"""Synthesize PDMX MIDI stems; optionally realify with SA3."""

from __future__ import annotations

import argparse
import multiprocessing
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import makedirs, remove
from os.path import dirname, exists, expanduser
from pathlib import Path

import mido
import pandas as pd
from tqdm import tqdm

from shared.config import (
    CHUNK_SIZE,
    DATA_DIR_NAME,
    DEFAULT_AUDIO_FORMAT,
    MAX_N_NOTES_IN_STEM,
    NA_STRING,
    REALIFY_BATCH_SIZE,
    REALIFY_CONTENT_FIDELITY_ENFORCE,
    REALIFY_SILENCE_ENFORCE,
    SONGS_TABLE_COLUMNS,
    SOUNDFONT_DIR,
    STEMS_FILE_NAME,
    STEMS_TABLE_COLUMNS,
)
from synthesis.audio import (
    get_waveform_tensor,
    pad_and_loudness_normalize,
    mixture_path,
    save_stem,
    song_is_complete,
    stem_is_valid,
    stem_path,
    synthesis_audio_format,
    write_mixture_from_song_dir,
    write_mixture_from_waveforms,
)
from synthesis.cli_common import add_synthesis_args, default_gm_register_path
from synthesis.dataset import listening_sample_path, prepare_ablation_dataset, prepare_full_dataset
from shared.csv_tables import append_rows_deduped, sanitize_track_name
from synthesis.paths import (
    ablation_raw_dir,
    ablation_realify_dir,
    full_stems_dir,
    full_stems_realify_dir,
)
from shared.repo_symlinks import link_ablations_in_repo
from synthesis.ddsp.config import DDSP_ROUTING_COLUMNS, DDSP_ROUTING_FILE_NAME
from synthesis.patches import PatchAssignment, apply_patch_to_midi_track
from synthesis.reuse import (
    copy_stem,
    donor_raw_stem_path,
    fallback_donor_mode,
    reused_source_label,
    song_rel_under_data,
    uses_ddsp,
    uses_slakh_recipes,
)

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        prog="Synthesize",
        description="Synthesize PDMX stems; pass --full for all valid songs, --realify for SA3.",
    )
    add_synthesis_args(parser)
    return parser.parse_args(args=args, namespace=namespace)


def song_output_dir(output_dir: str, original_dataset_dir: str, json_path: str) -> str:
    rel = json_path[len(original_dataset_dir):]
    rel_no_ext = ".".join(rel.split(".")[:-1])
    return f"{output_dir}{rel_no_ext}"


def _require_mixture(args) -> bool:
    return not bool(getattr(args, "no_mixture", False))


def synthesize_song_at_index(
    i: int,
    dataset: pd.DataFrame,
    completed_paths: set[str],
    args,
) -> tuple[dict | None, list[dict], list[dict]]:
    """Synthesize one song. Returns (song_row, stem_rows, ddsp_routing_rows).

    For DDSP render modes, ``args.ddsp_pass`` selects a global phase:
    ``ddsp_piano`` / ``midi_ddsp`` only render that neural backend; ``finalize``
    fills donor/soundfont stems, mixtures, and CSV rows.
    """
    path_output = dataset.at[i, "path_output"]
    song_dir = Path(path_output)
    audio_format = synthesis_audio_format(args.flac)
    require_mixture = _require_mixture(args)
    ddsp_pass = getattr(args, "ddsp_pass", None)

    midi = mido.MidiFile(filename=dataset.at[i, "mid"], charset="utf8")
    n_tracks = len(midi.tracks)

    if (
        path_output in completed_paths
        and song_is_complete(song_dir, n_tracks, audio_format, require_mixture=require_mixture)
        and not args.reset
    ):
        del midi
        return None, [], []
    stems_complete = all(
        stem_is_valid(stem_path(song_dir, j, audio_format)) for j in range(n_tracks)
    )
    # Neural-only passes still run when some stems exist (render missing neural).
    need_to_synthesize = args.reset or not stems_complete
    if uses_ddsp(args.render_mode) and ddsp_pass in ("ddsp_piano", "midi_ddsp"):
        need_to_synthesize = True
    stem_rows: list[dict] = []
    routing_rows: list[dict] = []

    if need_to_synthesize:
        temp_dir = tempfile.TemporaryDirectory()
        track_paths = [f"{temp_dir.name}/{j}.mid" for j in range(len(midi.tracks))]
        track_render_meta: list[dict] = []

    for j, track in enumerate(midi.tracks):
        if need_to_synthesize:
            track_midi = mido.MidiFile(ticks_per_beat=midi.ticks_per_beat, charset="utf8")
            track_midi_track = mido.MidiTrack()

        program = 0
        is_drum = False
        track_name = None
        has_lyrics = False
        n_notes = 0
        determined_whether_track_is_drum = False

        for message in track:
            if message.type == "note_on" and message.velocity > 0:
                n_notes += 1
            elif message.type == "program_change":
                program = message.program
            elif message.type == "track_name":
                track_name = sanitize_track_name(
                    " ".join(message.name.replace(",", " ").split())
                )
            elif message.type == "lyrics":
                has_lyrics = True
            if not determined_whether_track_is_drum and hasattr(message, "channel"):
                is_drum = message.channel == 9
                determined_whether_track_is_drum = True
            if need_to_synthesize and n_notes <= MAX_N_NOTES_IN_STEM:
                track_midi_track.append(message)

        # Apply GM register correction (track-name → program) before render routing.
        register_lookup = getattr(args, "gm_register_lookup", None)
        if register_lookup is not None:
            from analysis.gm_register import lookup_corrected_program

            mid_key = dataset.at[i, "mid"]
            corrected = lookup_corrected_program(
                register_lookup,
                mid=mid_key,
                track=j,
                default=program,
            )
            if corrected != program:
                program = corrected
                if need_to_synthesize:
                    apply_patch_to_midi_track(
                        track_midi_track,
                        PatchAssignment(program=program, is_drum=is_drum),
                    )

        if need_to_synthesize:
            track_midi.tracks.append(track_midi_track)
            slakh_cfg: dict = {}
            if uses_slakh_recipes(args.render_mode):
                from synthesis.patches import (
                    select_patch,
                    slakh_render_for_track,
                )

                slakh_cfg = slakh_render_for_track(
                    program=program,
                    is_drum=is_drum,
                    track_name=track_name,
                )
                apply_patch_to_midi_track(
                    track_midi_track,
                    select_patch(
                        program=program,
                        is_drum=is_drum,
                        pool_id=None,
                        category=slakh_cfg.get("category"),
                    ),
                )
            track_midi.save(track_paths[j])
            route_meta: dict = {}
            if uses_ddsp(args.render_mode):
                from synthesis.ddsp.routing import route_stem

                route = route_stem(
                    program=program,
                    is_drum=is_drum,
                    track_name=track_name,
                    track=track_midi_track,
                    ticks_per_beat=midi.ticks_per_beat,
                    check_monophony=True,
                )
                route_meta = {
                    "ddsp_backend": route.backend,
                    "ddsp_instrument_key": route.instrument_key,
                    "ddsp_reason": route.reason,
                    "n_notes": n_notes,
                }
            track_render_meta.append({
                "soundfont_filepath": args.soundfont_filepath,
                "fx_profile": None,
                **slakh_cfg,
                **route_meta,
            })

        stem_rows.append(dict(zip(STEMS_TABLE_COLUMNS, (
            path_output, j, program, is_drum,
            track_name if track_name and len(track_name) > 0 else None,
            has_lyrics,
        ))))

    del midi

    if need_to_synthesize:
        donor_mode = fallback_donor_mode(args.render_mode)
        song_rel = None
        if uses_ddsp(args.render_mode) and donor_mode is not None:
            song_rel = song_rel_under_data(
                song_dir,
                ablation_raw_dir(args.output_dir, args.render_mode),
            )

        if uses_ddsp(args.render_mode):
            from synthesis.ddsp.pool import ddsp_oneshot_enabled, get_ddsp_pool
            from synthesis.ddsp.routing import StemRoute
            from synthesis.ddsp.synthesize import synthesize_stem_neural

            # Global two-pass: neural phases only render one backend; finalize does the rest.
            if ddsp_pass in ("ddsp_piano", "midi_ddsp"):
                neural_jobs: list[tuple[int, str, StemRoute]] = []
                for j, track_path in enumerate(track_paths):
                    meta = track_render_meta[j]
                    if meta.get("ddsp_backend") != ddsp_pass:
                        continue
                    out_stem = stem_path(song_dir, j, audio_format)
                    if stem_is_valid(out_stem) and not args.reset:
                        continue
                    neural_jobs.append((
                        j,
                        track_path,
                        StemRoute(
                            backend=ddsp_pass,
                            instrument_key=meta.get("ddsp_instrument_key"),
                            reason=meta.get("ddsp_reason") or "",
                        ),
                    ))

                if neural_jobs:
                    def _neural_one(job: tuple[int, str, StemRoute]):
                        idx, mid_path, route = job
                        return idx, synthesize_stem_neural(mid_path, route)

                    if ddsp_oneshot_enabled():
                        max_workers = 1
                    else:
                        max_workers = max(1, get_ddsp_pool().size)
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(_neural_one, job) for job in neural_jobs
                        ]
                        for fut in as_completed(futures):
                            idx, waveform = fut.result()
                            waveform = pad_and_loudness_normalize([waveform])[0]
                            save_stem(waveform, song_dir, idx, audio_format)

                for path in track_paths:
                    if exists(path):
                        remove(path)
                temp_dir.cleanup()
                # CSV / mixtures only on finalize pass.
                return None, [], []

            # Finalize (default when ddsp_pass is None or "finalize"): non-neural stems + mix.
            for j, track_path in enumerate(track_paths):
                meta = track_render_meta[j]
                backend = meta.get("ddsp_backend")
                out_stem = stem_path(song_dir, j, audio_format)
                source = "rendered"
                original_path = None

                if backend in ("midi_ddsp", "ddsp_piano"):
                    if not stem_is_valid(out_stem):
                        for path in track_paths:
                            if exists(path):
                                remove(path)
                        temp_dir.cleanup()
                        raise RuntimeError(
                            f"Missing neural DDSP stem after neural passes: {out_stem}\n"
                            f"backend={backend} song={path_output} track={j}"
                        )
                elif stem_is_valid(out_stem) and not args.reset:
                    pass
                elif donor_mode is not None and song_rel is not None:
                    donor_stem = donor_raw_stem_path(
                        args.output_dir,
                        donor_mode,
                        song_rel,
                        j,
                        audio_format,
                    )
                    if stem_is_valid(donor_stem):
                        copy_stem(donor_stem, out_stem)
                        source = reused_source_label(donor_mode)
                        original_path = str(donor_stem.resolve())
                    elif getattr(args, "allow_fallback_render", False):
                        waveform = _render_soundfont_stem(
                            track_path, meta, args, path_output,
                        )
                        waveform = pad_and_loudness_normalize([waveform])[0]
                        save_stem(waveform, song_dir, j, audio_format)
                    else:
                        for path in track_paths:
                            if exists(path):
                                remove(path)
                        temp_dir.cleanup()
                        raise RuntimeError(
                            f"Missing donor stem for DDSP fallback: {donor_stem}\n"
                            f"Generate the donor ablation first:\n"
                            f"  uv run python -m synthesis.synthesize --render-mode {donor_mode}\n"
                            "Or pass --allow-fallback-render to Fluidsynth-render missing donors."
                        )
                else:
                    waveform = _render_soundfont_stem(
                        track_path, meta, args, path_output,
                    )
                    waveform = pad_and_loudness_normalize([waveform])[0]
                    save_stem(waveform, song_dir, j, audio_format)

                routing_rows.append({
                    "path": path_output,
                    "track": j,
                    "program": stem_rows[j]["program"],
                    "is_drum": stem_rows[j]["is_drum"],
                    "name": stem_rows[j]["name"],
                    "backend": meta.get("ddsp_backend"),
                    "instrument_key": meta.get("ddsp_instrument_key"),
                    "reason": meta.get("ddsp_reason"),
                    "n_notes": meta.get("n_notes"),
                    "source": source,
                    "original_path": original_path,
                })
                remove(track_path)
            temp_dir.cleanup()
            if require_mixture:
                write_mixture_from_song_dir(song_dir, list(range(n_tracks)), audio_format)
        else:
            waveforms = []
            for j, track_path in enumerate(track_paths):
                meta = track_render_meta[j]
                waveforms.append(
                    _render_soundfont_stem(track_path, meta, args, path_output)
                )
                remove(track_path)
            temp_dir.cleanup()
            waveforms = pad_and_loudness_normalize(waveforms)
            for j, waveform in enumerate(waveforms):
                save_stem(waveform, song_dir, j, audio_format)
            if require_mixture:
                write_mixture_from_waveforms(waveforms, song_dir, audio_format)
    elif (
        require_mixture
        and stems_complete
        and not mixture_path(song_dir, audio_format).exists()
    ):
        write_mixture_from_song_dir(song_dir, list(range(n_tracks)), audio_format)

    song_info = dataset.loc[i].to_dict()
    song_info["path"] = path_output
    del song_info["path_output"], song_info["mid"]
    return song_info, stem_rows, routing_rows


def _render_soundfont_stem(track_path: str, meta: dict, args, path_output: str):
    soundfont_filepath = meta.get("soundfont_filepath") or args.soundfont_filepath
    fx_profile = meta.get("fx_profile")
    if uses_slakh_recipes(args.render_mode):
        from experiments.patch_sweep.config import soundfont_file_for_id
        from experiments.patch_sweep.winners import pick_fx_profile, pick_soundfont_id

        soundfont_ids = meta.get("soundfont_ids") or []
        if not soundfont_ids and meta.get("soundfont_id"):
            soundfont_ids = [meta["soundfont_id"]]
        category = meta.get("category") or "default"
        if soundfont_ids:
            picked = pick_soundfont_id(
                list(soundfont_ids),
                category=category,
                song_path=path_output,
                sample_seed=args.sample_seed,
            )
            soundfont_filepath = str(
                Path(SOUNDFONT_DIR) / soundfont_file_for_id(picked)
            )
        elif meta.get("soundfont"):
            soundfont_filepath = str(Path(SOUNDFONT_DIR) / meta["soundfont"])

        fx_profiles = meta.get("fx_profiles") or []
        if not fx_profiles and meta.get("fx_profile"):
            fx_profiles = [meta["fx_profile"]]
        if fx_profiles:
            fx_profile = pick_fx_profile(
                list(fx_profiles),
                category=category,
                song_path=path_output,
                sample_seed=args.sample_seed,
            )
    elif meta.get("soundfont"):
        soundfont_filepath = str(Path(SOUNDFONT_DIR) / meta["soundfont"])
    return get_waveform_tensor(
        track_path,
        soundfont_filepath,
        fx_profile=fx_profile,
    )


_WORKER_CTX: dict = {}


def _init_synthesis_worker(dataset, completed_paths, args):
    global _WORKER_CTX
    _WORKER_CTX = {
        "dataset": dataset,
        "completed_paths": completed_paths,
        "args": args,
    }
    if uses_ddsp(args.render_mode):
        from synthesis.ddsp.pool import ddsp_oneshot_enabled, ensure_ddsp_pool

        ddsp_pass = getattr(args, "ddsp_pass", None)
        if (
            ddsp_pass in ("ddsp_piano", "midi_ddsp")
            and not ddsp_oneshot_enabled()
        ):
            ensure_ddsp_pool()


def _synthesis_worker(i: int) -> tuple[dict | None, list[dict], list[dict]]:
    return synthesize_song_at_index(
        i,
        _WORKER_CTX["dataset"],
        _WORKER_CTX["completed_paths"],
        _WORKER_CTX["args"],
    )


def _run_song_pool(
    *,
    dataset: pd.DataFrame,
    completed_paths: set[str],
    args,
    work_indices: list,
    jobs: int,
    desc: str,
    stems_output_filepath: str,
    routing_output_filepath: str | None,
    output_filepath: str,
    write_tables: bool,
) -> None:
    """Run song workers once (one DDSP pass or the non-DDSP path)."""
    pool_ctx = (
        multiprocessing.get_context("spawn")
        if uses_ddsp(args.render_mode)
        else multiprocessing
    )
    with pool_ctx.Pool(
        processes=jobs,
        initializer=_init_synthesis_worker,
        initargs=(dataset, completed_paths, args),
    ) as pool:
        for song_info, stem_rows, routing_rows in tqdm(
            pool.imap(_synthesis_worker, work_indices, chunksize=CHUNK_SIZE),
            desc=desc,
            total=len(work_indices),
            unit="song",
        ):
            if not write_tables or song_info is None:
                continue
            if stem_rows:
                append_rows_deduped(
                    stems_output_filepath,
                    STEMS_TABLE_COLUMNS,
                    stem_rows,
                )
            if routing_rows and routing_output_filepath is not None:
                append_rows_deduped(
                    routing_output_filepath,
                    DDSP_ROUTING_COLUMNS,
                    routing_rows,
                )
            append_rows_deduped(
                output_filepath,
                SONGS_TABLE_COLUMNS,
                [song_info],
            )


def reset_synthesis_output(output_dir: str) -> None:
    """Remove all prior synthesis artifacts under output_dir."""
    if exists(output_dir):
        shutil.rmtree(output_dir)
    makedirs(output_dir, exist_ok=True)


def run_synthesis(args, output_dir: str):
    if args.reset:
        reset_synthesis_output(output_dir)
    else:
        makedirs(output_dir, exist_ok=True)
    output_filepath = f"{output_dir}/{DATA_DIR_NAME}.csv"
    stems_output_filepath = f"{output_dir}/{STEMS_FILE_NAME}.csv"
    makedirs(f"{output_dir}/{DATA_DIR_NAME}", exist_ok=True)

    if args.soundfont_filepath is None:
        args.soundfont_filepath = f"{expanduser('~')}/.muspy/musescore-general/MuseScore_General.sf3"
    if not exists(args.soundfont_filepath):
        raise RuntimeError("Soundfont not found.")

    # GM register: required step-0 corrections unless --no-register.
    args.gm_register_lookup = None
    register_df = None
    register_path = getattr(args, "register", None) or default_gm_register_path(args.output_dir)
    if not getattr(args, "no_register", False):
        from analysis.gm_register import load_register_lookup

        if not exists(register_path):
            raise RuntimeError(
                f"GM register not found at {register_path}\n"
                "Run register correction before any ablation:\n"
                "  uv run python -m analysis.analyze_gm_register --subset all_valid -j 8\n"
                "Or pass --no-register to synthesize with raw MIDI programs."
            )
        pdmx_root = dirname(args.dataset_filepath)
        args.gm_register_lookup = load_register_lookup(register_path, pdmx_root=pdmx_root)
        register_df = pd.read_csv(register_path)
        print(f"Loaded GM register ({len(args.gm_register_lookup)} keys) from {register_path}")

    if uses_ddsp(args.render_mode) and not args.full:
        require_donor_ablation(args, realify=False)

    dataset = pd.read_csv(args.dataset_filepath, sep=",", header=0, index_col=False)
    dataset = dataset[dataset["subset:all_valid"]].reset_index(drop=True)
    dataset = dataset.drop(columns=["metadata", "mxl", "pdf", "version", "subset:all_valid"])
    if args.full:
        dataset = prepare_full_dataset(dataset)
    else:
        sample_file = listening_sample_path(args.output_dir)
        dataset = prepare_ablation_dataset(
            dataset,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
            min_stems_per_category=args.min_stems_per_category,
            register_df=register_df,
            listening_sample_file=sample_file,
            persist_sample=True,
        )
        if sample_file.is_file():
            print(f"Ablation sample: {sample_file} ({len(dataset)} songs)")
    original_dataset_dir = dirname(args.dataset_filepath)
    dataset["path"] = [original_dataset_dir + p[1:] for p in dataset["path"]]
    dataset["mid"] = [original_dataset_dir + p[1:] for p in dataset["mid"]]
    dataset["path_output"] = [
        song_output_dir(output_dir, original_dataset_dir, p) for p in dataset["path"]
    ]
    dataset = dataset.reset_index(drop=True)

    for song_dir in set(dataset["path_output"]):
        makedirs(song_dir, exist_ok=True)

    if not exists(output_filepath) or args.reset:
        pd.DataFrame(columns=SONGS_TABLE_COLUMNS).to_csv(
            output_filepath, sep=",", na_rep=NA_STRING, header=True, index=False, mode="w",
        )
    completed_paths = set()
    if exists(output_filepath) and not args.reset:
        completed_paths = set(
            pd.read_csv(output_filepath, sep=",", header=0, index_col=False, usecols=["path"])["path"]
        )
    if not exists(stems_output_filepath) or args.reset:
        pd.DataFrame(columns=STEMS_TABLE_COLUMNS).to_csv(
            stems_output_filepath, sep=",", na_rep=NA_STRING, header=True, index=False, mode="w",
        )
    routing_output_filepath = f"{output_dir}/{DDSP_ROUTING_FILE_NAME}"
    if uses_ddsp(args.render_mode) and (
        not exists(routing_output_filepath) or args.reset
    ):
        pd.DataFrame(columns=DDSP_ROUTING_COLUMNS).to_csv(
            routing_output_filepath, sep=",", na_rep=NA_STRING, header=True, index=False, mode="w",
        )

    # Neural DDSP needs Torch in workers for resample. Fork-after-CUDA causes
    # SIGKILL (exit -9); use spawn so each worker initializes cleanly.
    jobs = 1 if uses_ddsp(args.render_mode) else args.jobs
    if uses_ddsp(args.render_mode) and args.jobs > 1:
        print(
            f"Note: {args.render_mode} uses spawn + -j 1 (was {args.jobs}) to avoid "
            "CUDA-after-fork kills (exit -9)."
        )

    require_mixture = _require_mixture(args)
    work_indices = []
    for i in dataset.index:
        if args.reset:
            work_indices.append(i)
            continue
        path_output = dataset.at[i, "path_output"]
        if path_output not in completed_paths:
            work_indices.append(i)
            continue
        n_tracks = int(dataset.at[i, "n_tracks"])
        audio_format = synthesis_audio_format(args.flac)
        if not song_is_complete(
            Path(path_output), n_tracks, audio_format, require_mixture=require_mixture,
        ):
            work_indices.append(i)

    if not work_indices:
        return

    if uses_ddsp(args.render_mode):
        from synthesis.ddsp.pool import shutdown_ddsp_pool

        # Global two-pass: keep one neural backend hot per phase, then finalize.
        print(
            "DDSP schedule: pass1=ddsp_piano, pass2=midi_ddsp, "
            "pass3=donors/soundfont+mixtures (pool restarts between neural passes).",
            flush=True,
        )
        ddsp_passes = (
            ("ddsp_piano", "DDSP piano stems", False),
            ("midi_ddsp", "DDSP mono stems", False),
            ("finalize", "DDSP finalize", True),
        )
        for pass_name, desc, write_tables in ddsp_passes:
            args.ddsp_pass = pass_name
            if pass_name == "finalize":
                shutdown_ddsp_pool()
            _run_song_pool(
                dataset=dataset,
                completed_paths=completed_paths,
                args=args,
                work_indices=work_indices,
                jobs=jobs,
                desc=desc,
                stems_output_filepath=stems_output_filepath,
                routing_output_filepath=routing_output_filepath,
                output_filepath=output_filepath,
                write_tables=write_tables,
            )
            if pass_name in ("ddsp_piano", "midi_ddsp"):
                # Drop resident TF models before the next backend.
                shutdown_ddsp_pool()
        args.ddsp_pass = None
        return

    _run_song_pool(
        dataset=dataset,
        completed_paths=completed_paths,
        args=args,
        work_indices=work_indices,
        jobs=jobs,
        desc="Synthesizing songs",
        stems_output_filepath=stems_output_filepath,
        routing_output_filepath=None,
        output_filepath=output_filepath,
        write_tables=True,
    )


def synthesis_is_complete(
    source_dir: str,
    audio_format: str,
    *,
    require_mixture: bool = True,
) -> bool:
    """True when data/stems tables exist and every listed song has stem files on disk."""
    source = Path(source_dir)
    data_csv = source / f"{DATA_DIR_NAME}.csv"
    stems_csv = source / f"{STEMS_FILE_NAME}.csv"
    if not data_csv.exists() or not stems_csv.exists():
        return False

    songs = pd.read_csv(data_csv, sep=",", header=0, index_col=False)
    stems = pd.read_csv(stems_csv, sep=",", header=0, index_col=False)
    if len(songs) == 0 or len(stems) == 0:
        return False

    for _, row in songs.iterrows():
        song_dir = Path(row["path"])
        n_tracks = int(row["n_tracks"])
        if not song_is_complete(
            song_dir, n_tracks, audio_format, require_mixture=require_mixture,
        ):
            return False
    return True


def require_raw_synthesis(
    source_dir: str,
    *,
    run_command: str,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    require_mixture: bool = True,
) -> None:
    """Raise if the non-realify synthesis pass has not completed successfully."""
    if synthesis_is_complete(source_dir, audio_format, require_mixture=require_mixture):
        return
    raise RuntimeError(
        "Cannot realify: raw stems are missing or incomplete at "
        f"{source_dir}\n"
        "Run the corresponding non-realify ablation first:\n"
        f"  {run_command}"
    )


def require_donor_ablation(args, *, realify: bool) -> None:
    """Ensure the soundfont-fallback donor ablation exists for DDSP modes."""
    donor_mode = fallback_donor_mode(args.render_mode)
    if donor_mode is None:
        return
    audio_format = synthesis_audio_format(args.flac)
    require_mixture = _require_mixture(args)
    if realify:
        donor_dir = ablation_realify_dir(args.output_dir, donor_mode)
        cmd = (
            f"uv run python -m synthesis.synthesize --render-mode {donor_mode} --realify"
        )
    else:
        donor_dir = ablation_raw_dir(args.output_dir, donor_mode)
        cmd = f"uv run python -m synthesis.synthesize --render-mode {donor_mode}"
    if getattr(args, "no_mixture", False):
        cmd += " --no-mixture"
    if args.flac:
        cmd += " --flac"
    if synthesis_is_complete(donor_dir, audio_format, require_mixture=require_mixture):
        return
    if getattr(args, "allow_fallback_render", False) and not realify:
        print(
            f"Warning: donor ablation incomplete at {donor_dir}; "
            "--allow-fallback-render will Fluidsynth-render missing stems."
        )
        return
    kind = "realify" if realify else "raw"
    raise RuntimeError(
        f"Cannot run {args.render_mode}{' --realify' if realify else ''}: "
        f"donor {kind} ablation incomplete at {donor_dir}\n"
        f"Run first:\n  {cmd}"
    )


def raw_synthesis_command(args) -> str:
    cmd = f"uv run python -m synthesis.synthesize --render-mode {args.render_mode}"
    if args.full:
        cmd += " --full"
    if args.flac:
        cmd += " --flac"
    if getattr(args, "no_mixture", False):
        cmd += " --no-mixture"
    return cmd


def run_realify_pass(args, source_dir: str, dest_dir: str):
    from synthesis.realify.realify import run_realify

    audio_format = synthesis_audio_format(args.flac)
    content_fidelity_enforce = REALIFY_CONTENT_FIDELITY_ENFORCE
    if getattr(args, "content_fidelity_enforce", False):
        content_fidelity_enforce = True
    if getattr(args, "no_content_fidelity_enforce", False):
        content_fidelity_enforce = False

    run_realify(
        source_dir=source_dir,
        output_dir=dest_dir,
        model=args.model,
        limit=args.realify_limit,
        jobs=args.jobs,
        batch_size=(
            REALIFY_BATCH_SIZE
            if args.realify_batch_size is None
            else args.realify_batch_size
        ),
        audio_format=audio_format,
        sample_seed=args.sample_seed,
        reset=args.reset,
        silence_enforce=REALIFY_SILENCE_ENFORCE and not args.no_silence_enforce,
        content_fidelity_enforce=content_fidelity_enforce,
        no_mixture=bool(getattr(args, "no_mixture", False)),
        output_root=args.output_dir,
        render_mode=args.render_mode,
    )


def main():
    args = parse_args()
    if args.full:
        source_dir = full_stems_dir(args.output_dir)
        dest_dir = full_stems_realify_dir(args.output_dir)
    else:
        source_dir = ablation_raw_dir(args.output_dir, args.render_mode)
        dest_dir = ablation_realify_dir(args.output_dir, args.render_mode)

    if args.realify:
        audio_format = synthesis_audio_format(args.flac)
        require_raw_synthesis(
            source_dir,
            run_command=raw_synthesis_command(args),
            audio_format=audio_format,
            require_mixture=_require_mixture(args),
        )
        if uses_ddsp(args.render_mode) and not args.full:
            require_donor_ablation(args, realify=True)
        run_realify_pass(args, source_dir, dest_dir)
    else:
        run_synthesis(args, source_dir)

    link_ablations_in_repo(args.output_dir)


if __name__ == "__main__":
    main()
