"""Synthesize PDMX MIDI stems; optionally realify with SA3."""

from __future__ import annotations

import argparse
import multiprocessing
import shutil
import tempfile
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
    OUTPUT_DIR,
    RENDER_MODE_BASIC,
    RENDER_MODE_SLAKH,
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
from synthesis.cli_common import add_synthesis_args
from synthesis.dataset import prepare_ablation_dataset, prepare_full_dataset
from shared.csv_tables import append_rows_deduped, sanitize_track_name
from synthesis.paths import (
    ablation_raw_dir,
    ablation_realify_dir,
    full_stems_dir,
    full_stems_realify_dir,
)
from shared.repo_symlinks import link_ablations_in_repo


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


def synthesize_song_at_index(
    i: int,
    dataset: pd.DataFrame,
    completed_paths: set[str],
    args,
) -> tuple[dict | None, list[dict]]:
    """Synthesize one song. Returns (song_row, stem_rows) for main-process CSV writes."""
    path_output = dataset.at[i, "path_output"]
    song_dir = Path(path_output)
    audio_format = synthesis_audio_format(args.mp3)

    midi = mido.MidiFile(filename=dataset.at[i, "mid"], charset="utf8")
    n_tracks = len(midi.tracks)

    if path_output in completed_paths and song_is_complete(song_dir, n_tracks, audio_format) and not args.reset:
        del midi
        return None, []
    stems_complete = all(
        stem_is_valid(stem_path(song_dir, j, audio_format)) for j in range(n_tracks)
    )
    need_to_synthesize = args.reset or not stems_complete
    stem_rows: list[dict] = []

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

        if need_to_synthesize:
            track_midi.tracks.append(track_midi_track)
            slakh_cfg: dict = {}
            if args.render_mode == RENDER_MODE_SLAKH:
                import random

                from synthesis.patches import (
                    apply_patch_to_midi_track,
                    patch_group_key,
                    patch_seed,
                    select_patch,
                    slakh_render_for_track,
                )

                slakh_cfg = slakh_render_for_track(
                    program=program,
                    is_drum=is_drum,
                    track_name=track_name,
                )
                pool_id = slakh_cfg.get("pool_id")
                rng = None
                if pool_id:
                    group = patch_group_key(program, is_drum)
                    rng = random.Random(
                        patch_seed(args.sample_seed, path_output, group)
                    )
                apply_patch_to_midi_track(
                    track_midi_track,
                    select_patch(
                        program=program,
                        is_drum=is_drum,
                        pool_id=pool_id,
                        category=slakh_cfg.get("category"),
                        rng=rng,
                    ),
                )
            track_midi.save(track_paths[j])
            track_render_meta.append({
                "soundfont_filepath": args.soundfont_filepath,
                "fx_profile": None,
                **slakh_cfg,
            })

        stem_rows.append(dict(zip(STEMS_TABLE_COLUMNS, (
            path_output, j, program, is_drum,
            track_name if track_name and len(track_name) > 0 else None,
            has_lyrics,
        ))))

    del midi

    if need_to_synthesize:
        waveforms = []
        for j, track_path in enumerate(track_paths):
            meta = track_render_meta[j]
            soundfont_filepath = meta.get("soundfont_filepath") or args.soundfont_filepath
            if meta.get("soundfont"):
                soundfont_filepath = str(Path(SOUNDFONT_DIR) / meta["soundfont"])
            fx_profile = meta.get("fx_profile")
            waveform = get_waveform_tensor(
                track_path,
                soundfont_filepath,
                fx_profile=fx_profile,
            )
            waveforms.append(waveform)
            remove(track_path)

        temp_dir.cleanup()
        waveforms = pad_and_loudness_normalize(waveforms)

        for j, waveform in enumerate(waveforms):
            save_stem(waveform, song_dir, j, audio_format)

        write_mixture_from_waveforms(waveforms, song_dir, audio_format)
    elif stems_complete and not mixture_path(song_dir, audio_format).exists():
        write_mixture_from_song_dir(song_dir, list(range(n_tracks)), audio_format)

    song_info = dataset.loc[i].to_dict()
    song_info["path"] = path_output
    del song_info["path_output"], song_info["mid"]
    return song_info, stem_rows


_WORKER_CTX: dict = {}


def _init_synthesis_worker(dataset, completed_paths, args):
    global _WORKER_CTX
    _WORKER_CTX = {
        "dataset": dataset,
        "completed_paths": completed_paths,
        "args": args,
    }


def _synthesis_worker(i: int) -> tuple[dict | None, list[dict]]:
    return synthesize_song_at_index(
        i,
        _WORKER_CTX["dataset"],
        _WORKER_CTX["completed_paths"],
        _WORKER_CTX["args"],
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

    dataset = pd.read_csv(args.dataset_filepath, sep=",", header=0, index_col=False)
    dataset = dataset[dataset["subset:all_valid"]].reset_index(drop=True)
    dataset = dataset.drop(columns=["metadata", "mxl", "pdf", "version", "subset:all_valid"])
    if args.full:
        dataset = prepare_full_dataset(dataset)
    else:
        dataset = prepare_ablation_dataset(
            dataset,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
        )
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
        audio_format = synthesis_audio_format(args.mp3)
        if not song_is_complete(Path(path_output), n_tracks, audio_format):
            work_indices.append(i)

    with multiprocessing.Pool(
        processes=args.jobs,
        initializer=_init_synthesis_worker,
        initargs=(dataset, completed_paths, args),
    ) as pool:
        for song_info, stem_rows in tqdm(
            pool.imap(_synthesis_worker, work_indices, chunksize=CHUNK_SIZE),
            desc="Synthesizing songs",
            total=len(work_indices),
            unit="song",
        ):
            if song_info is None:
                continue
            if stem_rows:
                append_rows_deduped(
                    stems_output_filepath,
                    STEMS_TABLE_COLUMNS,
                    stem_rows,
                )
            append_rows_deduped(
                output_filepath,
                SONGS_TABLE_COLUMNS,
                [song_info],
            )


def synthesis_is_complete(source_dir: str, audio_format: str) -> bool:
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
        if not song_is_complete(song_dir, n_tracks, audio_format):
            return False
    return True


def require_raw_synthesis(
    source_dir: str,
    *,
    run_command: str,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
) -> None:
    """Raise if the non-realify synthesis pass has not completed successfully."""
    if synthesis_is_complete(source_dir, audio_format):
        return
    raise RuntimeError(
        "Cannot realify: raw stems are missing or incomplete at "
        f"{source_dir}\n"
        "Run the corresponding non-realify ablation first:\n"
        f"  {run_command}"
    )


def raw_synthesis_command(args) -> str:
    cmd = f"uv run python -m synthesis.synthesize --render-mode {args.render_mode}"
    if args.full:
        cmd += " --full"
    if args.mp3:
        cmd += " --mp3"
    return cmd


def run_realify_pass(args, source_dir: str, dest_dir: str):
    from synthesis.realify.realify import run_realify

    audio_format = synthesis_audio_format(args.mp3)
    run_realify(
        source_dir=source_dir,
        output_dir=dest_dir,
        model=args.model,
        limit=args.realify_limit,
        jobs=args.jobs,
        batch_size=args.realify_batch_size or REALIFY_BATCH_SIZE,
        audio_format=audio_format,
        sample_seed=args.sample_seed,
        reset=args.reset,
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
        audio_format = synthesis_audio_format(args.mp3)
        require_raw_synthesis(
            source_dir,
            run_command=raw_synthesis_command(args),
            audio_format=audio_format,
        )
        run_realify_pass(args, source_dir, dest_dir)
    else:
        run_synthesis(args, source_dir)

    link_ablations_in_repo(args.output_dir)


if __name__ == "__main__":
    main()
