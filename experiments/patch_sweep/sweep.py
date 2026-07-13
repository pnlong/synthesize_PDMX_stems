"""Run phased Slakh-style patch sweeps over the probe set."""

from __future__ import annotations

import argparse
import multiprocessing
import random
import tempfile
from os import remove
from os.path import exists
from os.path import expanduser
from pathlib import Path

import mido
import pandas as pd
import yaml
from tqdm import tqdm

from experiments.patch_sweep.config import (
    EXPERIMENT_DIR,
    PHASE1,
    PHASE2,
    PHASE3,
    PHASE_GRID_FILES,
    PHASE_OUTPUT_SUBDIRS,
    PHASES,
    load_soundfont_catalog,
    load_yaml,
    phase_output_dir,
    soundfont_file_for_id,
    soundfont_path,
)
from experiments.patch_sweep.winners import (
    phase_is_complete,
    phase_winners,
    resolve_phase1_soundfont_id,
    resolve_phase2_fx_profile,
)
from experiments.paths import DEFAULT_PROBE_STEMS, patch_sweep_output_root
from experiments.probe_stems import validate_probe_stems, resolve_mid_path
from shared.config import (
    ABLATION_SAMPLE_SEED,
    CHUNK_SIZE,
    DATA_DIR_NAME,
    DEFAULT_AUDIO_FORMAT,
    MAX_N_NOTES_IN_STEM,
    OUTPUT_DIR,
    PDMX_FILEPATH,
    SOUNDFONT_DIR,
)
from shared.repo_symlinks import REPO_PATCH_SWEEP_OUTPUT_SYMLINK
from synthesis.audio import (
    get_waveform_tensor,
    pad_and_loudness_normalize,
    save_stem,
    stem_filename,
    stem_is_valid,
    stem_path,
    synthesis_audio_format,
)
from synthesis.listening.catalog import default_ablations_dir
from synthesis.patches import apply_patch_to_midi_track, patch_group_key, patch_seed, select_patch
from synthesis.paths import ablation_raw_dir

MANIFEST_FILENAME = "manifest.csv"
VARIANTS_DIR_NAME = "variants"

MANIFEST_COLUMNS = [
    "phase",
    "variant_id",
    "soundfont_id",
    "soundfont_file",
    "fx_profile",
    "pool_id",
    "program",
    "gm_class",
    "stem_id",
    "category",
    "path",
    "track",
    "out_path",
]


def default_output_dir(output_root: str = OUTPUT_DIR) -> Path:
    if REPO_PATCH_SWEEP_OUTPUT_SYMLINK.is_dir():
        return REPO_PATCH_SWEEP_OUTPUT_SYMLINK.resolve()
    return Path(patch_sweep_output_root(output_root))


def default_source_dir(output_root: str = OUTPUT_DIR) -> Path:
    ablations = default_ablations_dir()
    basic = ablations / "basic"
    if basic.is_dir():
        return basic
    return Path(ablation_raw_dir(output_root, "basic"))


def song_path_from_id(source_dir: Path, song_id: str) -> Path:
    return source_dir / DATA_DIR_NAME / song_id


def variant_output_path(
    output_dir: Path,
    variant_id: str,
    song_path: Path,
    source_dir: Path,
    track: int,
    audio_format: str,
) -> Path:
    song_id = str(song_path.relative_to(source_dir / DATA_DIR_NAME))
    return (
        output_dir
        / VARIANTS_DIR_NAME
        / variant_id
        / DATA_DIR_NAME
        / song_id
        / stem_filename(track, audio_format)
    )


def _require_phase_prerequisites(phase: str, winners_path: Path) -> None:
    if phase == PHASE2 and not phase_is_complete(PHASE1, winners_path):
        raise RuntimeError(
            f"{PHASE2} requires completed {PHASE1} winners in {winners_path}. "
            f"Run the phase 1 sweep, listening test, and record_winners first."
        )
    if phase == PHASE3:
        if not phase_is_complete(PHASE1, winners_path):
            raise RuntimeError(f"{PHASE3} requires completed {PHASE1} winners.")
        if not phase_is_complete(PHASE2, winners_path):
            raise RuntimeError(f"{PHASE3} requires completed {PHASE2} winners.")


def resolve_render_settings(
    *,
    phase: str,
    variant: dict,
    grid_cfg: dict,
    category: str | None,
    catalog: dict,
    winners_path: Path,
    soundfont_dir: str | Path,
) -> tuple[str, str, str | None, str | None]:
    """Return (soundfont_path, soundfont_id, fx_profile, pool_id)."""
    default_fx = grid_cfg.get("fx_profile")
    default_pool = grid_cfg.get("pool_id")

    if phase == PHASE1:
        soundfont_id = variant["soundfont_id"]
        sf_file = soundfont_file_for_id(soundfont_id, catalog)
        fx_profile = variant.get("fx_profile") or default_fx or "dry"
        pool_id = variant.get("pool_id") or default_pool
        return str(soundfont_path(sf_file, soundfont_dir)), soundfont_id, fx_profile, pool_id

    if phase == PHASE2:
        soundfont_id = resolve_phase1_soundfont_id(category or "", winners_path)
        if soundfont_id is None:
            raise RuntimeError(f"No phase-1 soundfont winner for category: {category}")
        sf_file = soundfont_file_for_id(soundfont_id, catalog)
        fx_profile = variant.get("fx_profile") or default_fx or "dry"
        pool_id = variant.get("pool_id") or default_pool
        return str(soundfont_path(sf_file, soundfont_dir)), soundfont_id, fx_profile, pool_id

    # phase 3
    soundfont_id = resolve_phase1_soundfont_id(category or "", winners_path)
    if soundfont_id is None:
        raise RuntimeError(f"No phase-1 soundfont winner for category: {category}")
    sf_file = soundfont_file_for_id(soundfont_id, catalog)
    fx_profile = resolve_phase2_fx_profile(category or "", winners_path)
    if fx_profile is None:
        raise RuntimeError(f"No phase-2 FX winner for category: {category}")
    pool_id = variant.get("pool_id") or default_pool
    return str(soundfont_path(sf_file, soundfont_dir)), soundfont_id, fx_profile, pool_id


def render_probe_stem(
    *,
    mid_path: Path,
    track: int,
    pool_id: str | None,
    category: str | None,
    soundfont_filepath: str,
    fx_profile: str | None,
    sample_seed: int,
    song_path: str,
    out_path: Path,
    audio_format: str,
) -> tuple[int, str]:
    midi = mido.MidiFile(filename=str(mid_path), charset="utf8")
    if track >= len(midi.tracks):
        raise ValueError(f"Track {track} out of range for {mid_path} ({len(midi.tracks)} tracks)")

    source_track = midi.tracks[track]
    program = 0
    is_drum = False
    n_notes = 0
    determined_drum = False

    track_midi = mido.MidiFile(ticks_per_beat=midi.ticks_per_beat, charset="utf8")
    track_midi_track = mido.MidiTrack()

    for message in source_track:
        if message.type == "note_on" and message.velocity > 0:
            n_notes += 1
        elif message.type == "program_change":
            program = message.program
        if not determined_drum and hasattr(message, "channel"):
            is_drum = message.channel == 9
            determined_drum = True
        if n_notes <= MAX_N_NOTES_IN_STEM:
            track_midi_track.append(message)

    rng = random.Random(
        patch_seed(sample_seed, song_path, patch_group_key(program, is_drum))
    )
    assignment = select_patch(
        program=program,
        is_drum=is_drum,
        pool_id=pool_id,
        category=category,
        rng=rng,
    )
    apply_patch_to_midi_track(track_midi_track, assignment)
    track_midi.tracks.append(track_midi_track)

    temp_dir = tempfile.TemporaryDirectory()
    track_mid = Path(temp_dir.name) / "track.mid"
    track_midi.save(track_mid)

    waveform = get_waveform_tensor(
        str(track_mid),
        soundfont_filepath,
        fx_profile=fx_profile,
    )
    remove(track_mid)
    temp_dir.cleanup()

    waveforms = pad_and_loudness_normalize([waveform])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_stem(waveforms[0], out_path.parent, track, audio_format)

    return assignment.program, assignment.gm_class or ""


def build_sweep_tasks(
    *,
    phase: str,
    source_dir: Path,
    output_dir: Path,
    probe_stems: list[dict],
    variants: list[dict],
    grid_cfg: dict,
    audio_format: str,
    sample_seed: int,
    catalog: dict,
    winners_path: Path,
    soundfont_dir: str | Path,
    pdmx_filepath: str,
) -> list[dict]:
    tasks = []
    for probe in probe_stems:
        song_path = song_path_from_id(source_dir, probe["song_id"])
        track = int(probe["track"])
        source_stem_path = stem_path(song_path, track, audio_format)
        if not stem_is_valid(source_stem_path):
            raise FileNotFoundError(f"Missing or invalid probe stem: {source_stem_path}")

        mid_path = resolve_mid_path(probe["song_id"], pdmx_filepath)
        category = probe.get("category")

        for variant in variants:
            variant_id = variant["id"]
            sf_path, soundfont_id, fx_profile, pool_id = resolve_render_settings(
                phase=phase,
                variant=variant,
                grid_cfg=grid_cfg,
                category=category,
                catalog=catalog,
                winners_path=winners_path,
                soundfont_dir=soundfont_dir,
            )
            out_path = variant_output_path(
                output_dir,
                variant_id,
                song_path,
                source_dir,
                track,
                audio_format,
            )
            if out_path.exists():
                continue

            tasks.append({
                "phase": phase,
                "mid_path": mid_path,
                "track": track,
                "pool_id": pool_id,
                "variant_id": variant_id,
                "soundfont_id": soundfont_id,
                "soundfont_filepath": sf_path,
                "fx_profile": fx_profile,
                "stem_id": probe["id"],
                "category": category,
                "song_path": str(song_path),
                "out_path": out_path,
                "audio_format": audio_format,
                "sample_seed": sample_seed,
            })
    return tasks


def build_manifest_rows(
    *,
    phase: str,
    source_dir: Path,
    output_dir: Path,
    probe_stems: list[dict],
    variants: list[dict],
    grid_cfg: dict,
    audio_format: str,
    catalog: dict,
    winners_path: Path,
    soundfont_dir: str | Path,
) -> list[dict]:
    rows = []
    for probe in probe_stems:
        song_path = song_path_from_id(source_dir, probe["song_id"])
        track = int(probe["track"])
        category = probe.get("category")
        for variant in variants:
            variant_id = variant["id"]
            sf_path, soundfont_id, fx_profile, pool_id = resolve_render_settings(
                phase=phase,
                variant=variant,
                grid_cfg=grid_cfg,
                category=category,
                catalog=catalog,
                winners_path=winners_path,
                soundfont_dir=soundfont_dir,
            )
            out_path = variant_output_path(
                output_dir,
                variant_id,
                song_path,
                source_dir,
                track,
                audio_format,
            )
            rows.append({
                "phase": phase,
                "variant_id": variant_id,
                "soundfont_id": soundfont_id,
                "soundfont_file": Path(sf_path).name,
                "fx_profile": fx_profile or "",
                "pool_id": pool_id or "",
                "program": "",
                "gm_class": "",
                "stem_id": probe["id"],
                "category": category,
                "path": str(song_path),
                "track": track,
                "out_path": str(out_path),
            })
    return rows


def _manifest_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    df["program"] = pd.to_numeric(df["program"], errors="coerce").astype("Int64")
    for col in (
        "phase",
        "variant_id",
        "soundfont_id",
        "soundfont_file",
        "fx_profile",
        "pool_id",
        "gm_class",
        "stem_id",
        "category",
        "path",
        "out_path",
    ):
        df[col] = df[col].astype("string")
    df["track"] = df["track"].astype("int64")
    return df


def _read_manifest_df(manifest_path: Path) -> pd.DataFrame:
    return _manifest_dataframe(pd.read_csv(manifest_path).to_dict("records"))


def write_manifest(output_dir: Path, rows: list[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME
    _manifest_dataframe(rows).to_csv(manifest_path, index=False)
    return manifest_path


def _run_task(task: dict) -> dict:
    program, gm_class = render_probe_stem(
        mid_path=task["mid_path"],
        track=task["track"],
        pool_id=task["pool_id"],
        category=task.get("category"),
        soundfont_filepath=task["soundfont_filepath"],
        fx_profile=task["fx_profile"],
        sample_seed=task["sample_seed"],
        song_path=task["song_path"],
        out_path=task["out_path"],
        audio_format=task["audio_format"],
    )
    return {
        "phase": task["phase"],
        "variant_id": task["variant_id"],
        "soundfont_id": task["soundfont_id"],
        "soundfont_file": Path(task["soundfont_filepath"]).name,
        "fx_profile": task["fx_profile"] or "",
        "pool_id": task["pool_id"] or "",
        "program": program,
        "gm_class": gm_class,
        "stem_id": task["stem_id"],
        "category": task["category"],
        "path": task["song_path"],
        "track": task["track"],
        "out_path": str(task["out_path"]),
    }


def run_patch_sweep(
    *,
    phase: str,
    source_dir: Path,
    output_dir: Path,
    probe_stems_path: Path,
    grid_path: Path,
    winners_path: Path,
    soundfont_dir: str | Path,
    jobs: int = 1,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    sample_seed: int = ABLATION_SAMPLE_SEED,
    pdmx_filepath: str = PDMX_FILEPATH,
    limit_stems: int | None = None,
    limit_variants: int | None = None,
) -> pd.DataFrame:
    if phase not in PHASES:
        raise ValueError(f"Unknown phase: {phase}")

    source_dir = source_dir.resolve()
    output_dir = phase_output_dir(output_dir.resolve(), phase)
    output_dir.mkdir(parents=True, exist_ok=True)

    _require_phase_prerequisites(phase, winners_path)

    probe_cfg = load_yaml(probe_stems_path)
    grid_cfg = load_yaml(grid_path)
    catalog = load_soundfont_catalog()

    probe_stems = list(probe_cfg["stems"])
    validate_probe_stems(probe_stems)
    variants = list(grid_cfg["variants"])
    if limit_stems is not None:
        probe_stems = probe_stems[:limit_stems]
    if limit_variants is not None:
        variants = variants[:limit_variants]

    for variant in variants:
        if phase == PHASE1 and "soundfont_id" not in variant:
            raise ValueError(f"Phase 1 variant missing soundfont_id: {variant}")
        if phase == PHASE2 and "fx_profile" not in variant:
            raise ValueError(f"Phase 2 variant missing fx_profile: {variant}")
        if phase == PHASE3 and "pool_id" not in variant:
            raise ValueError(f"Phase 3 variant missing pool_id: {variant}")

    tasks = build_sweep_tasks(
        phase=phase,
        source_dir=source_dir,
        output_dir=output_dir,
        probe_stems=probe_stems,
        variants=variants,
        grid_cfg=grid_cfg,
        audio_format=audio_format,
        sample_seed=sample_seed,
        catalog=catalog,
        winners_path=winners_path,
        soundfont_dir=soundfont_dir,
        pdmx_filepath=pdmx_filepath,
    )

    manifest_rows = build_manifest_rows(
        phase=phase,
        source_dir=source_dir,
        output_dir=output_dir,
        probe_stems=probe_stems,
        variants=variants,
        grid_cfg=grid_cfg,
        audio_format=audio_format,
        catalog=catalog,
        winners_path=winners_path,
        soundfont_dir=soundfont_dir,
    )
    manifest_path = write_manifest(output_dir, manifest_rows)

    print(f"Patch sweep phase: {phase}")
    print(f"Patch sweep source: {source_dir}")
    print(f"Patch sweep output: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Probe stems: {len(probe_stems)}")
    print(f"Variants: {len(variants)}")
    print(f"Tasks queued: {len(tasks)} (skipped existing outputs)")

    completed_rows = []
    if tasks:
        progress_desc = f"Patch sweep ({phase})"
        if jobs <= 1:
            completed_rows = list(
                tqdm(
                    map(_run_task, tasks),
                    total=len(tasks),
                    desc=progress_desc,
                    unit="stem",
                )
            )
        else:
            with multiprocessing.Pool(processes=jobs) as pool:
                completed_rows = list(
                    tqdm(
                        pool.imap(_run_task, tasks, chunksize=CHUNK_SIZE),
                        total=len(tasks),
                        desc=progress_desc,
                        unit="stem",
                    )
                )

    if completed_rows:
        manifest_df = _read_manifest_df(manifest_path)
        for row in completed_rows:
            mask = (
                (manifest_df["variant_id"] == row["variant_id"])
                & (manifest_df["stem_id"] == row["stem_id"])
            )
            manifest_df.loc[mask, "program"] = int(row["program"])
            manifest_df.loc[mask, "gm_class"] = str(row["gm_class"])
        manifest_df.to_csv(manifest_path, index=False)

    return _read_manifest_df(manifest_path)


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="Run phased Slakh-style patch sweep on probe stems.",
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=list(PHASES),
        help="Tuning phase to run.",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        type=Path,
        help="Raw ablation dir (default: dev/ablations/basic).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Patch sweep root (phase subdir is appended automatically).",
    )
    parser.add_argument(
        "--probe-stems",
        default=DEFAULT_PROBE_STEMS,
        type=Path,
        help="Probe stem manifest YAML.",
    )
    parser.add_argument(
        "--grid",
        default=None,
        type=Path,
        help="Phase grid YAML (default: grids/phaseN_*.yaml for --phase).",
    )
    parser.add_argument(
        "--winners",
        default=EXPERIMENT_DIR / "winners.yaml",
        type=Path,
        help="Per-phase winners YAML (phase 2/3 read prior winners).",
    )
    parser.add_argument(
        "--soundfont-dir",
        default=SOUNDFONT_DIR,
        type=Path,
        help="Directory containing candidate .sf2 files.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        default=max(1, int(multiprocessing.cpu_count() / 4)),
        type=int,
        help="CPU workers for parallel renders.",
    )
    parser.add_argument("--limit-stems", default=None, type=int)
    parser.add_argument("--limit-variants", default=None, type=int)
    parser.add_argument("--sample-seed", default=ABLATION_SAMPLE_SEED, type=int)
    parser.add_argument(
        "--mp3",
        action="store_true",
        help="Write MP3 stems.",
    )
    return parser.parse_args(args=args, namespace=namespace)


def main():
    args = parse_args()
    grid_path = args.grid or PHASE_GRID_FILES[args.phase]

    run_patch_sweep(
        phase=args.phase,
        source_dir=args.source_dir or default_source_dir(),
        output_dir=args.output_dir or default_output_dir(),
        probe_stems_path=args.probe_stems,
        grid_path=grid_path,
        winners_path=args.winners,
        soundfont_dir=args.soundfont_dir,
        jobs=args.jobs,
        audio_format=synthesis_audio_format(args.mp3),
        sample_seed=args.sample_seed,
    )


if __name__ == "__main__":
    main()
