"""Prepare and validate ablation listening trial clips (8 conditions, per-category stems)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from experiments.ablation_listening.conditions import (
    ABLATION_MUSHRA_CONDITIONS,
    DEFAULT_STEMS_PER_CATEGORY,
    STEM_TRIAL_CATEGORIES,
    condition_roots,
)
from experiments.ablation_listening.paths import (
    DEFAULT_CLIPS_DIR,
    DEFAULT_MANIFEST,
)
from experiments.preset_sweep.diverse_stems import (
    DEFAULT_CLIP_SECONDS,
    DEFAULT_MIN_RMS,
    clip_stem_waveform,
    find_audible_clip_start,
)
from experiments.probe_stems import load_probe_stems
from shared.config import DATA_DIR_NAME, STEMS_FILE_NAME
from synthesis.audio import (
    mixture_path,
    stem_is_valid,
    stem_path,
    write_audio,
)
from synthesis.listening.catalog import default_ablations_dir, song_id_from_path
from synthesis.patches import resolve_probe_category


def _song_dir(root: Path, song_id: str) -> Path:
    return root / DATA_DIR_NAME / song_id


def _detect_format(song_dir: Path) -> str | None:
    for ext in ("mp3", "flac"):
        if mixture_path(song_dir, ext).is_file():
            return ext
        if any(song_dir.glob(f"stem_*.{ext}")):
            return ext
    return None


def stem_has_all_conditions(
    song_id: str,
    track: int,
    roots: dict[str, Path],
) -> bool:
    """True when ``stem_{track}`` exists under every ablation condition."""
    for root in roots.values():
        song_dir = _song_dir(root, song_id)
        audio_format = _detect_format(song_dir)
        if audio_format is None:
            return False
        if not stem_is_valid(stem_path(song_dir, track, audio_format)):
            return False
    return True


def song_has_all_mixtures(song_id: str, roots: dict[str, Path]) -> bool:
    for root in roots.values():
        song_dir = _song_dir(root, song_id)
        audio_format = _detect_format(song_dir)
        if audio_format is None:
            return False
        if not mixture_path(song_dir, audio_format).is_file():
            return False
    return True


def _stem_row_category(row: pd.Series) -> str | None:
    category = resolve_probe_category(
        program=int(row.get("program", 0) or 0),
        is_drum=bool(row.get("is_drum", False)),
        track_name=row.get("name"),
    )
    if category not in STEM_TRIAL_CATEGORIES:
        return None
    return category


def select_stem_trials(
    ablations_dir: Path,
    *,
    categories: tuple[str, ...] = STEM_TRIAL_CATEGORIES,
    stems_per_category: int = DEFAULT_STEMS_PER_CATEGORY,
    seed: int = 43,
) -> list[dict]:
    """Pick ``stems_per_category`` eligible stems for each listening category."""
    roots = condition_roots(ablations_dir)
    basic_dir = roots["basic"]
    stems_df = pd.read_csv(basic_dir / f"{STEMS_FILE_NAME}.csv")
    rng = random.Random(seed)

    probe_by_category: dict[str, list[dict]] = {cat: [] for cat in categories}
    for stem in load_probe_stems():
        if stem.get("category") in probe_by_category:
            probe_by_category[stem["category"]].append(stem)

    trials: list[dict] = []
    used_keys: set[tuple[str, int]] = set()

    for category in categories:
        candidates: list[dict] = []
        for _, row in stems_df.iterrows():
            if _stem_row_category(row) != category:
                continue
            song_path = str(row["path"])
            song_id = song_id_from_path(song_path)
            track = int(row["track"])
            key = (song_id, track)
            if key in used_keys:
                continue
            if not stem_has_all_conditions(song_id, track, roots):
                continue
            candidates.append({
                "song_path": song_path,
                "song_id": song_id,
                "track": track,
                "category": category,
                "note": str(row.get("name") or "").strip() or None,
            })

        # Prefer probe stems that are eligible, then fill from the rest.
        preferred_keys = {
            (p["song_id"], int(p["track"]))
            for p in (probe_by_category.get(category) or [])
        }
        preferred = [c for c in candidates if (c["song_id"], c["track"]) in preferred_keys]
        others = [c for c in candidates if (c["song_id"], c["track"]) not in preferred_keys]
        rng.shuffle(preferred)
        rng.shuffle(others)
        ordered = preferred + others

        if len(ordered) < stems_per_category:
            raise RuntimeError(
                f"Need {stems_per_category} stem trials for category {category!r}, "
                f"but only {len(ordered)} stems exist under all "
                f"{len(ABLATION_MUSHRA_CONDITIONS)} conditions in {ablations_dir}."
            )

        for index, picked in enumerate(ordered[:stems_per_category]):
            used_keys.add((picked["song_id"], picked["track"]))
            suffix = f"_{index + 1:02d}" if stems_per_category > 1 else ""
            trials.append({
                "id": f"stem_{category}{suffix}",
                "type": "stem",
                "song_id": picked["song_id"],
                "song_path": picked["song_path"],
                "track": picked["track"],
                "category": category,
                "note": picked.get("note"),
            })

    return trials


def select_mixture_trials(
    ablations_dir: Path,
    *,
    count: int = 4,
    seed: int = 43,
) -> list[dict]:
    roots = condition_roots(ablations_dir)
    songs_df = pd.read_csv(roots["basic"] / f"{DATA_DIR_NAME}.csv")
    paths = [
        str(row["path"])
        for _, row in songs_df.iterrows()
        if song_has_all_mixtures(song_id_from_path(str(row["path"])), roots)
    ]
    if len(paths) < count:
        raise RuntimeError(
            f"Need {count} mixture trials but only {len(paths)} songs have mixtures "
            f"under all conditions (mixtures are optional via synthesis.mix)."
        )
    rng = random.Random(seed)
    rng.shuffle(paths)
    return [
        {
            "id": f"mix_{index + 1:02d}",
            "type": "mixture",
            "song_id": song_id_from_path(song_path),
            "song_path": song_path,
            "track": None,
            "category": None,
        }
        for index, song_path in enumerate(paths[:count])
    ]


def _clip_reference_path(
    trial: dict,
    ablations_dir: Path,
    *,
    clip_seconds: float,
) -> tuple[Path, str, float]:
    roots = condition_roots(ablations_dir)
    ref_root = roots["basic"]
    song_dir = _song_dir(ref_root, trial["song_id"])
    audio_format = _detect_format(song_dir)
    if audio_format is None:
        raise FileNotFoundError(f"No audio in reference song dir: {song_dir}")

    if trial["type"] == "mixture":
        source_path = mixture_path(song_dir, audio_format)
    else:
        source_path = stem_path(song_dir, int(trial["track"]), audio_format)

    if not source_path.is_file():
        raise FileNotFoundError(f"Missing reference audio: {source_path}")

    start = find_audible_clip_start(
        source_path,
        clip_seconds=clip_seconds,
        min_rms=DEFAULT_MIN_RMS,
    )
    if start is None:
        raise RuntimeError(f"No audible {clip_seconds}s clip in {source_path}")
    return source_path, audio_format, start


def is_silent_from_waveform(waveform: np.ndarray, *, min_rms: float = DEFAULT_MIN_RMS) -> bool:
    if waveform.size == 0:
        return True
    return float(np.sqrt(np.mean(np.square(waveform)))) < min_rms


def write_trial_clips(
    trial: dict,
    ablations_dir: Path,
    clips_dir: Path,
    *,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
) -> dict:
    roots = condition_roots(ablations_dir)
    _, audio_format, start_seconds = _clip_reference_path(
        trial,
        ablations_dir,
        clip_seconds=clip_seconds,
    )

    out_dir = clips_dir / trial["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    for condition_id in ABLATION_MUSHRA_CONDITIONS:
        song_dir = _song_dir(roots[condition_id], trial["song_id"])
        if trial["type"] == "mixture":
            source_path = mixture_path(song_dir, audio_format)
        else:
            source_path = stem_path(song_dir, int(trial["track"]), audio_format)
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing audio: {source_path}")
        waveform = clip_stem_waveform(
            source_path,
            clip_seconds=clip_seconds,
            start_seconds=start_seconds,
        )
        if waveform.size == 0 or is_silent_from_waveform(waveform):
            raise RuntimeError(f"Silent clip for {trial['id']} / {condition_id}")

        out_path = out_dir / f"{condition_id}.{audio_format}"
        write_audio(torch.from_numpy(waveform), out_path, audio_format)
        written[condition_id] = str(out_path.relative_to(clips_dir))

    return {
        **trial,
        "clip_seconds": clip_seconds,
        "clip_start_seconds": start_seconds,
        "audio_format": audio_format,
        "conditions": written,
    }


def build_manifest(
    ablations_dir: Path,
    *,
    seed: int = 43,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    stems_per_category: int = DEFAULT_STEMS_PER_CATEGORY,
    include_mixtures: bool = False,
    mixture_count: int = 4,
) -> list[dict]:
    trials = select_stem_trials(
        ablations_dir,
        stems_per_category=stems_per_category,
        seed=seed,
    )
    if include_mixtures:
        trials = select_mixture_trials(
            ablations_dir, count=mixture_count, seed=seed,
        ) + trials
    return trials


def prepare_clips(
    ablations_dir: Path,
    clips_dir: Path,
    manifest_path: Path,
    *,
    seed: int = 43,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    stems_per_category: int = DEFAULT_STEMS_PER_CATEGORY,
    include_mixtures: bool = False,
) -> dict:
    ablations_dir = ablations_dir.resolve()
    clips_dir = clips_dir.resolve()
    manifest_path = manifest_path.resolve()
    clips_dir.mkdir(parents=True, exist_ok=True)

    trials = build_manifest(
        ablations_dir,
        seed=seed,
        clip_seconds=clip_seconds,
        stems_per_category=stems_per_category,
        include_mixtures=include_mixtures,
    )
    prepared = [
        write_trial_clips(trial, ablations_dir, clips_dir, clip_seconds=clip_seconds)
        for trial in trials
    ]

    doc = {
        "test_id": "ablation_listening_v2_8cond",
        "clip_seconds": clip_seconds,
        "seed": seed,
        "stems_per_category": stems_per_category,
        "include_mixtures": include_mixtures,
        "conditions": list(ABLATION_MUSHRA_CONDITIONS),
        "categories": list(STEM_TRIAL_CATEGORIES),
        "ablations_dir": str(ablations_dir),
        "trials": prepared,
    }
    with open(manifest_path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    return doc


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=(
            "Prepare 10s clips for the 8-condition ablation MUSHRA "
            "(stem trials × listening categories)."
        ),
    )
    parser.add_argument(
        "--ablations-dir",
        default=str(default_ablations_dir()),
        type=Path,
    )
    parser.add_argument("--clips-dir", default=DEFAULT_CLIPS_DIR, type=Path)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument("--seed", default=43, type=int)
    parser.add_argument("--clip-seconds", default=DEFAULT_CLIP_SECONDS, type=float)
    parser.add_argument(
        "--stems-per-category",
        default=DEFAULT_STEMS_PER_CATEGORY,
        type=int,
        help="Stem trials per listening category (default: 2).",
    )
    parser.add_argument(
        "--include-mixtures",
        action="store_true",
        help="Also include mixture trials (requires mixture files under all conditions).",
    )
    return parser.parse_args(args)


def main(args=None) -> None:
    opts = parse_args(args)
    doc = prepare_clips(
        opts.ablations_dir,
        opts.clips_dir,
        opts.manifest,
        seed=opts.seed,
        clip_seconds=opts.clip_seconds,
        stems_per_category=opts.stems_per_category,
        include_mixtures=opts.include_mixtures,
    )
    n_stem = sum(1 for t in doc["trials"] if t["type"] == "stem")
    n_mix = sum(1 for t in doc["trials"] if t["type"] == "mixture")
    print(
        f"Prepared {len(doc['trials'])} trials "
        f"({n_stem} stem, {n_mix} mixture) × "
        f"{len(ABLATION_MUSHRA_CONDITIONS)} conditions"
    )
    print(f"Manifest: {opts.manifest.resolve()}")
    print(f"Clips: {opts.clips_dir.resolve()}")


if __name__ == "__main__":
    main()
