"""Prepare and validate ablation listening trial clips."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from experiments.ablation_listening.paths import (
    DEFAULT_CLIPS_DIR,
    DEFAULT_MANIFEST,
)
from experiments.preset_sweep.diverse_stems import (
    DEFAULT_CLIP_SECONDS,
    DEFAULT_MIN_RMS,
    clip_stem_waveform,
    find_audible_clip_start,
    is_silent,
)
from experiments.probe_stems import PROBE_CATEGORIES, load_probe_stems
from shared.config import DATA_DIR_NAME, STEMS_FILE_NAME
from synthesis.audio import (
    mixture_path,
    stem_is_valid,
    stem_path,
    write_audio,
)
from synthesis.listening.catalog import (
    CONDITION_ORDER,
    default_ablations_dir,
    song_id_from_path,
)
from synthesis.patches import resolve_probe_category

STEM_TRIAL_CATEGORIES = (
    "piano",
    "drums",
    "strings",
    "wind",
    "voice",
    "polyphonic",
)

MIXTURE_TRIAL_COUNT = 4
STEM_TRIAL_COUNT = len(STEM_TRIAL_CATEGORIES)


def condition_roots(ablations_dir: Path) -> dict[str, Path]:
    return {
        "basic": ablations_dir / "basic",
        "basic_realify": ablations_dir / "basic_realify",
        "slakh": ablations_dir / "slakh",
        "slakh_realify": ablations_dir / "slakh_realify",
    }


def _song_dir(root: Path, song_id: str) -> Path:
    return root / DATA_DIR_NAME / song_id


def _detect_format(song_dir: Path) -> str | None:
    for ext in ("mp3", "flac"):
        if mixture_path(song_dir, ext).is_file():
            return ext
        if stem_path(song_dir, 0, ext).is_file():
            return ext
    return None


def song_has_all_conditions(
    song_path: str,
    *,
    roots: dict[str, Path],
    stems_df: pd.DataFrame,
    require_mixture: bool,
) -> bool:
    song_id = song_id_from_path(song_path)
    group = stems_df[stems_df["path"] == song_path]
    if group.empty:
        return False

    for root in roots.values():
        song_dir = _song_dir(root, song_id)
        audio_format = _detect_format(song_dir)
        if audio_format is None:
            return False
        if require_mixture:
            if not mixture_path(song_dir, audio_format).is_file():
                return False
        else:
            for track in group["track"]:
                if not stem_is_valid(stem_path(song_dir, int(track), audio_format)):
                    return False
    return True


def complete_song_paths(
    ablations_dir: Path,
    *,
    require_mixture: bool,
) -> list[str]:
    roots = condition_roots(ablations_dir)
    basic_dir = roots["basic"]
    songs_df = pd.read_csv(basic_dir / f"{DATA_DIR_NAME}.csv")
    stems_df = pd.read_csv(basic_dir / f"{STEMS_FILE_NAME}.csv")
    return [
        str(row["path"])
        for _, row in songs_df.iterrows()
        if song_has_all_conditions(
            str(row["path"]),
            roots=roots,
            stems_df=stems_df,
            require_mixture=require_mixture,
        )
    ]


def _stem_row_category(row: pd.Series) -> str | None:
    category = resolve_probe_category(
        program=int(row.get("program", 0) or 0),
        is_drum=bool(row.get("is_drum", False)),
        track_name=row.get("name"),
    )
    if category not in PROBE_CATEGORIES:
        return None
    return category


def select_mixture_trials(
    ablations_dir: Path,
    *,
    count: int = MIXTURE_TRIAL_COUNT,
    seed: int = 42,
) -> list[dict]:
    paths = complete_song_paths(ablations_dir, require_mixture=True)
    if len(paths) < count:
        raise RuntimeError(
            f"Need {count} mixture trials but only {len(paths)} songs have all conditions."
        )
    rng = random.Random(seed)
    rng.shuffle(paths)
    trials = []
    for index, song_path in enumerate(paths[:count]):
        trials.append({
            "id": f"mix_{index + 1:02d}",
            "type": "mixture",
            "song_id": song_id_from_path(song_path),
            "song_path": song_path,
            "track": None,
            "category": None,
        })
    return trials


def select_stem_trials(
    ablations_dir: Path,
    *,
    categories: tuple[str, ...] = STEM_TRIAL_CATEGORIES,
    seed: int = 42,
) -> list[dict]:
    roots = condition_roots(ablations_dir)
    basic_dir = roots["basic"]
    stems_df = pd.read_csv(basic_dir / f"{STEMS_FILE_NAME}.csv")
    complete_paths = set(complete_song_paths(ablations_dir, require_mixture=False))
    rng = random.Random(seed)

    probe_by_category: dict[str, list[dict]] = {cat: [] for cat in categories}
    for stem in load_probe_stems():
        if stem["category"] in probe_by_category:
            probe_by_category[stem["category"]].append(stem)

    trials: list[dict] = []
    used_keys: set[tuple[str, int]] = set()

    for category in categories:
        candidates: list[dict] = []
        for _, row in stems_df.iterrows():
            song_path = str(row["path"])
            if song_path not in complete_paths:
                continue
            if _stem_row_category(row) != category:
                continue
            track = int(row["track"])
            key = (song_id_from_path(song_path), track)
            if key in used_keys:
                continue
            candidates.append({
                "song_path": song_path,
                "song_id": key[0],
                "track": track,
                "category": category,
                "note": str(row.get("name") or "").strip() or None,
            })

        preferred = probe_by_category.get(category) or []
        rng.shuffle(preferred)
        picked = None
        for probe in preferred:
            key = (probe["song_id"], int(probe["track"]))
            if key in used_keys:
                continue
            if any(
                c["song_id"] == key[0] and c["track"] == key[1]
                for c in candidates
            ):
                picked = {
                    "song_path": next(
                        c["song_path"]
                        for c in candidates
                        if c["song_id"] == key[0] and c["track"] == key[1]
                    ),
                    "song_id": key[0],
                    "track": key[1],
                    "category": category,
                    "note": probe.get("note"),
                }
                break

        if picked is None and candidates:
            rng.shuffle(candidates)
            picked = candidates[0]

        if picked is None:
            raise RuntimeError(f"No eligible stem trial for category {category!r}")

        used_keys.add((picked["song_id"], picked["track"]))
        trials.append({
            "id": f"stem_{category}",
            "type": "stem",
            "song_id": picked["song_id"],
            "song_path": picked["song_path"],
            "track": picked["track"],
            "category": category,
            "note": picked.get("note"),
        })

    return trials


def _read_mixture_clip(
    song_dir: Path,
    audio_format: str,
    *,
    clip_seconds: float,
    start_seconds: float,
) -> np.ndarray:
    mix_path = mixture_path(song_dir, audio_format)
    if mix_path.is_file():
        return clip_stem_waveform(
            mix_path,
            clip_seconds=clip_seconds,
            start_seconds=start_seconds,
        )

    stems_df_path = song_dir.parents[2] / f"{STEMS_FILE_NAME}.csv"
    raise FileNotFoundError(
        f"Missing mixture for clip prep: {mix_path} (stems table: {stems_df_path})"
    )


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

    for condition_id in CONDITION_ORDER:
        song_dir = _song_dir(roots[condition_id], trial["song_id"])
        if trial["type"] == "mixture":
            waveform = _read_mixture_clip(
                song_dir,
                audio_format,
                clip_seconds=clip_seconds,
                start_seconds=start_seconds,
            )
        else:
            source_path = stem_path(song_dir, int(trial["track"]), audio_format)
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing stem: {source_path}")
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


def is_silent_from_waveform(waveform: np.ndarray, *, min_rms: float = DEFAULT_MIN_RMS) -> bool:
    if waveform.size == 0:
        return True
    return float(np.sqrt(np.mean(np.square(waveform)))) < min_rms


def build_manifest(
    ablations_dir: Path,
    *,
    seed: int = 42,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
) -> list[dict]:
    mixture_trials = select_mixture_trials(ablations_dir, seed=seed)
    stem_trials = select_stem_trials(ablations_dir, seed=seed)
    return mixture_trials + stem_trials


def prepare_clips(
    ablations_dir: Path,
    clips_dir: Path,
    manifest_path: Path,
    *,
    seed: int = 42,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
) -> dict:
    ablations_dir = ablations_dir.resolve()
    clips_dir = clips_dir.resolve()
    manifest_path = manifest_path.resolve()
    clips_dir.mkdir(parents=True, exist_ok=True)

    trials = build_manifest(ablations_dir, seed=seed, clip_seconds=clip_seconds)
    prepared = []
    for trial in trials:
        prepared.append(
            write_trial_clips(
                trial,
                ablations_dir,
                clips_dir,
                clip_seconds=clip_seconds,
            )
        )

    doc = {
        "test_id": "ablation_listening_v1",
        "clip_seconds": clip_seconds,
        "seed": seed,
        "ablations_dir": str(ablations_dir),
        "trials": prepared,
    }
    with open(manifest_path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    return doc


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Prepare clipped audio for the ablation listening test.",
    )
    parser.add_argument(
        "--ablations-dir",
        default=str(default_ablations_dir()),
        type=Path,
    )
    parser.add_argument("--clips-dir", default=DEFAULT_CLIPS_DIR, type=Path)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--clip-seconds",
        default=DEFAULT_CLIP_SECONDS,
        type=float,
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
    )
    print(f"Prepared {len(doc['trials'])} trials")
    print(f"Manifest: {opts.manifest.resolve()}")
    print(f"Clips: {opts.clips_dir.resolve()}")


if __name__ == "__main__":
    main()
