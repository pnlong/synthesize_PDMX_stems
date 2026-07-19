"""Select diverse non-silent probe stems and write fixed-length clips for audits."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from experiments.paths import DEFAULT_PROBE_STEMS
from experiments.probe_stems import PROBE_CATEGORIES, load_probe_stems
from shared.config import DATA_DIR_NAME, SAMPLE_RATE, STEMS_FILE_NAME
from synthesis.audio import (
    stem_duration_seconds,
    stem_is_valid,
    stem_n_samples,
    stem_path,
    write_audio,
)
from synthesis.listening.catalog import song_id_from_path
from synthesis.patches import resolve_probe_category

DEFAULT_CLIP_SECONDS = 10.0
DEFAULT_DIVERSE_PER_CATEGORY = 5
DEFAULT_MIN_RMS = 0.01


def stem_rms(path: Path) -> float:
    """Root-mean-square amplitude of a stem file."""
    if not path.is_file():
        return 0.0
    try:
        audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
    except (RuntimeError, OSError, ValueError):
        return 0.0
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def is_silent(path: Path, *, min_rms: float = DEFAULT_MIN_RMS) -> bool:
    return stem_rms(path) < min_rms


def probe_stem_keys(probe_stems_path: Path = DEFAULT_PROBE_STEMS) -> set[tuple[str, int]]:
    return {
        (str(stem["song_id"]), int(stem["track"]))
        for stem in load_probe_stems(probe_stems_path)
    }


def _stem_row_category(row: pd.Series) -> str | None:
    category = resolve_probe_category(
        program=int(row.get("program", 0) or 0),
        is_drum=bool(row.get("is_drum", False)),
        track_name=row.get("name"),
    )
    if category not in PROBE_CATEGORIES:
        return None
    return category


def select_diverse_stems(
    source_dir: Path,
    *,
    per_category: int = DEFAULT_DIVERSE_PER_CATEGORY,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    min_rms: float = DEFAULT_MIN_RMS,
    exclude_keys: set[tuple[str, int]] | None = None,
    seed: int = 0,
) -> list[dict]:
    """Pick diverse stems from the ablation dataset for noise-audit listening."""
    source_dir = source_dir.resolve()
    stems_csv = source_dir / f"{STEMS_FILE_NAME}.csv"
    if not stems_csv.is_file():
        raise FileNotFoundError(f"Missing stems table: {stems_csv}")

    exclude_keys = exclude_keys or probe_stem_keys()
    stems = pd.read_csv(stems_csv)
    min_samples = int(clip_seconds * SAMPLE_RATE)

    candidates: dict[str, list[dict]] = {category: [] for category in PROBE_CATEGORIES}
    for _, row in stems.iterrows():
        song_path = Path(str(row["path"]))
        track = int(row["track"])
        song_id = song_id_from_path(song_path)
        key = (song_id, track)
        if key in exclude_keys:
            continue

        category = _stem_row_category(row)
        if category is None:
            continue

        audio_path = None
        audio_format = None
        for ext in ("flac", "mp3"):
            candidate_path = stem_path(song_path, track, ext)
            if stem_is_valid(candidate_path):
                audio_path = candidate_path
                audio_format = ext
                break
        if audio_path is None:
            continue

        if stem_n_samples(audio_path) < min_samples:
            continue
        if is_silent(audio_path, min_rms=min_rms):
            continue

        candidates[category].append({
            "id": f"{category}_{song_id.replace('/', '_')}_t{track}",
            "category": category,
            "song_id": song_id,
            "track": track,
            "note": str(row.get("name") or "").strip() or None,
            "rms": round(stem_rms(audio_path), 4),
            "duration_seconds": round(stem_duration_seconds(audio_path), 2),
            "audio_format": audio_format,
        })

    rng = random.Random(seed)
    selected: list[dict] = []
    problems: list[str] = []
    for category in PROBE_CATEGORIES:
        pool = candidates[category]
        if not pool:
            problems.append(f"{category}: no eligible stems")
            continue

        rng.shuffle(pool)
        by_song: dict[str, list[dict]] = {}
        for stem in pool:
            by_song.setdefault(stem["song_id"], []).append(stem)

        picked: list[dict] = []
        song_ids = list(by_song)
        rng.shuffle(song_ids)
        for song_id in song_ids:
            if len(picked) >= per_category:
                break
            picked.append(by_song[song_id][0])

        if len(picked) < per_category:
            remaining = [stem for stem in pool if stem not in picked]
            rng.shuffle(remaining)
            for stem in remaining:
                if len(picked) >= per_category:
                    break
                if stem not in picked:
                    picked.append(stem)

        if len(picked) < per_category:
            problems.append(f"{category}: only {len(picked)} eligible stems")

        selected.extend(picked[:per_category])

    if problems:
        raise ValueError(
            "Could not select enough diverse stems: " + "; ".join(problems)
        )

    return selected


def clip_stem_waveform(
    path: Path,
    *,
    clip_seconds: float,
    start_seconds: float = 0.0,
) -> np.ndarray:
    start_frame = int(start_seconds * SAMPLE_RATE)
    max_frames = int(clip_seconds * SAMPLE_RATE)
    audio, _ = sf.read(
        str(path),
        start=start_frame,
        frames=max_frames,
        dtype="float32",
        always_2d=True,
    )
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    return np.asarray(audio.T, dtype=np.float32)


def clip_rms(path: Path, *, clip_seconds: float, start_seconds: float = 0.0) -> float:
    audio = clip_stem_waveform(
        path,
        clip_seconds=clip_seconds,
        start_seconds=start_seconds,
    )
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def find_audible_clip_start(
    path: Path,
    *,
    clip_seconds: float,
    min_rms: float = DEFAULT_MIN_RMS,
    hop_seconds: float = 1.0,
    min_start_seconds: float = 0.0,
) -> float | None:
    """Return a start offset (seconds) where clip_seconds has RMS >= min_rms."""
    if not stem_is_valid(path):
        return None

    clip_frames = int(clip_seconds * SAMPLE_RATE)
    total_frames = stem_n_samples(path)
    if total_frames < clip_frames:
        return None

    hop_frames = max(1, int(hop_seconds * SAMPLE_RATE))
    first_frame = int(min_start_seconds * SAMPLE_RATE)
    for start_frame in range(first_frame, total_frames - clip_frames + 1, hop_frames):
        start_seconds = start_frame / SAMPLE_RATE
        if clip_rms(path, clip_seconds=clip_seconds, start_seconds=start_seconds) >= min_rms:
            return start_seconds
    return None


def write_diverse_clip_dataset(
    source_dir: Path,
    stems: list[dict],
    clips_dir: Path,
    *,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
) -> Path:
    """Write a mini ablation tree of clipped reference stems for sweep/listening."""
    import torch

    source_dir = source_dir.resolve()
    clips_dir = clips_dir.resolve()
    clips_dir.mkdir(parents=True, exist_ok=True)

    songs = pd.read_csv(source_dir / f"{DATA_DIR_NAME}.csv")
    stems_df = pd.read_csv(source_dir / f"{STEMS_FILE_NAME}.csv")

    used_song_paths: set[str] = set()
    clip_rows: list[dict] = []
    manifest_stems: list[dict] = []

    for stem in stems:
        song_id = stem["song_id"]
        track = int(stem["track"])
        audio_format = stem["audio_format"]
        song_path = source_dir / DATA_DIR_NAME / song_id
        source_stem_path = stem_path(song_path, track, audio_format)
        out_song_dir = clips_dir / DATA_DIR_NAME / song_id
        out_stem_path = stem_path(out_song_dir, track, audio_format)
        clip_start_seconds = float(stem.get("clip_start_seconds") or 0.0)

        waveform = clip_stem_waveform(
            source_stem_path,
            clip_seconds=clip_seconds,
            start_seconds=clip_start_seconds,
        )
        write_audio(torch.from_numpy(waveform), out_stem_path, audio_format)

        used_song_paths.add(str(song_path))
        source_row = stems_df[
            (stems_df["path"] == str(song_path)) & (stems_df["track"] == track)
        ]
        if source_row.empty:
            raise ValueError(f"No stems.csv row for {song_path} track {track}")
        clip_row = source_row.iloc[0].to_dict()
        clip_row["path"] = str(out_song_dir)
        clip_rows.append(clip_row)
        manifest_stems.append({
            **stem,
            "clip_seconds": clip_seconds,
            "clip_start_seconds": clip_start_seconds,
            "source_stem_path": str(source_stem_path),
            "clip_stem_path": str(out_stem_path),
        })

    songs_clip = songs[songs["path"].isin(used_song_paths)].copy()
    songs_clip["path"] = songs_clip["path"].map(
        lambda path: str(clips_dir / DATA_DIR_NAME / song_id_from_path(str(path)))
    )
    songs_clip.to_csv(
        clips_dir / f"{DATA_DIR_NAME}.csv",
        index=False,
    )
    pd.DataFrame(clip_rows).to_csv(clips_dir / f"{STEMS_FILE_NAME}.csv", index=False)

    manifest_path = clips_dir.parent / "diverse_stems.yaml"
    with open(manifest_path, "w") as f:
        yaml.safe_dump(
            {
                "clip_seconds": clip_seconds,
                "stems": manifest_stems,
            },
            f,
            sort_keys=False,
            default_flow_style=False,
        )
    return manifest_path


def load_diverse_stems_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    return list(doc.get("stems") or [])


def probe_clip_path(clips_dir: Path, probe: dict) -> Path:
    song_path = clips_dir / DATA_DIR_NAME / probe["song_id"]
    track = int(probe["track"])
    stem_format = str(probe.get("audio_format") or "flac")
    return stem_path(song_path, track, stem_format)


def probe_clip_usable(clips_dir: Path, probe: dict, *, min_rms: float = DEFAULT_MIN_RMS) -> bool:
    clip_path = probe_clip_path(clips_dir, probe)
    return stem_is_valid(clip_path) and not is_silent(clip_path, min_rms=min_rms)


def source_stem_path(source_dir: Path, probe: dict) -> Path:
    song_path = source_dir / DATA_DIR_NAME / probe["song_id"]
    track = int(probe["track"])
    stem_format = str(probe.get("audio_format") or "flac")
    return stem_path(song_path, track, stem_format)


def source_stem_clip_usable(
    song_path: Path,
    track: int,
    audio_format: str,
    *,
    clip_seconds: float,
    min_rms: float,
) -> bool:
    source_path = stem_path(song_path, track, audio_format)
    if not stem_is_valid(source_path):
        return False
    if stem_n_samples(source_path) < int(clip_seconds * SAMPLE_RATE):
        return False
    return clip_rms(source_path, clip_seconds=clip_seconds, start_seconds=0.0) >= min_rms


def build_diverse_candidates(
    source_dir: Path,
    *,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    min_rms: float = DEFAULT_MIN_RMS,
    exclude_keys: set[tuple[str, int]] | None = None,
) -> dict[str, list[dict]]:
    """Eligible replacement stems per category (non-silent in the first clip_seconds)."""
    source_dir = source_dir.resolve()
    stems_csv = source_dir / f"{STEMS_FILE_NAME}.csv"
    if not stems_csv.is_file():
        raise FileNotFoundError(f"Missing stems table: {stems_csv}")

    exclude_keys = exclude_keys or probe_stem_keys()
    stems = pd.read_csv(stems_csv)
    candidates: dict[str, list[dict]] = {category: [] for category in PROBE_CATEGORIES}

    for _, row in stems.iterrows():
        song_path = Path(str(row["path"]))
        track = int(row["track"])
        song_id = song_id_from_path(song_path)
        key = (song_id, track)
        if key in exclude_keys:
            continue

        category = _stem_row_category(row)
        if category is None:
            continue

        audio_path = None
        audio_format = None
        for ext in ("flac", "mp3"):
            candidate_path = stem_path(song_path, track, ext)
            if source_stem_clip_usable(
                song_path,
                track,
                ext,
                clip_seconds=clip_seconds,
                min_rms=min_rms,
            ):
                audio_path = candidate_path
                audio_format = ext
                break
        if audio_path is None:
            continue

        candidates[category].append({
            "id": f"{category}_{song_id.replace('/', '_')}_t{track}",
            "category": category,
            "song_id": song_id,
            "track": track,
            "note": str(row.get("name") or "").strip() or None,
            "rms": round(clip_rms(audio_path, clip_seconds=clip_seconds, start_seconds=0.0), 4),
            "duration_seconds": round(stem_duration_seconds(audio_path), 2),
            "audio_format": audio_format,
        })

    return candidates


def _pick_replacement(
    pool: list[dict],
    *,
    used_keys: set[tuple[str, int]],
    rng: random.Random,
) -> dict | None:
    for stem in pool:
        key = (str(stem["song_id"]), int(stem["track"]))
        if key not in used_keys:
            return stem
    return None


def replace_silent_probe_clips(
    probe_stems: list[dict],
    *,
    reference_clips_dir: Path,
    source_dir: Path,
    output_clips_dir: Path,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    min_rms: float = DEFAULT_MIN_RMS,
    hop_seconds: float = 1.0,
    seed: int = 0,
    exclude_keys: set[tuple[str, int]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Swap silent clip entries for new diverse stems; materialize clip tree when needed."""
    reference_clips_dir = reference_clips_dir.resolve()
    source_dir = source_dir.resolve()
    clip_seconds = float(clip_seconds)

    candidates = build_diverse_candidates(
        source_dir,
        clip_seconds=clip_seconds,
        min_rms=min_rms,
        exclude_keys=exclude_keys,
    )
    rng = random.Random(seed)
    for pool in candidates.values():
        rng.shuffle(pool)

    used_keys: set[tuple[str, int]] = set()
    updated: list[dict] = []
    replacements: list[dict] = []

    for stem in probe_stems:
        category = str(stem["category"])
        if probe_clip_usable(reference_clips_dir, stem, min_rms=min_rms):
            kept = dict(stem)
            updated.append(kept)
            used_keys.add((str(kept["song_id"]), int(kept["track"])))
            continue

        source_path = source_stem_path(source_dir, stem)
        clip_start = find_audible_clip_start(
            source_path,
            clip_seconds=clip_seconds,
            min_rms=min_rms,
            min_start_seconds=hop_seconds,
        )
        if clip_start is not None:
            reclipped = dict(stem)
            reclipped["clip_start_seconds"] = round(clip_start, 3)
            updated.append(reclipped)
            used_keys.add((str(reclipped["song_id"]), int(reclipped["track"])))
            replacements.append({
                "category": category,
                "from_id": stem["id"],
                "to_id": stem["id"],
                "clip_start_seconds": reclipped["clip_start_seconds"],
            })
            continue

        replacement = _pick_replacement(
            candidates.get(category, []),
            used_keys=used_keys,
            rng=rng,
        )
        if replacement is None:
            raise ValueError(
                f"No non-silent clip replacement for {category} "
                f"(silent clip: {stem['id']})"
            )

        used_keys.add((str(replacement["song_id"]), int(replacement["track"])))
        new_stem = dict(replacement)
        new_stem["category"] = category
        replacements.append({
            "category": category,
            "from_id": stem["id"],
            "to_id": new_stem["id"],
        })
        updated.append(new_stem)

    if not replacements:
        return probe_stems, replacements

    write_diverse_clip_dataset(
        source_dir,
        updated,
        output_clips_dir,
        clip_seconds=clip_seconds,
    )
    return updated, replacements
