"""Build a JSON-serializable catalog from ablation CSV metadata and audio files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from shared.config import (
    ABLATION_SAMPLE_SEED,
    DATA_DIR_NAME,
    DEFAULT_AUDIO_FORMAT,
    OUTPUT_DIR,
    PROTOTYPE_AUDIO_FORMAT,
    STEMS_FILE_NAME,
)
from shared.repo_symlinks import REPO_ABLATIONS_SYMLINK
from synthesis.audio import mixture_filename, stem_filename
from synthesis.paths import ablations_root
from synthesis.realify.captions.generate import generate_captions_from_tables
from synthesis.realify.realify import resolve_output_song_dir

CONDITION_ORDER = ("basic", "basic_realify", "slakh", "slakh_realify")

CONDITION_LABELS: dict[str, str] = {
    "basic": "A1",
    "basic_realify": "A2",
    "slakh": "B1",
    "slakh_realify": "B2",
}

REALIFY_CONDITIONS = frozenset({"basic_realify", "slakh_realify"})


@dataclass(frozen=True)
class ConditionInfo:
    id: str
    label: str
    name: str
    available: bool


def default_ablations_dir() -> Path:
    if REPO_ABLATIONS_SYMLINK.is_dir():
        return REPO_ABLATIONS_SYMLINK.resolve()
    return Path(ablations_root(OUTPUT_DIR))


def song_id_from_path(song_path: str | Path) -> str:
    """Relative path under ``data/`` (e.g. ``7/19/QmPfj...``)."""
    parts = Path(song_path).parts
    if DATA_DIR_NAME not in parts:
        raise ValueError(f"Song path missing {DATA_DIR_NAME}/ segment: {song_path}")
    idx = parts.index(DATA_DIR_NAME)
    return str(Path(*parts[idx + 1:]))


def detect_audio_format(song_dir: Path) -> str | None:
    if (song_dir / mixture_filename(PROTOTYPE_AUDIO_FORMAT)).exists():
        return PROTOTYPE_AUDIO_FORMAT
    if (song_dir / mixture_filename(DEFAULT_AUDIO_FORMAT)).exists():
        return DEFAULT_AUDIO_FORMAT
    return None


def _condition_dir(ablations_dir: Path, condition: str) -> Path:
    return ablations_dir / condition


def _song_dir_for_condition(
    ablations_dir: Path,
    reference_condition: str,
    song_path: str | Path,
    target_condition: str,
) -> Path:
    source_dir = _condition_dir(ablations_dir, reference_condition)
    target_dir = _condition_dir(ablations_dir, target_condition)
    return resolve_output_song_dir(Path(song_path), source_dir, target_dir)


def _audio_cell(
    ablations_dir: Path,
    reference_condition: str,
    song_path: str | Path,
    target_condition: str,
    filename: str,
) -> dict:
    song_dir = _song_dir_for_condition(
        ablations_dir, reference_condition, song_path, target_condition
    )
    audio_path = song_dir / filename
    available = audio_path.is_file()
    song_id = song_id_from_path(song_path)
    return {
        "available": available,
        "url": f"/audio/{target_condition}/{song_id}/{filename}" if available else None,
    }


class AblationCatalog:
    def __init__(self, ablations_dir: Path, caption_seed: int = ABLATION_SAMPLE_SEED):
        self.ablations_dir = ablations_dir.resolve()
        self.caption_seed = caption_seed
        self.reference_condition = self._pick_reference_condition()
        self._songs_df = self._load_songs()
        self._stems_df = self._load_stems()
        self._captions_df = self._build_captions()

    def _pick_reference_condition(self) -> str:
        for condition in CONDITION_ORDER:
            if (self.ablations_dir / condition / f"{DATA_DIR_NAME}.csv").is_file():
                return condition
        raise FileNotFoundError(
            f"No ablation condition with {DATA_DIR_NAME}.csv under {self.ablations_dir}"
        )

    def _load_songs(self) -> pd.DataFrame:
        path = self.ablations_dir / self.reference_condition / f"{DATA_DIR_NAME}.csv"
        return pd.read_csv(path)

    def _load_stems(self) -> pd.DataFrame:
        path = self.ablations_dir / self.reference_condition / f"{STEMS_FILE_NAME}.csv"
        if not path.is_file():
            return pd.DataFrame(columns=["path", "track", "program", "is_drum", "name", "has_lyrics"])
        return pd.read_csv(path)

    def _build_captions(self) -> pd.DataFrame:
        if self._stems_df.empty or self._songs_df.empty:
            return pd.DataFrame(columns=["path", "track", "prompt"])
        return generate_captions_from_tables(
            self._songs_df,
            self._stems_df,
            seed=self.caption_seed,
        )

    def conditions(self) -> list[dict]:
        result = []
        for condition in CONDITION_ORDER:
            cond_dir = self.ablations_dir / condition
            result.append(
                ConditionInfo(
                    id=condition,
                    label=CONDITION_LABELS[condition],
                    name=condition,
                    available=(cond_dir / f"{DATA_DIR_NAME}.csv").is_file(),
                ).__dict__
            )
        return result

    def list_songs(self) -> list[dict]:
        songs = []
        for _, row in self._songs_df.iterrows():
            song_path = row["path"]
            song_id = song_id_from_path(song_path)
            duration = row.get("song_length.seconds")
            songs.append({
                "id": song_id,
                "title": _na_to_none(row.get("title")),
                "song_name": _na_to_none(row.get("song_name")),
                "artist_name": _na_to_none(row.get("artist_name")),
                "n_tracks": int(row.get("n_tracks", 0)),
                "genres": _na_to_none(row.get("genres")),
                "duration_seconds": _safe_float(duration),
            })
        return songs

    def get_song(self, song_id: str) -> dict | None:
        for _, row in self._songs_df.iterrows():
            if song_id_from_path(row["path"]) == song_id:
                return self._build_song_detail(row)
        return None

    def _build_song_detail(self, row: pd.Series) -> dict:
        song_path = row["path"]
        song_id = song_id_from_path(song_path)
        ref_song_dir = _song_dir_for_condition(
            self.ablations_dir,
            self.reference_condition,
            song_path,
            self.reference_condition,
        )
        audio_format = detect_audio_format(ref_song_dir) or DEFAULT_AUDIO_FORMAT

        stems_rows = self._stems_df[self._stems_df["path"] == song_path].sort_values("track")
        stems = []
        for _, stem_row in stems_rows.iterrows():
            track = int(stem_row["track"])
            stem_filename_str = stem_filename(track, audio_format)
            conditions = {
                condition: _audio_cell(
                    self.ablations_dir,
                    self.reference_condition,
                    song_path,
                    condition,
                    stem_filename_str,
                )
                for condition in CONDITION_ORDER
            }
            caption = None
            if not self._captions_df.empty:
                cap_rows = self._captions_df[
                    (self._captions_df["path"] == song_path)
                    & (self._captions_df["track"] == track)
                ]
                if not cap_rows.empty:
                    caption = _na_to_none(cap_rows.iloc[0].get("prompt"))

            stems.append({
                "track": track,
                "name": _na_to_none(stem_row.get("name")) or f"Track {track}",
                "program": int(stem_row["program"]) if pd.notna(stem_row.get("program")) else None,
                "is_drum": bool(stem_row.get("is_drum", False)),
                "caption": caption,
                "conditions": conditions,
            })

        mixture_filename_str = mixture_filename(audio_format)
        mixture = {
            condition: _audio_cell(
                self.ablations_dir,
                self.reference_condition,
                song_path,
                condition,
                mixture_filename_str,
            )
            for condition in CONDITION_ORDER
        }

        return {
            "id": song_id,
            "path": str(song_path),
            "song_dirs": {
                condition: str(
                    _song_dir_for_condition(
                        self.ablations_dir,
                        self.reference_condition,
                        song_path,
                        condition,
                    )
                )
                for condition in CONDITION_ORDER
            },
            "title": _na_to_none(row.get("title")),
            "song_name": _na_to_none(row.get("song_name")),
            "subtitle": _na_to_none(row.get("subtitle")),
            "artist_name": _na_to_none(row.get("artist_name")),
            "composer_name": _na_to_none(row.get("composer_name")),
            "genres": _na_to_none(row.get("genres")),
            "n_tracks": int(row.get("n_tracks", 0)),
            "duration_seconds": _safe_float(row.get("song_length.seconds")),
            "audio_format": audio_format,
            "mixture": mixture,
            "stems": stems,
        }

    def resolve_audio_path(self, condition: str, song_id: str, filename: str) -> Path | None:
        if condition not in CONDITION_ORDER:
            return None
        if ".." in Path(song_id).parts or ".." in Path(filename).parts:
            return None
        if "/" in filename or "\\" in filename:
            return None

        for _, row in self._songs_df.iterrows():
            if song_id_from_path(row["path"]) == song_id:
                song_dir = _song_dir_for_condition(
                    self.ablations_dir,
                    self.reference_condition,
                    row["path"],
                    condition,
                )
                audio_path = (song_dir / filename).resolve()
                if not str(audio_path).startswith(str(self.ablations_dir)):
                    return None
                return audio_path if audio_path.is_file() else None
        return None


def _na_to_none(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text == "NA":
        return None
    return text


def _safe_float(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
