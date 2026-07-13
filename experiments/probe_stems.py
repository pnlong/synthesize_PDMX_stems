"""Probe stem set helpers shared by patch and preset sweeps."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from os.path import dirname
from pathlib import Path

import mido
import pandas as pd
import yaml

from experiments.paths import DEFAULT_PROBE_STEMS
from shared.config import PDMX_FILEPATH
from synthesis.patches import _gm_class

TARGET_SAMPLES_PER_CATEGORY = 3

PROBE_CATEGORIES = (
    "piano",
    "drums",
    "strings",
    "wind",
    "voice",
    "mallet",
    "organ",
    "polyphonic",
)

# Probe listening category -> GM classes that Fluidsynth will actually render.
PROBE_CATEGORY_GM_CLASSES: dict[str, frozenset[str]] = {
    "piano": frozenset({"piano"}),
    "drums": frozenset({"drums"}),
    "strings": frozenset({"strings"}),
    "wind": frozenset({"pipe", "reed", "brass"}),
    "voice": frozenset({"ensemble"}),
    "mallet": frozenset({"chromatic_percussion"}),
    "organ": frozenset({"organ"}),
}


@dataclass(frozen=True)
class TrackMidiMeta:
    program: int | None
    is_drum: bool
    track_name: str | None
    has_program_change: bool

    @property
    def effective_program(self) -> int:
        if self.is_drum:
            return 0
        return 0 if self.program is None else self.program

    @property
    def gm_class(self) -> str:
        return _gm_class(self.effective_program, self.is_drum)


def load_probe_stems(path: Path = DEFAULT_PROBE_STEMS) -> list[dict]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return list(cfg.get("stems") or [])


def category_counts(stems: list[dict]) -> Counter:
    return Counter(stem.get("category") for stem in stems)


@lru_cache(maxsize=4)
def _pdmx_mid_paths(pdmx_filepath: str) -> dict[str, str]:
    df = pd.read_csv(pdmx_filepath, usecols=["mid"])
    pdmx_root = dirname(pdmx_filepath)
    mapping: dict[str, str] = {}
    for mid in df["mid"]:
        if not isinstance(mid, str) or not mid.startswith("./mid/") or not mid.endswith(".mid"):
            continue
        song_id = mid[len("./mid/") : -len(".mid")]
        mapping[song_id] = f"{pdmx_root}{mid[1:]}"
    return mapping


def resolve_mid_path(song_id: str, pdmx_filepath: str = PDMX_FILEPATH) -> Path:
    mid_paths = _pdmx_mid_paths(pdmx_filepath)
    mid_rel = mid_paths.get(song_id)
    if mid_rel is None:
        raise FileNotFoundError(
            f"song_id {song_id} not found in PDMX dataset {pdmx_filepath}"
        )
    mid_path = Path(mid_rel)
    if not mid_path.is_file():
        raise FileNotFoundError(f"MIDI not found for song_id {song_id}: {mid_path}")
    return mid_path


def read_track_midi_meta(mid_path: Path, track: int) -> TrackMidiMeta:
    midi = mido.MidiFile(filename=str(mid_path), charset="utf8")
    if track >= len(midi.tracks):
        raise ValueError(f"Track {track} out of range for {mid_path} ({len(midi.tracks)} tracks)")

    program: int | None = None
    is_drum = False
    track_name: str | None = None
    has_program_change = False

    for message in midi.tracks[track]:
        if message.type == "program_change":
            program = message.program
            has_program_change = True
        elif message.type == "track_name":
            track_name = message.name
        elif hasattr(message, "channel") and message.channel == 9:
            is_drum = True

    return TrackMidiMeta(
        program=program,
        is_drum=is_drum,
        track_name=track_name,
        has_program_change=has_program_change,
    )


def stem_matches_category(meta: TrackMidiMeta, category: str) -> bool:
    if category == "polyphonic":
        return meta.has_program_change and not meta.is_drum

    allowed = PROBE_CATEGORY_GM_CLASSES.get(category)
    if allowed is None:
        return False
    return meta.gm_class in allowed


def validate_probe_stem_midi_programs(
    stems: list[dict],
    *,
    pdmx_filepath: str = PDMX_FILEPATH,
) -> None:
    """Ensure each stem's MIDI program matches its listening category."""
    problems: list[str] = []
    for stem in stems:
        stem_id = stem.get("id")
        category = stem.get("category")
        song_id = stem.get("song_id")
        track = stem.get("track")
        if not stem_id or not category or not song_id or track is None:
            problems.append(f"{stem_id or '?'}: missing id/category/song_id/track")
            continue

        try:
            mid_path = resolve_mid_path(str(song_id), pdmx_filepath)
            meta = read_track_midi_meta(mid_path, int(track))
        except (FileNotFoundError, ValueError) as exc:
            problems.append(f"{stem_id}: {exc}")
            continue

        if stem_matches_category(meta, str(category)):
            continue

        name = (meta.track_name or "").strip() or "?"
        if category == "polyphonic":
            detail = "polyphonic stems need an explicit program_change"
        else:
            allowed = ", ".join(sorted(PROBE_CATEGORY_GM_CLASSES.get(str(category), ())))
            detail = f"expected GM class in [{allowed}], got {meta.gm_class}"
        problems.append(
            f"{stem_id} ({category}): program={meta.effective_program}, "
            f"track_name={name!r}, {detail}"
        )

    if problems:
        raise ValueError(
            "probe stems must use MIDI programs that match their category; "
            + "; ".join(problems)
        )


def validate_probe_stems(
    stems: list[dict],
    *,
    target_per_category: int = TARGET_SAMPLES_PER_CATEGORY,
    required_categories: tuple[str, ...] = PROBE_CATEGORIES,
    validate_midi_programs: bool = True,
    pdmx_filepath: str = PDMX_FILEPATH,
) -> None:
    """Raise ValueError when category coverage or MIDI programs do not meet targets."""
    if not stems:
        raise ValueError("probe stem set is empty")

    counts = category_counts(stems)
    missing = [cat for cat in required_categories if cat not in counts]
    if missing:
        raise ValueError(f"probe stems missing categories: {', '.join(missing)}")

    problems = []
    for category in required_categories:
        count = counts[category]
        if count < target_per_category:
            problems.append(f"{category}: {count} < {target_per_category}")
        elif count > target_per_category:
            problems.append(f"{category}: {count} > {target_per_category}")

    if problems:
        raise ValueError(
            "probe stems must have exactly "
            f"{target_per_category} samples per category; "
            + "; ".join(problems)
        )

    ids = [stem.get("id") for stem in stems]
    if len(ids) != len(set(ids)):
        raise ValueError("probe stem ids must be unique")

    keys = [(stem.get("song_id"), stem.get("track")) for stem in stems]
    if len(keys) != len(set(keys)):
        raise ValueError("probe stems must use unique (song_id, track) pairs")

    if validate_midi_programs and Path(pdmx_filepath).is_file():
        validate_probe_stem_midi_programs(stems, pdmx_filepath=pdmx_filepath)
