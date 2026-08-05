"""Detect DDSP donor-copy equivalences via ``route_stem`` (not audio hashing)."""

from __future__ import annotations

from pathlib import Path

import mido
import pandas as pd

from shared.config import OUTPUT_DIR, PDMX_FILEPATH, STEMS_FILE_NAME
from synthesis.ddsp.routing import BACKEND_SOUNDFONT, route_stem
from synthesis.listening.catalog import song_id_from_path
from synthesis.mix import resolve_song_midi

# Duplicate condition → donor whose audio it copies when routed to soundfont.
DONOR_EQUIVALENCE_PAIRS: dict[str, str] = {
    "ddsp_basic": "basic",
    "ddsp_slakh": "slakh",
    "ddsp_basic_realify": "basic_realify",
    "ddsp_slakh_realify": "slakh_realify",
}


def donor_equivalences_for_backend(backend: str) -> dict[str, str]:
    """All four DDSP↔donor pairs when the stem is a soundfont fallback."""
    if backend == BACKEND_SOUNDFONT:
        return dict(DONOR_EQUIVALENCE_PAIRS)
    return {}


def detect_equivalences_for_stem(
    *,
    program: int,
    is_drum: bool,
    track_name: str | None = None,
    midi_path: str | Path | None = None,
    track=None,
    ticks_per_beat: int = 480,
    check_monophony: bool = True,
) -> dict[str, str]:
    """Return ``{duplicate: donor}`` when ``route_stem`` chooses soundfont."""
    route = route_stem(
        program=int(program),
        is_drum=bool(is_drum),
        track_name=track_name,
        midi_path=midi_path,
        track=track,
        ticks_per_beat=ticks_per_beat,
        check_monophony=check_monophony,
    )
    return donor_equivalences_for_backend(route.backend)


def load_stem_row(
    ablations_dir: Path,
    song_id: str,
    track: int,
) -> pd.Series:
    """Look up ``program`` / ``is_drum`` / ``name`` from the basic ``stems.csv``."""
    stems_csv = Path(ablations_dir) / "basic" / f"{STEMS_FILE_NAME}.csv"
    if not stems_csv.is_file():
        raise FileNotFoundError(f"Missing stems table: {stems_csv}")
    stems_df = pd.read_csv(stems_csv)
    track_i = int(track)
    matches = []
    for _, row in stems_df.iterrows():
        if int(row["track"]) != track_i:
            continue
        if song_id_from_path(str(row["path"])) != song_id:
            continue
        matches.append(row)
    if not matches:
        raise KeyError(f"No stems.csv row for song_id={song_id!r} track={track_i}")
    return matches[0]


def load_stem_midi_track(
    song_dir: Path,
    track: int,
    *,
    pdmx_root: Path | None = None,
    spdmx_output_dir: str | Path = OUTPUT_DIR,
) -> tuple[object, int, Path]:
    """Return ``(midi_track, ticks_per_beat, midi_path)`` for a stem index."""
    root = Path(pdmx_root) if pdmx_root is not None else Path(PDMX_FILEPATH).parent
    midi_path = resolve_song_midi(
        Path(song_dir),
        pdmx_root=root,
        output_dir=str(spdmx_output_dir),
    )
    midi = mido.MidiFile(filename=str(midi_path), charset="utf8")
    track_i = int(track)
    if track_i < 0 or track_i >= len(midi.tracks):
        raise IndexError(
            f"Track {track_i} out of range for {midi_path} "
            f"({len(midi.tracks)} tracks)"
        )
    return midi.tracks[track_i], int(midi.ticks_per_beat), midi_path


def detect_equivalences_for_trial(
    trial: dict,
    ablations_dir: Path,
    *,
    pdmx_root: Path | None = None,
    spdmx_output_dir: str | Path = OUTPUT_DIR,
) -> dict[str, str]:
    """Route-based equivalences for a stem trial; mixtures get none."""
    if (trial.get("type") or "stem") == "mixture":
        return {}
    track = trial.get("track")
    song_id = trial.get("song_id")
    if track is None or not song_id:
        return {}

    row = load_stem_row(ablations_dir, str(song_id), int(track))
    program = int(row.get("program", 0) or 0)
    is_drum = bool(row.get("is_drum", False))
    track_name = row.get("name")
    if track_name is not None and (not str(track_name).strip() or str(track_name) == "nan"):
        track_name = None
    else:
        track_name = str(track_name) if track_name is not None else None

    song_dir = Path(ablations_dir) / "basic" / "data" / str(song_id)
    try:
        midi_track, ticks_per_beat, _midi_path = load_stem_midi_track(
            song_dir,
            int(track),
            pdmx_root=pdmx_root,
            spdmx_output_dir=spdmx_output_dir,
        )
    except (FileNotFoundError, IndexError, ValueError):
        # Metadata-only routing (no monophony / empty-track check via MIDI).
        return detect_equivalences_for_stem(
            program=program,
            is_drum=is_drum,
            track_name=track_name,
            check_monophony=False,
        )

    return detect_equivalences_for_stem(
        program=program,
        is_drum=is_drum,
        track_name=track_name,
        track=midi_track,
        ticks_per_beat=ticks_per_beat,
        check_monophony=True,
    )


def unique_condition_ids(
    all_conditions: tuple[str, ...] | list[str],
    equivalences: dict[str, str] | None,
) -> list[str]:
    """Conditions to present as blind samples (omit donor-copy duplicates)."""
    skip = set(equivalences or {})
    return [c for c in all_conditions if c not in skip]


def trial_equivalences(trial: dict) -> dict[str, str]:
    """Normalize a trial's ``equivalences`` map (duplicate → donor)."""
    raw = trial.get("equivalences") or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for duplicate, donor in raw.items():
        dup = str(duplicate)
        don = str(donor)
        if dup in DONOR_EQUIVALENCE_PAIRS and DONOR_EQUIVALENCE_PAIRS[dup] == don:
            out[dup] = don
    return out


def equivalences_by_trial_id(manifest: dict) -> dict[str, dict[str, str]]:
    """``{trial_id: {duplicate: donor}}`` from a trial manifest."""
    out: dict[str, dict[str, str]] = {}
    for trial in manifest.get("trials") or []:
        trial_id = trial.get("id")
        if not trial_id:
            continue
        equiv = trial_equivalences(trial)
        if equiv:
            out[str(trial_id)] = equiv
    return out


def load_manifest_equivalences(manifest_path: Path | None) -> dict[str, dict[str, str]]:
    if manifest_path is None or not Path(manifest_path).is_file():
        return {}
    import yaml

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f) or {}
    return equivalences_by_trial_id(manifest)


def expand_equivalence_scores(
    df: "pd.DataFrame",
    equivalences_by_trial: dict[str, dict[str, str]],
    *,
    score_columns: tuple[str, ...] = ("score",),
    scale_key: str | None = "scale",
) -> "pd.DataFrame":
    """Synthesize ratings for omitted donor-copy conditions from rated donors.

    Supports:
    - long format: ``score`` + ``scale`` columns
    - wide format (custom JSON): ``content`` + ``realism`` columns (``scale_key=None``)
    """
    import pandas as pd

    if df.empty or not equivalences_by_trial:
        if "auto_assigned" not in df.columns:
            df = df.copy()
            df["auto_assigned"] = False
            df["source_condition"] = None
        return df

    df = df.copy()
    if "auto_assigned" not in df.columns:
        df["auto_assigned"] = False
    if "source_condition" not in df.columns:
        df["source_condition"] = None

    def _row_key(row) -> tuple:
        parts = [row.listener_id, row.trial_id, row.condition_id]
        if scale_key is not None:
            parts.insert(2, getattr(row, scale_key))
        return tuple(parts)

    existing = {_row_key(row) for row in df.itertuples(index=False)}

    extra: list[dict] = []
    for row in df.itertuples(index=False):
        if bool(getattr(row, "auto_assigned", False)):
            continue
        equiv = equivalences_by_trial.get(str(row.trial_id)) or {}
        for duplicate, donor in equiv.items():
            if row.condition_id != donor:
                continue
            key_parts = [row.listener_id, row.trial_id, duplicate]
            if scale_key is not None:
                key_parts.insert(2, getattr(row, scale_key))
            key = tuple(key_parts)
            if key in existing:
                continue
            existing.add(key)
            new_row = {
                "listener_id": row.listener_id,
                "trial_id": row.trial_id,
                "category": row.category,
                "trial_type": row.trial_type,
                "condition_id": duplicate,
                "auto_assigned": True,
                "source_condition": donor,
            }
            if scale_key is not None:
                new_row[scale_key] = getattr(row, scale_key)
                if hasattr(row, "page_id"):
                    new_row["page_id"] = row.page_id
            if hasattr(row, "is_reference"):
                new_row["is_reference"] = False
            if hasattr(row, "condition_label"):
                from synthesis.listening.catalog import CONDITION_LABELS

                new_row["condition_label"] = CONDITION_LABELS.get(duplicate, duplicate)
            for col in score_columns:
                new_row[col] = getattr(row, col)
            extra.append(new_row)

    if not extra:
        return df
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)


__all__ = [
    "DONOR_EQUIVALENCE_PAIRS",
    "detect_equivalences_for_stem",
    "detect_equivalences_for_trial",
    "donor_equivalences_for_backend",
    "equivalences_by_trial_id",
    "expand_equivalence_scores",
    "load_manifest_equivalences",
    "load_stem_midi_track",
    "load_stem_row",
    "trial_equivalences",
    "unique_condition_ids",
]
