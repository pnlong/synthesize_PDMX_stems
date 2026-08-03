"""Feature flag for dense corrected MIDI (empty tracks dropped)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from shared.config import MID_CORRECTED_DIR_NAME, OUTPUT_DIR
from synthesis.paths import mid_corrected_dir


def _env_flag(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def dense_midi_enabled(args: argparse.Namespace | None = None) -> bool:
    """True when dense corrected MIDI should be used.

    Precedence: CLI ``--dense-midi`` / ``--no-dense-midi`` > ``SPDMX_DENSE_MIDI`` env >
    default off.
    """
    if args is not None:
        explicit = getattr(args, "dense_midi", None)
        if explicit is True:
            return True
        if explicit is False:
            return False
    env = _env_flag("SPDMX_DENSE_MIDI")
    if env is not None:
        return env
    return False


def default_corrected_midi_dir(output_dir: str = OUTPUT_DIR) -> str:
    return mid_corrected_dir(output_dir)


def resolve_synthesis_midi(
    pdmx_mid: str | Path,
    *,
    args: argparse.Namespace,
    pdmx_root: str | Path,
) -> tuple[Path, dict[int, dict] | None]:
    """Return (midi_path, track_map_or_none).

    When dense MIDI is enabled, ``track_map`` maps dense track →
    ``{original_track, program, is_drum, name}``.
    """
    from analysis.corrected_midi import load_track_map, resolve_corrected_midi_path

    pdmx_mid = Path(pdmx_mid)
    if not dense_midi_enabled(args):
        return pdmx_mid, None

    corrected_root = Path(
        getattr(args, "corrected_midi_dir", None) or default_corrected_midi_dir(args.output_dir)
    )
    corrected = resolve_corrected_midi_path(
        pdmx_mid,
        pdmx_root=pdmx_root,
        corrected_midi_dir=corrected_root,
    )
    if not corrected.is_file():
        raise FileNotFoundError(
            f"Dense MIDI enabled but corrected MIDI missing: {corrected}\n"
            "Generate corrected midis first:\n"
            "  uv run python -m analysis.prepare_synthesis --subset all_valid "
            "-j 8\n"
            "Or disable with --no-dense-midi / unset SPDMX_DENSE_MIDI."
        )
    return corrected, load_track_map(corrected)


def stem_original_track(row: dict | object, track: int | None = None) -> int:
    """Read ``original_track`` from a stems row; fall back to ``track``."""
    if track is None:
        track = int(row["track"] if isinstance(row, dict) else row["track"])
    if isinstance(row, dict):
        value = row.get("original_track")
    else:
        value = row["original_track"] if "original_track" in getattr(row, "index", []) else None
        if value is None and hasattr(row, "get"):
            value = row.get("original_track")
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return int(track)
    try:
        import pandas as pd

        if pd.isna(value):
            return int(track)
    except Exception:
        pass
    return int(value)
