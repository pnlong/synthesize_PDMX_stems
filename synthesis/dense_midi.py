"""Dense corrected MIDI resolution for synthesis (empty tracks dropped)."""

from __future__ import annotations

import argparse
from pathlib import Path

from shared.config import OUTPUT_DIR
from synthesis.paths import mid_corrected_dir


def default_corrected_midi_dir(output_dir: str = OUTPUT_DIR) -> str:
    return mid_corrected_dir(output_dir)


def resolve_synthesis_midi(
    pdmx_mid: str | Path,
    *,
    args: argparse.Namespace,
    pdmx_root: str | Path,
) -> tuple[Path, dict[int, dict]]:
    """Return (corrected_midi_path, track_map).

    ``track_map`` maps dense track → ``{original_track, program, is_drum, name}``.
    """
    from analysis.corrected_midi import load_track_map, resolve_corrected_midi_path

    pdmx_mid = Path(pdmx_mid)
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
            f"Corrected MIDI missing: {corrected}\n"
            "Generate corrected midis first:\n"
            "  uv run python -m analysis.prepare_synthesis --subset all_valid -j 8"
        )
    return corrected, load_track_map(
        corrected,
        corrected_midi_dir=corrected_root,
        track_maps=getattr(args, "track_maps", None),
    )


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
