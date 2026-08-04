"""MIDI velocity dynamics for cross-stem level restoration.

Independent per-stem LUFS normalization removes relative dynamics (piano vs forte
across tracks). After LUFS, multiply each stem by

    velocity_scale[track] = max_velocity(track) / max_velocity(song)

using note-on velocities (> 0) from the song MIDI.
"""

from __future__ import annotations

from pathlib import Path

import mido
import torch


def max_note_on_velocity(track) -> int:
    """Maximum note-on velocity (> 0) on a MIDI track; 0 if none."""
    peak = 0
    for message in track:
        if message.type == "note_on" and message.velocity > 0:
            peak = max(peak, int(message.velocity))
    return peak


def velocity_scales_for_midi(midi_path: str | Path) -> dict[int, float]:
    """Per-track velocity scales: track_max / song_max.

    Empty tracks (no note-ons) get scale ``0.0``. If the song has no note-ons,
    every track gets ``1.0`` (avoid divide-by-zero; leave dynamics untouched).
    """
    midi = mido.MidiFile(filename=str(midi_path), charset="utf8")
    track_max: dict[int, int] = {
        i: max_note_on_velocity(track) for i, track in enumerate(midi.tracks)
    }
    song_max = max(track_max.values()) if track_max else 0
    if song_max <= 0:
        return {i: 1.0 for i in track_max}
    return {
        i: (float(v) / float(song_max)) if v > 0 else 0.0
        for i, v in track_max.items()
    }


def velocity_scales_from_track_maxima(track_maxima: dict[int, int]) -> dict[int, float]:
    """Build scales from precomputed per-track max velocities."""
    if not track_maxima:
        return {}
    song_max = max(track_maxima.values())
    if song_max <= 0:
        return {i: 1.0 for i in track_maxima}
    return {
        i: (float(v) / float(song_max)) if v > 0 else 0.0
        for i, v in track_maxima.items()
    }


def apply_velocity_scales(
    waveforms: list[torch.Tensor],
    track_indices: list[int],
    scales: dict[int, float] | None,
) -> list[torch.Tensor]:
    """Multiply each waveform by ``scales[track]`` (missing → 1.0)."""
    if not scales:
        return waveforms
    if len(waveforms) != len(track_indices):
        raise ValueError("waveforms and track_indices must have the same length")
    out = []
    for waveform, track in zip(waveforms, track_indices):
        scale = float(scales.get(track, 1.0))
        out.append(waveform if scale == 1.0 else waveform * scale)
    return out


def pdmx_mid_from_song_dir(song_dir: Path, pdmx_root: Path) -> Path:
    """Map an ablation song dir ``.../data/a/b/Qm…`` to ``{pdmx_root}/mid/a/b/Qm….mid``."""
    parts = song_dir.resolve().parts
    try:
        data_idx = parts.index("data")
    except ValueError as exc:
        raise FileNotFoundError(
            f"Cannot resolve PDMX MIDI: song path has no /data/ segment: {song_dir}"
        ) from exc
    rel = Path(*parts[data_idx + 1 :])
    mid_path = Path(pdmx_root) / "mid" / rel.with_suffix(".mid")
    return mid_path
