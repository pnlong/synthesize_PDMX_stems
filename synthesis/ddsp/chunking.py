"""Overlap-and-stitch helpers for chunked DDSP inference (piano + MIDI-DDSP)."""

from __future__ import annotations

import numpy as np


def plan_chunk_spans(
    total: int,
    chunk: int | None = None,
    overlap: int | None = None,
    *,
    chunk_frames: int | None = None,
    overlap_frames: int | None = None,
) -> list[tuple[int, int]]:
    """Return [start, end) integer spans covering ``total`` with overlap.

    Used for piano conditioning frames and for MIDI-DDSP sample/time grids.
    Accepts either ``chunk``/``overlap`` or legacy ``chunk_frames``/``overlap_frames``.
    """
    if chunk is None:
        chunk = chunk_frames
    if overlap is None:
        overlap = overlap_frames
    if chunk is None or overlap is None:
        raise TypeError("plan_chunk_spans requires chunk and overlap sizes")
    if total <= 0:
        return []
    if total <= chunk:
        return [(0, total)]
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk:
        raise ValueError("overlap must be smaller than chunk size")

    hop = chunk - overlap
    spans: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(start + chunk, total)
        spans.append((start, end))
        if end >= total:
            break
        start += hop
    return spans


# Back-compat alias used by older tests / piano path.
plan_chunk_frame_spans = plan_chunk_spans


def frames_to_samples(n_frames: int, frame_rate: int, sample_rate: int) -> int:
    return int(n_frames / frame_rate * sample_rate)


def stitch_audio_chunks(
    chunks: list[np.ndarray],
    spans: list[tuple[int, int]],
    overlap_samples: int,
) -> np.ndarray:
    """Linear-crossfade overlapping mono chunks into one waveform."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) != len(spans):
        raise ValueError("chunks and spans must have the same length")
    if len(chunks) == 1:
        length = spans[0][1] - spans[0][0]
        audio = np.asarray(chunks[0], dtype=np.float32).reshape(-1)
        if audio.shape[0] >= length:
            return audio[:length]
        return np.pad(audio, (0, length - audio.shape[0]))

    total_samples = spans[-1][1]
    output = np.zeros(total_samples, dtype=np.float64)
    weights = np.zeros(total_samples, dtype=np.float64)

    for i, (chunk, (start, end)) in enumerate(zip(chunks, spans)):
        length = end - start
        audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if audio.shape[0] >= length:
            audio = audio[:length]
        else:
            audio = np.pad(audio, (0, length - audio.shape[0]))

        weight = np.ones(length, dtype=np.float64)
        if i > 0 and overlap_samples > 0:
            fade = min(overlap_samples, length)
            weight[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float64)
        if i < len(chunks) - 1 and overlap_samples > 0:
            fade = min(overlap_samples, length)
            fade_out = np.linspace(1.0, 0.0, fade, dtype=np.float64)
            weight[-fade:] = np.minimum(weight[-fade:], fade_out)

        output[start:end] += audio * weight
        weights[start:end] += weight

    weights = np.maximum(weights, 1e-8)
    return (output / weights).astype(np.float32)
