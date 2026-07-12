"""Overlap-and-stitch chunking for long realify stems."""

from __future__ import annotations

import torch

from shared.config import (
    REALIFY_CHUNK_OVERLAP_SEC,
    REALIFY_DURATION_PADDING_SEC,
    SAMPLE_RATE,
)


def max_realify_chunk_seconds(model) -> float:
    """Longest stem segment SA3 can realify in one pass for this model."""
    sample_size = model.model_config["sample_size"]
    sample_rate = getattr(getattr(model, "model", None), "sample_rate", SAMPLE_RATE)
    return max(1.0, (sample_size / sample_rate) - REALIFY_DURATION_PADDING_SEC)


def max_realify_chunk_samples(model) -> int:
    return int(max_realify_chunk_seconds(model) * SAMPLE_RATE)


def realify_overlap_samples() -> int:
    return int(REALIFY_CHUNK_OVERLAP_SEC * SAMPLE_RATE)


def needs_chunking(total_samples: int, model) -> bool:
    return total_samples > max_realify_chunk_samples(model)


def plan_chunk_spans(
    total_samples: int,
    chunk_samples: int,
    overlap_samples: int,
) -> list[tuple[int, int]]:
    """Return [start, end) sample spans covering the full stem with overlap."""
    if total_samples <= 0:
        return []
    if total_samples <= chunk_samples:
        return [(0, total_samples)]
    if overlap_samples >= chunk_samples:
        raise ValueError("overlap must be smaller than chunk size")

    hop = chunk_samples - overlap_samples
    spans: list[tuple[int, int]] = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        spans.append((start, end))
        if end >= total_samples:
            break
        start += hop
    return spans


def fit_chunk_length(chunk: torch.Tensor, length: int) -> torch.Tensor:
    current = chunk.shape[-1]
    if current == length:
        return chunk
    if current > length:
        return chunk[..., :length]
    return torch.nn.functional.pad(chunk, (0, length - current))


def stitch_chunk_outputs(
    chunks: list[torch.Tensor],
    spans: list[tuple[int, int]],
    overlap_samples: int,
) -> torch.Tensor:
    """Crossfade overlapping realified chunks back into one stem."""
    if not chunks:
        raise ValueError("need at least one chunk")
    if len(chunks) != len(spans):
        raise ValueError("chunks and spans must have the same length")
    if len(chunks) == 1:
        return fit_chunk_length(chunks[0].detach().cpu(), spans[0][1] - spans[0][0])

    total_samples = spans[-1][1]
    channels = chunks[0].shape[0]
    output = torch.zeros(channels, total_samples)
    weights = torch.zeros(1, total_samples)

    for i, (chunk, (start, end)) in enumerate(zip(chunks, spans)):
        length = end - start
        chunk = fit_chunk_length(chunk.detach().cpu(), length)
        weight = torch.ones(1, length)

        if i > 0:
            fade = min(overlap_samples, length)
            weight[..., :fade] = torch.linspace(0.0, 1.0, fade)

        if i < len(chunks) - 1:
            fade = min(overlap_samples, length)
            fade_out = torch.linspace(1.0, 0.0, fade)
            weight[..., -fade:] = torch.minimum(weight[..., -fade:], fade_out)

        output[..., start:end] += chunk * weight
        weights[..., start:end] += weight

    return output / weights.clamp(min=1e-8)
