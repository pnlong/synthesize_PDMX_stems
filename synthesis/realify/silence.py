"""Post-SA3 silence enforcement for realified stems."""

from __future__ import annotations

import math

import torch

from shared.config import (
    REALIFY_SILENCE_ACTIVE_MARGIN_MS,
    REALIFY_SILENCE_CHUNK_MS,
    REALIFY_SILENCE_ENFORCE,
    REALIFY_SILENCE_FADE_MS,
    REALIFY_SILENCE_OVERLAP_RATIO,
    REALIFY_SILENCE_THRESHOLD_DB,
    SAMPLE_RATE,
)
from synthesis.audio import ensure_stem_channels


def ms_to_samples(ms: float, sample_rate: int = SAMPLE_RATE) -> int:
    if ms <= 0:
        return 0
    return max(1, int(ms * sample_rate / 1000.0))


def linear_threshold(threshold_db: float) -> float:
    return 10 ** (threshold_db / 20.0)


def window_peak_db(waveform: torch.Tensor, start: int, end: int) -> float:
    """Peak amplitude of a waveform slice in dBFS."""
    chunk = waveform[..., start:end]
    if chunk.numel() == 0:
        return -math.inf
    peak = float(chunk.abs().max().item())
    return 20.0 * math.log10(peak + 1e-10)


def dilate_sample_mask(mask: torch.Tensor, margin_samples: int) -> torch.Tensor:
    """Expand True regions by margin_samples on each side."""
    if margin_samples <= 0:
        return mask
    mask = mask.flatten().bool()
    if not mask.any():
        return mask
    n = mask.numel()
    kernel = 2 * margin_samples + 1
    if kernel >= n:
        return torch.ones(n, dtype=torch.bool, device=mask.device)
    x = mask.float().view(1, 1, n)
    dilated = torch.nn.functional.max_pool1d(
        x,
        kernel_size=kernel,
        stride=1,
        padding=margin_samples,
    )
    return dilated.view(n).bool()


def detect_hallucination_mask(
    reference: torch.Tensor,
    realified: torch.Tensor,
    *,
    chunk_samples: int,
    overlap_samples: int,
    threshold_db: float = REALIFY_SILENCE_THRESHOLD_DB,
    margin_samples: int,
) -> torch.Tensor:
    """Return per-sample mask where reference is silent but realified is not."""
    reference = ensure_stem_channels(reference)
    realified = ensure_stem_channels(realified)
    n_samples = min(reference.shape[-1], realified.shape[-1])
    if n_samples <= 0:
        return torch.zeros(0, dtype=torch.bool, device=reference.device)

    reference = reference[..., :n_samples]
    realified = realified[..., :n_samples]

    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    if overlap_samples >= chunk_samples:
        raise ValueError("overlap_samples must be smaller than chunk_samples")

    hop = chunk_samples - overlap_samples
    ref_amp = reference.abs().amax(dim=0)
    ref_silent_samples = ref_amp < linear_threshold(threshold_db)
    ref_active = ~ref_silent_samples

    raw_hallucination = torch.zeros(n_samples, dtype=torch.bool, device=reference.device)

    start = 0
    while start < n_samples:
        end = min(start + chunk_samples, n_samples)
        real_silent = window_peak_db(realified, start, end) < threshold_db
        if not real_silent:
            raw_hallucination[start:end] |= ref_silent_samples[start:end]
        if end >= n_samples:
            break
        start += hop

    protected = dilate_sample_mask(ref_active, margin_samples)
    return raw_hallucination & ~protected


def apply_boundary_crossfade(
    output: torch.Tensor,
    original: torch.Tensor,
    silent_mask: torch.Tensor,
    fade_samples: int,
) -> torch.Tensor:
    """Linear ramps at force_silent boundaries only."""
    if fade_samples <= 0:
        return output

    mask = silent_mask.flatten().bool()
    n = mask.numel()
    if n == 0 or not mask.any():
        return output

    result = output.clone()
    original = original[..., :n]
    result = result[..., :n]

    prev = torch.cat([torch.zeros(1, dtype=torch.bool, device=mask.device), mask[:-1]])
    rising = mask & ~prev
    falling = (~mask) & prev

    for i in rising.nonzero(as_tuple=True)[0].tolist():
        for k in range(fade_samples):
            idx = i + k
            if idx >= n:
                break
            t = (k + 1) / fade_samples
            result[..., idx] = original[..., idx] * (1.0 - t)

    for i in falling.nonzero(as_tuple=True)[0].tolist():
        for k in range(fade_samples):
            idx = i + k
            if idx >= n:
                break
            t = (k + 1) / fade_samples
            result[..., idx] = original[..., idx] * t

    return result


def apply_silence_enforcement(
    reference: torch.Tensor,
    realified: torch.Tensor,
    *,
    enabled: bool = REALIFY_SILENCE_ENFORCE,
    chunk_ms: float = REALIFY_SILENCE_CHUNK_MS,
    overlap_ratio: float = REALIFY_SILENCE_OVERLAP_RATIO,
    threshold_db: float = REALIFY_SILENCE_THRESHOLD_DB,
    active_margin_ms: float = REALIFY_SILENCE_ACTIVE_MARGIN_MS,
    fade_ms: float = REALIFY_SILENCE_FADE_MS,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Detect SA3 hallucinations during reference-silent regions and force them to zero."""
    if not enabled:
        return realified

    reference = ensure_stem_channels(reference)
    realified = ensure_stem_channels(realified)
    n_samples = min(reference.shape[-1], realified.shape[-1])
    if n_samples <= 0:
        return realified

    reference = reference[..., :n_samples]
    realified = realified[..., :n_samples]

    chunk_samples = ms_to_samples(chunk_ms, sample_rate)
    overlap_samples = max(0, int(chunk_samples * overlap_ratio))
    margin_samples = ms_to_samples(active_margin_ms, sample_rate)
    fade_samples = ms_to_samples(fade_ms, sample_rate)

    force_silent = detect_hallucination_mask(
        reference,
        realified,
        chunk_samples=chunk_samples,
        overlap_samples=overlap_samples,
        threshold_db=threshold_db,
        margin_samples=margin_samples,
    )
    if not force_silent.any():
        return realified

    output = realified.clone()
    output[..., force_silent] = 0.0
    return apply_boundary_crossfade(output, realified, force_silent, fade_samples)
