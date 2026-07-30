"""Post-SA3 content fidelity scoring via onset alignment."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import torch

from shared.config import (
    REALIFY_CONTENT_FIDELITY_ONSET_TOLERANCE_MS,
    REALIFY_CONTENT_FIDELITY_THRESHOLD,
    REALIFY_SILENCE_THRESHOLD_DB,
    SAMPLE_RATE,
)
from synthesis.realify.silence import dilate_sample_mask, linear_threshold, ms_to_samples


@dataclass(frozen=True)
class ContentFidelityResult:
    score: float
    matched_onsets: int
    extra_onsets: int
    missing_onsets: int
    n_reference_onsets: int
    n_realified_onsets: int
    passed: bool


def waveform_to_mono_numpy(
    waveform: torch.Tensor | np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Convert a stem waveform to mono float32 numpy."""
    del sample_rate
    if isinstance(waveform, torch.Tensor):
        array = waveform.detach().cpu().numpy()
    else:
        array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == 1:
        return array.astype(np.float32, copy=False)
    if array.ndim == 2:
        if array.shape[0] == 1:
            return array[0].astype(np.float32, copy=False)
        return array.mean(axis=0).astype(np.float32, copy=False)
    raise ValueError(f"Expected 1D or 2D waveform, got shape {array.shape}")


def reference_active_mask(
    reference: np.ndarray,
    *,
    threshold_db: float = REALIFY_SILENCE_THRESHOLD_DB,
    active_margin_ms: float = 0.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Return per-sample mask where the reference is considered active."""
    ref_amp = np.abs(reference)
    active = ref_amp >= linear_threshold(threshold_db)
    if active_margin_ms <= 0:
        return active
    margin_samples = ms_to_samples(active_margin_ms, sample_rate)
    active_tensor = torch.from_numpy(active)
    dilated = dilate_sample_mask(active_tensor, margin_samples)
    return dilated.cpu().numpy()


def detect_onset_times(
    waveform: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Return onset times in seconds."""
    if waveform.size == 0:
        return np.array([], dtype=np.float64)
    onsets = librosa.onset.onset_detect(
        y=waveform,
        sr=sample_rate,
        units="time",
        backtrack=True,
    )
    return np.asarray(onsets, dtype=np.float64)


def filter_onsets_to_active_regions(
    onsets: np.ndarray,
    active_mask: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Keep onsets whose nearest sample falls inside the active mask."""
    if onsets.size == 0 or active_mask.size == 0:
        return np.array([], dtype=np.float64)
    kept = []
    for onset in onsets:
        sample_index = int(round(onset * sample_rate))
        sample_index = min(max(sample_index, 0), active_mask.size - 1)
        if active_mask[sample_index]:
            kept.append(onset)
    return np.asarray(kept, dtype=np.float64)


def match_onsets(
    reference_onsets: np.ndarray,
    realified_onsets: np.ndarray,
    *,
    tolerance_sec: float,
) -> tuple[int, int, int]:
    """Greedy nearest-neighbor onset matching within tolerance."""
    ref = np.sort(reference_onsets)
    real = np.sort(realified_onsets)
    if ref.size == 0 and real.size == 0:
        return 0, 0, 0
    if ref.size == 0:
        return 0, int(real.size), 0
    if real.size == 0:
        return 0, 0, int(ref.size)

    used_real = np.zeros(real.size, dtype=bool)
    matched = 0
    for ref_onset in ref:
        distances = np.abs(real - ref_onset)
        order = np.argsort(distances)
        for index in order:
            if used_real[index]:
                continue
            if distances[index] <= tolerance_sec:
                used_real[index] = True
                matched += 1
            break

    missing = int(ref.size - matched)
    extra = int(real.size - matched)
    return matched, extra, missing


def compute_f1_score(
    matched: int,
    *,
    n_reference_onsets: int,
    n_realified_onsets: int,
) -> float:
    if n_reference_onsets == 0 and n_realified_onsets == 0:
        return 1.0
    precision = matched / max(n_realified_onsets, 1)
    recall = matched / max(n_reference_onsets, 1)
    if precision + recall <= 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def score_content_fidelity(
    reference: torch.Tensor | np.ndarray,
    realified: torch.Tensor | np.ndarray,
    *,
    threshold: float = REALIFY_CONTENT_FIDELITY_THRESHOLD,
    onset_tolerance_ms: float = REALIFY_CONTENT_FIDELITY_ONSET_TOLERANCE_MS,
    active_margin_ms: float = 0.0,
    threshold_db: float = REALIFY_SILENCE_THRESHOLD_DB,
    sample_rate: int = SAMPLE_RATE,
) -> ContentFidelityResult:
    """Compare reference and realified stems using active-region onset F1."""
    reference_mono = waveform_to_mono_numpy(reference, sample_rate=sample_rate)
    realified_mono = waveform_to_mono_numpy(realified, sample_rate=sample_rate)
    n_samples = min(reference_mono.size, realified_mono.size)
    if n_samples <= 0:
        return ContentFidelityResult(
            score=1.0,
            matched_onsets=0,
            extra_onsets=0,
            missing_onsets=0,
            n_reference_onsets=0,
            n_realified_onsets=0,
            passed=True,
        )

    reference_mono = reference_mono[:n_samples]
    realified_mono = realified_mono[:n_samples]
    active_mask = reference_active_mask(
        reference_mono,
        threshold_db=threshold_db,
        active_margin_ms=active_margin_ms,
        sample_rate=sample_rate,
    )

    ref_onsets = filter_onsets_to_active_regions(
        detect_onset_times(reference_mono, sample_rate=sample_rate),
        active_mask,
        sample_rate=sample_rate,
    )
    real_onsets = filter_onsets_to_active_regions(
        detect_onset_times(realified_mono, sample_rate=sample_rate),
        active_mask,
        sample_rate=sample_rate,
    )

    tolerance_sec = onset_tolerance_ms / 1000.0
    matched, extra, missing = match_onsets(ref_onsets, real_onsets, tolerance_sec=tolerance_sec)
    score = compute_f1_score(
        matched,
        n_reference_onsets=int(ref_onsets.size),
        n_realified_onsets=int(real_onsets.size),
    )
    return ContentFidelityResult(
        score=score,
        matched_onsets=matched,
        extra_onsets=extra,
        missing_onsets=missing,
        n_reference_onsets=int(ref_onsets.size),
        n_realified_onsets=int(real_onsets.size),
        passed=score >= threshold,
    )
