"""Unit tests for post-SA3 content fidelity scoring."""

import math

import numpy as np
import torch

from shared.config import SAMPLE_RATE
from synthesis.realify.content_fidelity import (
    compute_f1_score,
    match_onsets,
    score_content_fidelity,
)


def _tone_burst(
    start_sec: float,
    *,
    n_samples: int,
    duration_sec: float = 0.05,
    frequency_hz: float = 1000.0,
    amplitude: float = 0.8,
) -> torch.Tensor:
    waveform = torch.zeros(1, n_samples)
    start = int(start_sec * SAMPLE_RATE)
    end = min(int((start_sec + duration_sec) * SAMPLE_RATE), n_samples)
    if end <= start:
        return waveform
    sample_indices = torch.arange(end - start, dtype=torch.float32) / SAMPLE_RATE
    waveform[0, start:end] = amplitude * torch.sin(2 * math.pi * frequency_hz * sample_indices)
    return waveform


def _burst_train(times_sec: list[float], *, n_samples: int) -> torch.Tensor:
    waveform = torch.zeros(1, n_samples)
    for time_sec in times_sec:
        waveform += _tone_burst(time_sec, n_samples=n_samples)
    return waveform


def test_match_onsets_perfect_alignment():
    ref = np.array([0.5, 1.0, 1.5])
    real = np.array([0.52, 1.01, 1.48])
    matched, extra, missing = match_onsets(ref, real, tolerance_sec=0.05)
    assert matched == 3
    assert extra == 0
    assert missing == 0


def test_match_onsets_detects_extra_and_missing():
    ref = np.array([0.5, 1.0, 1.5])
    real = np.array([0.5, 0.75, 1.0, 2.0])
    matched, extra, missing = match_onsets(ref, real, tolerance_sec=0.05)
    assert matched == 2
    assert extra == 2
    assert missing == 1


def test_compute_f1_score():
    assert compute_f1_score(3, n_reference_onsets=3, n_realified_onsets=3) == 1.0
    assert compute_f1_score(0, n_reference_onsets=0, n_realified_onsets=0) == 1.0


def test_score_content_fidelity_passes_identical_onsets():
    n_samples = SAMPLE_RATE * 3
    times = [0.5, 1.0, 1.5, 2.0]
    reference = _burst_train(times, n_samples=n_samples)
    realified = reference.clone()
    result = score_content_fidelity(reference, realified, threshold=0.85)
    assert result.n_reference_onsets > 0
    assert result.passed
    assert result.score >= 0.85


def test_score_content_fidelity_fails_on_extra_onsets(monkeypatch):
    n_samples = SAMPLE_RATE * 3
    reference = _burst_train([0.5, 1.0, 1.5], n_samples=n_samples)
    realified = reference.clone()
    call = {"n": 0}

    def fake_detect(_waveform, **kwargs):
        call["n"] += 1
        if call["n"] == 1:
            return np.array([0.5, 1.0, 1.5], dtype=np.float64)
        return np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=np.float64)

    monkeypatch.setattr(
        "synthesis.realify.content_fidelity.detect_onset_times",
        fake_detect,
    )
    monkeypatch.setattr(
        "synthesis.realify.content_fidelity.reference_active_mask",
        lambda reference, **kwargs: np.ones(reference.shape[-1], dtype=bool),
    )

    result = score_content_fidelity(reference, realified, threshold=0.85)
    assert not result.passed
    assert result.extra_onsets > 0


def test_score_content_fidelity_fails_on_missing_onsets(monkeypatch):
    n_samples = SAMPLE_RATE * 3
    reference = _burst_train([0.5, 1.0, 1.5, 2.0], n_samples=n_samples)
    realified = _burst_train([0.5, 1.5], n_samples=n_samples)
    call = {"n": 0}

    def fake_detect(_waveform, **kwargs):
        call["n"] += 1
        if call["n"] == 1:
            return np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        return np.array([0.5, 1.5], dtype=np.float64)

    monkeypatch.setattr(
        "synthesis.realify.content_fidelity.detect_onset_times",
        fake_detect,
    )
    monkeypatch.setattr(
        "synthesis.realify.content_fidelity.reference_active_mask",
        lambda reference, **kwargs: np.ones(reference.shape[-1], dtype=bool),
    )

    result = score_content_fidelity(reference, realified, threshold=0.85)
    assert not result.passed
    assert result.missing_onsets > 0


def test_realify_with_content_fidelity_backoff_passthrough(monkeypatch, tmp_path):
    from synthesis.realify.content_fidelity import ContentFidelityResult
    from synthesis.realify.realify import _realify_with_content_fidelity_backoff

    n_samples = SAMPLE_RATE * 2
    reference = _burst_train([0.25, 0.5], n_samples=n_samples)
    attempts = {"count": 0}

    def fake_generate_and_enforce(**kwargs):
        attempts["count"] += 1
        return _burst_train([0.25, 0.5, 0.75], n_samples=n_samples)

    def fake_score(reference, realified, **kwargs):
        return ContentFidelityResult(
            score=0.2,
            matched_onsets=0,
            extra_onsets=1,
            missing_onsets=1,
            n_reference_onsets=2,
            n_realified_onsets=3,
            passed=False,
        )

    monkeypatch.setattr(
        "synthesis.realify.realify._generate_and_enforce",
        fake_generate_and_enforce,
    )
    monkeypatch.setattr(
        "synthesis.realify.content_fidelity.score_content_fidelity",
        fake_score,
    )

    audio = _realify_with_content_fidelity_backoff(
        reference=reference,
        preset={"init_noise_level": 0.45},
        model=object(),
        prompt="solo piano",
        init_audio=(SAMPLE_RATE, reference),
        duration_seconds=2.0,
        seed=1,
        silence_enforce=False,
        output_path=tmp_path / "out.flac",
    )
    assert attempts["count"] >= 2
    assert torch.allclose(audio, reference)
