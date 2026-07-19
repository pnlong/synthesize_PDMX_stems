"""Unit tests for post-SA3 silence enforcement."""

import pytest
import torch

from synthesis.realify.silence import (
    apply_boundary_crossfade,
    apply_silence_enforcement,
    detect_hallucination_mask,
    dilate_sample_mask,
    ms_to_samples,
    window_peak_db,
)


def _tone(start: int, end: int, n_samples: int, *, amplitude: float = 0.5) -> torch.Tensor:
    waveform = torch.zeros(1, n_samples)
    waveform[0, start:end] = amplitude
    return waveform


def test_window_peak_db_silent():
    waveform = torch.zeros(1, 1000)
    assert window_peak_db(waveform, 0, 1000) < -50.0


def test_window_peak_db_active():
    waveform = torch.ones(1, 1000) * 0.5
    assert window_peak_db(waveform, 0, 1000) > -10.0


def test_dilate_sample_mask_expands_active_region():
    mask = torch.zeros(20, dtype=torch.bool)
    mask[5:10] = True
    dilated = dilate_sample_mask(mask, margin_samples=3)
    assert dilated[2:13].all()
    assert not dilated[0]
    assert not dilated[-1]


def test_detect_hallucination_mask_finds_mismatch():
    n = ms_to_samples(3000)
    chunk = ms_to_samples(500)
    overlap = chunk // 2
    reference = _tone(ms_to_samples(500), ms_to_samples(1500), n)
    realified = reference.clone()
    realified[0, ms_to_samples(1600) : ms_to_samples(2400)] = 0.3

    mask = detect_hallucination_mask(
        reference,
        realified,
        chunk_samples=chunk,
        overlap_samples=overlap,
        threshold_db=-40.0,
        margin_samples=0,
    )
    assert mask[ms_to_samples(1600) : ms_to_samples(2400)].any()
    assert not mask[ms_to_samples(500) : ms_to_samples(1500)].any()


def test_detect_hallucination_mask_skips_when_both_silent():
    n = ms_to_samples(2000)
    chunk = ms_to_samples(500)
    overlap = chunk // 2
    reference = torch.zeros(1, n)
    realified = torch.zeros(1, n)
    realified[0, ms_to_samples(500) : ms_to_samples(1000)] = 0.01

    mask = detect_hallucination_mask(
        reference,
        realified,
        chunk_samples=chunk,
        overlap_samples=overlap,
        threshold_db=-40.0,
        margin_samples=0,
    )
    assert not mask.any()


def test_active_margin_protects_tail():
    n = ms_to_samples(3000)
    chunk = ms_to_samples(500)
    overlap = chunk // 2
    reference = _tone(ms_to_samples(500), ms_to_samples(1200), n, amplitude=0.5)
    realified = reference.clone()
    realified[0, ms_to_samples(1300) : ms_to_samples(2000)] = 0.2

    without_margin = detect_hallucination_mask(
        reference,
        realified,
        chunk_samples=chunk,
        overlap_samples=overlap,
        threshold_db=-40.0,
        margin_samples=0,
    )
    with_margin = detect_hallucination_mask(
        reference,
        realified,
        chunk_samples=chunk,
        overlap_samples=overlap,
        threshold_db=-40.0,
        margin_samples=ms_to_samples(400),
    )
    assert without_margin[ms_to_samples(1300) : ms_to_samples(1500)].any()
    assert not with_margin[ms_to_samples(1300) : ms_to_samples(1500)].any()


def test_apply_silence_enforcement_zeros_hallucination():
    n = ms_to_samples(3000)
    reference = _tone(ms_to_samples(500), ms_to_samples(1500), n)
    realified = reference.clone()
    realified[0, ms_to_samples(1600) : ms_to_samples(2400)] = 0.4

    output = apply_silence_enforcement(
        reference,
        realified,
        chunk_ms=500.0,
        overlap_ratio=0.5,
        threshold_db=-40.0,
        active_margin_ms=0.0,
        fade_ms=0.0,
    )
    silent_region = output[0, ms_to_samples(1600) : ms_to_samples(2400)]
    assert float(silent_region.abs().max()) == 0.0
    active_region = output[0, ms_to_samples(600) : ms_to_samples(1400)]
    assert float(active_region.abs().max()) > 0.3


def test_apply_silence_enforcement_disabled_passthrough():
    reference = torch.zeros(1, 1000)
    realified = torch.ones(1, 1000) * 0.5
    output = apply_silence_enforcement(
        reference,
        realified,
        enabled=False,
    )
    assert torch.allclose(output, realified)


def test_apply_boundary_crossfade_smooths_edges():
    n = 200
    original = torch.ones(1, n) * 0.8
    output = torch.zeros(1, n)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[80:120] = True

    faded = apply_boundary_crossfade(output, original, mask, fade_samples=10)
    assert faded[0, 79] == 0.0
    assert 0.0 < float(faded[0, 80]) < 0.8
    assert float(faded[0, 119]) < 0.8
    assert float(faded[0, 120]) > 0.0
    assert float(faded[0, 129]) == pytest.approx(0.8, abs=0.01)


def test_apply_boundary_crossfade_noop_when_no_silent_regions():
    original = torch.ones(1, 100) * 0.5
    output = original.clone()
    mask = torch.zeros(100, dtype=torch.bool)
    result = apply_boundary_crossfade(output, original, mask, fade_samples=10)
    assert torch.allclose(result, output)
