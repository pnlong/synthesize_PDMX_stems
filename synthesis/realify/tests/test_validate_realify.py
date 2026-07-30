"""Tests for validate_realify and silence hallucination scoring."""

from pathlib import Path

import numpy as np
import torch

from synthesis.realify.silence import score_silence_hallucinations
from synthesis.realify.validate_realify import (
    score_existing_pair,
    validate_with_backoff,
)


def test_score_silence_hallucinations_clean():
    sr = 44100
    reference = torch.zeros(1, sr)
    reference[0, 1000:2000] = 0.5
    realified = reference.clone()
    result = score_silence_hallucinations(reference, realified)
    assert result.passed
    assert result.n_hallucination_samples == 0


def test_score_silence_hallucinations_detects_rest_noise():
    sr = 44100
    reference = torch.zeros(1, sr * 4)
    reference[0, sr: sr + 44100] = 0.5
    realified = reference.clone()
    realified[0, sr * 3: sr * 3 + 5000] = 0.4
    result = score_silence_hallucinations(reference, realified)
    assert not result.passed
    assert result.n_hallucination_samples > 0


def test_score_existing_pair(tmp_path: Path):
    import soundfile as sf

    sr = 44100
    t = np.linspace(0, 1, sr)
    ref = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    ref_path = tmp_path / "ref.flac"
    out_path = tmp_path / "out.flac"
    sf.write(str(ref_path), ref, sr, format="FLAC")
    sf.write(str(out_path), ref, sr, format="FLAC")

    result = score_existing_pair(
        reference_path=ref_path,
        realified_path=out_path,
        threshold=0.85,
        silence_enforce=True,
        stem_id="test",
    )
    assert result.overall_passed


def test_validate_with_backoff_passthrough(monkeypatch, tmp_path: Path):
    from synthesis.realify.content_fidelity import ContentFidelityResult

    sr = 44100
    reference = torch.zeros(1, sr)
    reference[0, 1000:5000] = 0.4

    class FakeModel:
        model_config = {"sample_size": sr * 120}

    fail = ContentFidelityResult(
        score=0.2,
        matched_onsets=1,
        extra_onsets=5,
        missing_onsets=0,
        n_reference_onsets=5,
        n_realified_onsets=10,
        passed=False,
    )

    monkeypatch.setattr(
        "synthesis.realify.validate_realify._generate_and_enforce",
        lambda **kwargs: reference + 0.01,
    )
    monkeypatch.setattr(
        "synthesis.realify.validate_realify.score_content_fidelity",
        lambda *args, **kwargs: fail,
    )
    monkeypatch.setattr(
        "synthesis.realify.validate_realify.load_stem",
        lambda path: reference,
    )

    out_path = tmp_path / "validated.flac"
    result = validate_with_backoff(
        stem_path=tmp_path / "stem.flac",
        output_path=out_path,
        row={"prompt": "piano", "stem_id": "test", "category": "piano"},
        preset={"init_noise_level": 0.55, "steps": 8, "cfg_scale": 1.0},
        model=FakeModel(),
        duration_seconds=1.0,
        seed=1,
        threshold=0.85,
        silence_enforce=False,
        audio_format="flac",
    )
    assert result.attempts[-1].used_reference_passthrough
    assert not result.overall_passed
    assert out_path.is_file()
    assert (tmp_path / "test_reference.flac").is_file()
