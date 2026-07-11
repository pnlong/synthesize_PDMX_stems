"""Tests for mono downmix and BS.1770 loudness normalization."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from shared.config import SAMPLE_RATE, TARGET_LOUDNESS_LUFS
from synthesis.audio import (
    build_mixture,
    loudness_normalize,
    pad_and_loudness_normalize,
    song_is_complete,
    to_mono_numpy,
    write_mixture_from_song_dir,
)


def test_to_mono_from_stereo_tensor():
    stereo = torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
    mono = to_mono_numpy(stereo)
    assert mono.shape == (3,)
    np.testing.assert_allclose(mono, [0.0, 0.0, 0.0])


def test_loudness_normalize_non_silent():
    t = torch.linspace(0, 1, SAMPLE_RATE)
    waveform = torch.sin(2 * np.pi * 440 * t).unsqueeze(0) * 0.01
    normalized = loudness_normalize(waveform)
    assert normalized.shape == waveform.shape
    assert normalized.abs().max() > waveform.abs().max()


def test_pad_and_loudness_equal_length():
    sr = SAMPLE_RATE
    w1 = torch.randn(1, sr)
    w2 = torch.randn(1, sr * 2)
    padded = pad_and_loudness_normalize([w1, w2])
    assert padded[0].shape[-1] == padded[1].shape[-1] == sr * 2


def test_build_mixture_scales_when_clipping():
    w1 = torch.ones(1, 4) * 0.8
    w2 = torch.ones(1, 4) * 0.8
    mixture = build_mixture([w1, w2], peak_limit=1.0)
    assert mixture.abs().max().item() <= 1.0 + 1e-6
    np.testing.assert_allclose(to_mono_numpy(mixture), [1.0, 1.0, 1.0, 1.0], rtol=1e-5)


def test_build_mixture_single_stem():
    w = torch.linspace(0, 1, 100).unsqueeze(0)
    mixture = build_mixture([w])
    np.testing.assert_allclose(to_mono_numpy(mixture), to_mono_numpy(w))


def test_write_mixture_from_song_dir(tmp_path: Path):
    song_dir = tmp_path / "song"
    song_dir.mkdir()
    sr = 44100
    sf.write(str(song_dir / "stem_0.flac"), np.full(sr, 0.5, dtype=np.float32), sr, format="FLAC")
    sf.write(str(song_dir / "stem_1.flac"), np.full(sr, 0.5, dtype=np.float32), sr, format="FLAC")
    out = write_mixture_from_song_dir(song_dir, [0, 1])
    assert out is not None
    assert out.name == "mixture.flac"
    assert out.exists()


def test_song_is_complete_requires_mixture(tmp_path: Path):
    song_dir = tmp_path / "song"
    song_dir.mkdir()
    sr = 44100
    sf.write(str(song_dir / "stem_0.flac"), np.zeros(sr, dtype=np.float32), sr, format="FLAC")
    assert not song_is_complete(song_dir, 1)
    write_mixture_from_song_dir(song_dir, [0])
    assert song_is_complete(song_dir, 1)


def test_loudness_normalize_silent_passthrough():
    silent = torch.zeros(1, SAMPLE_RATE)
    out = loudness_normalize(silent)
    assert out.shape == silent.shape
