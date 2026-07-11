"""Tests for mono downmix and BS.1770 loudness normalization."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from shared.config import (
    FLAC_SUBTYPE,
    MAX_N_SAMPLES_IN_STEM,
    PROTOTYPE_AUDIO_FORMAT,
    SAMPLE_RATE,
)
from synthesis.audio import (
    build_mixture,
    load_stem_flac,
    loudness_normalize,
    pad_and_loudness_normalize,
    save_stem,
    song_is_complete,
    stem_flac_is_valid,
    to_mono_numpy,
    truncate_waveform,
    write_flac,
    write_mixture_from_song_dir,
    write_mp3,
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
    assert normalized.abs().max() <= 1.0 + 1e-6


def test_loudness_normalize_caps_peak_for_quiet_sparse_stem():
    """Quiet stem with loud peaks: unlimited LUFS gain would clip; we peak-limit."""
    sr = SAMPLE_RATE
    audio = np.zeros(sr, dtype=np.float64)
    audio[1000] = 0.5
    audio[5000] = 0.3
    waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    normalized = loudness_normalize(waveform)
    assert normalized.abs().max().item() <= 1.0 + 1e-6
    assert normalized.abs().max().item() > 0.3


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


def test_truncate_waveform():
    w = torch.randn(1, MAX_N_SAMPLES_IN_STEM + 1000)
    out = truncate_waveform(w)
    assert out.shape[-1] == MAX_N_SAMPLES_IN_STEM
    short = torch.randn(1, 100)
    assert truncate_waveform(short).shape == short.shape


def test_stem_flac_is_valid_rejects_oversized(tmp_path: Path, monkeypatch):
    path = tmp_path / "stem_0.flac"
    path.write_bytes(b"not a real flac")
    assert not stem_flac_is_valid(path)

    import synthesis.audio as audio_mod

    class FakeInfo:
        frames = MAX_N_SAMPLES_IN_STEM + 1
        samplerate = SAMPLE_RATE

    monkeypatch.setattr(audio_mod.sf, "info", lambda _: FakeInfo())
    path.write_bytes(b"x" * 8)
    assert not stem_flac_is_valid(path)


def test_write_mp3(tmp_path: Path, monkeypatch):
    import synthesis.audio as audio_mod

    captured = {}

    def fake_save(path, tensor, sample_rate, format):
        captured.update(path=path, format=format, shape=tuple(tensor.shape))

    monkeypatch.setattr("torchaudio.save", fake_save)
    write_mp3(torch.ones(1, 100), tmp_path / "stem_0.mp3")
    assert captured["format"] == "mp3"
    assert captured["shape"] == (1, 100)


def test_save_stem_mp3_uses_mp3_extension(tmp_path: Path, monkeypatch):
    import synthesis.audio as audio_mod

    monkeypatch.setattr("torchaudio.save", lambda *args, **kwargs: None)
    out = save_stem(torch.ones(1, 100), tmp_path, 0, PROTOTYPE_AUDIO_FORMAT)
    assert out.name == "stem_0.mp3"


def test_write_flac_uses_pcm_16_subtype(tmp_path: Path, monkeypatch):
    import synthesis.audio as audio_mod

    captured = {}

    def fake_write(path, audio, sr, format, subtype):
        captured.update(path=path, subtype=subtype, dtype=audio.dtype)

    monkeypatch.setattr(audio_mod.sf, "write", fake_write)
    write_flac(torch.ones(1, 100), tmp_path / "stem_0.flac")
    assert captured["subtype"] == FLAC_SUBTYPE
    assert captured["dtype"] == np.float32


def test_load_stem_flac_caps_frames(tmp_path: Path, monkeypatch):
    import synthesis.audio as audio_mod

    class FakeInfo:
        frames = MAX_N_SAMPLES_IN_STEM + 5000
        samplerate = SAMPLE_RATE

    monkeypatch.setattr(audio_mod.sf, "info", lambda _: FakeInfo())

    def fake_read(path, frames, dtype, always_2d):
        assert frames == MAX_N_SAMPLES_IN_STEM
        return np.zeros(frames, dtype=np.float32), SAMPLE_RATE

    monkeypatch.setattr(audio_mod.sf, "read", fake_read)
    out = load_stem_flac(tmp_path / "stem_0.flac")
    assert out.shape == (1, MAX_N_SAMPLES_IN_STEM)
