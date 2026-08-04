"""Tests for MIDI velocity dynamics scaling."""

from pathlib import Path

import mido
import numpy as np
import pandas as pd
import pytest
import torch

from synthesis.velocity import (
    apply_velocity_scales,
    max_note_on_velocity,
    pdmx_mid_from_song_dir,
    velocity_scales_for_midi,
    velocity_scales_from_track_maxima,
)


def _track_with_velocities(*velocities: int) -> mido.MidiTrack:
    track = mido.MidiTrack()
    for vel in velocities:
        track.append(mido.Message("note_on", note=60, velocity=vel, time=0))
        track.append(mido.Message("note_off", note=60, velocity=0, time=10))
    return track


def test_max_note_on_velocity_ignores_zero():
    track = _track_with_velocities(0, 64, 40)
    assert max_note_on_velocity(track) == 64


def test_velocity_scales_from_track_maxima():
    scales = velocity_scales_from_track_maxima({0: 64, 1: 127, 2: 0})
    assert scales[0] == pytest.approx(64 / 127)
    assert scales[1] == pytest.approx(1.0)
    assert scales[2] == 0.0


def test_velocity_scales_song_max_zero_is_identity():
    scales = velocity_scales_from_track_maxima({0: 0, 1: 0})
    assert scales == {0: 1.0, 1: 1.0}


def test_velocity_scales_for_midi(tmp_path: Path):
    mid = mido.MidiFile()
    mid.tracks.append(_track_with_velocities(64))
    mid.tracks.append(_track_with_velocities(127))
    path = tmp_path / "song.mid"
    mid.save(path)
    scales = velocity_scales_for_midi(path)
    assert scales[0] == pytest.approx(64 / 127)
    assert scales[1] == pytest.approx(1.0)


def test_apply_velocity_scales():
    w0 = torch.ones(1, 4)
    w1 = torch.ones(1, 4) * 2
    out = apply_velocity_scales([w0, w1], [0, 1], {0: 0.5, 1: 1.0})
    np.testing.assert_allclose(out[0].numpy(), 0.5)
    np.testing.assert_allclose(out[1].numpy(), 2.0)


def test_pdmx_mid_from_song_dir():
    song = Path("/deepfreeze/x/ablations/basic/data/4/5/QmHash")
    mid = pdmx_mid_from_song_dir(song, Path("/pdmx"))
    assert mid == Path("/pdmx/mid/4/5/QmHash.mid")


def test_normalize_applies_velocity_before_peak(tmp_path: Path):
    """Quieter MIDI track stays quieter after LUFS + velocity + peak."""
    from synthesis.audio import normalize_stems_in_song_dir, load_stem, to_mono_numpy
    import soundfile as sf
    from shared.config import SAMPLE_RATE, FLAC_AUDIO_FORMAT

    song = tmp_path / "song"
    song.mkdir()
    sr = SAMPLE_RATE
    # Same raw level; velocity scales will differentiate after LUFS.
    sf.write(str(song / "stem_0.flac"), np.full(sr, 0.2, np.float32), sr, format="FLAC")
    sf.write(str(song / "stem_1.flac"), np.full(sr, 0.2, np.float32), sr, format="FLAC")
    gain = normalize_stems_in_song_dir(
        song,
        [0, 1],
        FLAC_AUDIO_FORMAT,
        velocity_scales={0: 0.5, 1: 1.0},
    )
    assert gain is not None
    s0 = load_stem(song / "stem_0.flac")
    s1 = load_stem(song / "stem_1.flac")
    assert float(s0.abs().mean()) < float(s1.abs().mean())
    assert (s0 + s1).abs().max().item() <= 1.0 + 1e-4


def test_mix_prefers_persisted_velocity_scale(tmp_path: Path):
    from synthesis.mix import build_mixture_tasks

    source = tmp_path / "basic"
    song = source / "data" / "song"
    song.mkdir(parents=True)
    stems = pd.DataFrame({
        "path": [str(song), str(song)],
        "track": [0, 1],
        "velocity_scale": [0.25, 1.0],
    })
    tasks = build_mixture_tasks(
        stems, source, source, "flac", use_velocity_dynamics=True,
    )
    assert tasks[0]["velocity_scales"] == {0: 0.25, 1: 1.0}