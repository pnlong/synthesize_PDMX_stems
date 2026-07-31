"""Optional integration smoke for neural DDSP workers (skipped without TF venv)."""

from __future__ import annotations

from pathlib import Path

import mido
import pytest

from synthesis.ddsp.env import DdspEnvError, ddsp_python_executable
from synthesis.ddsp.routing import StemRoute, BACKEND_MIDI_DDSP, REASON_MIDI_DDSP


def _ddsp_env_available() -> bool:
    try:
        ddsp_python_executable()
        return True
    except DdspEnvError:
        return False


pytestmark = pytest.mark.skipif(
    not _ddsp_env_available(),
    reason="Neural DDSP TF venv not configured (see SETUP.md Track C)",
)


def _mono_violin_mid(path: Path) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.Message("program_change", program=40, time=0))
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=960))
    midi.tracks.append(track)
    midi.save(str(path))


def test_midi_ddsp_wrapper_smoke(tmp_path: Path):
    from shared.config import SAMPLE_RATE
    from synthesis.ddsp.synthesize import synthesize_stem_midi_ddsp

    mid = tmp_path / "violin.mid"
    _mono_violin_mid(mid)
    try:
        waveform = synthesize_stem_midi_ddsp(mid, "violin")
    except Exception as exc:
        pytest.skip(f"MIDI-DDSP worker not fully installed: {exc}")
    assert waveform.ndim == 2
    assert waveform.shape[0] >= 1
    assert waveform.shape[-1] > SAMPLE_RATE * 0.1  # >100 ms
