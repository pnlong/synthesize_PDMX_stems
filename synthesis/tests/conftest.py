"""Pytest configuration for synthesis tests."""

from pathlib import Path

import mido
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TINY_MID = FIXTURES_DIR / "tiny.mid"


@pytest.fixture(scope="session")
def tiny_mid_path() -> Path:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    if not TINY_MID.exists():
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        track.append(mido.Message("program_change", program=0, time=0))
        track.append(mido.Message("note_on", note=60, velocity=80, time=0, channel=0))
        track.append(mido.Message("note_off", note=60, velocity=0, time=480, channel=0))
        midi.tracks.append(track)
        midi.save(str(TINY_MID))
    return TINY_MID
