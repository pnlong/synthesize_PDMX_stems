"""Unit tests for neural-DDSP stem routing and monophony checks."""

from __future__ import annotations

from pathlib import Path

import mido
import pytest

from synthesis.ddsp.routing import (
    BACKEND_DDSP_PIANO,
    BACKEND_MIDI_DDSP,
    BACKEND_SOUNDFONT,
    REASON_BASS_GUITAR,
    REASON_DRUM,
    REASON_MIDI_DDSP,
    REASON_PIANO,
    REASON_POLYPHONIC,
    REASON_UNSUPPORTED,
    REASON_VOCAL,
    is_monophonic_messages,
    is_monophonic_midi,
    route_stem,
)


def _track(*messages) -> mido.MidiTrack:
    track = mido.MidiTrack()
    for message in messages:
        track.append(message)
    return track


def test_route_piano_by_program():
    route = route_stem(program=0, is_drum=False, check_monophony=False)
    assert route.backend == BACKEND_DDSP_PIANO
    assert route.reason == REASON_PIANO


def test_route_piano_by_name():
    route = route_stem(program=48, is_drum=False, track_name="Grand Piano", check_monophony=False)
    assert route.backend == BACKEND_DDSP_PIANO


def test_route_violin_mono():
    track = _track(
        mido.Message("program_change", program=40, time=0),
        mido.Message("note_on", note=60, velocity=80, time=0),
        mido.Message("note_off", note=60, velocity=0, time=480),
        mido.Message("note_on", note=62, velocity=80, time=0),
        mido.Message("note_off", note=62, velocity=0, time=480),
    )
    route = route_stem(program=40, is_drum=False, track=track, check_monophony=True)
    assert route.backend == BACKEND_MIDI_DDSP
    assert route.instrument_key == "violin"
    assert route.reason == REASON_MIDI_DDSP


def test_route_violin_polyphonic_falls_back():
    track = _track(
        mido.Message("program_change", program=40, time=0),
        mido.Message("note_on", note=60, velocity=80, time=0),
        mido.Message("note_on", note=64, velocity=80, time=0),
        mido.Message("note_off", note=60, velocity=0, time=480),
        mido.Message("note_off", note=64, velocity=0, time=0),
    )
    route = route_stem(program=40, is_drum=False, track=track, check_monophony=True)
    assert route.backend == BACKEND_SOUNDFONT
    assert route.reason == REASON_POLYPHONIC
    assert route.instrument_key == "violin"


def test_route_double_bass_not_bass_guitar():
    route = route_stem(program=43, is_drum=False, check_monophony=False)
    assert route.backend == BACKEND_MIDI_DDSP
    assert route.instrument_key == "double bass"

    bass_guitar = route_stem(program=33, is_drum=False, check_monophony=False)
    assert bass_guitar.backend == BACKEND_SOUNDFONT
    assert bass_guitar.reason == REASON_BASS_GUITAR


def test_route_drums_and_vocals():
    assert route_stem(program=0, is_drum=True, check_monophony=False).reason == REASON_DRUM
    vocal = route_stem(program=52, is_drum=False, track_name="Choir", check_monophony=False)
    assert vocal.backend == BACKEND_SOUNDFONT
    assert vocal.reason == REASON_VOCAL


def test_route_unsupported():
    route = route_stem(program=80, is_drum=False, check_monophony=False)  # synth lead
    assert route.backend == BACKEND_SOUNDFONT
    assert route.reason == REASON_UNSUPPORTED


def test_is_monophonic_messages_and_file(tmp_path: Path):
    mono = _track(
        mido.Message("note_on", note=60, velocity=80, time=0),
        mido.Message("note_off", note=60, velocity=0, time=100),
    )
    assert is_monophonic_messages(mono)

    poly = _track(
        mido.Message("note_on", note=60, velocity=80, time=0),
        mido.Message("note_on", note=64, velocity=80, time=0),
        mido.Message("note_off", note=60, velocity=0, time=100),
        mido.Message("note_off", note=64, velocity=0, time=0),
    )
    assert not is_monophonic_messages(poly)

    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(mono)
    path = tmp_path / "mono.mid"
    midi.save(str(path))
    assert is_monophonic_midi(path)


def test_synthesize_stem_neural_mocked(monkeypatch, tmp_path: Path):
    import torch

    from synthesis.ddsp.routing import StemRoute
    from synthesis.ddsp import synthesize as synth_mod

    midi = mido.MidiFile(ticks_per_beat=480)
    track = _track(
        mido.Message("program_change", program=40, time=0),
        mido.Message("note_on", note=60, velocity=80, time=0),
        mido.Message("note_off", note=60, velocity=0, time=480),
    )
    midi.tracks.append(track)
    mid_path = tmp_path / "v.mid"
    midi.save(str(mid_path))

    fake = torch.zeros(1, 1000)

    def fake_midi_ddsp(midi_path, instrument_name, **kwargs):
        assert instrument_name == "violin"
        return fake

    monkeypatch.setattr(synth_mod, "synthesize_stem_midi_ddsp", fake_midi_ddsp)
    out = synth_mod.synthesize_stem_neural(
        mid_path,
        StemRoute(BACKEND_MIDI_DDSP, "violin", REASON_MIDI_DDSP),
    )
    assert out is fake
