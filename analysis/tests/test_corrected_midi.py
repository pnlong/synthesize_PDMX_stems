"""Tests for dense corrected MIDI writing and feature flag."""

from __future__ import annotations

from pathlib import Path

import mido
import pandas as pd
import pytest

from analysis.corrected_midi import (
    load_track_map,
    note_on_count,
    resolve_corrected_midi_path,
    write_corrected_midi,
)
from synthesis.dense_midi import dense_midi_enabled, stem_original_track


def _song_with_empty_stub(tmp_path: Path) -> Path:
    mid = mido.MidiFile(ticks_per_beat=480)
    # Track 0: conductor tempo only
    t0 = mido.MidiTrack()
    t0.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    t0.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    t0.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(t0)
    # Track 1: empty grand-staff stub
    t1 = mido.MidiTrack()
    t1.append(mido.MetaMessage("track_name", name="Marimba (Grand Staff)", time=0))
    t1.append(mido.MetaMessage("end_of_track", time=1))
    mid.tracks.append(t1)
    # Track 2: notes + program
    t2 = mido.MidiTrack()
    t2.append(mido.MetaMessage("track_name", name="Marimba", time=0))
    t2.append(mido.Message("program_change", program=0, channel=0, time=0))
    t2.append(mido.Message("note_on", note=60, velocity=80, channel=0, time=0))
    t2.append(mido.Message("note_on", note=60, velocity=0, channel=0, time=480))
    t2.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(t2)
    path = tmp_path / "src.mid"
    mid.save(path)
    return path


def test_write_corrected_midi_drops_empty_and_applies_register(tmp_path: Path):
    src = _song_with_empty_stub(tmp_path)
    dest = tmp_path / "out" / "song.mid"
    rows = write_corrected_midi(
        src,
        dest,
        program_by_original_track={2: 12},
        mid_rel="./mid/x.mid",
    )
    assert dest.is_file()
    assert len(rows) == 1
    assert rows[0]["track"] == 0
    assert rows[0]["original_track"] == 2
    assert rows[0]["program"] == 12

    out = mido.MidiFile(filename=str(dest), charset="utf8")
    assert len(out.tracks) == 1
    assert note_on_count(out.tracks[0]) > 0
    programs = [m.program for m in out.tracks[0] if m.type == "program_change"]
    assert programs and programs[0] == 12
    tempos = [m for m in out.tracks[0] if m.type == "set_tempo"]
    assert tempos

    tmap = load_track_map(dest)
    assert tmap[0]["original_track"] == 2


def test_resolve_corrected_midi_path_strips_mid_prefix(tmp_path: Path):
    pdmx_root = tmp_path / "PDMX"
    src = pdmx_root / "mid" / "8" / "44" / "Qm.mid"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"mthd")
    corrected_root = tmp_path / "mid_corrected"
    path = resolve_corrected_midi_path(
        src, pdmx_root=pdmx_root, corrected_midi_dir=corrected_root
    )
    assert path == corrected_root / "8" / "44" / "Qm.mid"


def test_dense_midi_flag_default_off(monkeypatch):
    monkeypatch.delenv("SPDMX_DENSE_MIDI", raising=False)

    class NS:
        dense_midi = None

    assert dense_midi_enabled(NS()) is False


def test_dense_midi_flag_env_and_cli(monkeypatch):
    monkeypatch.setenv("SPDMX_DENSE_MIDI", "1")

    class NS:
        dense_midi = None

    assert dense_midi_enabled(NS()) is True

    class Off:
        dense_midi = False

    assert dense_midi_enabled(Off()) is False

    class On:
        dense_midi = True

    monkeypatch.delenv("SPDMX_DENSE_MIDI", raising=False)
    assert dense_midi_enabled(On()) is True


def test_stem_original_track_legacy_fallback():
    assert stem_original_track({"track": 3}) == 3
    assert stem_original_track({"track": 1, "original_track": 5}) == 5
    assert stem_original_track(pd.Series({"track": 2, "original_track": float("nan")})) == 2
