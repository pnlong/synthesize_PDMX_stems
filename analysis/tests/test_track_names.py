"""Tests for track name analysis."""

from pathlib import Path

import matplotlib
import mido
import pandas as pd

matplotlib.use("Agg")

from analysis.analyze_track_names import analyze_track_names
from analysis.plots import plot_track_name_bar
from analysis.track_names import (
    UNNAMED_TRACK,
    build_track_name_report,
    extract_named_stems_from_mid,
    normalize_track_name,
)


def _write_minimal_midi(path: Path, *, program: int = 0, channel: int = 0, name: str | None = "Piano"):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    if name is not None:
        track.append(mido.MetaMessage("track_name", name=name, time=0))
    track.append(mido.Message("program_change", program=program, channel=channel, time=0))
    track.append(mido.Message("note_on", note=60, velocity=80, channel=channel, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, channel=channel, time=480))
    midi.tracks.append(track)
    midi.save(path)


def test_normalize_track_name():
    assert normalize_track_name("Piano") == "piano"
    assert normalize_track_name("  SOPRANO ") == "soprano"
    assert normalize_track_name("") == UNNAMED_TRACK
    assert normalize_track_name(None) == UNNAMED_TRACK


def test_extract_named_stems_from_mid(tmp_path: Path):
    mid_path = tmp_path / "song.mid"
    _write_minimal_midi(mid_path, program=56, channel=0, name="Trumpet")
    rows = extract_named_stems_from_mid(mid_path)
    assert rows is not None
    assert len(rows) == 1
    assert rows[0]["track_name"] == "trumpet"
    assert rows[0]["category"] == "brass"


def test_extract_unnamed_track(tmp_path: Path):
    mid_path = tmp_path / "song.mid"
    _write_minimal_midi(mid_path, name=None)
    rows = extract_named_stems_from_mid(mid_path)
    assert rows is not None
    assert rows[0]["track_name"] == UNNAMED_TRACK


def test_build_track_name_report():
    stems = pd.DataFrame([
        {"track_name": "piano", "display_name": "Piano", "program": 0, "is_drum": False, "gm_class": "piano", "category": "piano"},
        {"track_name": "piano", "display_name": "Piano", "program": 0, "is_drum": False, "gm_class": "piano", "category": "piano"},
        {"track_name": UNNAMED_TRACK, "display_name": UNNAMED_TRACK, "program": 0, "is_drum": False, "gm_class": "piano", "category": "default"},
    ])
    report = build_track_name_report(stems, subset="all_valid", n_songs=1, n_songs_failed=0)
    assert report["n_stems"] == 3
    assert report["n_named_stems"] == 2
    assert report["n_unnamed_stems"] == 1


def test_plot_track_name_bar(tmp_path: Path):
    stems = pd.DataFrame([
        {"track_name": "piano", "display_name": "Piano", "program": 0, "is_drum": False, "gm_class": "piano", "category": "piano"},
        {"track_name": "flute", "display_name": "Flute", "program": 73, "is_drum": False, "gm_class": "pipe", "category": "wind"},
    ])
    out = tmp_path / "track_name_counts.png"
    plot_track_name_bar(stems, out, top_n=10)
    assert out.exists()


def test_analyze_track_names_on_fake_pdmx(tmp_path: Path):
    pdmx_root = tmp_path / "PDMX"
    pdmx_root.mkdir()
    mid_rel = "/0/test/song.mid"
    mid_path = pdmx_root / "0/test/song.mid"
    mid_path.parent.mkdir(parents=True)
    _write_minimal_midi(mid_path, program=0, channel=0, name="Piano")

    csv_path = pdmx_root / "PDMX.csv"
    pd.DataFrame({
        "mid": [mid_rel],
        "subset:all_valid": [True],
    }).to_csv(csv_path, index=False)

    report, stems = analyze_track_names(csv_path, subset="all_valid", jobs=1)
    assert report["n_songs"] == 1
    assert report["n_stems"] == 1
    assert stems.iloc[0]["track_name"] == "piano"
