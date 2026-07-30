"""Tests for GM program analysis."""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from analysis.analyze_gm_programs import analyze_gm_programs
from analysis.gm_programs import (
    build_gm_report,
    gm_id_label,
    gm_program_name,
    parse_tracks_cell,
    tracks_cell_to_stem_records,
)
from analysis.plots import plot_gm_program_bar


def test_gm_id_labels():
    assert gm_program_name(0) == "Acoustic Grand Piano"
    assert gm_program_name(56) == "Trumpet"
    assert gm_id_label(0).startswith("0:")
    assert gm_id_label(56) == "56: Trumpet"


def test_parse_tracks_cell():
    assert parse_tracks_cell("0") == [0]
    assert parse_tracks_cell("0-42-43-48-48") == [0, 42, 43, 48, 48]
    assert parse_tracks_cell("") is None
    assert parse_tracks_cell(float("nan")) is None
    assert parse_tracks_cell("0-bad") is None


def test_tracks_cell_to_stem_records():
    rows = tracks_cell_to_stem_records("0-56")
    assert len(rows) == 2
    assert rows[0]["gm_id"] == 0
    assert rows[0]["gm_class"] == "piano"
    assert rows[1]["gm_id"] == 56
    assert rows[1]["gm_class"] == "brass"


def test_build_gm_report():
    stems = pd.DataFrame([
        {"gm_id": 0, "program": 0, "gm_class": "piano", "category": "default"},
        {"gm_id": 56, "program": 56, "gm_class": "brass", "category": "default"},
    ])
    report = build_gm_report(stems, subset="all_valid", n_songs=1, n_songs_failed=0)
    assert report["n_stems"] == 2
    assert report["program_0_count"] == 1
    assert len(report["gm_programs"]) == 2


def test_plot_gm_program_bar(tmp_path: Path):
    stems = pd.DataFrame([
        {"gm_id": 0, "program": 0, "gm_class": "piano", "category": "default"},
        {"gm_id": 56, "program": 56, "gm_class": "brass", "category": "default"},
    ])
    out = tmp_path / "gm_program_counts.png"
    plot_gm_program_bar(stems, out, top_n=10)
    assert out.exists()


def test_analyze_gm_programs_on_fake_pdmx(tmp_path: Path):
    csv_path = tmp_path / "PDMX.csv"
    pd.DataFrame({
        "tracks": ["0", "0-56-56", ""],
        "subset:all_valid": [True, True, True],
    }).to_csv(csv_path, index=False)

    report, stems = analyze_gm_programs(csv_path, subset="all_valid", jobs=1)
    assert report["n_songs"] == 3
    assert report["n_songs_failed"] == 1
    assert report["n_stems"] == 4
    assert (stems["gm_id"] == 0).sum() == 2
    assert (stems["gm_id"] == 56).sum() == 2
