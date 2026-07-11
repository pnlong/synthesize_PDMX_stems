"""Tests for CSV table helpers."""

import pandas as pd

from shared.config import STEMS_TABLE_COLUMNS
from shared.csv_tables import append_rows_deduped, sanitize_track_name


def test_sanitize_track_name_strips_null_bytes():
    assert sanitize_track_name("Piano\x00") == "Piano"
    assert sanitize_track_name("  a\x00b  ") == "ab"
    assert sanitize_track_name("") is None
    assert sanitize_track_name(None) is None


def test_append_rows_deduped_replaces_path(tmp_path):
    csv_path = tmp_path / "stems.csv"
    path = "/songs/a"
    rows_a = [
        {"path": path, "track": 0, "program": 0, "is_drum": False, "name": sanitize_track_name("Piano\x00"), "has_lyrics": False},
        {"path": path, "track": 1, "program": 1, "is_drum": False, "name": "Bass", "has_lyrics": False},
    ]
    append_rows_deduped(str(csv_path), STEMS_TABLE_COLUMNS, rows_a)
    append_rows_deduped(
        str(csv_path),
        STEMS_TABLE_COLUMNS,
        [{**rows_a[0], "name": "Piano"}, rows_a[1]],
    )
    stems = pd.read_csv(csv_path)
    assert len(stems) == 2
    assert stems[stems["track"] == 0].iloc[0]["name"] == "Piano"
