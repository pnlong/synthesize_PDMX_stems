"""Tests for PDMX subset filtering."""

import pandas as pd

from analysis.pdmx_subset import filter_pdmx_subset, subset_output_dir


def test_filter_pdmx_subset_all_valid():
    df = pd.DataFrame({
        "subset:all_valid": [True, False, True],
        "subset:rated_deduplicated": [True, True, False],
        "tracks": ["0", "0-1", "0-2"],
    })
    out = filter_pdmx_subset(df, "all_valid")
    assert len(out) == 2


def test_filter_pdmx_subset_rated_deduplicated():
    df = pd.DataFrame({
        "subset:all_valid": [True, True, True],
        "subset:rated_deduplicated": [True, False, True],
        "tracks": ["0", "0-1", "0-2"],
    })
    out = filter_pdmx_subset(df, "rated_deduplicated")
    assert len(out) == 2


def test_subset_output_dir(tmp_path):
    assert subset_output_dir(tmp_path / "instruments", "all_valid") == tmp_path / "instruments" / "all_valid"
