"""Tests for ablation listening result plots."""

from pathlib import Path

import pandas as pd

from experiments.ablation_listening.plot_results import (
    category_leaderboard,
    write_plots,
)
from synthesis.listening.catalog import CONDITION_ORDER


def _fake_df() -> pd.DataFrame:
    rows = []
    for listener in ("a", "b"):
        for category in ("piano", "drums"):
            for i, condition in enumerate(CONDITION_ORDER):
                rows.append({
                    "listener_id": listener,
                    "trial_id": f"stem_{category}_01",
                    "trial_type": "stem",
                    "category": category,
                    "condition_id": condition,
                    "condition_label": condition,
                    "is_reference": False,
                    "content": 40 + i * 5,
                    "realism": 30 + i * 6,
                    "auto_assigned": False,
                    "source_condition": None,
                })
    return pd.DataFrame(rows)


def test_write_plots(tmp_path: Path):
    out = tmp_path / "plots"
    result = write_plots(_fake_df(), out)
    assert result["overview"].is_file()
    assert result["overview_by_category"].is_file()
    assert result["category_winners"].is_file()
    names = {p.name for p in result["by_category"]}
    assert names == {"piano.pdf", "drums.pdf"}


def test_combined_score_penalizes_low_content():
    from experiments.ablation_listening.plot_results import with_combined_score

    df = pd.DataFrame({
        "content": [100.0, 50.0, 0.0],
        "realism": [80.0, 80.0, 100.0],
    })
    out = with_combined_score(df)
    assert list(out["combined"]) == [80.0, 40.0, 0.0]


def test_category_leaderboard_reports_margin():
    board = category_leaderboard(_fake_df())
    assert set(board["category"]) == {"piano", "drums"}
    assert "combined" in board.columns
    assert "margin_vs_2nd" in board.columns
    # Highest index condition wins on content, realism, and combined with the fake data.
    assert (board["winner"] == "ddsp_slakh_realify").all()
