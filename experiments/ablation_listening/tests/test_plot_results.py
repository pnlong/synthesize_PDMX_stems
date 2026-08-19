"""Tests for ablation listening result plots."""

from pathlib import Path

import pandas as pd
import pytest

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


def test_content_difference_pairs_realify_to_base():
    from experiments.ablation_listening.plot_results import with_content_difference

    df = pd.DataFrame({
        "listener_id": ["a", "a", "a", "a"],
        "trial_id": ["stem_piano_01"] * 4,
        "condition_id": ["basic", "basic_realify", "slakh", "slakh_realify"],
        "content": [79.0, 56.0, 70.0, 70.0],
        "realism": [80.0, 70.0, 75.0, 72.0],
        "trial_type": ["stem"] * 4,
        "category": ["piano"] * 4,
    })
    out = with_content_difference(df)
    by_cond = dict(zip(out["condition_id"], out["content_diff"]))
    assert by_cond["basic"] == 0.0
    assert by_cond["slakh"] == 0.0
    assert by_cond["basic_realify"] == pytest.approx(-23.0)
    assert by_cond["slakh_realify"] == pytest.approx(0.0)


def test_content_delta_panel_skips_synthetic_bars():
    from experiments.ablation_listening.plot_results import (
        CONTENT_DIFF_METRIC,
        _draw_metric_panel,
        condition_metric_stats,
        with_content_difference,
    )
    import matplotlib.pyplot as plt

    stats = condition_metric_stats(
        with_content_difference(_fake_df()),
        CONTENT_DIFF_METRIC,
    )
    fig, ax = plt.subplots()
    _draw_metric_panel(ax, stats, CONTENT_DIFF_METRIC)
    # One realified bar per family (A/B/CA/CB); synthetics are not drawn.
    assert len(ax.patches) == 8  # 4 bars + 4 hatch overlays
    plt.close(fig)


def test_category_leaderboard_reports_margin():
    board = category_leaderboard(_fake_df())
    assert set(board["category"]) == {"piano", "drums"}
    assert "combined" not in board.columns
    assert "content_diff" in board.columns
    assert "margin_vs_2nd" in board.columns
    # Highest index condition wins on realism (and content) with the fake data.
    assert (board["winner"] == "ddsp_slakh_realify").all()


def test_hidden_equivalent_conditions_only_when_all_auto_assigned():
    from experiments.ablation_listening.plot_results import hidden_equivalent_conditions
    from experiments.ablation_listening.equivalence import DONOR_EQUIVALENCE_PAIRS

    rows = []
    for condition in CONDITION_ORDER:
        auto = condition in DONOR_EQUIVALENCE_PAIRS
        rows.append({
            "category": "drums",
            "condition_id": condition,
            "auto_assigned": auto,
        })
    drums = pd.DataFrame(rows)
    assert hidden_equivalent_conditions(drums) == set(DONOR_EQUIVALENCE_PAIRS)

    mixed = drums.copy()
    mixed.loc[mixed["condition_id"] == "ddsp_basic", "auto_assigned"] = False
    assert "ddsp_basic" not in hidden_equivalent_conditions(mixed)
    assert "ddsp_slakh" in hidden_equivalent_conditions(mixed)


def test_leaderboard_hides_equivalent_duplicates():
    from experiments.ablation_listening.equivalence import DONOR_EQUIVALENCE_PAIRS

    rows = []
    for condition in CONDITION_ORDER:
        is_ddsp = condition in DONOR_EQUIVALENCE_PAIRS
        rows.append({
            "listener_id": "a",
            "trial_id": "stem_drums_01",
            "trial_type": "stem",
            "category": "drums",
            "condition_id": condition,
            "condition_label": condition,
            "is_reference": False,
            "content": 90.0 if condition in ("basic", "ddsp_basic") else 40.0,
            "realism": 80.0 if condition in ("basic", "ddsp_basic") else 30.0,
            "auto_assigned": is_ddsp,
            "source_condition": DONOR_EQUIVALENCE_PAIRS.get(condition),
        })
    df = pd.DataFrame(rows)
    shown = category_leaderboard(df, hide_equivalences=False)
    hidden = category_leaderboard(df, hide_equivalences=True)
    default = category_leaderboard(df)
    assert "ddsp_basic" in shown.iloc[0]["winner"]
    assert hidden.iloc[0]["winner"] == "basic"
    assert default.iloc[0]["winner"] == "basic"
