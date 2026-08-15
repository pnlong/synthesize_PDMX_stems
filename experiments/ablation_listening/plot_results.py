"""Plot ablation listening results: overview + per-category content/realism panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.ablation_listening.aggregate import aggregate_responses
from experiments.ablation_listening.conditions import STEM_TRIAL_CATEGORIES
from experiments.ablation_listening.paths import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESPONSES_DIR,
)
from synthesis.listening.catalog import CONDITION_LABELS, CONDITION_ORDER

# Distinct family colors (synthetic solid; realified lighter + hatch).
FAMILY_COLOR = {
    "basic": "#4C78A8",
    "slakh": "#54A24B",
    "ddsp_basic": "#F58518",
    "ddsp_slakh": "#E45756",
}
CONDITION_FAMILY = {
    "basic": "basic",
    "basic_realify": "basic",
    "slakh": "slakh",
    "slakh_realify": "slakh",
    "ddsp_basic": "ddsp_basic",
    "ddsp_basic_realify": "ddsp_basic",
    "ddsp_slakh": "ddsp_slakh",
    "ddsp_slakh_realify": "ddsp_slakh",
}
REALIFIED = {
    "basic_realify",
    "slakh_realify",
    "ddsp_basic_realify",
    "ddsp_slakh_realify",
}

DEFAULT_PLOTS_DIR = DEFAULT_OUTPUT_DIR / "plots"

# content · realism/100-scaled product rewards high content adherence.
COMBINED_METRIC = "combined"
PLOT_METRICS = ("content", "realism", COMBINED_METRIC)
METRIC_TITLES = {
    "content": "Content",
    "realism": "Realism",
    COMBINED_METRIC: r"Combined $=(\mathrm{content}/100)\times\mathrm{realism}$",
}

# X-axis families: synthetic + realified share each tick.
AXIS_GROUPS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("A", ("basic", "basic_realify")),
    ("B", ("slakh", "slakh_realify")),
    ("CA", ("ddsp_basic", "ddsp_basic_realify")),
    ("CB", ("ddsp_slakh", "ddsp_slakh_realify")),
)


def with_combined_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``combined = (content/100) * realism`` (NaN if either score missing)."""
    out = df.copy()
    content = pd.to_numeric(out.get("content"), errors="coerce")
    realism = pd.to_numeric(out.get("realism"), errors="coerce")
    out[COMBINED_METRIC] = (content / 100.0) * realism
    return out


def _stem_df(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["trial_type"] == "stem"].copy() if "trial_type" in df.columns else df.copy()
    return with_combined_score(work)


def _savefig(fig: plt.Figure, output_path: Path) -> Path:
    """Save PDF with transparent background (paper-friendly)."""
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".pdf":
        output_path = output_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.patch.set_alpha(0.0)
    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    return output_path


def _short_label(condition_id: str) -> str:
    return CONDITION_LABELS.get(condition_id, condition_id)


def _bar_style(condition_id: str, *, winner: bool = False) -> dict:
    family = CONDITION_FAMILY.get(condition_id, condition_id)
    color = FAMILY_COLOR.get(family, "#888888")
    if condition_id in REALIFIED:
        style = {
            "color": color,
            "alpha": 0.45,
            "hatch": "//",
            "edgecolor": color,
            "linewidth": 1.0,
        }
    else:
        style = {
            "color": color,
            "alpha": 0.92,
            "hatch": None,
            "edgecolor": "white",
            "linewidth": 0.6,
        }
    if winner:
        style["edgecolor"] = "#111111"
        style["linewidth"] = 2.8
        style["alpha"] = min(1.0, float(style["alpha"]) + 0.08)
    return style


def condition_metric_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean ± SEM per condition.

    With multiple listeners, average within listener first, then across listeners.
    """
    work = df.dropna(subset=[metric]).copy()
    if work.empty:
        return pd.DataFrame(columns=["mean", "sem", "n"])

    if work["listener_id"].nunique(dropna=True) > 1:
        per_listener = (
            work.groupby(["listener_id", "condition_id"], dropna=False)[metric]
            .mean()
            .reset_index()
        )
        grouped = per_listener.groupby("condition_id")[metric]
    else:
        grouped = work.groupby("condition_id")[metric]

    out = grouped.agg(mean="mean", sem="sem", n="count")
    out["sem"] = out["sem"].fillna(0.0)
    return out.reindex(CONDITION_ORDER)


def _metric_winners(stats: pd.DataFrame, *, tol: float = 1e-9) -> set[str]:
    """All conditions tied for the highest mean (within ``tol``)."""
    best_mean = float("-inf")
    for condition_id in CONDITION_ORDER:
        if condition_id not in stats.index or pd.isna(stats.loc[condition_id, "mean"]):
            continue
        mean = float(stats.loc[condition_id, "mean"])
        if mean > best_mean:
            best_mean = mean
    if best_mean == float("-inf"):
        return set()
    winners = set()
    for condition_id in CONDITION_ORDER:
        if condition_id not in stats.index or pd.isna(stats.loc[condition_id, "mean"]):
            continue
        if abs(float(stats.loc[condition_id, "mean"]) - best_mean) <= tol:
            winners.add(condition_id)
    return winners


def _stat_value(stats: pd.DataFrame, condition_id: str, field: str) -> float:
    if condition_id not in stats.index or pd.isna(stats.loc[condition_id, field]):
        return 0.0
    return float(stats.loc[condition_id, field])


def _draw_condition_bars(
    ax: plt.Axes,
    stats: pd.DataFrame,
    *,
    title: str,
    ylabel: str = "Score (0–100)",
) -> None:
    """Grouped bars: x ticks are A/B/CA/CB; each tick has synthetic + realified."""
    winners = _metric_winners(stats)
    group_xs = np.arange(len(AXIS_GROUPS), dtype=float)
    bar_width = 0.36
    offsets = (-bar_width / 2, bar_width / 2)  # synthetic, realified

    for group_x, (_group_label, conditions) in zip(group_xs, AXIS_GROUPS):
        for offset, condition_id in zip(offsets, conditions):
            mean = _stat_value(stats, condition_id, "mean")
            sem = _stat_value(stats, condition_id, "sem")
            is_winner = condition_id in winners
            x = group_x + offset
            ax.bar(
                x,
                mean,
                width=bar_width * 0.95,
                yerr=sem,
                capsize=3,
                error_kw={"ecolor": "#333333", "elinewidth": 1},
                **_bar_style(condition_id, winner=is_winner),
            )
            if mean > 0:
                ax.text(
                    x,
                    mean + sem + 1.5,
                    f"{mean:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold" if is_winner else "normal",
                    color="#111111" if is_winner else "#333333",
                )
                if is_winner:
                    ax.text(
                        x,
                        mean + sem + 9.0,
                        "★",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                        color="#111111",
                    )

    ax.set_xticks(group_xs)
    ax.set_xticklabels([group_label for group_label, _ in AXIS_GROUPS])
    ax.set_xlim(-0.6, len(AXIS_GROUPS) - 0.4)
    ax.set_ylim(0, 118)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(50, color="#bbbbbb", linewidth=0.8, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _legend_handles():
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=FAMILY_COLOR["basic"], edgecolor="white", label="A (basic)"),
        Patch(facecolor=FAMILY_COLOR["slakh"], edgecolor="white", label="B (slakh)"),
        Patch(facecolor=FAMILY_COLOR["ddsp_basic"], edgecolor="white", label="CA (ddsp_basic)"),
        Patch(facecolor=FAMILY_COLOR["ddsp_slakh"], edgecolor="white", label="CB (ddsp_slakh)"),
        Patch(
            facecolor="#888888",
            alpha=0.45,
            hatch="//",
            edgecolor="#888888",
            label="realified (right bar)",
        ),
        Patch(
            facecolor="none",
            edgecolor="#111111",
            linewidth=2.5,
            label="★ winner",
        ),
    ]
    return handles


def plot_overview(df: pd.DataFrame, output_path: Path) -> Path:
    """Three-panel overall means across all stem trials (content, realism, combined)."""
    stem = _stem_df(df)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, metric in zip(axes, PLOT_METRICS):
        stats = condition_metric_stats(stem, metric)
        _draw_condition_bars(ax, stats, title=METRIC_TITLES[metric])
    axes[0].legend(handles=_legend_handles(), loc="upper right", fontsize=8, frameon=False)
    n_listeners = int(stem["listener_id"].nunique(dropna=True)) if not stem.empty else 0
    fig.suptitle(
        f"Ablation listening — overall means (±SEM; n_listeners={n_listeners})",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    output_path = Path(output_path)
    _savefig(fig, output_path)
    plt.close(fig)
    return Path(output_path).with_suffix(".pdf")


def plot_category_grid(df: pd.DataFrame, output_path: Path) -> Path:
    """Big plot: one row per category, content | realism | combined panels."""
    stem = _stem_df(df)
    categories = [
        c for c in STEM_TRIAL_CATEGORIES
        if c in set(stem["category"].dropna().astype(str))
    ]
    if not categories:
        categories = sorted(stem["category"].dropna().astype(str).unique())
    n = max(len(categories), 1)
    fig, axes = plt.subplots(n, 3, figsize=(15, 2.4 * n), sharex=True, sharey=True)
    if n == 1:
        axes = np.array([axes])
    for row, category in enumerate(categories):
        cat_df = stem[stem["category"] == category]
        for col, metric in enumerate(PLOT_METRICS):
            ax = axes[row, col]
            stats = condition_metric_stats(cat_df, metric)
            short = {
                "content": "content",
                "realism": "realism",
                COMBINED_METRIC: "combined",
            }[metric]
            title = f"{category.capitalize()} — {short}"
            _draw_condition_bars(ax, stats, title=title)
            if row < n - 1:
                ax.set_xlabel("")
    axes[0, 2].legend(handles=_legend_handles(), loc="upper right", fontsize=7, frameon=False)
    fig.suptitle(
        r"Ablation listening — by category "
        r"(combined $=(\mathrm{content}/100)\times\mathrm{realism}$)",
        fontsize=12,
        y=1.005,
    )
    fig.tight_layout()
    output_path = Path(output_path)
    _savefig(fig, output_path)
    plt.close(fig)
    return Path(output_path).with_suffix(".pdf")


def plot_category_panels(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write one three-panel (content|realism|combined) figure per category."""
    stem = _stem_df(df)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = [
        c for c in STEM_TRIAL_CATEGORIES
        if c in set(stem["category"].dropna().astype(str))
    ]
    if not categories:
        categories = sorted(stem["category"].dropna().astype(str).unique())

    written: list[Path] = []
    for category in categories:
        cat_df = stem[stem["category"] == category]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
        for ax, metric in zip(axes, PLOT_METRICS):
            stats = condition_metric_stats(cat_df, metric)
            _draw_condition_bars(ax, stats, title=METRIC_TITLES[metric])
        n_ratings = len(cat_df)
        n_listeners = int(cat_df["listener_id"].nunique(dropna=True))
        fig.suptitle(
            f"{category.capitalize()} (n_ratings={n_ratings}, n_listeners={n_listeners})",
            fontsize=12,
        )
        handles = _legend_handles()
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(handles),
            fontsize=8,
            frameon=False,
            columnspacing=1.2,
            handlelength=1.6,
        )
        fig.tight_layout(rect=(0, 0.10, 1, 0.95))
        path = output_dir / f"{category}.pdf"
        _savefig(fig, path)
        plt.close(fig)
        written.append(path)
    return written


def category_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Per-category winner on combined=(content/100)*realism."""
    stem = _stem_df(df)
    rows = []
    for category in sorted(stem["category"].dropna().astype(str).unique()):
        cat_df = stem[stem["category"] == category]
        combined = condition_metric_stats(cat_df, COMBINED_METRIC)
        realism = condition_metric_stats(cat_df, "realism")
        content = condition_metric_stats(cat_df, "content")
        merged = (
            combined[["mean"]].rename(columns={"mean": "combined"})
            .join(realism[["mean"]].rename(columns={"mean": "realism"}), how="outer")
            .join(content[["mean"]].rename(columns={"mean": "content"}), how="outer")
        )
        merged = merged.dropna(subset=["combined"]).sort_values(
            ["combined", "realism", "content"], ascending=False,
        )
        if merged.empty:
            continue
        winner = merged.index[0]
        winner_combined = float(merged.iloc[0]["combined"])
        second_combined = (
            float(merged.iloc[1]["combined"]) if len(merged) > 1 else float("nan")
        )
        margin = (
            winner_combined - second_combined if pd.notna(second_combined) else float("nan")
        )
        donor_margin = None
        if str(winner).startswith("ddsp_"):
            donor = str(winner).removeprefix("ddsp_")
            if donor in merged.index:
                donor_margin = winner_combined - float(merged.loc[donor, "combined"])
        rows.append({
            "category": category,
            "winner": winner,
            "winner_label": _short_label(winner),
            "combined": round(winner_combined, 2),
            "content": round(float(merged.iloc[0]["content"]), 2)
            if pd.notna(merged.iloc[0]["content"]) else None,
            "realism": round(float(merged.iloc[0]["realism"]), 2)
            if pd.notna(merged.iloc[0]["realism"]) else None,
            "margin_vs_2nd": round(margin, 2) if pd.notna(margin) else None,
            "ddsp_margin_vs_donor": round(donor_margin, 2) if donor_margin is not None else None,
        })
    return pd.DataFrame(rows)


def write_plots(
    df: pd.DataFrame,
    plots_dir: Path,
) -> dict:
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    by_cat = plots_dir / "by_category"
    overview = plot_overview(df, plots_dir / "overview.pdf")
    grid = plot_category_grid(df, plots_dir / "overview_by_category.pdf")
    panels = plot_category_panels(df, by_cat)
    board = category_leaderboard(df)
    board_path = plots_dir / "category_winners.csv"
    board.to_csv(board_path, index=False)
    return {
        "overview": overview,
        "overview_by_category": grid,
        "by_category": panels,
        "category_winners": board_path,
        "leaderboard": board,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=(
            "Plot ablation listening results: overall bars + per-category "
            "content/realism panels."
        ),
    )
    parser.add_argument(
        "--responses",
        nargs="+",
        type=Path,
        help="Completed response JSON file(s) and/or directories.",
    )
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=None,
        help=f"Response directory (default: {DEFAULT_RESPONSES_DIR}).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help=f"Output directory for plots (default: {DEFAULT_PLOTS_DIR}).",
    )
    return parser.parse_args(args)


def main(args=None) -> None:
    opts = parse_args(args)
    sources: list[Path] = list(opts.responses or [])
    if opts.responses_dir is not None:
        sources.append(opts.responses_dir)
    elif not sources:
        sources.append(DEFAULT_RESPONSES_DIR)

    df, summary = aggregate_responses(sources, manifest_path=opts.manifest)
    if df.empty or summary.get("error"):
        raise SystemExit(summary.get("error") or "no ratings to plot")

    result = write_plots(df, opts.plots_dir)
    print(f"Wrote {result['overview']}")
    print(f"Wrote {result['overview_by_category']}")
    print(f"Wrote {len(result['by_category'])} category panels under {opts.plots_dir / 'by_category'}")
    print(f"Wrote {result['category_winners']}")
    board = result["leaderboard"]
    if not board.empty:
        print("\nPer-category winners on combined=(content/100)*realism:")
        print(board.to_string(index=False))


if __name__ == "__main__":
    main()
