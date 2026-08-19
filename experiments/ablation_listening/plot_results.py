"""Plot ablation listening results: overview + per-category content/realism panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["hatch.linewidth"] = 1.35
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
REALIFY_BASE = {
    "basic_realify": "basic",
    "slakh_realify": "slakh",
    "ddsp_basic_realify": "ddsp_basic",
    "ddsp_slakh_realify": "ddsp_slakh",
}

DEFAULT_PLOTS_DIR = DEFAULT_OUTPUT_DIR / "plots"

CONTENT_DIFF_METRIC = "content_diff"
PLOT_METRICS = ("content", CONTENT_DIFF_METRIC, "realism")
METRIC_TITLES = {
    "content": "Content",
    CONTENT_DIFF_METRIC: "Content Δ",
    "realism": "Realism",
}
MOS_YLIM = (0.0, 118.0)

# X-axis families: synthetic + realified share each tick.
AXIS_GROUPS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("A", ("basic", "basic_realify")),
    ("B", ("slakh", "slakh_realify")),
    ("CA", ("ddsp_basic", "ddsp_basic_realify")),
    ("CB", ("ddsp_slakh", "ddsp_slakh_realify")),
)


def with_content_difference(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``content_diff``: realified minus paired non-realified content.

    Pairing is per listener × trial × family (A2−A1, B2−B1, CA2−CA1, CB2−CB1).
    Non-realified rows are 0 (no generative content change by construction).
    """
    out = df.copy()
    if out.empty or "content" not in out.columns:
        out[CONTENT_DIFF_METRIC] = pd.Series(dtype=float)
        return out
    work = out.copy()
    work["_content"] = pd.to_numeric(work["content"], errors="coerce")
    bases = work.loc[
        ~work["condition_id"].isin(REALIFIED),
        ["listener_id", "trial_id", "condition_id", "_content"],
    ].rename(columns={"condition_id": "base_id", "_content": "base_content"})
    bases = bases.drop_duplicates(
        subset=["listener_id", "trial_id", "base_id"], keep="first",
    )
    work = work.reset_index(drop=True)
    work["base_id"] = work["condition_id"].map(REALIFY_BASE)
    merged = work.merge(
        bases,
        on=["listener_id", "trial_id", "base_id"],
        how="left",
    )
    if len(merged) != len(work):
        raise ValueError("content Δ pairing produced duplicate rows")
    is_realified = merged["condition_id"].isin(REALIFIED)
    merged[CONTENT_DIFF_METRIC] = np.where(
        is_realified,
        merged["_content"] - merged["base_content"],
        np.where(merged["_content"].notna(), 0.0, np.nan),
    )
    out[CONTENT_DIFF_METRIC] = merged[CONTENT_DIFF_METRIC].to_numpy()
    return out


def _stem_df(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["trial_type"] == "stem"].copy() if "trial_type" in df.columns else df.copy()
    return with_content_difference(work)


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


def _family_color(condition_id: str) -> str:
    family = CONDITION_FAMILY.get(condition_id, condition_id)
    return FAMILY_COLOR.get(family, "#888888")


def _bar_style(condition_id: str, *, winner: bool = False) -> dict:
    color = _family_color(condition_id)
    if condition_id in REALIFIED:
        r, g, b, _ = matplotlib.colors.to_rgba(color)
        # Hatch is drawn as a second layer so winner outlines cannot hide it.
        style = {
            "facecolor": (r, g, b, 0.38),
            "edgecolor": color,
            "linewidth": 1.0,
        }
    else:
        style = {
            "facecolor": color,
            "alpha": 0.92,
            "edgecolor": "white",
            "linewidth": 0.6,
        }
    if winner:
        style["edgecolor"] = "#111111"
        style["linewidth"] = 2.8
    return style


def hidden_equivalent_conditions(df: pd.DataFrame) -> set[str]:
    """Conditions that are donor-copies for every rating in ``df``.

    Aggregation expands omitted DDSP samples with ``auto_assigned=True``. If
    every row for a condition is auto-assigned (e.g. drums CA1=A1), hide it.
    Mixed categories (some neural, some fallback) keep the bar.
    """
    if df.empty or "auto_assigned" not in df.columns:
        return set()
    hidden: set[str] = set()
    flags = df["auto_assigned"].fillna(False).astype(bool)
    for condition_id in CONDITION_ORDER:
        mask = df["condition_id"] == condition_id
        if not mask.any():
            continue
        if bool(flags[mask].all()):
            hidden.add(condition_id)
    return hidden


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


def _metric_winners(
    stats: pd.DataFrame,
    *,
    hide: set[str] | None = None,
    candidates: set[str] | None = None,
    tol: float = 1e-9,
) -> set[str]:
    """All visible conditions tied for the highest mean (within ``tol``)."""
    hide = hide or set()
    allowed = set(CONDITION_ORDER) if candidates is None else set(candidates)
    best_mean = float("-inf")
    for condition_id in CONDITION_ORDER:
        if condition_id in hide or condition_id not in allowed:
            continue
        if condition_id not in stats.index or pd.isna(stats.loc[condition_id, "mean"]):
            continue
        mean = float(stats.loc[condition_id, "mean"])
        if mean > best_mean:
            best_mean = mean
    if best_mean == float("-inf"):
        return set()
    winners = set()
    for condition_id in CONDITION_ORDER:
        if condition_id in hide or condition_id not in allowed:
            continue
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
    hide: set[str] | None = None,
    ylim: tuple[float, float] | None = None,
    signed: bool = False,
) -> None:
    """Grouped bars: x ticks are A/B/CA/CB; each tick has synthetic + realified.

    Content Δ only draws realified bars (centered on the tick). Synthetics are
    the Δ baseline, not a plotted condition.
    """
    hide = hide or set()
    winner_pool = REALIFIED if signed else None
    winners = _metric_winners(stats, hide=hide, candidates=winner_pool)
    group_xs = np.arange(len(AXIS_GROUPS), dtype=float)
    bar_width = 0.36
    offsets = (0.0, 0.0) if signed else (-bar_width / 2, bar_width / 2)

    if ylim is None:
        if signed:
            values = []
            for condition_id in CONDITION_ORDER:
                if condition_id in hide or condition_id not in REALIFIED:
                    continue
                if condition_id not in stats.index or pd.isna(stats.loc[condition_id, "mean"]):
                    continue
                mean = float(stats.loc[condition_id, "mean"])
                sem = _stat_value(stats, condition_id, "sem")
                values.extend((mean - sem, mean + sem))
            if values:
                lo, hi = min(values), max(values)
                pad = max(8.0, 0.12 * (hi - lo if hi > lo else 20.0))
                ylim = (min(-12.0, lo - pad), max(12.0, hi + pad))
            else:
                ylim = (-50.0, 20.0)
        else:
            ylim = MOS_YLIM
    y_span = ylim[1] - ylim[0]
    label_pad = 0.015 * y_span
    star_pad = 0.075 * y_span

    for group_x, (_group_label, conditions) in zip(group_xs, AXIS_GROUPS):
        for offset, condition_id in zip(offsets, conditions):
            if condition_id in hide:
                continue
            if signed and condition_id not in REALIFIED:
                continue
            mean = _stat_value(stats, condition_id, "mean")
            sem = _stat_value(stats, condition_id, "sem")
            is_winner = condition_id in winners
            x = group_x + offset
            width = bar_width * 0.95
            ax.bar(
                x,
                mean,
                width=width,
                yerr=sem,
                capsize=3,
                error_kw={"ecolor": "#333333", "elinewidth": 1},
                **_bar_style(condition_id, winner=is_winner),
            )
            if condition_id in REALIFIED:
                ax.bar(
                    x,
                    mean,
                    width=width,
                    facecolor="none",
                    hatch="///",
                    edgecolor=_family_color(condition_id),
                    linewidth=0.0,
                    zorder=3,
                )
            if signed or mean > 0:
                label = f"{mean:+.0f}" if signed else f"{mean:.0f}"
                above = mean >= 0
                text_y = mean + sem + label_pad if above else mean - sem - label_pad
                ax.text(
                    x,
                    text_y,
                    label,
                    ha="center",
                    va="bottom" if above else "top",
                    fontsize=8,
                    fontweight="bold" if is_winner else "normal",
                    color="#111111" if is_winner else "#333333",
                )
                if is_winner:
                    star_y = (
                        mean + sem + star_pad if above else mean - sem - star_pad
                    )
                    ax.text(
                        x,
                        star_y,
                        "★",
                        ha="center",
                        va="bottom" if above else "top",
                        fontsize=11,
                        fontweight="bold",
                        color="#111111",
                    )

    ax.set_xticks(group_xs)
    ax.set_xticklabels([group_label for group_label, _ in AXIS_GROUPS])
    ax.set_xlim(-0.6, len(AXIS_GROUPS) - 0.4)
    ax.set_ylim(*ylim)
    if signed:
        ax.axhline(0.0, color="#888888", linewidth=0.8, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _set_grid_headers(
    axes: np.ndarray,
    *,
    col_labels: tuple[str, ...] | list[str],
    row_labels: tuple[str, ...] | list[str] | None = None,
) -> None:
    """Column titles on the top row; category names as left-column y labels."""
    axes = np.atleast_2d(axes)
    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label)
    if row_labels is not None:
        for row, label in enumerate(row_labels):
            axes[row, 0].set_ylabel(label)


def _legend_handles():
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=FAMILY_COLOR["basic"], edgecolor="white", label="A (basic)"),
        Patch(facecolor=FAMILY_COLOR["slakh"], edgecolor="white", label="B (slakh)"),
        Patch(facecolor=FAMILY_COLOR["ddsp_basic"], edgecolor="white", label="CA (ddsp_basic)"),
        Patch(facecolor=FAMILY_COLOR["ddsp_slakh"], edgecolor="white", label="CB (ddsp_slakh)"),
        Patch(
            facecolor="#d0d0d0",
            hatch="///",
            edgecolor="#444444",
            label="Realified",
        ),
    ]
    return handles


def _draw_metric_panel(
    ax: plt.Axes,
    stats: pd.DataFrame,
    metric: str,
    *,
    hide: set[str] | None = None,
) -> None:
    signed = metric == CONTENT_DIFF_METRIC
    _draw_condition_bars(
        ax,
        stats,
        hide=hide,
        ylim=None if signed else MOS_YLIM,
        signed=signed,
    )


def plot_overview(
    df: pd.DataFrame,
    output_path: Path,
    *,
    hide_equivalences: bool = True,
) -> Path:
    """Overall means across stem trials (content, content Δ, realism)."""
    stem = _stem_df(df)
    hide = hidden_equivalent_conditions(stem) if hide_equivalences else set()
    fig, axes = plt.subplots(
        1, len(PLOT_METRICS), figsize=(5.2 * len(PLOT_METRICS), 4.2), sharey=False,
    )
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, PLOT_METRICS):
        stats = condition_metric_stats(stem, metric)
        _draw_metric_panel(ax, stats, metric, hide=hide)
    _set_grid_headers(
        np.atleast_2d(axes),
        col_labels=[METRIC_TITLES[m] for m in PLOT_METRICS],
    )
    axes[0].legend(handles=_legend_handles(), loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    output_path = Path(output_path)
    _savefig(fig, output_path)
    plt.close(fig)
    return Path(output_path).with_suffix(".pdf")


def plot_category_grid(
    df: pd.DataFrame,
    output_path: Path,
    *,
    hide_equivalences: bool = True,
) -> Path:
    """Big plot: one row per category, content | content Δ | realism panels."""
    stem = _stem_df(df)
    categories = [
        c for c in STEM_TRIAL_CATEGORIES
        if c in set(stem["category"].dropna().astype(str))
    ]
    if not categories:
        categories = sorted(stem["category"].dropna().astype(str).unique())
    n = max(len(categories), 1)
    n_cols = len(PLOT_METRICS)
    fig, axes = plt.subplots(
        n, n_cols, figsize=(5.2 * n_cols, 2.4 * n), sharex=True, sharey=False,
    )
    axes = np.atleast_2d(axes)
    for row, category in enumerate(categories):
        cat_df = stem[stem["category"] == category]
        hide = hidden_equivalent_conditions(cat_df) if hide_equivalences else set()
        for col, metric in enumerate(PLOT_METRICS):
            ax = axes[row, col]
            stats = condition_metric_stats(cat_df, metric)
            _draw_metric_panel(ax, stats, metric, hide=hide)
            if row < n - 1:
                ax.set_xlabel("")
    _set_grid_headers(
        axes,
        col_labels=[METRIC_TITLES[m] for m in PLOT_METRICS],
        row_labels=[c.capitalize() for c in categories],
    )
    axes[0, -1].legend(handles=_legend_handles(), loc="upper right", fontsize=7, frameon=False)
    fig.tight_layout()
    output_path = Path(output_path)
    _savefig(fig, output_path)
    plt.close(fig)
    return Path(output_path).with_suffix(".pdf")


def plot_category_panels(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    hide_equivalences: bool = True,
) -> list[Path]:
    """Write one content | content Δ | realism figure per category."""
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
        hide = hidden_equivalent_conditions(cat_df) if hide_equivalences else set()
        fig, axes = plt.subplots(
            1, len(PLOT_METRICS), figsize=(5.2 * len(PLOT_METRICS), 4.6), sharey=False,
        )
        axes = np.atleast_1d(axes)
        for ax, metric in zip(axes, PLOT_METRICS):
            stats = condition_metric_stats(cat_df, metric)
            _draw_metric_panel(ax, stats, metric, hide=hide)
        _set_grid_headers(
            np.atleast_2d(axes),
            col_labels=[METRIC_TITLES[m] for m in PLOT_METRICS],
            row_labels=[category.capitalize()],
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
        fig.tight_layout(rect=(0, 0.10, 1, 0.92))
        path = output_dir / f"{category}.pdf"
        _savefig(fig, path)
        plt.close(fig)
        written.append(path)
    return written


def category_leaderboard(
    df: pd.DataFrame,
    *,
    hide_equivalences: bool = True,
) -> pd.DataFrame:
    """Per-category winner(s) ranked by realism."""
    stem = _stem_df(df)
    rank_key = "realism"
    rows = []
    for category in sorted(stem["category"].dropna().astype(str).unique()):
        cat_df = stem[stem["category"] == category]
        hide = hidden_equivalent_conditions(cat_df) if hide_equivalences else set()
        realism = condition_metric_stats(cat_df, "realism")
        content = condition_metric_stats(cat_df, "content")
        content_diff = condition_metric_stats(cat_df, CONTENT_DIFF_METRIC)
        merged = (
            realism[["mean"]].rename(columns={"mean": "realism"})
            .join(content[["mean"]].rename(columns={"mean": "content"}), how="outer")
            .join(
                content_diff[["mean"]].rename(columns={"mean": CONTENT_DIFF_METRIC}),
                how="outer",
            )
        )
        if hide:
            merged = merged.drop(index=[c for c in hide if c in merged.index], errors="ignore")
        sort_cols = [rank_key, "content"]
        sort_cols = list(dict.fromkeys(c for c in sort_cols if c in merged.columns))
        merged = merged.dropna(subset=[rank_key]).sort_values(sort_cols, ascending=False)
        if merged.empty:
            continue
        best = float(merged.iloc[0][rank_key])
        winners = [
            idx for idx, row in merged.iterrows()
            if abs(float(row[rank_key]) - best) <= 1e-9
        ]
        runner_up = None
        for idx, row in merged.iterrows():
            if idx not in winners:
                runner_up = float(row[rank_key])
                break
        margin = best - runner_up if runner_up is not None else None
        donor_margin = None
        ddsp_winners = [w for w in winners if str(w).startswith("ddsp_")]
        if ddsp_winners:
            donor = str(ddsp_winners[0]).removeprefix("ddsp_")
            if donor in merged.index:
                donor_margin = best - float(merged.loc[donor, rank_key])
        first = merged.loc[winners[0]]
        row = {
            "category": category,
            "winner": ",".join(winners),
            "winner_label": ",".join(_short_label(w) for w in winners),
            "content": round(float(first["content"]), 2)
            if pd.notna(first["content"]) else None,
            "content_diff": round(float(first[CONTENT_DIFF_METRIC]), 2)
            if pd.notna(first[CONTENT_DIFF_METRIC]) else None,
            "realism": round(float(first["realism"]), 2)
            if pd.notna(first["realism"]) else None,
            "margin_vs_2nd": round(margin, 2) if margin is not None else None,
            "ddsp_margin_vs_donor": round(donor_margin, 2) if donor_margin is not None else None,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def write_plots(
    df: pd.DataFrame,
    plots_dir: Path,
    *,
    hide_equivalences: bool = True,
) -> dict:
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    by_cat = plots_dir / "by_category"
    kw = {"hide_equivalences": hide_equivalences}
    overview = plot_overview(df, plots_dir / "overview.pdf", **kw)
    grid = plot_category_grid(df, plots_dir / "overview_by_category.pdf", **kw)
    panels = plot_category_panels(df, by_cat, **kw)
    board = category_leaderboard(df, **kw)
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
    parser.add_argument(
        "--show-equivalences",
        action="store_true",
        help=(
            "Show bars whose scores were auto-copied from a donor (e.g. drums "
            "CA1/CA2/CB1/CB2 when they equal A/B). Hidden by default."
        ),
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

    result = write_plots(
        df,
        opts.plots_dir,
        hide_equivalences=not opts.show_equivalences,
    )
    print(f"Wrote {result['overview']}")
    print(f"Wrote {result['overview_by_category']}")
    print(f"Wrote {len(result['by_category'])} category panels under {opts.plots_dir / 'by_category'}")
    print(f"Wrote {result['category_winners']}")
    board = result["leaderboard"]
    if not board.empty:
        print("\nPer-category winners on realism:")
        print(board.to_string(index=False))


if __name__ == "__main__":
    main()
