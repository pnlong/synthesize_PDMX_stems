"""Plot song-length distributions for SA3 model selection."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from shared.config import SA3_MEDIUM_MAX_DURATION, SA3_SMALL_MUSIC_MAX_DURATION


def _add_sa3_limits(ax: plt.Axes):
    ax.axvline(
        SA3_SMALL_MUSIC_MAX_DURATION,
        color="C1",
        linestyle="--",
        linewidth=1.5,
        label=f"small-music ({SA3_SMALL_MUSIC_MAX_DURATION}s)",
    )
    ax.axvline(
        SA3_MEDIUM_MAX_DURATION,
        color="C2",
        linestyle="--",
        linewidth=1.5,
        label=f"medium ({SA3_MEDIUM_MAX_DURATION}s)",
    )


def plot_histogram(
    durations: pd.Series,
    output_path: str | Path,
    *,
    max_seconds: float = 600,
    bins: int = 60,
):
    """Histogram of song lengths with SA3 model duration limits marked."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    clipped = durations.clip(upper=max_seconds)
    ax.hist(clipped, bins=bins, color="C0", alpha=0.85, edgecolor="white")
    _add_sa3_limits(ax)
    ax.set_xlabel("Song length (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("PDMX song length distribution")
    ax.set_xlim(0, max_seconds)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_percentiles(
    durations: pd.Series,
    output_path: str | Path,
    *,
    max_seconds: float = 600,
):
    """Empirical CDF (percentile curve) with SA3 model duration limits marked."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_durations = durations.sort_values().to_numpy()
    cumulative_pct = (pd.Series(range(1, len(sorted_durations) + 1)) / len(sorted_durations) * 100).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sorted_durations, cumulative_pct, color="C0", linewidth=2)
    _add_sa3_limits(ax)
    ax.set_xlabel("Song length (seconds)")
    ax.set_ylabel("Percentile")
    ax.set_title("PDMX song length percentiles")
    ax.set_xlim(0, max_seconds)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _gm_count_series(stems: pd.DataFrame) -> pd.Series:
    if stems is None or stems.empty:
        return pd.Series(dtype=int)
    return stems["gm_id"].value_counts()


def _select_gm_ids_for_plot(
    counts: pd.Series,
    *,
    top_n: int,
    drum_id: int,
) -> list:
    """Top-N gm_ids by count, keeping drums visible; optional Other via -1 elsewhere."""
    if counts.empty:
        return []
    if top_n <= 0 or len(counts) <= top_n:
        return list(counts.index)
    head_index = list(counts.head(top_n).index)
    if drum_id in counts.index and drum_id not in head_index:
        head_index = list(counts.head(top_n - 1).index)
        if drum_id not in head_index:
            head_index.append(drum_id)
    return head_index


def _annotate_pct_inside_bars(
    ax,
    bars,
    values,
    total: int,
    *,
    min_pct: float = 1.0,
    fontsize: int = 7,
) -> None:
    """Write ``12%`` inside bars (near the tip) when share of ``total`` exceeds ``min_pct``."""
    if total <= 0:
        return
    xmax = max(ax.get_xlim()[1], max(values, default=0), 1)
    inset = xmax * 0.012
    for bar, val in zip(bars, values):
        pct = 100.0 * float(val) / float(total)
        if pct <= min_pct or val <= 0:
            continue
        y = bar.get_y() + bar.get_height() / 2.0
        x = float(bar.get_width()) - inset
        if x <= 0:
            continue
        ax.text(
            x,
            y,
            f"{round(pct)}%",
            va="center",
            ha="right",
            fontsize=fontsize,
            color="white",
            fontweight="bold",
            clip_on=True,
        )


def plot_gm_program_bar(
    stems: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int = 40,
    title: str = "PDMX General MIDI program usage (drums = channel 10)",
):
    """Horizontal bar chart of GM program id counts (top N + Other)."""
    from analysis.gm_programs import DRUM_GM_ID, gm_id_label

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = _gm_count_series(stems)
    total = int(counts.sum())
    head_index = _select_gm_ids_for_plot(counts, top_n=top_n, drum_id=DRUM_GM_ID)
    if head_index:
        head = counts.loc[head_index]
        other = int(counts.drop(labels=head_index, errors="ignore").sum())
        plot_counts = head.copy()
        if other:
            plot_counts.loc[-1] = other
    else:
        plot_counts = counts

    plot_counts = plot_counts.sort_values(ascending=True)
    values = [int(v) for v in plot_counts.values]

    labels = [
        gm_id_label(int(v)) if int(v) >= 0 else "Other" for v in plot_counts.index
    ]

    fig_height = max(6, 0.28 * len(plot_counts))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    bars = ax.barh(labels, values, color="C0", alpha=0.9)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
    ax.set_xlim(0, max(values, default=0) * 1.18 + 1)
    _annotate_pct_inside_bars(ax, bars, values, total, fontsize=8)
    ax.set_xlabel("Stem count (non-empty MIDI tracks)")
    ax.set_ylabel("GM program id")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_gm_program_compare(
    stems_original: pd.DataFrame,
    stems_corrected: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int = 40,
    title: str = "GM program usage: original vs corrected",
):
    """Two-panel horizontal bar chart with a shared GM-id y-axis.

    Left = raw MIDI programs; right = register ``program_corrected``.
    Y-order follows corrected counts (ascending). Each panel has its own x-scale.
    """
    from analysis.gm_programs import DRUM_GM_ID, gm_id_label

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    left_counts = _gm_count_series(stems_original)
    right_counts = _gm_count_series(stems_corrected)
    left_total = int(left_counts.sum())
    right_total = int(right_counts.sum())

    left_ids = set(_select_gm_ids_for_plot(left_counts, top_n=top_n, drum_id=DRUM_GM_ID))
    right_ids = set(_select_gm_ids_for_plot(right_counts, top_n=top_n, drum_id=DRUM_GM_ID))
    shared_ids = left_ids | right_ids
    if not shared_ids:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_title(title)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    # Sort by corrected count so the right panel defines the row order.
    ordered = sorted(shared_ids, key=lambda g: int(right_counts.get(g, 0)))
    labels = [gm_id_label(int(g)) for g in ordered]
    left_vals = [int(left_counts.get(g, 0)) for g in ordered]
    right_vals = [int(right_counts.get(g, 0)) for g in ordered]

    y = range(len(ordered))
    fig_height = max(6, 0.28 * len(ordered))
    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(14, fig_height),
        sharey=True,
        constrained_layout=True,
    )

    bars_l = ax_l.barh(y, left_vals, color="C0", alpha=0.9)
    bars_r = ax_r.barh(y, right_vals, color="C0", alpha=0.9)
    ax_l.bar_label(bars_l, fmt="%d", padding=3, fontsize=7)
    ax_r.bar_label(bars_r, fmt="%d", padding=3, fontsize=7)

    ax_l.set_yticks(list(y))
    ax_l.set_yticklabels(labels)
    ax_l.set_xlabel("Stem count")
    ax_r.set_xlabel("Stem count")
    ax_l.set_title("Original (MIDI program_change)")
    ax_r.set_title("Corrected (GM register)")
    ax_l.set_ylabel("GM program id")

    # Independent x-scales (only y is shared).
    ax_l.set_xlim(0, max(left_vals, default=0) * 1.18 + 1)
    ax_r.set_xlim(0, max(right_vals, default=0) * 1.18 + 1)
    _annotate_pct_inside_bars(ax_l, bars_l, left_vals, left_total)
    _annotate_pct_inside_bars(ax_r, bars_r, right_vals, right_total)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_track_name_bar(
    stems: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int = 40,
):
    """Horizontal bar chart of track name counts (top N + Other)."""
    from analysis.track_names import UNNAMED_TRACK

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if stems.empty:
        counts = pd.Series(dtype=int)
    else:
        counts = stems["track_name"].value_counts()

    if top_n > 0 and len(counts) > top_n:
        head = counts.head(top_n)
        other = int(counts.iloc[top_n:].sum())
        plot_counts = head.copy()
        if other:
            plot_counts.loc["__other__"] = other
    else:
        plot_counts = counts

    plot_counts = plot_counts.sort_values(ascending=True)
    labels = [
        name if name != "__other__" else "Other"
        for name in plot_counts.index
    ]

    fig_height = max(6, 0.28 * len(plot_counts))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    bars = ax.barh(labels, plot_counts.values, color="C0", alpha=0.9)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
    ax.set_xlabel("Track count (non-empty MIDI tracks)")
    ax.set_ylabel("Track name")
    ax.set_title("PDMX MIDI track name usage")
    if UNNAMED_TRACK in labels:
        idx = labels.index(UNNAMED_TRACK)
        bars[idx].set_color("C3")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
