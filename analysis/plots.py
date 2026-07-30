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


def plot_gm_program_bar(
    stems: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int = 40,
):
    """Horizontal bar chart of GM program id counts (top N + Other)."""
    from analysis.gm_programs import gm_id_label

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if stems.empty:
        counts = pd.Series(dtype=int)
    else:
        counts = stems["gm_id"].value_counts()

    if top_n > 0 and len(counts) > top_n:
        head = counts.head(top_n)
        other = int(counts.iloc[top_n:].sum())
        plot_counts = head.copy()
        if other:
            plot_counts.loc[-1] = other
    else:
        plot_counts = counts

    plot_counts = plot_counts.sort_values(ascending=True)

    fig_height = max(6, 0.28 * len(plot_counts))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    bars = ax.barh(
        [gm_id_label(int(v)) if v >= 0 else "Other" for v in plot_counts.index],
        plot_counts.values,
        color="C0",
        alpha=0.9,
    )
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
    ax.set_xlabel("Stem count (non-empty MIDI tracks)")
    ax.set_ylabel("GM program id")
    ax.set_title("PDMX General MIDI program usage")
    fig.tight_layout()
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
