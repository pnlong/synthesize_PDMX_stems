"""Analyze PDMX song lengths from metadata and recommend an SA3 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis.pdmx_lengths import load_pdmx_song_lengths, percentile_table, sa3_limit_percentiles
from analysis.plots import plot_histogram, plot_percentiles
from analysis.report import recommend_model
from shared.config import OUTPUT_DIR, PDMX_FILEPATH
from shared.repo_symlinks import link_analysis_in_repo
from synthesis.paths import song_lengths_dir


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="Analyze PDMX song_length.seconds and recommend an SA3 model.",
    )
    parser.add_argument("-df", "--dataset_filepath", default=PDMX_FILEPATH, type=str)
    parser.add_argument("-o", "--output_dir", default=song_lengths_dir(OUTPUT_DIR), type=str)
    parser.add_argument(
        "--max-seconds",
        default=600,
        type=float,
        help="X-axis limit for plots (songs longer than this appear in the rightmost bin / edge).",
    )
    return parser.parse_args(args=args, namespace=namespace)


def build_report(durations: pd.Series) -> dict:
    return {
        "n_songs": int(len(durations)),
        "summary": recommend_model(durations),
        "percentiles": percentile_table(durations),
        "sa3_limits": sa3_limit_percentiles(durations),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    durations = load_pdmx_song_lengths(args.dataset_filepath)
    report = build_report(durations)

    histogram_path = output_dir / "song_length_histogram.png"
    percentile_path = output_dir / "song_length_percentiles.png"
    report_path = output_dir / "song_length_report.json"

    plot_histogram(durations, histogram_path, max_seconds=args.max_seconds)
    plot_percentiles(durations, percentile_path, max_seconds=args.max_seconds)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    link, target = link_analysis_in_repo(OUTPUT_DIR)

    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {report_path}")
    print(f"Symlinked {link} -> {target}")
    print(f"Wrote {histogram_path}")
    print(f"Wrote {percentile_path}")


if __name__ == "__main__":
    main()
