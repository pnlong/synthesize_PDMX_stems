"""Report duration statistics and recommend SA3 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from shared.config import (
    OUTPUT_DIR,
    RENDER_MODE_BASIC,
    SA3_MEDIUM_MAX_DURATION,
    SA3_SMALL_MUSIC_MAX_DURATION,
)
from synthesis.paths import ablation_raw_dir


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(description="Report stem duration stats and SA3 model recommendation.")
    parser.add_argument("-i", "--input_filepath", default=None, type=str)
    parser.add_argument("-o", "--output_filepath", default=None, type=str)
    return parser.parse_args(args=args, namespace=namespace)


def song_max_durations(df: pd.DataFrame) -> pd.Series:
    return df.groupby("path")["duration_seconds"].max()


def recommend_model(song_durations: pd.Series, threshold: float = 0.95) -> dict:
    under_small = (song_durations <= SA3_SMALL_MUSIC_MAX_DURATION).mean()
    under_medium = (song_durations <= SA3_MEDIUM_MAX_DURATION).mean()
    over_medium = (song_durations > SA3_MEDIUM_MAX_DURATION).sum()

    if under_small >= threshold:
        model = "small-music"
        reason = f"{100 * under_small:.1f}% of songs fit within {SA3_SMALL_MUSIC_MAX_DURATION}s"
    else:
        model = "medium"
        reason = f"only {100 * under_small:.1f}% of songs fit within {SA3_SMALL_MUSIC_MAX_DURATION}s; use medium (up to {SA3_MEDIUM_MAX_DURATION}s)"

    return {
        "recommended_model": model,
        "reason": reason,
        "pct_songs_under_120s": round(100 * under_small, 2),
        "pct_songs_under_380s": round(100 * under_medium, 2),
        "n_songs_over_380s": int(over_medium),
        "median_song_duration_seconds": round(float(song_durations.median()), 2),
        "p95_song_duration_seconds": round(float(song_durations.quantile(0.95)), 2),
        "p99_song_duration_seconds": round(float(song_durations.quantile(0.99)), 2),
    }


def breakdown_by_column(df: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
    song_df = df.groupby("path").agg(
        duration_seconds=("duration_seconds", "max"),
        **{column: (column, "first")},
    ).reset_index()
    grouped = song_df.groupby(column)["duration_seconds"].agg(["count", "median", "max"]).reset_index()
    return grouped.sort_values("count", ascending=False).head(top_n)


def build_report(duration_df: pd.DataFrame) -> dict:
    song_durations = song_max_durations(duration_df)
    report = {
        "summary": recommend_model(song_durations),
        "by_program": breakdown_by_column(duration_df, "program").to_dict(orient="records"),
        "by_is_drum": breakdown_by_column(duration_df, "is_drum").to_dict(orient="records"),
    }
    if "genres" in duration_df.columns:
        report["by_genres"] = breakdown_by_column(duration_df, "genres").to_dict(orient="records")
    return report


def main():
    args = parse_args()
    default_dir = ablation_raw_dir(OUTPUT_DIR, RENDER_MODE_BASIC)
    default_input = f"{default_dir}/duration_analysis.csv"
    input_filepath = args.input_filepath or default_input
    output_filepath = args.output_filepath or f"{default_dir}/duration_report.json"

    df = pd.read_csv(input_filepath)
    report = build_report(df)

    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["summary"], indent=2))
    print(f"Full report written to {output_filepath}")


if __name__ == "__main__":
    main()
