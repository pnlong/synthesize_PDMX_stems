"""Analyze synthesized FLAC stem durations."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from shared.config import CHUNK_SIZE, OUTPUT_DIR, RENDER_MODE_BASIC, STEMS_FILE_NAME
from synthesis.paths import ablation_raw_dir


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(description="Analyze FLAC stem durations in sPDMX_stems.")
    parser.add_argument(
        "-d", "--dataset_dir",
        default=ablation_raw_dir(OUTPUT_DIR, RENDER_MODE_BASIC),
        type=str,
    )
    parser.add_argument("-o", "--output_filepath", default=None, type=str)
    parser.add_argument("-j", "--jobs", default=int(multiprocessing.cpu_count() / 4), type=int)
    return parser.parse_args(args=args, namespace=namespace)


def stem_duration_seconds(flac_path: Path) -> float:
    info = sf.info(str(flac_path))
    return info.frames / info.samplerate


def analyze_stem_row(row: pd.Series) -> dict:
    song_dir = Path(row["path"])
    flac = song_dir / f"stem_{int(row['track'])}.flac"
    duration = stem_duration_seconds(flac) if flac.exists() else float("nan")
    return {
        "path": row["path"],
        "track": int(row["track"]),
        "duration_seconds": duration,
        "program": row.get("program"),
        "is_drum": row.get("is_drum"),
        "name": row.get("name"),
        "genres": row.get("genres"),
    }


def analyze_durations(dataset_dir: str, jobs: int = 1) -> pd.DataFrame:
    dataset_dir = Path(dataset_dir)
    stems = pd.read_csv(dataset_dir / f"{STEMS_FILE_NAME}.csv")
    songs = pd.read_csv(dataset_dir / "data.csv", usecols=["path", "genres"])
    merged = stems.merge(songs, on="path", how="left")

    if jobs <= 1:
        records = [analyze_stem_row(row) for _, row in tqdm(merged.iterrows(), total=len(merged))]
    else:
        with multiprocessing.Pool(processes=jobs) as pool:
            records = list(tqdm(
                pool.imap(analyze_stem_row, (row for _, row in merged.iterrows()), chunksize=CHUNK_SIZE),
                total=len(merged),
            ))

    return pd.DataFrame(records)


def main():
    args = parse_args()
    output_filepath = args.output_filepath or f"{args.dataset_dir}/duration_analysis.csv"
    df = analyze_durations(args.dataset_dir, jobs=args.jobs)
    df.to_csv(output_filepath, index=False)
    print(f"Wrote {len(df)} stem duration records to {output_filepath}")


if __name__ == "__main__":
    main()
