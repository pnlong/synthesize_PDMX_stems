"""Batch-generate captions.csv from synthesis metadata tables."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from shared.config import (
    CAPTIONS_FILE_NAME,
    CAPTIONS_TABLE_COLUMNS,
    NA_STRING,
    OUTPUT_DIR,
    STEMS_FILE_NAME,
)
from synthesis.paths import full_stems_dir
from synthesis.realify.captions.metadata import get_caption


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(description="Generate captions from stem metadata.")
    parser.add_argument("-d", "--dataset_dir", default=None, type=str)
    parser.add_argument("-s", "--seed", default=0, type=int)
    parser.add_argument("-o", "--output_filepath", default=None, type=str)
    return parser.parse_args(args=args, namespace=namespace)


def generate_captions(dataset_dir: str, seed: int = 0) -> pd.DataFrame:
    dataset_dir = Path(dataset_dir)
    songs = pd.read_csv(dataset_dir / "data.csv", sep=",", header=0, index_col=False)
    stems = pd.read_csv(dataset_dir / f"{STEMS_FILE_NAME}.csv", sep=",", header=0, index_col=False)

    merged = stems.merge(songs, on="path", suffixes=("_stem", "_song"))
    rows = []
    for i, row in merged.iterrows():
        rng = random.Random(seed + i)
        prompt = get_caption(row.to_dict(), rng=rng)
        rows.append({"path": row["path"], "track": row["track"], "prompt": prompt})

    return pd.DataFrame(rows, columns=CAPTIONS_TABLE_COLUMNS)


def write_captions(dataset_dir: str, seed: int = 0, output_filepath: str | None = None) -> Path:
    dataset_dir = Path(dataset_dir)
    output_filepath = Path(output_filepath or dataset_dir / f"{CAPTIONS_FILE_NAME}.csv")
    captions = generate_captions(str(dataset_dir), seed=seed)
    captions.to_csv(output_filepath, sep=",", na_rep=NA_STRING, header=True, index=False, mode="w")
    return output_filepath


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir or full_stems_dir(OUTPUT_DIR)
    out = write_captions(dataset_dir, seed=args.seed, output_filepath=args.output_filepath)
    print(f"Wrote captions to {out}")


if __name__ == "__main__":
    main()
