"""Generate SA3 captions in memory from synthesis metadata tables."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from shared.config import (
    ABLATION_SAMPLE_SEED,
    CAPTIONS_TABLE_COLUMNS,
    DATA_DIR_NAME,
    OUTPUT_DIR,
    STEMS_FILE_NAME,
)
from synthesis.paths import full_stems_dir
from synthesis.realify.captions.metadata import get_caption


def generate_captions_from_tables(
    songs: pd.DataFrame,
    stems: pd.DataFrame,
    seed: int = ABLATION_SAMPLE_SEED,
    prompt_variant: str = "current",
    presets: dict | None = None,
) -> pd.DataFrame:
    merged = stems.merge(songs, on="path", suffixes=("_stem", "_song"))
    rows = []
    for i, row in merged.iterrows():
        rng = random.Random(seed + i)
        if presets is not None:
            from synthesis.realify.preset_config import select_preset

            preset = select_preset(presets, row)
            prompt_variant = preset.get("prompt_variant", prompt_variant)
        prompt = get_caption(
            row.to_dict(),
            rng=rng,
            prompt_variant=prompt_variant,
        )
        rows.append({"path": row["path"], "track": row["track"], "prompt": prompt})
    return pd.DataFrame(rows, columns=CAPTIONS_TABLE_COLUMNS)


def generate_captions(
    dataset_dir: str | Path,
    seed: int = ABLATION_SAMPLE_SEED,
    presets: dict | None = None,
) -> pd.DataFrame:
    dataset_dir = Path(dataset_dir)
    songs = pd.read_csv(dataset_dir / f"{DATA_DIR_NAME}.csv")
    stems = pd.read_csv(dataset_dir / f"{STEMS_FILE_NAME}.csv")
    return generate_captions_from_tables(songs, stems, seed=seed, presets=presets)


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="Preview in-memory SA3 captions for a synthesized dataset directory."
    )
    parser.add_argument("-d", "--dataset_dir", default=None, type=str)
    parser.add_argument("-s", "--seed", default=ABLATION_SAMPLE_SEED, type=int)
    parser.add_argument("-n", "--limit", default=3, type=int, help="Number of prompts to print.")
    return parser.parse_args(args=args, namespace=namespace)


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir or full_stems_dir(OUTPUT_DIR)
    captions = generate_captions(dataset_dir, seed=args.seed)
    print(f"Generated {len(captions)} captions in memory from {dataset_dir}")
    for _, row in captions.head(args.limit).iterrows():
        print(f"\n[{row['path']} track {row['track']}]")
        print(row["prompt"])


if __name__ == "__main__":
    main()
