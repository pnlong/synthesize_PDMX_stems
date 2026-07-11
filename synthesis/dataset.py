"""PDMX dataset filtering and sampling for synthesis."""

from __future__ import annotations

import pandas as pd

from shared.config import (
    ABLATION_SAMPLE_SEED,
    ABLATION_SAMPLE_SIZE,
    ABLATION_SUBSET_COLUMN,
)


def prepare_ablation_dataset(
    dataset: pd.DataFrame,
    sample_size: int = ABLATION_SAMPLE_SIZE,
    sample_seed: int = ABLATION_SAMPLE_SEED,
    subset_column: str = ABLATION_SUBSET_COLUMN,
) -> pd.DataFrame:
    """Random sample from ``subset:rated_deduplicated`` for the listening-test ablation."""
    if subset_column not in dataset.columns:
        raise KeyError(f"Missing subset column {subset_column!r}")
    dataset = dataset[dataset[subset_column]].reset_index(drop=True)
    if len(dataset) > sample_size:
        dataset = dataset.sample(n=sample_size, random_state=sample_seed).reset_index(drop=True)
    return dataset


def prepare_full_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """All valid PDMX rows (caller should filter ``subset:all_valid`` first)."""
    return dataset.reset_index(drop=True)
