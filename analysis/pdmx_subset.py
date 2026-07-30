"""Shared PDMX subset filtering for analysis scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PDMX_SUBSETS = ("all_valid", "rated_deduplicated", "all")


def filter_pdmx_subset(dataset: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "all":
        return dataset
    if subset == "all_valid":
        return dataset[dataset["subset:all_valid"] == True]  # noqa: E712
    if subset == "rated_deduplicated":
        from shared.config import ABLATION_SUBSET_COLUMN

        return dataset[
            (dataset["subset:all_valid"] == True)  # noqa: E712
            & (dataset[ABLATION_SUBSET_COLUMN] == True)  # noqa: E712
        ]
    raise ValueError(f"Unknown subset {subset!r}; expected one of {PDMX_SUBSETS}")


def subset_output_dir(base_dir: str | Path, subset: str) -> Path:
    """Place analysis artifacts under ``{base_dir}/{subset}/``."""
    return Path(base_dir) / subset
