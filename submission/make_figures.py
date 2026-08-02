"""Regenerate ICASSP paper figures as transparent PDFs under submission/figures/."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.plots import plot_gm_program_compare
from shared.config import OUTPUT_DIR

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
INSTRUMENTS_DIR = (
    Path(OUTPUT_DIR) / "dev" / "analysis" / "instruments" / "all_valid"
)
ORIGINAL_STEMS = INSTRUMENTS_DIR / "gm_program_stems.csv"
CORRECTED_STEMS = INSTRUMENTS_DIR / "gm_program_stems_corrected.csv"


def make_gm_program_compare_figure(
    *,
    top_n: int = 10,
    rank_by: str = "corrected",
    show_percentages: bool = False,
) -> Path:
    original = pd.read_csv(ORIGINAL_STEMS)
    corrected = pd.read_csv(CORRECTED_STEMS)
    out = FIGURES_DIR / "gm_program_counts_compare.pdf"
    plot_gm_program_compare(
        original,
        corrected,
        out,
        top_n=top_n,
        rank_by=rank_by,
        show_percentages=show_percentages,
        figsize=(8.0, 4.0),  # 2:1 wide; large canvas so text scales down in-column
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top-N programs for named bars (default: 10).",
    )
    parser.add_argument(
        "--rank-by",
        choices=("corrected", "original"),
        default="corrected",
        help="Inventory used to pick/order top-N + Other (default: corrected).",
    )
    parser.add_argument(
        "--show-percentages",
        action="store_true",
        help="Annotate bars with percentage labels.",
    )
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = make_gm_program_compare_figure(
        top_n=args.top_n,
        rank_by=args.rank_by,
        show_percentages=args.show_percentages,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
