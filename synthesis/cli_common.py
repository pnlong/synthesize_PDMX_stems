"""Shared synthesis/realify CLI arguments."""

from __future__ import annotations

import argparse
import multiprocessing

from shared.config import (
    ABLATION_SAMPLE_SEED,
    ABLATION_SAMPLE_SIZE,
    OUTPUT_DIR,
    PDMX_FILEPATH,
    RENDER_MODE_BASIC,
    RENDER_MODE_SLAKH,
    SOUNDFONT_PATH,
)


def add_synthesis_args(parser: argparse.ArgumentParser):
    parser.add_argument("-df", "--dataset_filepath", default=PDMX_FILEPATH, type=str)
    parser.add_argument("-o", "--output_dir", default=OUTPUT_DIR, type=str)
    parser.add_argument("-sf", "--soundfont_filepath", default=SOUNDFONT_PATH, type=str)
    parser.add_argument("-r", "--reset", action="store_true")
    parser.add_argument("-rt", "--reset_tables", action="store_true")
    parser.add_argument("-j", "--jobs", default=int(multiprocessing.cpu_count() / 4), type=int)
    parser.add_argument(
        "--render-mode",
        default=RENDER_MODE_BASIC,
        choices=[RENDER_MODE_BASIC, RENDER_MODE_SLAKH],
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Synthesize all valid PDMX songs (default: random ablation sample from rated_deduplicated).",
    )
    parser.add_argument("--realify", action="store_true")
    parser.add_argument("-m", "--model", default="medium", choices=["small-music", "medium"])
    parser.add_argument("--realify-limit", default=None, type=int)
    parser.add_argument("--sample-size", default=ABLATION_SAMPLE_SIZE, type=int)
    parser.add_argument("--sample-seed", default=ABLATION_SAMPLE_SEED, type=int)
