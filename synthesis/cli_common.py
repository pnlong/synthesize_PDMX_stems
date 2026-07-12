"""Shared synthesis/realify CLI arguments."""

from __future__ import annotations

import argparse
import multiprocessing

from shared.config import (
    ABLATION_SAMPLE_SEED,
    ABLATION_SAMPLE_SIZE,
    OUTPUT_DIR,
    PDMX_FILEPATH,
    REALIFY_BATCH_SIZE,
    RENDER_MODE_BASIC,
    RENDER_MODE_SLAKH,
    SOUNDFONT_PATH,
)


def add_synthesis_args(parser: argparse.ArgumentParser):
    parser.add_argument("-df", "--dataset_filepath", default=PDMX_FILEPATH, type=str)
    parser.add_argument("-o", "--output_dir", default=OUTPUT_DIR, type=str)
    parser.add_argument("-sf", "--soundfont_filepath", default=SOUNDFONT_PATH, type=str)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the output directory and rerun from scratch (raw synthesis or realify target).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        "--workers",
        default=int(multiprocessing.cpu_count() / 4),
        type=int,
        help="CPU workers for synthesis, CPU realify (small-music), and realify mixture writes.",
    )
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
    parser.add_argument(
        "--realify-limit",
        default=None,
        type=int,
        help="Realify only the first N stems (smoke tests); default: all stems.",
    )
    parser.add_argument(
        "--realify-batch-size",
        default=None,
        type=int,
        help="SA3 stems per GPU forward pass (default: REALIFY_BATCH_SIZE in shared/config.py).",
    )
    parser.add_argument(
        "-n",
        "--sample-size",
        default=ABLATION_SAMPLE_SIZE,
        type=int,
        help="Ablation sample size (default: from shared/config).",
    )
    parser.add_argument(
        "--mp3",
        action="store_true",
        help="Write stems and mixtures as MP3 instead of FLAC (prototyping; smaller files).",
    )
    parser.add_argument("--sample-seed", default=ABLATION_SAMPLE_SEED, type=int)
