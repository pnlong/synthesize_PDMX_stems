"""Shared synthesis/realify CLI arguments."""

from __future__ import annotations

import argparse
import multiprocessing

from shared.config import (
    ABLATION_MIN_STEMS_PER_CATEGORY,
    ABLATION_SAMPLE_SEED,
    ABLATION_SAMPLE_SIZE,
    OUTPUT_DIR,
    PDMX_FILEPATH,
    REALIFY_BATCH_SIZE,
    REALIFY_CONTENT_FIDELITY_ENFORCE,
    RENDER_MODE_BASIC,
    RENDER_MODES,
    SOUNDFONT_PATH,
)
from synthesis.paths import instruments_dir


def default_gm_register_path(output_dir: str = OUTPUT_DIR) -> str:
    """Default register written by ``analysis.analyze_gm_register --subset all_valid``."""
    return f"{instruments_dir(output_dir)}/all_valid/register.csv"


def add_audio_format_arg(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--flac",
        action="store_true",
        help="Read/write FLAC stems instead of the default MP3.",
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
        choices=list(RENDER_MODES),
        help=(
            "basic = single soundfont; slakh = locked multi-SF recipes; "
            "ddsp_basic / ddsp_slakh = neural DDSP with basic/slakh soundfont fallback "
            "(copies donor stems; skips Fluidsynth for fallbacks)."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Synthesize all valid PDMX songs (default: stratified ablation sample from rated_deduplicated).",
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
        help="Max songs for stratified ablation fill (default: from shared/config).",
    )
    parser.add_argument(
        "--min-stems-per-category",
        default=ABLATION_MIN_STEMS_PER_CATEGORY,
        type=int,
        help="Stratified ablation: keep songs until each listening category has this many stems.",
    )
    parser.add_argument(
        "--no-mixture",
        action="store_true",
        help="Skip writing mixture files; song completeness is stems-only.",
    )
    parser.add_argument(
        "--allow-fallback-render",
        action="store_true",
        help=(
            "DDSP modes: if a donor soundfont stem is missing, render Fluidsynth instead of erroring."
        ),
    )
    add_audio_format_arg(parser)
    parser.add_argument(
        "--no-silence-enforce",
        action="store_true",
        help="Disable post-SA3 silence enforcement on realified stems.",
    )
    parser.add_argument(
        "--content-fidelity-enforce",
        action="store_true",
        help="Enable onset-based content fidelity gate with init_noise_level backoff.",
    )
    parser.add_argument(
        "--no-content-fidelity-enforce",
        action="store_true",
        help="Disable content fidelity gate even if REALIFY_CONTENT_FIDELITY_ENFORCE is set.",
    )
    parser.add_argument("--sample-seed", default=ABLATION_SAMPLE_SEED, type=int)
    parser.add_argument(
        "--register",
        default=None,
        type=str,
        help=(
            "GM register CSV from analysis.analyze_gm_register "
            f"(default: {default_gm_register_path()}). "
            "Corrects per-track program ids before rendering."
        ),
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Do not apply GM register corrections (use MIDI program_change as-is).",
    )
