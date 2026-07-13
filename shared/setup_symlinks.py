"""CLI to create in-repo symlinks to deepfreeze dev artifact directories."""

from __future__ import annotations

import argparse

from shared.config import OUTPUT_DIR, SOUNDFONT_DIR
from shared.repo_symlinks import setup_dev_symlinks


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="Create gitignored in-repo symlinks to {OUTPUT_DIR}/dev/ artifacts.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=OUTPUT_DIR,
        type=str,
        help=f"SPDMX output root (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--soundfont-dir",
        default=SOUNDFONT_DIR,
        type=str,
        help=f"Soundfont library directory (default: {SOUNDFONT_DIR}).",
    )
    return parser.parse_args(args=args, namespace=namespace)


def main():
    args = parse_args()
    for link, target in setup_dev_symlinks(
        args.output_dir,
        soundfont_dir=args.soundfont_dir,
    ):
        print(f"{link} -> {target}")


if __name__ == "__main__":
    main()
