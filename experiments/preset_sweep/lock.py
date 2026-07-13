"""Lock preset phase winners into production categories.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.preset_sweep.config import CATEGORIES_YAML_PATH, WINNERS_LOCKED_PATH
from experiments.preset_sweep.winners import phase_is_complete, write_locked_config
from experiments.preset_sweep.config import PHASE3


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Write winners_locked.yaml and update categories.yaml for production realify.",
    )
    parser.add_argument(
        "--winners",
        default=Path(__file__).resolve().parent / "winners.yaml",
        type=Path,
        help="Per-phase winners YAML.",
    )
    parser.add_argument(
        "--output",
        default=WINNERS_LOCKED_PATH,
        type=Path,
        help="Locked preset config output path.",
    )
    parser.add_argument(
        "--categories",
        default=CATEGORIES_YAML_PATH,
        type=Path,
        help="Production categories.yaml to update.",
    )
    parser.add_argument(
        "--skip-categories-yaml",
        action="store_true",
        help="Write winners_locked.yaml only; do not modify categories.yaml.",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    out = write_locked_config(
        args.winners,
        args.output,
        args.categories,
        update_categories_yaml=not args.skip_categories_yaml,
    )
    print(f"Wrote locked preset config: {out}")
    if not args.skip_categories_yaml:
        print(f"Updated production presets: {args.categories}")
    if not phase_is_complete(PHASE3, args.winners):
        print("Note: phase 3 not completed — locked config uses default steps/cfg_scale.")
    print("Run A2 ablation: uv run python -m synthesis.synthesize --render-mode basic --realify")


if __name__ == "__main__":
    main()
