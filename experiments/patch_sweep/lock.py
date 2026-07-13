"""Lock all phase winners into production slakh render config."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.patch_sweep.config import EXPERIMENT_DIR, WINNERS_LOCKED_PATH
from experiments.patch_sweep.winners import write_locked_config


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Write winners_locked.yaml for slakh production rendering.",
    )
    parser.add_argument(
        "--winners",
        default=EXPERIMENT_DIR / "winners.yaml",
        type=Path,
        help="Per-phase winners YAML.",
    )
    parser.add_argument(
        "--output",
        default=WINNERS_LOCKED_PATH,
        type=Path,
        help="Locked production config output path.",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    out = write_locked_config(args.winners, args.output)
    print(f"Wrote locked slakh config: {out}")
    print("Production slakh mode will load this on next import of synthesis.patches.")
    print("Run B1 ablation: uv run python -m synthesis.synthesize --render-mode slakh")


if __name__ == "__main__":
    main()
