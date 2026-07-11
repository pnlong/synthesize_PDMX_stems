"""Assemble the complete sPDMX dataset at ``{OUTPUT_DIR}/SPDMX/``.

Planned behavior (not implemented):

1. Copy PDMX metadata tables and any other required assets from PDMX.
2. Call ``synthesis.synthesize --full`` internally with the chosen render mode / realify settings.
3. Lay out the final dataset tree under ``{OUTPUT_DIR}/SPDMX/``.

You should not need to run ``synthesize --full`` directly in normal workflows.
"""

from __future__ import annotations

import argparse
import sys

from shared.config import OUTPUT_DIR, RENDER_MODE_BASIC, RENDER_MODE_SLAKH
from synthesis.paths import spdmx_dataset_dir


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="Build the full sPDMX dataset (metadata + stems).",
    )
    parser.add_argument("-o", "--output_dir", default=OUTPUT_DIR, type=str)
    parser.add_argument(
        "--render-mode",
        default=RENDER_MODE_BASIC,
        choices=[RENDER_MODE_BASIC, RENDER_MODE_SLAKH],
    )
    parser.add_argument("--realify", action="store_true")
    return parser.parse_args(args=args, namespace=namespace)


def main():
    args = parse_args()
    target = spdmx_dataset_dir(args.output_dir)
    print(
        f"build_spdmx is not implemented yet.\n"
        f"  Target: {target}\n"
        f"  Render mode: {args.render_mode}\n"
        f"  Realify: {args.realify}\n"
        f"\n"
        f"When implemented, this script will:\n"
        f"  1. Copy PDMX metadata into {target}\n"
        f"  2. Call synthesis.synthesize --full internally\n"
        f"  3. Assemble the final sPDMX layout\n",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
