"""Rebuild summable stems for realified ablations and optionally resume realify."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from shared.config import OUTPUT_DIR
from synthesis.mix import normalize_stems_for_dataset
from synthesis.paths import ablation_raw_dir, ablation_realify_dir


def rebuild_mixtures(*, jobs: int = 8, audio_format: str = "mp3") -> None:
    for mode in ("basic", "slakh"):
        source = Path(ablation_raw_dir(OUTPUT_DIR, mode))
        output = Path(ablation_realify_dir(OUTPUT_DIR, mode))
        if not output.is_dir():
            print(f"skip {mode}_realify: missing {output}")
            continue
        print(f"Normalizing stems for summability in {mode}_realify …")
        normalize_stems_for_dataset(source, output, jobs=jobs, audio_format=audio_format)
        print(f"  done: {output}")


def run_realify(mode: str) -> None:
    cmd = [
        "uv", "run", "python", "-m", "synthesis.synthesize",
        "--render-mode", mode,
        "--realify",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Finish A2/B2 ablations: realify + stem summability normalize.",
    )
    parser.add_argument(
        "--realify",
        choices=("basic", "slakh", "both"),
        help="Run SA3 realify for incomplete ablation(s).",
    )
    parser.add_argument(
        "--mixtures-only",
        action="store_true",
        help="Only peak-normalize existing realified stems for summability.",
    )
    parser.add_argument("-j", "--jobs", default=8, type=int)
    return parser.parse_args(args)


def main(args=None) -> None:
    opts = parse_args(args)
    if opts.mixtures_only or not opts.realify:
        rebuild_mixtures(jobs=opts.jobs)
    if opts.realify and not opts.mixtures_only:
        if opts.realify in ("basic", "both"):
            run_realify("basic")
        if opts.realify in ("slakh", "both"):
            run_realify("slakh")
        rebuild_mixtures(jobs=opts.jobs)


if __name__ == "__main__":
    main()
