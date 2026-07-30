"""Render probe stems with many soundfonts for quick listening comparisons."""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

import yaml
from tqdm import tqdm

from experiments.patch_sweep.config import (
    SOUNDFONTS_CATALOG,
    load_soundfont_catalog,
    soundfont_path,
)
from experiments.patch_sweep.sweep import render_probe_stem
from experiments.paths import DEFAULT_PROBE_STEMS, patch_sweep_output_root
from experiments.probe_stems import load_probe_stems, resolve_mid_path, validate_probe_stems
from shared.config import (
    ABLATION_SAMPLE_SEED,
    CHUNK_SIZE,
    OUTPUT_DIR,
    PDMX_FILEPATH,
    SOUNDFONT_DIR,
)
from synthesis.audio import synthesis_audio_format
from synthesis.cli_common import add_audio_format_arg

ARCHIVE_CATALOG = Path(__file__).resolve().parent / "archive_soundfonts.yaml"
DEFAULT_OUTPUT = Path(patch_sweep_output_root(OUTPUT_DIR)) / "soundfont_audition"

# GM-weighted default: piano dominates all_valid; choir + flute next.
DEFAULT_CATEGORIES = ("piano", "voice", "wind", "strings", "brass", "drums")


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="Render probe stems across soundfont candidates for audition.",
    )
    parser.add_argument(
        "--catalog",
        default=str(ARCHIVE_CATALOG),
        type=str,
        help="Soundfont catalog YAML (archive_soundfonts.yaml or soundfonts.yaml).",
    )
    parser.add_argument(
        "--soundfont-dir",
        default=SOUNDFONT_DIR,
        type=str,
        help="Root directory containing soundfont files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        type=str,
    )
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        type=str,
        help=f"Comma-separated probe categories (default: {','.join(DEFAULT_CATEGORIES)}).",
    )
    parser.add_argument(
        "--tags",
        default=None,
        type=str,
        help="Only soundfonts whose tags intersect this list (e.g. piano,orchestra).",
    )
    parser.add_argument(
        "--ids",
        default=None,
        type=str,
        help="Comma-separated soundfont ids to render (overrides --tags).",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Optional cap on number of soundfonts.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        default=max(1, multiprocessing.cpu_count() // 2),
        type=int,
    )
    add_audio_format_arg(parser)
    return parser.parse_args(args=args, namespace=namespace)


def load_catalog(path: Path) -> dict:
    if not path.is_file():
        return load_soundfont_catalog(SOUNDFONTS_CATALOG)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def filter_candidates(catalog: dict, *, tags: str | None, ids: str | None, limit: int | None) -> list[dict]:
    candidates = list(catalog.get("candidates", []))
    if ids:
        wanted = {part.strip() for part in ids.split(",") if part.strip()}
        candidates = [entry for entry in candidates if entry["id"] in wanted]
    elif tags:
        wanted = {part.strip() for part in tags.split(",") if part.strip()}
        candidates = [
            entry for entry in candidates
            if wanted.intersection(set(entry.get("tags") or []))
        ]
    if limit is not None:
        candidates = candidates[:limit]
    return candidates


def _render_task(args: tuple) -> str:
    (
        candidate,
        probe,
        soundfont_dir,
        output_dir,
        audio_format,
        sample_seed,
        pdmx_filepath,
    ) = args

    mid_path = resolve_mid_path(probe["song_id"], pdmx_filepath=pdmx_filepath)
    song_path = str(Path("data") / probe["song_id"])
    sf_path = soundfont_path(candidate["file"], soundfont_dir)
    out_dir = Path(output_dir) / candidate["id"] / probe["category"] / probe["id"]
    out_path = out_dir / f"stem_{int(probe['track'])}.{audio_format}"

    if out_path.is_file():
        return str(out_path)

    render_probe_stem(
        mid_path=mid_path,
        track=int(probe["track"]),
        pool_id=None,
        category=probe.get("category"),
        soundfont_filepath=str(sf_path),
        fx_profile=None,
        sample_seed=sample_seed,
        song_path=song_path,
        out_path=out_path,
        audio_format=audio_format,
    )
    return str(out_path)


def main():
    args = parse_args()
    catalog = load_catalog(Path(args.catalog))
    candidates = filter_candidates(
        catalog,
        tags=args.tags,
        ids=args.ids,
        limit=args.limit,
    )
    if not candidates:
        raise SystemExit("No soundfont candidates matched filters.")

    categories = {part.strip() for part in args.categories.split(",") if part.strip()}
    probes = load_probe_stems(DEFAULT_PROBE_STEMS)
    validate_probe_stems(probes, pdmx_filepath=PDMX_FILEPATH)
    probes = [probe for probe in probes if probe["category"] in categories]

    audio_format = synthesis_audio_format(args.flac)
    tasks = [
        (candidate, probe, args.soundfont_dir, args.output_dir, audio_format, ABLATION_SAMPLE_SEED, PDMX_FILEPATH)
        for candidate in candidates
        for probe in probes
    ]

    print(f"Catalog: {args.catalog}")
    print(f"Soundfonts: {len(candidates)}  Probes: {len(probes)}  Renders: {len(tasks)}")
    print(f"Output: {args.output_dir}")

    if args.jobs <= 1:
        for task in tqdm(tasks, desc="Audition renders", unit="clip"):
            _render_task(task)
    else:
        with multiprocessing.Pool(processes=args.jobs) as pool:
            list(tqdm(
                pool.imap(_render_task, tasks, chunksize=CHUNK_SIZE),
                total=len(tasks),
                desc="Audition renders",
                unit="clip",
            ))

    print("\nListen locally:")
    print(f"  find {args.output_dir} -name '*.{audio_format}' | sort")
    print("Or browse with your file manager / ffplay / synthesis.listening.serve patterns.")


if __name__ == "__main__":
    main()
