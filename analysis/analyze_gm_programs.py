"""Analyze General MIDI program usage from PDMX ``tracks`` metadata."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

from tqdm import tqdm

from analysis.gm_programs import (
    build_gm_report,
    load_pdmx_tracks,
    parse_tracks_cell,
    print_gm_report,
    stems_dataframe,
    tracks_cell_to_stem_records,
)
from analysis.pdmx_subset import subset_output_dir
from analysis.plots import plot_gm_program_bar
from shared.config import CHUNK_SIZE, OUTPUT_DIR, PDMX_FILEPATH
from shared.repo_symlinks import link_analysis_in_repo
from synthesis.paths import instruments_dir


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description=(
            "Summarize General MIDI program ids from PDMX.csv ``tracks`` "
            "(hyphen-separated GM programs per song)."
        ),
    )
    parser.add_argument("-df", "--dataset_filepath", default=PDMX_FILEPATH, type=str)
    parser.add_argument("-o", "--output_dir", default=instruments_dir(OUTPUT_DIR), type=str)
    parser.add_argument(
        "--subset",
        choices=("all_valid", "rated_deduplicated", "all"),
        default="all_valid",
        help="PDMX rows to include (default: all_valid, same as synthesize --full).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        default=max(1, multiprocessing.cpu_count() // 2),
        type=int,
        help="Worker processes for parsing ``tracks`` cells.",
    )
    parser.add_argument(
        "-n",
        "--limit",
        default=None,
        type=int,
        help="Optional cap on number of songs (for smoke tests).",
    )
    parser.add_argument(
        "--top-n",
        default=40,
        type=int,
        help="Number of GM ids to show in the bar chart (rest grouped as Other).",
    )
    return parser.parse_args(args=args, namespace=namespace)


def _worker(tracks_values: list) -> tuple[list[dict], int]:
    records: list[dict] = []
    n_failed = 0
    for tracks in tracks_values:
        programs = parse_tracks_cell(tracks)
        if programs is None:
            n_failed += 1
            continue
        records.extend(tracks_cell_to_stem_records(tracks))
    return records, n_failed


def _chunked(values: list, chunk_size: int) -> list[list]:
    if chunk_size <= 0:
        chunk_size = 1
    return [values[i:i + chunk_size] for i in range(0, len(values), chunk_size)]


def analyze_gm_programs(
    dataset_filepath: str | Path,
    *,
    subset: str = "all_valid",
    jobs: int = 1,
    limit: int | None = None,
) -> tuple[dict, object]:
    tracks = load_pdmx_tracks(dataset_filepath, subset=subset)
    if limit is not None:
        tracks = tracks.head(limit)

    track_values = tracks.tolist()
    n_songs = len(track_values)
    chunk_size = max(1, len(track_values) // max(jobs * 4, 1))
    chunks = _chunked(track_values, chunk_size)

    stem_records: list[dict] = []
    n_songs_failed = 0

    if jobs <= 1 or len(chunks) <= 1:
        iterator = chunks
        if len(chunks) > 1:
            iterator = tqdm(chunks, desc="Parsing tracks", unit="chunk")
        for chunk in iterator:
            rows, failed = _worker(chunk)
            stem_records.extend(rows)
            n_songs_failed += failed
    else:
        with multiprocessing.Pool(processes=jobs) as pool:
            for rows, failed in tqdm(
                pool.imap(_worker, chunks, chunksize=CHUNK_SIZE),
                total=len(chunks),
                desc="Parsing tracks",
                unit="chunk",
            ):
                stem_records.extend(rows)
                n_songs_failed += failed

    stems = stems_dataframe(stem_records)
    report = build_gm_report(
        stems,
        subset=subset,
        n_songs=n_songs,
        n_songs_failed=n_songs_failed,
    )
    return report, stems


def main():
    args = parse_args()
    output_dir = subset_output_dir(args.output_dir, args.subset)
    output_dir.mkdir(parents=True, exist_ok=True)

    report, stems = analyze_gm_programs(
        args.dataset_filepath,
        subset=args.subset,
        jobs=args.jobs,
        limit=args.limit,
    )

    chart_path = output_dir / "gm_program_counts.png"
    plot_gm_program_bar(stems, chart_path, top_n=args.top_n)

    report_path = output_dir / "gm_program_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    stems_path = output_dir / "gm_program_stems.csv"
    stems.to_csv(stems_path, index=False)

    link, target = link_analysis_in_repo(OUTPUT_DIR)

    print_gm_report(report)
    print(f"\nWrote {report_path}")
    print(f"Wrote {chart_path}")
    print(f"Wrote {stems_path}")
    print(f"Symlinked {link} -> {target}")


if __name__ == "__main__":
    main()
