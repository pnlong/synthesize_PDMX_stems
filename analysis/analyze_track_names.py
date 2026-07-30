"""Analyze MIDI track names across PDMX via multiprocessing."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

from tqdm import tqdm

from analysis.pdmx_subset import subset_output_dir
from analysis.plots import plot_track_name_bar
from analysis.track_names import (
    build_track_name_report,
    extract_named_stems_from_mid,
    load_pdmx_mid_paths,
    mid_path_for_row,
    print_track_name_report,
    stems_dataframe,
)
from shared.config import CHUNK_SIZE, OUTPUT_DIR, PDMX_FILEPATH
from shared.repo_symlinks import link_analysis_in_repo
from synthesis.paths import track_names_dir


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description=(
            "Summarize MIDI track names from PDMX (track_name meta-events per "
            "non-empty track)."
        ),
    )
    parser.add_argument("-df", "--dataset_filepath", default=PDMX_FILEPATH, type=str)
    parser.add_argument("-o", "--output_dir", default=track_names_dir(OUTPUT_DIR), type=str)
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
        help="Worker processes for MIDI parsing.",
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
        help="Number of track names to show in the bar chart (rest grouped as Other).",
    )
    return parser.parse_args(args=args, namespace=namespace)


def _worker(args: tuple[str, str]) -> tuple[list[dict], bool]:
    mid_rel, pdmx_root = args
    mid_path = mid_path_for_row(mid_rel, pdmx_root)
    if not mid_path.is_file():
        return [], True
    rows = extract_named_stems_from_mid(mid_path)
    if rows is None:
        return [], True
    return rows, False


def analyze_track_names(
    dataset_filepath: str | Path,
    *,
    subset: str = "all_valid",
    jobs: int = 1,
    limit: int | None = None,
) -> tuple[dict, object]:
    mid_paths, pdmx_root = load_pdmx_mid_paths(dataset_filepath, subset=subset)
    if limit is not None:
        mid_paths = mid_paths.head(limit)

    n_songs = len(mid_paths)
    worker_args = [(mid_rel, str(pdmx_root)) for mid_rel in mid_paths]

    stem_records: list[dict] = []
    n_songs_failed = 0

    if jobs <= 1:
        iterator = worker_args
        if n_songs > 1:
            iterator = tqdm(worker_args, total=n_songs, desc="Parsing MIDI", unit="song")
        for mid_rel, root in iterator:
            rows, failed = _worker((mid_rel, root))
            stem_records.extend(rows)
            n_songs_failed += int(failed)
    else:
        with multiprocessing.Pool(processes=jobs) as pool:
            for rows, failed in tqdm(
                pool.imap(_worker, worker_args, chunksize=CHUNK_SIZE),
                total=n_songs,
                desc="Parsing MIDI",
                unit="song",
            ):
                stem_records.extend(rows)
                n_songs_failed += int(failed)

    stems = stems_dataframe(stem_records)
    report = build_track_name_report(
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

    report, stems = analyze_track_names(
        args.dataset_filepath,
        subset=args.subset,
        jobs=args.jobs,
        limit=args.limit,
    )

    chart_path = output_dir / "track_name_counts.png"
    plot_track_name_bar(stems, chart_path, top_n=args.top_n)

    report_path = output_dir / "track_name_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    stems_path = output_dir / "track_name_stems.csv"
    stems.to_csv(stems_path, index=False)

    link, target = link_analysis_in_repo(OUTPUT_DIR)

    print_track_name_report(report, top_n=max(args.top_n, 50))
    print(f"\nWrote {report_path}")
    print(f"Wrote {chart_path}")
    print(f"Wrote {stems_path}")
    print(f"Symlinked {link} -> {target}")


if __name__ == "__main__":
    main()
