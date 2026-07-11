"""Symbolic stem length analysis (note counts) via mido."""

from __future__ import annotations

import argparse
import multiprocessing

import mido
import numpy as np
import pandas as pd
from tqdm import tqdm

from shared.config import CHUNK_SIZE, PDMX_FILEPATH

QUANTILES = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(description="Analyze PDMX stem note counts via MIDI.")
    parser.add_argument("-df", "--dataset_filepath", default=PDMX_FILEPATH, type=str)
    parser.add_argument("-o", "--output_filepath", default=None, type=str)
    parser.add_argument("-j", "--jobs", default=int(multiprocessing.cpu_count() / 4), type=int)
    return parser.parse_args(args=args, namespace=namespace)


def note_counts_for_mid(mid_path: str) -> list[int]:
    midi = mido.MidiFile(filename=mid_path, charset="utf8")
    counts = []
    for track in midi.tracks:
        n_notes = sum(
            1 for msg in track
            if msg.type == "note_on" and msg.velocity > 0
        )
        counts.append(n_notes)
    return counts


def analyze_stems(dataset_filepath: str, jobs: int = 1) -> np.ndarray:
    dataset = pd.read_csv(dataset_filepath)
    if "subset:all_valid" in dataset.columns:
        dataset = dataset[dataset["subset:all_valid"]]
    from os.path import dirname
    original_dir = dirname(dataset_filepath)
    mid_paths = [original_dir + p[1:] for p in dataset["mid"]]

    if jobs <= 1:
        results = [note_counts_for_mid(p) for p in tqdm(mid_paths)]
    else:
        with multiprocessing.Pool(processes=jobs) as pool:
            results = list(tqdm(pool.imap(note_counts_for_mid, mid_paths, chunksize=CHUNK_SIZE), total=len(mid_paths)))

    flat = sorted(sum(results, []))
    return np.array(flat)


def main():
    args = parse_args()
    arr = analyze_stems(args.dataset_filepath, jobs=args.jobs)
    if args.output_filepath:
        np.save(args.output_filepath, arr)
    print("Quantiles (note counts):")
    for q, v in zip(QUANTILES, np.quantile(arr, QUANTILES)):
        print(f"  {100 * q:.1f}%: {v:.0f}")


if __name__ == "__main__":
    main()
