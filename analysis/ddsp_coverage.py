"""Estimate neural-DDSP coverage on PDMX (MIDI-DDSP + DDSP-Piano vs soundfont).

Program-level stats use the PDMX ``tracks`` column (fast, no MIDI I/O).
Optional ``--check-monophony`` opens MIDI files to apply the MIDI-DDSP
monophony gate (slower; needed for paper-ready eligible %).

Examples:

  uv run python -m analysis.ddsp_coverage
  uv run python -m analysis.ddsp_coverage --subset rated_deduplicated --check-monophony -n 500
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import mido
import pandas as pd
from tqdm import tqdm

from analysis.gm_programs import parse_tracks_cell
from analysis.pdmx_subset import filter_pdmx_subset
from shared.config import OUTPUT_DIR, PDMX_FILEPATH
from synthesis.ddsp.routing import (
    BACKEND_DDSP_PIANO,
    BACKEND_MIDI_DDSP,
    BACKEND_SOUNDFONT,
    route_stem,
)
from synthesis.paths import analysis_root


def _song_mid_path(dataset_dir: Path, mid_rel: str) -> Path:
    # PDMX mid paths look like ./mid/... — mirror synthesize.py string join
    # (Path / "/abs" would discard dataset_dir).
    rel = mid_rel[1:] if mid_rel.startswith(".") else mid_rel
    return Path(str(dataset_dir) + rel)


def analyze_programs(dataset: pd.DataFrame) -> dict:
    """Coverage from GM programs alone (monophony not checked)."""
    reason_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    n_stems = 0
    for tracks in dataset["tracks"]:
        programs = parse_tracks_cell(tracks)
        if programs is None:
            continue
        for program in programs:
            n_stems += 1
            # Program-only: treat drums if we only have program ids — drums are
            # not in the tracks melodic list typically; channel-9 is MIDI-only.
            route = route_stem(
                program=int(program),
                is_drum=False,
                track_name=None,
                check_monophony=False,
            )
            backend_counts[route.backend] += 1
            reason_counts[route.reason] += 1

    def pct(n: int) -> float:
        return round(100.0 * n / n_stems, 4) if n_stems else 0.0

    return {
        "n_songs": int(len(dataset)),
        "n_stems": n_stems,
        "backends": {
            backend: {"count": count, "pct_of_stems": pct(count)}
            for backend, count in sorted(backend_counts.items(), key=lambda x: -x[1])
        },
        "reasons": {
            reason: {"count": count, "pct_of_stems": pct(count)}
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1])
        },
        "neural_pct_of_stems": pct(
            backend_counts[BACKEND_MIDI_DDSP] + backend_counts[BACKEND_DDSP_PIANO]
        ),
        "note": (
            "Program-only estimate; MIDI-DDSP monophony not applied. "
            "Pass --check-monophony for eligible-after-mono stats."
        ),
    }


def analyze_with_monophony(
    dataset: pd.DataFrame,
    *,
    dataset_dir: Path,
    limit: int | None,
) -> dict:
    """Open MIDI files and apply full routing including monophony."""
    reason_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    n_stems = 0
    n_notes_total = 0
    n_notes_by_backend: Counter[str] = Counter()
    songs = dataset if limit is None else dataset.head(limit)

    for _, row in tqdm(songs.iterrows(), total=len(songs), desc="Routing MIDI", unit="song"):
        mid_path = _song_mid_path(dataset_dir, str(row["mid"]))
        if not mid_path.is_file():
            continue
        try:
            midi = mido.MidiFile(filename=str(mid_path), charset="utf8")
        except Exception:
            continue
        for track in midi.tracks:
            program = 0
            is_drum = False
            track_name = None
            n_notes = 0
            saw_channel = False
            for message in track:
                if message.type == "note_on" and message.velocity > 0:
                    n_notes += 1
                elif message.type == "program_change":
                    program = message.program
                elif message.type == "track_name":
                    track_name = message.name
                if not saw_channel and hasattr(message, "channel"):
                    is_drum = message.channel == 9
                    saw_channel = True
            if n_notes == 0 and not any(
                m.type in ("note_on", "note_off") for m in track
            ):
                # Skip meta-only tracks.
                if all(
                    m.type in (
                        "track_name",
                        "copyright",
                        "marker",
                        "lyrics",
                        "text",
                        "end_of_track",
                        "set_tempo",
                        "time_signature",
                        "key_signature",
                        "smpte_offset",
                        "midi_port",
                        "instrument_name",
                        "sequence_number",
                        "channel_prefix",
                        "sequencer_specific",
                    )
                    or m.is_meta
                    for m in track
                ):
                    continue
            n_stems += 1
            n_notes_total += n_notes
            route = route_stem(
                program=program,
                is_drum=is_drum,
                track_name=track_name,
                track=track,
                ticks_per_beat=midi.ticks_per_beat,
                check_monophony=True,
            )
            backend_counts[route.backend] += 1
            reason_counts[route.reason] += 1
            n_notes_by_backend[route.backend] += n_notes

    def pct_stems(n: int) -> float:
        return round(100.0 * n / n_stems, 4) if n_stems else 0.0

    def pct_notes(n: int) -> float:
        return round(100.0 * n / n_notes_total, 4) if n_notes_total else 0.0

    return {
        "n_songs_scanned": int(len(songs)),
        "n_stems": n_stems,
        "n_notes": n_notes_total,
        "backends": {
            backend: {
                "count": count,
                "pct_of_stems": pct_stems(count),
                "n_notes": n_notes_by_backend[backend],
                "pct_of_notes": pct_notes(n_notes_by_backend[backend]),
            }
            for backend, count in sorted(backend_counts.items(), key=lambda x: -x[1])
        },
        "reasons": {
            reason: {"count": count, "pct_of_stems": pct_stems(count)}
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1])
        },
        "neural_pct_of_stems": pct_stems(
            backend_counts[BACKEND_MIDI_DDSP] + backend_counts[BACKEND_DDSP_PIANO]
        ),
        "neural_pct_of_notes": pct_notes(
            n_notes_by_backend[BACKEND_MIDI_DDSP] + n_notes_by_backend[BACKEND_DDSP_PIANO]
        ),
        "midi_ddsp_mono_eligible_stems": backend_counts[BACKEND_MIDI_DDSP],
        "ddsp_piano_stems": backend_counts[BACKEND_DDSP_PIANO],
        "soundfont_stems": backend_counts[BACKEND_SOUNDFONT],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-df", "--dataset_filepath", default=PDMX_FILEPATH)
    parser.add_argument("-o", "--output_dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--subset",
        default="all_valid",
        help="PDMX subset filter (default: all_valid).",
    )
    parser.add_argument(
        "--check-monophony",
        action="store_true",
        help="Open MIDI files and apply MIDI-DDSP monophony gate.",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Max songs when --check-monophony (default: all).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dataset = pd.read_csv(args.dataset_filepath, sep=",", header=0, index_col=False)
    dataset = filter_pdmx_subset(dataset, args.subset)
    dataset_dir = Path(args.dataset_filepath).parent

    program_report = analyze_programs(dataset)
    report: dict = {"subset": args.subset, "program_only": program_report}

    if args.check_monophony:
        report["with_monophony"] = analyze_with_monophony(
            dataset,
            dataset_dir=dataset_dir,
            limit=args.limit,
        )

    out_dir = Path(analysis_root(args.output_dir)) / "ddsp_coverage"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Fallback when OUTPUT_DIR is not writable in this environment.
        out_dir = Path("analysis") / "output" / "ddsp_coverage"
        out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.subset}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {out_path}")
    print(json.dumps(report, indent=2))

    # Listening guidance (printed for the paper protocol).
    print(
        "\nListening protocol notes:\n"
        "- Isolated-stem A/B: same notes under B1 (slakh) vs B2 (slakh_realify) vs B3 "
        "(ddsp_slakh) for piano and one MIDI-DDSP instrument.\n"
        "- Full-mix showcase: prefer pieces with high neural coverage "
        "(piano + strings/winds), not random draws dominated by drums/guitar.\n"
        "- Vocals stay on soundfont(+SA3); lyric SVS excluded for provenance.\n"
    )
    return report


if __name__ == "__main__":
    main()
