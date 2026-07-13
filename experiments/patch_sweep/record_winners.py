"""Record per-category winners from a phase listening test into winners.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.listening.aggregate import aggregate_winners, load_responses, ratings_dataframe
from experiments.listening.catalog import SweepCatalog
from experiments.patch_sweep.config import EXPERIMENT_DIR, PHASES, phase_output_dir
from experiments.patch_sweep.sweep import default_output_dir
from experiments.patch_sweep.winners import record_phase_winners


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Record blinded listening winners for a patch sweep phase.",
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=list(PHASES),
        help="Phase whose winners to record.",
    )
    parser.add_argument(
        "--responses",
        required=True,
        type=Path,
        help="Exported responses JSON from listening test.",
    )
    parser.add_argument(
        "--sweep-dir",
        default=None,
        type=Path,
        help="Phase sweep output dir (default: output/<phase>).",
    )
    parser.add_argument(
        "--winners",
        default=EXPERIMENT_DIR / "winners.yaml",
        type=Path,
        help="Winners YAML to update.",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    sweep_root = args.sweep_dir or phase_output_dir(default_output_dir(), args.phase)

    responses = load_responses(args.responses)
    df = ratings_dataframe(responses)
    _, winners_df = aggregate_winners(df)

    if winners_df.empty:
        raise RuntimeError("No winners produced from responses.")

    winner_map = {
        str(row["category"]): str(row["variant_id"])
        for _, row in winners_df.iterrows()
    }

    doc = record_phase_winners(args.phase, winner_map, path=args.winners)

    print(f"Recorded {args.phase} winners to {args.winners}")
    for category, variant_id in sorted(winner_map.items()):
        print(f"  {category}: {variant_id}")

    catalog = SweepCatalog("patch", sweep_root)
    manifest = catalog._manifest
    if not manifest.empty:
        print("\nResolved settings:")
        for category, variant_id in sorted(winner_map.items()):
            match = manifest[manifest["variant_id"] == variant_id].iloc[0]
            details = [f"variant={variant_id}"]
            if "soundfont_id" in match and str(match["soundfont_id"]):
                details.append(f"soundfont={match['soundfont_id']}")
            if "fx_profile" in match and str(match["fx_profile"]):
                details.append(f"fx={match['fx_profile']}")
            if "pool_id" in match and str(match["pool_id"]):
                details.append(f"pool={match['pool_id']}")
            print(f"  {category}: {', '.join(details)}")

    incomplete = [
        phase for phase in PHASES
        if not doc["phases"].get(phase, {}).get("completed")
    ]
    if incomplete:
        print(f"\nRemaining phases: {', '.join(incomplete)}")
    else:
        print("\nAll phases complete. Run: uv run python -m experiments.patch_sweep.lock")


if __name__ == "__main__":
    main()
