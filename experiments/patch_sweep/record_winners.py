"""Record per-category winners from a phase listening test into winners.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.listening.aggregate import (
    DEFAULT_MEAN_RATING_THRESHOLD,
    aggregate_winners,
    load_responses,
    ratings_dataframe,
    shortlist_dataframe,
    shortlist_variants,
)
from experiments.listening.catalog import SweepCatalog
from experiments.patch_sweep.config import EXPERIMENT_DIR, PHASE1, PHASES, phase_output_dir
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
    parser.add_argument(
        "--mean-rating-threshold",
        default=DEFAULT_MEAN_RATING_THRESHOLD,
        type=float,
        help="Phase 1 only: include soundfonts with mean(content, realism)/2 >= this.",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    sweep_root = args.sweep_dir or phase_output_dir(default_output_dir(), args.phase)

    responses = load_responses(args.responses)
    df = ratings_dataframe(responses)

    if args.phase == PHASE1:
        winner_map = shortlist_variants(
            df,
            mean_rating_threshold=args.mean_rating_threshold,
        )
        if not winner_map:
            raise RuntimeError("No phase-1 soundfont shortlists produced from responses.")
        stats = shortlist_dataframe(
            df,
            mean_rating_threshold=args.mean_rating_threshold,
        )
    else:
        _, winners_df = aggregate_winners(df)
        if winners_df.empty:
            raise RuntimeError("No winners produced from responses.")
        winner_map = {
            str(row["category"]): str(row["variant_id"])
            for _, row in winners_df.iterrows()
        }
        stats = None

    doc = record_phase_winners(args.phase, winner_map, path=args.winners)

    print(f"Recorded {args.phase} winners to {args.winners}")
    for category, value in sorted(winner_map.items()):
        if isinstance(value, list):
            print(f"  {category}: [{', '.join(value)}]")
        else:
            print(f"  {category}: {value}")

    if stats is not None and not stats.empty:
        print(f"\nPhase 1 shortlist threshold: mean rating >= {args.mean_rating_threshold}")
        for category in sorted(stats["category"].unique()):
            group = stats[stats["category"] == category]
            print(f"  {category}:")
            for _, row in group.iterrows():
                flag = "✓" if row["shortlisted"] else " "
                print(
                    f"    {flag} {row['variant_id']}: "
                    f"rating={row['mean_rating']:.2f} "
                    f"(content={row['mean_content']}, realism={row['mean_realism']})"
                )

    catalog = SweepCatalog("patch", sweep_root)
    manifest = catalog._manifest
    if not manifest.empty:
        print("\nResolved settings:")
        for category, value in sorted(winner_map.items()):
            variant_ids = value if isinstance(value, list) else [value]
            for variant_id in variant_ids:
                match_rows = manifest[
                    (manifest["variant_id"] == variant_id)
                    & (manifest["category"] == category)
                ]
                if match_rows.empty:
                    match_rows = manifest[manifest["variant_id"] == variant_id]
                match = match_rows.iloc[0]
                details = [f"variant={variant_id}"]
                if "soundfont_id" in match and str(match["soundfont_id"]):
                    details.append(f"soundfont={match['soundfont_id']}")
                if "fx_profile" in match and str(match["fx_profile"]):
                    details.append(f"fx={match['fx_profile']}")
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
