"""Record per-category winners from a phase listening test into winners.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.listening.aggregate import (
    DEFAULT_MEAN_RATING_THRESHOLD,
    DEFAULT_MIN_SWIPE_WINNERS,
    DEFAULT_REALISM_THRESHOLD,
    aggregate_winners,
    load_responses,
    ratings_dataframe,
    shortlist_dataframe,
    shortlist_variants,
    swipe_winners,
)
from experiments.listening.catalog import SweepCatalog
from experiments.listening.swipe import (
    filter_swipe_votes_to_manifest,
    sanitize_cross_variant_cascade_votes,
)
from experiments.patch_sweep.config import (
    EXPERIMENT_DIR,
    PHASE1,
    PHASE1_ARCHIVE,
    PHASE2,
    PHASES,
    phase_output_dir,
)
from experiments.patch_sweep.sweep import default_output_dir
from experiments.patch_sweep.winners import record_phase_winners


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Record blinded listening winners for a patch sweep phase.",
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=list(PHASES) + [PHASE1_ARCHIVE],
        help="Phase whose winners to record.",
    )
    parser.add_argument(
        "--responses",
        required=True,
        type=Path,
        help="Exported responses JSON from listening test.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=("swipe", "rating"),
        help="Response mode (default: swipe for archive phase 1).",
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
        help="Legacy phase 1: include soundfonts with mean(content, realism)/2 >= this.",
    )
    parser.add_argument(
        "--realism-threshold",
        default=DEFAULT_REALISM_THRESHOLD,
        type=float,
        help="Phase 1 rating mode: include soundfonts with mean realism >= this.",
    )
    parser.add_argument(
        "--min-winners",
        default=DEFAULT_MIN_SWIPE_WINNERS,
        type=int,
        help="Swipe mode: max winners when falling back to weak accepts.",
    )
    parser.add_argument(
        "--allow-reject-fallback",
        action="store_true",
        help="Swipe mode: allow weak/strong reject tiers when no accepts exist.",
    )
    parser.add_argument(
        "--use-mean-rating",
        action="store_true",
        help="Rating mode: use mean(content, realism)/2 threshold instead of realism-only.",
    )
    return parser.parse_args(args)


def _winner_phase_key(phase: str) -> str:
    """Archive phase 1 results are stored under the canonical phase1 key."""
    if phase == PHASE1_ARCHIVE:
        return PHASE1
    return phase


def _default_mode(phase: str, responses: dict) -> str:
    if responses.get("mode") == "swipe":
        return "swipe"
    if phase in (PHASE1_ARCHIVE, PHASE2):
        return "swipe"
    return "rating"


def main():
    args = parse_args()
    sweep_phase = args.phase
    winner_phase = _winner_phase_key(sweep_phase)
    sweep_root = args.sweep_dir or phase_output_dir(default_output_dir(), sweep_phase)

    responses = load_responses(args.responses)
    mode = args.mode or _default_mode(sweep_phase, responses)
    stats = None

    catalog = SweepCatalog("patch", sweep_root)
    manifest = catalog._clip_manifest if not catalog._clip_manifest.empty else catalog._manifest
    responses, dropped_votes = filter_swipe_votes_to_manifest(responses, manifest)
    if dropped_votes:
        print(
            f"Ignored {dropped_votes} votes that do not match this phase's manifest "
            f"({len(responses.get('votes') or [])} kept)."
        )

    votes = list(responses.get("votes") or [])
    votes, dropped_cascades = sanitize_cross_variant_cascade_votes(votes)
    if dropped_cascades:
        print(
            f"Ignored {dropped_cascades} cross-variant cascade votes "
            f"({len(votes)} kept)."
        )
        responses = {**responses, "votes": votes}

    if mode == "swipe":
        winner_map = swipe_winners(
            responses,
            min_winners=args.min_winners,
            strict=not args.allow_reject_fallback,
        )
        if not winner_map:
            raise RuntimeError(f"No {winner_phase} swipe winners produced from responses.")
    elif winner_phase == PHASE1:
        df = ratings_dataframe(responses)
        realism_threshold = None if args.use_mean_rating else args.realism_threshold
        winner_map = shortlist_variants(
            df,
            mean_rating_threshold=args.mean_rating_threshold,
            realism_threshold=realism_threshold,
        )
        if not winner_map:
            raise RuntimeError("No phase-1 soundfont shortlists produced from responses.")
        stats = shortlist_dataframe(
            df,
            mean_rating_threshold=args.mean_rating_threshold,
            realism_threshold=realism_threshold,
        )
    else:
        df = ratings_dataframe(responses)
        _, winners_df = aggregate_winners(df)
        if winners_df.empty:
            raise RuntimeError("No winners produced from responses.")
        winner_map = {
            str(row["category"]): str(row["variant_id"])
            for _, row in winners_df.iterrows()
        }

    doc = record_phase_winners(winner_phase, winner_map, path=args.winners)

    print(f"Recorded {winner_phase} winners to {args.winners} ({mode} mode)")
    if sweep_phase != winner_phase:
        print(f"  (from sweep phase {sweep_phase})")
    for category, value in sorted(winner_map.items()):
        if isinstance(value, list):
            print(f"  {category}: [{', '.join(value)}]")
        else:
            print(f"  {category}: {value}")

    if stats is not None and not stats.empty:
        if args.use_mean_rating:
            print(f"\nPhase 1 shortlist threshold: mean rating >= {args.mean_rating_threshold}")
        else:
            print(f"\nPhase 1 shortlist threshold: mean realism >= {args.realism_threshold}")
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
                if match_rows.empty:
                    print(f"  {category}: variant={variant_id} (not in manifest)")
                    continue
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
