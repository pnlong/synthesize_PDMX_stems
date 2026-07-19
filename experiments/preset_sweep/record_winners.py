"""Record per-category winners from a phase listening test into winners.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.listening.aggregate import (
    DEFAULT_NOISE_CONTENT_THRESHOLD,
    aggregate_winners,
    load_responses,
    noise_audit_winners,
    noise_level_dataframe,
    noise_level_winners,
    ratings_dataframe,
)
from experiments.listening.catalog import SweepCatalog
from experiments.preset_sweep.config import EXPERIMENT_DIR, PHASE1, PHASE1B, PHASES, phase_output_dir
from experiments.preset_sweep.sweep import default_output_dir
from experiments.preset_sweep.winners import (
    load_winners,
    phase_winners,
    record_phase_winners,
    save_winners,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Record blinded listening winners for a preset sweep phase.",
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
        "--noise-content-threshold",
        default=DEFAULT_NOISE_CONTENT_THRESHOLD,
        type=float,
        help="Phase 1 / 1b: content threshold for reporting and audit gating.",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    sweep_root = args.sweep_dir or phase_output_dir(default_output_dir(), args.phase)

    responses = load_responses(args.responses)
    df = ratings_dataframe(responses)

    if args.phase == PHASE1:
        winners_df = noise_level_winners(
            df,
            content_threshold=args.noise_content_threshold,
        )
        stats = noise_level_dataframe(
            df,
            content_threshold=args.noise_content_threshold,
        )
        revisions = {}
    elif args.phase == PHASE1B:
        phase1_winners = phase_winners(PHASE1, args.winners)
        winners_df, revisions = noise_audit_winners(
            df,
            phase1_winners,
            content_threshold=args.noise_content_threshold,
        )
        stats = noise_level_dataframe(
            df,
            content_threshold=args.noise_content_threshold,
        )
    else:
        _, winners_df = aggregate_winners(df)
        stats = None
        revisions = {}

    if winners_df.empty:
        raise RuntimeError("No winners produced from responses.")

    winner_map = {
        str(row["category"]): str(row["variant_id"])
        for _, row in winners_df.iterrows()
    }

    doc = record_phase_winners(args.phase, winner_map, path=args.winners)

    if revisions:
        doc = load_winners(args.winners)
        phase1_doc = doc["phases"].setdefault(PHASE1, {"completed": False, "winners": {}})
        phase1_doc["winners"].update(revisions)
        phase1_doc["revised_from_phase1b_at"] = doc["phases"][PHASE1B].get("recorded_at")
        save_winners(doc, args.winners)
        print("\nLowered phase-1 noise from audit:")
        for category, variant_id in sorted(revisions.items()):
            print(f"  {category}: {variant_id}")

    print(f"Recorded {args.phase} winners to {args.winners}")
    for category, variant_id in sorted(winner_map.items()):
        print(f"  {category}: {variant_id}")

    if args.phase in (PHASE1, PHASE1B) and stats is not None and not stats.empty:
        label = "Phase 1" if args.phase == PHASE1 else "Phase 1b audit"
        rule = (
            f"mean content >= {args.noise_content_threshold}, then highest realism "
            f"(ties → higher noise)"
            if args.phase == PHASE1
            else f"highest realism among phase-1 winner vs one-step-lower after "
            f"content >= {args.noise_content_threshold} "
            f"(ties → lower noise; silence enforcement applied at render); "
            f"may revise phase-1 winners"
        )
        print(f"\n{label} rule: {rule}")
        for category in sorted(stats["category"].unique()):
            group = stats[stats["category"] == category]
            winner_row = winners_df[winners_df["category"] == category].iloc[0]
            print(f"  {category}:")
            for _, row in group.iterrows():
                flag = "✓" if row["passed_content_threshold"] else " "
                picked = "← winner" if row["variant_id"] == winner_row["variant_id"] else ""
                print(
                    f"    {flag} {row['variant_id']}: "
                    f"content={row['mean_content']:.2f} realism={row['mean_realism']:.2f} "
                    f"{picked}"
                )
            if not winner_row["passed_content_threshold"]:
                print(
                    f"    (no variant met content >= {args.noise_content_threshold}; "
                    f"fell back to highest realism)"
                )

    catalog = SweepCatalog("preset", sweep_root)
    manifest = catalog._manifest
    if not manifest.empty:
        print("\nResolved settings:")
        for category, variant_id in sorted(winner_map.items()):
            match_rows = manifest[
                (manifest["variant_id"] == variant_id)
                & (manifest["category"] == category)
            ]
            if match_rows.empty:
                match_rows = manifest[manifest["variant_id"] == variant_id]
            match = match_rows.iloc[0]
            details = [
                f"variant={variant_id}",
                f"noise={float(match['init_noise_level'])}",
                f"prompt={match['prompt_variant']}",
            ]
            if "steps" in match and str(match["steps"]):
                details.append(f"steps={int(match['steps'])}")
            if "cfg_scale" in match and str(match["cfg_scale"]):
                details.append(f"cfg={float(match['cfg_scale'])}")
            print(f"  {category}: {', '.join(details)}")

    incomplete = [
        phase for phase in PHASES
        if not doc["phases"].get(phase, {}).get("completed")
    ]
    if incomplete:
        if len(incomplete) == 1 and incomplete[0] == "phase3_diffusion":
            print(
                "\nPhases 1–2 complete. Optional: run phase 3 diffusion sweep, "
                "or lock now with default steps/cfg."
            )
            print("Lock: uv run python -m experiments.preset_sweep.lock")
        elif PHASE1B in incomplete and PHASE2 in incomplete:
            print(f"\nNext: complete {PHASE1B} noise audit, then phase 2.")
        else:
            print(f"\nRemaining phases: {', '.join(incomplete)}")
    else:
        print("\nAll phases complete. Run: uv run python -m experiments.preset_sweep.lock")


if __name__ == "__main__":
    main()
