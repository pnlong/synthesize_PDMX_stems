"""Aggregate sweep listening-test responses into per-category winners."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from experiments.listening.catalog import SweepCatalog, default_sweep_dir
from experiments.paths import DEFAULT_PROBE_STEMS

DEFAULT_CONTENT_THRESHOLD = 3.0
DEFAULT_CONTENT_MEAN_THRESHOLD = 3.5
DEFAULT_MEAN_RATING_THRESHOLD = 4.1
DEFAULT_NOISE_CONTENT_THRESHOLD = 4.5


def load_responses(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def ratings_dataframe(responses: dict) -> pd.DataFrame:
    rows = []
    for entry in responses.get("ratings", []):
        stem_id = entry["stem_id"]
        category = entry.get("category")
        for rating in entry.get("samples", []):
            rows.append({
                "stem_id": stem_id,
                "category": category,
                "variant_id": rating["variant_id"],
                "content": float(rating["content"]),
                "realism": float(rating["realism"]),
            })
    return pd.DataFrame(rows)


def aggregate_winners(
    df: pd.DataFrame,
    *,
    content_threshold: float = DEFAULT_CONTENT_THRESHOLD,
    content_mean_threshold: float = DEFAULT_CONTENT_MEAN_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per_stem_scores, per_category_winners)."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    from experiments.listening.verification import auto_winners, filter_eligible, variant_stats

    per_stem = (
        df.groupby(["stem_id", "category", "variant_id"], dropna=False)
        .agg(content=("content", "mean"), realism=("realism", "mean"))
        .reset_index()
    )

    stats = variant_stats(df)
    eligible = filter_eligible(
        stats,
        content_threshold=content_threshold,
        content_mean_threshold=content_mean_threshold,
    )
    winners = auto_winners(eligible)

    return per_stem, winners


def _noise_level_from_variant_id(variant_id: str) -> float:
    prefix = "noise"
    if str(variant_id).startswith(prefix):
        return float(str(variant_id)[len(prefix):])
    return float("-inf")


def noise_level_winners(
    df: pd.DataFrame,
    *,
    content_threshold: float = DEFAULT_NOISE_CONTENT_THRESHOLD,
) -> pd.DataFrame:
    """Pick phase-1 noise winners: mean content >= threshold, then highest realism.

    Tie-break equal realism by higher init_noise_level.
    """
    if df.empty:
        return pd.DataFrame()

    from experiments.listening.verification import variant_stats

    stats = variant_stats(df)
    if stats.empty:
        return stats

    stats = stats.copy()
    stats["init_noise_level"] = stats["variant_id"].map(_noise_level_from_variant_id)

    def _rank(group: pd.DataFrame) -> pd.DataFrame:
        return group.sort_values(
            ["mean_realism", "init_noise_level", "mean_content", "variant_id"],
            ascending=[False, False, False, True],
        )

    winners = []
    for category, group in stats.groupby("category", dropna=False):
        passing = group[group["mean_content"] >= content_threshold]
        ranked = _rank(passing if not passing.empty else group)
        best = ranked.iloc[0]
        winners.append({
            "category": category,
            "variant_id": best["variant_id"],
            "mean_content": round(float(best["mean_content"]), 2),
            "mean_realism": round(float(best["mean_realism"]), 2),
            "n_stems": int(best["n_stems"]),
            "passed_content_threshold": bool(best["mean_content"] >= content_threshold),
        })
    return pd.DataFrame(winners).sort_values("category")


def noise_level_dataframe(
    df: pd.DataFrame,
    *,
    content_threshold: float = DEFAULT_NOISE_CONTENT_THRESHOLD,
) -> pd.DataFrame:
    """Tabular phase-1 noise stats with threshold flag for reporting."""
    from experiments.listening.verification import variant_stats

    stats = variant_stats(df)
    if stats.empty:
        return stats

    stats = stats.copy()
    stats["passed_content_threshold"] = stats["mean_content"] >= content_threshold
    return stats.sort_values(["category", "mean_realism"], ascending=[True, False])


def noise_audit_candidates(
    phase1_winners: dict[str, str],
) -> dict[str, set[str]]:
    """Per-category variant ids to compare in phase-1b (winner vs one-step-lower)."""
    from experiments.preset_sweep.config import (
        init_noise_level_from_variant_id,
        lower_noise_level,
        noise_variant_id,
    )

    candidates: dict[str, set[str]] = {}
    for category, variant_id in phase1_winners.items():
        winner_level = init_noise_level_from_variant_id(variant_id)
        lower_level = lower_noise_level(winner_level)
        candidates[str(category)] = {
            noise_variant_id(winner_level),
            noise_variant_id(lower_level),
        }
    return candidates


def noise_audit_winners(
    df: pd.DataFrame,
    phase1_winners: dict[str, str],
    *,
    content_threshold: float = DEFAULT_NOISE_CONTENT_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Pick audit winners after production silence enforcement: content gate, then realism."""
    if df.empty:
        return pd.DataFrame(), {}

    from experiments.listening.verification import variant_stats
    from experiments.preset_sweep.config import init_noise_level_from_variant_id

    stats = variant_stats(df)
    if stats.empty:
        return pd.DataFrame(), {}

    stats = stats.copy()
    stats["init_noise_level"] = stats["variant_id"].map(_noise_level_from_variant_id)
    candidates = noise_audit_candidates(phase1_winners)

    winners = []
    revisions: dict[str, str] = {}
    for category, group in stats.groupby("category", dropna=False):
        category = str(category)
        allowed = candidates.get(category)
        if not allowed:
            continue
        subset = group[group["variant_id"].isin(allowed)]
        if subset.empty:
            continue
        passing = subset[subset["mean_content"] >= content_threshold]
        pool = passing if not passing.empty else subset
        pool = pool.sort_values(
            ["mean_realism", "init_noise_level", "mean_content", "variant_id"],
            ascending=[False, True, False, True],
        )
        best = pool.iloc[0]
        winners.append({
            "category": category,
            "variant_id": best["variant_id"],
            "mean_content": round(float(best["mean_content"]), 2),
            "mean_realism": round(float(best["mean_realism"]), 2),
            "n_stems": int(best["n_stems"]),
            "passed_content_threshold": bool(best["mean_content"] >= content_threshold),
        })

        phase1_variant = phase1_winners.get(category)
        if phase1_variant and best["variant_id"] != phase1_variant:
            best_level = init_noise_level_from_variant_id(str(best["variant_id"]))
            phase1_level = init_noise_level_from_variant_id(phase1_variant)
            if best_level < phase1_level:
                revisions[category] = str(best["variant_id"])

    winners_df = pd.DataFrame(winners).sort_values("category") if winners else pd.DataFrame()
    return winners_df, revisions


def shortlist_variants(
    df: pd.DataFrame,
    *,
    mean_rating_threshold: float = DEFAULT_MEAN_RATING_THRESHOLD,
) -> dict[str, list[str]]:
    """Return per-category variant shortlists with mean(content, realism) >= threshold."""
    if df.empty:
        return {}

    from experiments.listening.verification import variant_stats

    stats = variant_stats(df)
    if stats.empty:
        return {}

    stats = stats.copy()
    stats["mean_rating"] = (stats["mean_content"] + stats["mean_realism"]) / 2

    shortlists: dict[str, list[str]] = {}
    for category, group in stats.groupby("category", dropna=False):
        passing = group[group["mean_rating"] >= mean_rating_threshold].sort_values(
            ["mean_rating", "mean_realism", "mean_content"],
            ascending=[False, False, False],
        )
        if passing.empty:
            best = group.sort_values(
                ["mean_rating", "mean_realism", "mean_content"],
                ascending=[False, False, False],
            ).iloc[0]
            passing = group[group["variant_id"] == best["variant_id"]]

        shortlists[str(category)] = [str(v) for v in passing["variant_id"].tolist()]

    return shortlists


def shortlist_dataframe(
    df: pd.DataFrame,
    *,
    mean_rating_threshold: float = DEFAULT_MEAN_RATING_THRESHOLD,
) -> pd.DataFrame:
    """Tabular shortlist with stats for reporting."""
    from experiments.listening.verification import variant_stats

    stats = variant_stats(df)
    if stats.empty:
        return stats

    stats = stats.copy()
    stats["mean_rating"] = (stats["mean_content"] + stats["mean_realism"]) / 2
    stats["shortlisted"] = stats["mean_rating"] >= mean_rating_threshold
    return stats.sort_values(["category", "mean_rating"], ascending=[True, False])


def preset_config_suggestions(
    winners: pd.DataFrame,
    catalog: SweepCatalog,
) -> str:
    if winners.empty or catalog._manifest.empty:
        return ""

    manifest = catalog._manifest
    lines = ["# Suggested categories.yaml overrides", ""]
    for _, row in winners.iterrows():
        variant_id = row["variant_id"]
        match = manifest[manifest["variant_id"] == variant_id].iloc[0]
        lines.append(f"  {row['category']}:")
        lines.append(f"    init_noise_level: {float(match['init_noise_level'])}")
        lines.append(f"    prompt_variant: {match['prompt_variant']}")
        if "steps" in match and pd.notna(match["steps"]):
            lines.append(f"    steps: {int(match['steps'])}")
        if "cfg_scale" in match and pd.notna(match["cfg_scale"]):
            lines.append(f"    cfg_scale: {float(match['cfg_scale'])}")
        lines.append("")
    lines.append("# After all phases: uv run python -m experiments.preset_sweep.lock")
    return "\n".join(lines)


def patch_config_suggestions(winners: pd.DataFrame, catalog: SweepCatalog) -> str:
    if winners.empty or catalog._manifest.empty:
        return ""

    manifest = catalog._manifest
    lines = ["# Suggested production config", ""]
    for _, row in winners.iterrows():
        variant_id = row["variant_id"]
        match = manifest[manifest["variant_id"] == variant_id].iloc[0]
        parts = [f"variant {variant_id}"]
        if "soundfont_id" in match and str(match["soundfont_id"]):
            parts.append(f"soundfont={match['soundfont_id']}")
        if "fx_profile" in match and str(match["fx_profile"]):
            parts.append(f"fx={match['fx_profile']}")
        if "pool_id" in match and str(match["pool_id"]):
            parts.append(f"pool={match['pool_id']}")
        lines.append(f"# {row['category']}: {', '.join(parts)}")
    lines.append("")
    lines.append("# After all phases: uv run python -m experiments.patch_sweep.lock")
    lines.append("# Or merge per-category winners into experiments/patch_sweep/winners.yaml")
    return "\n".join(lines)


def format_results_markdown(
    *,
    sweep_type: str,
    winners: pd.DataFrame,
    catalog: SweepCatalog,
    responses_path: Path,
) -> str:
    lines = [
        "# Sweep Listening Test Results",
        "",
        f"Source: `{responses_path}`",
        f"Sweep: {sweep_type}",
        "",
        "## Per instrument class",
        "",
    ]

    manifest = catalog._manifest

    if sweep_type == "preset":
        has_diffusion = "steps" in manifest.columns and "cfg_scale" in manifest.columns
        if has_diffusion:
            lines.append(
                "| Category | Winner variant_id | init_noise_level | prompt_variant "
                "| steps | cfg_scale | mean_content | mean_realism | Notes |"
            )
            lines.append(
                "|----------|-------------------|------------------|----------------|"
                "-------|-----------|--------------|--------------|-------|"
            )
        else:
            lines.append(
                "| Category | Winner variant_id | init_noise_level | prompt_variant "
                "| mean_content | mean_realism | Notes |"
            )
            lines.append(
                "|----------|-------------------|------------------|----------------|"
                "--------------|--------------|-------|"
            )
        for _, row in winners.iterrows():
            match = manifest[manifest["variant_id"] == row["variant_id"]].iloc[0]
            if has_diffusion:
                lines.append(
                    f"| {row['category']} | {row['variant_id']} | "
                    f"{float(match['init_noise_level'])} | {match['prompt_variant']} | "
                    f"{int(match['steps'])} | {float(match['cfg_scale'])} | "
                    f"{row['mean_content']} | {row['mean_realism']} | |"
                )
            else:
                lines.append(
                    f"| {row['category']} | {row['variant_id']} | "
                    f"{float(match['init_noise_level'])} | {match['prompt_variant']} | "
                    f"{row['mean_content']} | {row['mean_realism']} | |"
                )
    else:
        has_sf = "soundfont_id" in manifest.columns
        has_fx = "fx_profile" in manifest.columns
        if has_sf and has_fx:
            header = (
                "| Category | Winner variant_id | soundfont_id | fx_profile | pool_id "
                "| mean_content | mean_realism | Notes |"
            )
            sep = (
                "|----------|-------------------|--------------|------------|---------|"
                "--------------|--------------|-------|"
            )
        else:
            header = (
                "| Category | Winner variant_id | pool_id | mean_content | mean_realism | Notes |"
            )
            sep = (
                "|----------|-------------------|---------|--------------|--------------|-------|"
            )
        lines.append(header)
        lines.append(sep)
        for _, row in winners.iterrows():
            match = manifest[manifest["variant_id"] == row["variant_id"]].iloc[0]
            if has_sf and has_fx:
                lines.append(
                    f"| {row['category']} | {row['variant_id']} | "
                    f"{match.get('soundfont_id', '')} | {match.get('fx_profile', '')} | "
                    f"{match.get('pool_id', '')} | "
                    f"{row['mean_content']} | {row['mean_realism']} | |"
                )
            else:
                lines.append(
                    f"| {row['category']} | {row['variant_id']} | {match['pool_id']} | "
                    f"{row['mean_content']} | {row['mean_realism']} | |"
                )

    lines.extend(["", "## Config suggestions", ""])
    if sweep_type == "preset":
        lines.append("```yaml")
        lines.append(preset_config_suggestions(winners, catalog).strip())
        lines.append("```")
    else:
        lines.append("```")
        lines.append(patch_config_suggestions(winners, catalog).strip())
        lines.append("```")

    return "\n".join(lines) + "\n"


def run_aggregate(
    *,
    sweep_type: str,
    responses_path: Path,
    output_path: Path | None,
    sweep_dir: Path | None,
    content_threshold: float,
    content_mean_threshold: float,
) -> pd.DataFrame:
    responses = load_responses(responses_path)
    df = ratings_dataframe(responses)
    _, winners = aggregate_winners(
        df,
        content_threshold=content_threshold,
        content_mean_threshold=content_mean_threshold,
    )

    catalog = SweepCatalog(
        sweep_type,
        sweep_dir or default_sweep_dir(sweep_type),
        probe_stems_path=DEFAULT_PROBE_STEMS,
    )

    markdown = format_results_markdown(
        sweep_type=sweep_type,
        winners=winners,
        catalog=catalog,
        responses_path=responses_path,
    )

    print(markdown)
    if output_path is not None:
        output_path.write_text(markdown)

    return winners


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="Aggregate sweep listening-test responses.",
    )
    parser.add_argument(
        "--sweep",
        required=True,
        choices=["preset", "patch"],
        help="Sweep type to aggregate.",
    )
    parser.add_argument(
        "--responses",
        required=True,
        type=Path,
        help="Exported responses JSON from listening test.",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Write results markdown (e.g. experiments/preset_sweep/results_notes.md).",
    )
    parser.add_argument(
        "--sweep-dir",
        default=None,
        type=Path,
        help="Sweep output root (default: experiments/*/output symlink).",
    )
    parser.add_argument(
        "--content-threshold",
        default=DEFAULT_CONTENT_THRESHOLD,
        type=float,
        help="Drop variants with min content below this on any stem.",
    )
    parser.add_argument(
        "--content-mean-threshold",
        default=DEFAULT_CONTENT_MEAN_THRESHOLD,
        type=float,
        help="Drop variants with mean content below this per category.",
    )
    return parser.parse_args(args=args, namespace=namespace)


def main():
    args = parse_args()
    run_aggregate(
        sweep_type=args.sweep,
        responses_path=args.responses,
        output_path=args.output,
        sweep_dir=args.sweep_dir,
        content_threshold=args.content_threshold,
        content_mean_threshold=args.content_mean_threshold,
    )


if __name__ == "__main__":
    main()
