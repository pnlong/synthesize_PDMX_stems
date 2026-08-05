"""Aggregate webMUSHRA mushra.csv results for ablation listening."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import yaml

from experiments.ablation_listening.aggregate import factorial_table, render_markdown
from experiments.ablation_listening.conditions import (
    ABLATION_MUSHRA_CONDITIONS,
    CONDITION_LABELS,
    RATING_SCALES,
    REFERENCE_CONDITION,
    STEM_TRIAL_CATEGORIES,
    category_from_trial_id,
    parse_mushra_page_id,
)
from experiments.ablation_listening.equivalence import equivalences_by_trial_id
from experiments.ablation_listening.paths import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WEBMUSHRA_ROOT,
    WEBMUSHRA_TEST_ID,
)

CONDITION_ALIASES = {
    "a1": "basic",
    "a2": "basic_realify",
    "b1": "slakh",
    "b2": "slakh_realify",
    "ca1": "ddsp_basic",
    "ca2": "ddsp_basic_realify",
    "cb1": "ddsp_slakh",
    "cb2": "ddsp_slakh_realify",
    "basic": "basic",
    "basic_realify": "basic_realify",
    "slakh": "slakh",
    "slakh_realify": "slakh_realify",
    "ddsp_basic": "ddsp_basic",
    "ddsp_basic_realify": "ddsp_basic_realify",
    "ddsp_slakh": "ddsp_slakh",
    "ddsp_slakh_realify": "ddsp_slakh_realify",
    # legacy dir name before rename
    "slakh_ddsp": "ddsp_slakh",
    "slakh_ddsp_realify": "ddsp_slakh_realify",
}


def normalize_condition(stimulus: str) -> str | None:
    key = str(stimulus).strip().lower()
    return CONDITION_ALIASES.get(key)


def load_manifest_equivalences(manifest_path: Path | None) -> dict[str, dict[str, str]]:
    if manifest_path is None or not Path(manifest_path).is_file():
        return {}
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f) or {}
    return equivalences_by_trial_id(manifest)


def expand_equivalence_scores(
    df: pd.DataFrame,
    equivalences_by_trial: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Synthesize ratings for omitted donor-copy conditions from rated donors."""
    if df.empty or not equivalences_by_trial:
        if "auto_assigned" not in df.columns:
            df = df.copy()
            df["auto_assigned"] = False
            df["source_condition"] = None
        return df

    df = df.copy()
    if "auto_assigned" not in df.columns:
        df["auto_assigned"] = False
    if "source_condition" not in df.columns:
        df["source_condition"] = None

    # Avoid double-inserting if a duplicate was somehow already rated.
    existing = {
        (row.listener_id, row.trial_id, row.scale, row.condition_id)
        for row in df.itertuples(index=False)
    }

    extra: list[dict] = []
    for row in df.itertuples(index=False):
        if bool(getattr(row, "auto_assigned", False)):
            continue
        equiv = equivalences_by_trial.get(str(row.trial_id)) or {}
        for duplicate, donor in equiv.items():
            if row.condition_id != donor:
                continue
            key = (row.listener_id, row.trial_id, row.scale, duplicate)
            if key in existing:
                continue
            existing.add(key)
            extra.append({
                "listener_id": row.listener_id,
                "page_id": row.page_id,
                "trial_id": row.trial_id,
                "scale": row.scale,
                "category": row.category,
                "trial_type": row.trial_type,
                "condition_id": duplicate,
                "score": row.score,
                "auto_assigned": True,
                "source_condition": donor,
            })

    if not extra:
        return df
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)


def load_mushra_csv(
    path: Path,
    *,
    manifest_path: Path | None = None,
    equivalences_by_trial: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Load ratings; parse ``trial_id__scale`` page ids into trial + scale columns.

    When trial equivalences are available (manifest or explicit map), omitted
    donor-copy conditions receive auto-assigned scores matching their donor.
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            condition_id = normalize_condition(row.get("rating_stimulus", ""))
            if condition_id is None:
                continue
            try:
                score = float(row["rating_score"])
            except (KeyError, TypeError, ValueError):
                continue

            page_id = row.get("trial_id") or ""
            trial_id, scale = parse_mushra_page_id(page_id)
            # Legacy single-scale BAQ exports → treat as realism.
            if scale is None:
                scale = "realism"
            category = category_from_trial_id(trial_id)
            trial_type = (
                "mixture" if str(trial_id).startswith("mix_")
                else "stem" if str(trial_id).startswith("stem_")
                else None
            )
            rows.append({
                "listener_id": row.get("listener_id") or row.get("session_uuid"),
                "page_id": page_id,
                "trial_id": trial_id,
                "scale": scale,
                "category": category,
                "trial_type": trial_type,
                "condition_id": condition_id,
                "score": score,
                "auto_assigned": False,
                "source_condition": None,
            })
    df = pd.DataFrame(rows)
    if equivalences_by_trial is None:
        equivalences_by_trial = load_manifest_equivalences(manifest_path)
    return expand_equivalence_scores(df, equivalences_by_trial)


def _means_dict(series: pd.Series) -> dict[str, float]:
    out = {}
    for cond in ABLATION_MUSHRA_CONDITIONS:
        if cond in series.index and pd.notna(series.loc[cond]):
            out[cond] = round(float(series.loc[cond]), 2)
    return out


def summarize_mushra(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"error": "no ratings"}

    content_df = df[df["scale"] == "content"]
    realism_df = df[df["scale"] == "realism"]

    content_means = (
        content_df.groupby("condition_id")["score"].mean().reindex(ABLATION_MUSHRA_CONDITIONS)
        if not content_df.empty else pd.Series(dtype=float)
    )
    realism_means = (
        realism_df.groupby("condition_id")["score"].mean().reindex(ABLATION_MUSHRA_CONDITIONS)
        if not realism_df.empty else pd.Series(dtype=float)
    )

    means_df = pd.DataFrame({
        "content": content_means,
        "realism": realism_means,
    }, index=list(ABLATION_MUSHRA_CONDITIONS))

    # Hidden reference: content ratings of A1 are usually near ceiling; keep them
    # but mark reference in reporting.
    winner = None
    if not realism_means.dropna().empty:
        # Prefer high realism among non-reference with decent content when available.
        variants = means_df.drop(index=REFERENCE_CONDITION, errors="ignore")
        if not variants.empty and variants["realism"].notna().any():
            if variants["content"].notna().any():
                eligible = variants[variants["content"].fillna(0) >= 40]
                if eligible.empty or eligible["realism"].isna().all():
                    eligible = variants
            else:
                eligible = variants
            winner = eligible["realism"].idxmax()

    means_by_condition = {}
    for cond in ABLATION_MUSHRA_CONDITIONS:
        c_val = means_df.loc[cond, "content"] if cond in means_df.index else float("nan")
        r_val = means_df.loc[cond, "realism"] if cond in means_df.index else float("nan")
        means_by_condition[cond] = {
            "content": None if pd.isna(c_val) else round(float(c_val), 2),
            "realism": None if pd.isna(r_val) else round(float(r_val), 2),
            "content_band": (
                "— (reference)" if cond == REFERENCE_CONDITION and pd.isna(c_val)
                else (f"{round(float(c_val), 1)}" if pd.notna(c_val) else "—")
            ),
            "realism_band": f"{round(float(r_val), 1)}" if pd.notna(r_val) else "—",
        }

    # Per-category × condition × scale
    by_category: dict[str, dict] = {}
    stem_df = df[df["trial_type"] == "stem"]
    for category in STEM_TRIAL_CATEGORIES:
        cat_df = stem_df[stem_df["category"] == category]
        if cat_df.empty:
            continue
        entry: dict = {}
        for scale in RATING_SCALES:
            scale_df = cat_df[cat_df["scale"] == scale]
            if scale_df.empty:
                continue
            entry[scale] = _means_dict(
                scale_df.groupby("condition_id")["score"].mean().reindex(ABLATION_MUSHRA_CONDITIONS)
            )
        if entry:
            by_category[category] = entry

    mix_df = df[df["trial_type"] == "mixture"]

    return {
        "n_ratings": int(len(df)),
        "n_listeners": int(df["listener_id"].nunique(dropna=True)),
        "winner": winner,
        "means_by_condition": means_by_condition,
        "by_category": by_category,
        "mixture_means": {
            scale: _means_dict(
                mix_df[mix_df["scale"] == scale]
                .groupby("condition_id")["score"]
                .mean()
                .reindex(ABLATION_MUSHRA_CONDITIONS)
            )
            for scale in RATING_SCALES
            if not mix_df[mix_df["scale"] == scale].empty
        },
        "stem_means": {
            scale: _means_dict(
                stem_df[stem_df["scale"] == scale]
                .groupby("condition_id")["score"]
                .mean()
                .reindex(ABLATION_MUSHRA_CONDITIONS)
            )
            for scale in RATING_SCALES
            if not stem_df[stem_df["scale"] == scale].empty
        },
        "factorial_content": (
            factorial_table(means_df, "content").to_dict(orient="index")
            if content_means.notna().any() else {}
        ),
        "factorial_realism": (
            factorial_table(means_df, "realism").to_dict(orient="index")
            if realism_means.notna().any() else {}
        ),
    }


def render_mushra_markdown(summary: dict, *, responses_path: Path) -> str:
    """Markdown with overall + per-category tables."""
    base = render_markdown(summary, responses_path=responses_path)
    by_category = summary.get("by_category") or {}
    if not by_category:
        return base

    lines = [base.rstrip(), "", "## Per-category means (stem trials)", ""]
    for category in STEM_TRIAL_CATEGORIES:
        entry = by_category.get(category)
        if not entry:
            continue
        lines.append(f"### {category}")
        lines.append("")
        lines.append("| Condition | Content | Realism |")
        lines.append("|-----------|---------|---------|")
        for cond in ABLATION_MUSHRA_CONDITIONS:
            label = CONDITION_LABELS.get(cond, cond)
            c = (entry.get("content") or {}).get(cond, "—")
            r = (entry.get("realism") or {}).get(cond, "—")
            lines.append(f"| {label} | {c} | {r} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def default_results_csv(webmushra_root: Path, test_id: str = WEBMUSHRA_TEST_ID) -> Path:
    return webmushra_root / "results" / test_id / "mushra.csv"


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Aggregate webMUSHRA mushra.csv results.")
    parser.add_argument(
        "--results",
        type=Path,
        help="Path to mushra.csv (default: webMUSHRA/results/<test_id>/mushra.csv).",
    )
    parser.add_argument("--webmushra-root", default=DEFAULT_WEBMUSHRA_ROOT, type=Path)
    parser.add_argument("--test-id", default=WEBMUSHRA_TEST_ID)
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        type=Path,
        help=(
            "Trial manifest with per-trial equivalences; omitted donor-copy "
            "conditions inherit the donor's score."
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR / "results_notes_webmushra.md",
        type=Path,
    )
    return parser.parse_args(args)


def main(args=None) -> None:
    opts = parse_args(args)
    results_path = opts.results or default_results_csv(opts.webmushra_root, opts.test_id)
    if not results_path.is_file():
        raise FileNotFoundError(f"No results at {results_path}")

    df = load_mushra_csv(results_path, manifest_path=opts.manifest)
    summary = summarize_mushra(df)
    n_auto = int(df["auto_assigned"].sum()) if "auto_assigned" in df.columns else 0
    summary["n_auto_assigned"] = n_auto

    opts.output.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_mushra_markdown(summary, responses_path=results_path)
    opts.output.write_text(markdown)
    json_path = opts.output.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2))
    print(markdown)
    print(f"Wrote {opts.output}")
    print(f"Wrote {json_path}")
    if n_auto:
        print(f"Auto-assigned {n_auto} ratings from donor equivalences")


if __name__ == "__main__":
    main()
