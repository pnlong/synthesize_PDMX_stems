"""Aggregate ablation listening test responses."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from experiments.ablation_listening.equivalence import (
    expand_equivalence_scores,
    load_manifest_equivalences,
)
from experiments.ablation_listening.paths import DEFAULT_MANIFEST, DEFAULT_OUTPUT_DIR
from experiments.ablation_listening.session import REFERENCE_CONDITION
from experiments.listening_shared.scale import (
    DEFAULT_CONTENT_MEAN_THRESHOLD,
    DEFAULT_CONTENT_THRESHOLD,
    band_index,
    band_label,
    likert_equivalent,
)
from synthesis.listening.catalog import CONDITION_LABELS, CONDITION_ORDER

# 4×2: render family × realify (covers A/B/CA/CB).
FACTORIAL_ROWS = {
    "basic": "basic",
    "basic_realify": "basic",
    "slakh": "slakh",
    "slakh_realify": "slakh",
    "ddsp_basic": "ddsp_basic",
    "ddsp_basic_realify": "ddsp_basic",
    "ddsp_slakh": "ddsp_slakh",
    "ddsp_slakh_realify": "ddsp_slakh",
}
FACTORIAL_COLS = {
    "basic": "synthetic",
    "slakh": "synthetic",
    "ddsp_basic": "synthetic",
    "ddsp_slakh": "synthetic",
    "basic_realify": "realified",
    "slakh_realify": "realified",
    "ddsp_basic_realify": "realified",
    "ddsp_slakh_realify": "realified",
}
FACTORIAL_ROW_ORDER = ("basic", "slakh", "ddsp_basic", "ddsp_slakh")


def load_responses(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def is_completed_responses_file(path: Path) -> bool:
    """True for finished exports; false for in-progress checkpoints."""
    name = Path(path).name
    if "in_progress" in name:
        return False
    return name.endswith(".json")


def resolve_completed_response_paths(paths: list[Path]) -> list[Path]:
    """Expand files/dirs and drop in-progress checkpoint JSONs."""
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            candidates = sorted(path.glob("responses_*.json")) + sorted(
                path.glob("responses.json")
            )
        elif path.exists():
            candidates = [path]
        else:
            candidates = []
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            if not is_completed_responses_file(candidate):
                continue
            # Prefer an explicit complete flag when present.
            try:
                doc = load_responses(candidate)
            except (OSError, json.JSONDecodeError):
                continue
            if doc.get("complete") is False:
                continue
            seen.add(candidate)
            resolved.append(candidate)
    return resolved


def ratings_dataframe(
    responses: dict,
    *,
    equivalences_by_trial: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    rows = []
    listener_id = responses.get("listener_id")
    for entry in responses.get("ratings", []):
        trial_id = entry["trial_id"]
        trial_type = entry.get("trial_type")
        category = entry.get("category")
        for sample in entry.get("samples", []):
            is_reference = bool(sample.get("is_reference"))
            content = sample.get("content")
            rows.append({
                "listener_id": listener_id,
                "trial_id": trial_id,
                "trial_type": trial_type,
                "category": category,
                "condition_id": sample["condition_id"],
                "condition_label": CONDITION_LABELS.get(
                    sample["condition_id"],
                    sample["condition_id"],
                ),
                "is_reference": is_reference,
                "content": float(content) if content is not None else float("nan"),
                "realism": float(sample["realism"]),
                "auto_assigned": False,
                "source_condition": None,
            })
    df = pd.DataFrame(rows)
    if equivalences_by_trial:
        df = expand_equivalence_scores(
            df,
            equivalences_by_trial,
            score_columns=("content", "realism"),
            scale_key=None,
        )
    return df


def content_filter(
    stats: pd.DataFrame,
    *,
    content_threshold: float = DEFAULT_CONTENT_THRESHOLD,
    content_mean_threshold: float = DEFAULT_CONTENT_MEAN_THRESHOLD,
) -> pd.DataFrame:
    if stats.empty:
        return stats
    best_content = stats["content"].max()
    floor = max(content_threshold, best_content - 20)
    eligible = stats[stats["content"] >= floor].copy()
    if eligible.empty:
        eligible = stats[stats["content"] >= content_mean_threshold].copy()
    if eligible.empty:
        eligible = stats.sort_values("content", ascending=False).head(1)
    return eligible


def pick_winner(stats: pd.DataFrame) -> str | None:
    if stats.empty:
        return None
    variants = stats[stats.index != REFERENCE_CONDITION].copy()
    if variants.empty:
        variants = stats
    eligible = content_filter(variants)
    winner_row = eligible.sort_values(
        ["realism", "content"],
        ascending=False,
    ).iloc[0]
    return str(winner_row.name if hasattr(winner_row, "name") else winner_row["condition_id"])


def band_breakdown(df: pd.DataFrame, field: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    counts = defaultdict(int)
    for value in df[field]:
        counts[band_index(value)] += 1
    total = sum(counts.values())
    rows = []
    for idx in range(5):
        count = counts.get(idx, 0)
        rows.append({
            "band": f"{idx * 20}–{(idx + 1) * 20 if idx < 4 else 100}",
            "count": count,
            "pct": round(100.0 * count / total, 1) if total else 0.0,
        })
    return pd.DataFrame(rows)


def factorial_table(means: pd.DataFrame, field: str) -> pd.DataFrame:
    table = pd.DataFrame(
        index=list(FACTORIAL_ROW_ORDER),
        columns=["synthetic", "realified"],
        dtype=float,
    )
    for condition_id, row in means.iterrows():
        value = row[field]
        if pd.isna(value):
            continue
        if condition_id not in FACTORIAL_ROWS:
            continue
        r = FACTORIAL_ROWS[condition_id]
        c = FACTORIAL_COLS[condition_id]
        table.loc[r, c] = round(float(value), 2)
    return table


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"error": "no ratings"}

    means = (
        df.groupby("condition_id")[["content", "realism"]]
        .mean()
        .reindex(CONDITION_ORDER)
    )
    content_means = (
        df[~df["is_reference"]]
        .groupby("condition_id")[["content"]]
        .mean()
        .reindex(CONDITION_ORDER)
    )
    for cond in CONDITION_ORDER:
        if cond in content_means.index and pd.notna(content_means.loc[cond, "content"]):
            means.loc[cond, "content"] = content_means.loc[cond, "content"]
        elif cond == REFERENCE_CONDITION:
            means.loc[cond, "content"] = float("nan")
    means["content_likert"] = means["content"].apply(
        lambda v: likert_equivalent(v) if pd.notna(v) else None
    )
    means["realism_likert"] = means["realism"].apply(
        lambda v: likert_equivalent(v) if pd.notna(v) else None
    )

    means_by_condition = {}
    for cond, row in means.iterrows():
        entry = {
            "realism": round(float(row["realism"]), 2) if pd.notna(row["realism"]) else None,
            "realism_band": band_label(row["realism"], "realism") if pd.notna(row["realism"]) else None,
        }
        if pd.notna(row["content"]):
            entry["content"] = round(float(row["content"]), 2)
            entry["content_band"] = band_label(row["content"], "content")
        elif cond == REFERENCE_CONDITION:
            entry["content"] = None
            entry["content_band"] = "— (reference)"
        means_by_condition[cond] = entry

    mix_df = df[df["trial_type"] == "mixture"]
    stem_df = df[df["trial_type"] == "stem"]

    return {
        "n_ratings": int(len(df)),
        "n_listeners": int(df["listener_id"].nunique(dropna=True)),
        "winner": pick_winner(means),
        "means_by_condition": means_by_condition,
        "mixture_means": (
            mix_df.groupby("condition_id")[["content", "realism"]].mean().round(2).to_dict()
            if not mix_df.empty else {}
        ),
        "stem_means": (
            stem_df.groupby("condition_id")[["content", "realism"]].mean().round(2).to_dict()
            if not stem_df.empty else {}
        ),
        "factorial_content": factorial_table(means, "content").to_dict(orient="index"),
        "factorial_realism": factorial_table(means, "realism").to_dict(orient="index"),
    }


def render_markdown(summary: dict, *, responses_path: Path) -> str:
    lines = [
        "# Ablation Listening Test Results",
        "",
        f"Responses: `{responses_path}`",
        "",
    ]
    if "error" in summary:
        lines.append(f"Error: {summary['error']}")
        return "\n".join(lines)

    winner = summary.get("winner")
    winner_label = CONDITION_LABELS.get(winner, winner)
    lines.extend([
        f"**Winner:** {winner_label} (`{winner}`)",
        "",
        "## Means by condition (0–100)",
        "",
        "| Condition | Content | Realism | Content band | Realism band |",
        "|-----------|---------|---------|--------------|--------------|",
    ])
    for cond, stats in summary.get("means_by_condition", {}).items():
        label = CONDITION_LABELS.get(cond, cond)
        content_val = stats.get("content")
        content_str = "— (reference)" if content_val is None and cond == REFERENCE_CONDITION else (
            str(content_val) if content_val is not None else "—"
        )
        content_band = stats.get("content_band") or "—"
        lines.append(
            f"| {label} | {content_str} | {stats['realism']} | "
            f"{content_band} | {stats['realism_band']} |"
        )

    lines.extend(["", "## 4×2 factorial (content)", ""])
    fc = summary.get("factorial_content", {})
    lines.append("| | Synthetic | Realified |")
    lines.append("|--|-----------|-----------|")
    for row in FACTORIAL_ROW_ORDER:
        lines.append(
            f"| {row} | {fc.get(row, {}).get('synthetic', '—')} | "
            f"{fc.get(row, {}).get('realified', '—')} |"
        )

    lines.extend(["", "## 4×2 factorial (realism)", ""])
    fr = summary.get("factorial_realism", {})
    lines.append("| | Synthetic | Realified |")
    lines.append("|--|-----------|-----------|")
    for row in FACTORIAL_ROW_ORDER:
        lines.append(
            f"| {row} | {fr.get(row, {}).get('synthetic', '—')} | "
            f"{fr.get(row, {}).get('realified', '—')} |"
        )
    return "\n".join(lines) + "\n"


def aggregate_responses(
    paths: list[Path],
    *,
    manifest_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    completed = resolve_completed_response_paths(list(paths))
    if not completed:
        return pd.DataFrame(), {"error": "no completed response files"}
    equivalences = load_manifest_equivalences(manifest_path)
    frames = []
    for path in completed:
        df = ratings_dataframe(
            load_responses(path),
            equivalences_by_trial=equivalences,
        )
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(), {"error": "no ratings"}
    combined = pd.concat(frames, ignore_index=True)
    summary = summarize(combined)
    summary["n_response_files"] = len(completed)
    summary["response_files"] = [str(p) for p in completed]
    if "auto_assigned" in combined.columns:
        summary["n_auto_assigned"] = int(combined["auto_assigned"].sum())
    return combined, summary


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Aggregate ablation listening responses.")
    parser.add_argument(
        "--responses",
        nargs="+",
        type=Path,
        help=(
            "Completed response JSON file(s) and/or directories. "
            "In-progress checkpoints (responses_in_progress_*.json) are ignored."
        ),
    )
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=None,
        help="Directory of response JSONs (default: experiments/ablation_listening/output/responses).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR / "results_notes.md",
        type=Path,
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        type=Path,
        help="Trial manifest (equivalences for auto-assigning omitted DDSP conditions).",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help=(
            "If set, also write overview + per-category content/realism bar plots "
            "(default folder when flag present with no path: output/plots)."
        ),
        nargs="?",
        const=DEFAULT_OUTPUT_DIR / "plots",
    )
    parser.add_argument(
        "--show-equivalences",
        action="store_true",
        help=(
            "When plotting, show bars auto-copied from a donor (e.g. drums CA/CB). "
            "Hidden by default. Only used with --plots-dir."
        ),
    )
    return parser.parse_args(args)


def main(args=None) -> None:
    from experiments.ablation_listening.paths import DEFAULT_RESPONSES_DIR

    opts = parse_args(args)
    sources: list[Path] = list(opts.responses or [])
    if opts.responses_dir is not None:
        sources.append(opts.responses_dir)
    elif not sources:
        sources.append(DEFAULT_RESPONSES_DIR)

    completed = resolve_completed_response_paths(sources)
    if not completed:
        raise FileNotFoundError(
            "No completed response files found "
            f"(looked in: {', '.join(str(s) for s in sources)}). "
            "Finish the test (not just checkpoints) before aggregating."
        )

    df, summary = aggregate_responses(completed, manifest_path=opts.manifest)
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(summary, responses_path=completed[0])
    opts.output.write_text(markdown)
    json_path = opts.output.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2))
    print(markdown)
    print(f"Wrote {opts.output}")
    print(f"Wrote {json_path}")
    print(f"Used {len(completed)} completed response file(s)")
    n_auto = summary.get("n_auto_assigned") or 0
    if n_auto:
        print(f"Auto-assigned {n_auto} ratings from donor equivalences")

    if opts.plots_dir is not None:
        from experiments.ablation_listening.plot_results import write_plots

        result = write_plots(
            df,
            opts.plots_dir,
            hide_equivalences=not opts.show_equivalences,
        )
        print(f"Wrote plots under {opts.plots_dir}")
        print(f"  overview: {result['overview'].name}")
        print(f"  by category: {result['overview_by_category'].name}")
        print(f"  panels: {len(result['by_category'])} in by_category/")
        print(f"  winners: {result['category_winners'].name}")


if __name__ == "__main__":
    main()
