"""Aggregate ablation listening test responses."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

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

FACTORIAL_ROWS = {
    "basic": "basic",
    "basic_realify": "basic",
    "slakh": "slakh",
    "slakh_realify": "slakh",
}
FACTORIAL_COLS = {
    "basic": "synthetic",
    "slakh": "synthetic",
    "basic_realify": "realified",
    "slakh_realify": "realified",
}


def load_responses(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def ratings_dataframe(responses: dict) -> pd.DataFrame:
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
            })
    return pd.DataFrame(rows)


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
        index=["basic", "slakh"],
        columns=["synthetic", "realified"],
        dtype=float,
    )
    for condition_id, row in means.iterrows():
        value = row[field]
        if pd.isna(value):
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
    means["realism_likert"] = means["realism"].map(likert_equivalent)

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

    lines.extend(["", "## 2×2 factorial (content)", ""])
    fc = summary.get("factorial_content", {})
    lines.append("| | Synthetic | Realified |")
    lines.append("|--|-----------|-----------|")
    for row in ("basic", "slakh"):
        lines.append(
            f"| {row} | {fc.get(row, {}).get('synthetic', '—')} | "
            f"{fc.get(row, {}).get('realified', '—')} |"
        )

    lines.extend(["", "## 2×2 factorial (realism)", ""])
    fr = summary.get("factorial_realism", {})
    lines.append("| | Synthetic | Realified |")
    lines.append("|--|-----------|-----------|")
    for row in ("basic", "slakh"):
        lines.append(
            f"| {row} | {fr.get(row, {}).get('synthetic', '—')} | "
            f"{fr.get(row, {}).get('realified', '—')} |"
        )
    return "\n".join(lines) + "\n"


def aggregate_responses(paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    frames = []
    for path in paths:
        df = ratings_dataframe(load_responses(path))
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(), {"error": "no ratings"}
    combined = pd.concat(frames, ignore_index=True)
    return combined, summarize(combined)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Aggregate ablation listening responses.")
    parser.add_argument(
        "--responses",
        nargs="+",
        required=True,
        type=Path,
        help="One or more exported response JSON files.",
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
        help="Trial manifest (for metadata only).",
    )
    return parser.parse_args(args)


def main(args=None) -> None:
    opts = parse_args(args)
    _, summary = aggregate_responses(opts.responses)
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(summary, responses_path=opts.responses[0])
    opts.output.write_text(markdown)
    json_path = opts.output.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2))
    print(markdown)
    print(f"Wrote {opts.output}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
