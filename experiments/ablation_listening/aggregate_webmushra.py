"""Aggregate webMUSHRA mushra.csv results for ablation listening."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from experiments.ablation_listening.aggregate import factorial_table, render_markdown
from experiments.ablation_listening.paths import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WEBMUSHRA_ROOT,
    WEBMUSHRA_TEST_ID,
)
from experiments.ablation_listening.session import REFERENCE_CONDITION
from synthesis.listening.catalog import CONDITION_LABELS, CONDITION_ORDER

CONDITION_ALIASES = {
    "a1": "basic",
    "a2": "basic_realify",
    "b1": "slakh",
    "b2": "slakh_realify",
    "basic": "basic",
    "basic_realify": "basic_realify",
    "slakh": "slakh",
    "slakh_realify": "slakh_realify",
}


def normalize_condition(stimulus: str) -> str | None:
    key = str(stimulus).strip().lower()
    return CONDITION_ALIASES.get(key)


def load_mushra_csv(path: Path) -> pd.DataFrame:
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
            rows.append({
                "listener_id": row.get("listener_id") or row.get("session_uuid"),
                "trial_id": row.get("trial_id"),
                "condition_id": condition_id,
                "score": score,
            })
    return pd.DataFrame(rows)


def summarize_mushra(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"error": "no ratings"}

    means = (
        df.groupby("condition_id")["score"]
        .mean()
        .reindex(CONDITION_ORDER)
    )
    winner = means.idxmax(skipna=True) if not means.dropna().empty else None

    means_df = pd.DataFrame({
        "content": [float("nan")] * len(CONDITION_ORDER),
        "realism": means,
    }, index=CONDITION_ORDER)
    means_df.loc[REFERENCE_CONDITION, "content"] = float("nan")

    means_by_condition = {}
    for cond, value in means.items():
        if pd.isna(value):
            continue
        label = CONDITION_LABELS.get(cond, cond)
        means_by_condition[cond] = {
            "mushra_score": round(float(value), 2),
            "content": None if cond == REFERENCE_CONDITION else None,
            "realism": round(float(value), 2),
            "content_band": "— (reference)" if cond == REFERENCE_CONDITION else "—",
            "realism_band": f"{round(float(value), 1)}",
        }

    mix_trials = {t for t in df["trial_id"].unique() if str(t).startswith("mix_")}
    stem_trials = {t for t in df["trial_id"].unique() if str(t).startswith("stem_")}

    return {
        "n_ratings": int(len(df)),
        "n_listeners": int(df["listener_id"].nunique(dropna=True)),
        "winner": winner,
        "means_by_condition": means_by_condition,
        "mixture_means": (
            df[df["trial_id"].isin(mix_trials)].groupby("condition_id")["score"].mean().round(2).to_dict()
            if mix_trials else {}
        ),
        "stem_means": (
            df[df["trial_id"].isin(stem_trials)].groupby("condition_id")["score"].mean().round(2).to_dict()
            if stem_trials else {}
        ),
        "factorial_realism": factorial_table(means_df, "realism").to_dict(orient="index"),
    }


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

    df = load_mushra_csv(results_path)
    summary = summarize_mushra(df)

    opts.output.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(summary, responses_path=results_path)
    markdown = markdown.replace("Content | Realism", "MUSHRA score | (BAQ 0–100)")
    opts.output.write_text(markdown)
    json_path = opts.output.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2))
    print(markdown)
    print(f"Wrote {opts.output}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
