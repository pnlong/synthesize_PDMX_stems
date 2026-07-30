"""Score preset-sweep outputs with onset-based content fidelity."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.listening.aggregate import load_responses, ratings_dataframe
from experiments.preset_sweep.sweep import MANIFEST_FILENAME
from shared.config import REALIFY_CONTENT_FIDELITY_THRESHOLD
from synthesis.audio import load_stem
from synthesis.realify.content_fidelity import score_content_fidelity


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Score realified sweep variants against references using onset fidelity.",
    )
    parser.add_argument(
        "--sweep-dir",
        required=True,
        type=Path,
        help="Phase sweep output directory containing manifest.csv.",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        type=Path,
        help="Reference audio root (default: sweep-dir/clips if present, else probe source).",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="CSV output path (default: sweep-dir/content_fidelity_scores.csv).",
    )
    parser.add_argument(
        "--responses",
        default=None,
        type=Path,
        help="Optional listening responses JSON for correlation with human content ratings.",
    )
    parser.add_argument(
        "--threshold",
        default=REALIFY_CONTENT_FIDELITY_THRESHOLD,
        type=float,
        help="Pass/fail threshold for suggested gate (default from shared/config.py).",
    )
    parser.add_argument(
        "--content-gate",
        default=4.5,
        type=float,
        help="Human content rating treated as pass when >= this value (1-5 scale).",
    )
    return parser.parse_args(args)


def reference_path_for_row(
    row: pd.Series,
    *,
    sweep_dir: Path,
    reference_dir: Path | None,
) -> Path:
    clips_dir = sweep_dir / "clips"
    if reference_dir is not None:
        root = reference_dir
    elif clips_dir.is_dir():
        root = clips_dir
    else:
        root = Path(row["path"]).parent

    stem_id = row.get("stem_id")
    track = int(row["track"])
    if isinstance(stem_id, str) and (root / f"{stem_id}.flac").is_file():
        return root / f"{stem_id}.flac"
    if isinstance(stem_id, str) and (root / f"{stem_id}.mp3").is_file():
        return root / f"{stem_id}.mp3"

    song_path = Path(row["path"])
    for ext in ("flac", "mp3", "wav"):
        candidate = song_path / f"stem_{track}.{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No reference audio found for manifest row stem_id={stem_id!r}")


def score_manifest(
    manifest: pd.DataFrame,
    *,
    sweep_dir: Path,
    reference_dir: Path | None,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for _, row in manifest.iterrows():
        out_path = Path(row["out_path"])
        if not out_path.is_file():
            continue
        ref_path = reference_path_for_row(
            row,
            sweep_dir=sweep_dir,
            reference_dir=reference_dir,
        )
        reference = load_stem(ref_path)
        realified = load_stem(out_path)
        result = score_content_fidelity(reference, realified, threshold=threshold)
        rows.append({
            "stem_id": row.get("stem_id"),
            "category": row.get("category"),
            "variant_id": row.get("variant_id"),
            "init_noise_level": row.get("init_noise_level"),
            "prompt_variant": row.get("prompt_variant"),
            "reference_path": str(ref_path),
            "out_path": str(out_path),
            "fidelity_score": result.score,
            "matched_onsets": result.matched_onsets,
            "extra_onsets": result.extra_onsets,
            "missing_onsets": result.missing_onsets,
            "n_reference_onsets": result.n_reference_onsets,
            "n_realified_onsets": result.n_realified_onsets,
            "passed": result.passed,
        })
    return pd.DataFrame(rows)


def correlate_with_ratings(
    scores: pd.DataFrame,
    responses_path: Path,
    *,
    content_gate: float,
) -> pd.DataFrame:
    responses = load_responses(responses_path)
    ratings = ratings_dataframe(responses)
    if ratings.empty or scores.empty:
        return pd.DataFrame()

    merged = scores.merge(
        ratings,
        on=["stem_id", "variant_id"],
        how="inner",
        suffixes=("", "_human"),
    )
    if merged.empty:
        return merged

    merged["human_content_pass"] = merged["content"] >= content_gate
    merged["agreement"] = merged["passed"] == merged["human_content_pass"]
    return merged


def print_threshold_suggestions(scores: pd.DataFrame, merged: pd.DataFrame) -> None:
    if scores.empty:
        print("No scored rows.")
        return

    print(f"\nScored variants: {len(scores)}")
    print(
        "Automated pass rate: "
        f"{scores['passed'].mean():.1%} at threshold in shared/config.py"
    )

    if merged.empty:
        print("No overlapping listening ratings — skip correlation.")
        return

    agreement = merged["agreement"].mean()
    print(f"Agreement with human content gate: {agreement:.1%} ({len(merged)} pairs)")

    best_threshold = None
    best_agreement = -1.0
    for threshold in [round(x * 0.05, 2) for x in range(0, 21)]:
        predicted_pass = merged["fidelity_score"] >= threshold
        acc = (predicted_pass == merged["human_content_pass"]).mean()
        if acc >= best_agreement:
            best_agreement = acc
            best_threshold = threshold

    print(
        f"Suggested threshold from grid search: {best_threshold:.2f} "
        f"(agreement {best_agreement:.1%})"
    )
    borderline = merged[
        merged["fidelity_score"].between(
            REALIFY_CONTENT_FIDELITY_THRESHOLD - 0.05,
            REALIFY_CONTENT_FIDELITY_THRESHOLD + 0.05,
        )
    ]
    if not borderline.empty:
        print(
            f"Borderline cases near default threshold: {len(borderline)} "
            "(listen to these during manual calibration)"
        )


def main():
    args = parse_args()
    sweep_dir = args.sweep_dir.resolve()
    manifest_path = sweep_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    scores = score_manifest(
        manifest,
        sweep_dir=sweep_dir,
        reference_dir=args.reference_dir.resolve() if args.reference_dir else None,
        threshold=args.threshold,
    )

    output_path = args.output or (sweep_dir / "content_fidelity_scores.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False)
    print(f"Wrote {len(scores)} scores to {output_path}")

    merged = pd.DataFrame()
    if args.responses is not None:
        merged = correlate_with_ratings(
            scores,
            args.responses.resolve(),
            content_gate=args.content_gate,
        )
        if not merged.empty:
            corr_path = output_path.with_name("content_fidelity_correlation.csv")
            merged.to_csv(corr_path, index=False)
            print(f"Wrote correlation table to {corr_path}")

    print_threshold_suggestions(scores, merged)


if __name__ == "__main__":
    main()
