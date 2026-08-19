"""Validate realified stems without a listening test.

Scores content fidelity (onset F1) and silence hallucinations (reference-silent
regions). Can render stems with the content-fidelity noise backoff loop and
report each attempt.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from shared.config import (
    ABLATION_SAMPLE_SEED,
    DATA_DIR_NAME,
    DEFAULT_AUDIO_FORMAT,
    REALIFY_CONTENT_FIDELITY_MAX_ATTEMPTS,
    REALIFY_CONTENT_FIDELITY_MIN_NOISE,
    REALIFY_CONTENT_FIDELITY_NOISE_STEP,
    REALIFY_CONTENT_FIDELITY_THRESHOLD,
    REALIFY_INIT_NOISE_LEVEL,
    STEMS_FILE_NAME,
)
from synthesis.realify.preset_config import DEFAULT_PRESETS_FILE, load_presets, realify_enabled, select_preset
from synthesis.audio import load_stem, write_audio
from synthesis.realify.captions.generate import generate_captions_from_tables
from synthesis.listening.catalog import song_id_from_path
from synthesis.realify.content_fidelity import score_content_fidelity
from synthesis.realify.realify import (
    _generate_and_enforce,
    configure_sa3_env,
    load_model,
    stem_seed,
)
from synthesis.realify.silence import apply_silence_enforcement, score_silence_hallucinations


def load_clip_manifest(clips_dir: Path) -> list[dict]:
    for candidate in (clips_dir.parent / "diverse_stems.yaml", clips_dir / "diverse_stems.yaml"):
        if candidate.is_file():
            from experiments.preset_sweep.diverse_stems import load_diverse_stems_manifest

            return load_diverse_stems_manifest(candidate)

    stems = []
    for path in sorted(clips_dir.glob("**/*.flac")) + sorted(clips_dir.glob("**/*.mp3")):
        if path.name.startswith("stem_") or path.stem.isdigit():
            stems.append({
                "id": path.stem,
                "path": str(path),
                "category": None,
            })
    return stems


@dataclass
class ValidationAttempt:
    attempt: int
    init_noise_level: float
    fidelity_score: float
    extra_onsets: int
    missing_onsets: int
    content_passed: bool
    silence_hallucination_samples: int
    silence_hallucination_sec: float
    silence_passed: bool
    used_reference_passthrough: bool


@dataclass
class StemValidationResult:
    stem_path: str
    stem_id: str
    category: str | None
    prompt: str
    final_noise_level: float | None
    overall_passed: bool
    output_path: str | None
    reference_output_path: str | None
    attempts: list[ValidationAttempt]


def reference_output_path_for(stem_id: str, output_dir: Path, audio_format: str) -> Path:
    return output_dir / f"{stem_id}_reference.{audio_format}"


def write_reference_copy(
    reference,
    *,
    stem_id: str,
    output_dir: Path,
    audio_format: str,
) -> Path:
    path = reference_output_path_for(stem_id, output_dir, audio_format)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_audio(reference, path, audio_format=audio_format)
    return path


def _score_pair(
    reference: pd.Series,
    realified,
    *,
    threshold: float,
    silence_enforce: bool,
) -> tuple:
    """Return (content_result, silence_before, silence_after)."""
    content = score_content_fidelity(reference, realified, threshold=threshold)
    silence_before = score_silence_hallucinations(reference, realified)
    if silence_enforce:
        cleaned = apply_silence_enforcement(reference, realified, enabled=True)
        silence_after = score_silence_hallucinations(reference, cleaned)
    else:
        cleaned = realified
        silence_after = silence_before
    return content, silence_before, silence_after, cleaned


def validate_with_backoff(
    *,
    stem_path: Path,
    output_path: Path,
    row: dict,
    preset: dict,
    model,
    duration_seconds: float,
    seed: int,
    threshold: float,
    silence_enforce: bool,
    audio_format: str,
) -> StemValidationResult:
    """Run realify with decreasing init_noise_level until content fidelity passes."""
    reference = load_stem(stem_path)
    prompt = row["prompt"]
    init_audio = (44100, reference)

    noise = float(preset.get("init_noise_level", REALIFY_INIT_NOISE_LEVEL))
    step = REALIFY_CONTENT_FIDELITY_NOISE_STEP
    min_noise = REALIFY_CONTENT_FIDELITY_MIN_NOISE
    attempts: list[ValidationAttempt] = []
    final_audio = reference
    final_noise: float | None = None
    used_passthrough = False

    for attempt_idx in range(1, REALIFY_CONTENT_FIDELITY_MAX_ATTEMPTS + 1):
        trial_preset = {**preset, "init_noise_level": noise}
        audio = _generate_and_enforce(
            reference=reference,
            preset=trial_preset,
            model=model,
            prompt=prompt,
            init_audio=init_audio,
            duration_seconds=duration_seconds,
            seed=seed,
            silence_enforce=silence_enforce,
        )
        content, silence_before, silence_after, cleaned = _score_pair(
            reference,
            audio,
            threshold=threshold,
            silence_enforce=False,
        )
        attempts.append(
            ValidationAttempt(
                attempt=attempt_idx,
                init_noise_level=noise,
                fidelity_score=content.score,
                extra_onsets=content.extra_onsets,
                missing_onsets=content.missing_onsets,
                content_passed=content.passed,
                silence_hallucination_samples=silence_before.n_hallucination_samples,
                silence_hallucination_sec=silence_before.hallucination_duration_sec,
                silence_passed=silence_after.passed,
                used_reference_passthrough=False,
            )
        )
        final_audio = cleaned
        final_noise = noise

        if content.passed:
            break

        noise -= step
        if noise < min_noise - 1e-9:
            break

    if attempts and not attempts[-1].content_passed:
        used_passthrough = True
        final_audio = reference
        final_noise = None
        attempts.append(
            ValidationAttempt(
                attempt=len(attempts) + 1,
                init_noise_level=float("nan"),
                fidelity_score=attempts[-1].fidelity_score,
                extra_onsets=attempts[-1].extra_onsets,
                missing_onsets=attempts[-1].missing_onsets,
                content_passed=True,
                silence_hallucination_samples=0,
                silence_hallucination_sec=0.0,
                silence_passed=True,
                used_reference_passthrough=True,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem_id = row.get("stem_id", stem_path.stem)
    ref_out = write_reference_copy(
        reference,
        stem_id=stem_id,
        output_dir=output_path.parent,
        audio_format=audio_format,
    )
    write_audio(final_audio, output_path, audio_format=audio_format)

    last = attempts[-1]
    overall_passed = last.content_passed and last.silence_passed and not used_passthrough
    return StemValidationResult(
        stem_path=str(stem_path),
        stem_id=stem_id,
        category=row.get("category"),
        prompt=prompt,
        final_noise_level=final_noise,
        overall_passed=overall_passed,
        output_path=str(output_path),
        reference_output_path=str(ref_out),
        attempts=attempts,
    )


def score_existing_pair(
    *,
    reference_path: Path,
    realified_path: Path,
    threshold: float,
    silence_enforce: bool,
    stem_id: str = "",
    category: str | None = None,
) -> StemValidationResult:
    reference = load_stem(reference_path)
    realified = load_stem(realified_path)
    content, silence_before, silence_after, _ = _score_pair(
        reference,
        realified,
        threshold=threshold,
        silence_enforce=silence_enforce,
    )
    attempt = ValidationAttempt(
        attempt=1,
        init_noise_level=float("nan"),
        fidelity_score=content.score,
        extra_onsets=content.extra_onsets,
        missing_onsets=content.missing_onsets,
        content_passed=content.passed,
        silence_hallucination_samples=silence_before.n_hallucination_samples,
        silence_hallucination_sec=silence_before.hallucination_duration_sec,
        silence_passed=silence_after.passed,
        used_reference_passthrough=False,
    )
    return StemValidationResult(
        stem_path=str(reference_path),
        stem_id=stem_id or reference_path.stem,
        category=category,
        prompt="",
        final_noise_level=None,
        overall_passed=attempt.content_passed and attempt.silence_passed,
        output_path=str(realified_path),
        reference_output_path=None,
        attempts=[attempt],
    )


def caption_for_stem(
    *,
    source_dir: Path,
    song_path: Path,
    track: int,
    prompt_variant: str,
    sample_seed: int,
) -> tuple[str, pd.Series]:
    songs = pd.read_csv(source_dir / f"{DATA_DIR_NAME}.csv")
    stems = pd.read_csv(source_dir / f"{STEMS_FILE_NAME}.csv")
    song_id = str(song_path.relative_to(source_dir / DATA_DIR_NAME))
    stems["_song_id"] = stems["path"].map(lambda p: song_id_from_path(str(p)))
    stems = stems[(stems["_song_id"] == song_id) & (stems["track"] == track)]
    if stems.empty:
        raise ValueError(f"No stem row for {song_path} track {track}")

    stem_row = stems.iloc[0]
    captions = generate_captions_from_tables(
        songs,
        stems,
        seed=sample_seed,
        prompt_variant=prompt_variant,
    )
    return captions.iloc[0]["prompt"], stem_row


def build_rows_for_stems(
    *,
    clips_dir: Path,
    source_dir: Path,
    clip_entries: list[dict],
    prompt_variant: str,
    sample_seed: int,
) -> list[dict]:
    from experiments.preset_sweep.diverse_stems import probe_clip_path

    rows = []
    for entry in clip_entries:
        if "song_id" in entry and "track" in entry:
            clip_path = probe_clip_path(clips_dir, entry)
            song_path = clips_dir / DATA_DIR_NAME / entry["song_id"]
            track = int(entry["track"])
        else:
            clip_path = Path(entry["path"])
            if not clip_path.is_file():
                clip_path = clips_dir / clip_path.name
            song_path = clip_path.parent
            track = 0
            if clip_path.stem.isdigit():
                track = int(clip_path.stem)
            elif clip_path.stem.startswith("stem_"):
                track = int(clip_path.stem.split("_", 1)[1])

        if not clip_path.is_file():
            raise FileNotFoundError(f"Missing clip: {entry}")

        prompt, stem_row = caption_for_stem(
            source_dir=source_dir,
            song_path=song_path,
            track=track,
            prompt_variant=prompt_variant,
            sample_seed=sample_seed,
        )
        rows.append({
            **stem_row.to_dict(),
            "path": str(song_path),
            "track": track,
            "prompt": prompt,
            "stem_id": entry.get("id", clip_path.stem),
            "category": entry.get("category"),
            "clip_path": str(clip_path),
        })
    return rows


def results_to_dataframe(results: list[StemValidationResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        for attempt in result.attempts:
            rows.append({
                "stem_id": result.stem_id,
                "category": result.category,
                "stem_path": result.stem_path,
                "reference_output_path": result.reference_output_path,
                "output_path": result.output_path,
                "prompt": result.prompt,
                "final_noise_level": result.final_noise_level,
                "overall_passed": result.overall_passed,
                "attempt": attempt.attempt,
                "init_noise_level": attempt.init_noise_level,
                "fidelity_score": attempt.fidelity_score,
                "extra_onsets": attempt.extra_onsets,
                "missing_onsets": attempt.missing_onsets,
                "content_passed": attempt.content_passed,
                "silence_hallucination_samples": attempt.silence_hallucination_samples,
                "silence_hallucination_sec": attempt.silence_hallucination_sec,
                "silence_passed": attempt.silence_passed,
                "used_reference_passthrough": attempt.used_reference_passthrough,
            })
    return pd.DataFrame(rows)


def print_summary(results: list[StemValidationResult]) -> None:
    if not results:
        print("No stems validated.")
        return

    n_pass = sum(1 for r in results if r.overall_passed)
    n_passthrough = sum(
        1 for r in results if r.attempts and r.attempts[-1].used_reference_passthrough
    )
    print(f"\nValidated {len(results)} stems: {n_pass} passed, {len(results) - n_pass} failed")
    if n_passthrough:
        print(f"  {n_passthrough} fell back to reference passthrough")

    failures = [r for r in results if not r.overall_passed]
    if failures:
        print("\nFailures:")
        for result in failures[:20]:
            last = result.attempts[-1]
            print(
                f"  {result.stem_id}: fidelity={last.fidelity_score:.3f} "
                f"extra={last.extra_onsets} missing={last.missing_onsets} "
                f"silence_samples={last.silence_hallucination_samples}"
            )
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Validate realified stems (content fidelity + silence hallucinations).",
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        help="Directory of reference clips (e.g. phase1b/phase2b clips/).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Single reference stem to render and validate.",
    )
    parser.add_argument(
        "--realified",
        type=Path,
        help="Existing realified stem to score against --reference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/preset_sweep/output/validate_realify"),
        help="Where to write outputs and validation_report.csv.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Ablation source dir for captions (default: parent of clips-dir).",
    )
    parser.add_argument(
        "--presets",
        type=Path,
        default=DEFAULT_PRESETS_FILE,
        help="Preset YAML (uses per-category init_noise_level).",
    )
    parser.add_argument(
        "--init-noise-level",
        type=float,
        default=None,
        help="Override preset init_noise_level for all stems.",
    )
    parser.add_argument(
        "--prompt-variant",
        default="current",
        help="Caption prompt variant.",
    )
    parser.add_argument("-m", "--model", default="medium", choices=["small-music", "medium"])
    parser.add_argument("--limit", type=int, default=None, help="Max stems to validate.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=REALIFY_CONTENT_FIDELITY_THRESHOLD,
        help="Content fidelity F1 pass threshold.",
    )
    parser.add_argument(
        "--no-silence-enforce",
        action="store_true",
        help="Skip silence enforcement during render (still scores hallucinations).",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Only score --reference vs --realified; do not render.",
    )
    parser.add_argument(
        "--audio-format",
        default=DEFAULT_AUDIO_FORMAT,
        choices=["flac", "mp3"],
    )
    parser.add_argument("--sample-seed", type=int, default=ABLATION_SAMPLE_SEED)
    return parser.parse_args(args)


def main(args=None) -> None:
    opts = parse_args(args)
    silence_enforce = not opts.no_silence_enforce

    if opts.score_only:
        if not opts.reference or not opts.realified:
            raise SystemExit("--score-only requires --reference and --realified")
        result = score_existing_pair(
            reference_path=opts.reference.resolve(),
            realified_path=opts.realified.resolve(),
            threshold=opts.threshold,
            silence_enforce=silence_enforce,
        )
        results = [result]
    else:
        configure_sa3_env()
        presets = load_presets(opts.presets)
        model = load_model(opts.model)

        if opts.reference:
            clip_entries = [{"id": opts.reference.stem, "path": str(opts.reference)}]
            source_dir = opts.source_dir or opts.reference.parent
            clips_dir = source_dir
        elif opts.clips_dir:
            clips_dir = opts.clips_dir.resolve()
            clip_entries = load_clip_manifest(clips_dir)
            source_dir = opts.source_dir or clips_dir
        else:
            raise SystemExit("Provide --clips-dir, --reference, or --score-only pair.")

        if opts.limit is not None:
            clip_entries = clip_entries[: opts.limit]

        rows = build_rows_for_stems(
            clips_dir=clips_dir if opts.clips_dir else source_dir,
            source_dir=source_dir,
            clip_entries=clip_entries,
            prompt_variant=opts.prompt_variant,
            sample_seed=opts.sample_seed,
        )

        output_dir = opts.output_dir.resolve()
        results: list[StemValidationResult] = []

        for row in rows:
            stem_path = Path(row["clip_path"])
            preset = select_preset(presets, pd.Series(row))
            if not realify_enabled(preset):
                print(f"Skipping {row['stem_id']}: realify disabled for category")
                continue
            if opts.init_noise_level is not None:
                preset = {**preset, "init_noise_level": opts.init_noise_level}

            out_path = output_dir / f"{row['stem_id']}_validated.{opts.audio_format}"
            waveform = load_stem(stem_path)
            duration_seconds = waveform.shape[-1] / 44100.0

            result = validate_with_backoff(
                stem_path=stem_path,
                output_path=out_path,
                row=row,
                preset=preset,
                model=model,
                duration_seconds=duration_seconds,
                seed=stem_seed(opts.sample_seed, str(stem_path.parent), 0),
                threshold=opts.threshold,
                silence_enforce=silence_enforce,
                audio_format=opts.audio_format,
            )
            results.append(result)
            last = result.attempts[-1]
            n_attempts = sum(1 for a in result.attempts if not a.used_reference_passthrough)
            passthrough = last.used_reference_passthrough
            print(
                f"{result.stem_id}: "
                f"attempts={n_attempts} "
                f"noise={result.final_noise_level} "
                f"fidelity={last.fidelity_score:.3f} "
                f"silence_ok={last.silence_passed} "
                f"passthrough={passthrough} "
                f"passed={result.overall_passed}"
            )
            if result.reference_output_path:
                print(f"  reference: {result.reference_output_path}")
            print(f"  validated: {result.output_path}")

    output_dir = opts.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_csv = output_dir / "validation_report.csv"
    df = results_to_dataframe(results)
    df.to_csv(report_csv, index=False)

    summary = {
        "n_stems": len(results),
        "n_passed": sum(1 for r in results if r.overall_passed),
        "threshold": opts.threshold,
        "silence_enforce": silence_enforce,
    }
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(results)
    print(f"\nWrote {report_csv}")
    if any(r.reference_output_path for r in results):
        print("Each stem has paired files: {stem_id}_reference.* and {stem_id}_validated.*")


if __name__ == "__main__":
    main()
