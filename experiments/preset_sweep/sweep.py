"""Run phased SA3 preset sweeps over the probe set."""

from __future__ import annotations

import argparse
import multiprocessing
import shutil
from pathlib import Path

import pandas as pd
import yaml

from experiments.paths import DEFAULT_PROBE_STEMS, preset_sweep_output_root
from experiments.probe_stems import validate_probe_stems
from experiments.preset_sweep.clips_dir import expose_clips_dir
from experiments.preset_sweep.config import (
    EXPERIMENT_DIR,
    LOCKED_VERIFY_VARIANT,
    PHASE1,
    PHASE1B,
    PHASE2,
    PHASE3,
    PHASE4,
    PHASE_GRID_FILES,
    SWEEP_PHASES,
    build_noise_audit_variants,
    load_yaml,
    phase_output_dir,
    resolve_silence_enforce,
)
from experiments.preset_sweep.diverse_stems import (
    DEFAULT_CLIP_SECONDS,
    DEFAULT_DIVERSE_PER_CATEGORY,
    DEFAULT_MIN_RMS,
    load_diverse_stems_manifest,
    replace_silent_probe_clips,
    select_diverse_stems,
    write_diverse_clip_dataset,
)
from experiments.preset_sweep.winners import (
    phase_is_complete,
    phase_winners,
    resolve_phase1_init_noise_level,
    resolve_phase2_prompt_variant,
    resolve_phase3_diffusion_variant_id,
)
from shared.config import (
    ABLATION_SAMPLE_SEED,
    DATA_DIR_NAME,
    DEFAULT_AUDIO_FORMAT,
    OUTPUT_DIR,
    REALIFY_BATCH_SIZE,
    REALIFY_CFG_SCALE,
    REALIFY_STEPS,
    STEMS_FILE_NAME,
)
from shared.repo_symlinks import REPO_PRESET_SWEEP_OUTPUT_SYMLINK
from synthesis.audio import stem_duration_seconds, stem_filename, stem_is_valid, stem_path
from synthesis.listening.catalog import default_ablations_dir, song_id_from_path
from synthesis.paths import ablation_raw_dir
from synthesis.realify.captions.generate import generate_captions_from_tables
from synthesis.realify.realify import (
    _run_realify_cpu,
    _run_realify_gpu,
    configure_sa3_env,
    load_model,
    log_realify_plan,
    process_realify_tasks,
    realify_uses_gpu,
    stem_seed,
)

REALIFY_PRESETS_PATH = (
    Path(__file__).resolve().parents[2]
    / "synthesis"
    / "realify"
    / "presets"
    / "categories.yaml"
)

MANIFEST_FILENAME = "manifest.csv"
VARIANTS_DIR_NAME = "variants"

MANIFEST_COLUMNS = [
    "phase",
    "variant_id",
    "init_noise_level",
    "prompt_variant",
    "steps",
    "cfg_scale",
    "prompt",
    "stem_id",
    "category",
    "path",
    "track",
    "out_path",
]


def default_output_dir(output_root: str = OUTPUT_DIR) -> Path:
    if REPO_PRESET_SWEEP_OUTPUT_SYMLINK.is_dir():
        return REPO_PRESET_SWEEP_OUTPUT_SYMLINK.resolve()
    return Path(preset_sweep_output_root(output_root))


def default_source_dir(output_root: str = OUTPUT_DIR) -> Path:
    ablations = default_ablations_dir()
    basic = ablations / "basic"
    if basic.is_dir():
        return basic
    return Path(ablation_raw_dir(output_root, "basic"))


def song_path_from_id(source_dir: Path, song_id: str) -> Path:
    return source_dir / DATA_DIR_NAME / song_id


def variant_output_path(
    output_dir: Path,
    variant_id: str,
    song_path: Path,
    source_dir: Path,
    track: int,
    audio_format: str,
) -> Path:
    song_id = str(song_path.relative_to(source_dir / DATA_DIR_NAME))
    return (
        output_dir
        / VARIANTS_DIR_NAME
        / variant_id
        / DATA_DIR_NAME
        / song_id
        / stem_filename(track, audio_format)
    )


def _require_phase_prerequisites(phase: str, winners_path: Path) -> None:
    if phase == PHASE1B and not phase_is_complete(PHASE1, winners_path):
        raise RuntimeError(
            f"{PHASE1B} requires completed {PHASE1} winners in {winners_path}. "
            f"Run phase 1, listening test, and record_winners first."
        )
    if phase == PHASE2:
        if not phase_is_complete(PHASE1, winners_path):
            raise RuntimeError(
                f"{PHASE2} requires completed {PHASE1} winners in {winners_path}. "
                f"Run the phase 1 sweep, listening test, and record_winners first."
            )
        if not phase_is_complete(PHASE1B, winners_path):
            raise RuntimeError(
                f"{PHASE2} requires completed {PHASE1B} noise audit in {winners_path}. "
                f"Run phase 1b, listening test, and record_winners first."
            )
    if phase == PHASE3:
        if not phase_is_complete(PHASE1, winners_path):
            raise RuntimeError(f"{PHASE3} requires completed {PHASE1} winners.")
        if not phase_is_complete(PHASE1B, winners_path):
            raise RuntimeError(f"{PHASE3} requires completed {PHASE1B} noise audit.")
        if not phase_is_complete(PHASE2, winners_path):
            raise RuntimeError(f"{PHASE3} requires completed {PHASE2} winners.")
    if phase == PHASE4:
        if not phase_is_complete(PHASE1, winners_path):
            raise RuntimeError(f"{PHASE4} requires completed {PHASE1} winners.")
        if not phase_is_complete(PHASE1B, winners_path):
            raise RuntimeError(
                f"{PHASE4} requires completed {PHASE1B} diverse clips in {winners_path}."
            )
        if not phase_is_complete(PHASE2, winners_path):
            raise RuntimeError(f"{PHASE4} requires completed {PHASE2} winners.")


def resolve_preset_settings(
    *,
    phase: str,
    variant: dict,
    grid_cfg: dict,
    category: str | None,
    winners_path: Path,
) -> tuple[float, str, int, float]:
    """Return (init_noise_level, prompt_variant, steps, cfg_scale)."""
    default_prompt = grid_cfg.get("prompt_variant", "current")
    default_steps = int(grid_cfg.get("steps", REALIFY_STEPS))
    default_cfg = float(grid_cfg.get("cfg_scale", REALIFY_CFG_SCALE))

    if phase == PHASE1:
        return (
            float(variant["init_noise_level"]),
            variant.get("prompt_variant") or default_prompt,
            int(variant.get("steps", default_steps)),
            float(variant.get("cfg_scale", default_cfg)),
        )

    if phase == PHASE1B:
        return (
            float(variant["init_noise_level"]),
            grid_cfg.get("prompt_variant", default_prompt),
            int(grid_cfg.get("steps", default_steps)),
            float(grid_cfg.get("cfg_scale", default_cfg)),
        )

    if phase == PHASE2:
        init_noise_level = resolve_phase1_init_noise_level(category or "", winners_path)
        if init_noise_level is None:
            raise RuntimeError(f"No phase-1 noise winner for category: {category}")
        return (
            init_noise_level,
            variant["prompt_variant"],
            int(variant.get("steps", default_steps)),
            float(variant.get("cfg_scale", default_cfg)),
        )

    init_noise_level = resolve_phase1_init_noise_level(category or "", winners_path)
    if init_noise_level is None:
        raise RuntimeError(f"No phase-1 noise winner for category: {category}")
    prompt_variant = resolve_phase2_prompt_variant(category or "", winners_path)
    if prompt_variant is None:
        raise RuntimeError(f"No phase-2 prompt winner for category: {category}")

    if phase == PHASE3:
        return (
            init_noise_level,
            prompt_variant,
            int(variant.get("steps", default_steps)),
            float(variant.get("cfg_scale", default_cfg)),
        )

    diffusion_variant = resolve_phase3_diffusion_variant_id(category or "", winners_path)
    if diffusion_variant:
        from experiments.preset_sweep.winners import _diffusion_from_variant_id

        diffusion = _diffusion_from_variant_id(diffusion_variant)
        steps = int(diffusion["steps"])
        cfg_scale = float(diffusion["cfg_scale"])
    else:
        steps = default_steps
        cfg_scale = default_cfg

    return (
        init_noise_level,
        prompt_variant,
        steps,
        cfg_scale,
    )


def build_probe_caption(
    *,
    source_dir: Path,
    song_path: Path,
    track: int,
    prompt_variant: str,
    sample_seed: int,
) -> str:
    songs = pd.read_csv(source_dir / f"{DATA_DIR_NAME}.csv")
    stems = pd.read_csv(source_dir / f"{STEMS_FILE_NAME}.csv")
    song_id = str(song_path.relative_to(source_dir / DATA_DIR_NAME))
    stems["_song_id"] = stems["path"].map(lambda p: song_id_from_path(str(p)))
    stems = stems[
        (stems["_song_id"] == song_id) & (stems["track"] == track)
    ]
    if stems.empty:
        raise ValueError(f"No stem row for {song_path} track {track}")

    captions = generate_captions_from_tables(
        songs,
        stems,
        seed=sample_seed,
        prompt_variant=prompt_variant,
    )
    return captions.iloc[0]["prompt"]


def build_sweep_tasks(
    *,
    phase: str,
    source_dir: Path,
    output_dir: Path,
    probe_stems: list[dict],
    variants: list[dict],
    grid_cfg: dict,
    audio_format: str,
    sample_seed: int,
    winners_path: Path,
) -> list[dict]:
    tasks = []
    for probe in probe_stems:
        song_path = song_path_from_id(source_dir, probe["song_id"])
        track = int(probe["track"])
        stem_format = str(probe.get("audio_format") or audio_format)
        source_stem_path = stem_path(song_path, track, stem_format)
        if not stem_is_valid(source_stem_path):
            raise FileNotFoundError(f"Missing or invalid probe stem: {source_stem_path}")

        category = probe.get("category")

        for variant in variants:
            variant_id = variant["id"]
            init_noise_level, prompt_variant, steps, cfg_scale = resolve_preset_settings(
                phase=phase,
                variant=variant,
                grid_cfg=grid_cfg,
                category=category,
                winners_path=winners_path,
            )
            out_path = variant_output_path(
                output_dir,
                variant_id,
                song_path,
                source_dir,
                track,
                audio_format,
            )
            if out_path.exists():
                continue

            prompt = build_probe_caption(
                source_dir=source_dir,
                song_path=song_path,
                track=track,
                prompt_variant=prompt_variant,
                sample_seed=sample_seed,
            )
            preset = {
                "steps": steps,
                "cfg_scale": cfg_scale,
                "init_noise_level": init_noise_level,
            }
            if variant.get("negative_prompt") is not None:
                preset["negative_prompt"] = variant["negative_prompt"]

            tasks.append({
                "row": {
                    "path": str(song_path),
                    "track": track,
                    "prompt": prompt,
                    "stem_id": probe["id"],
                    "category": category,
                    "variant_id": variant_id,
                    "prompt_variant": prompt_variant,
                    "init_noise_level": init_noise_level,
                },
                "preset": preset,
                "out_path": str(out_path),
                "stem_path": str(source_stem_path),
                "duration": stem_duration_seconds(source_stem_path),
                "audio_format": stem_format,
                "seed": stem_seed(sample_seed, str(song_path), track),
            })
    return tasks


def build_manifest_rows(
    *,
    phase: str,
    source_dir: Path,
    output_dir: Path,
    probe_stems: list[dict],
    variants: list[dict],
    grid_cfg: dict,
    audio_format: str,
    sample_seed: int,
    winners_path: Path,
) -> list[dict]:
    rows = []
    for probe in probe_stems:
        song_path = song_path_from_id(source_dir, probe["song_id"])
        track = int(probe["track"])
        stem_format = str(probe.get("audio_format") or audio_format)
        category = probe.get("category")
        for variant in variants:
            variant_id = variant["id"]
            init_noise_level, prompt_variant, steps, cfg_scale = resolve_preset_settings(
                phase=phase,
                variant=variant,
                grid_cfg=grid_cfg,
                category=category,
                winners_path=winners_path,
            )
            prompt = build_probe_caption(
                source_dir=source_dir,
                song_path=song_path,
                track=track,
                prompt_variant=prompt_variant,
                sample_seed=sample_seed,
            )
            out_path = variant_output_path(
                output_dir,
                variant_id,
                song_path,
                source_dir,
                track,
                stem_format,
            )
            rows.append({
                "phase": phase,
                "variant_id": variant_id,
                "init_noise_level": init_noise_level,
                "prompt_variant": prompt_variant,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "prompt": prompt,
                "stem_id": probe["id"],
                "category": category,
                "path": str(song_path),
                "track": track,
                "out_path": str(out_path),
            })
    return rows


def write_manifest(output_dir: Path, rows: list[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest_path, index=False)
    return manifest_path


def prepare_phase1b_audit(
    *,
    source_dir: Path,
    output_dir: Path,
    grid_cfg: dict,
    winners_path: Path,
    sample_seed: int,
) -> tuple[Path, list[dict], list[dict]]:
    clip_seconds = float(grid_cfg.get("clip_seconds", DEFAULT_CLIP_SECONDS))
    per_category = int(grid_cfg.get("diverse_stems_per_category", DEFAULT_DIVERSE_PER_CATEGORY))
    min_rms = float(grid_cfg.get("min_rms", DEFAULT_MIN_RMS))
    clips_dir = output_dir / "clips"
    manifest_path = output_dir / "diverse_stems.yaml"

    if manifest_path.is_file():
        probe_stems = load_diverse_stems_manifest(manifest_path)
    else:
        probe_stems = select_diverse_stems(
            source_dir,
            per_category=per_category,
            clip_seconds=clip_seconds,
            min_rms=min_rms,
            seed=sample_seed,
        )
        write_diverse_clip_dataset(
            source_dir,
            probe_stems,
            clips_dir,
            clip_seconds=clip_seconds,
        )

    phase1_winners = phase_winners(PHASE1, winners_path)
    variants = build_noise_audit_variants(phase1_winners)
    return clips_dir, probe_stems, variants



def _link_phase4_clip_assets(*, sweep_root: Path, output_dir: Path) -> Path:
    """Expose phase-1b clips + manifest under the phase-4 output tree for listening."""
    phase1b_dir = phase_output_dir(sweep_root, PHASE1B)
    source_clips = phase1b_dir / "clips"
    source_manifest = phase1b_dir / "diverse_stems.yaml"
    if not source_manifest.is_file():
        raise RuntimeError(
            f"{PHASE4} requires phase-1b diverse_stems.yaml: {source_manifest}. "
            f"Run phase 1b first."
        )
    if not source_clips.is_dir():
        raise RuntimeError(
            f"{PHASE4} requires phase-1b clips dir: {source_clips}. Run phase 1b first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dest_manifest = output_dir / "diverse_stems.yaml"
    if not dest_manifest.exists():
        shutil.copy2(source_manifest, dest_manifest)
    expose_clips_dir(output_dir=output_dir, source_clips=source_clips)
    return source_clips


def prepare_phase4_verify(
    *,
    sweep_root: Path,
    output_dir: Path,
    source_dir: Path,
    grid_cfg: dict,
    sample_seed: int,
) -> tuple[Path, list[dict], list[dict]]:
    phase1b_dir = phase_output_dir(sweep_root, PHASE1B)
    reference_clips = phase1b_dir / "clips"
    reference_manifest = phase1b_dir / "diverse_stems.yaml"
    if not reference_manifest.is_file():
        raise RuntimeError(
            f"{PHASE4} requires phase-1b diverse_stems.yaml: {reference_manifest}. "
            f"Run phase 1b first."
        )
    if not reference_clips.is_dir():
        raise RuntimeError(
            f"{PHASE4} requires phase-1b clips dir: {reference_clips}. Run phase 1b first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "diverse_stems.yaml"
    if not manifest_path.is_file():
        shutil.copy2(reference_manifest, manifest_path)

    probe_stems = load_diverse_stems_manifest(manifest_path)
    clip_seconds = float(grid_cfg.get("clip_seconds", DEFAULT_CLIP_SECONDS))
    min_rms = float(grid_cfg.get("min_rms", DEFAULT_MIN_RMS))
    variants = [{"id": LOCKED_VERIFY_VARIANT}]

    if grid_cfg.get("replace_silent_clips", True):
        output_clips = output_dir / "clips"
        probe_stems, replacements = replace_silent_probe_clips(
            probe_stems,
            reference_clips_dir=reference_clips,
            source_dir=source_dir,
            output_clips_dir=output_clips,
            clip_seconds=clip_seconds,
            min_rms=min_rms,
            seed=sample_seed,
        )
        if replacements:
            with open(manifest_path, "w") as f:
                yaml.safe_dump(
                    {"clip_seconds": clip_seconds, "stems": probe_stems},
                    f,
                    sort_keys=False,
                    default_flow_style=False,
                )
            for entry in replacements:
                if entry["from_id"] == entry["to_id"]:
                    print(
                        f"Re-clipped silent stem in {entry['category']}: "
                        f"{entry['from_id']} @ {entry['clip_start_seconds']}s"
                    )
                else:
                    print(
                        f"Replaced silent clip in {entry['category']}: "
                        f"{entry['from_id']} -> {entry['to_id']}"
                    )
            clips_dir = output_clips
        else:
            clips_dir = _link_phase4_clip_assets(sweep_root=sweep_root, output_dir=output_dir)
    else:
        clips_dir = _link_phase4_clip_assets(sweep_root=sweep_root, output_dir=output_dir)

    return clips_dir, probe_stems, variants


def run_preset_sweep(
    *,
    phase: str,
    source_dir: Path,
    output_dir: Path,
    probe_stems_path: Path,
    grid_path: Path,
    winners_path: Path,
    model: str = "medium",
    jobs: int = 1,
    batch_size: int = REALIFY_BATCH_SIZE,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    sample_seed: int = ABLATION_SAMPLE_SEED,
    limit_stems: int | None = None,
    limit_variants: int | None = None,
    silence_enforce: bool | None = None,
) -> pd.DataFrame:
    if phase not in SWEEP_PHASES:
        raise ValueError(f"Unknown phase: {phase}")

    configure_sa3_env()
    source_dir = source_dir.resolve()
    sweep_root = output_dir.resolve()
    output_dir = phase_output_dir(sweep_root, phase)
    output_dir.mkdir(parents=True, exist_ok=True)

    _require_phase_prerequisites(phase, winners_path)

    probe_cfg = load_yaml(probe_stems_path)
    grid_cfg = load_yaml(grid_path)
    realify_source_dir = source_dir

    if phase == PHASE1B:
        clips_dir, probe_stems, variants = prepare_phase1b_audit(
            source_dir=source_dir,
            output_dir=output_dir,
            grid_cfg=grid_cfg,
            winners_path=winners_path,
            sample_seed=sample_seed,
        )
        realify_source_dir = clips_dir
    elif phase == PHASE4:
        clips_dir, probe_stems, variants = prepare_phase4_verify(
            sweep_root=sweep_root,
            output_dir=output_dir,
            source_dir=source_dir,
            grid_cfg=grid_cfg,
            sample_seed=sample_seed,
        )
        realify_source_dir = clips_dir
    else:
        probe_stems = list(probe_cfg["stems"])
        validate_probe_stems(probe_stems)
        variants = list(grid_cfg["variants"])

    if limit_stems is not None:
        probe_stems = probe_stems[:limit_stems]
    if limit_variants is not None:
        variants = variants[:limit_variants]

    if silence_enforce is None:
        silence_enforce = resolve_silence_enforce(phase, grid_cfg)

    for variant in variants:
        if phase in (PHASE1, PHASE1B) and "init_noise_level" not in variant:
            raise ValueError(f"Phase 1 variant missing init_noise_level: {variant}")
        if phase == PHASE2 and "prompt_variant" not in variant:
            raise ValueError(f"Phase 2 variant missing prompt_variant: {variant}")
        if phase == PHASE3 and ("steps" not in variant or "cfg_scale" not in variant):
            raise ValueError(f"Phase 3 variant missing steps/cfg_scale: {variant}")
        if phase == PHASE4 and variant.get("id") != LOCKED_VERIFY_VARIANT:
            raise ValueError(f"Phase 4 expects locked variant id {LOCKED_VERIFY_VARIANT!r}")

    tasks = build_sweep_tasks(
        phase=phase,
        source_dir=realify_source_dir,
        output_dir=output_dir,
        probe_stems=probe_stems,
        variants=variants,
        grid_cfg=grid_cfg,
        audio_format=audio_format,
        sample_seed=sample_seed,
        winners_path=winners_path,
    )

    manifest_rows = build_manifest_rows(
        phase=phase,
        source_dir=realify_source_dir,
        output_dir=output_dir,
        probe_stems=probe_stems,
        variants=variants,
        grid_cfg=grid_cfg,
        audio_format=audio_format,
        sample_seed=sample_seed,
        winners_path=winners_path,
    )
    manifest_path = write_manifest(output_dir, manifest_rows)

    use_gpu = realify_uses_gpu(model) if tasks else False
    print(f"Preset sweep phase: {phase}")
    print(f"Preset sweep source: {realify_source_dir}")
    print(f"Preset sweep output: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Probe stems: {len(probe_stems)}")
    print(f"Variants: {len(variants)}")
    print(f"Tasks queued: {len(tasks)} (skipped existing outputs)")
    if silence_enforce:
        print("Silence enforcement: enabled (reference-gated post-SA3 silence)")
    log_realify_plan(
        source_dir=source_dir,
        output_dir=output_dir,
        model=model,
        n_tasks=len(tasks),
        n_captions=len(manifest_rows),
        use_gpu=use_gpu,
        batch_size=batch_size,
    )

    if not tasks:
        return pd.read_csv(manifest_path)

    empty_presets: dict = {}
    if use_gpu:
        if len(tasks) == 1:
            sa3_model = load_model(model)
            process_realify_tasks(
                tasks,
                model=sa3_model,
                presets=empty_presets,
                audio_format=audio_format,
                batch_size=1,
                desc="Preset sweep (GPU)",
                silence_enforce=silence_enforce,
            )
        else:
            _run_realify_gpu(
                tasks,
                model=model,
                presets_filepath=REALIFY_PRESETS_PATH,
                batch_size=batch_size,
                audio_format=audio_format,
                silence_enforce=silence_enforce,
            )
    else:
        _run_realify_cpu(
            tasks,
            model=model,
            presets_filepath=REALIFY_PRESETS_PATH,
            jobs=jobs,
            batch_size=batch_size,
            audio_format=audio_format,
            silence_enforce=silence_enforce,
        )

    return pd.read_csv(manifest_path)


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="Run phased SA3 preset sweep on curated probe stems.",
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=list(SWEEP_PHASES),
        help="Tuning phase to run.",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        type=Path,
        help="Raw ablation dir (default: dev/ablations/basic).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Preset sweep root (phase subdir is appended automatically).",
    )
    parser.add_argument(
        "--probe-stems",
        default=DEFAULT_PROBE_STEMS,
        type=Path,
        help="Probe stem manifest YAML.",
    )
    parser.add_argument(
        "--grid",
        default=None,
        type=Path,
        help="Phase grid YAML (default: grids/phaseN_*.yaml for --phase).",
    )
    parser.add_argument(
        "--winners",
        default=EXPERIMENT_DIR / "winners.yaml",
        type=Path,
        help="Per-phase winners YAML (phase 2/3 read prior winners).",
    )
    parser.add_argument("-m", "--model", default="medium", choices=["small-music", "medium"])
    parser.add_argument(
        "-j",
        "--jobs",
        "--workers",
        default=int(multiprocessing.cpu_count() / 4),
        type=int,
        help="CPU workers for small-music CPU realify.",
    )
    parser.add_argument(
        "--realify-batch-size",
        default=REALIFY_BATCH_SIZE,
        type=int,
        help="SA3 stems per GPU forward pass.",
    )
    parser.add_argument("--limit-stems", default=None, type=int)
    parser.add_argument("--limit-variants", default=None, type=int)
    parser.add_argument("--sample-seed", default=ABLATION_SAMPLE_SEED, type=int)
    parser.add_argument(
        "--mp3",
        action="store_true",
        help="Read/write MP3 stems (must match source ablation format).",
    )
    return parser.parse_args(args=args, namespace=namespace)


def main():
    args = parse_args()
    from synthesis.audio import synthesis_audio_format

    grid_path = args.grid or PHASE_GRID_FILES[args.phase]

    run_preset_sweep(
        phase=args.phase,
        source_dir=args.source_dir or default_source_dir(),
        output_dir=args.output_dir or default_output_dir(),
        probe_stems_path=args.probe_stems,
        grid_path=grid_path,
        winners_path=args.winners,
        model=args.model,
        jobs=args.jobs,
        batch_size=args.realify_batch_size,
        audio_format=synthesis_audio_format(args.mp3),
        sample_seed=args.sample_seed,
        limit_stems=args.limit_stems,
        limit_variants=args.limit_variants,
    )


if __name__ == "__main__":
    main()
