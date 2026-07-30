"""Shared constants and helpers for phased preset sweeps."""

from __future__ import annotations

from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
GRIDS_DIR = EXPERIMENT_DIR / "grids"
WINNERS_PATH = EXPERIMENT_DIR / "winners.yaml"
WINNERS_LOCKED_PATH = EXPERIMENT_DIR / "winners_locked.yaml"
CATEGORIES_YAML_PATH = (
    Path(__file__).resolve().parents[2]
    / "synthesis"
    / "realify"
    / "presets"
    / "categories.yaml"
)

PHASE1 = "phase1_noise"
PHASE1B = "phase1b_noise_audit"
PHASE2 = "phase2_prompts"
PHASE2B = "phase2b_adherence_prompt"
PHASE3 = "phase3_diffusion"
PHASE4 = "phase4_verify_diverse"

TUNING_PHASES = (PHASE1, PHASE1B, PHASE2, PHASE2B, PHASE3)
PHASES = TUNING_PHASES
SWEEP_PHASES = TUNING_PHASES + (PHASE4,)
REQUIRED_LOCK_PHASES = (PHASE1, PHASE2)
LOCKED_VERIFY_VARIANT = "locked"

NOISE_LEVELS = (0.25, 0.35, 0.45, 0.55, 0.65)

PHASE_GRID_FILES = {
    PHASE1: GRIDS_DIR / "phase1_noise.yaml",
    PHASE1B: GRIDS_DIR / "phase1b_noise_audit.yaml",
    PHASE2: GRIDS_DIR / "phase2_prompts.yaml",
    PHASE2B: GRIDS_DIR / "phase2b_adherence_prompt.yaml",
    PHASE3: GRIDS_DIR / "phase3_diffusion.yaml",
    PHASE4: GRIDS_DIR / "phase4_verify_diverse.yaml",
}

PHASE_OUTPUT_SUBDIRS = {
    PHASE1: "phase1_noise",
    PHASE1B: "phase1b_noise_audit",
    PHASE2: "phase2_prompts",
    PHASE2B: "phase2b_adherence_prompt",
    PHASE3: "phase3_diffusion",
    PHASE4: "phase4_verify_diverse",
}


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def phase_output_dir(base_output_dir: Path, phase: str) -> Path:
    return base_output_dir / PHASE_OUTPUT_SUBDIRS[phase]


def init_noise_level_from_variant_id(variant_id: str) -> float:
    prefix = "noise"
    if not variant_id.startswith(prefix):
        raise ValueError(f"Expected phase-1 variant id like noise0.45, got {variant_id!r}")
    return float(variant_id[len(prefix):])


def lower_noise_level(level: float, *, levels: tuple[float, ...] = NOISE_LEVELS) -> float:
    """Next lower init_noise_level on the phase-1 grid (or same if already minimum)."""
    lower = [value for value in levels if value < level]
    return max(lower) if lower else level


def higher_noise_level(level: float, *, levels: tuple[float, ...] = NOISE_LEVELS) -> float:
    """Next higher init_noise_level on the phase-1 grid (or same if already maximum)."""
    higher = [value for value in levels if value > level]
    return min(higher) if higher else level


def noise_variant_id(level: float) -> str:
    text = f"{level:.2f}".rstrip("0").rstrip(".")
    return f"noise{text}"


def build_noise_audit_variants(
    phase1_winners: dict[str, str],
    *,
    levels: tuple[float, ...] = NOISE_LEVELS,
) -> list[dict]:
    """Build winner-vs-lower noise variants for the phase-1b audit."""
    needed_levels: set[float] = set()
    for variant_id in phase1_winners.values():
        winner_level = init_noise_level_from_variant_id(variant_id)
        needed_levels.add(winner_level)
        needed_levels.add(lower_noise_level(winner_level, levels=levels))

    variants = []
    for level in sorted(needed_levels):
        variants.append({
            "id": noise_variant_id(level),
            "init_noise_level": level,
            "note": "Phase-1 winner or one-step-lower audit candidate",
        })
    return variants


def build_adherence_audit_variants(
    phase1_winners: dict[str, str],
    *,
    levels: tuple[float, ...] = NOISE_LEVELS,
    include_negative: bool = False,
) -> list[dict]:
    """Build baseline-vs-adherence prompt audit variants at winner and higher noise."""
    from synthesis.realify.captions.metadata import ADHERENCE_NEGATIVE_PROMPT

    variants: list[dict] = []
    seen: set[str] = set()

    def add_variant(variant: dict) -> None:
        variant_id = variant["id"]
        if variant_id in seen:
            return
        seen.add(variant_id)
        variants.append(variant)

    for variant_id in phase1_winners.values():
        winner_level = init_noise_level_from_variant_id(variant_id)
        higher_level = higher_noise_level(winner_level, levels=levels)
        for noise_source, level in (
            ("winner", winner_level),
            ("higher", higher_level),
        ):
            add_variant({
                "id": f"baseline_{noise_source}",
                "prompt_source": "phase2",
                "noise_source": noise_source,
                "note": f"Phase-2 prompt winner at {level} ({noise_source})",
            })
            add_variant({
                "id": f"adherence_{noise_source}",
                "prompt_variant": "adherence",
                "noise_source": noise_source,
                "note": f"Adherence prompt at {level} ({noise_source})",
            })
            if include_negative:
                add_variant({
                    "id": f"adherence_neg_{noise_source}",
                    "prompt_variant": "adherence",
                    "negative_prompt": ADHERENCE_NEGATIVE_PROMPT,
                    "noise_source": noise_source,
                    "note": f"Adherence + negative prompt at {level} ({noise_source})",
                })

    return variants


def resolve_silence_enforce(phase: str, grid_cfg: dict) -> bool:
    """Whether preset-sweep realify applies post-SA3 silence enforcement."""
    if "silence_enforce" in grid_cfg:
        return bool(grid_cfg["silence_enforce"])
    return phase in (PHASE1B, PHASE2B)
