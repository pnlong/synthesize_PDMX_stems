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
PHASE2 = "phase2_prompts"
PHASE3 = "phase3_diffusion"

PHASES = (PHASE1, PHASE2, PHASE3)
REQUIRED_LOCK_PHASES = (PHASE1, PHASE2)

PHASE_GRID_FILES = {
    PHASE1: GRIDS_DIR / "phase1_noise.yaml",
    PHASE2: GRIDS_DIR / "phase2_prompts.yaml",
    PHASE3: GRIDS_DIR / "phase3_diffusion.yaml",
}

PHASE_OUTPUT_SUBDIRS = {
    PHASE1: "phase1_noise",
    PHASE2: "phase2_prompts",
    PHASE3: "phase3_diffusion",
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
