"""Shared constants and helpers for phased patch sweeps."""

from __future__ import annotations

from pathlib import Path

import yaml

from shared.config import SOUNDFONT_DIR

EXPERIMENT_DIR = Path(__file__).resolve().parent
GRIDS_DIR = EXPERIMENT_DIR / "grids"
SOUNDFONTS_CATALOG = EXPERIMENT_DIR / "soundfonts.yaml"
WINNERS_PATH = EXPERIMENT_DIR / "winners.yaml"
WINNERS_LOCKED_PATH = EXPERIMENT_DIR / "winners_locked.yaml"

PHASE1 = "phase1_soundfonts"
PHASE2 = "phase2_fx"
PHASE3 = "phase3_pools"

PHASES = (PHASE1, PHASE2, PHASE3)

PHASE_GRID_FILES = {
    PHASE1: GRIDS_DIR / "phase1_soundfonts.yaml",
    PHASE2: GRIDS_DIR / "phase2_fx.yaml",
    PHASE3: GRIDS_DIR / "phase3_pools.yaml",
}

PHASE_OUTPUT_SUBDIRS = {
    PHASE1: "phase1_soundfonts",
    PHASE2: "phase2_fx",
    PHASE3: "phase3_pools",
}


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def soundfont_path(file_name: str, soundfont_dir: str | Path | None = None) -> Path:
    root = Path(soundfont_dir or SOUNDFONT_DIR)
    return root / file_name


def load_soundfont_catalog(path: Path = SOUNDFONTS_CATALOG) -> dict:
    return load_yaml(path)


def soundfont_file_for_id(soundfont_id: str, catalog: dict | None = None) -> str:
    catalog = catalog or load_soundfont_catalog()
    for entry in catalog.get("candidates", []):
        if entry["id"] == soundfont_id:
            return entry["file"]
    raise KeyError(f"Unknown soundfont id: {soundfont_id}")


def phase_output_dir(base_output_dir: Path, phase: str) -> Path:
    return base_output_dir / PHASE_OUTPUT_SUBDIRS[phase]
