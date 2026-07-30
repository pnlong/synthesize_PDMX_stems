"""Shared constants and helpers for phased patch sweeps."""

from __future__ import annotations

from pathlib import Path

import yaml

from shared.config import SOUNDFONT_DIR

EXPERIMENT_DIR = Path(__file__).resolve().parent
GRIDS_DIR = EXPERIMENT_DIR / "grids"
SOUNDFONTS_CATALOG = EXPERIMENT_DIR / "soundfonts.yaml"
ARCHIVE_SOUNDFONTS_CATALOG = EXPERIMENT_DIR / "archive_soundfonts.yaml"
WINNERS_PATH = EXPERIMENT_DIR / "winners.yaml"
WINNERS_LOCKED_PATH = EXPERIMENT_DIR / "winners_locked.yaml"

PHASE1 = "phase1_soundfonts"
PHASE1_ARCHIVE = "phase1_archive_soundfonts"
PHASE2 = "phase2_fx"
PHASE3 = "phase3_pools"

PHASE1_PHASES = frozenset({PHASE1, PHASE1_ARCHIVE})
PHASES = (PHASE1, PHASE2)
SWEEP_PHASES = (PHASE1, PHASE1_ARCHIVE, PHASE2, PHASE3)
REQUIRED_LOCK_PHASES = (PHASE1, PHASE2)
DEPRECATED_PHASES = (PHASE3,)

PHASE_GRID_FILES = {
    PHASE1: GRIDS_DIR / "phase1_soundfonts.yaml",
    PHASE1_ARCHIVE: GRIDS_DIR / "phase1_archive_soundfonts.yaml",
    PHASE2: GRIDS_DIR / "phase2_fx.yaml",
    PHASE3: GRIDS_DIR / "phase3_pools.yaml",
}

PHASE_OUTPUT_SUBDIRS = {
    PHASE1: "phase1_soundfonts",
    PHASE1_ARCHIVE: "phase1_archive_soundfonts",
    PHASE2: "phase2_fx",
    PHASE3: "phase3_pools",
}


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def soundfont_path(file_name: str, soundfont_dir: str | Path | None = None) -> Path:
    root = Path(soundfont_dir or SOUNDFONT_DIR)
    return root / file_name


def load_soundfont_catalog(path: Path | None = None) -> dict:
    return load_yaml(path or SOUNDFONTS_CATALOG)


def load_archive_soundfont_catalog(path: Path | None = None) -> dict:
    return load_yaml(path or ARCHIVE_SOUNDFONTS_CATALOG)


def load_combined_soundfont_catalog() -> dict:
    """Merge core + archive catalogs; archive ids win on duplicate ids."""
    combined: dict[str, dict] = {}
    for catalog in (load_soundfont_catalog(), load_archive_soundfont_catalog()):
        for entry in catalog.get("candidates", []):
            combined[entry["id"]] = entry
    return {"candidates": list(combined.values())}


def archive_variants_from_catalog(catalog: dict | None = None) -> list[dict]:
    """Build sweep variants for every archive soundfont (no tag filtering)."""
    catalog = catalog or load_archive_soundfont_catalog()
    variants = []
    for entry in catalog.get("candidates", []):
        name = entry.get("archive_name") or entry["id"]
        variants.append({
            "id": entry["id"],
            "soundfont_id": entry["id"],
            "note": name,
        })
    return variants


def soundfont_file_for_id(
    soundfont_id: str,
    catalog: dict | None = None,
    *,
    include_archive: bool = True,
) -> str:
    catalogs = [catalog] if catalog is not None else [load_soundfont_catalog()]
    if include_archive and catalog is None:
        catalogs.append(load_archive_soundfont_catalog())
    for cat in catalogs:
        for entry in cat.get("candidates", []):
            if entry["id"] == soundfont_id:
                return entry["file"]
    raise KeyError(f"Unknown soundfont id: {soundfont_id}")


def phase_output_dir(base_output_dir: Path, phase: str) -> Path:
    return base_output_dir / PHASE_OUTPUT_SUBDIRS[phase]
