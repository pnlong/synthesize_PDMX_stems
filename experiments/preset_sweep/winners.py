"""Load and update per-phase preset tuning winners."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from experiments.preset_sweep.config import (
    CATEGORIES_YAML_PATH,
    PHASE1,
    PHASE2,
    PHASE3,
    PHASES,
    REQUIRED_LOCK_PHASES,
    WINNERS_LOCKED_PATH,
    WINNERS_PATH,
    init_noise_level_from_variant_id,
    load_yaml,
)
from shared.config import REALIFY_CFG_SCALE, REALIFY_STEPS


def _empty_winners_doc() -> dict:
    return {
        "phases": {
            PHASE1: {"completed": False, "winners": {}},
            PHASE2: {"completed": False, "winners": {}},
            PHASE3: {"completed": False, "winners": {}},
        }
    }


def load_winners(path: Path = WINNERS_PATH) -> dict:
    if not path.is_file():
        return _empty_winners_doc()
    doc = load_yaml(path)
    if "phases" not in doc:
        doc = _empty_winners_doc()
    for phase in PHASES:
        doc["phases"].setdefault(phase, {"completed": False, "winners": {}})
    return doc


def save_winners(doc: dict, path: Path = WINNERS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    return path


def record_phase_winners(
    phase: str,
    winners: dict[str, str],
    *,
    path: Path = WINNERS_PATH,
) -> dict:
    """Persist per-category variant_id winners for a phase."""
    doc = load_winners(path)
    phase_doc = doc["phases"].setdefault(phase, {"completed": False, "winners": {}})
    phase_doc["winners"] = dict(winners)
    phase_doc["completed"] = True
    phase_doc["recorded_at"] = datetime.now(timezone.utc).isoformat()
    save_winners(doc, path)
    return doc


def phase_winners(phase: str, path: Path = WINNERS_PATH) -> dict[str, str]:
    doc = load_winners(path)
    return dict(doc["phases"].get(phase, {}).get("winners") or {})


def phase_is_complete(phase: str, path: Path = WINNERS_PATH) -> bool:
    doc = load_winners(path)
    return bool(doc["phases"].get(phase, {}).get("completed"))


def resolve_phase1_init_noise_level(category: str, winners_path: Path = WINNERS_PATH) -> float | None:
    variant_id = phase_winners(PHASE1, winners_path).get(category)
    if variant_id is None:
        return None
    return init_noise_level_from_variant_id(variant_id)


def resolve_phase2_prompt_variant(category: str, winners_path: Path = WINNERS_PATH) -> str | None:
    return phase_winners(PHASE2, winners_path).get(category)


def resolve_phase3_diffusion_variant_id(category: str, winners_path: Path = WINNERS_PATH) -> str | None:
    return phase_winners(PHASE3, winners_path).get(category)


def _default_diffusion_settings() -> dict:
    return {"steps": REALIFY_STEPS, "cfg_scale": REALIFY_CFG_SCALE}


def _diffusion_from_variant_id(variant_id: str) -> dict:
    """Parse steps/cfg from phase-3 variant ids like steps8_cfg1.0."""
    if not variant_id.startswith("steps") or "_cfg" not in variant_id:
        raise ValueError(f"Expected phase-3 variant id like steps8_cfg1.0, got {variant_id!r}")
    steps_part, cfg_part = variant_id.split("_cfg", 1)
    return {
        "steps": int(steps_part.removeprefix("steps")),
        "cfg_scale": float(cfg_part),
    }


def build_locked_preset_config(
    winners_path: Path = WINNERS_PATH,
    *,
    diffusion_defaults: dict | None = None,
) -> dict:
    """Merge phase winners into per-category production preset overrides."""
    doc = load_winners(winners_path)
    diffusion_defaults = diffusion_defaults or _default_diffusion_settings()

    phase1 = doc["phases"][PHASE1].get("winners") or {}
    phase2 = doc["phases"][PHASE2].get("winners") or {}
    phase3 = doc["phases"][PHASE3].get("winners") or {}
    phase3_complete = doc["phases"][PHASE3].get("completed", False)

    categories = sorted(set(phase1) | set(phase2) | set(phase3))
    locked = {"categories": {}}
    for category in categories:
        noise_variant = phase1.get(category)
        prompt_variant = phase2.get(category)
        if noise_variant is None or prompt_variant is None:
            continue

        preset = {
            "init_noise_level": init_noise_level_from_variant_id(noise_variant),
            "prompt_variant": prompt_variant,
        }

        if phase3_complete and category in phase3:
            preset.update(_diffusion_from_variant_id(phase3[category]))
        else:
            preset.update(diffusion_defaults)

        locked["categories"][category] = preset

    locked["locked_at"] = datetime.now(timezone.utc).isoformat()
    locked["source_phases"] = {
        phase: {
            "completed": doc["phases"][phase].get("completed", False),
            "recorded_at": doc["phases"][phase].get("recorded_at"),
        }
        for phase in PHASES
    }
    return locked


def merge_locked_into_categories_yaml(
    locked: dict,
    categories_path: Path = CATEGORIES_YAML_PATH,
) -> dict:
    """Apply locked category overrides to production categories.yaml."""
    doc = load_yaml(categories_path)
    doc.setdefault("categories", {})
    for category, preset in locked.get("categories", {}).items():
        doc["categories"][category] = dict(preset)
    return doc


def write_locked_config(
    winners_path: Path = WINNERS_PATH,
    output_path: Path = WINNERS_LOCKED_PATH,
    categories_path: Path = CATEGORIES_YAML_PATH,
    *,
    update_categories_yaml: bool = True,
) -> Path:
    for phase in REQUIRED_LOCK_PHASES:
        if not phase_is_complete(phase, winners_path):
            raise RuntimeError(f"Phase not complete: {phase}")

    locked = build_locked_preset_config(winners_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(locked, f, sort_keys=False, default_flow_style=False)

    if update_categories_yaml:
        merged = merge_locked_into_categories_yaml(locked, categories_path)
        with open(categories_path, "w") as f:
            yaml.safe_dump(merged, f, sort_keys=False, default_flow_style=False)

    return output_path
