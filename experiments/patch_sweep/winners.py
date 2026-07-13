"""Load and update per-phase tuning winners."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from experiments.patch_sweep.config import (
    PHASE1,
    PHASE2,
    PHASE3,
    PHASES,
    WINNERS_LOCKED_PATH,
    WINNERS_PATH,
    load_soundfont_catalog,
    load_yaml,
    soundfont_file_for_id,
)


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


def resolve_phase1_soundfont_id(category: str, winners_path: Path = WINNERS_PATH) -> str | None:
    return phase_winners(PHASE1, winners_path).get(category)


def resolve_phase2_fx_profile(category: str, winners_path: Path = WINNERS_PATH) -> str | None:
    return phase_winners(PHASE2, winners_path).get(category)


def build_locked_render_config(
    winners_path: Path = WINNERS_PATH,
    soundfonts_catalog_path: Path | None = None,
) -> dict:
    """Merge all phase winners into a production render config."""
    doc = load_winners(winners_path)
    catalog = load_soundfont_catalog(soundfonts_catalog_path)

    phase1 = doc["phases"][PHASE1].get("winners") or {}
    phase2 = doc["phases"][PHASE2].get("winners") or {}
    phase3 = doc["phases"][PHASE3].get("winners") or {}

    categories = sorted(set(phase1) | set(phase2) | set(phase3))
    locked = {"categories": {}}
    for category in categories:
        soundfont_id = phase1.get(category)
        if soundfont_id is None:
            continue
        locked["categories"][category] = {
            "soundfont_id": soundfont_id,
            "soundfont": soundfont_file_for_id(soundfont_id, catalog),
            "fx_profile": phase2.get(category, "light"),
            "pool_id": phase3.get(category, "pool_v1_conservative"),
        }

    locked["locked_at"] = datetime.now(timezone.utc).isoformat()
    return locked


def write_locked_config(
    winners_path: Path = WINNERS_PATH,
    output_path: Path = WINNERS_LOCKED_PATH,
) -> Path:
    for phase in PHASES:
        if not phase_is_complete(phase, winners_path):
            raise RuntimeError(f"Phase not complete: {phase}")

    locked = build_locked_render_config(winners_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(locked, f, sort_keys=False, default_flow_style=False)
    return output_path
