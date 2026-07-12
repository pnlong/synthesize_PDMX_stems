"""Load and resolve per-category SA3 realify presets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

REALIFY_PRESETS_DIR = Path(__file__).resolve().parent / "presets"
DEFAULT_PRESETS_FILE = REALIFY_PRESETS_DIR / "categories.yaml"


def load_presets(presets_filepath: Path | None = None) -> dict:
    presets_filepath = presets_filepath or DEFAULT_PRESETS_FILE
    with open(presets_filepath) as f:
        return yaml.safe_load(f)


def _normalize_name(name) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def resolve_category(row: pd.Series, presets: dict) -> str:
    """Map a stem metadata row to a category key (or implicit default)."""
    routing = presets.get("routing", [])
    name = _normalize_name(row.get("name"))

    for rule in routing:
        if rule.get("is_drum") and bool(row.get("is_drum")):
            return rule["category"]

        keywords = rule.get("name_keywords", [])
        if name and any(keyword in name for keyword in keywords):
            return rule["category"]

    return "default"


def select_preset(presets: dict, row: pd.Series) -> dict:
    """Merge default preset with category-specific overrides."""
    preset = dict(presets.get("default", {}))
    category = resolve_category(row, presets)
    if category != "default":
        categories = presets.get("categories", {})
        if category in categories:
            preset.update(categories[category] or {})
    return preset


def preset_key(preset: dict) -> tuple:
    from shared.config import (
        REALIFY_CFG_SCALE,
        REALIFY_INIT_NOISE_LEVEL,
        REALIFY_STEPS,
    )

    return (
        preset.get("steps", REALIFY_STEPS),
        preset.get("cfg_scale", REALIFY_CFG_SCALE),
        preset.get("init_noise_level", REALIFY_INIT_NOISE_LEVEL),
        preset.get("prompt_variant", "current"),
        preset.get("negative_prompt"),
    )
