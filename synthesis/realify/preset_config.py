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


def routing_rule_matches_row(rule: dict, row: pd.Series) -> bool:
    """Return True when a routing rule matches stem metadata."""
    name = _normalize_name(row.get("name"))
    is_drum = bool(row.get("is_drum"))
    program = int(row.get("program", 0) or 0)

    if rule.get("is_drum") and is_drum:
        return True

    rule_program = rule.get("program")
    if rule_program is not None and program == int(rule_program):
        rule_is_drum = rule.get("is_drum")
        if rule_is_drum is not None and is_drum != bool(rule_is_drum):
            return False
        return True

    keywords = rule.get("name_keywords", [])
    if name and keywords and any(keyword in name for keyword in keywords):
        return True

    return False


def resolve_category(row: pd.Series, presets: dict) -> str:
    """Map a stem metadata row to a category key (or implicit default)."""
    routing = presets.get("routing", [])
    for rule in routing:
        if routing_rule_matches_row(rule, row):
            return rule["category"]
    return "default"


def select_preset(presets: dict, row: pd.Series) -> dict:
    """Merge default preset with category-specific overrides."""
    preset = dict(presets.get("default", {}))
    routing = presets.get("routing", [])
    matched_rule: dict | None = None
    category = "default"

    for rule in routing:
        if routing_rule_matches_row(rule, row):
            matched_rule = rule
            category = rule["category"]
            break

    if category != "default":
        categories = presets.get("categories", {})
        if category in categories:
            preset.update(categories[category] or {})

    if matched_rule is not None and matched_rule.get("realify") is False:
        preset["realify"] = False

    return preset


def realify_enabled(preset: dict) -> bool:
    """Return False when a category is locked with ``realify: false`` (passthrough basic stems)."""
    return preset.get("realify", True) is not False


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
