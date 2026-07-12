"""Tests for per-category realify preset loading and routing."""

import pandas as pd

from synthesis.realify.preset_config import (
    load_presets,
    resolve_category,
    select_preset,
)


def _sample_presets() -> dict:
    return {
        "default": {
            "init_noise_level": 0.45,
            "prompt_variant": "current",
            "steps": 8,
            "cfg_scale": 1.0,
        },
        "categories": {
            "drums": {"init_noise_level": 0.55},
            "piano": {"prompt_variant": "minimal"},
        },
        "routing": [
            {"category": "drums", "is_drum": True},
            {"category": "piano", "name_keywords": ["piano"]},
            {"category": "wind", "name_keywords": ["flute"]},
        ],
    }


def test_load_presets_reads_repo_defaults():
    presets = load_presets()
    assert presets["default"]["init_noise_level"] == 0.45
    assert presets["default"]["prompt_variant"] == "current"


def test_resolve_category_drums_and_piano():
    presets = _sample_presets()
    drum_row = pd.Series({"is_drum": True, "name": "Kick"})
    piano_row = pd.Series({"is_drum": False, "name": "Piano"})
    unknown_row = pd.Series({"is_drum": False, "name": "Banjo"})

    assert resolve_category(drum_row, presets) == "drums"
    assert resolve_category(piano_row, presets) == "piano"
    assert resolve_category(unknown_row, presets) == "default"


def test_select_preset_merges_default_and_category():
    presets = _sample_presets()
    drum_row = pd.Series({"is_drum": True, "name": "Kick"})
    piano_row = pd.Series({"is_drum": False, "name": "Piano"})
    flute_row = pd.Series({"is_drum": False, "name": "Flute"})

    assert select_preset(presets, drum_row) == {
        "init_noise_level": 0.55,
        "prompt_variant": "current",
        "steps": 8,
        "cfg_scale": 1.0,
    }
    assert select_preset(presets, piano_row) == {
        "init_noise_level": 0.45,
        "prompt_variant": "minimal",
        "steps": 8,
        "cfg_scale": 1.0,
    }
    assert select_preset(presets, flute_row)["init_noise_level"] == 0.45
