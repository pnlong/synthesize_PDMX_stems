"""Tests for per-instrument bypass routing rules."""

import pandas as pd

from experiments.preset_sweep.bypass_rules import (
    bypass_rule_from_stem,
    merge_bypass_rules,
    track_name_keywords,
)
from synthesis.realify.preset_config import routing_rule_matches_row, select_preset


def test_track_name_keywords_from_stem_name():
    assert track_name_keywords("Soprano Voice") == ["soprano", "voice"]


def test_bypass_rule_prefers_name_keywords():
    rule = bypass_rule_from_stem(
        category="voice",
        track_name="Soprano",
        program=52,
        is_drum=False,
    )
    assert rule["name_keywords"] == ["soprano"]
    assert rule["realify"] is False


def test_bypass_rule_falls_back_to_program():
    rule = bypass_rule_from_stem(
        category="wind",
        track_name="",
        program=73,
        is_drum=False,
    )
    assert rule["program"] == 73
    assert "name_keywords" not in rule


def test_merge_bypass_rules_deduplicates():
    rule = bypass_rule_from_stem(
        category="organ",
        track_name="Organ",
        program=19,
        is_drum=False,
    )
    merged = merge_bypass_rules([rule], [dict(rule)])
    assert len(merged) == 1


def test_select_preset_honors_rule_level_realify_false():
    presets = {
        "default": {"init_noise_level": 0.45},
        "categories": {"voice": {"init_noise_level": 0.45, "prompt_variant": "current"}},
        "routing": [
            {"category": "voice", "name_keywords": ["soprano"], "realify": False},
            {"category": "voice", "name_keywords": ["voice"]},
        ],
    }
    soprano = pd.Series({"name": "Soprano", "program": 52, "is_drum": False})
    choir = pd.Series({"name": "Choir", "program": 52, "is_drum": False})

    assert select_preset(presets, soprano)["realify"] is False
    assert select_preset(presets, choir).get("realify", True) is not False


def test_routing_rule_matches_program_fallback():
    rule = {"category": "wind", "program": 73, "realify": False}
    row = pd.Series({"name": "", "program": 73, "is_drum": False})
    assert routing_rule_matches_row(rule, row)
