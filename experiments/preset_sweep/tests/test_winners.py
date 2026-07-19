"""Tests for phased preset sweep winner recording."""

from pathlib import Path

import pytest
import yaml

from experiments.preset_sweep.config import (
    PHASE1,
    PHASE1B,
    PHASE2,
    PHASE3,
    init_noise_level_from_variant_id,
)
from experiments.preset_sweep.winners import (
    build_locked_preset_config,
    load_winners,
    merge_locked_into_categories_yaml,
    record_bypass_realify,
    record_bypass_realify_rules,
    record_phase_winners,
    write_locked_config,
)


def _winners_doc():
    return {
        "phases": {
            PHASE1: {"completed": True, "winners": {"piano": "noise0.45", "drums": "noise0.55"}},
            PHASE1B: {"completed": True, "winners": {"piano": "noise0.45", "drums": "noise0.55"}},
            PHASE2: {"completed": True, "winners": {"piano": "minimal", "drums": "preservation"}},
            PHASE3: {"completed": True, "winners": {"piano": "steps8_cfg1.0", "drums": "steps10_cfg1.2"}},
        }
    }


def test_init_noise_level_from_variant_id():
    assert init_noise_level_from_variant_id("noise0.45") == 0.45


def test_record_phase_winners(tmp_path: Path):
    path = tmp_path / "winners.yaml"
    record_phase_winners(PHASE1, {"piano": "noise0.45"}, path=path)
    doc = load_winners(path)
    assert doc["phases"][PHASE1]["completed"] is True
    assert doc["phases"][PHASE1]["winners"]["piano"] == "noise0.45"


def test_build_locked_preset_config(tmp_path: Path):
    winners_path = tmp_path / "winners.yaml"
    winners_path.write_text(yaml.safe_dump(_winners_doc()))

    locked = build_locked_preset_config(winners_path)
    assert locked["categories"]["piano"]["init_noise_level"] == 0.45
    assert locked["categories"]["piano"]["prompt_variant"] == "minimal"
    assert locked["categories"]["piano"]["steps"] == 8
    assert locked["categories"]["drums"]["cfg_scale"] == 1.2


def test_build_locked_preset_config_with_bypass(tmp_path: Path):
    winners_path = tmp_path / "winners.yaml"
    winners_path.write_text(yaml.safe_dump(_winners_doc()))
    record_bypass_realify({"organ": True}, path=winners_path)

    locked = build_locked_preset_config(winners_path)
    assert locked["categories"]["organ"] == {"realify": False}
    assert locked["categories"]["piano"]["init_noise_level"] == 0.45


def test_build_locked_preset_config_without_phase3(tmp_path: Path):
    winners_path = tmp_path / "winners.yaml"
    winners_path.write_text(yaml.safe_dump({
        "phases": {
            PHASE1: {"completed": True, "winners": {"piano": "noise0.35"}},
            PHASE2: {"completed": True, "winners": {"piano": "current"}},
            PHASE3: {"completed": False, "winners": {}},
        }
    }))

    locked = build_locked_preset_config(winners_path)
    assert locked["categories"]["piano"]["steps"] == 8
    assert locked["categories"]["piano"]["cfg_scale"] == 1.0


def test_merge_locked_into_categories_yaml(tmp_path: Path):
    categories_path = tmp_path / "categories.yaml"
    categories_path.write_text(yaml.safe_dump({
        "default": {"init_noise_level": 0.45, "prompt_variant": "current", "steps": 8, "cfg_scale": 1.0},
        "categories": {"piano": {}},
        "routing": [{"category": "piano", "name_keywords": ["piano"]}],
    }))

    locked = {"categories": {"piano": {"init_noise_level": 0.35, "prompt_variant": "minimal", "steps": 8, "cfg_scale": 1.0}}}
    merged = merge_locked_into_categories_yaml(locked, categories_path)
    assert merged["categories"]["piano"]["init_noise_level"] == 0.35
    assert merged["routing"] == [{"category": "piano", "name_keywords": ["piano"]}]


def test_merge_locked_into_categories_yaml_adds_bypass_rules(tmp_path: Path):
    categories_path = tmp_path / "categories.yaml"
    categories_path.write_text(yaml.safe_dump({
        "default": {"init_noise_level": 0.45},
        "categories": {"voice": {"init_noise_level": 0.45}},
        "routing": [{"category": "voice", "name_keywords": ["voice"]}],
    }))

    locked = {
        "categories": {"voice": {"init_noise_level": 0.45, "prompt_variant": "current"}},
        "bypass_routing_rules": [{
            "category": "voice",
            "name_keywords": ["soprano"],
            "realify": False,
        }],
    }
    merged = merge_locked_into_categories_yaml(locked, categories_path)
    assert len(merged["routing"]) == 2
    assert merged["routing"][-1]["realify"] is False


def test_record_bypass_realify_rules(tmp_path: Path):
    path = tmp_path / "winners.yaml"
    path.write_text(yaml.safe_dump(_winners_doc()))
    record_bypass_realify_rules([{
        "category": "voice",
        "name_keywords": ["soprano"],
        "realify": False,
    }], path=path)
    locked = build_locked_preset_config(path)
    assert locked["bypass_routing_rules"][0]["name_keywords"] == ["soprano"]


def test_write_locked_config_requires_phases_1_and_2(tmp_path: Path):
    path = tmp_path / "winners.yaml"
    path.write_text(yaml.safe_dump({
        "phases": {
            PHASE1: {"completed": True, "winners": {"piano": "noise0.45"}},
            PHASE2: {"completed": False, "winners": {}},
            PHASE3: {"completed": False, "winners": {}},
        }
    }))
    with pytest.raises(RuntimeError, match=PHASE2):
        write_locked_config(
            path,
            tmp_path / "locked.yaml",
            tmp_path / "categories.yaml",
            update_categories_yaml=False,
        )
