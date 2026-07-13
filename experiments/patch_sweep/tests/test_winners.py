"""Tests for phased patch sweep winner recording."""

from pathlib import Path

import pytest
import yaml

from experiments.patch_sweep.config import PHASE1, PHASE2, PHASE3
from experiments.patch_sweep.winners import (
    build_locked_render_config,
    load_winners,
    record_phase_winners,
    write_locked_config,
)


def _winners_doc():
    return {
        "phases": {
            PHASE1: {"completed": True, "winners": {"piano": "sgm_v2", "strings": "arachno"}},
            PHASE2: {"completed": True, "winners": {"piano": "fx_light", "strings": "fx_dry"}},
            PHASE3: {"completed": True, "winners": {"piano": "pool_v2_diverse", "strings": "pool_v1_conservative"}},
        }
    }


def test_record_phase_winners(tmp_path: Path):
    path = tmp_path / "winners.yaml"
    record_phase_winners(PHASE1, {"piano": "sgm_v2"}, path=path)
    doc = load_winners(path)
    assert doc["phases"][PHASE1]["completed"] is True
    assert doc["phases"][PHASE1]["winners"]["piano"] == "sgm_v2"


def test_build_locked_render_config(tmp_path: Path):
    winners_path = tmp_path / "winners.yaml"
    winners_path.write_text(yaml.safe_dump(_winners_doc()))

    catalog_path = tmp_path / "soundfonts.yaml"
    catalog_path.write_text(yaml.safe_dump({
        "candidates": [
            {"id": "sgm_v2", "file": "SGM-V2.01.sf2"},
            {"id": "arachno", "file": "Arachno.sf2"},
        ]
    }))

    locked = build_locked_render_config(winners_path, catalog_path)
    assert locked["categories"]["piano"]["soundfont"] == "SGM-V2.01.sf2"
    assert locked["categories"]["piano"]["fx_profile"] == "fx_light"
    assert locked["categories"]["piano"]["pool_id"] == "pool_v2_diverse"


def test_write_locked_config_requires_all_phases(tmp_path: Path):
    path = tmp_path / "winners.yaml"
    path.write_text(yaml.safe_dump({
        "phases": {
            PHASE1: {"completed": True, "winners": {"piano": "sgm_v2"}},
            PHASE2: {"completed": False, "winners": {}},
            PHASE3: {"completed": False, "winners": {}},
        }
    }))
    with pytest.raises(RuntimeError, match=PHASE2):
        write_locked_config(path, tmp_path / "locked.yaml")
