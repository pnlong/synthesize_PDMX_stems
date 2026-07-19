"""Tests for phased patch sweep winner recording."""

from pathlib import Path

import pytest
import yaml

from experiments.patch_sweep.config import PHASE1, PHASE2, REQUIRED_LOCK_PHASES
from experiments.patch_sweep.winners import (
    build_locked_render_config,
    fx_profile_from_phase2_variant_id,
    load_winners,
    pick_soundfont_id,
    phase1_soundfont_ids,
    record_phase_winners,
    soundfont_seed,
    write_locked_config,
)


def _winners_doc():
    return {
        "phases": {
            PHASE1: {
                "completed": True,
                "winners": {
                    "piano": ["sgm_v2", "airfont_380"],
                    "strings": "arachno",
                },
            },
            PHASE2: {
                "completed": True,
                "winners": {"piano": "fx_light", "strings": "fx_dry"},
            },
        }
    }


def test_fx_profile_from_phase2_variant_id():
    assert fx_profile_from_phase2_variant_id("fx_dry") == "dry"
    assert fx_profile_from_phase2_variant_id("fx_light") == "light"
    assert fx_profile_from_phase2_variant_id("fx_warm") == "warm"


def test_phase1_soundfont_ids_normalizes_scalar():
    path = Path("/tmp/winners.yaml")
    path.write_text(yaml.safe_dump(_winners_doc()))
    assert phase1_soundfont_ids("piano", path) == ["sgm_v2", "airfont_380"]
    assert phase1_soundfont_ids("strings", path) == ["arachno"]


def test_pick_soundfont_id_stable_per_song_category():
    ids = ["sgm_v2", "airfont_380", "generaluser"]
    a = pick_soundfont_id(
        ids,
        category="piano",
        song_path="/song/a",
        sample_seed=42,
    )
    b = pick_soundfont_id(
        ids,
        category="piano",
        song_path="/song/a",
        sample_seed=42,
    )
    assert a == b
    assert a in ids


def test_pick_soundfont_id_can_differ_across_songs():
    ids = ["sgm_v2", "airfont_380", "generaluser", "arachno"]
    picks = {
        pick_soundfont_id(
            ids,
            category="strings",
            song_path=f"/song/{i}",
            sample_seed=7,
        )
        for i in range(6)
    }
    assert len(picks) >= 2


def test_soundfont_seed_differs_by_category():
    assert soundfont_seed(42, "/song", "piano") != soundfont_seed(42, "/song", "strings")


def test_record_phase_winners_shortlist(tmp_path: Path):
    path = tmp_path / "winners.yaml"
    record_phase_winners(
        PHASE1,
        {"piano": ["sgm_v2", "airfont_380"]},
        path=path,
    )
    doc = load_winners(path)
    assert doc["phases"][PHASE1]["winners"]["piano"] == ["sgm_v2", "airfont_380"]


def test_build_locked_render_config(tmp_path: Path):
    winners_path = tmp_path / "winners.yaml"
    winners_path.write_text(yaml.safe_dump(_winners_doc()))

    catalog_path = tmp_path / "soundfonts.yaml"
    catalog_path.write_text(yaml.safe_dump({
        "candidates": [
            {"id": "sgm_v2", "file": "SGM-V2.01.sf2"},
            {"id": "airfont_380", "file": "airfont_380_final.sf2"},
            {"id": "arachno", "file": "Arachno.sf2"},
        ]
    }))

    locked = build_locked_render_config(winners_path, catalog_path)
    assert locked["categories"]["piano"]["soundfont_ids"] == ["sgm_v2", "airfont_380"]
    assert locked["categories"]["piano"]["soundfont"] == "SGM-V2.01.sf2"
    assert locked["categories"]["piano"]["fx_profile"] == "light"
    assert "pool_id" not in locked["categories"]["piano"]


def test_write_locked_config_requires_phases_1_and_2(tmp_path: Path):
    path = tmp_path / "winners.yaml"
    path.write_text(yaml.safe_dump({
        "phases": {
            PHASE1: {"completed": True, "winners": {"piano": ["sgm_v2"]}},
            PHASE2: {"completed": False, "winners": {}},
        }
    }))
    with pytest.raises(RuntimeError, match=PHASE2):
        write_locked_config(path, tmp_path / "locked.yaml")

    assert REQUIRED_LOCK_PHASES == (PHASE1, PHASE2)
