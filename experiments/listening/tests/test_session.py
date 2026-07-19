"""Tests for listening session helpers."""

from pathlib import Path

import pandas as pd

from experiments.listening.catalog import SweepCatalog
from experiments.listening.session import rubric_for_catalog


def test_rubric_for_catalog_uses_phase1b_guidance(tmp_path: Path):
    sweep_dir = tmp_path / "phase1b"
    sweep_dir.mkdir()
    pd.DataFrame([{
        "phase": "phase1b_noise_audit",
        "variant_id": "noise0.45",
        "stem_id": "piano_test",
        "path": "/song",
        "track": 0,
        "out_path": str(sweep_dir / "variants/noise0.45/data/song/stem_0.flac"),
    }]).to_csv(sweep_dir / "manifest.csv", index=False)

    catalog = SweepCatalog("preset", sweep_dir)
    rubric = rubric_for_catalog(catalog)
    assert "played sections" in rubric["content_help"]
    assert "silence-corrected" in rubric["content_help"]


def test_rubric_for_catalog_default_preset(tmp_path: Path):
    sweep_dir = tmp_path / "phase1"
    sweep_dir.mkdir()
    pd.DataFrame([{
        "phase": "phase1_noise",
        "variant_id": "noise0.45",
        "stem_id": "piano_test",
        "path": "/song",
        "track": 0,
        "out_path": str(sweep_dir / "variants/noise0.45/data/song/stem_0.flac"),
    }]).to_csv(sweep_dir / "manifest.csv", index=False)

    catalog = SweepCatalog("preset", sweep_dir)
    rubric = rubric_for_catalog(catalog)
    assert rubric["content_help"] == "Same melody, rhythm, and timing as the reference?"
