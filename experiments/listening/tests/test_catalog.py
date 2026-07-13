"""Tests for sweep listening catalog."""

from pathlib import Path

import pandas as pd
import yaml

from experiments.listening.catalog import SweepCatalog


def _write_preset_sweep_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_dir = tmp_path / "basic"
    sweep_dir = tmp_path / "sweep"
    song_id = "0/13/QmTest"
    song_dir = source_dir / "data" / song_id
    song_dir.mkdir(parents=True)
    (song_dir / "stem_0.flac").write_bytes(b"fake")

    variant_dir = sweep_dir / "variants" / "noise0.25_current" / "data" / song_id
    variant_dir.mkdir(parents=True)
    (variant_dir / "stem_0.flac").write_bytes(b"fake")

    manifest = pd.DataFrame([{
        "variant_id": "noise0.25_current",
        "init_noise_level": 0.25,
        "prompt_variant": "current",
        "prompt": "solo piano",
        "stem_id": "piano_test",
        "category": "piano",
        "path": str(song_dir),
        "track": 0,
        "out_path": str(variant_dir / "stem_0.flac"),
    }])
    manifest.to_csv(sweep_dir / "manifest.csv", index=False)

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [{
            "id": "piano_test",
            "category": "piano",
            "song_id": song_id,
            "track": 0,
            "note": "test piano",
        }],
    }))
    return sweep_dir, source_dir, probe_path


def test_sweep_catalog_preset_lists_and_resolves_audio(tmp_path: Path):
    sweep_dir, source_dir, probe_path = _write_preset_sweep_tree(tmp_path)
    catalog = SweepCatalog(
        "preset",
        sweep_dir,
        source_dir,
        probe_stems_path=probe_path,
    )

    assert catalog.available() is True
    stems = catalog.list_stems()
    assert len(stems) == 1
    assert stems[0]["id"] == "piano_test"

    detail = catalog.get_stem_test("piano_test", session_seed=42)
    assert detail is not None
    assert detail["reference"]["available"] is True
    assert len(detail["samples"]) == 1
    assert detail["samples"][0]["audio"]["available"] is True

    ref = catalog.resolve_reference_audio("piano_test", "stem_0.flac")
    assert ref is not None
    var = catalog.resolve_variant_audio("noise0.25_current", "0/13/QmTest", "stem_0.flac")
    assert var is not None


def _write_patch_sweep_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_dir = tmp_path / "basic"
    sweep_dir = tmp_path / "sweep"
    song_id = "0/13/QmTest"
    song_dir = source_dir / "data" / song_id
    song_dir.mkdir(parents=True)
    (song_dir / "stem_0.flac").write_bytes(b"fake")

    variant_dir = sweep_dir / "variants" / "pool_v1" / "data" / song_id
    variant_dir.mkdir(parents=True)
    (variant_dir / "stem_0.flac").write_bytes(b"fake")

    manifest = pd.DataFrame([{
        "variant_id": "pool_v1",
        "pool_id": "pool_v1",
        "program": 0,
        "gm_class": "piano",
        "stem_id": "piano_test",
        "category": "piano",
        "path": str(song_dir),
        "track": 0,
        "out_path": str(variant_dir / "stem_0.flac"),
    }])
    manifest.to_csv(sweep_dir / "manifest.csv", index=False)

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [{
            "id": "piano_test",
            "category": "piano",
            "song_id": song_id,
            "track": 0,
        }],
    }))
    return sweep_dir, source_dir, probe_path


def test_sweep_catalog_patch_variants(tmp_path: Path):
    sweep_dir, source_dir, probe_path = _write_patch_sweep_tree(tmp_path)
    catalog = SweepCatalog(
        "patch",
        sweep_dir,
        source_dir,
        probe_stems_path=probe_path,
    )
    variants = catalog.variants()
    assert len(variants) == 1
    assert variants[0]["pool_id"] == "pool_v1"


def test_sweep_catalog_patch_phase1_empty_pool_id(tmp_path: Path):
    sweep_dir, source_dir, probe_path = _write_patch_sweep_tree(tmp_path)
    manifest = pd.DataFrame([{
        "phase": "phase1_soundfonts",
        "variant_id": "sgm_v2",
        "soundfont_id": "sgm_v2",
        "soundfont_file": "SGM-V2.01.sf2",
        "fx_profile": "dry",
        "pool_id": "",
        "program": 0,
        "gm_class": "piano",
        "stem_id": "piano_test",
        "category": "piano",
        "path": str(source_dir / "data" / "0/13/QmTest"),
        "track": 0,
        "out_path": str(
            sweep_dir / "variants" / "sgm_v2" / "data" / "0/13/QmTest" / "stem_0.flac"
        ),
    }])
    manifest.to_csv(sweep_dir / "manifest.csv", index=False)

    catalog = SweepCatalog(
        "patch",
        sweep_dir,
        source_dir,
        probe_stems_path=probe_path,
    )
    variants = catalog.variants()
    assert variants[0]["pool_id"] == ""
