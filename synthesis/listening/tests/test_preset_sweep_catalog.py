"""Tests for preset-sweep listening catalog."""

from pathlib import Path

import pandas as pd
import yaml

from synthesis.listening.preset_sweep_catalog import PresetSweepCatalog


def _write_sweep_tree(tmp_path: Path) -> tuple[Path, Path]:
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


def test_preset_sweep_catalog_lists_and_resolves_audio(tmp_path: Path):
    sweep_dir, source_dir, probe_path = _write_sweep_tree(tmp_path)
    catalog = PresetSweepCatalog(sweep_dir, source_dir, probe_stems_path=probe_path)

    assert catalog.available() is True
    stems = catalog.list_stems()
    assert len(stems) == 1
    assert stems[0]["id"] == "piano_test"

    detail = catalog.get_stem("piano_test")
    assert detail is not None
    assert detail["reference"]["available"] is True
    assert detail["variants"][0]["audio"]["available"] is True

    ref = catalog.resolve_reference_audio("piano_test", "stem_0.flac")
    assert ref is not None
    var = catalog.resolve_variant_audio("noise0.25_current", "0/13/QmTest", "stem_0.flac")
    assert var is not None
