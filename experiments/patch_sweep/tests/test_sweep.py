"""Tests for patch sweep experiment helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from experiments.patch_sweep.config import PHASE1, PHASE3
from experiments.patch_sweep.sweep import (
    MANIFEST_COLUMNS,
    build_manifest_rows,
    build_sweep_tasks,
    song_path_from_id,
    variant_output_path,
    write_manifest,
)
from experiments.probe_stems import resolve_mid_path
from shared.config import DATA_DIR_NAME


def _write_basic_dataset(tmp_path: Path) -> tuple[Path, Path]:
    source_dir = tmp_path / "basic"
    song_id = "0/13/QmTest"
    song_dir = source_dir / DATA_DIR_NAME / song_id
    song_dir.mkdir(parents=True)

    sr = 44100
    sf.write(str(song_dir / "stem_0.flac"), np.zeros(sr), sr, format="FLAC")

    pd.DataFrame({
        "path": [str(song_dir)],
        "title": ["Test Song"],
        "genres": ["classical"],
    }).to_csv(source_dir / f"{DATA_DIR_NAME}.csv", index=False)
    pd.DataFrame({
        "path": [str(song_dir)],
        "track": [0],
        "program": [0],
        "is_drum": [False],
        "name": ["Piano"],
        "has_lyrics": [False],
    }).to_csv(source_dir / "stems.csv", index=False)
    return source_dir, song_dir


def _write_soundfont_catalog(tmp_path: Path) -> Path:
    catalog = {
        "candidates": [
            {"id": "sgm_v2", "file": "SGM-V2.01.sf2"},
        ]
    }
    path = tmp_path / "soundfonts.yaml"
    path.write_text(yaml.safe_dump(catalog))
    return path


def test_song_path_from_id(tmp_path: Path):
    source_dir = tmp_path / "basic"
    assert song_path_from_id(source_dir, "7/19/QmTest") == source_dir / "data" / "7/19/QmTest"


def test_variant_output_path(tmp_path: Path):
    source_dir = tmp_path / "basic"
    song_path = source_dir / "data" / "0/13/QmTest"
    out = variant_output_path(
        tmp_path / "output",
        "pool_v1_conservative",
        song_path,
        source_dir,
        0,
        "flac",
    )
    assert out == (
        tmp_path
        / "output"
        / "variants"
        / "pool_v1_conservative"
        / "data"
        / "0/13/QmTest"
        / "stem_0.flac"
    )


def test_build_manifest_rows_phase1(tmp_path: Path, monkeypatch):
    source_dir, song_dir = _write_basic_dataset(tmp_path)
    output_dir = tmp_path / "sweep"
    catalog_path = _write_soundfont_catalog(tmp_path)
    sf_dir = tmp_path / "fonts"
    sf_dir.mkdir()
    (sf_dir / "SGM-V2.01.sf2").write_bytes(b"sf2")

    monkeypatch.setattr(
        "experiments.patch_sweep.sweep.load_soundfont_catalog",
        lambda: yaml.safe_load(catalog_path.read_text()),
    )

    probe_stems = [{
        "id": "piano_test",
        "category": "piano",
        "song_id": "0/13/QmTest",
        "track": 0,
    }]
    variants = [{
        "id": "sgm_v2",
        "soundfont_id": "sgm_v2",
    }]
    grid_cfg = {"fx_profile": "dry", "pool_id": None}

    rows = build_manifest_rows(
        phase=PHASE1,
        source_dir=source_dir,
        output_dir=output_dir,
        probe_stems=probe_stems,
        variants=variants,
        grid_cfg=grid_cfg,
        audio_format="flac",
        catalog=yaml.safe_load(catalog_path.read_text()),
        winners_path=tmp_path / "winners.yaml",
        soundfont_dir=sf_dir,
    )
    assert len(rows) == 1
    assert rows[0]["variant_id"] == "sgm_v2"
    assert rows[0]["soundfont_id"] == "sgm_v2"
    assert rows[0]["fx_profile"] == "dry"


def test_build_sweep_tasks_skips_existing(tmp_path: Path, monkeypatch):
    source_dir, song_dir = _write_basic_dataset(tmp_path)
    output_dir = tmp_path / "sweep"
    catalog_path = _write_soundfont_catalog(tmp_path)
    sf_dir = tmp_path / "fonts"
    sf_dir.mkdir()
    (sf_dir / "SGM-V2.01.sf2").write_bytes(b"sf2")

    mid_path = tmp_path / "song.mid"
    mid_path.write_bytes(b"")
    monkeypatch.setattr(
        "experiments.patch_sweep.sweep.resolve_mid_path",
        lambda song_id, pdmx_filepath=None: mid_path,
    )
    monkeypatch.setattr(
        "experiments.patch_sweep.sweep.load_soundfont_catalog",
        lambda: yaml.safe_load(catalog_path.read_text()),
    )

    variant_dir = output_dir / "variants" / "sgm_v2" / "data" / "0/13/QmTest"
    variant_dir.mkdir(parents=True)
    sf.write(str(variant_dir / "stem_0.flac"), np.zeros(44100), 44100, format="FLAC")

    probe_stems = [{
        "id": "piano_test",
        "category": "piano",
        "song_id": "0/13/QmTest",
        "track": 0,
    }]
    variants = [{
        "id": "sgm_v2",
        "soundfont_id": "sgm_v2",
    }]

    tasks = build_sweep_tasks(
        phase=PHASE1,
        source_dir=source_dir,
        output_dir=output_dir,
        probe_stems=probe_stems,
        variants=variants,
        grid_cfg={"fx_profile": "dry", "pool_id": None},
        audio_format="flac",
        sample_seed=42,
        catalog=yaml.safe_load(catalog_path.read_text()),
        winners_path=tmp_path / "winners.yaml",
        soundfont_dir=sf_dir,
        pdmx_filepath=str(tmp_path / "PDMX.csv"),
    )
    assert tasks == []


def test_write_manifest(tmp_path: Path):
    output_dir = tmp_path / "out"
    rows = [{
        "phase": PHASE3,
        "variant_id": "pool_v1",
        "soundfont_id": "sgm_v2",
        "soundfont_file": "SGM-V2.01.sf2",
        "fx_profile": "light",
        "pool_id": "pool_v1",
        "program": 0,
        "gm_class": "piano",
        "stem_id": "piano_test",
        "category": "piano",
        "path": "/song",
        "track": 0,
        "out_path": "/out/stem.flac",
    }]
    path = write_manifest(output_dir, rows)
    df = pd.read_csv(path)
    assert list(df.columns) == MANIFEST_COLUMNS
    assert len(df) == 1


def test_resolve_mid_path(tmp_path: Path):
    pdmx_root = tmp_path / "PDMX"
    pdmx_root.mkdir()
    mid_path = pdmx_root / "mid" / "0/13/QmTest.mid"
    mid_path.parent.mkdir(parents=True)
    mid_path.write_bytes(b"MThd")
    pdmx_csv = pdmx_root / "PDMX.csv"
    pd.DataFrame({
        "path": ["./data/0/13/QmTest.json"],
        "mid": ["./mid/0/13/QmTest.mid"],
    }).to_csv(pdmx_csv, index=False)
    resolved = resolve_mid_path("0/13/QmTest", pdmx_filepath=str(pdmx_csv))
    assert resolved == mid_path


def test_resolve_mid_path_missing(tmp_path: Path):
    pdmx_root = tmp_path / "PDMX"
    pdmx_root.mkdir()
    pdmx_csv = pdmx_root / "PDMX.csv"
    pd.DataFrame({
        "path": ["./data/0/13/QmOther.json"],
        "mid": ["./mid/0/13/QmOther.mid"],
    }).to_csv(pdmx_csv, index=False)
    with pytest.raises(FileNotFoundError, match="not found in PDMX"):
        resolve_mid_path("0/13/QmMissing", pdmx_filepath=str(pdmx_csv))


def test_resolve_mid_path_missing_file(tmp_path: Path):
    pdmx_root = tmp_path / "PDMX"
    pdmx_root.mkdir()
    pdmx_csv = pdmx_root / "PDMX.csv"
    pd.DataFrame({
        "path": ["./data/0/13/QmTest.json"],
        "mid": ["./mid/0/13/QmTest.mid"],
    }).to_csv(pdmx_csv, index=False)
    with pytest.raises(FileNotFoundError, match="MIDI not found"):
        resolve_mid_path("0/13/QmTest", pdmx_filepath=str(pdmx_csv))
