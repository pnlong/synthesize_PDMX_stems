"""Tests for preset sweep experiment helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from experiments.preset_sweep.sweep import (
    MANIFEST_COLUMNS,
    build_manifest_rows,
    build_sweep_tasks,
    load_yaml,
    run_preset_sweep,
    song_path_from_id,
    variant_output_path,
    write_manifest,
)
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


def test_song_path_from_id(tmp_path: Path):
    source_dir = tmp_path / "basic"
    assert song_path_from_id(source_dir, "7/19/QmTest") == source_dir / "data" / "7/19/QmTest"


def test_variant_output_path(tmp_path: Path):
    source_dir = tmp_path / "basic"
    song_path = source_dir / "data" / "0/13/QmTest"
    out = variant_output_path(
        tmp_path / "output",
        "noise0.45_minimal",
        song_path,
        source_dir,
        0,
        "flac",
    )
    assert out == (
        tmp_path
        / "output"
        / "variants"
        / "noise0.45_minimal"
        / "data"
        / "0/13/QmTest"
        / "stem_0.flac"
    )


def test_build_sweep_tasks_and_manifest(tmp_path: Path):
    source_dir, song_dir = _write_basic_dataset(tmp_path)
    output_dir = tmp_path / "sweep"

    probe_stems = [{
        "id": "piano_test",
        "category": "piano",
        "song_id": "0/13/QmTest",
        "track": 0,
    }]
    variants = [
        {"id": "noise0.25_current", "init_noise_level": 0.25, "prompt_variant": "current"},
        {"id": "noise0.45_minimal", "init_noise_level": 0.45, "prompt_variant": "minimal"},
    ]

    tasks = build_sweep_tasks(
        source_dir=source_dir,
        output_dir=output_dir,
        probe_stems=probe_stems,
        variants=variants,
        audio_format="flac",
        sample_seed=42,
    )
    assert len(tasks) == 2
    assert tasks[0]["preset"]["init_noise_level"] == 0.25
    assert tasks[1]["preset"]["init_noise_level"] == 0.45
    assert "solo piano" in tasks[1]["row"]["prompt"]

    manifest_rows = build_manifest_rows(
        source_dir=source_dir,
        output_dir=output_dir,
        probe_stems=probe_stems,
        variants=variants,
        audio_format="flac",
        sample_seed=42,
    )
    manifest_path = write_manifest(output_dir, manifest_rows)
    manifest = pd.read_csv(manifest_path)
    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert len(manifest) == 2


def test_build_sweep_tasks_skip_existing(tmp_path: Path):
    source_dir, song_dir = _write_basic_dataset(tmp_path)
    output_dir = tmp_path / "sweep"

    probe_stems = [{
        "id": "piano_test",
        "category": "piano",
        "song_id": "0/13/QmTest",
        "track": 0,
    }]
    variants = [
        {"id": "noise0.25_current", "init_noise_level": 0.25, "prompt_variant": "current"},
        {"id": "noise0.45_minimal", "init_noise_level": 0.45, "prompt_variant": "minimal"},
    ]

    existing = variant_output_path(
        output_dir,
        "noise0.25_current",
        song_dir,
        source_dir,
        0,
        "flac",
    )
    existing.parent.mkdir(parents=True)
    sf.write(str(existing), np.zeros(44100), 44100, format="FLAC")

    tasks = build_sweep_tasks(
        source_dir=source_dir,
        output_dir=output_dir,
        probe_stems=probe_stems,
        variants=variants,
        audio_format="flac",
        sample_seed=42,
    )
    assert len(tasks) == 1
    assert tasks[0]["row"]["variant_id"] == "noise0.45_minimal"


def test_run_preset_sweep_writes_manifest_without_gpu(tmp_path: Path, monkeypatch):
    source_dir, _ = _write_basic_dataset(tmp_path)
    output_dir = tmp_path / "sweep"

    probe_path = tmp_path / "probe_stems.yaml"
    grid_path = tmp_path / "preset_grid.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [{
            "id": "piano_test",
            "category": "piano",
            "song_id": "0/13/QmTest",
            "track": 0,
        }],
    }))
    grid_path.write_text(yaml.dump({
        "variants": [
            {"id": "noise0.25_current", "init_noise_level": 0.25, "prompt_variant": "current"},
        ],
    }))

    monkeypatch.setattr(
        "experiments.preset_sweep.sweep.realify_uses_gpu",
        lambda model: False,
    )
    monkeypatch.setattr(
        "experiments.preset_sweep.sweep._run_realify_cpu",
        lambda *args, **kwargs: None,
    )

    manifest = run_preset_sweep(
        source_dir=source_dir,
        output_dir=output_dir,
        probe_stems_path=probe_path,
        preset_grid_path=grid_path,
        model="small-music",
        jobs=1,
    )
    assert len(manifest) == 1
    assert (output_dir / "manifest.csv").is_file()


def test_load_yaml_roundtrip(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text("variants:\n  - id: test\n")
    assert load_yaml(path)["variants"][0]["id"] == "test"
