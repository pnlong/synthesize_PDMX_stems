"""Tests for preset sweep experiment helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from experiments.preset_sweep.config import (
    PHASE1,
    PHASE1B,
    PHASE2,
    PHASE3,
    load_yaml,
    resolve_silence_enforce,
)
from experiments.preset_sweep.sweep import (
    MANIFEST_COLUMNS,
    build_manifest_rows,
    build_sweep_tasks,
    resolve_preset_settings,
    run_preset_sweep,
    song_path_from_id,
    variant_output_path,
    write_manifest,
)
from experiments.preset_sweep.winners import record_phase_winners
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
        "noise0.45",
        song_path,
        source_dir,
        0,
        "flac",
    )
    assert out == (
        tmp_path
        / "output"
        / "variants"
        / "noise0.45"
        / "data"
        / "0/13/QmTest"
        / "stem_0.flac"
    )


def test_resolve_preset_settings_phase1():
    noise, prompt, steps, cfg = resolve_preset_settings(
        phase=PHASE1,
        variant={"id": "noise0.45", "init_noise_level": 0.45},
        grid_cfg={"prompt_variant": "current", "steps": 8, "cfg_scale": 1.0},
        category="piano",
        winners_path=Path("/nonexistent/winners.yaml"),
    )
    assert noise == 0.45
    assert prompt == "current"
    assert steps == 8
    assert cfg == 1.0


def test_resolve_preset_settings_phase2(tmp_path: Path):
    winners_path = tmp_path / "winners.yaml"
    record_phase_winners(PHASE1, {"piano": "noise0.55"}, path=winners_path)

    noise, prompt, steps, cfg = resolve_preset_settings(
        phase=PHASE2,
        variant={"id": "minimal", "prompt_variant": "minimal"},
        grid_cfg={"steps": 8, "cfg_scale": 1.0},
        category="piano",
        winners_path=winners_path,
    )
    assert noise == 0.55
    assert prompt == "minimal"


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
        {"id": "noise0.25", "init_noise_level": 0.25},
        {"id": "noise0.45", "init_noise_level": 0.45},
    ]
    grid_cfg = {"prompt_variant": "current", "steps": 8, "cfg_scale": 1.0}
    winners_path = tmp_path / "winners.yaml"

    tasks = build_sweep_tasks(
        phase=PHASE1,
        source_dir=source_dir,
        output_dir=output_dir,
        probe_stems=probe_stems,
        variants=variants,
        grid_cfg=grid_cfg,
        audio_format="flac",
        sample_seed=42,
        winners_path=winners_path,
    )
    assert len(tasks) == 2
    assert tasks[0]["preset"]["init_noise_level"] == 0.25
    assert tasks[1]["preset"]["init_noise_level"] == 0.45
    assert "piano" in tasks[1]["row"]["prompt"].lower()

    manifest_rows = build_manifest_rows(
        phase=PHASE1,
        source_dir=source_dir,
        output_dir=output_dir,
        probe_stems=probe_stems,
        variants=variants,
        grid_cfg=grid_cfg,
        audio_format="flac",
        sample_seed=42,
        winners_path=winners_path,
    )
    manifest_path = write_manifest(output_dir, manifest_rows)
    manifest = pd.read_csv(manifest_path)
    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert len(manifest) == 2
    assert manifest.iloc[0]["phase"] == PHASE1


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
        {"id": "noise0.25", "init_noise_level": 0.25},
        {"id": "noise0.45", "init_noise_level": 0.45},
    ]
    grid_cfg = {"prompt_variant": "current", "steps": 8, "cfg_scale": 1.0}

    existing = variant_output_path(
        output_dir,
        "noise0.25",
        song_dir,
        source_dir,
        0,
        "flac",
    )
    existing.parent.mkdir(parents=True)
    sf.write(str(existing), np.zeros(44100), 44100, format="FLAC")

    tasks = build_sweep_tasks(
        phase=PHASE1,
        source_dir=source_dir,
        output_dir=output_dir,
        probe_stems=probe_stems,
        variants=variants,
        grid_cfg=grid_cfg,
        audio_format="flac",
        sample_seed=42,
        winners_path=tmp_path / "winners.yaml",
    )
    assert len(tasks) == 1
    assert tasks[0]["row"]["variant_id"] == "noise0.45"


def test_run_preset_sweep_writes_manifest_without_gpu(tmp_path: Path, monkeypatch):
    source_dir, _ = _write_basic_dataset(tmp_path)
    output_dir = tmp_path / "sweep"

    probe_path = tmp_path / "probe_stems.yaml"
    grid_path = tmp_path / "phase1_noise.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [{
            "id": "piano_test",
            "category": "piano",
            "song_id": "0/13/QmTest",
            "track": 0,
        }],
    }))
    grid_path.write_text(yaml.dump({
        "prompt_variant": "current",
        "steps": 8,
        "cfg_scale": 1.0,
        "variants": [
            {"id": "noise0.25", "init_noise_level": 0.25},
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
    monkeypatch.setattr(
        "experiments.preset_sweep.sweep.validate_probe_stems",
        lambda stems, **kwargs: None,
    )

    manifest = run_preset_sweep(
        phase=PHASE1,
        source_dir=source_dir,
        output_dir=output_dir,
        probe_stems_path=probe_path,
        grid_path=grid_path,
        winners_path=tmp_path / "winners.yaml",
        model="small-music",
        jobs=1,
    )
    assert len(manifest) == 1
    assert (output_dir / "phase1_noise" / "manifest.csv").is_file()


def test_resolve_silence_enforce_defaults_phase1b_only():
    assert resolve_silence_enforce(PHASE1, {}) is False
    assert resolve_silence_enforce(PHASE1B, {}) is True
    assert resolve_silence_enforce(PHASE2, {}) is False
    assert resolve_silence_enforce(PHASE1, {"silence_enforce": True}) is True


def test_run_phase1b_passes_silence_enforce(tmp_path: Path, monkeypatch):
    source_dir, _ = _write_basic_dataset(tmp_path)
    output_dir = tmp_path / "sweep"
    winners_path = tmp_path / "winners.yaml"
    record_phase_winners(PHASE1, {"piano": "noise0.45"}, path=winners_path)

    grid_path = tmp_path / "phase1b.yaml"
    grid_path.write_text(yaml.dump({
        "prompt_variant": "current",
        "steps": 8,
        "cfg_scale": 1.0,
        "silence_enforce": True,
        "clip_seconds": 10,
        "diverse_stems_per_category": 1,
        "min_rms": 0.01,
        "noise_levels": [0.25, 0.35, 0.45, 0.55, 0.65],
        "variants": [],
    }))

    captured = {}

    def fake_cpu(*args, **kwargs):
        captured["silence_enforce"] = kwargs.get("silence_enforce")

    def fake_prepare(**kwargs):
        return source_dir, [{
            "id": "piano_test",
            "category": "piano",
            "song_id": "0/13/QmTest",
            "track": 0,
        }], [
            {"id": "noise0.35", "init_noise_level": 0.35},
            {"id": "noise0.45", "init_noise_level": 0.45},
        ]

    monkeypatch.setattr(
        "experiments.preset_sweep.sweep.realify_uses_gpu",
        lambda model: False,
    )
    monkeypatch.setattr(
        "experiments.preset_sweep.sweep._run_realify_cpu",
        fake_cpu,
    )
    monkeypatch.setattr(
        "experiments.preset_sweep.sweep.prepare_phase1b_audit",
        fake_prepare,
    )

    probe_path = tmp_path / "probe.yaml"
    probe_path.write_text(yaml.dump({"stems": []}))

    run_preset_sweep(
        phase=PHASE1B,
        source_dir=source_dir,
        output_dir=output_dir,
        probe_stems_path=probe_path,
        grid_path=grid_path,
        winners_path=winners_path,
        model="small-music",
        jobs=1,
    )
    assert captured["silence_enforce"] is True


def test_phase2_requires_phase1_and_phase1b_winners(tmp_path: Path):
    source_dir, _ = _write_basic_dataset(tmp_path)
    probe_path = tmp_path / "probe_stems.yaml"
    grid_path = tmp_path / "phase2_prompts.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [{
            "id": "piano_test",
            "category": "piano",
            "song_id": "0/13/QmTest",
            "track": 0,
        }],
    }))
    grid_path.write_text(yaml.dump({
        "variants": [{"id": "current", "prompt_variant": "current"}],
    }))
    winners_path = tmp_path / "winners.yaml"

    with pytest.raises(RuntimeError, match=PHASE1):
        run_preset_sweep(
            phase=PHASE2,
            source_dir=source_dir,
            output_dir=tmp_path / "sweep",
            probe_stems_path=probe_path,
            grid_path=grid_path,
            winners_path=winners_path,
            model="small-music",
            jobs=1,
        )

    record_phase_winners(PHASE1, {"piano": "noise0.45"}, path=winners_path)
    with pytest.raises(RuntimeError, match="phase1b_noise_audit"):
        run_preset_sweep(
            phase=PHASE2,
            source_dir=source_dir,
            output_dir=tmp_path / "sweep",
            probe_stems_path=probe_path,
            grid_path=grid_path,
            winners_path=winners_path,
            model="small-music",
            jobs=1,
        )


def test_load_yaml_roundtrip(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text("variants:\n  - id: test\n")
    assert load_yaml(path)["variants"][0]["id"] == "test"
