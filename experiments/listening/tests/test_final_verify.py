"""Tests for end-of-pipeline verification."""

from pathlib import Path

import yaml

from experiments.listening.final_verify import (
    composed_config,
    final_phase_winners,
    readiness_errors,
    verification_phase,
)
from experiments.preset_sweep.config import LOCKED_VERIFY_VARIANT, PHASE4, phase_output_dir


def _write_preset_winners(path: Path, *, phase3_complete: bool = False, phase1b_complete: bool = True):
    doc = {
        "phases": {
            "phase1_noise": {
                "completed": True,
                "winners": {"piano": "noise0.45"},
            },
            "phase1b_noise_audit": {
                "completed": phase1b_complete,
                "winners": {"piano": "noise0.45"},
            },
            "phase2_prompts": {
                "completed": True,
                "winners": {"piano": "minimal"},
            },
            "phase3_diffusion": {
                "completed": phase3_complete,
                "winners": {"piano": "steps8_cfg1.0"} if phase3_complete else {},
            },
        },
    }
    path.write_text(yaml.dump(doc))


def test_verification_phase_uses_phase4_when_manifest_exists(tmp_path: Path, monkeypatch):
    winners_path = tmp_path / "winners.yaml"
    _write_preset_winners(winners_path, phase3_complete=True)
    sweep_root = tmp_path / "sweep"
    phase4_dir = phase_output_dir(sweep_root, PHASE4)
    phase4_dir.mkdir(parents=True)
    (phase4_dir / "manifest.csv").write_text("phase,variant_id\n")

    monkeypatch.setattr(
        "experiments.listening.final_verify.winners_path_for",
        lambda sweep_type, path=None: winners_path,
    )
    monkeypatch.setattr(
        "experiments.listening.final_verify.preset_default_output_dir",
        lambda output_root=None: sweep_root,
    )
    assert verification_phase("preset", winners_path) == PHASE4


def test_final_phase_winners_use_locked_variant_for_phase4(tmp_path: Path, monkeypatch):
    winners_path = tmp_path / "winners.yaml"
    _write_preset_winners(winners_path, phase3_complete=True)
    sweep_root = tmp_path / "sweep"
    phase4_dir = phase_output_dir(sweep_root, PHASE4)
    phase4_dir.mkdir(parents=True)
    (phase4_dir / "manifest.csv").write_text("phase,variant_id\n")

    monkeypatch.setattr(
        "experiments.listening.final_verify.winners_path_for",
        lambda sweep_type, path=None: winners_path,
    )
    monkeypatch.setattr(
        "experiments.listening.final_verify.preset_default_output_dir",
        lambda output_root=None: sweep_root,
    )
    assert final_phase_winners("preset", winners_path) == {
        "piano": LOCKED_VERIFY_VARIANT,
    }


def test_composed_config_phase4_uses_locked_winners(tmp_path: Path, monkeypatch):
    winners_path = tmp_path / "winners.yaml"
    _write_preset_winners(winners_path, phase3_complete=True)
    sweep_root = tmp_path / "sweep"
    phase4_dir = phase_output_dir(sweep_root, PHASE4)
    phase4_dir.mkdir(parents=True)
    (phase4_dir / "manifest.csv").write_text("phase,variant_id\n")

    monkeypatch.setattr(
        "experiments.listening.final_verify.winners_path_for",
        lambda sweep_type, path=None: winners_path,
    )
    monkeypatch.setattr(
        "experiments.listening.final_verify.preset_default_output_dir",
        lambda output_root=None: sweep_root,
    )
    config = composed_config("preset", "piano", LOCKED_VERIFY_VARIANT, winners_path)
    assert config["variant_id"] == LOCKED_VERIFY_VARIANT
    assert config["init_noise_level"] == 0.45
    assert config["prompt_variant"] == "minimal"
    assert config["steps"] == 8


def test_verification_phase_uses_phase2_when_phase3_incomplete(tmp_path: Path, monkeypatch):
    winners_path = tmp_path / "winners.yaml"
    _write_preset_winners(winners_path, phase3_complete=False)
    monkeypatch.setattr(
        "experiments.listening.final_verify.PRESET_EXPERIMENT_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        "experiments.listening.final_verify.winners_path_for",
        lambda sweep_type, path=None: winners_path,
    )
    assert verification_phase("preset", winners_path) == "phase2_prompts"


def test_verification_phase_uses_phase3_when_complete(tmp_path: Path, monkeypatch):
    winners_path = tmp_path / "winners.yaml"
    _write_preset_winners(winners_path, phase3_complete=True)
    monkeypatch.setattr(
        "experiments.listening.final_verify.winners_path_for",
        lambda sweep_type, path=None: winners_path,
    )
    assert verification_phase("preset", winners_path) == "phase3_diffusion"


def test_readiness_errors_when_phase1_incomplete(tmp_path: Path, monkeypatch):
    winners_path = tmp_path / "winners.yaml"
    winners_path.write_text(yaml.dump({
        "phases": {
            "phase1_noise": {"completed": False, "winners": {}},
            "phase2_prompts": {"completed": False, "winners": {}},
            "phase3_diffusion": {"completed": False, "winners": {}},
        },
    }))
    monkeypatch.setattr(
        "experiments.listening.final_verify.winners_path_for",
        lambda sweep_type, path=None: winners_path,
    )
    errors = readiness_errors("preset", winners_path)
    assert any("phase1_noise" in err for err in errors)


def test_verification_phase_patch_uses_phase1(tmp_path: Path, monkeypatch):
    winners_path = tmp_path / "winners.yaml"
    winners_path.write_text(yaml.dump({
        "phases": {
            "phase1_soundfonts": {"completed": True, "winners": {"piano": ["sgm_v2"]}},
            "phase2_fx": {"completed": True, "winners": {"piano": "fx_light"}},
        },
    }))
    monkeypatch.setattr(
        "experiments.listening.final_verify.winners_path_for",
        lambda sweep_type, path=None: winners_path,
    )
    assert verification_phase("patch", winners_path) == "phase1_soundfonts"


def test_final_phase_winners(tmp_path: Path, monkeypatch):
    winners_path = tmp_path / "winners.yaml"
    _write_preset_winners(winners_path, phase3_complete=True)
    monkeypatch.setattr(
        "experiments.listening.final_verify.winners_path_for",
        lambda sweep_type, path=None: winners_path,
    )
    winners = __import__(
        "experiments.listening.final_verify",
        fromlist=["final_phase_winners"],
    ).final_phase_winners("preset", winners_path)
    assert winners == {"piano": "steps8_cfg1.0"}


def test_composed_config_patch_soundfont(tmp_path: Path, monkeypatch):
    winners_path = tmp_path / "winners.yaml"
    winners_path.write_text(yaml.dump({
        "phases": {
            "phase1_soundfonts": {
                "completed": True,
                "winners": {"piano": ["sgm_v2"]},
            },
            "phase2_fx": {
                "completed": True,
                "winners": {"piano": "fx_light"},
            },
        },
    }))
    monkeypatch.setattr(
        "experiments.listening.final_verify.winners_path_for",
        lambda sweep_type, path=None: winners_path,
    )
    config = composed_config("patch", "piano", "sgm_v2", winners_path)
    assert config == {
        "variant_id": "sgm_v2",
        "soundfont_id": "sgm_v2",
        "fx_profile": "dry",
    }
