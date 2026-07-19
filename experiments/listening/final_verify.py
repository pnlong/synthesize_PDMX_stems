"""Final verification after all tuning phases complete (before lock)."""

from __future__ import annotations

from pathlib import Path

from experiments.listening.catalog import SweepCatalog
from experiments.patch_sweep.config import (
    EXPERIMENT_DIR as PATCH_EXPERIMENT_DIR,
    PHASE1 as PATCH_PHASE1,
    PHASE2 as PATCH_PHASE2,
    PHASE3 as PATCH_PHASE3,
    PHASES as PATCH_PHASES,
    phase_output_dir as patch_phase_output_dir,
)
from experiments.patch_sweep.sweep import default_output_dir as patch_default_output_dir
from experiments.patch_sweep.winners import (
    load_winners as load_patch_winners,
    phase_is_complete as patch_phase_is_complete,
    phase_winners as patch_phase_winners,
)
from experiments.preset_sweep.config import (
    EXPERIMENT_DIR as PRESET_EXPERIMENT_DIR,
    LOCKED_VERIFY_VARIANT,
    PHASE1 as PRESET_PHASE1,
    PHASE1B as PRESET_PHASE1B,
    PHASE2 as PRESET_PHASE2,
    PHASE3 as PRESET_PHASE3,
    PHASE4 as PRESET_PHASE4,
    TUNING_PHASES as PRESET_TUNING_PHASES,
    REQUIRED_LOCK_PHASES as PRESET_REQUIRED_LOCK_PHASES,
    phase_output_dir as preset_phase_output_dir,
)
from experiments.preset_sweep.sweep import default_output_dir as preset_default_output_dir
from experiments.preset_sweep.winners import (
    load_winners as load_preset_winners,
    phase_is_complete as preset_phase_is_complete,
    phase_winners as preset_phase_winners,
    _diffusion_from_variant_id,
)
from experiments.preset_sweep.config import init_noise_level_from_variant_id


def _patch_config():
    return {
        "experiment_dir": PATCH_EXPERIMENT_DIR,
        "phases": PATCH_PHASES,
        "required_phases": PATCH_PHASES,
        "phase1": PATCH_PHASE1,
        "phase2": PATCH_PHASE2,
        "load_winners": load_patch_winners,
        "phase_is_complete": patch_phase_is_complete,
        "phase_winners": patch_phase_winners,
        "default_output_dir": patch_default_output_dir,
        "phase_output_dir": patch_phase_output_dir,
    }


def _preset_config():
    return {
        "experiment_dir": PRESET_EXPERIMENT_DIR,
        "phases": PRESET_TUNING_PHASES,
        "required_phases": PRESET_REQUIRED_LOCK_PHASES,
        "phase1": PRESET_PHASE1,
        "phase1b": PRESET_PHASE1B,
        "phase2": PRESET_PHASE2,
        "phase3": PRESET_PHASE3,
        "phase4": PRESET_PHASE4,
        "load_winners": load_preset_winners,
        "phase_is_complete": preset_phase_is_complete,
        "phase_winners": preset_phase_winners,
        "default_output_dir": preset_default_output_dir,
        "phase_output_dir": preset_phase_output_dir,
    }


def experiment_config(sweep_type: str) -> dict:
    if sweep_type == "patch":
        return _patch_config()
    if sweep_type == "preset":
        return _preset_config()
    raise ValueError(f"Unknown sweep type: {sweep_type}")


def winners_path_for(sweep_type: str, path: Path | None = None) -> Path:
    if path is not None:
        return path
    return experiment_config(sweep_type)["experiment_dir"] / "winners.yaml"


def verification_phase(sweep_type: str, winners_path: Path | None = None) -> str:
    cfg = experiment_config(sweep_type)
    path = winners_path_for(sweep_type, winners_path)
    if sweep_type == "patch":
        return cfg["phase1"]
    if sweep_type == "preset":
        phase4_dir = cfg["phase_output_dir"](cfg["default_output_dir"](), cfg["phase4"])
        if (phase4_dir / "manifest.csv").is_file():
            return cfg["phase4"]
    if cfg["phase_is_complete"](cfg["phase3"], path):
        return cfg["phase3"]
    return cfg["phase2"]


def readiness_errors(sweep_type: str, winners_path: Path | None = None) -> list[str]:
    cfg = experiment_config(sweep_type)
    path = winners_path_for(sweep_type, winners_path)
    errors = []
    for phase in cfg["required_phases"]:
        if not cfg["phase_is_complete"](phase, path):
            errors.append(f"{phase} not complete in {path}")
    if sweep_type == "preset" and not cfg["phase_is_complete"](cfg["phase1b"], path):
        errors.append(f"{cfg['phase1b']} not complete in {path}")
    verify_phase = verification_phase(sweep_type, path)
    sweep_dir = cfg["phase_output_dir"](
        cfg["default_output_dir"](),
        verify_phase,
    )
    if not (sweep_dir / "manifest.csv").is_file():
        if sweep_type == "preset" and verify_phase == cfg["phase4"]:
            errors.append(
                f"Missing phase-4 verification render: run "
                f"`uv run python -m experiments.preset_sweep.sweep --phase {cfg['phase4']}`"
            )
        else:
            errors.append(f"Missing manifest for {verify_phase}: {sweep_dir / 'manifest.csv'}")
    return errors


def final_sweep_dir(sweep_type: str, winners_path: Path | None = None) -> Path:
    cfg = experiment_config(sweep_type)
    phase = verification_phase(sweep_type, winners_path)
    return cfg["phase_output_dir"](cfg["default_output_dir"](), phase)


def final_phase_winners(
    sweep_type: str,
    winners_path: Path | None = None,
) -> dict[str, str]:
    """Per-category variant ids from the verification phase (locked in winners.yaml)."""
    cfg = experiment_config(sweep_type)
    path = winners_path_for(sweep_type, winners_path)
    phase = verification_phase(sweep_type, path)
    if sweep_type == "preset" and phase == cfg["phase4"]:
        phase1_winners = cfg["phase_winners"](cfg["phase1"], path)
        return {
            str(category): LOCKED_VERIFY_VARIANT
            for category in phase1_winners
        }
    return dict(cfg["phase_winners"](phase, path))


def final_catalog(
    sweep_type: str,
    winners_path: Path | None = None,
) -> tuple[SweepCatalog, str]:
    errors = readiness_errors(sweep_type, winners_path)
    if errors:
        raise RuntimeError("; ".join(errors))
    phase = verification_phase(sweep_type, winners_path)
    catalog = SweepCatalog(sweep_type, final_sweep_dir(sweep_type, winners_path))
    return catalog, phase


def composed_config(
    sweep_type: str,
    category: str,
    variant_id: str,
    winners_path: Path | None = None,
) -> dict:
    """Full per-category production config for a final-phase variant."""
    cfg = experiment_config(sweep_type)
    path = winners_path_for(sweep_type, winners_path)
    phase1_winners = cfg["phase_winners"](cfg["phase1"], path)
    phase2_winners = cfg["phase_winners"](cfg["phase2"], path)

    if sweep_type == "patch":
        return {
            "variant_id": variant_id,
            "soundfont_id": variant_id,
            "fx_profile": "dry",
        }

    noise_variant = phase1_winners.get(category)
    base = {
        "variant_id": variant_id,
        "init_noise_level": (
            init_noise_level_from_variant_id(noise_variant)
            if noise_variant
            else None
        ),
    }
    verify_phase = verification_phase(sweep_type, path)
    phase3_winners = cfg["phase_winners"](cfg["phase3"], path)
    phase3_complete = cfg["phase_is_complete"](cfg["phase3"], path)

    if verify_phase == cfg["phase4"]:
        preset = {
            **base,
            "prompt_variant": phase2_winners.get(category),
        }
        if phase3_complete and category in phase3_winners:
            preset.update(_diffusion_from_variant_id(phase3_winners[category]))
        else:
            from shared.config import REALIFY_CFG_SCALE, REALIFY_STEPS

            preset.update({"steps": REALIFY_STEPS, "cfg_scale": REALIFY_CFG_SCALE})
        return preset

    if verify_phase == cfg["phase3"]:
        return {
            **base,
            "prompt_variant": phase2_winners.get(category),
            **_diffusion_from_variant_id(variant_id),
        }
    return {
        **base,
        "prompt_variant": variant_id,
    }


def apply_verification_to_winners(
    verification: dict,
    *,
    sweep_type: str,
    winners_path: Path | None = None,
) -> dict:
    """Override final-phase winners in winners.yaml from verification JSON."""
    from experiments.listening.verification import (
        bypass_realify_from_verification,
        bypass_routing_rules_from_verification,
        winners_from_verification,
    )
    from experiments.patch_sweep.winners import record_phase_winners as record_patch
    from experiments.preset_sweep.winners import (
        record_bypass_realify,
        record_bypass_realify_rules,
        record_phase_winners as record_preset,
    )

    winner_map = winners_from_verification(verification, sweep_type=sweep_type)
    bypass_map = bypass_realify_from_verification(verification)
    bypass_rules = bypass_routing_rules_from_verification(verification)
    if not winner_map and not bypass_map and not bypass_rules:
        raise RuntimeError("No winners or bypass entries in verification file.")

    path = winners_path_for(sweep_type, winners_path)
    if sweep_type == "patch":
        if not winner_map:
            raise RuntimeError("No winners in verification file.")
        from experiments.patch_sweep.config import PHASE1 as PATCH_PHASE1
        return record_patch(PATCH_PHASE1, winner_map, path=path)

    cfg = experiment_config(sweep_type)
    phase = verification_phase(sweep_type, path)
    if bypass_map:
        record_bypass_realify(bypass_map, path=path)
    if bypass_rules:
        record_bypass_realify_rules(bypass_rules, path=path)
    if winner_map and phase != cfg["phase4"]:
        record_preset(phase, winner_map, path=path)
    from experiments.preset_sweep.winners import load_winners

    return load_winners(path)
