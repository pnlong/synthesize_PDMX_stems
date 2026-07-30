"""Tests for preset sweep config helpers."""

from experiments.preset_sweep.config import (
    PHASE2B,
    build_adherence_audit_variants,
    higher_noise_level,
    resolve_silence_enforce,
)


def test_higher_noise_level_steps_up_grid():
    assert higher_noise_level(0.25) == 0.35
    assert higher_noise_level(0.65) == 0.65


def test_build_adherence_audit_variants():
    phase1_winners = {
        "piano": "noise0.25",
        "drums": "noise0.35",
    }
    variants = build_adherence_audit_variants(phase1_winners)
    ids = {variant["id"] for variant in variants}
    assert ids == {
        "baseline_winner",
        "baseline_higher",
        "adherence_winner",
        "adherence_higher",
    }
    assert resolve_silence_enforce(PHASE2B, {}) is True
