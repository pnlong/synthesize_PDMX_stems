import pytest

from experiments.ablation_listening.session import (
    blind_labels,
    blinded_condition_order,
    trial_order,
)


def test_blind_labels():
    assert blind_labels(4) == ["A", "B", "C", "D"]


def test_blinded_condition_order_is_deterministic():
    conditions = ["basic", "basic_realify", "slakh", "slakh_realify"]
    first = blinded_condition_order(conditions, trial_id="mix_01", session_seed=42)
    second = blinded_condition_order(conditions, trial_id="mix_01", session_seed=42)
    assert first == second
    labels = [label for label, _ in first]
    assert len(set(labels)) == 4


def test_trial_order_shuffled():
    ordered = trial_order(["a", "b", "c", "d"], 42)
    assert sorted(ordered) == ["a", "b", "c", "d"]
