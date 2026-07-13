"""Tests for sweep listening session helpers."""

from experiments.listening.session import (
    blind_labels,
    blinded_variant_order,
    stem_order,
    storage_key,
)


def test_blind_labels():
    assert blind_labels(3) == ["A", "B", "C"]
    assert blind_labels(27)[-1] == "AA"


def test_blinded_variant_order_reproducible():
    first = blinded_variant_order(
        ["v1", "v2", "v3"],
        stem_id="piano_world",
        session_seed=42,
    )
    second = blinded_variant_order(
        ["v1", "v2", "v3"],
        stem_id="piano_world",
        session_seed=42,
    )
    assert first == second
    labels = [label for label, _ in first]
    assert len(set(labels)) == 3


def test_blinded_variant_order_differs_by_stem():
    a = blinded_variant_order(["v1", "v2", "v3", "v4"], stem_id="stem_a", session_seed=42)
    b = blinded_variant_order(["v1", "v2", "v3", "v4"], stem_id="stem_b", session_seed=42)
    assert [variant for _, variant in a] != [variant for _, variant in b]


def test_stem_order_shuffles():
    ordered = stem_order(["a", "b", "c", "d"], session_seed=7)
    assert sorted(ordered) == ["a", "b", "c", "d"]
    assert stem_order(["a", "b", "c", "d"], session_seed=7) == ordered


def test_storage_key():
    assert storage_key("preset", "abc123") == "sweep_test_preset_abc123"
