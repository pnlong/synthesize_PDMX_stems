import pytest

from experiments.listening_shared.scale import (
    band_index,
    band_label,
    band_midpoint,
    clamp_score,
    content_label,
    likert_equivalent,
    realism_label,
    snap_to_band_midpoint,
)


@pytest.mark.parametrize(
    ("score", "index"),
    [(0, 0), (19, 0), (20, 1), (50, 2), (79, 3), (100, 4)],
)
def test_band_index(score, index):
    assert band_index(score) == index


def test_band_midpoints():
    assert band_midpoint(0) == 10
    assert band_midpoint(4) == 90


def test_snap_to_band_midpoint():
    assert snap_to_band_midpoint(15) == 10
    assert snap_to_band_midpoint(72) == 70


def test_likert_equivalent():
    assert likert_equivalent(10) == 1
    assert likert_equivalent(85) == 5


def test_band_labels():
    assert content_label(45) == "Mostly same"
    assert realism_label(45) == "Mixed"
    assert band_label(45, "content") == content_label(45)


def test_clamp_score():
    assert clamp_score(150) == 100
    assert clamp_score(-5) == 0
