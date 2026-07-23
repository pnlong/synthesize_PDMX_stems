"""0–100 listening scale helpers with 20-point band labels."""

from __future__ import annotations

CONTENT_BAND_LABELS = (
    "Very different",
    "Different",
    "Mostly same",
    "Same",
    "Identical",
)

REALISM_BAND_LABELS = (
    "Very synthetic",
    "Synthetic",
    "Mixed",
    "Realistic",
    "Very realistic",
)

BAND_MIDPOINTS = (10, 30, 50, 70, 90)
TICK_VALUES = (0, 20, 40, 60, 80, 100)

DEFAULT_CONTENT_THRESHOLD = 50
DEFAULT_CONTENT_MEAN_THRESHOLD = 60
DEFAULT_REALISM_THRESHOLD = 70

CONTENT_FIELDS = ("content",)
REALISM_FIELDS = ("realism",)
RATING_FIELDS = ("content", "realism")


def clamp_score(value: int | float) -> int:
    return max(0, min(100, int(round(value))))


def band_index(score: int | float) -> int:
    """Return 0–4 band index for a 0–100 score."""
    score = clamp_score(score)
    if score >= 80:
        return 4
    if score >= 60:
        return 3
    if score >= 40:
        return 2
    if score >= 20:
        return 1
    return 0


def band_midpoint(index: int) -> int:
    if index < 0 or index >= len(BAND_MIDPOINTS):
        raise ValueError(f"band index out of range: {index}")
    return BAND_MIDPOINTS[index]


def snap_to_band_midpoint(score: int | float) -> int:
    return band_midpoint(band_index(score))


def likert_equivalent(score: int | float) -> int:
    """Map 0–100 to 1–5 Likert equivalent."""
    return band_index(score) + 1


def content_label(score: int | float) -> str:
    return CONTENT_BAND_LABELS[band_index(score)]


def realism_label(score: int | float) -> str:
    return REALISM_BAND_LABELS[band_index(score)]


def band_label(score: int | float, rubric: str) -> str:
    if rubric == "realism":
        return realism_label(score)
    return content_label(score)


def is_rated(value) -> bool:
    if value is None:
        return False
    try:
        score = int(value)
    except (TypeError, ValueError):
        return False
    return 0 <= score <= 100
