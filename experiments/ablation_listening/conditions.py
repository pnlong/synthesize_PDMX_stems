"""Condition IDs for the formal ablation MUSHRA (A1–CB2)."""

from __future__ import annotations

from pathlib import Path

from synthesis.listening.catalog import CONDITION_LABELS, CONDITION_ORDER
from synthesis.patches import LISTENING_CATEGORY_GM_CLASSES

# Explicit order for this test (matches synthesis.listening.catalog).
ABLATION_MUSHRA_CONDITIONS: tuple[str, ...] = tuple(CONDITION_ORDER)

REFERENCE_CONDITION = "basic"

# All listening categories — stem trials only (mixtures have no single category).
STEM_TRIAL_CATEGORIES: tuple[str, ...] = tuple(LISTENING_CATEGORY_GM_CLASSES.keys())

# Multiple stems per category for more stable per-category estimates.
DEFAULT_STEMS_PER_CATEGORY = 2

RATING_SCALES: tuple[str, ...] = ("content", "realism")

SCALE_LABELS = {
    "content": "Content adherence",
    "realism": "Realism",
}

SCALE_HELP = {
    "content": (
        "How well does this condition preserve the melody, rhythm, and timing of the Reference? "
        "100 = identical musical content; 0 = completely different or unrecognizable."
    ),
    "realism": (
        "How realistic / natural does this condition sound as an instrument recording "
        "(timbre, articulation, absence of artifacts)? "
        "100 = transparent / as realistic as the Reference; 0 = highly artificial or degraded."
    ),
}


def condition_roots(ablations_dir: Path) -> dict[str, Path]:
    ablations_dir = Path(ablations_dir)
    return {cid: ablations_dir / cid for cid in ABLATION_MUSHRA_CONDITIONS}


def mushra_page_id(trial_id: str, scale: str) -> str:
    return f"{trial_id}__{scale}"


def parse_mushra_page_id(page_id: str) -> tuple[str, str | None]:
    """Return (trial_id, scale) for ``stem_piano__content``-style page ids."""
    text = str(page_id)
    if "__" in text:
        trial_id, scale = text.rsplit("__", 1)
        if scale in RATING_SCALES:
            return trial_id, scale
    return text, None


def category_from_trial_id(trial_id: str) -> str | None:
    """``stem_piano`` / ``stem_piano_02`` → ``piano``."""
    text = str(trial_id)
    if not text.startswith("stem_"):
        return None
    rest = text[len("stem_") :]
    # strip trailing _NN index
    parts = rest.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return rest


__all__ = [
    "ABLATION_MUSHRA_CONDITIONS",
    "CONDITION_LABELS",
    "DEFAULT_STEMS_PER_CATEGORY",
    "RATING_SCALES",
    "REFERENCE_CONDITION",
    "SCALE_HELP",
    "SCALE_LABELS",
    "STEM_TRIAL_CATEGORIES",
    "category_from_trial_id",
    "condition_roots",
    "mushra_page_id",
    "parse_mushra_page_id",
]
