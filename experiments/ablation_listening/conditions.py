"""Condition IDs for the formal ablation listening test (A1–CB2)."""

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


def gm_instrument_label(*, program: int, is_drum: bool) -> str:
    """Human-readable GM instrument for the listening UI."""
    from analysis.gm_programs import DRUM_GM_ID, gm_program_display_name

    if is_drum:
        return gm_program_display_name(DRUM_GM_ID)
    return gm_program_display_name(int(program))


def condition_roots(ablations_dir: Path) -> dict[str, Path]:
    ablations_dir = Path(ablations_dir)
    return {cid: ablations_dir / cid for cid in ABLATION_MUSHRA_CONDITIONS}


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
    "STEM_TRIAL_CATEGORIES",
    "category_from_trial_id",
    "condition_roots",
    "gm_instrument_label",
]
