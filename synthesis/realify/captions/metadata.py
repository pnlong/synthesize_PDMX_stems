"""Caption generation from PDMX metadata."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd

from shared.config import CAPTION_MD_COLUMNS

REALIFY_ANCHOR = (
    "TrackType: Instrument. Preserve the exact melody and rhythm. "
    "Same performance and timing. Realistic recorded sound, expressive performance."
)

PROMPT_VARIANTS = ("current", "minimal", "preservation")

PRESERVATION_ANCHOR = (
    "Preserve the exact melody, rhythm, and timing. Same notes and phrasing."
)


def instrument_hint(program: int, is_drum: bool, name: str | None) -> str:
    if is_drum:
        return "format: solo | instruments: drums"
    if isinstance(name, str) and name.strip():
        return f"format: solo | instruments: {name.strip().lower()}"
    return f"format: solo | instruments: midi program {program}"


def instrument_label(program: int, is_drum: bool, name: str | None) -> str:
    if is_drum:
        return "drums"
    if isinstance(name, str) and name.strip():
        return name.strip().lower()
    return f"midi program {program}"


def get_caption(
    metadata: dict[str, Any],
    rng: random.Random | None = None,
    include_instrument_hint: bool = True,
    prompt_variant: str = "current",
) -> str:
    """Build a caption for SA3 realify or preset-sweep variants."""
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(
            f"Unknown prompt_variant {prompt_variant!r}; expected one of {PROMPT_VARIANTS}"
        )

    if rng is None:
        rng = random.Random()

    program = int(metadata.get("program", 0))
    is_drum = bool(metadata.get("is_drum", False))
    name = metadata.get("name")
    label = instrument_label(program, is_drum, name)

    if prompt_variant == "minimal":
        return (
            f"solo {label}, realistic studio recording, expressive performance"
        )

    if prompt_variant == "preservation":
        parts = [f"{PRESERVATION_ANCHOR} Realistic recorded {label}."]
        if include_instrument_hint:
            parts.append(instrument_hint(program, is_drum, name))
        return ". ".join(parts)

    vals = [
        f"{k}: {v}"
        for k, v in metadata.items()
        if k in CAPTION_MD_COLUMNS and pd.notna(v)
    ]
    rng.shuffle(vals)

    parts = [REALIFY_ANCHOR]
    if vals:
        parts.append(", ".join(vals))
    if include_instrument_hint:
        parts.append(instrument_hint(program, is_drum, name))
    return ". ".join(parts)
