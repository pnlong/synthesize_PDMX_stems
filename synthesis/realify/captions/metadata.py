"""Caption generation from PDMX metadata."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd

from shared.config import CAPTION_MD_COLUMNS


def instrument_hint(program: int, is_drum: bool, name: str | None) -> str:
    if is_drum:
        return "format: solo | instruments: drums"
    if name:
        return f"format: solo | instruments: {name.lower()}"
    return f"format: solo | instruments: midi program {program}"


def get_caption(
    metadata: dict[str, Any],
    rng: random.Random | None = None,
    include_instrument_hint: bool = True,
) -> str:
    """Build a shuffled comma-separated caption from metadata fields."""
    if rng is None:
        rng = random.Random()

    vals = [
        f"{k}: {v}"
        for k, v in metadata.items()
        if k in CAPTION_MD_COLUMNS and pd.notna(v)
    ]

    if include_instrument_hint:
        hint = instrument_hint(
            program=int(metadata.get("program", 0)),
            is_drum=bool(metadata.get("is_drum", False)),
            name=metadata.get("name"),
        )
        vals.append(hint)

    rng.shuffle(vals)
    return ", ".join(vals)
