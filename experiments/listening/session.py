"""Blinded trial ordering and rubric definitions for sweep listening tests."""

from __future__ import annotations

import random
import string

SWEEP_TYPES = frozenset({"preset", "patch"})

RUBRICS = {
    "preset": {
        "content_label": "Content",
        "content_help": "Same melody, rhythm, and timing as the reference?",
        "realism_label": "Realism",
        "realism_help": "Sounds like a recorded instrument?",
        "reference_label": "Reference (A1 raw)",
    },
    "patch": {
        "content_label": "Musical content",
        "content_help": "Same melody, rhythm, and timing as the reference?",
        "realism_label": "Timbre quality",
        "realism_help": "Sounds like a realistic, appropriate instrument?",
        "reference_label": "Reference (basic synthesis)",
    },
}


def blind_labels(n: int) -> list[str]:
    """Return n blind labels: A, B, …, Z, AA, …"""
    labels = []
    for i in range(n):
        label = ""
        x = i
        while True:
            label = string.ascii_uppercase[x % 26] + label
            x = x // 26 - 1
            if x < 0:
                break
        labels.append(label)
    return labels


def blinded_variant_order(
    variant_ids: list[str],
    *,
    stem_id: str,
    session_seed: int,
) -> list[tuple[str, str]]:
    """Shuffle variant ids and pair with blind labels."""
    rng = random.Random(hash((session_seed, stem_id)) & 0x7FFFFFFF)
    shuffled = list(variant_ids)
    rng.shuffle(shuffled)
    labels = blind_labels(len(shuffled))
    return list(zip(labels, shuffled))


def stem_order(stem_ids: list[str], session_seed: int) -> list[str]:
    """Shuffle probe stem presentation order."""
    rng = random.Random(session_seed & 0x7FFFFFFF)
    ordered = list(stem_ids)
    rng.shuffle(ordered)
    return ordered


def storage_key(sweep_type: str, manifest_id: str) -> str:
    return f"sweep_test_{sweep_type}_{manifest_id}"
