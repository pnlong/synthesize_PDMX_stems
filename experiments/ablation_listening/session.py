"""Blinded trial ordering for ablation listening tests."""

from __future__ import annotations

import random
import string

from synthesis.listening.catalog import CONDITION_ORDER

REFERENCE_CONDITION = "basic"
VARIANT_CONDITIONS = tuple(c for c in CONDITION_ORDER if c != REFERENCE_CONDITION)

RUBRICS = {
    "content": {
        "label": "Content",
        "help": "Same melody, rhythm, and timing as the reference (A1)?",
    },
    "realism_stem": {
        "label": "Realism",
        "help": "Sounds like a realistic, appropriate instrument?",
    },
    "realism_mix": {
        "label": "Realism",
        "help": "Sounds like a realistic recording of a full mix?",
    },
}


def blind_labels(n: int) -> list[str]:
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


def blinded_condition_order(
    condition_ids: list[str],
    *,
    trial_id: str,
    session_seed: int,
) -> list[tuple[str, str]]:
    rng = random.Random(hash((session_seed, trial_id)) & 0x7FFFFFFF)
    shuffled = list(condition_ids)
    rng.shuffle(shuffled)
    labels = blind_labels(len(shuffled))
    return list(zip(labels, shuffled))


def trial_order(trial_ids: list[str], session_seed: int) -> list[str]:
    rng = random.Random(session_seed & 0x7FFFFFFF)
    ordered = list(trial_ids)
    rng.shuffle(ordered)
    return ordered


def variant_condition_ids() -> list[str]:
    return list(VARIANT_CONDITIONS)


def default_condition_ids() -> list[str]:
    return list(CONDITION_ORDER)


def storage_key(test_id: str, session_seed: int) -> str:
    return f"ablation_listening_{test_id}_{session_seed}"


def realism_rubric(trial_type: str) -> dict:
    if trial_type == "mixture":
        return RUBRICS["realism_mix"]
    return RUBRICS["realism_stem"]
