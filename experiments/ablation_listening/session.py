"""Blinded trial ordering for ablation listening tests."""

from __future__ import annotations

import random
import string

from experiments.ablation_listening.conditions import ABLATION_MUSHRA_CONDITIONS
from experiments.ablation_listening.equivalence import (
    trial_equivalences,
    unique_condition_ids,
)

REFERENCE_CONDITION = "basic"

RUBRICS = {
    "content": {
        "label": "Content",
        "help": "Same melody, rhythm, and timing as the Reference?",
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


def unique_blind_condition_ids(trial: dict) -> list[str]:
    """Conditions shown as blind samples (includes ``basic`` when not a donor-copy omit)."""
    return unique_condition_ids(ABLATION_MUSHRA_CONDITIONS, trial_equivalences(trial))


def default_condition_ids() -> list[str]:
    return list(ABLATION_MUSHRA_CONDITIONS)


def storage_key(test_id: str, session_seed: int, listener_id: str = "") -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(listener_id or "anon"))
    return f"ablation_listening_{test_id}_{safe}_{session_seed}"


def safe_listener_id(listener_id: str | None) -> str:
    text = str(listener_id or "anonymous").strip() or "anonymous"
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def realism_rubric(trial_type: str) -> dict:
    if trial_type == "mixture":
        return RUBRICS["realism_mix"]
    return RUBRICS["realism_stem"]
