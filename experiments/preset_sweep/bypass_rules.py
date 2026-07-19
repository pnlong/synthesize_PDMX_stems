"""Build per-instrument realify bypass routing rules from stem metadata."""

from __future__ import annotations

import re

from experiments.probe_stems import PROBE_CATEGORIES


def _normalize_name(name) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def track_name_keywords(track_name: str | None) -> list[str]:
    """Extract keyword tokens from a MIDI track name for routing rules."""
    name = _normalize_name(track_name)
    if not name:
        return []
    tokens = [token for token in re.split(r"[\s_/\-]+", name) if len(token) >= 3]
    if not tokens:
        return [name]
    return tokens[:3]


def bypass_rule_from_stem(
    *,
    category: str,
    track_name: str | None,
    program: int,
    is_drum: bool,
) -> dict:
    """Return a routing rule that disables SA3 for matching stems."""
    if category not in PROBE_CATEGORIES:
        raise ValueError(f"Unknown probe category: {category}")

    rule: dict = {"category": category, "realify": False}
    keywords = track_name_keywords(track_name)
    if keywords:
        rule["name_keywords"] = keywords
        return rule

    rule["program"] = int(program)
    if is_drum:
        rule["is_drum"] = True
    return rule


def rule_fingerprint(rule: dict) -> tuple:
    return (
        str(rule.get("category")),
        tuple(rule.get("name_keywords") or ()),
        rule.get("program"),
        rule.get("is_drum"),
        rule.get("realify"),
    )


def merge_bypass_rules(existing: list[dict], new_rules: list[dict]) -> list[dict]:
    """Append bypass rules without duplicates."""
    merged = [dict(rule) for rule in existing]
    seen = {rule_fingerprint(rule) for rule in merged}
    for rule in new_rules:
        fingerprint = rule_fingerprint(rule)
        if fingerprint in seen:
            continue
        merged.append(dict(rule))
        seen.add(fingerprint)
    return merged
