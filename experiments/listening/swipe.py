"""Swipe listening queue and response helpers."""

from __future__ import annotations

import random
from pathlib import Path

SWIPE_TIERS = ("strong_reject", "weak_reject", "weak_accept", "strong_accept")
SWIPE_MODES = frozenset({"shuffle", "group_stem", "sequential"})
DEFAULT_SWIPE_ORDER = "shuffle"


def swipe_storage_key(sweep_type: str, session_id: str) -> str:
    return f"swipe_{sweep_type}_{session_id}"


def swipe_session_id(sweep_dir: Path) -> str:
    """Stable swipe session id across clip-manifest appends."""
    return sweep_dir.resolve().name


def merge_swipe_vote_lists(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged = votes_by_card_id(existing)
    for vote in incoming:
        card_id = vote.get("card_id") or vote_key(vote)
        if card_id:
            merged[str(card_id)] = {**vote, "card_id": str(card_id)}
    return list(merged.values())


def vote_key(vote: dict) -> str:
    return f"{vote['variant_id']}|{vote['clip_id']}"


def card_soundfont_id(card: dict) -> str:
    return str(card.get("soundfont_id") or card["variant_id"])


def same_cascade_group(left: dict, right: dict) -> bool:
    """Group cards for accept/reject cascades within a category (by variant_id)."""
    return (
        str(left.get("category") or "") == str(right.get("category") or "")
        and str(left["variant_id"]) == str(right["variant_id"])
    )


def same_soundfont_in_category(left: dict, right: dict) -> bool:
    """Backward-compatible alias; cascades key off variant_id, not soundfont_id."""
    return same_cascade_group(left, right)


def make_swipe_vote(card: dict, tier: str, **extra) -> dict:
    vote = {
        "card_id": card["card_id"],
        "category": card.get("category"),
        "stem_id": card.get("stem_id"),
        "clip_id": card.get("clip_id"),
        "variant_id": card.get("variant_id"),
        "tier": tier,
    }
    vote.update(extra)
    return vote


def votes_by_card_id(votes: list[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for vote in votes:
        card_id = vote.get("card_id") or vote_key(vote)
        if not card_id:
            continue
        indexed[str(card_id)] = {**vote, "card_id": str(card_id)}
    return indexed


def cascade_strong_reject(
    cards: list[dict],
    votes: dict[str, dict],
    trigger_card_id: str,
    *,
    overwrite: bool = True,
) -> list[tuple[str, dict | None]]:
    """Strong-reject sibling clips for the same variant in one category."""
    trigger = next((card for card in cards if card["card_id"] == trigger_card_id), None)
    if trigger is None:
        return []

    changes: list[tuple[str, dict | None]] = []
    for sibling in cards:
        if sibling["card_id"] == trigger_card_id:
            continue
        if not same_soundfont_in_category(trigger, sibling):
            continue
        previous = votes.get(sibling["card_id"])
        if previous and previous.get("tier") == "strong_reject":
            continue
        if previous and not overwrite:
            continue
        changes.append((sibling["card_id"], previous))
        votes[sibling["card_id"]] = make_swipe_vote(
            sibling,
            "strong_reject",
            cascaded_from=trigger_card_id,
        )
    return changes


def cascade_strong_accept(
    cards: list[dict],
    votes: dict[str, dict],
    trigger_card_id: str,
) -> list[tuple[str, dict | None]]:
    """Strong-accept unvoted sibling clips for the same variant in one category."""
    trigger = next((card for card in cards if card["card_id"] == trigger_card_id), None)
    if trigger is None:
        return []

    changes: list[tuple[str, dict | None]] = []
    for sibling in cards:
        if sibling["card_id"] == trigger_card_id:
            continue
        if not same_soundfont_in_category(trigger, sibling):
            continue
        previous = votes.get(sibling["card_id"])
        if previous:
            continue
        changes.append((sibling["card_id"], previous))
        votes[sibling["card_id"]] = make_swipe_vote(
            sibling,
            "strong_accept",
            cascaded_from=trigger_card_id,
        )
    return changes


def normalize_swipe_strong_reject_cascades(
    cards: list[dict],
    votes: list[dict],
    *,
    overwrite: bool = True,
) -> list[dict]:
    """Ensure every strong reject applies to all clips from that soundfont in-category."""
    indexed = votes_by_card_id(votes)
    for card in cards:
        vote = indexed.get(card["card_id"])
        if vote and vote.get("tier") == "strong_reject":
            cascade_strong_reject(
                cards,
                indexed,
                card["card_id"],
                overwrite=overwrite,
            )
    return list(indexed.values())


def normalize_swipe_cascades(
    cards: list[dict],
    votes: list[dict],
    *,
    reject_overwrite: bool = True,
) -> list[dict]:
    """Apply strong-reject and strong-accept cascades for saved swipe responses."""
    indexed = votes_by_card_id(
        normalize_swipe_strong_reject_cascades(
            cards,
            votes,
            overwrite=reject_overwrite,
        )
    )
    for card in cards:
        vote = indexed.get(card["card_id"])
        if vote and vote.get("tier") == "strong_accept":
            cascade_strong_accept(cards, indexed, card["card_id"])
    return list(indexed.values())


def sanitize_cross_variant_cascade_votes(votes: list[dict]) -> tuple[list[dict], int]:
    """Drop cascaded votes copied onto a different variant (phase-2 FX bug)."""
    cleaned: list[dict] = []
    dropped = 0
    for vote in votes:
        cascaded_from = vote.get("cascaded_from")
        if cascaded_from:
            source_variant = str(cascaded_from).split("|", 1)[0]
            if source_variant != str(vote.get("variant_id")):
                dropped += 1
                continue
        cleaned.append(vote)
    return cleaned, dropped


def filter_swipe_votes_to_manifest(responses: dict, manifest) -> tuple[dict, int]:
    """Drop swipe votes whose variant_id is absent from the sweep manifest."""
    if manifest.empty or "variant_id" not in manifest.columns:
        return responses, 0

    allowed = set(manifest["variant_id"].astype(str).unique())
    votes = list(responses.get("votes") or [])
    filtered = [vote for vote in votes if str(vote.get("variant_id")) in allowed]
    dropped = len(votes) - len(filtered)
    if dropped == 0:
        return responses, 0
    return {**responses, "votes": filtered}, dropped


def merge_votes(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged = {vote_key(vote): vote for vote in existing if vote.get("variant_id") and vote.get("clip_id")}
    for vote in incoming:
        key = vote_key(vote)
        if key:
            merged[key] = vote
    return list(merged.values())


def order_swipe_cards(cards: list[dict], *, order: str, seed: int) -> list[dict]:
    if order == "sequential":
        return list(cards)

    rng = random.Random(seed & 0x7FFFFFFF)
    if order == "group_stem":
        groups: dict[tuple[str, str], list[dict]] = {}
        for card in cards:
            key = (str(card["stem_id"]), str(card["clip_id"]))
            groups.setdefault(key, []).append(card)
        ordered: list[dict] = []
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)
        for key in group_keys:
            group = list(groups[key])
            rng.shuffle(group)
            ordered.extend(group)
        return ordered

    shuffled = list(cards)
    rng.shuffle(shuffled)
    return shuffled


def build_swipe_cards(catalog, *, category: str | None = None) -> list[dict]:
    """Build swipe card payloads from clip manifest rows."""
    clip_manifest = catalog._clip_manifest
    if clip_manifest.empty:
        return []

    probes = catalog._probe_by_id
    clips_per_stem: dict[str, int] = {}
    for stem_id in clip_manifest["stem_id"].unique():
        clips_per_stem[str(stem_id)] = int(
            clip_manifest[clip_manifest["stem_id"] == stem_id]["clip_id"].nunique()
        )

    cards: list[dict] = []
    for _, row in clip_manifest.iterrows():
        row_category = str(row.get("category") or "")
        if category and row_category != category:
            continue

        variant_id = str(row["variant_id"])
        clip_id = str(row["clip_id"])
        stem_id = str(row["stem_id"])
        probe = probes.get(stem_id, {})
        clip_index = int(row.get("clip_index") or 0)
        n_clips = clips_per_stem.get(stem_id, clip_index + 1)
        track = int(row.get("track") or probe.get("track") or 0)
        note = probe.get("note") or stem_id
        category_label = row_category or probe.get("category") or "unknown"
        out_path = Path(str(row["out_path"]))
        filename = out_path.name
        from experiments.listening.catalog import song_id_from_manifest_row

        song_id = song_id_from_manifest_row(row)
        audio = catalog._variant_cell_from_row(row)
        soundfont_id = str(row.get("soundfont_id") or variant_id)
        cards.append({
            "card_id": f"{variant_id}|{clip_id}",
            "variant_id": variant_id,
            "soundfont_id": soundfont_id,
            "clip_id": clip_id,
            "stem_id": stem_id,
            "category": row_category,
            "legacy": bool(probe.get("legacy")),
            "track": track,
            "clip_index": clip_index,
            "clips_per_stem": n_clips,
            "label": f"{category_label} · {note} · clip {clip_index + 1}/{n_clips}",
            "audio": audio,
            "filename": filename,
            "song_id": song_id,
        })
    return cards
