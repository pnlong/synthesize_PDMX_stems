"""Tests for swipe queue helpers."""

from pathlib import Path

import pandas as pd

from experiments.listening.swipe import (
    cascade_strong_accept,
    cascade_strong_reject,
    filter_swipe_votes_to_manifest,
    make_swipe_vote,
    merge_swipe_vote_lists,
    normalize_swipe_cascades,
    normalize_swipe_strong_reject_cascades,
    sanitize_cross_variant_cascade_votes,
    swipe_session_id,
    swipe_storage_key,
    votes_by_card_id,
)


def _card(
    variant_id: str,
    clip_id: str,
    *,
    category: str = "piano",
    soundfont_id: str | None = None,
) -> dict:
    return {
        "card_id": f"{variant_id}|{clip_id}",
        "variant_id": variant_id,
        "soundfont_id": soundfont_id or variant_id,
        "clip_id": clip_id,
        "category": category,
        "stem_id": "piano_test",
    }


def test_cascade_strong_reject_marks_sibling_clips():
    cards = [
        _card("sf_a", "c0"),
        _card("sf_a", "c1"),
        _card("sf_a", "c2"),
        _card("sf_b", "c0"),
    ]
    votes = votes_by_card_id([
        make_swipe_vote(cards[0], "strong_reject"),
    ])

    changes = cascade_strong_reject(cards, votes, cards[0]["card_id"])

    assert len(changes) == 2
    assert votes[cards[1]["card_id"]]["tier"] == "strong_reject"
    assert votes[cards[2]["card_id"]]["tier"] == "strong_reject"
    assert cards[3]["card_id"] not in votes


def test_cascade_strong_reject_is_category_scoped():
    cards = [
        _card("sf_a", "c0", category="piano"),
        _card("sf_a", "c1", category="voice"),
    ]
    votes = votes_by_card_id([
        make_swipe_vote(cards[0], "strong_reject"),
    ])

    cascade_strong_reject(cards, votes, cards[0]["card_id"])

    assert cards[1]["card_id"] not in votes


def test_normalize_swipe_strong_reject_cascades_on_save():
    cards = [
        _card("sf_a", "c0"),
        _card("sf_a", "c1"),
    ]
    votes = normalize_swipe_strong_reject_cascades(
        cards,
        [make_swipe_vote(cards[0], "strong_reject")],
    )

    assert len(votes) == 2
    assert all(vote["tier"] == "strong_reject" for vote in votes)


def test_cascade_strong_accept_skips_unvoted_siblings_only():
    cards = [
        _card("sf_a", "c0"),
        _card("sf_a", "c1"),
        _card("sf_a", "c2"),
    ]
    votes = votes_by_card_id([
        make_swipe_vote(cards[0], "strong_accept"),
    ])

    changes = cascade_strong_accept(cards, votes, cards[0]["card_id"])

    assert len(changes) == 2
    assert votes[cards[1]["card_id"]]["tier"] == "strong_accept"
    assert votes[cards[2]["card_id"]]["tier"] == "strong_accept"


def test_cascade_strong_accept_does_not_overwrite_existing_votes():
    cards = [
        _card("sf_a", "c0"),
        _card("sf_a", "c1"),
        _card("sf_a", "c2"),
    ]
    votes = votes_by_card_id([
        make_swipe_vote(cards[0], "strong_accept"),
        make_swipe_vote(cards[1], "weak_reject"),
    ])

    changes = cascade_strong_accept(cards, votes, cards[0]["card_id"])

    assert len(changes) == 1
    assert votes[cards[1]["card_id"]]["tier"] == "weak_reject"
    assert votes[cards[2]["card_id"]]["tier"] == "strong_accept"


def test_normalize_swipe_cascades_applies_both_tiers():
    cards = [
        _card("sf_a", "c0"),
        _card("sf_a", "c1"),
        _card("sf_b", "c0"),
        _card("sf_b", "c1"),
    ]
    votes = normalize_swipe_cascades(
        cards,
        [
            make_swipe_vote(cards[0], "strong_accept"),
            make_swipe_vote(cards[2], "strong_reject"),
        ],
    )

    indexed = votes_by_card_id(votes)
    assert indexed[cards[1]["card_id"]]["tier"] == "strong_accept"
    assert indexed[cards[3]["card_id"]]["tier"] == "strong_reject"


def test_merge_swipe_vote_lists_preserves_existing_votes():
    existing = [make_swipe_vote(_card("sf_a", "c0"), "strong_accept")]
    incoming = [make_swipe_vote(_card("sf_b", "c0"), "weak_accept")]
    merged = merge_swipe_vote_lists(existing, incoming)
    assert len(merged) == 2


def test_cascade_strong_accept_does_not_cross_fx_variants():
    """Phase 2 shares one soundfont per category; cascades must stay on variant_id."""
    cards = [
        _card("fx_dry", "c0", soundfont_id="airfont_380_final"),
        _card("fx_light", "c0", soundfont_id="airfont_380_final"),
        _card("fx_warm", "c0", soundfont_id="airfont_380_final"),
        _card("fx_dry", "c1", soundfont_id="airfont_380_final"),
    ]
    votes = votes_by_card_id([
        make_swipe_vote(cards[0], "strong_accept"),
    ])

    changes = cascade_strong_accept(cards, votes, cards[0]["card_id"])

    assert len(changes) == 1
    assert votes[cards[3]["card_id"]]["tier"] == "strong_accept"
    assert cards[1]["card_id"] not in votes
    assert cards[2]["card_id"] not in votes


def test_sanitize_cross_variant_cascade_votes():
    votes = [
        {
            "variant_id": "fx_dry",
            "clip_id": "c1",
            "tier": "strong_accept",
            "cascaded_from": "fx_dry|c0",
        },
        {
            "variant_id": "fx_light",
            "clip_id": "c0",
            "tier": "strong_accept",
            "cascaded_from": "fx_dry|c0",
        },
    ]
    cleaned, dropped = sanitize_cross_variant_cascade_votes(votes)
    assert dropped == 1
    assert cleaned == [votes[0]]


def test_filter_swipe_votes_to_manifest_drops_other_sessions():
    manifest = pd.DataFrame({
        "variant_id": ["fx_dry", "fx_light", "fx_warm"],
        "category": ["piano", "piano", "piano"],
    })
    responses = {
        "votes": [
            {"variant_id": "fx_dry", "clip_id": "c0", "category": "piano", "tier": "strong_accept"},
            {"variant_id": "airfont_380_final", "clip_id": "c0", "category": "piano", "tier": "strong_accept"},
        ],
    }
    filtered, dropped = filter_swipe_votes_to_manifest(responses, manifest)
    assert dropped == 1
    assert len(filtered["votes"]) == 1
    assert filtered["votes"][0]["variant_id"] == "fx_dry"


def test_swipe_storage_key_is_stable_across_manifest_versions():
    assert swipe_storage_key("patch", "phase1_archive_soundfonts") == (
        "swipe_patch_phase1_archive_soundfonts"
    )
    assert swipe_session_id(Path("/tmp/output/phase1_archive_soundfonts")) == (
        "phase1_archive_soundfonts"
    )
