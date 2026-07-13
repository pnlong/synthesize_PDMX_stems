"""Tests for Slakh patch selection."""

import random

import mido

from synthesis.patches import (
    PATCH_POOLS,
    PatchAssignment,
    apply_patch_to_midi_track,
    patch_group_key,
    patch_seed,
    select_patch,
)


def test_select_patch_passthrough():
    assignment = select_patch(program=11, is_drum=False)
    assert assignment == PatchAssignment(
        program=11,
        is_drum=False,
        pool_id=None,
        gm_class="chromatic_percussion",
    )


def test_select_patch_drum():
    assignment = select_patch(program=0, is_drum=True)
    assert assignment.is_drum is True
    assert assignment.gm_class == "drums"


def test_select_patch_from_pool(monkeypatch):
    monkeypatch.setitem(
        PATCH_POOLS,
        "test_pool",
        {"piano": [5, 6]},
    )
    rng = random.Random(0)
    assignment = select_patch(program=0, is_drum=False, pool_id="test_pool", rng=rng)
    assert assignment.program in (5, 6)
    assert assignment.pool_id == "test_pool"


def test_select_patch_same_seed_same_group_same_program(monkeypatch):
    monkeypatch.setitem(
        PATCH_POOLS,
        "test_pool",
        {"strings": [40, 41, 42, 43]},
    )
    rng1 = random.Random(patch_seed(42, "/song/a", "strings"))
    rng2 = random.Random(patch_seed(42, "/song/a", "strings"))
    a1 = select_patch(program=40, is_drum=False, pool_id="test_pool", rng=rng1)
    a2 = select_patch(program=42, is_drum=False, pool_id="test_pool", rng=rng2)
    assert a1.program == a2.program


def test_select_patch_different_songs_can_differ(monkeypatch):
    monkeypatch.setitem(
        PATCH_POOLS,
        "test_pool",
        {"strings": [40, 41, 42, 43, 44, 45]},
    )
    programs = set()
    for song in ("/song/a", "/song/b", "/song/c"):
        rng = random.Random(patch_seed(42, song, "strings"))
        programs.add(
            select_patch(program=40, is_drum=False, pool_id="test_pool", rng=rng).program
        )
    assert len(programs) >= 2
    monkeypatch.setitem(
        PATCH_POOLS,
        "test_pool",
        {
            "pipe": [73],
            "reed": [65, 66],
            "piano": [0],
        },
    )
    rng = random.Random(0)
    assignment = select_patch(
        program=0,
        is_drum=False,
        pool_id="test_pool",
        category="wind",
        rng=rng,
    )
    assert assignment.program in (73, 65, 66)


def test_apply_patch_to_midi_track():
    track = mido.MidiTrack()
    track.append(mido.Message("program_change", program=0, time=0))
    assignment = PatchAssignment(program=42, is_drum=False, pool_id="x", gm_class="piano")
    apply_patch_to_midi_track(track, assignment)
    assert track[0].program == 42


def test_patch_seed_stable_within_song_and_group():
    group = patch_group_key(44, False)
    assert patch_seed(42, "/song/path", group) == patch_seed(42, "/song/path", group)
    assert patch_seed(42, "/song/a", group) != patch_seed(42, "/song/b", group)


def test_patch_seed_differs_by_instrument_class():
    assert patch_seed(42, "/song", "strings") != patch_seed(42, "/song", "piano")
