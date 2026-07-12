"""Tests for caption generation."""

import random

import pandas as pd

from synthesis.realify.captions.generate import generate_captions, generate_captions_from_tables
from synthesis.realify.captions.metadata import REALIFY_ANCHOR, get_caption, instrument_hint


def test_get_caption_filters_na_and_shuffles_with_seed():
    metadata = {
        "genres": "rock",
        "groups": pd.NA,
        "tags": "guitar",
        "song_name": "Test Song",
        "title": pd.NA,
        "subtitle": pd.NA,
        "artist_name": "Artist",
        "composer_name": pd.NA,
        "publisher": pd.NA,
        "complexity": 3,
        "license": "CC",
        "program": 0,
        "is_drum": False,
        "name": "Piano",
    }
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    assert get_caption(metadata, rng=rng1) == get_caption(metadata, rng=rng2)
    caption = get_caption(metadata, rng=random.Random(0))
    assert caption.startswith(REALIFY_ANCHOR)
    assert "Preserve the exact melody" in caption
    assert "TrackType: Instrument" in caption
    assert "rock" in caption
    assert "NA" not in caption
    assert "format: solo" in caption


def test_instrument_hint_drum():
    assert "drums" in instrument_hint(program=0, is_drum=True, name=None)


def test_instrument_hint_missing_name_uses_program():
    assert instrument_hint(program=42, is_drum=False, name=None) == (
        "format: solo | instruments: midi program 42"
    )
    assert instrument_hint(program=42, is_drum=False, name=float("nan")) == (
        "format: solo | instruments: midi program 42"
    )
    assert instrument_hint(program=42, is_drum=False, name="") == (
        "format: solo | instruments: midi program 42"
    )


def test_get_caption_minimal_variant():
    metadata = {
        "program": 0,
        "is_drum": False,
        "name": "Piano",
    }
    caption = get_caption(metadata, prompt_variant="minimal")
    assert caption == "solo piano, realistic studio recording, expressive performance"


def test_get_caption_preservation_variant():
    metadata = {
        "program": 0,
        "is_drum": False,
        "name": "Piano",
        "genres": "rock",
    }
    caption = get_caption(metadata, prompt_variant="preservation")
    assert "Preserve the exact melody" in caption
    assert "rock" not in caption
    assert "format: solo | instruments: piano" in caption


def test_generate_captions_join(tmp_path):
    song_dir = str(tmp_path / "data" / "song1")
    songs = pd.DataFrame({
        "path": [song_dir],
        "genres": ["jazz"],
        "groups": [pd.NA],
        "tags": ["swing"],
        "song_name": ["Tune"],
        "title": [pd.NA],
        "subtitle": [pd.NA],
        "artist_name": [pd.NA],
        "composer_name": [pd.NA],
        "publisher": [pd.NA],
        "complexity": [1],
        "license": [pd.NA],
    })
    stems = pd.DataFrame({
        "path": [song_dir],
        "track": [0],
        "program": [11],
        "is_drum": [False],
        "name": ["Vibraphone"],
        "has_lyrics": [False],
    })
    songs.to_csv(tmp_path / "data.csv", index=False)
    stems.to_csv(tmp_path / "stems.csv", index=False)

    captions = generate_captions(str(tmp_path), seed=1)
    assert len(captions) == 1
    assert captions.iloc[0]["path"] == song_dir
    assert "jazz" in captions.iloc[0]["prompt"]


def test_generate_captions_uses_preset_prompt_variant(tmp_path):
    song_dir = str(tmp_path / "data" / "song1")
    songs = pd.DataFrame({"path": [song_dir], "genres": ["jazz"]})
    stems = pd.DataFrame({
        "path": [song_dir],
        "track": [0],
        "program": [0],
        "is_drum": [False],
        "name": ["Piano"],
        "has_lyrics": [False],
    })
    presets = {
        "default": {"prompt_variant": "current"},
        "categories": {"piano": {"prompt_variant": "minimal"}},
        "routing": [{"category": "piano", "name_keywords": ["piano"]}],
    }

    captions = generate_captions_from_tables(
        songs, stems, seed=1, presets=presets,
    )
    assert captions.iloc[0]["prompt"].startswith(
        "solo piano, realistic studio recording"
    )
