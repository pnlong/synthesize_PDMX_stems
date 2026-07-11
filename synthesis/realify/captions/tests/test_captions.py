"""Tests for caption generation."""

import random

import pandas as pd

from synthesis.realify.captions.generate import generate_captions
from synthesis.realify.captions.metadata import get_caption, instrument_hint


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
    assert "rock" in caption
    assert "NA" not in caption
    assert "format: solo" in caption


def test_instrument_hint_drum():
    assert "drums" in instrument_hint(program=0, is_drum=True, name=None)


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
