"""Tests for shortlist aggregation."""

import pandas as pd

from experiments.listening.aggregate import (
    DEFAULT_MEAN_RATING_THRESHOLD,
    shortlist_variants,
)


def test_shortlist_variants_includes_passing_ratings():
    df = pd.DataFrame([
        {"stem_id": "piano_a", "category": "piano", "variant_id": "sgm_v2",
         "content": 5.0, "realism": 4.0},
        {"stem_id": "piano_a", "category": "piano", "variant_id": "airfont_380",
         "content": 4.0, "realism": 5.0},
        {"stem_id": "piano_a", "category": "piano", "variant_id": "fluidr3",
         "content": 2.0, "realism": 5.0},
        {"stem_id": "piano_b", "category": "piano", "variant_id": "sgm_v2",
         "content": 4.0, "realism": 4.0},
        {"stem_id": "piano_b", "category": "piano", "variant_id": "airfont_380",
         "content": 4.0, "realism": 4.0},
        {"stem_id": "piano_b", "category": "piano", "variant_id": "fluidr3",
         "content": 2.0, "realism": 2.0},
    ])
    shortlists = shortlist_variants(df, mean_rating_threshold=DEFAULT_MEAN_RATING_THRESHOLD)
    assert shortlists["piano"] == ["airfont_380", "sgm_v2"]


def test_shortlist_variants_falls_back_to_best_when_none_pass():
    df = pd.DataFrame([
        {"stem_id": "piano_a", "category": "piano", "variant_id": "sgm_v2",
         "content": 3.0, "realism": 3.0},
        {"stem_id": "piano_a", "category": "piano", "variant_id": "fluidr3",
         "content": 2.0, "realism": 2.0},
    ])
    shortlists = shortlist_variants(df, mean_rating_threshold=DEFAULT_MEAN_RATING_THRESHOLD)
    assert shortlists["piano"] == ["sgm_v2"]
