"""Tests for webMUSHRA CSV aggregation (category × condition × scale)."""

from pathlib import Path

import pandas as pd
import yaml

from experiments.ablation_listening.aggregate_webmushra import (
    expand_equivalence_scores,
    load_mushra_csv,
    summarize_mushra,
)


def test_load_and_summarize_dual_scale(tmp_path: Path):
    csv_path = tmp_path / "mushra.csv"
    csv_path.write_text(
        "\n".join([
            "session_uuid,listener_id,trial_id,rating_stimulus,rating_score",
            "s1,alice,stem_piano_01__content,basic,95",
            "s1,alice,stem_piano_01__content,basic_realify,80",
            "s1,alice,stem_piano_01__content,slakh,70",
            "s1,alice,stem_piano_01__content,slakh_realify,75",
            "s1,alice,stem_piano_01__content,ddsp_basic,72",
            "s1,alice,stem_piano_01__content,ddsp_basic_realify,68",
            "s1,alice,stem_piano_01__content,ddsp_slakh,65",
            "s1,alice,stem_piano_01__content,ddsp_slakh_realify,60",
            "s1,alice,stem_piano_01__realism,basic,50",
            "s1,alice,stem_piano_01__realism,basic_realify,85",
            "s1,alice,stem_piano_01__realism,slakh,55",
            "s1,alice,stem_piano_01__realism,slakh_realify,90",
            "s1,alice,stem_piano_01__realism,ddsp_basic,70",
            "s1,alice,stem_piano_01__realism,ddsp_basic_realify,88",
            "s1,alice,stem_piano_01__realism,ddsp_slakh,60",
            "s1,alice,stem_piano_01__realism,ddsp_slakh_realify,82",
            "s1,alice,stem_drums_01__content,basic,90",
            "s1,alice,stem_drums_01__content,basic_realify,70",
            "s1,alice,stem_drums_01__content,slakh,88",
            "s1,alice,stem_drums_01__content,slakh_realify,85",
            "s1,alice,stem_drums_01__content,ddsp_basic,80",
            "s1,alice,stem_drums_01__content,ddsp_basic_realify,78",
            "s1,alice,stem_drums_01__content,ddsp_slakh,84",
            "s1,alice,stem_drums_01__content,ddsp_slakh_realify,81",
            "s1,alice,stem_drums_01__realism,basic,40",
            "s1,alice,stem_drums_01__realism,basic_realify,75",
            "s1,alice,stem_drums_01__realism,slakh,80",
            "s1,alice,stem_drums_01__realism,slakh_realify,92",
            "s1,alice,stem_drums_01__realism,ddsp_basic,77",
            "s1,alice,stem_drums_01__realism,ddsp_basic_realify,86",
            "s1,alice,stem_drums_01__realism,ddsp_slakh,79",
            "s1,alice,stem_drums_01__realism,ddsp_slakh_realify,89",
            "",
        ])
    )
    df = load_mushra_csv(csv_path)
    assert set(df["scale"]) == {"content", "realism"}
    assert set(df["category"]) == {"piano", "drums"}
    assert len(df) == 32
    assert not df["auto_assigned"].any()

    summary = summarize_mushra(df)
    assert "piano" in summary["by_category"]
    assert "drums" in summary["by_category"]
    assert summary["by_category"]["piano"]["realism"]["slakh_realify"] == 90.0
    assert summary["by_category"]["drums"]["content"]["slakh"] == 88.0
    assert summary["factorial_realism"]["slakh"]["realified"] == 91.0  # mean(90, 92)
    assert summary["winner"] == "slakh_realify"


def test_legacy_page_id_defaults_to_realism(tmp_path: Path):
    csv_path = tmp_path / "mushra.csv"
    csv_path.write_text(
        "session_uuid,listener_id,trial_id,rating_stimulus,rating_score\n"
        "s1,bob,stem_piano,basic_realify,77\n"
    )
    df = load_mushra_csv(csv_path)
    assert df.iloc[0]["scale"] == "realism"
    assert df.iloc[0]["category"] == "piano"


def test_expand_equivalence_scores_from_manifest(tmp_path: Path):
    """Drums page rated only unique donors; ddsp_* scores are auto-copied."""
    manifest_path = tmp_path / "trial_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump({
            "trials": [{
                "id": "stem_drums_01",
                "equivalences": {
                    "ddsp_basic": "basic",
                    "ddsp_slakh": "slakh",
                    "ddsp_basic_realify": "basic_realify",
                    "ddsp_slakh_realify": "slakh_realify",
                },
            }],
        })
    )
    csv_path = tmp_path / "mushra.csv"
    csv_path.write_text(
        "\n".join([
            "session_uuid,listener_id,trial_id,rating_stimulus,rating_score",
            "s1,alice,stem_drums_01__content,basic,90",
            "s1,alice,stem_drums_01__content,basic_realify,70",
            "s1,alice,stem_drums_01__content,slakh,88",
            "s1,alice,stem_drums_01__content,slakh_realify,85",
            "s1,alice,stem_drums_01__realism,basic,40",
            "s1,alice,stem_drums_01__realism,basic_realify,75",
            "s1,alice,stem_drums_01__realism,slakh,80",
            "s1,alice,stem_drums_01__realism,slakh_realify,92",
            "",
        ])
    )
    df = load_mushra_csv(csv_path, manifest_path=manifest_path)
    assert len(df) == 16  # 8 rated + 8 auto
    assert int(df["auto_assigned"].sum()) == 8

    auto = df[df["auto_assigned"]]
    assert set(auto["condition_id"]) == {
        "ddsp_basic",
        "ddsp_slakh",
        "ddsp_basic_realify",
        "ddsp_slakh_realify",
    }
    content = df[(df["scale"] == "content") & (df["condition_id"] == "ddsp_basic")]
    assert float(content.iloc[0]["score"]) == 90.0
    assert content.iloc[0]["source_condition"] == "basic"

    realism_slakh = df[
        (df["scale"] == "realism") & (df["condition_id"] == "ddsp_slakh_realify")
    ]
    assert float(realism_slakh.iloc[0]["score"]) == 92.0

    summary = summarize_mushra(df)
    assert summary["by_category"]["drums"]["content"]["ddsp_basic"] == 90.0
    assert summary["by_category"]["drums"]["realism"]["ddsp_slakh_realify"] == 92.0


def test_expand_skips_when_duplicate_already_rated():
    df = pd.DataFrame([
        {
            "listener_id": "alice",
            "page_id": "stem_drums_01__content",
            "trial_id": "stem_drums_01",
            "scale": "content",
            "category": "drums",
            "trial_type": "stem",
            "condition_id": "basic",
            "score": 90.0,
            "auto_assigned": False,
            "source_condition": None,
        },
        {
            "listener_id": "alice",
            "page_id": "stem_drums_01__content",
            "trial_id": "stem_drums_01",
            "scale": "content",
            "category": "drums",
            "trial_type": "stem",
            "condition_id": "ddsp_basic",
            "score": 91.0,
            "auto_assigned": False,
            "source_condition": None,
        },
    ])
    out = expand_equivalence_scores(df, {"stem_drums_01": {"ddsp_basic": "basic"}})
    assert len(out) == 2
    assert not out["auto_assigned"].any()
