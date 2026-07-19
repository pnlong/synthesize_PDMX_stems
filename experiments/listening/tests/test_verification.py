"""Tests for verification listening workflow."""

import json
from pathlib import Path

import pandas as pd

from experiments.listening.verification import (
    PRESET_VERIFY_SOURCE,
    analyze_responses,
    build_preset_realify_verification_meta,
    build_verification_meta,
    validate_verification,
    winners_from_verification,
)


def _responses():
    return {
        "ratings": [
            {
                "stem_id": "piano_a",
                "category": "piano",
                "samples": [
                    {"variant_id": "good", "content": 5, "realism": 4},
                    {"variant_id": "bad", "content": 2, "realism": 5},
                    {"variant_id": "ok", "content": 4, "realism": 3},
                ],
            },
            {
                "stem_id": "piano_b",
                "category": "piano",
                "samples": [
                    {"variant_id": "good", "content": 4, "realism": 5},
                    {"variant_id": "bad", "content": 3, "realism": 4},
                    {"variant_id": "ok", "content": 4, "realism": 4},
                ],
            },
        ],
    }


def test_analyze_responses_filters_low_content():
    _, stats, eligible, winners = analyze_responses(_responses())
    assert "bad" not in set(eligible["variant_id"])
    assert winners.iloc[0]["variant_id"] == "good"


def test_winners_from_verification():
    doc = {
        "categories": [
            {
                "category": "piano",
                "approved": ["good", "ok"],
                "winner_variant_id": "ok",
            },
        ],
    }
    assert winners_from_verification(doc) == {"piano": "ok"}


def test_winners_from_verification_patch_shortlist():
    doc = {
        "sweep_type": "patch",
        "categories": [
            {
                "category": "piano",
                "approved": ["sgm_v2", "airfont_380"],
            },
        ],
    }
    assert winners_from_verification(doc, sweep_type="patch") == {
        "piano": ["sgm_v2", "airfont_380"],
    }


def test_validate_verification_patch_requires_one_soundfont():
    errors = validate_verification({
        "sweep_type": "patch",
        "categories": [{
            "category": "piano",
            "approved": [],
        }],
    })
    assert any("keep at least one soundfont" in err for err in errors)

    errors = validate_verification({
        "sweep_type": "patch",
        "categories": [{
            "category": "piano",
            "approved": ["sgm_v2"],
        }],
    })
    assert errors == []


def test_validate_verification_requires_approved_winner():
    errors = validate_verification({
        "sweep_type": "preset",
        "categories": [{
            "category": "piano",
            "approved": [],
            "winner_variant_id": "good",
        }],
    })
    assert any("approve" in err for err in errors)

    errors = validate_verification({
        "sweep_type": "preset",
        "categories": [{
            "category": "piano",
            "approved": ["good"],
            "winner_variant_id": "ok",
        }],
    })
    assert any("winner must be among approved" in err for err in errors)


def test_bypass_routing_rules_from_verification_partial_category():
    from experiments.listening.verification import (
        bypass_realify_from_verification,
        bypass_routing_rules_from_verification,
    )

    doc = {
        "categories": [{
            "category": "voice",
            "bypass_realify": False,
            "stems": [
                {
                    "stem_id": "voice_a",
                    "track_name": "Soprano",
                    "program": 52,
                    "is_drum": False,
                    "bypass_realify": True,
                },
                {
                    "stem_id": "voice_b",
                    "track_name": "Choir",
                    "program": 52,
                    "is_drum": False,
                    "bypass_realify": False,
                },
            ],
        }],
    }
    assert bypass_realify_from_verification(doc) == {}
    rules = bypass_routing_rules_from_verification(doc)
    assert len(rules) == 1
    assert rules[0]["name_keywords"] == ["soprano"]


def test_bypass_realify_from_verification_all_stems():
    from experiments.listening.verification import bypass_realify_from_verification

    doc = {
        "categories": [{
            "category": "organ",
            "bypass_realify": False,
            "stems": [
                {"stem_id": "organ_a", "track_name": "Organ", "bypass_realify": True},
                {"stem_id": "organ_b", "track_name": "Organ 2", "bypass_realify": True},
            ],
        }],
    }
    assert bypass_realify_from_verification(doc) == {"organ": True}


def test_validate_verification_preset_bypass_skips_winner():
    errors = validate_verification({
        "sweep_type": "preset",
        "categories": [{
            "category": "organ",
            "approved": [],
            "winner_variant_id": None,
            "bypass_realify": True,
        }],
    })
    assert errors == []


def test_bypass_realify_from_verification():
    from experiments.listening.verification import bypass_realify_from_verification

    doc = {
        "categories": [
            {"category": "organ", "bypass_realify": True},
            {"category": "piano", "bypass_realify": False, "winner_variant_id": "good"},
        ],
    }
    assert bypass_realify_from_verification(doc) == {"organ": True}


def test_winners_from_verification_skips_bypass():
    doc = {
        "sweep_type": "preset",
        "categories": [
            {"category": "organ", "bypass_realify": True},
            {"category": "piano", "approved": ["good"], "winner_variant_id": "good"},
        ],
    }
    assert winners_from_verification(doc, sweep_type="preset") == {"piano": "good"}


def test_build_verification_meta_marks_filter_pass(tmp_path: Path):
    from experiments.listening.catalog import SweepCatalog
    import yaml

    sweep_dir = tmp_path / "sweep"
    song_dir = tmp_path / "basic" / "data" / "0/13/QmTest"
    song_dir.mkdir(parents=True)
    (song_dir / "stem_0.flac").write_bytes(b"x")

    for variant_id in ("good", "bad", "ok"):
        variant_dir = sweep_dir / "variants" / variant_id / "data" / "0/13/QmTest"
        variant_dir.mkdir(parents=True)
        (variant_dir / "stem_0.flac").write_bytes(b"x")

    rows = []
    for stem_id in ("piano_a", "piano_b"):
        for variant_id in ("good", "bad", "ok"):
            rows.append({
                "variant_id": variant_id,
                "init_noise_level": 0.45,
                "prompt_variant": "minimal",
                "stem_id": stem_id,
                "category": "piano",
                "path": str(song_dir),
                "track": 0,
                "out_path": str(sweep_dir / "variants" / variant_id / "data" / "0/13/QmTest" / "stem_0.flac"),
            })
    pd.DataFrame(rows).to_csv(sweep_dir / "manifest.csv", index=False)

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [
            {"id": "piano_a", "category": "piano", "song_id": "0/13/QmTest", "track": 0},
            {"id": "piano_b", "category": "piano", "song_id": "0/13/QmTest", "track": 0},
        ],
    }))

    catalog = SweepCatalog("preset", sweep_dir, tmp_path / "basic", probe_stems_path=probe_path)
    meta = build_verification_meta(
        catalog,
        _responses(),
        source_responses="responses_test.json",
    )
    piano = next(entry for entry in meta["categories"] if entry["category"] == "piano")
    passed = {v["variant_id"]: v["passed_filter"] for v in piano["variants"]}
    assert passed["good"] is True
    assert passed["bad"] is False
    assert piano["auto_winner_variant_id"] == "good"


def test_build_preset_realify_verification_meta(tmp_path: Path):
    from experiments.listening.catalog import SweepCatalog
    import yaml

    sweep_dir = tmp_path / "sweep"
    song_dir = tmp_path / "basic" / "data" / "0/13/QmTest"
    song_dir.mkdir(parents=True)
    (song_dir / "stem_0.flac").write_bytes(b"x")

    variant_dir = sweep_dir / "variants" / "steps8_cfg1.0" / "data" / "0/13/QmTest"
    variant_dir.mkdir(parents=True)
    (variant_dir / "stem_0.flac").write_bytes(b"x")

    pd.DataFrame([{
        "variant_id": "steps8_cfg1.0",
        "init_noise_level": 0.45,
        "prompt_variant": "minimal",
        "steps": 8,
        "cfg_scale": 1.0,
        "stem_id": "piano_a",
        "category": "piano",
        "path": str(song_dir),
        "track": 0,
        "out_path": str(variant_dir / "stem_0.flac"),
    }]).to_csv(sweep_dir / "manifest.csv", index=False)

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [
            {"id": "piano_a", "category": "piano", "song_id": "0/13/QmTest", "track": 0},
        ],
    }))

    catalog = SweepCatalog("preset", sweep_dir, tmp_path / "basic", probe_stems_path=probe_path)
    meta = build_preset_realify_verification_meta(
        catalog,
        category_winners={"piano": "steps8_cfg1.0"},
        composed_config_fn=lambda category, variant_id: {
            "variant_id": variant_id,
            "init_noise_level": 0.45,
            "prompt_variant": "minimal",
            "steps": 8,
            "cfg_scale": 1.0,
        },
        verification_phase="phase3_diffusion",
    )
    assert meta["verification_mode"] == "preset_realify"
    assert meta["source_responses"] == PRESET_VERIFY_SOURCE
    piano = meta["categories"][0]
    assert piano["auto_winner_variant_id"] == "steps8_cfg1.0"
    assert len(piano["variants"]) == 1
