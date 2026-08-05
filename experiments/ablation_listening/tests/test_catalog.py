import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from experiments.ablation_listening.aggregate import aggregate_responses, ratings_dataframe
from experiments.ablation_listening.catalog import AblationListeningCatalog
from experiments.ablation_listening.conditions import ABLATION_MUSHRA_CONDITIONS
from experiments.ablation_listening.equivalence import DONOR_EQUIVALENCE_PAIRS


@pytest.fixture
def manifest_and_clips(tmp_path: Path):
    clips_dir = tmp_path / "clips"
    trial_dir = clips_dir / "mix_01"
    trial_dir.mkdir(parents=True)
    for cond in ABLATION_MUSHRA_CONDITIONS:
        (trial_dir / f"{cond}.mp3").write_bytes(b"\x00" * 128)

    manifest = tmp_path / "trial_manifest.yaml"
    doc = {
        "test_id": "test_v1",
        "trials": [{
            "id": "mix_01",
            "type": "mixture",
            "song_id": "1/2/QmTest",
            "track": None,
            "category": None,
            "clip_seconds": 10.0,
            "audio_format": "mp3",
            "conditions": {
                cond: f"mix_01/{cond}.mp3"
                for cond in ABLATION_MUSHRA_CONDITIONS
            },
        }],
    }
    with open(manifest, "w") as f:
        yaml.safe_dump(doc, f)
    return manifest, clips_dir


def test_catalog_get_trial_includes_basic(manifest_and_clips):
    manifest, clips_dir = manifest_and_clips
    catalog = AblationListeningCatalog(manifest, clips_dir)
    detail = catalog.get_trial("mix_01", session_seed=42)
    assert detail is not None
    assert detail["reference"]["condition_id"] == "basic"
    assert detail["reference"]["available"] is True
    assert detail["n_unique"] == len(ABLATION_MUSHRA_CONDITIONS)
    assert len(detail["samples"]) == len(ABLATION_MUSHRA_CONDITIONS)
    assert {s["condition_id"] for s in detail["samples"]} == set(ABLATION_MUSHRA_CONDITIONS)
    assert all(sample["available"] for sample in detail["samples"])


def test_catalog_omits_equivalences(tmp_path: Path):
    clips_dir = tmp_path / "clips"
    trial_dir = clips_dir / "stem_drums_01"
    trial_dir.mkdir(parents=True)
    for cond in ABLATION_MUSHRA_CONDITIONS:
        (trial_dir / f"{cond}.mp3").write_bytes(b"\x00" * 64)

    manifest = tmp_path / "trial_manifest.yaml"
    with open(manifest, "w") as f:
        yaml.safe_dump({
            "test_id": "test_v1",
            "trials": [{
                "id": "stem_drums_01",
                "type": "stem",
                "category": "drums",
                "song_id": "1/2/QmDrums",
                "track": 0,
                "clip_seconds": 10.0,
                "audio_format": "mp3",
                "equivalences": dict(DONOR_EQUIVALENCE_PAIRS),
                "conditions": {
                    c: f"stem_drums_01/{c}.mp3" for c in ABLATION_MUSHRA_CONDITIONS
                },
            }],
        }, f)

    catalog = AblationListeningCatalog(manifest, clips_dir)
    detail = catalog.get_trial("stem_drums_01", session_seed=7)
    assert detail["n_unique"] == 4
    assert len(detail["samples"]) == 4
    assert {s["condition_id"] for s in detail["samples"]} == {
        "basic",
        "basic_realify",
        "slakh",
        "slakh_realify",
    }
    assert detail["equivalences"] == dict(DONOR_EQUIVALENCE_PAIRS)


def test_aggregate_responses(manifest_and_clips):
    responses = {
        "listener_id": "tester",
        "ratings": [{
            "trial_id": "mix_01",
            "trial_type": "mixture",
            "category": None,
            "samples": [
                {"blind_label": "A", "condition_id": "basic", "content": 95, "realism": 55},
                {"blind_label": "B", "condition_id": "basic_realify", "content": 75, "realism": 85},
                {"blind_label": "C", "condition_id": "slakh", "content": 78, "realism": 72},
                {"blind_label": "D", "condition_id": "slakh_realify", "content": 82, "realism": 88},
                {"blind_label": "E", "condition_id": "ddsp_basic", "content": 90, "realism": 60},
                {"blind_label": "F", "condition_id": "ddsp_basic_realify", "content": 70, "realism": 80},
                {"blind_label": "G", "condition_id": "ddsp_slakh", "content": 76, "realism": 70},
                {"blind_label": "H", "condition_id": "ddsp_slakh_realify", "content": 80, "realism": 86},
            ],
        }],
    }
    path = manifest_and_clips[0].parent / "responses.json"
    path.write_text(json.dumps(responses))
    df = ratings_dataframe(responses)
    assert len(df) == 8
    assert float(df.loc[df["condition_id"] == "basic", "content"].iloc[0]) == 95.0
    _, summary = aggregate_responses([path])
    assert summary["winner"] == "slakh_realify"


def test_aggregate_expands_equivalences(tmp_path: Path):
    manifest = tmp_path / "trial_manifest.yaml"
    manifest.write_text(
        yaml.safe_dump({
            "trials": [{
                "id": "stem_drums_01",
                "equivalences": dict(DONOR_EQUIVALENCE_PAIRS),
            }],
        })
    )
    responses = {
        "listener_id": "alice",
        "ratings": [{
            "trial_id": "stem_drums_01",
            "trial_type": "stem",
            "category": "drums",
            "samples": [
                {"blind_label": "A", "condition_id": "basic", "content": 90, "realism": 40},
                {"blind_label": "B", "condition_id": "basic_realify", "content": 70, "realism": 75},
                {"blind_label": "C", "condition_id": "slakh", "content": 88, "realism": 80},
                {"blind_label": "D", "condition_id": "slakh_realify", "content": 85, "realism": 92},
            ],
        }],
    }
    path = tmp_path / "responses.json"
    path.write_text(json.dumps(responses))
    df, summary = aggregate_responses([path], manifest_path=manifest)
    assert len(df) == 8
    assert int(df["auto_assigned"].sum()) == 4
    assert summary["n_auto_assigned"] == 4
    auto = df[df["condition_id"] == "ddsp_basic"].iloc[0]
    assert float(auto["content"]) == 90.0
    assert float(auto["realism"]) == 40.0
    assert auto["source_condition"] == "basic"


def test_aggregate_skips_in_progress_files(tmp_path: Path):
    from experiments.ablation_listening.aggregate import resolve_completed_response_paths

    responses_dir = tmp_path / "responses"
    responses_dir.mkdir()
    done = {
        "listener_id": "alice",
        "complete": True,
        "ratings": [{
            "trial_id": "stem_drums_01",
            "trial_type": "stem",
            "category": "drums",
            "samples": [
                {"blind_label": "A", "condition_id": "basic", "content": 90, "realism": 40},
                {"blind_label": "B", "condition_id": "slakh", "content": 80, "realism": 70},
            ],
        }],
    }
    progress = {
        "listener_id": "bob",
        "complete": False,
        "ratings": [{
            "trial_id": "stem_drums_01",
            "trial_type": "stem",
            "category": "drums",
            "samples": [
                {"blind_label": "A", "condition_id": "basic", "content": 10, "realism": 10},
            ],
        }],
    }
    done_path = responses_dir / "responses_alice_20260101T000000Z.json"
    progress_path = responses_dir / "responses_in_progress_bob.json"
    done_path.write_text(json.dumps(done))
    progress_path.write_text(json.dumps(progress))

    completed = resolve_completed_response_paths([responses_dir])
    assert completed == [done_path.resolve()]

    df, summary = aggregate_responses([responses_dir])
    assert set(df["listener_id"]) == {"alice"}
    assert summary["n_response_files"] == 1
    assert float(df[df["condition_id"] == "basic"]["content"].iloc[0]) == 90.0
