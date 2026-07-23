import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from experiments.ablation_listening.aggregate import aggregate_responses, ratings_dataframe
from experiments.ablation_listening.catalog import AblationListeningCatalog


@pytest.fixture
def manifest_and_clips(tmp_path: Path):
    clips_dir = tmp_path / "clips"
    trial_dir = clips_dir / "mix_01"
    trial_dir.mkdir(parents=True)
    for cond in ("basic", "basic_realify", "slakh", "slakh_realify"):
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
                for cond in ("basic", "basic_realify", "slakh", "slakh_realify")
            },
        }],
    }
    with open(manifest, "w") as f:
        yaml.safe_dump(doc, f)
    return manifest, clips_dir


def test_catalog_get_trial(manifest_and_clips):
    manifest, clips_dir = manifest_and_clips
    catalog = AblationListeningCatalog(manifest, clips_dir)
    detail = catalog.get_trial("mix_01", session_seed=42)
    assert detail is not None
    assert detail["reference"]["condition_id"] == "basic"
    assert detail["reference"]["available"] is True
    assert len(detail["samples"]) == 3
    assert all(sample["available"] for sample in detail["samples"])


def test_aggregate_responses(manifest_and_clips):
    responses = {
        "listener_id": "tester",
        "ratings": [{
            "trial_id": "mix_01",
            "trial_type": "mixture",
            "category": None,
            "samples": [
                {"is_reference": True, "condition_id": "basic", "realism": 55},
                {"blind_label": "A", "condition_id": "basic_realify", "content": 75, "realism": 85},
                {"blind_label": "B", "condition_id": "slakh", "content": 78, "realism": 72},
                {"blind_label": "C", "condition_id": "slakh_realify", "content": 82, "realism": 88},
            ],
        }],
    }
    path = manifest_and_clips[0].parent / "responses.json"
    path.write_text(json.dumps(responses))
    df = ratings_dataframe(responses)
    assert len(df) == 4
    assert pd.isna(df.loc[df["condition_id"] == "basic", "content"].iloc[0])
    _, summary = aggregate_responses([path])
    assert summary["winner"] == "slakh_realify"
