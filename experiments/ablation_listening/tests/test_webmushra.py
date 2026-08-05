"""Tests for webMUSHRA config generation."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from experiments.ablation_listening.conditions import (
    ABLATION_MUSHRA_CONDITIONS,
    RATING_SCALES,
    category_from_trial_id,
    mushra_page_id,
    parse_mushra_page_id,
)
from experiments.ablation_listening.webmushra import (
    build_mushra_trial_page,
    build_webmushra_config,
    export_trial_wavs,
)


@pytest.fixture
def mini_trial(tmp_path: Path):
    clips = tmp_path / "clips" / "stem_piano_01"
    clips.mkdir(parents=True)
    for cond in ABLATION_MUSHRA_CONDITIONS:
        sf.write(
            str(clips / f"{cond}.wav"),
            np.zeros((44100, 1), dtype="float32"),
            44100,
        )
    return {
        "id": "stem_piano_01",
        "type": "stem",
        "category": "piano",
        "track": 0,
        "song_id": "1/2/QmTest",
        "conditions": {
            c: f"stem_piano_01/{c}.wav" for c in ABLATION_MUSHRA_CONDITIONS
        },
    }


def test_page_id_roundtrip():
    page_id = mushra_page_id("stem_piano_01", "content")
    assert page_id == "stem_piano_01__content"
    trial_id, scale = parse_mushra_page_id(page_id)
    assert trial_id == "stem_piano_01"
    assert scale == "content"
    assert category_from_trial_id(trial_id) == "piano"
    assert category_from_trial_id("stem_drums") == "drums"


def test_build_mushra_trial_hidden_reference():
    wav_paths = {
        c: f"stimuli/spdmx_ablation/stem_piano_01/{c}.wav"
        for c in ABLATION_MUSHRA_CONDITIONS
    }
    page = build_mushra_trial_page(
        {"id": "stem_piano_01", "type": "stem", "category": "piano", "track": 0},
        wav_paths,
        scale="content",
    )
    assert page["id"] == "stem_piano_01__content"
    assert page["reference"] == wav_paths["basic"]
    assert set(page["stimuli"]) == set(ABLATION_MUSHRA_CONDITIONS)
    assert page["randomize"] is True
    assert page["showConditionNames"] is False
    assert page["createAnchor35"] is False


def test_build_mushra_trial_omits_equivalences():
    wav_paths = {
        c: f"stimuli/spdmx_ablation/stem_drums_01/{c}.wav"
        for c in ABLATION_MUSHRA_CONDITIONS
    }
    trial = {
        "id": "stem_drums_01",
        "type": "stem",
        "category": "drums",
        "track": 0,
        "equivalences": {
            "ddsp_basic": "basic",
            "ddsp_slakh": "slakh",
            "ddsp_basic_realify": "basic_realify",
            "ddsp_slakh_realify": "slakh_realify",
        },
    }
    page = build_mushra_trial_page(trial, wav_paths, scale="realism")
    assert set(page["stimuli"]) == {
        "basic",
        "basic_realify",
        "slakh",
        "slakh_realify",
    }
    assert page["reference"] == wav_paths["basic"]
    assert "4 blind condition" in page["content"]


def test_export_trial_wavs(tmp_path: Path, mini_trial: dict):
    webmushra = tmp_path / "webMUSHRA"
    webmushra.mkdir()
    (webmushra / "index.html").write_text("<html></html>")
    clips_dir = tmp_path / "clips"
    paths = export_trial_wavs(mini_trial, clips_dir, webmushra)
    assert set(paths) == set(ABLATION_MUSHRA_CONDITIONS)
    for cond, rel in paths.items():
        assert (webmushra / rel).is_file(), cond


def test_build_config_dual_scale_pages(mini_trial: dict):
    mini_trial["webmushra_wav_paths"] = {
        c: f"stimuli/spdmx_ablation/stem_piano_01/{c}.wav"
        for c in ABLATION_MUSHRA_CONDITIONS
    }
    config = build_webmushra_config(
        {"trials": [mini_trial]},
        volume_stimulus="stimuli/spdmx_ablation/stem_piano_01/basic.wav",
    )
    assert config["testId"] == "spdmx_ablation"
    pages = config["pages"]
    assert pages[2] == "random"
    mushra = [p for p in pages if isinstance(p, dict) and p.get("type") == "mushra"]
    assert len(mushra) == len(RATING_SCALES)
    assert {p["id"] for p in mushra} == {
        "stem_piano_01__content",
        "stem_piano_01__realism",
    }
