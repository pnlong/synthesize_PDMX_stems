"""Tests for webMUSHRA config generation."""

from pathlib import Path

import pytest
import yaml

from experiments.ablation_listening.webmushra import (
    build_mushra_trial_page,
    build_webmushra_config,
    convert_to_wav,
    export_trial_wavs,
)


@pytest.fixture
def mini_trial(tmp_path: Path):
    clips = tmp_path / "clips" / "mix_01"
    clips.mkdir(parents=True)
    for cond in ("basic", "basic_realify", "slakh", "slakh_realify"):
        # minimal valid wav
        import numpy as np
        import soundfile as sf
        sf.write(str(clips / f"{cond}.wav"), np.zeros((44100, 1), dtype="float32"), 44100)
    return {
        "id": "mix_01",
        "type": "mixture",
        "song_id": "1/2/QmTest",
        "conditions": {c: f"mix_01/{c}.wav" for c in ("basic", "basic_realify", "slakh", "slakh_realify")},
    }


def test_build_mushra_trial_hidden_reference():
    wav_paths = {
        "basic": "stimuli/spdmx_ablation/mix_01/basic.wav",
        "basic_realify": "stimuli/spdmx_ablation/mix_01/basic_realify.wav",
        "slakh": "stimuli/spdmx_ablation/mix_01/slakh.wav",
        "slakh_realify": "stimuli/spdmx_ablation/mix_01/slakh_realify.wav",
    }
    page = build_mushra_trial_page({"id": "mix_01", "type": "mixture"}, wav_paths)
    assert page["reference"] == wav_paths["basic"]
    assert page["stimuli"]["basic"] == wav_paths["basic"]
    assert page["randomize"] is True
    assert page["showConditionNames"] is False
    assert page["createAnchor35"] is False


def test_export_trial_wavs(tmp_path: Path, mini_trial: dict):
    webmushra = tmp_path / "webMUSHRA"
    webmushra.mkdir()
    (webmushra / "index.html").write_text("<html></html>")
    clips_dir = tmp_path / "clips"
    paths = export_trial_wavs(mini_trial, clips_dir, webmushra)
    for cond, rel in paths.items():
        assert (webmushra / rel).is_file(), cond


def test_build_config_has_randomized_trials(mini_trial: dict):
    mini_trial["webmushra_wav_paths"] = {
        c: f"stimuli/spdmx_ablation/mix_01/{c}.wav"
        for c in ("basic", "basic_realify", "slakh", "slakh_realify")
    }
    config = build_webmushra_config(
        {"trials": [mini_trial]},
        volume_stimulus="stimuli/spdmx_ablation/mix_01/basic.wav",
    )
    assert config["testId"] == "spdmx_ablation"
    pages = config["pages"]
    assert pages[2] == "random"
    mushra = [p for p in pages if isinstance(p, dict) and p.get("type") == "mushra"]
    assert len(mushra) == 1
