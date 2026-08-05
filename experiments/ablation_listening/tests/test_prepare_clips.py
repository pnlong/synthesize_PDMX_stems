"""Tests for ablation MUSHRA clip preparation."""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from experiments.ablation_listening.conditions import ABLATION_MUSHRA_CONDITIONS
from experiments.ablation_listening.prepare_clips import (
    annotate_manifest_equivalences,
    reference_has_audible_clip,
    select_stem_trials,
    write_trial_clips,
)


def _write_stem(path: Path, *, amplitude: float, seconds: float = 12.0, sr: int = 44100):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.full(int(sr * seconds), amplitude, dtype=np.float32)
    sf.write(str(path), audio, sr, format="MP3")


def _ablation_tree(tmp_path: Path) -> Path:
    """Minimal 8-condition tree: one audible piano stem, one silent piano stem."""
    song_rel = "0/0/QmAudible"
    silent_rel = "0/0/QmSilent"
    for condition in ABLATION_MUSHRA_CONDITIONS:
        root = tmp_path / condition
        root.mkdir(parents=True)
        for song_rel_i, amp, track in (
            (song_rel, 0.2, 0),
            (silent_rel, 0.0, 0),
        ):
            song_dir = root / "data" / song_rel_i
            _write_stem(song_dir / f"stem_{track}.mp3", amplitude=amp)

    # stems.csv / data.csv only needed under basic
    basic = tmp_path / "basic"
    rows = [
        {
            "path": str(basic / "data" / song_rel),
            "track": 0,
            "original_track": 0,
            "program": 0,
            "is_drum": False,
            "name": "Piano",
            "has_lyrics": False,
        },
        {
            "path": str(basic / "data" / silent_rel),
            "track": 0,
            "original_track": 0,
            "program": 0,
            "is_drum": False,
            "name": "Piano Silent",
            "has_lyrics": False,
        },
    ]
    pd.DataFrame(rows).to_csv(basic / "stems.csv", index=False)
    pd.DataFrame({
        "path": [str(basic / "data" / song_rel), str(basic / "data" / silent_rel)],
        "n_tracks": [1, 1],
        "title": ["Audible", "Silent"],
    }).to_csv(basic / "data.csv", index=False)
    return tmp_path


def test_reference_has_audible_clip_rejects_sparse_end_burst(tmp_path: Path):
    """Overall RMS can pass while material only fills the last ~2s — reject those."""
    song_rel = "0/0/QmSparse"
    for condition in ABLATION_MUSHRA_CONDITIONS:
        song_dir = tmp_path / condition / "data" / song_rel
        song_dir.mkdir(parents=True, exist_ok=True)
        sr = 44100
        audio = np.zeros(sr * 12, dtype=np.float32)
        audio[sr * 10 :] = 0.3
        sf.write(str(song_dir / "stem_0.mp3"), audio, sr, format="MP3")

    basic = tmp_path / "basic"
    pd.DataFrame([{
        "path": str(basic / "data" / song_rel),
        "track": 0,
        "program": 0,
        "is_drum": False,
        "name": "Piano",
        "has_lyrics": False,
    }]).to_csv(basic / "stems.csv", index=False)

    from experiments.ablation_listening.conditions import condition_roots

    roots = condition_roots(tmp_path)
    assert reference_has_audible_clip(song_rel, 0, roots) is False


def test_reference_has_audible_clip_rejects_silent(tmp_path: Path):
    root = _ablation_tree(tmp_path)
    from experiments.ablation_listening.conditions import condition_roots

    roots = condition_roots(root)
    assert reference_has_audible_clip("0/0/QmAudible", 0, roots) is True
    assert reference_has_audible_clip("0/0/QmSilent", 0, roots) is False


def test_select_stem_trials_skips_silent_reference(tmp_path: Path, monkeypatch):
    root = _ablation_tree(tmp_path)
    monkeypatch.setattr(
        "experiments.ablation_listening.prepare_clips.load_probe_stems",
        lambda: [],
    )
    # Only need one category with enough audible candidates.
    trials = select_stem_trials(
        root,
        categories=("piano",),
        stems_per_category=1,
        seed=0,
    )
    assert len(trials) == 1
    assert trials[0]["song_id"] == "0/0/QmAudible"


def test_write_trial_clips_records_donor_equivalences(tmp_path: Path):
    """Drums route to soundfont → all four DDSP↔donor equivalences."""
    song_rel = "0/0/QmCopy"
    for condition in ABLATION_MUSHRA_CONDITIONS:
        song_dir = tmp_path / condition / "data" / song_rel
        _write_stem(song_dir / "stem_0.mp3", amplitude=0.2)

    basic = tmp_path / "basic"
    pd.DataFrame([{
        "path": str(basic / "data" / song_rel),
        "track": 0,
        "program": 0,
        "is_drum": True,
        "name": "Drums",
        "has_lyrics": False,
    }]).to_csv(basic / "stems.csv", index=False)

    trial = {
        "id": "stem_drums_01",
        "type": "stem",
        "song_id": song_rel,
        "track": 0,
        "category": "drums",
    }
    clips_dir = tmp_path / "clips"
    prepared = write_trial_clips(trial, tmp_path, clips_dir, clip_seconds=2.0)
    assert prepared["equivalences"] == {
        "ddsp_basic": "basic",
        "ddsp_basic_realify": "basic_realify",
        "ddsp_slakh": "slakh",
        "ddsp_slakh_realify": "slakh_realify",
    }


def test_write_trial_clips_piano_has_no_equivalences(tmp_path: Path):
    song_rel = "0/0/QmPiano"
    for condition in ABLATION_MUSHRA_CONDITIONS:
        song_dir = tmp_path / condition / "data" / song_rel
        _write_stem(song_dir / "stem_0.mp3", amplitude=0.2)

    basic = tmp_path / "basic"
    pd.DataFrame([{
        "path": str(basic / "data" / song_rel),
        "track": 0,
        "program": 0,
        "is_drum": False,
        "name": "Piano",
        "has_lyrics": False,
    }]).to_csv(basic / "stems.csv", index=False)

    trial = {
        "id": "stem_piano_01",
        "type": "stem",
        "song_id": song_rel,
        "track": 0,
        "category": "piano",
    }
    prepared = write_trial_clips(trial, tmp_path, tmp_path / "clips", clip_seconds=2.0)
    assert "equivalences" not in prepared or prepared.get("equivalences") == {}


def test_annotate_manifest_equivalences(tmp_path: Path):
    song_rel = "0/0/QmDrums"
    basic = tmp_path / "basic"
    (basic / "data" / song_rel).mkdir(parents=True)
    pd.DataFrame([{
        "path": str(basic / "data" / song_rel),
        "track": 0,
        "program": 0,
        "is_drum": True,
        "name": "Drums",
        "has_lyrics": False,
    }]).to_csv(basic / "stems.csv", index=False)

    manifest_path = tmp_path / "trial_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump({
            "ablations_dir": str(tmp_path),
            "trials": [{
                "id": "stem_drums_01",
                "type": "stem",
                "song_id": song_rel,
                "track": 0,
                "category": "drums",
                "conditions": {
                    c: f"stem_drums_01/{c}.wav" for c in ABLATION_MUSHRA_CONDITIONS
                },
            }],
        })
    )
    doc = annotate_manifest_equivalences(manifest_path)
    assert doc["trials"][0]["equivalences"] == {
        "ddsp_basic": "basic",
        "ddsp_basic_realify": "basic_realify",
        "ddsp_slakh": "slakh",
        "ddsp_slakh_realify": "slakh_realify",
    }