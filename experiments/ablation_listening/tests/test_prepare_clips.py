"""Tests for ablation MUSHRA clip preparation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from experiments.ablation_listening.conditions import ABLATION_MUSHRA_CONDITIONS
from experiments.ablation_listening.prepare_clips import (
    reference_has_audible_clip,
    select_stem_trials,
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
