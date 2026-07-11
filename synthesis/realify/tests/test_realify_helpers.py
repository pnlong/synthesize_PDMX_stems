"""Unit tests for realify task building and device selection."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from synthesis.realify.realify import (
    build_realify_tasks,
    realify_uses_gpu,
)


def test_realify_uses_gpu_medium_requires_cuda():
    with patch("synthesis.realify.realify.visible_cuda_count", return_value=0):
        with pytest.raises(RuntimeError, match="medium requires a GPU"):
            realify_uses_gpu("medium")


def test_realify_uses_gpu_medium_with_cuda():
    with patch("synthesis.realify.realify.visible_cuda_count", return_value=2):
        assert realify_uses_gpu("medium") is True


def test_realify_uses_gpu_small_music_prefers_cuda():
    with patch("synthesis.realify.realify.visible_cuda_count", return_value=1):
        assert realify_uses_gpu("small-music") is True


def test_realify_uses_gpu_small_music_falls_back_to_cpu():
    with patch("synthesis.realify.realify.visible_cuda_count", return_value=0):
        assert realify_uses_gpu("small-music") is False


def test_build_realify_tasks_skips_existing(tmp_path: Path):
    source_dir = tmp_path / "basic"
    output_dir = tmp_path / "basic_realify"
    song_dir = source_dir / "data" / "song"
    out_song_dir = output_dir / "data" / "song"
    song_dir.mkdir(parents=True)
    out_song_dir.mkdir(parents=True)

    sr = 44100
    sf.write(str(song_dir / "stem_0.flac"), np.zeros(sr), sr, format="FLAC")
    sf.write(str(song_dir / "stem_1.flac"), np.zeros(sr), sr, format="FLAC")
    sf.write(str(out_song_dir / "stem_0.flac"), np.zeros(sr), sr, format="FLAC")

    captions = pd.DataFrame({
        "path": [str(song_dir), str(song_dir)],
        "track": [0, 1],
        "prompt": ["piano", "drums"],
        "is_drum": [False, True],
        "name": [None, None],
    })

    tasks = build_realify_tasks(captions, source_dir, output_dir)
    assert len(tasks) == 1
    assert tasks[0]["out_path"] == str(out_song_dir / "stem_1.flac")


def test_build_realify_tasks_skips_invalid_stem(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "basic"
    output_dir = tmp_path / "basic_realify"
    song_dir = source_dir / "data" / "song"
    song_dir.mkdir(parents=True)

    sr = 44100
    sf.write(str(song_dir / "stem_0.flac"), np.zeros(sr), sr, format="FLAC")
    (song_dir / "stem_1.flac").write_bytes(b"bad")

    captions = pd.DataFrame({
        "path": [str(song_dir), str(song_dir)],
        "track": [0, 1],
        "prompt": ["piano", "drums"],
    })

    monkeypatch.setattr(
        "synthesis.realify.realify.stem_is_valid",
        lambda path: path.name == "stem_0.flac",
    )
    tasks = build_realify_tasks(captions, source_dir, output_dir)
    assert len(tasks) == 1
    assert tasks[0]["stem_path"].endswith("stem_0.flac")


def test_build_realify_tasks_uses_mp3_when_requested(tmp_path: Path):
    from shared.config import PROTOTYPE_AUDIO_FORMAT

    source_dir = tmp_path / "basic"
    output_dir = tmp_path / "basic_realify"
    song_dir = source_dir / "data" / "song"
    out_song_dir = output_dir / "data" / "song"
    song_dir.mkdir(parents=True)
    out_song_dir.mkdir(parents=True)

    sr = 44100
    sf.write(str(song_dir / "stem_0.mp3"), np.zeros(sr), sr, format="MP3")

    captions = pd.DataFrame({
        "path": [str(song_dir)],
        "track": [0],
        "prompt": ["piano"],
    })

    tasks = build_realify_tasks(
        captions, source_dir, output_dir, audio_format=PROTOTYPE_AUDIO_FORMAT,
    )
    assert len(tasks) == 1
    assert tasks[0]["stem_path"].endswith("stem_0.mp3")
    assert tasks[0]["out_path"].endswith("stem_0.mp3")
