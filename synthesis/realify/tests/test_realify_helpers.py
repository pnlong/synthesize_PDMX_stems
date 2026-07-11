"""Unit tests for realify task building and GPU parsing."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from synthesis.realify.realify import build_realify_tasks, parse_gpu_ids


def test_parse_gpu_ids_explicit():
    with patch("torch.cuda.device_count", return_value=4):
        assert parse_gpu_ids("0,2") == [0, 2]
        assert parse_gpu_ids("all") == [0, 1, 2, 3]


def test_parse_gpu_ids_default():
    with patch("torch.cuda.device_count", return_value=2):
        assert parse_gpu_ids(None) == [0, 1]


def test_parse_gpu_ids_no_cuda():
    with patch("torch.cuda.device_count", return_value=0):
        with pytest.raises(RuntimeError, match="No CUDA GPUs"):
            parse_gpu_ids(None)


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
