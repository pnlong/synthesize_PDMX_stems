"""Tests for content-rich clip window selection."""

from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.patch_sweep.clip_windows import find_content_rich_clips
from experiments.patch_sweep.make_clips import build_clip_windows
from shared.config import SAMPLE_RATE


def _write_test_stem(path: Path, *, duration_seconds: float = 60.0) -> None:
    samples = int(duration_seconds * SAMPLE_RATE)
    audio = np.zeros((samples, 1), dtype=np.float32)
    # Sparse intro/outro, dense middle burst.
    audio[int(5 * SAMPLE_RATE) : int(8 * SAMPLE_RATE)] = 0.05
    audio[int(25 * SAMPLE_RATE) : int(40 * SAMPLE_RATE)] = 0.25
    audio[int(50 * SAMPLE_RATE) : int(53 * SAMPLE_RATE)] = 0.04
    sf.write(str(path), audio, SAMPLE_RATE)


def test_find_content_rich_clips_avoids_sparse_edges(tmp_path: Path):
    path = tmp_path / "stem_0.flac"
    _write_test_stem(path)

    starts = find_content_rich_clips(path, n_clips=2, clip_seconds=10.0)
    assert len(starts) == 2
    assert all(start >= 10.0 for start in starts)
    assert all(start <= 40.0 for start in starts)
    assert abs(starts[0] - starts[1]) >= 5.0


def test_find_content_rich_clips_non_overlapping(tmp_path: Path):
    path = tmp_path / "stem_0.flac"
    _write_test_stem(path, duration_seconds=90.0)

    starts = find_content_rich_clips(
        path,
        n_clips=3,
        clip_seconds=10.0,
        min_separation_seconds=5.0,
    )
    assert len(starts) == 3
    for left, right in zip(starts, starts[1:]):
        assert right - left >= 5.0


def test_build_clip_windows_parallel_matches_serial(tmp_path: Path):
    source_dir = tmp_path / "basic"
    song_dir = source_dir / "data" / "0/13/QmTest"
    song_dir.mkdir(parents=True)
    _write_test_stem(song_dir / "stem_0.flac", duration_seconds=90.0)

    probes = [{
        "id": "piano_test",
        "category": "piano",
        "song_id": "0/13/QmTest",
        "track": 0,
    }]
    serial = build_clip_windows(
        probes=probes,
        source_dir=source_dir,
        clips_per_stem=2,
        clip_seconds=10.0,
        jobs=1,
    )
    parallel = build_clip_windows(
        probes=probes,
        source_dir=source_dir,
        clips_per_stem=2,
        clip_seconds=10.0,
        jobs=2,
    )
    assert parallel == serial
