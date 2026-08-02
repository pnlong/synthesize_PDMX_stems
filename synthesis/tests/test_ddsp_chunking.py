"""Unit tests for DDSP-Piano overlap chunk planning / stitching."""

from __future__ import annotations

import numpy as np
import pytest

from synthesis.ddsp.chunking import (
    frames_to_samples,
    plan_chunk_frame_spans,
    stitch_audio_chunks,
)


def test_plan_chunk_frame_spans_single():
    assert plan_chunk_frame_spans(100, chunk_frames=200, overlap_frames=50) == [(0, 100)]


def test_plan_chunk_frame_spans_with_overlap():
    spans = plan_chunk_frame_spans(1000, chunk_frames=400, overlap_frames=100)
    assert spans[0] == (0, 400)
    assert spans[1] == (300, 700)
    assert spans[-1][1] == 1000
    for prev, cur in zip(spans, spans[1:]):
        assert cur[0] == prev[0] + 300
        assert cur[0] < prev[1]


def test_stitch_audio_chunks_blends_overlap():
    chunks = [
        np.ones(8, dtype=np.float32),
        np.full(8, 3.0, dtype=np.float32),
    ]
    spans = [(0, 8), (4, 12)]
    stitched = stitch_audio_chunks(chunks, spans, overlap_samples=4)
    assert stitched.shape == (12,)
    assert stitched[0] == pytest.approx(1.0)
    assert stitched[11] == pytest.approx(3.0)
    # At global sample 6: chunk1 weight 1/3, chunk2 weight 2/3 → 7/3.
    assert stitched[6] == pytest.approx(7.0 / 3.0)


def test_stitch_zero_overlap_matches_concat_boundaries():
    chunks = [np.arange(4, dtype=np.float32), np.arange(4, 8, dtype=np.float32)]
    spans = [(0, 4), (4, 8)]
    stitched = stitch_audio_chunks(chunks, spans, overlap_samples=0)
    np.testing.assert_allclose(stitched, np.arange(8, dtype=np.float32))


def test_frames_to_samples():
    assert frames_to_samples(250, frame_rate=250, sample_rate=16000) == 16000
