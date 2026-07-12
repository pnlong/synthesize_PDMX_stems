"""Tests for realify stem chunking."""

import torch

from synthesis.realify.chunking import (
    fit_chunk_length,
    max_realify_chunk_seconds,
    needs_chunking,
    plan_chunk_spans,
    realify_overlap_samples,
    stitch_chunk_outputs,
)


class FakeModel:
    def __init__(self, sample_size: int, sample_rate: int = 44100):
        self.model_config = {"sample_size": sample_size}
        self.model = type("M", (), {"sample_rate": sample_rate})()


def test_max_realify_chunk_seconds_accounts_for_padding():
    model = FakeModel(sample_size=44100 * 120)
    assert max_realify_chunk_seconds(model) == 120 - 6.0


def test_needs_chunking():
    model = FakeModel(sample_size=44100 * 120)
    chunk_samples = int(max_realify_chunk_seconds(model) * 44100)
    assert not needs_chunking(chunk_samples, model)
    assert needs_chunking(chunk_samples + 1, model)


def test_plan_chunk_spans_single_chunk():
    assert plan_chunk_spans(1000, 5000, 200) == [(0, 1000)]


def test_plan_chunk_spans_multiple_with_overlap():
    spans = plan_chunk_spans(total_samples=1000, chunk_samples=400, overlap_samples=100)
    assert spans[0] == (0, 400)
    assert spans[1] == (300, 700)
    assert spans[2] == (600, 1000)


def test_stitch_chunk_outputs_blends_overlap():
    spans = [(0, 4), (2, 6)]
    chunks = [
        torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    ]
    stitched = stitch_chunk_outputs(chunks, spans, overlap_samples=2)
    assert stitched.shape == (1, 6)
    assert stitched[0, 0] == 1.0
    assert stitched[0, -1] == 1.0
    assert stitched[0, 3] < 1.0


def test_fit_chunk_length_pads_and_trims():
    chunk = torch.ones(1, 3)
    assert fit_chunk_length(chunk, 5).shape[-1] == 5
    assert fit_chunk_length(chunk, 2).shape[-1] == 2


def test_stitch_chunk_outputs_supports_stereo_chunks():
    spans = [(0, 4), (2, 6)]
    chunks = [
        torch.tensor([[1.0, 1.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.5]]),
    ]
    stitched = stitch_chunk_outputs(chunks, spans, overlap_samples=2)
    assert stitched.shape == (2, 6)


def test_realify_overlap_samples():
    assert realify_overlap_samples() == int(2.0 * 44100)
