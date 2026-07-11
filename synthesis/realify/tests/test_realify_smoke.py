"""Optional smoke test for realify (requires GPU + SA3 submodule)."""

from pathlib import Path

import pytest


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _sa3_available() -> bool:
    sa3_path = Path(__file__).parents[1] / "stable-audio-3"
    return sa3_path.exists() and (sa3_path / "stable_audio_3").exists()


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
@pytest.mark.skipif(not _sa3_available(), reason="stable-audio-3 submodule not initialized")
def test_realify_import():
    from stable_audio_3 import StableAudioModel  # noqa: F401
