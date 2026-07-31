"""Resample neural-DDSP waveforms to the pipeline sample rate / channel layout."""

from __future__ import annotations

import numpy as np
import torch
import torchaudio

from shared.config import SAMPLE_RATE, STEM_CHANNELS
from synthesis.audio import ensure_stem_channels, truncate_waveform


def numpy_audio_to_stem_tensor(
    audio: np.ndarray,
    *,
    source_sr: int,
    target_sr: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Convert mono or (samples,) / (channels, samples) float audio to stem tensor."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1:
        waveform = torch.from_numpy(arr.copy()).unsqueeze(0)
    elif arr.ndim == 2:
        # Prefer (channels, samples); if (samples, channels) with few channels, transpose.
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
            waveform = torch.from_numpy(arr.copy())
        else:
            waveform = torch.from_numpy(arr.T.copy())
    else:
        arr = arr.reshape(-1)
        waveform = torch.from_numpy(arr.copy()).unsqueeze(0)

    if source_sr != target_sr and waveform.numel() > 0:
        waveform = torchaudio.functional.resample(waveform, source_sr, target_sr)

    waveform = ensure_stem_channels(truncate_waveform(waveform))
    if STEM_CHANNELS == 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform
