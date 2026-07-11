"""Audio synthesis helpers using fluidsynth."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch

from shared.config import (
    GAIN,
    MIXTURE_FILE_NAME,
    MIXTURE_PEAK_LIMIT,
    SAMPLE_RATE,
    STEM_FILE_PATTERN,
    TARGET_LOUDNESS_LUFS,
)


def get_waveform_tensor(midi_path: str, soundfont_filepath: str) -> torch.Tensor:
    """Synthesize a MIDI file to a mono float waveform tensor (1, samples)."""
    result = subprocess.run(
        args=[
            "fluidsynth",
            "-T", "raw",
            "-F-", "-r", str(SAMPLE_RATE), "-g", str(GAIN),
            "-i", soundfont_filepath,
            midi_path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    stereo = np.frombuffer(result.stdout, dtype=np.int16).reshape(-1, 2)
    mono = stereo.mean(axis=1, keepdims=True).astype(np.float32) / np.iinfo(np.int16).max
    return torch.from_numpy(mono.T.copy())  # (1, samples)


def to_mono_numpy(waveform: torch.Tensor) -> np.ndarray:
    """Convert waveform tensor to mono numpy array (samples,)."""
    audio = waveform.numpy()
    if audio.ndim == 2:
        if audio.shape[0] == 1:
            return audio[0]
        return audio.mean(axis=0)
    return audio


def loudness_normalize(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    target_lufs: float = TARGET_LOUDNESS_LUFS,
) -> torch.Tensor:
    """Normalize integrated loudness to target LUFS (ITU-R BS.1770-4 via pyloudnorm)."""
    audio = to_mono_numpy(waveform).astype(np.float64)
    if audio.size == 0 or np.max(np.abs(audio)) == 0:
        return waveform.type(torch.float)

    min_samples = int(sample_rate * 0.4)  # pyloudnorm default block size
    if audio.size < min_samples:
        return waveform.type(torch.float)

    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)
    if np.isinf(loudness):
        return waveform.type(torch.float)

    normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
    return torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)


def pad_and_loudness_normalize(
    waveforms: list[torch.Tensor],
    target_lufs: float = TARGET_LOUDNESS_LUFS,
) -> list[torch.Tensor]:
    """Loudness-normalize each stem, then zero-pad to equal length."""
    normalized = [loudness_normalize(w, target_lufs=target_lufs) for w in waveforms]
    max_length = max(w.shape[-1] for w in normalized)
    padded = []
    for waveform in normalized:
        padded.append(torch.nn.functional.pad(
            waveform,
            pad=(0, max_length - waveform.shape[-1]),
            mode="constant",
            value=0,
        ))
    return padded


def build_mixture(
    waveforms: list[torch.Tensor],
    peak_limit: float = MIXTURE_PEAK_LIMIT,
) -> torch.Tensor:
    """Sum loudness-normalized stems and apply uniform anti-clip gain (Slakh-style)."""
    if not waveforms:
        raise ValueError("need at least one stem to build a mixture")
    summed = torch.stack(waveforms).sum(dim=0)
    audio = to_mono_numpy(summed).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > peak_limit:
        audio = audio * (peak_limit / peak)
    return torch.from_numpy(audio).unsqueeze(0)


def load_stem_flac(path: Path) -> torch.Tensor:
    audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return torch.from_numpy(audio).unsqueeze(0)


def save_stem_flac(waveform: torch.Tensor, song_dir: Path, track: int) -> Path:
    """Write a single mono stem waveform as FLAC."""
    song_dir.mkdir(parents=True, exist_ok=True)
    output_path = song_dir / STEM_FILE_PATTERN.format(track=track)
    audio = to_mono_numpy(waveform)
    sf.write(str(output_path), audio, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return output_path


def save_mixture_flac(waveform: torch.Tensor, song_dir: Path) -> Path:
    """Write mixture.flac for a song directory."""
    song_dir.mkdir(parents=True, exist_ok=True)
    output_path = song_dir / MIXTURE_FILE_NAME
    audio = to_mono_numpy(waveform)
    sf.write(str(output_path), audio, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return output_path


def write_mixture_from_waveforms(waveforms: list[torch.Tensor], song_dir: Path) -> Path:
    return save_mixture_flac(build_mixture(waveforms), song_dir)


def write_mixture_from_song_dir(song_dir: Path, track_indices: list[int]) -> Path | None:
    """Build mixture.flac from existing stem FLACs. Returns None if any stem is missing."""
    stem_paths = [stem_flac_path(song_dir, track) for track in track_indices]
    if not all(path.exists() for path in stem_paths):
        return None
    waveforms = [load_stem_flac(path) for path in stem_paths]
    return write_mixture_from_waveforms(waveforms, song_dir)


def stem_flac_path(song_dir: Path, track: int) -> Path:
    return song_dir / STEM_FILE_PATTERN.format(track=track)


def mixture_flac_path(song_dir: Path) -> Path:
    return song_dir / MIXTURE_FILE_NAME


def song_is_complete(song_dir: Path, n_tracks: int) -> bool:
    """Return True if all expected stem FLAC files and mixture.flac exist."""
    if not song_dir.is_dir():
        return False
    stems_ok = all(stem_flac_path(song_dir, j).exists() for j in range(n_tracks))
    return stems_ok and mixture_flac_path(song_dir).exists()
