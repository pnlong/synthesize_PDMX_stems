"""Tests for offline content fidelity scoring."""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from experiments.preset_sweep.score_content_fidelity import score_manifest


def test_score_manifest_writes_fidelity_columns(tmp_path):
    sr = 44100
    reference = tmp_path / "piano_test.flac"
    realified = tmp_path / "out.flac"
    times = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    ref_wave = np.zeros(sr * 2, dtype=np.float32)
    real_wave = ref_wave.copy()
    burst = int(0.05 * sr)
    for time_sec in times:
        start = int(time_sec * sr)
        end = min(start + burst, ref_wave.size)
        t = np.arange(end - start, dtype=np.float32) / sr
        ref_wave[start:end] = 0.8 * np.sin(2 * np.pi * 1000.0 * t)
        real_wave[start:end] = ref_wave[start:end]
    sf.write(reference, ref_wave, sr, format="FLAC")
    sf.write(realified, real_wave, sr, format="FLAC")

    manifest = pd.DataFrame([{
        "stem_id": "piano_test",
        "category": "piano",
        "variant_id": "noise0.45",
        "init_noise_level": 0.45,
        "prompt_variant": "current",
        "path": str(tmp_path),
        "track": 0,
        "out_path": str(realified),
    }])

    scores = score_manifest(
        manifest,
        sweep_dir=tmp_path,
        reference_dir=tmp_path,
        threshold=0.85,
    )
    assert len(scores) == 1
    assert scores.iloc[0]["fidelity_score"] >= 0.85
    assert scores.iloc[0]["passed"]
