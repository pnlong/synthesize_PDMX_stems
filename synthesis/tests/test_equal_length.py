"""Assert all stems in a song directory share the same sample length."""

from pathlib import Path

import soundfile as sf


def flac_sample_counts(song_dir: Path) -> list[int]:
    stems = sorted(song_dir.glob("stem_*.flac"))
    counts = []
    for stem in stems:
        info = sf.info(str(stem))
        counts.append(info.frames)
    return counts


def test_equal_length_flacs(tmp_path: Path):
    import numpy as np

    song_dir = tmp_path / "song"
    song_dir.mkdir()
    sr = 44100
    for i, n_samples in enumerate([sr, sr * 2, sr]):
        audio = np.zeros(n_samples, dtype=np.float32)
        sf.write(str(song_dir / f"stem_{i}.flac"), audio, sr, format="FLAC")

    # Simulate padding to max length (2s)
    max_samples = sr * 2
    for i in range(3):
        data, _ = sf.read(str(song_dir / f"stem_{i}.flac"))
        if len(data) < max_samples:
            pad = np.zeros(max_samples - len(data), dtype=np.float32)
            data = np.concatenate([data, pad])
        sf.write(str(song_dir / f"stem_{i}.flac"), data, sr, format="FLAC")

    counts = flac_sample_counts(song_dir)
    assert len(counts) == 3
    assert len(set(counts)) == 1
