"""Validate synthesized FLAC stem directories."""

from pathlib import Path

import pandas as pd
import pytest
import soundfile as sf

from shared.config import NA_STRING, STEMS_FILE_NAME
from synthesis.audio import stem_flac_path


OUTPUT_COLUMNS = ["path", "valid", "complete", "n_missing_stems"]


def validate_song_dir(path: str, expected_stems: int) -> dict:
    song_dir = Path(path)
    try:
        valid = song_dir.is_dir()
        actual = 0
        if valid:
            for j in range(expected_stems):
                flac = stem_flac_path(song_dir, j)
                if flac.exists():
                    sf.info(str(flac))
                    actual += 1
        complete = valid and actual == expected_stems
    except Exception:
        valid = False
        complete = False
        actual = 0
    return {
        "path": path,
        "valid": valid,
        "complete": complete,
        "n_missing_stems": expected_stems - actual,
    }


def test_validate_song_dir_complete(tmp_path: Path):
    song_dir = tmp_path / "song_a"
    song_dir.mkdir()
    import numpy as np
    sr = 44100
    for j in range(2):
        sf.write(str(song_dir / f"stem_{j}.flac"), np.zeros((sr, 2)), sr, format="FLAC")

    result = validate_song_dir(str(song_dir), expected_stems=2)
    assert result["valid"]
    assert result["complete"]
    assert result["n_missing_stems"] == 0


def test_validate_song_dir_missing_stem(tmp_path: Path):
    song_dir = tmp_path / "song_b"
    song_dir.mkdir()
    result = validate_song_dir(str(song_dir), expected_stems=3)
    assert result["valid"]
    assert not result["complete"]
    assert result["n_missing_stems"] == 3


@pytest.fixture
def stems_csv_fixture(tmp_path: Path):
    song_dir = tmp_path / "data" / "song"
    song_dir.mkdir(parents=True)
    import numpy as np
    sr = 44100
    sf.write(str(song_dir / "stem_0.flac"), np.zeros(sr), sr, format="FLAC")
    sf.write(str(song_dir / "stem_1.flac"), np.zeros(sr), sr, format="FLAC")

    stems_path = tmp_path / f"{STEMS_FILE_NAME}.csv"
    pd.DataFrame({
        "path": [str(song_dir), str(song_dir)],
        "track": [0, 1],
    }).to_csv(stems_path, index=False)

    return tmp_path, str(song_dir)


def test_stems_csv_integration(stems_csv_fixture):
    dataset_dir, song_dir = stems_csv_fixture
    stems = pd.read_csv(f"{dataset_dir}/{STEMS_FILE_NAME}.csv")
    grouped = stems.groupby("path").size()
    result = validate_song_dir(song_dir, int(grouped[song_dir]))
    assert result["complete"]
