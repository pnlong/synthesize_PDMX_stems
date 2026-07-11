"""Tests for synthesize CLI orchestration."""

from pathlib import Path

import pandas as pd
import pytest

from shared.config import DATA_DIR_NAME, DEFAULT_AUDIO_FORMAT, STEMS_FILE_NAME
from synthesis.synthesize import (
    require_raw_synthesis,
    reset_synthesis_output,
    synthesis_is_complete,
)


def _write_complete_ablation(ablation_dir: Path, n_tracks: int = 1):
    ablation_dir.mkdir(parents=True)
    song_dir = ablation_dir / "data" / "song"
    song_dir.mkdir(parents=True)
    import numpy as np
    import soundfile as sf

    sr = 44100
    for j in range(n_tracks):
        sf.write(str(song_dir / f"stem_{j}.flac"), np.zeros(sr), sr, format="FLAC")
    sf.write(str(song_dir / "mixture.flac"), np.zeros(sr), sr, format="FLAC")

    pd.DataFrame({"path": [str(song_dir)], "n_tracks": [n_tracks]}).to_csv(
        ablation_dir / f"{DATA_DIR_NAME}.csv", index=False
    )
    pd.DataFrame({
        "path": [str(song_dir)] * n_tracks,
        "track": list(range(n_tracks)),
    }).to_csv(ablation_dir / f"{STEMS_FILE_NAME}.csv", index=False)


def test_synthesis_is_complete_false_when_stems_missing(tmp_path: Path):
    ablation_dir = tmp_path / "basic"
    ablation_dir.mkdir()
    song_dir = ablation_dir / "data" / "song"
    song_dir.mkdir(parents=True)
    pd.DataFrame({"path": [str(song_dir)], "n_tracks": [1]}).to_csv(
        ablation_dir / f"{DATA_DIR_NAME}.csv", index=False
    )
    pd.DataFrame({"path": [str(song_dir)], "track": [0]}).to_csv(
        ablation_dir / f"{STEMS_FILE_NAME}.csv", index=False
    )
    assert not synthesis_is_complete(str(ablation_dir), DEFAULT_AUDIO_FORMAT)


def test_require_raw_synthesis_raises_with_command(tmp_path: Path):
    ablation_dir = tmp_path / "basic"
    ablation_dir.mkdir()
    with pytest.raises(RuntimeError, match="Run the corresponding non-realify ablation first"):
        require_raw_synthesis(
            str(ablation_dir),
            run_command="uv run python -m synthesis.synthesize --render-mode basic",
        )


def test_require_raw_synthesis_passes_when_complete(tmp_path: Path):
    ablation_dir = tmp_path / "basic"
    _write_complete_ablation(ablation_dir)
    require_raw_synthesis(
        str(ablation_dir),
        run_command="uv run python -m synthesis.synthesize --render-mode basic",
    )


def test_reset_synthesis_output_removes_stems_and_tables(tmp_path: Path):
    ablation_dir = tmp_path / "basic"
    _write_complete_ablation(ablation_dir)
    assert (ablation_dir / "data.csv").exists()
    assert (ablation_dir / "data" / "song" / "stem_0.flac").exists()

    reset_synthesis_output(str(ablation_dir))

    assert ablation_dir.is_dir()
    assert not (ablation_dir / "data.csv").exists()
    assert not (ablation_dir / "data" / "song" / "stem_0.flac").exists()
