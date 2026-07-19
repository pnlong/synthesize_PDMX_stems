"""Tests for phase-4 diverse verification render."""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from experiments.preset_sweep.clips_dir import CLIPS_SOURCE_MARKER, resolve_sweep_clips_dir
from experiments.preset_sweep.config import LOCKED_VERIFY_VARIANT, PHASE1B, PHASE4, phase_output_dir
from experiments.preset_sweep.diverse_stems import write_diverse_clip_dataset
from experiments.preset_sweep.sweep import prepare_phase4_verify
from shared.config import DATA_DIR_NAME


def _write_source_and_phase1b(
    tmp_path: Path,
) -> tuple[Path, Path, list[dict]]:
    source_dir = tmp_path / "basic"
    song_id = "0/1/QmTest"
    song_dir = source_dir / DATA_DIR_NAME / song_id
    song_dir.mkdir(parents=True)
    sr = 44100
    sf.write(
        str(song_dir / "stem_0.flac"),
        np.full(sr * 12, 0.2, dtype=np.float32),
        sr,
        format="FLAC",
    )
    pd.DataFrame([{"path": str(song_dir), "title": song_id, "genres": "classical"}]).to_csv(
        source_dir / f"{DATA_DIR_NAME}.csv",
        index=False,
    )
    pd.DataFrame([{
        "path": str(song_dir),
        "track": 0,
        "program": 0,
        "is_drum": False,
        "name": "Piano",
        "has_lyrics": False,
    }]).to_csv(source_dir / "stems.csv", index=False)

    sweep_root = tmp_path / "output"
    phase1b_dir = phase_output_dir(sweep_root, PHASE1B)
    stems = [{
        "id": "piano_clip",
        "category": "piano",
        "song_id": song_id,
        "track": 0,
        "audio_format": "flac",
    }]
    write_diverse_clip_dataset(
        source_dir,
        stems,
        phase1b_dir / "clips",
        clip_seconds=10,
    )
    return source_dir, phase1b_dir, stems


def test_prepare_phase4_verify_reuses_phase1b_assets(tmp_path: Path):
    source_dir, phase1b_dir, _stems = _write_source_and_phase1b(tmp_path)
    sweep_root = tmp_path / "output"
    phase4_dir = phase_output_dir(sweep_root, PHASE4)

    source_clips, probe_stems, variants = prepare_phase4_verify(
        sweep_root=sweep_root,
        output_dir=phase4_dir,
        source_dir=source_dir,
        grid_cfg={"replace_silent_clips": True},
        sample_seed=1,
    )

    assert source_clips == phase1b_dir / "clips"
    assert resolve_sweep_clips_dir(phase4_dir) == (phase1b_dir / "clips").resolve()
    assert (phase4_dir / "clips").is_symlink() or (phase4_dir / CLIPS_SOURCE_MARKER).is_file()
    assert len(probe_stems) == 1
    assert variants == [{"id": LOCKED_VERIFY_VARIANT}]
