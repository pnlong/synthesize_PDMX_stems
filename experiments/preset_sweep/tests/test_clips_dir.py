"""Tests for sweep clip directory resolution."""

from pathlib import Path

from experiments.preset_sweep.clips_dir import (
    CLIPS_SOURCE_MARKER,
    clips_dataset_ready,
    expose_clips_dir,
    resolve_sweep_clips_dir,
)
from shared.config import DATA_DIR_NAME, STEMS_FILE_NAME


def _write_clip_dataset(clips_dir: Path) -> None:
    clips_dir.mkdir(parents=True, exist_ok=True)
    (clips_dir / f"{DATA_DIR_NAME}.csv").write_text("path\n")
    (clips_dir / f"{STEMS_FILE_NAME}.csv").write_text("path,track\n")


def test_resolve_sweep_clips_dir_from_symlink(tmp_path: Path):
    source = tmp_path / "phase1b" / "clips"
    _write_clip_dataset(source)
    phase_dir = tmp_path / "phase4"
    phase_dir.mkdir()
    (phase_dir / "clips").symlink_to(source, target_is_directory=True)

    assert resolve_sweep_clips_dir(phase_dir) == source.resolve()


def test_resolve_sweep_clips_dir_from_marker(tmp_path: Path):
    source = tmp_path / "phase1b" / "clips"
    _write_clip_dataset(source)
    phase_dir = tmp_path / "phase4"
    phase_dir.mkdir()
    (phase_dir / CLIPS_SOURCE_MARKER).write_text("../phase1b/clips\n")

    assert resolve_sweep_clips_dir(phase_dir) == source.resolve()


def test_expose_clips_dir_writes_marker_when_symlink_unusable(tmp_path: Path, monkeypatch):
    source = tmp_path / "phase1b" / "clips"
    _write_clip_dataset(source)
    phase_dir = tmp_path / "phase4"

    def _broken_symlink(relative_target, link_path, target_is_directory=True):
        Path(link_path).symlink_to("missing-target")

    monkeypatch.setattr("experiments.preset_sweep.clips_dir.os.symlink", _broken_symlink)

    returned = expose_clips_dir(output_dir=phase_dir, source_clips=source)

    assert returned == source.resolve()
    assert not clips_dataset_ready(phase_dir / "clips")
    assert resolve_sweep_clips_dir(phase_dir) == source.resolve()
