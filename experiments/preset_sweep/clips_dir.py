"""Resolve sweep clip directories (symlink or clips.source marker)."""

from __future__ import annotations

import os
from pathlib import Path

from shared.config import DATA_DIR_NAME, STEMS_FILE_NAME

CLIPS_SOURCE_MARKER = "clips.source"


def clips_dataset_ready(clips_dir: Path) -> bool:
    return (
        clips_dir.is_dir()
        and (clips_dir / f"{DATA_DIR_NAME}.csv").is_file()
        and (clips_dir / f"{STEMS_FILE_NAME}.csv").is_file()
    )


def resolve_sweep_clips_dir(sweep_dir: Path) -> Path | None:
    """Return the clip dataset directory for a sweep phase output tree."""
    sweep_dir = sweep_dir.resolve()
    clips_dir = sweep_dir / "clips"
    if clips_dataset_ready(clips_dir):
        return clips_dir.resolve()

    marker = sweep_dir / CLIPS_SOURCE_MARKER
    if marker.is_file():
        relative_target = marker.read_text().strip()
        if relative_target:
            target = (sweep_dir / relative_target).resolve()
            if clips_dataset_ready(target):
                return target
    return None


def expose_clips_dir(*, output_dir: Path, source_clips: Path) -> Path:
    """Expose source_clips under output_dir for listening; returns source_clips."""
    source_clips = source_clips.resolve()
    if not source_clips.is_dir():
        raise FileNotFoundError(f"Clip source is not a directory: {source_clips}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dest_clips = output_dir / "clips"
    marker = output_dir / CLIPS_SOURCE_MARKER

    if dest_clips.is_symlink() or (dest_clips.exists() and not clips_dataset_ready(dest_clips)):
        dest_clips.unlink(missing_ok=True)

    relative_target = os.path.relpath(source_clips, output_dir.resolve())

    if not dest_clips.exists():
        try:
            os.symlink(relative_target, dest_clips, target_is_directory=True)
        except OSError:
            pass

    if clips_dataset_ready(dest_clips):
        marker.unlink(missing_ok=True)
        return source_clips

    dest_clips.unlink(missing_ok=True)
    marker.write_text(f"{relative_target}\n")
    return source_clips
