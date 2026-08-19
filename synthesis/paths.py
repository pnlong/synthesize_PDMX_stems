"""Output path helpers."""

from __future__ import annotations

from pathlib import Path

from shared.config import (
    ABLATIONS_DIR_NAME,
    ANALYSIS_DIR_NAME,
    DEV_DIR_NAME,
    INSTRUMENTS_DIR_NAME,
    MID_CORRECTED_DIR_NAME,
    SONG_LENGTHS_DIR_NAME,
    SPDMX_AUDIO_DIR_NAME,
    SPDMX_DATASET_DIR_NAME,
    SPDMX_MID_DIR_NAME,
    STEMS_DIR_NAME,
    STEMS_REALIFY_DIR_NAME,
    TRACK_NAMES_DIR_NAME,
)


def condition_name(render_mode: str, realify: bool = False) -> str:
    return f"{render_mode}_realify" if realify else render_mode


def dev_root(output_dir: str) -> str:
    return f"{output_dir}/{DEV_DIR_NAME}"


def ablations_root(output_dir: str) -> str:
    return f"{dev_root(output_dir)}/{ABLATIONS_DIR_NAME}"


def ablation_dir(output_dir: str, condition: str) -> str:
    return f"{ablations_root(output_dir)}/{condition}"


def ablation_raw_dir(output_dir: str, render_mode: str) -> str:
    return ablation_dir(output_dir, render_mode)


def ablation_realify_dir(output_dir: str, render_mode: str) -> str:
    return ablation_dir(output_dir, condition_name(render_mode, realify=True))


def full_stems_dir(output_dir: str) -> str:
    return f"{dev_root(output_dir)}/{STEMS_DIR_NAME}"


def full_stems_realify_dir(output_dir: str) -> str:
    return f"{dev_root(output_dir)}/{STEMS_REALIFY_DIR_NAME}"


def analysis_root(output_dir: str) -> str:
    return f"{dev_root(output_dir)}/{ANALYSIS_DIR_NAME}"


def mid_corrected_dir(output_dir: str) -> str:
    return f"{dev_root(output_dir)}/{MID_CORRECTED_DIR_NAME}"


def song_lengths_dir(output_dir: str) -> str:
    return f"{analysis_root(output_dir)}/{SONG_LENGTHS_DIR_NAME}"


def instruments_dir(output_dir: str) -> str:
    return f"{analysis_root(output_dir)}/{INSTRUMENTS_DIR_NAME}"


def track_names_dir(output_dir: str) -> str:
    return f"{analysis_root(output_dir)}/{TRACK_NAMES_DIR_NAME}"


def spdmx_dataset_dir(output_dir: str) -> str:
    return f"{output_dir}/{SPDMX_DATASET_DIR_NAME}"


def spdmx_audio_dir(output_dir: str) -> str:
    return f"{spdmx_dataset_dir(output_dir)}/{SPDMX_AUDIO_DIR_NAME}"


def spdmx_mid_dir(output_dir: str) -> str:
    return f"{spdmx_dataset_dir(output_dir)}/{SPDMX_MID_DIR_NAME}"


def remap_path_prefix(value: str, source_dir: Path, output_dir: Path) -> str:
    """Rewrite an absolute path under ``source_dir`` to the mirrored path under ``output_dir``."""
    source_prefix = str(source_dir)
    if value.startswith(source_prefix):
        return str(output_dir) + value[len(source_prefix):]
    return value


def resolve_output_song_dir(song_dir: Path, source_dir: Path, output_dir: Path) -> Path:
    """Map a song directory under ``source_dir`` to the mirrored path under ``output_dir``."""
    if output_dir == source_dir:
        return song_dir
    remapped = remap_path_prefix(str(song_dir), source_dir, output_dir)
    if remapped == str(song_dir):
        raise ValueError(f"Song path {song_dir} is not under source dir {source_dir}")
    return Path(remapped)
