"""Output path helpers."""

from __future__ import annotations

from shared.config import (
    ABLATIONS_DIR_NAME,
    ANALYSIS_DIR_NAME,
    DEV_DIR_NAME,
    SONG_LENGTHS_DIR_NAME,
    SPDMX_DATASET_DIR_NAME,
    STEMS_DIR_NAME,
    STEMS_REALIFY_DIR_NAME,
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


def song_lengths_dir(output_dir: str) -> str:
    return f"{analysis_root(output_dir)}/{SONG_LENGTHS_DIR_NAME}"


def spdmx_dataset_dir(output_dir: str) -> str:
    return f"{output_dir}/{SPDMX_DATASET_DIR_NAME}"
