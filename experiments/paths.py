"""Output path helpers for experiments."""

from __future__ import annotations

from shared.config import DEV_DIR_NAME, EXPERIMENTS_DIR_NAME, PRESET_SWEEP_DIR_NAME
from synthesis.paths import dev_root


def experiments_root(output_dir: str) -> str:
    return f"{dev_root(output_dir)}/{EXPERIMENTS_DIR_NAME}"


def preset_sweep_output_root(output_dir: str) -> str:
    return f"{experiments_root(output_dir)}/{PRESET_SWEEP_DIR_NAME}"
