"""Output paths for model listening test."""

from __future__ import annotations

from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = MODULE_DIR / "output"
DEFAULT_CLIPS_DIR = DEFAULT_OUTPUT_DIR / "clips"
DEFAULT_MANIFEST = MODULE_DIR / "trial_manifest.yaml"
DEFAULT_RESPONSES_DIR = DEFAULT_OUTPUT_DIR / "responses"
