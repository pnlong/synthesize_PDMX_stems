"""Output paths for ablation listening test."""

from __future__ import annotations

from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
DEFAULT_OUTPUT_DIR = MODULE_DIR / "output"
DEFAULT_CLIPS_DIR = DEFAULT_OUTPUT_DIR / "clips"
DEFAULT_MANIFEST = MODULE_DIR / "trial_manifest.yaml"
DEFAULT_RESPONSES_DIR = DEFAULT_OUTPUT_DIR / "responses"

DEFAULT_WEBMUSHRA_ROOT = REPO_ROOT / "third_party" / "webMUSHRA"
WEBMUSHRA_CONFIG_NAME = "spdmx_ablation.yaml"
WEBMUSHRA_STIMULI_DIR = "stimuli/spdmx_ablation"
WEBMUSHRA_TEST_ID = "spdmx_ablation"
