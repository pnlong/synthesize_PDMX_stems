"""Paths and knobs for the isolated TF/DDSP neural-synthesis tier."""

from __future__ import annotations

import os
from pathlib import Path

from shared.config import OUTPUT_DIR, SAMPLE_RATE

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DDSP_DIR = Path(__file__).resolve().parent

# Isolated TensorFlow venv (not the root uv .venv — avoids TF/Torch CUDA clashes).
DDSP_VENV_DIR = Path(
    os.environ.get("SPDMX_DDSP_VENV", str(_REPO_ROOT / ".venv-ddsp"))
)
DDSP_PYTHON = DDSP_VENV_DIR / "bin" / "python"

# MIDI-DDSP pretrained URMP weights.
# After `midi_ddsp_download_model_weights`, weights also live inside the package.
# Override with MIDI_DDSP_WEIGHTS_DIR pointing at a folder that contains
# synthesis_generator/50000 and expression_generator/5000 (or the nested
# midi_ddsp_model_weights_urmp_9_10/ layout from the upstream zip).
MIDI_DDSP_WEIGHTS_DIR = Path(
    os.environ.get(
        "MIDI_DDSP_WEIGHTS_DIR",
        str(_REPO_ROOT / "models" / "midi_ddsp"),
    )
)
# Manual zip (fallback): https://github.com/magenta/midi-ddsp/raw/models/midi_ddsp_model_weights_urmp_9_10.zip
MIDI_DDSP_WEIGHTS_URL = (
    "https://github.com/magenta/midi-ddsp/raw/models/midi_ddsp_model_weights_urmp_9_10.zip"
)
MIDI_DDSP_SAMPLE_RATE = 16000

# DDSP-Piano (lrenault/ddsp-piano) checkout + checkpoint.
DDSP_PIANO_ROOT = Path(
    os.environ.get(
        "DDSP_PIANO_ROOT",
        str(_DDSP_DIR / "third_party" / "ddsp-piano"),
    )
)
# Paper model (DAFx / JAES): dafx22 @ 16 kHz. Override via env if needed.
DDSP_PIANO_CKPT = Path(
    os.environ.get(
        "DDSP_PIANO_CKPT",
        str(DDSP_PIANO_ROOT / "ddsp_piano" / "model_weights" / "dafx22"),
    )
)
DDSP_PIANO_GIN = Path(
    os.environ.get(
        "DDSP_PIANO_GIN",
        str(DDSP_PIANO_ROOT / "ddsp_piano" / "configs" / "dafx22.gin"),
    )
)
# MAESTRO competition year index 0–9 (Disklavier edition).
DDSP_PIANO_TYPE = int(os.environ.get("DDSP_PIANO_TYPE", "0"))
# Paper train @ 16 kHz; verify at install. Always resample to SAMPLE_RATE in wrappers.
DDSP_PIANO_SAMPLE_RATE = int(os.environ.get("DDSP_PIANO_SAMPLE_RATE", "16000"))

PIPELINE_SAMPLE_RATE = SAMPLE_RATE

# Routing / coverage CSV written beside stems during B3 runs.
DDSP_ROUTING_FILE_NAME = "ddsp_routing.csv"
DDSP_ROUTING_COLUMNS = [
    "path",
    "track",
    "program",
    "is_drum",
    "name",
    "backend",
    "instrument_key",
    "reason",
    "n_notes",
    "source",
    # Absolute stem filepath this row was copied from; NA when newly rendered.
    "original_path",
]
