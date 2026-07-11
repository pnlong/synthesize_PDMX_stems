"""Smoke test: fluidsynth can render the tiny MIDI fixture."""

import shutil
import subprocess
from pathlib import Path

import pytest

from shared.config import SOUNDFONT_PATH


def _fluidsynth_available() -> bool:
    return shutil.which("fluidsynth") is not None


def _soundfont_available() -> bool:
    return Path(SOUNDFONT_PATH).exists()


@pytest.mark.skipif(not _fluidsynth_available(), reason="fluidsynth not installed")
@pytest.mark.skipif(not _soundfont_available(), reason="default soundfont not found")
def test_soundfont_renders_tiny_mid(tiny_mid_path: Path, tmp_path: Path):
    output_wav = tmp_path / "out.wav"
    subprocess.run(
        [
            "fluidsynth", "-ni", "-F", str(output_wav), "-T", "wav",
            "-r", "44100", "-g", "1.0", SOUNDFONT_PATH, str(tiny_mid_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert output_wav.exists()
    assert output_wav.stat().st_size > 44
