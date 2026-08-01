"""Tests for donor stem copy helpers."""

from pathlib import Path

import pytest

from shared.config import RENDER_MODE_DDSP_BASIC, RENDER_MODE_DDSP_SLAKH
from synthesis.reuse import (
    copy_stem,
    fallback_donor_mode,
    is_reused_source,
    reused_source_label,
    song_rel_under_data,
    uses_ddsp,
    uses_slakh_recipes,
)


def test_uses_ddsp_and_slakh_recipes():
    assert uses_ddsp(RENDER_MODE_DDSP_BASIC)
    assert uses_ddsp(RENDER_MODE_DDSP_SLAKH)
    assert not uses_ddsp("basic")
    assert uses_slakh_recipes("slakh")
    assert uses_slakh_recipes(RENDER_MODE_DDSP_SLAKH)
    assert not uses_slakh_recipes(RENDER_MODE_DDSP_BASIC)


def test_fallback_donor_mode():
    assert fallback_donor_mode(RENDER_MODE_DDSP_BASIC) == "basic"
    assert fallback_donor_mode(RENDER_MODE_DDSP_SLAKH) == "slakh"
    assert fallback_donor_mode("basic") is None


def test_reused_source_label():
    assert reused_source_label("basic") == "reused:basic"
    assert is_reused_source("reused:slakh")
    assert not is_reused_source("rendered")


def test_copy_stem(tmp_path: Path):
    src = tmp_path / "a" / "stem_0.mp3"
    dst = tmp_path / "b" / "stem_0.mp3"
    src.parent.mkdir()
    src.write_bytes(b"audio")
    copy_stem(src, dst)
    assert dst.read_bytes() == b"audio"


def test_song_rel_under_data(tmp_path: Path):
    abl = tmp_path / "basic"
    song = abl / "data" / "7" / "19" / "QmTest"
    song.mkdir(parents=True)
    assert song_rel_under_data(song, abl) == "7/19/QmTest"


def test_copy_stem_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        copy_stem(tmp_path / "missing.mp3", tmp_path / "out.mp3")


def test_ddsp_routing_has_original_path_column():
    from synthesis.ddsp.config import DDSP_ROUTING_COLUMNS

    assert "original_path" in DDSP_ROUTING_COLUMNS
    assert "source" in DDSP_ROUTING_COLUMNS
