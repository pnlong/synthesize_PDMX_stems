"""Tests for archive soundfont download helpers."""

from experiments.patch_sweep.download_archive_soundfonts import slugify, tag_soundfont


def test_slugify():
    assert slugify("merlin_grand(v5.37).sf2") == "merlin_grand_v5_37"


def test_tag_soundfont_piano():
    tags = tag_soundfont("SGM-v2.01-NicePianosGuitarsBass-V1.2.sf2")
    assert "piano" in tags


def test_tag_soundfont_general_fallback():
    assert tag_soundfont("CT2MGM.SF2") == ["general"]
