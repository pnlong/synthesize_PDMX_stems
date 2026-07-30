"""Tests for probe listening category resolution."""

from synthesis.patches import resolve_probe_category


def test_resolve_probe_category_guitar_and_brass_by_gm_class():
    assert resolve_probe_category(program=24, is_drum=False) == "guitar"
    assert resolve_probe_category(program=59, is_drum=False) == "brass"


def test_resolve_probe_category_wind_excludes_brass():
    assert resolve_probe_category(program=73, is_drum=False) == "wind"
    assert resolve_probe_category(program=65, is_drum=False) == "wind"


def test_resolve_probe_category_polyphonic_synth_and_ethnic():
    assert resolve_probe_category(program=83, is_drum=False) == "polyphonic"
    assert resolve_probe_category(program=105, is_drum=False, track_name="Banjo") == "polyphonic"


def test_resolve_probe_category_trombone_name_is_brass_not_voice():
    assert (
        resolve_probe_category(program=57, is_drum=False, track_name="Tenor")
        == "brass"
    )
