"""Tests for synthesis FX profiles."""

from synthesis.fx import fluidsynth_fx_args


def test_fluidsynth_fx_args_dry_disables_reverb_and_chorus():
    args = fluidsynth_fx_args("dry")
    assert "-R" in args and "0" in args
    assert "-C" in args and "0" in args


def test_fluidsynth_fx_args_light_sets_reverb_level():
    args = fluidsynth_fx_args("light")
    assert any("synth.reverb.level=0.2" in arg for arg in args)
