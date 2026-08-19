"""CLI tests for hybrid final synthesis."""

from synthesis.final import FINAL_CONDITION, hybrid_dirs, parse_args
from synthesis.paths import ablation_raw_dir, full_stems_dir
from synthesis.recipe import DEFAULT_RECIPE_PATH


def test_parse_args_defaults_full():
    args = parse_args([])
    assert args.full is True
    assert args.recipe == str(DEFAULT_RECIPE_PATH)
    assert args.only_pass is None
    assert args.yes is False
    assert not hasattr(args, "render_mode") or getattr(args, "render_mode", None) is None


def test_parse_args_ablation_sample_and_only_pass():
    args = parse_args(["--ablation-sample", "--only-pass", "fluidsynth", "-j", "4", "-y"])
    assert args.full is False
    assert args.only_pass == "fluidsynth"
    assert args.jobs == 4
    assert args.yes is True


def test_hybrid_dirs_full_vs_sample():
    full = parse_args(["-o", "/tmp/spdmx"])
    sample = parse_args(["-o", "/tmp/spdmx", "--ablation-sample"])
    assert hybrid_dirs(full)[0] == full_stems_dir("/tmp/spdmx")
    assert hybrid_dirs(sample)[0] == ablation_raw_dir("/tmp/spdmx", FINAL_CONDITION)
    assert hybrid_dirs(sample)[1].endswith("/final_realify")
