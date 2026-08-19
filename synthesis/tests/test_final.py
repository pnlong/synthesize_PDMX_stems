"""CLI tests for hybrid final synthesis."""

from pathlib import Path

import pandas as pd
import pytest

from shared.config import DATA_DIR_NAME, STEMS_FILE_NAME
from synthesis.final import (
    FINAL_CONDITION,
    hybrid_dirs,
    parse_args,
    pass_sequence,
)
from synthesis.paths import ablation_raw_dir, spdmx_dataset_dir
from synthesis.recipe import (
    DEFAULT_RECIPE_PATH,
    STEM_RECIPE_FILE_NAME,
    CategoryRecipe,
    CategorySpec,
)
from synthesis.synthesize import run_layout_pass


def test_parse_args_requires_only_pass():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_layout_defaults_full_flac():
    args = parse_args(["--only-pass", "layout"])
    assert args.full is True
    assert args.flac is True
    assert args.recipe == str(DEFAULT_RECIPE_PATH)
    assert args.only_pass == "layout"
    assert args.yes is False


def test_parse_args_mp3_and_only_pass():
    args = parse_args(["--ablation-sample", "--only-pass", "fluidsynth", "--mp3", "-j", "4", "-y"])
    assert args.full is False
    assert args.flac is False
    assert args.only_pass == "fluidsynth"
    assert args.jobs == 4
    assert args.yes is True


def test_hybrid_dirs_write_one_tree():
    full = parse_args(["-o", "/tmp/spdmx", "--only-pass", "layout"])
    sample = parse_args(["-o", "/tmp/spdmx", "--ablation-sample", "--only-pass", "layout"])
    assert hybrid_dirs(full) == (
        spdmx_dataset_dir("/tmp/spdmx"),
        spdmx_dataset_dir("/tmp/spdmx"),
    )
    dest = ablation_raw_dir("/tmp/spdmx", FINAL_CONDITION)
    assert hybrid_dirs(sample) == (dest, dest)


def test_pass_sequence_starts_with_layout():
    no_realify = CategoryRecipe(
        specs={"piano": CategorySpec("basic", False, "basic", "basic")},
    )
    with_realify = CategoryRecipe(
        specs={"piano": CategorySpec("basic", True, "basic", "basic_realify")},
    )
    with_ddsp = CategoryRecipe(
        specs={"strings": CategorySpec("midi-ddsp", False, "basic", "ddsp_basic")},
    )
    assert pass_sequence(no_realify) == ("layout", "fluidsynth", "mix")
    assert pass_sequence(with_ddsp) == ("layout", "fluidsynth", "ddsp", "mix")
    assert pass_sequence(with_realify) == ("layout", "fluidsynth", "realify", "mix")


def test_layout_pass_creates_song_dirs(tmp_path: Path):
    pdmx_root = tmp_path / "PDMX"
    pdmx_root.mkdir()
    csv_path = pdmx_root / "PDMX.csv"
    pd.DataFrame({
        "path": ["./data/1/11/QmTestSong.json"],
        "mid": ["./mid/1/11/QmTestSong.mid"],
        "subset:all_valid": [True],
        "n_tracks": [2],
    }).to_csv(csv_path, index=False)

    out = tmp_path / "SPDMX"
    args = parse_args([
        "--only-pass", "layout",
        "-o", str(out),
        "-df", str(csv_path),
        "--no-register",
    ])
    args.recipe = CategoryRecipe(
        specs={"piano": CategorySpec("basic", False, "basic", "basic")},
    )
    dest = spdmx_dataset_dir(str(out))
    dataset = run_layout_pass(args, dest)
    song_dir = Path(dataset.iloc[0]["path_output"])
    assert song_dir.is_dir()
    assert song_dir == Path(dest) / "audio" / "1" / "11" / "QmTestSong"
    assert (Path(dest) / "mid" / "1" / "11").is_dir()
    assert (Path(dest) / f"{DATA_DIR_NAME}.csv").is_file()
    assert (Path(dest) / f"{STEMS_FILE_NAME}.csv").is_file()
    assert (Path(dest) / STEM_RECIPE_FILE_NAME).is_file()
