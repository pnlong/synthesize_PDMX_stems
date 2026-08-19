"""Tests for the separate mixture + stem-normalization CLI."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from shared.config import SAMPLE_RATE
from synthesis.audio import load_stem, to_mono_numpy
from synthesis.mix import (
    confirm_overwrite,
    default_dest_dir,
    main,
    mix_command,
    normalize_stems_for_dataset,
    resolve_stems_dir,
)


def test_mix_command_includes_stems_dir_and_jobs():
    cmd = mix_command("/tmp/ablations/basic", jobs=8)
    assert cmd == "uv run python -m synthesis.mix --stems-dir /tmp/ablations/basic -j 8"


def test_mix_command_flac_flag():
    cmd = mix_command("/tmp/x", jobs=2, flac=True)
    assert "--flac" in cmd


def test_resolve_stems_dir_explicit(tmp_path: Path):
    assert resolve_stems_dir(stems_dir=str(tmp_path)) == tmp_path


def test_resolve_stems_dir_render_mode(tmp_path: Path):
    out = resolve_stems_dir(output_dir=str(tmp_path), render_mode="basic")
    assert out == tmp_path / "dev" / "ablations" / "basic"


def test_resolve_stems_dir_realify(tmp_path: Path):
    out = resolve_stems_dir(
        output_dir=str(tmp_path), render_mode="slakh", realify=True,
    )
    assert out == tmp_path / "dev" / "ablations" / "slakh_realify"


def test_default_dest_dir_sibling():
    assert default_dest_dir(Path("/a/basic")) == Path("/a/basic_summable")


def test_main_errors_on_missing_dir(tmp_path: Path):
    missing = tmp_path / "nope"
    with pytest.raises(SystemExit, match="Stem directory not found"):
        main(["--stems-dir", str(missing)])


def test_confirm_overwrite_yes(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert confirm_overwrite(Path("/tmp/x")) is True


def test_confirm_overwrite_no(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert confirm_overwrite(Path("/tmp/x")) is False


def test_main_aborts_without_confirm(tmp_path: Path, monkeypatch):
    stems_dir = tmp_path / "basic"
    song = stems_dir / "data" / "song"
    song.mkdir(parents=True)
    sf.write(str(song / "stem_0.flac"), np.zeros(100, np.float32), SAMPLE_RATE, format="FLAC")
    pd.DataFrame({"path": [str(song)], "track": [0]}).to_csv(stems_dir / "stems.csv", index=False)
    monkeypatch.setattr("synthesis.mix.confirm_overwrite", lambda _: False)
    with pytest.raises(SystemExit, match="Aborted"):
        main(["--stems-dir", str(stems_dir), "--flac"])


def _seed_song_tree(root: Path) -> Path:
    song = root / "data" / "song"
    song.mkdir(parents=True)
    sr = SAMPLE_RATE
    sf.write(str(song / "stem_0.flac"), np.full(sr, 0.9, np.float32), sr, format="FLAC")
    sf.write(str(song / "stem_1.flac"), np.full(sr, 0.9, np.float32), sr, format="FLAC")
    pd.DataFrame({
        "path": [str(song), str(song)],
        "track": [0, 1],
    }).to_csv(root / "stems.csv", index=False)
    pd.DataFrame({"path": [str(song)], "n_tracks": [2]}).to_csv(root / "data.csv", index=False)
    return song


def test_no_overwrite_writes_dest_and_mixture(tmp_path: Path):
    source = tmp_path / "basic"
    song = _seed_song_tree(source)
    dest = tmp_path / "basic_summable"
    main([
        "--stems-dir", str(source),
        "--no-overwrite",
        "--dest-dir", str(dest),
        "--write-mixture",
        "--no-velocity-dynamics",
        "--flac",
        "-j", "1",
    ])
    # Originals untouched (still loud).
    orig0 = load_stem(song / "stem_0.flac")
    assert orig0.abs().max().item() > 0.8
    assert not (song / "mixture.flac").exists()

    out_song = dest / "data" / "song"
    stem0 = load_stem(out_song / "0.flac")
    stem1 = load_stem(out_song / "1.flac")
    mixture = load_stem(out_song / "mixture.flac")
    assert (stem0 + stem1).abs().max().item() <= 1.0 + 1e-4
    np.testing.assert_allclose(
        to_mono_numpy(stem0 + stem1), to_mono_numpy(mixture), rtol=1e-3, atol=1e-3,
    )
    remapped = pd.read_csv(dest / "stems.csv")
    assert str(remapped.iloc[0]["path"]).startswith(str(dest))


def test_overwrite_with_yes_skips_prompt(tmp_path: Path, monkeypatch):
    source = tmp_path / "basic"
    song = _seed_song_tree(source)
    called = {"n": 0}

    def boom(_):
        called["n"] += 1
        return False

    monkeypatch.setattr("synthesis.mix.confirm_overwrite", boom)
    main(["--stems-dir", str(source), "--flac", "--yes", "--no-velocity-dynamics", "-j", "1"])
    assert called["n"] == 0
    stem0 = load_stem(song / "stem_0.flac")
    stem1 = load_stem(song / "stem_1.flac")
    assert (stem0 + stem1).abs().max().item() <= 1.0 + 1e-4
