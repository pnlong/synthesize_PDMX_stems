"""Tests for in-repo dev artifact symlinks."""

from pathlib import Path

from shared.repo_symlinks import (
    REPO_ABLATIONS_SYMLINK,
    REPO_ANALYSIS_SYMLINK,
    link_ablations_in_repo,
    link_analysis_in_repo,
    setup_dev_symlinks,
)
from synthesis.paths import ablations_root, analysis_root


def test_link_analysis_in_repo(tmp_path: Path, monkeypatch):
    spdmx_root = str(tmp_path / "SPDMX")
    analysis_dir = Path(analysis_root(spdmx_root))
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "song_lengths").mkdir()

    symlink = tmp_path / "repo" / "analysis" / "output"
    symlink.parent.mkdir(parents=True)
    monkeypatch.setattr("shared.repo_symlinks.REPO_ANALYSIS_SYMLINK", symlink)
    monkeypatch.setattr("shared.repo_symlinks.LEGACY_ANALYSIS_SYMLINKS", ())

    link, target = link_analysis_in_repo(spdmx_root)
    assert link == symlink
    assert symlink.is_symlink()
    assert symlink.resolve() == analysis_dir.resolve()


def test_link_ablations_in_repo(tmp_path: Path, monkeypatch):
    spdmx_root = str(tmp_path / "SPDMX")
    ablations_dir = Path(ablations_root(spdmx_root))
    ablations_dir.mkdir(parents=True)
    (ablations_dir / "basic").mkdir()

    symlink = tmp_path / "repo" / "synthesis" / "ablations_output"
    symlink.parent.mkdir(parents=True)
    monkeypatch.setattr("shared.repo_symlinks.REPO_ABLATIONS_SYMLINK", symlink)

    link, target = link_ablations_in_repo(spdmx_root)
    assert link == symlink
    assert symlink.is_symlink()
    assert symlink.resolve() == ablations_dir.resolve()
    assert (symlink / "basic").is_dir()


def test_setup_dev_symlinks(tmp_path: Path, monkeypatch):
    spdmx_root = str(tmp_path / "SPDMX")
    Path(analysis_root(spdmx_root)).mkdir(parents=True)
    Path(ablations_root(spdmx_root)).mkdir(parents=True)

    analysis_link = tmp_path / "repo" / "analysis" / "output"
    ablations_link = tmp_path / "repo" / "synthesis" / "ablations_output"
    analysis_link.parent.mkdir(parents=True)
    ablations_link.parent.mkdir(parents=True)
    monkeypatch.setattr("shared.repo_symlinks.REPO_ANALYSIS_SYMLINK", analysis_link)
    monkeypatch.setattr("shared.repo_symlinks.REPO_ABLATIONS_SYMLINK", ablations_link)
    monkeypatch.setattr("shared.repo_symlinks.LEGACY_ANALYSIS_SYMLINKS", ())

    links = setup_dev_symlinks(spdmx_root)
    assert len(links) == 2
    assert analysis_link.is_symlink()
    assert ablations_link.is_symlink()
