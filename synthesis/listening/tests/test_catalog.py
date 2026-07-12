"""Tests for ablation listening catalog."""

from pathlib import Path

import pandas as pd
import pytest

from synthesis.listening.catalog import (
    AblationCatalog,
    CONDITION_LABELS,
    CONDITION_ORDER,
    detect_audio_format,
    song_id_from_path,
)


def _write_ablation_tree(
    root: Path,
    *,
    conditions: tuple[str, ...] = ("basic", "basic_realify"),
    song_rel: str = "7/19/QmTestSong",
    n_tracks: int = 2,
    audio_format: str = "mp3",
    include_realify_audio: bool = True,
):
    song_path = root / "basic" / "data" / song_rel
    for condition in conditions:
        cond_dir = root / condition
        cond_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "path": [str(song_path)],
        "title": ["Test Title"],
        "song_name": ["test song"],
        "artist_name": ["Test Artist"],
        "genres": ["classical"],
        "n_tracks": [n_tracks],
        "song_length.seconds": [120.5],
    }).to_csv(root / "basic" / "data.csv", index=False)

    stems_rows = []
    for track in range(n_tracks):
        stems_rows.append({
            "path": str(song_path),
            "track": track,
            "program": 0,
            "is_drum": False,
            "name": f"Stem {track}",
            "has_lyrics": False,
        })
    pd.DataFrame(stems_rows).to_csv(root / "basic" / "stems.csv", index=False)

    for condition in ("basic", "basic_realify"):
        if condition not in conditions:
            continue
        out_song_dir = root / condition / "data" / song_rel
        out_song_dir.mkdir(parents=True, exist_ok=True)
        write_audio = condition == "basic" or include_realify_audio
        if not write_audio:
            continue
        for track in range(n_tracks):
            (out_song_dir / f"stem_{track}.{audio_format}").write_bytes(b"stem")
        (out_song_dir / f"mixture.{audio_format}").write_bytes(b"mix")

    if "basic_realify" in conditions:
        for name in ("data.csv", "stems.csv"):
            src = root / "basic" / name
            if src.exists():
                (root / "basic_realify" / name).write_text(src.read_text())


def test_song_id_from_path():
    path = "/deepfreeze/pnlong/SPDMX/dev/ablations/basic/data/7/19/QmTest"
    assert song_id_from_path(path) == "7/19/QmTest"


def test_detect_audio_format_mp3(tmp_path: Path):
    song_dir = tmp_path / "song"
    song_dir.mkdir()
    (song_dir / "mixture.mp3").write_bytes(b"x")
    assert detect_audio_format(song_dir) == "mp3"


def test_detect_audio_format_flac(tmp_path: Path):
    song_dir = tmp_path / "song"
    song_dir.mkdir()
    (song_dir / "mixture.flac").write_bytes(b"x")
    assert detect_audio_format(song_dir) == "flac"


def test_conditions_and_list_songs(tmp_path: Path):
    _write_ablation_tree(tmp_path)
    catalog = AblationCatalog(tmp_path)

    conditions = catalog.conditions()
    assert [c["id"] for c in conditions] == list(CONDITION_ORDER)
    assert conditions[0]["available"] is True
    assert conditions[0]["label"] == CONDITION_LABELS["basic"]
    assert conditions[2]["available"] is False

    songs = catalog.list_songs()
    assert len(songs) == 1
    assert songs[0]["id"] == "7/19/QmTestSong"
    assert songs[0]["title"] == "Test Title"
    assert songs[0]["n_tracks"] == 2


def test_get_song_resolves_cross_condition_paths(tmp_path: Path):
    _write_ablation_tree(tmp_path)
    catalog = AblationCatalog(tmp_path)
    detail = catalog.get_song("7/19/QmTestSong")

    assert detail is not None
    assert detail["audio_format"] == "mp3"
    assert detail["path"].endswith("7/19/QmTestSong")
    assert detail["song_dirs"]["basic"].endswith("basic/data/7/19/QmTestSong")
    assert detail["song_dirs"]["basic_realify"].endswith("basic_realify/data/7/19/QmTestSong")
    assert detail["mixture"]["basic"]["available"] is True
    assert detail["mixture"]["basic_realify"]["available"] is True
    assert detail["mixture"]["basic_realify"]["url"] == (
        "/audio/basic_realify/7/19/QmTestSong/mixture.mp3"
    )
    assert detail["mixture"]["slakh"]["available"] is False
    assert len(detail["stems"]) == 2
    assert detail["stems"][0]["caption"] is not None
    assert "Preserve the exact melody" in detail["stems"][0]["caption"]


def test_missing_realify_audio_marked_unavailable(tmp_path: Path):
    _write_ablation_tree(tmp_path, include_realify_audio=False)
    catalog = AblationCatalog(tmp_path)
    detail = catalog.get_song("7/19/QmTestSong")

    assert detail["mixture"]["basic"]["available"] is True
    assert detail["mixture"]["basic_realify"]["available"] is False


def test_resolve_audio_path_rejects_traversal(tmp_path: Path):
    _write_ablation_tree(tmp_path)
    catalog = AblationCatalog(tmp_path)
    assert catalog.resolve_audio_path("basic", "../evil", "mixture.mp3") is None
    assert catalog.resolve_audio_path("basic", "7/19/QmTestSong", "../data.csv") is None


def test_resolve_audio_path_returns_file(tmp_path: Path):
    _write_ablation_tree(tmp_path)
    catalog = AblationCatalog(tmp_path)
    path = catalog.resolve_audio_path("basic_realify", "7/19/QmTestSong", "stem_1.mp3")
    assert path is not None
    assert path.name == "stem_1.mp3"


def test_get_song_unknown_returns_none(tmp_path: Path):
    _write_ablation_tree(tmp_path)
    catalog = AblationCatalog(tmp_path)
    assert catalog.get_song("missing/song") is None


def test_catalog_requires_data_csv(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No ablation condition"):
        AblationCatalog(tmp_path)
