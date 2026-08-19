"""Tests for packaging sanitized MIDIs into the sPDMX mid/ tree."""

from pathlib import Path

import pandas as pd

from analysis.corrected_midi import track_map_path_for_midi
from shared.config import SPDMX_AUDIO_DIR_NAME
from synthesis.package_midi import package_corrected_midis, parse_args
from synthesis.paths import spdmx_dataset_dir
from synthesis.synthesize import song_output_dir


def test_song_output_dir_swaps_data_for_audio():
    json_path = "/pdmx/data/1/11/QmSong.json"
    assert song_output_dir("/out", "/pdmx", json_path) == "/out/data/1/11/QmSong"
    assert song_output_dir(
        "/out", "/pdmx", json_path, tree_dir_name=SPDMX_AUDIO_DIR_NAME,
    ) == "/out/audio/1/11/QmSong"


def test_package_corrected_midis_copies_mid_and_track_map(tmp_path: Path):
    pdmx_root = tmp_path / "PDMX"
    (pdmx_root / "mid" / "1" / "11").mkdir(parents=True)
    (pdmx_root / "data" / "1" / "11").mkdir(parents=True)
    (pdmx_root / "data" / "1" / "11" / "QmSong.json").write_text("{}")
    pd.DataFrame({
        "path": ["./data/1/11/QmSong.json"],
        "mid": ["./mid/1/11/QmSong.mid"],
        "subset:all_valid": [True],
        "n_tracks": [1],
    }).to_csv(pdmx_root / "PDMX.csv", index=False)

    out = tmp_path / "SPDMX"
    corrected = out / "dev" / "mid_corrected" / "1" / "11"
    corrected.mkdir(parents=True)
    src_mid = corrected / "QmSong.mid"
    src_mid.write_bytes(b"MThd-fake")
    track_map_path_for_midi(src_mid).write_text("mid,track\n./mid/1/11/QmSong.mid,0\n")

    args = parse_args([
        "-o", str(out),
        "-df", str(pdmx_root / "PDMX.csv"),
        "--no-register",
        "--corrected-midi-dir", str(out / "dev" / "mid_corrected"),
    ])
    args.recipe = None
    dest_root = spdmx_dataset_dir(str(out))
    copied, skipped = package_corrected_midis(args, dest_root)
    assert copied == 1
    assert skipped == 0
    dest = Path(dest_root) / "mid" / "1" / "11" / "QmSong.mid"
    assert dest.is_file()
    assert dest.read_bytes() == b"MThd-fake"
    assert track_map_path_for_midi(dest).is_file()

    copied2, skipped2 = package_corrected_midis(args, dest_root)
    assert copied2 == 0
    assert skipped2 == 1
