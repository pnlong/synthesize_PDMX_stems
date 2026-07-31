"""Tests for GM register name→program correction."""

from __future__ import annotations

from pathlib import Path

import mido
import pandas as pd
import pytest

from analysis.gm_register import (
    STATUS_CORRECTED,
    STATUS_KEEP,
    STATUS_SKIPPED_DRUM,
    STATUS_SKIPPED_GENERIC,
    STATUS_SKIPPED_UNNAMED,
    build_register_report,
    extract_register_rows_from_mid,
    load_alias_config,
    load_register_lookup,
    lookup_corrected_program,
    resolve_program,
)


@pytest.fixture(scope="module")
def config():
    from analysis import gm_register as gr

    gr.load_alias_config.cache_clear()
    gr._gm_name_needles.cache_clear()
    return load_alias_config()


def test_keep_piano_name_with_piano_program(config):
    result = resolve_program(track_name="Piano", program=0, is_drum=False, config=config)
    assert result.status == STATUS_KEEP
    assert result.program == 0


def test_correct_harpsichord_from_program_zero(config):
    result = resolve_program(track_name="Harpsichord", program=0, is_drum=False, config=config)
    assert result.status == STATUS_CORRECTED
    assert result.program == 6
    assert result.match_key == "harpsichord"


def test_satb_soprano_to_choir(config):
    result = resolve_program(track_name="Soprano", program=0, is_drum=False, config=config)
    assert result.status == STATUS_CORRECTED
    assert result.program == 52


def test_alto_sax_not_choir(config):
    result = resolve_program(track_name="Alto Sax", program=0, is_drum=False, config=config)
    assert result.status == STATUS_CORRECTED
    assert result.program == 65
    assert "sax" in result.match_key


def test_bare_bass_no_auto_correct(config):
    result = resolve_program(track_name="Bass", program=0, is_drum=False, config=config)
    # Bare "bass" is not an alias; may be no_match or family miss → not corrected to choir.
    assert result.program == 0
    assert result.status != STATUS_CORRECTED


def test_bass_singer_to_choir(config):
    result = resolve_program(track_name="Bass Singer", program=0, is_drum=False, config=config)
    assert result.status == STATUS_CORRECTED
    assert result.program == 52


def test_drum_skip(config):
    result = resolve_program(track_name="Drums", program=0, is_drum=True, config=config)
    assert result.status == STATUS_SKIPPED_DRUM
    assert result.program == 0


def test_unnamed_skip(config):
    result = resolve_program(track_name=None, program=0, is_drum=False, config=config)
    assert result.status == STATUS_SKIPPED_UNNAMED


def test_generic_melody_skip(config):
    result = resolve_program(track_name="Melody", program=0, is_drum=False, config=config)
    assert result.status == STATUS_SKIPPED_GENERIC
    assert result.program == 0


def test_family_agree_guitar_already_in_class(config):
    result = resolve_program(track_name="Guitar", program=27, is_drum=False, config=config)
    assert result.status == STATUS_KEEP
    assert result.program == 27


def test_family_fix_guitar_from_piano(config):
    result = resolve_program(track_name="Guitar", program=0, is_drum=False, config=config)
    assert result.status == STATUS_CORRECTED
    assert result.program == 24


def test_synth_family_does_not_map_to_square_lead(config):
    result = resolve_program(
        track_name="Warm Synthesizer", program=0, is_drum=False, config=config
    )
    # Specific warm synthesizer alias → pad; not bare synth → 80.
    assert result.program == 89
    assert result.status == STATUS_CORRECTED


def test_musicxml_part_not_music_box(config):
    result = resolve_program(
        track_name="MusicXML Part", program=0, is_drum=False, config=config
    )
    assert result.program == 0
    assert result.status != STATUS_CORRECTED


def test_tenore_to_choir(config):
    result = resolve_program(track_name="Tenore", program=0, is_drum=False, config=config)
    assert result.status == STATUS_CORRECTED
    assert result.program == 52
    assert result.match_key == "tenore"


def test_accented_tenor_folds(config):
    result = resolve_program(track_name="Ténor", program=0, is_drum=False, config=config)
    assert result.status == STATUS_CORRECTED
    assert result.program == 52


def test_flote_german(config):
    result = resolve_program(track_name="Flöte", program=0, is_drum=False, config=config)
    assert result.status == STATUS_CORRECTED
    assert result.program == 73


def test_arpa_italian_harp(config):
    result = resolve_program(track_name="Arpa", program=0, is_drum=False, config=config)
    assert result.status == STATUS_CORRECTED
    assert result.program == 46


def test_cembalo_harpsichord(config):
    result = resolve_program(track_name="Cembalo", program=0, is_drum=False, config=config)
    assert result.status == STATUS_CORRECTED
    assert result.program == 6


def test_chor_does_not_match_inside_harpsichord(config):
    result = resolve_program(track_name="Harpsichord", program=0, is_drum=False, config=config)
    assert result.program == 6
    assert result.match_key == "harpsichord"


def test_bassoon_not_basso_vocal(config):
    result = resolve_program(track_name="Bassoon", program=0, is_drum=False, config=config)
    assert result.program == 70


def test_basso_vocal(config):
    result = resolve_program(track_name="Basso", program=0, is_drum=False, config=config)
    assert result.status == STATUS_CORRECTED
    assert result.program == 52


def test_extract_register_rows_indexing(tmp_path: Path, config):
    mid_path = tmp_path / "song.mid"
    midi = mido.MidiFile(ticks_per_beat=480)

    empty = mido.MidiTrack()
    empty.append(mido.MetaMessage("track_name", name="Empty", time=0))
    midi.tracks.append(empty)

    harp = mido.MidiTrack()
    harp.append(mido.MetaMessage("track_name", name="Harpsichord", time=0))
    harp.append(mido.Message("program_change", program=0, channel=0, time=0))
    harp.append(mido.Message("note_on", note=60, velocity=80, channel=0, time=0))
    harp.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=480))
    midi.tracks.append(harp)

    midi.save(mid_path)
    rows = extract_register_rows_from_mid(
        mid_path,
        mid_rel="./mid/song.mid",
        config=config,
    )
    assert rows is not None
    assert len(rows) == 2
    assert rows[0]["track"] == 0
    assert rows[1]["track"] == 1
    assert rows[1]["program_original"] == 0
    assert rows[1]["program_corrected"] == 6
    assert rows[1]["status"] == STATUS_CORRECTED


def test_build_register_report_stats():
    df = pd.DataFrame(
        [
            {
                "mid": "./a.mid",
                "track": 0,
                "name": "Harpsichord",
                "is_drum": False,
                "program_original": 0,
                "program_corrected": 6,
                "gm_name_original": "Acoustic Grand Piano",
                "gm_name_corrected": "Harpsichord",
                "status": STATUS_CORRECTED,
                "match_key": "harpsichord",
            },
            {
                "mid": "./a.mid",
                "track": 1,
                "name": "Piano",
                "is_drum": False,
                "program_original": 0,
                "program_corrected": 0,
                "gm_name_original": "Acoustic Grand Piano",
                "gm_name_corrected": "Acoustic Grand Piano",
                "status": STATUS_KEEP,
                "match_key": None,
            },
            {
                "mid": "./a.mid",
                "track": 2,
                "name": "Soprano",
                "is_drum": False,
                "program_original": 0,
                "program_corrected": 52,
                "gm_name_original": "Acoustic Grand Piano",
                "gm_name_corrected": "Choir Aahs",
                "status": STATUS_CORRECTED,
                "match_key": "soprano",
            },
        ]
    )
    report = build_register_report(df, subset="all_valid", n_songs=1, top_n=5)
    assert report["n_tracks"] == 3
    assert report["n_corrected"] == 2
    assert report["n_fine_as_is"] == 1
    assert report["pct_corrected"] == pytest.approx(66.67, abs=0.01)
    assert report["top_corrections"]
    assert report["top_match_keys"][0]["match_key"] in ("harpsichord", "soprano")

    from analysis.gm_register import format_register_report

    text = format_register_report(report)
    assert "GM register summary" in text
    assert "corrected: 2" in text
    assert "top corrections:" in text


def test_load_register_lookup(tmp_path: Path):
    csv_path = tmp_path / "register.csv"
    pd.DataFrame(
        [
            {
                "mid": "./mid/x.mid",
                "track": 0,
                "name": "Harpsichord",
                "is_drum": False,
                "program_original": 0,
                "program_corrected": 6,
                "gm_name_original": "Acoustic Grand Piano",
                "gm_name_corrected": "Harpsichord",
                "status": STATUS_CORRECTED,
                "match_key": "harpsichord",
            }
        ]
    ).to_csv(csv_path, index=False)

    lookup = load_register_lookup(csv_path, pdmx_root=tmp_path)
    assert lookup_corrected_program(lookup, mid="./mid/x.mid", track=0, default=0) == 6
    assert lookup_corrected_program(lookup, mid="./mid/x.mid", track=1, default=3) == 3
