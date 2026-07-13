"""Tests for shared probe stem manifest."""

from pathlib import Path

import mido
import pytest
import yaml

from experiments.probe_stems import (
    PROBE_CATEGORIES,
    TARGET_SAMPLES_PER_CATEGORY,
    category_counts,
    load_probe_stems,
    read_track_midi_meta,
    stem_matches_category,
    validate_probe_stem_midi_programs,
    validate_probe_stems,
)
from experiments.paths import DEFAULT_PROBE_STEMS
from shared.config import PDMX_FILEPATH


def test_default_probe_stems_has_three_per_category():
    stems = load_probe_stems()
    validate_probe_stems(stems, validate_midi_programs=Path(PDMX_FILEPATH).is_file())
    counts = category_counts(stems)
    assert len(stems) == len(PROBE_CATEGORIES) * TARGET_SAMPLES_PER_CATEGORY
    for category in PROBE_CATEGORIES:
        assert counts[category] == TARGET_SAMPLES_PER_CATEGORY


def test_validate_probe_stems_rejects_short_category():
    stems = load_probe_stems()
    bad = [s for s in stems if s["category"] != "drums"] + [
        s for s in stems if s["category"] == "drums"
    ][:1]
    with pytest.raises(ValueError, match="drums"):
        validate_probe_stems(bad, validate_midi_programs=False)


def test_probe_stem_ids_are_unique():
    stems = load_probe_stems()
    ids = [stem["id"] for stem in stems]
    assert len(ids) == len(set(ids))


def test_validate_probe_stem_midi_programs_rejects_piano_for_voice(tmp_path: Path):
    mid_path = tmp_path / "song.mid"
    midi = mido.MidiFile(ticks_per_beat=480, charset="utf8")
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="SOPRANO", time=0))
    track.append(mido.Message("program_change", program=0, time=0, channel=0))
    midi.tracks.append(track)
    midi.save(str(mid_path))

    stems = [{
        "id": "bad_voice",
        "category": "voice",
        "song_id": "0/0/QmTest",
        "track": 0,
    }]

    pdmx_csv = tmp_path / "PDMX.csv"
    pdmx_csv.write_text("mid\n./mid/0/0/QmTest.mid\n")
    mid_copy = tmp_path / "mid" / "0/0" / "QmTest.mid"
    mid_copy.parent.mkdir(parents=True)
    mid_copy.write_bytes(mid_path.read_bytes())

    with pytest.raises(ValueError, match="bad_voice"):
        validate_probe_stem_midi_programs(stems, pdmx_filepath=str(pdmx_csv))


def test_stem_matches_category_voice_program_52(tmp_path: Path):
    mid_path = tmp_path / "voice.mid"
    midi = mido.MidiFile(ticks_per_beat=480, charset="utf8")
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="Soprano", time=0))
    track.append(mido.Message("program_change", program=52, time=0, channel=0))
    midi.tracks.append(track)
    midi.save(str(mid_path))

    meta = read_track_midi_meta(mid_path, 0)
    assert stem_matches_category(meta, "voice")


@pytest.mark.skipif(not Path(PDMX_FILEPATH).is_file(), reason="PDMX not available")
def test_default_probe_stem_midi_programs_match_categories():
    validate_probe_stem_midi_programs(load_probe_stems())
