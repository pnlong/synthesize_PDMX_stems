"""Tests for route_stem-based donor-copy equivalence detection."""

from pathlib import Path

import mido
import pandas as pd

from experiments.ablation_listening.conditions import ABLATION_MUSHRA_CONDITIONS
from experiments.ablation_listening.equivalence import (
    DONOR_EQUIVALENCE_PAIRS,
    detect_equivalences_for_stem,
    detect_equivalences_for_trial,
    donor_equivalences_for_backend,
    equivalences_by_trial_id,
    unique_condition_ids,
)
from synthesis.ddsp.routing import BACKEND_DDSP_PIANO, BACKEND_SOUNDFONT


def _track(*messages) -> mido.MidiTrack:
    track = mido.MidiTrack()
    for message in messages:
        track.append(message)
    return track


def test_donor_equivalences_for_backend():
    assert donor_equivalences_for_backend(BACKEND_SOUNDFONT) == DONOR_EQUIVALENCE_PAIRS
    assert donor_equivalences_for_backend(BACKEND_DDSP_PIANO) == {}


def test_detect_equivalences_drums_are_fallback():
    equiv = detect_equivalences_for_stem(program=0, is_drum=True, check_monophony=False)
    assert equiv == DONOR_EQUIVALENCE_PAIRS


def test_detect_equivalences_piano_is_neural():
    equiv = detect_equivalences_for_stem(program=0, is_drum=False, check_monophony=False)
    assert equiv == {}


def test_detect_equivalences_poly_violin_is_fallback():
    track = _track(
        mido.Message("program_change", program=40, time=0),
        mido.Message("note_on", note=60, velocity=80, time=0),
        mido.Message("note_on", note=64, velocity=80, time=0),  # overlap
        mido.Message("note_off", note=60, velocity=0, time=480),
        mido.Message("note_off", note=64, velocity=0, time=0),
    )
    equiv = detect_equivalences_for_stem(
        program=40,
        is_drum=False,
        track_name="Violin",
        track=track,
        check_monophony=True,
    )
    assert equiv == DONOR_EQUIVALENCE_PAIRS


def test_detect_equivalences_mono_violin_is_neural():
    track = _track(
        mido.Message("program_change", program=40, time=0),
        mido.Message("note_on", note=60, velocity=80, time=0),
        mido.Message("note_off", note=60, velocity=0, time=480),
        mido.Message("note_on", note=62, velocity=80, time=0),
        mido.Message("note_off", note=62, velocity=0, time=480),
    )
    equiv = detect_equivalences_for_stem(
        program=40,
        is_drum=False,
        track_name="Violin",
        track=track,
        check_monophony=True,
    )
    assert equiv == {}


def test_detect_equivalences_for_trial_reads_stems_csv(tmp_path: Path):
    song_id = "0/0/QmDrums"
    basic = tmp_path / "basic"
    (basic / "data" / song_id).mkdir(parents=True)
    pd.DataFrame([{
        "path": str(basic / "data" / song_id),
        "track": 3,
        "program": 0,
        "is_drum": True,
        "name": "Drums",
        "has_lyrics": False,
    }]).to_csv(basic / "stems.csv", index=False)

    trial = {
        "id": "stem_drums_01",
        "type": "stem",
        "song_id": song_id,
        "track": 3,
        "category": "drums",
    }
    # No MIDI needed for drums (route returns early).
    equiv = detect_equivalences_for_trial(trial, tmp_path)
    assert equiv == DONOR_EQUIVALENCE_PAIRS


def test_detect_equivalences_for_mixture_empty(tmp_path: Path):
    assert detect_equivalences_for_trial(
        {"id": "mix_01", "type": "mixture", "song_id": "0/0/Qm"},
        tmp_path,
    ) == {}


def test_unique_condition_ids_omits_duplicates():
    equiv = {"ddsp_basic": "basic", "ddsp_slakh": "slakh"}
    unique = unique_condition_ids(ABLATION_MUSHRA_CONDITIONS, equiv)
    assert "ddsp_basic" not in unique
    assert "ddsp_slakh" not in unique
    assert "basic" in unique
    assert len(unique) == len(ABLATION_MUSHRA_CONDITIONS) - 2


def test_equivalences_by_trial_id_filters_invalid():
    manifest = {
        "trials": [
            {
                "id": "stem_drums_01",
                "equivalences": {
                    "ddsp_basic": "basic",
                    "ddsp_basic_realify": "slakh",  # invalid donor pairing
                },
            },
            {"id": "stem_piano_01"},
        ]
    }
    by_id = equivalences_by_trial_id(manifest)
    assert by_id == {"stem_drums_01": {"ddsp_basic": "basic"}}
