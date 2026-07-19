"""Tests for phase-1b diverse stem selection and noise audit helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from experiments.listening.aggregate import noise_audit_winners
from experiments.preset_sweep.config import (
    PHASE1,
    build_noise_audit_variants,
    lower_noise_level,
    noise_variant_id,
)
from experiments.preset_sweep.diverse_stems import (
    is_silent,
    replace_silent_probe_clips,
    select_diverse_stems,
    write_diverse_clip_dataset,
)
from experiments.preset_sweep.winners import record_phase_winners
from shared.config import DATA_DIR_NAME


def _write_routing_presets(path: Path):
    path.write_text(yaml.dump({
        "default": {"init_noise_level": 0.45, "prompt_variant": "current"},
        "routing": [
            {"category": "drums", "is_drum": True},
            {"category": "piano", "name_keywords": ["piano"]},
            {"category": "strings", "name_keywords": ["violin"]},
            {"category": "wind", "name_keywords": ["flute"]},
            {"category": "voice", "name_keywords": ["voice"]},
            {"category": "mallet", "name_keywords": ["marimba"]},
            {"category": "organ", "name_keywords": ["organ"]},
            {"category": "polyphonic", "name_keywords": ["ensemble"]},
        ],
    }))


def _write_dataset(tmp_path: Path, rows: list[dict]) -> Path:
    source_dir = tmp_path / "basic"
    songs = []
    stems = []
    for row in rows:
        song_id = row["song_id"]
        track = row["track"]
        song_dir = source_dir / DATA_DIR_NAME / song_id
        song_dir.mkdir(parents=True, exist_ok=True)
        sr = 44100
        duration = row.get("duration_seconds", 12)
        amplitude = row.get("amplitude", 0.2)
        sf.write(
            str(song_dir / f"stem_{track}.flac"),
            np.full(int(sr * duration), amplitude, dtype=np.float32),
            sr,
            format="FLAC",
        )
        path = str(song_dir)
        if path not in songs:
            songs.append({"path": path, "title": song_id, "genres": "classical"})
        stems.append({
            "path": path,
            "track": track,
            "program": row.get("program", 0),
            "is_drum": row.get("is_drum", False),
            "name": row.get("name", "Piano"),
            "has_lyrics": False,
        })

    pd.DataFrame(songs).to_csv(source_dir / f"{DATA_DIR_NAME}.csv", index=False)
    pd.DataFrame(stems).to_csv(source_dir / "stems.csv", index=False)
    return source_dir


def test_lower_noise_level_steps_down_grid():
    assert lower_noise_level(0.45) == 0.35
    assert lower_noise_level(0.25) == 0.25


def test_build_noise_audit_variants_union():
    variants = build_noise_audit_variants({
        "piano": "noise0.45",
        "drums": "noise0.25",
    })
    ids = {variant["id"] for variant in variants}
    assert ids == {"noise0.25", "noise0.35", "noise0.45"}


def test_noise_audit_winners_prefers_realism_after_content_gate_and_lowers_phase1():
    df = pd.DataFrame([
        {"stem_id": "a", "category": "piano", "variant_id": "noise0.45", "content": 4.8, "realism": 4.5},
        {"stem_id": "b", "category": "piano", "variant_id": "noise0.35", "content": 4.7, "realism": 5.0},
        {"stem_id": "c", "category": "piano", "variant_id": "noise0.45", "content": 4.7, "realism": 4.4},
        {"stem_id": "d", "category": "piano", "variant_id": "noise0.35", "content": 4.6, "realism": 4.9},
    ])
    winners_df, revisions = noise_audit_winners(
        df,
        {"piano": "noise0.45"},
        content_threshold=4.5,
    )
    assert winners_df.iloc[0]["variant_id"] == "noise0.35"
    assert revisions == {"piano": "noise0.35"}


def test_select_diverse_stems_filters_silence_and_short_audio(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "experiments.preset_sweep.diverse_stems.PROBE_CATEGORIES",
        ("piano",),
    )
    source_dir = _write_dataset(tmp_path, [
        {"song_id": "0/1/QmA", "track": 0, "name": "Piano", "amplitude": 0.0},
        {"song_id": "0/2/QmB", "track": 0, "name": "Piano", "duration_seconds": 5},
        {"song_id": "0/3/QmC", "track": 0, "name": "Piano", "amplitude": 0.3},
        {"song_id": "0/4/QmD", "track": 0, "name": "Piano", "amplitude": 0.25},
    ])

    stems = select_diverse_stems(
        source_dir,
        per_category=2,
        clip_seconds=10,
        min_rms=0.01,
        seed=1,
    )
    piano = [stem for stem in stems if stem["category"] == "piano"]
    assert len(piano) == 2
    assert all(stem["song_id"] in {"0/3/QmC", "0/4/QmD"} for stem in piano)


def test_write_diverse_clip_dataset_clips_to_ten_seconds(tmp_path: Path):
    source_dir = _write_dataset(tmp_path, [
        {"song_id": "0/9/QmClip", "track": 0, "name": "Piano", "duration_seconds": 20},
    ])
    stems = [{
        "id": "piano_clip",
        "category": "piano",
        "song_id": "0/9/QmClip",
        "track": 0,
        "audio_format": "flac",
    }]
    output_dir = tmp_path / "phase1b"
    write_diverse_clip_dataset(source_dir, stems, output_dir / "clips", clip_seconds=10)
    clip_path = output_dir / "clips" / DATA_DIR_NAME / "0/9/QmClip" / "stem_0.flac"
    assert clip_path.is_file()
    info = sf.info(str(clip_path))
    assert info.frames == pytest.approx(44100 * 10, rel=0.01)

    clip_stems = pd.read_csv(output_dir / "clips" / "stems.csv")
    clip_song_dir = str(output_dir / "clips" / DATA_DIR_NAME / "0/9/QmClip")
    assert clip_stems.iloc[0]["path"] == clip_song_dir


def test_replace_silent_probe_clips_reclips_late_entry(tmp_path: Path, monkeypatch):
    source_dir = _write_dataset(tmp_path, [{
        "song_id": "0/1/QmLate",
        "track": 0,
        "name": "Piano",
        "duration_seconds": 20,
        "amplitude": 0.0,
    }])
    stem_path = source_dir / DATA_DIR_NAME / "0/1/QmLate" / "stem_0.flac"
    sr = 44100
    audio = np.zeros(sr * 20, dtype=np.float32)
    audio[sr * 11: sr * 18] = 0.2
    sf.write(str(stem_path), audio, sr, format="FLAC")

    stems = [{
        "id": "piano_late",
        "category": "piano",
        "song_id": "0/1/QmLate",
        "track": 0,
        "audio_format": "flac",
    }]
    reference_clips = tmp_path / "phase1b" / "clips"
    write_diverse_clip_dataset(source_dir, stems, reference_clips, clip_seconds=10)
    monkeypatch.setattr(
        "experiments.preset_sweep.diverse_stems.probe_clip_usable",
        lambda clips_dir, probe, min_rms=0.01: False,
    )

    output_clips = tmp_path / "phase4" / "clips"
    updated, replacements = replace_silent_probe_clips(
        stems,
        reference_clips_dir=reference_clips,
        source_dir=source_dir,
        output_clips_dir=output_clips,
        clip_seconds=10,
        min_rms=0.01,
        seed=1,
    )

    assert replacements[0]["from_id"] == replacements[0]["to_id"] == "piano_late"
    assert replacements[0]["clip_start_seconds"] >= 1.0
    assert updated[0]["clip_start_seconds"] >= 1.0
    assert (output_clips / DATA_DIR_NAME / "0/1/QmLate" / "stem_0.flac").is_file()


def test_replace_silent_probe_clips_swaps_stem(tmp_path: Path, monkeypatch):
    source_dir = _write_dataset(tmp_path, [
        {"song_id": "0/1/QmA", "track": 0, "name": "Piano", "duration_seconds": 12},
        {"song_id": "0/2/QmB", "track": 0, "name": "Piano", "duration_seconds": 12, "amplitude": 0.0},
        {"song_id": "0/3/QmC", "track": 0, "name": "Piano", "duration_seconds": 12},
    ])
    silent_source = source_dir / DATA_DIR_NAME / "0/2/QmB" / "stem_0.flac"
    sf.write(str(silent_source), np.zeros(44100 * 12, dtype=np.float32), 44100, format="FLAC")
    stems = [
        {
            "id": "piano_a",
            "category": "piano",
            "song_id": "0/1/QmA",
            "track": 0,
            "audio_format": "flac",
        },
        {
            "id": "piano_b",
            "category": "piano",
            "song_id": "0/2/QmB",
            "track": 0,
            "audio_format": "flac",
        },
    ]
    reference_clips = tmp_path / "phase1b" / "clips"
    write_diverse_clip_dataset(source_dir, stems, reference_clips, clip_seconds=10)
    silent_path = reference_clips / DATA_DIR_NAME / "0/2/QmB" / "stem_0.flac"
    monkeypatch.setattr(
        "experiments.preset_sweep.diverse_stems.is_silent",
        lambda path, min_rms=0.01: path == silent_path,
    )

    output_clips = tmp_path / "phase4" / "clips"
    updated, replacements = replace_silent_probe_clips(
        stems,
        reference_clips_dir=reference_clips,
        source_dir=source_dir,
        output_clips_dir=output_clips,
        clip_seconds=10,
        min_rms=0.01,
        seed=1,
    )

    assert replacements == [{
        "category": "piano",
        "from_id": "piano_b",
        "to_id": "piano_0_3_QmC_t0",
    }]
    assert updated[0]["id"] == "piano_a"
    assert updated[1]["id"] == "piano_0_3_QmC_t0"
    assert (output_clips / DATA_DIR_NAME / "0/3/QmC" / "stem_0.flac").is_file()
