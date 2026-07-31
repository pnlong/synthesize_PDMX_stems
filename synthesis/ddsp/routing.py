"""Route stems to MIDI-DDSP, DDSP-Piano, or slakh soundfont fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido

from synthesis.patches import _gm_class, _normalize_name

BACKEND_MIDI_DDSP = "midi_ddsp"
BACKEND_DDSP_PIANO = "ddsp_piano"
BACKEND_SOUNDFONT = "soundfont"

REASON_PIANO = "piano"
REASON_MIDI_DDSP = "midi_ddsp_eligible"
REASON_POLYPHONIC = "soundfont_polyphonic"
REASON_UNSUPPORTED = "soundfont_unsupported"
REASON_DRUM = "soundfont_drum"
REASON_BASS_GUITAR = "soundfont_bass_guitar"
REASON_VOCAL = "soundfont_vocal"
REASON_GUITAR = "soundfont_guitar"

# Canonical MIDI-DDSP / URMP names → GM program (0-indexed), matching magenta midi-ddsp.
MIDI_DDSP_NAME_TO_PROGRAM: dict[str, int] = {
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "double bass": 43,
    "flute": 73,
    "oboe": 68,
    "clarinet": 71,
    "saxophone": 66,
    "bassoon": 70,
    "trumpet": 56,
    "horn": 60,
    "trombone": 57,
    "tuba": 58,
}

MIDI_DDSP_PROGRAM_TO_NAME: dict[int, str] = {
    program: name for name, program in MIDI_DDSP_NAME_TO_PROGRAM.items()
}

# Track-name substrings → MIDI-DDSP instrument (checked before coarse GM class).
_NAME_ALIASES: tuple[tuple[str, str], ...] = (
    ("contrabass", "double bass"),
    ("double bass", "double bass"),
    ("doublebass", "double bass"),
    ("violoncello", "cello"),
    ("cello", "cello"),
    ("viola", "viola"),
    ("violin", "violin"),
    ("fiddle", "violin"),
    ("flute", "flute"),
    ("piccolo", "flute"),
    ("oboe", "oboe"),
    ("clarinet", "clarinet"),
    ("bassoon", "bassoon"),
    ("saxophone", "saxophone"),
    ("alto sax", "saxophone"),
    ("tenor sax", "saxophone"),
    ("bari sax", "saxophone"),
    ("sax", "saxophone"),
    ("trumpet", "trumpet"),
    ("cornet", "trumpet"),
    ("trombone", "trombone"),
    ("tuba", "tuba"),
    ("french horn", "horn"),
    ("horn", "horn"),
)

# GM programs that look like "bass" but are bass guitar / synth bass — never double bass.
BASS_GUITAR_PROGRAMS = frozenset(range(32, 40))

# GM piano family routed to DDSP-Piano (0–7 per plan).
PIANO_PROGRAMS = frozenset(range(0, 8))


@dataclass(frozen=True)
class StemRoute:
    backend: str
    instrument_key: str | None
    reason: str


def midi_ddsp_instrument_from_name(track_name: str | None) -> str | None:
    name = _normalize_name(track_name)
    if not name:
        return None
    # Prefer longer / more specific aliases first (tuple order).
    for needle, instrument in _NAME_ALIASES:
        if needle in name:
            return instrument
    return None


def midi_ddsp_instrument_from_program(program: int, is_drum: bool) -> str | None:
    if is_drum:
        return None
    if program in BASS_GUITAR_PROGRAMS:
        return None
    return MIDI_DDSP_PROGRAM_TO_NAME.get(int(program))


def is_piano_stem(*, program: int, is_drum: bool, track_name: str | None) -> bool:
    if is_drum:
        return False
    name = _normalize_name(track_name)
    if "piano" in name and "guitar" not in name:
        return True
    if any(k in name for k in ("harpsichord", "clavinet", "clavi", "electric piano", "epiano")):
        return True
    return int(program) in PIANO_PROGRAMS and _gm_class(program, is_drum) == "piano"


def is_vocal_stem(*, program: int, is_drum: bool, track_name: str | None) -> bool:
    if is_drum:
        return False
    name = _normalize_name(track_name)
    if any(k in name for k in ("voice", "vocal", "choir", "soprano", "alto", "tenor", "baritone", "bass singer")):
        # "bass" alone is ambiguous; require singer-ish cues above except choir/voice.
        return True
    gm = _gm_class(program, is_drum)
    return gm == "ensemble" and int(program) in (52, 53, 54)  # Choir Aahs, Voice Oohs, Synth Voice


def is_monophonic_messages(messages, *, ticks_per_beat: int = 480) -> bool:
    """True if at most one note is sounding at any time (velocity>0 note_on overlaps)."""
    del ticks_per_beat  # absolute tick timeline from delta times is enough
    open_notes: set[int] = set()
    for message in messages:
        if message.type == "note_on" and message.velocity > 0:
            if open_notes:
                return False
            open_notes.add(message.note)
        elif message.type in ("note_off", "note_on") and (
            message.type == "note_off" or message.velocity == 0
        ):
            open_notes.discard(message.note)
    return True


def is_monophonic_midi(midi_path: str | Path) -> bool:
    midi = mido.MidiFile(filename=str(midi_path), charset="utf8")
    # Merge all tracks onto one timeline for the stem file (usually single-track).
    if len(midi.tracks) == 1:
        return is_monophonic_messages(midi.tracks[0], ticks_per_beat=midi.ticks_per_beat)
    merged = mido.merge_tracks(midi.tracks)
    return is_monophonic_messages(merged, ticks_per_beat=midi.ticks_per_beat)


def is_monophonic_track(track, *, ticks_per_beat: int = 480) -> bool:
    return is_monophonic_messages(track, ticks_per_beat=ticks_per_beat)


def route_stem(
    *,
    program: int,
    is_drum: bool,
    track_name: str | None = None,
    midi_path: str | Path | None = None,
    track=None,
    ticks_per_beat: int = 480,
    check_monophony: bool = True,
) -> StemRoute:
    """Decide neural vs soundfont backend for one stem.

    Monophony is only required for MIDI-DDSP. DDSP-Piano accepts polyphony.
    When ``check_monophony`` is True, pass ``midi_path`` and/or ``track``.
    """
    if is_drum:
        return StemRoute(BACKEND_SOUNDFONT, None, REASON_DRUM)

    if is_vocal_stem(program=program, is_drum=is_drum, track_name=track_name):
        return StemRoute(BACKEND_SOUNDFONT, None, REASON_VOCAL)

    if is_piano_stem(program=program, is_drum=is_drum, track_name=track_name):
        return StemRoute(BACKEND_DDSP_PIANO, "piano", REASON_PIANO)

    name = _normalize_name(track_name)
    if any(k in name for k in ("guitar", "gitar", "gitarre", "luit")):
        return StemRoute(BACKEND_SOUNDFONT, None, REASON_GUITAR)
    if int(program) in BASS_GUITAR_PROGRAMS or (
        _gm_class(program, is_drum) == "bass" and int(program) in BASS_GUITAR_PROGRAMS
    ):
        return StemRoute(BACKEND_SOUNDFONT, None, REASON_BASS_GUITAR)
    if _gm_class(program, is_drum) == "bass":
        # Acoustic/electric bass class without mapping to double bass.
        return StemRoute(BACKEND_SOUNDFONT, None, REASON_BASS_GUITAR)
    if _gm_class(program, is_drum) == "guitar":
        return StemRoute(BACKEND_SOUNDFONT, None, REASON_GUITAR)

    instrument = midi_ddsp_instrument_from_name(track_name)
    if instrument is None:
        instrument = midi_ddsp_instrument_from_program(program, is_drum)

    if instrument is None:
        return StemRoute(BACKEND_SOUNDFONT, None, REASON_UNSUPPORTED)

    if check_monophony:
        mono = True
        if track is not None:
            mono = is_monophonic_track(track, ticks_per_beat=ticks_per_beat)
        elif midi_path is not None:
            mono = is_monophonic_midi(midi_path)
        if not mono:
            return StemRoute(BACKEND_SOUNDFONT, instrument, REASON_POLYPHONIC)

    return StemRoute(BACKEND_MIDI_DDSP, instrument, REASON_MIDI_DDSP)
