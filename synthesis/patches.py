"""Slakh-style rendering configuration.

Per-category soundfont shortlists and FX profiles are loaded from
``experiments/patch_sweep/winners_locked.yaml`` after the tuning pipeline.
Production slakh mode randomly picks a soundfont per (song, category) from
each shortlist while keeping MIDI programs unchanged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import yaml

# pool_id -> GM class -> list of General MIDI program numbers (0–127).
_PATCH_V1: dict[str, list[int]] = {
    "piano": [0, 1, 2],
    "chromatic_percussion": [9, 11, 12],
    "organ": [16, 17, 18],
    "guitar": [24, 25, 26],
    "bass": [32, 33, 34],
    "strings": [40, 42, 48],
    "ensemble": [52, 53],
    "brass": [56, 57, 60],
    "reed": [65, 66, 72],
    "pipe": [73, 74],
    "synth_lead": [80, 81],
    "synth_pad": [89, 90],
    "drums": [0],
}

_PATCH_V2: dict[str, list[int]] = {
    **_PATCH_V1,
    "piano": [0, 1, 2, 3, 4, 5],
    "chromatic_percussion": [8, 9, 11, 12, 13],
    "organ": [16, 17, 18, 19, 20],
    "guitar": [24, 25, 26, 27, 28],
    "bass": [32, 33, 34, 35, 36],
    "strings": [40, 41, 42, 43, 44, 45, 48],
    "ensemble": [48, 52, 53, 54],
    "brass": [56, 57, 58, 59, 60, 61],
    "reed": [64, 65, 66, 67, 68, 71, 72],
    "pipe": [71, 72, 73, 74, 75],
}

_PATCH_V3: dict[str, list[int]] = {
    **_PATCH_V2,
    "piano": [0, 1, 2, 4, 7, 3, 5, 6],
    "chromatic_percussion": [8, 9, 10, 11, 12, 13, 14, 15],
    "organ": [16, 17, 18, 19, 20, 21, 22, 23],
    "guitar": [24, 25, 26, 27, 28, 29, 30, 31],
    "bass": [32, 33, 34, 35, 36, 37, 38],
    "strings": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
    "ensemble": [48, 49, 50, 52, 53, 54, 55],
    "brass": [56, 57, 58, 59, 60, 61, 62, 63],
    "reed": [64, 65, 66, 67, 68, 69, 70, 71, 72],
    "pipe": [71, 72, 73, 74, 75, 76, 77, 78, 79],
}

PATCH_POOLS: dict[str, dict[str, list[int]]] = {
    "pool_v1_conservative": dict(_PATCH_V1),
    "pool_v2_diverse": dict(_PATCH_V2),
    "pool_v3_slakh_like": dict(_PATCH_V3),
}

# Listening categories (probe / locked winners) -> GM pool keys used for random patch choice.
LISTENING_CATEGORY_GM_CLASSES: dict[str, tuple[str, ...]] = {
    "piano": ("piano",),
    "drums": ("drums",),
    "strings": ("strings",),
    "wind": ("pipe", "reed", "brass"),
    "voice": ("ensemble",),
    "mallet": ("chromatic_percussion",),
    "organ": ("organ",),
}

WINNERS_LOCKED_PATH = (
    Path(__file__).resolve().parent.parent / "experiments" / "patch_sweep" / "winners_locked.yaml"
)

# probe listening category -> production render recipe (filled by patch_sweep.lock).
SLAKH_CATEGORY_RENDER: dict[str, dict] = {}


def _load_slakh_category_render() -> dict[str, dict]:
    if not WINNERS_LOCKED_PATH.is_file():
        return {}
    with open(WINNERS_LOCKED_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    return dict(cfg.get("categories") or {})


SLAKH_CATEGORY_RENDER.update(_load_slakh_category_render())


def _gm_class(program: int, is_drum: bool) -> str:
    if is_drum:
        return "drums"
    if program < 8:
        return "piano"
    if program < 16:
        return "chromatic_percussion"
    if program < 24:
        return "organ"
    if program < 32:
        return "guitar"
    if program < 40:
        return "bass"
    if program < 48:
        return "strings"
    if program < 56:
        return "ensemble"
    if program < 64:
        return "brass"
    if program < 72:
        return "reed"
    if program < 80:
        return "pipe"
    if program < 88:
        return "synth_lead"
    if program < 96:
        return "synth_pad"
    if program < 104:
        return "synth_effects"
    if program < 112:
        return "ethnic"
    if program < 120:
        return "percussive"
    return "sound_effects"


@dataclass(frozen=True)
class PatchAssignment:
    program: int
    is_drum: bool
    pool_id: str | None = None
    gm_class: str | None = None


def patch_group_key(program: int, is_drum: bool) -> str:
    """Stable within-song grouping for patch randomization (GM instrument class)."""
    return _gm_class(program, is_drum)


def patch_seed(sample_seed: int, song_path: str, group_key: str) -> int:
    """Seed patch RNG per (song, instrument class) — same class in a song, varied across songs."""
    return hash((sample_seed, song_path, group_key)) & 0x7FFFFFFF


def pool_candidates(
    pool_id: str,
    *,
    program: int,
    is_drum: bool,
    category: str | None = None,
) -> list[int]:
    """Programs to sample from for a track, optionally scoped to a listening category."""
    pool = PATCH_POOLS.get(pool_id, {})
    gm_class = _gm_class(program, is_drum)
    if category and category in LISTENING_CATEGORY_GM_CLASSES:
        gm_keys = LISTENING_CATEGORY_GM_CLASSES[category]
    else:
        gm_keys = (gm_class,)

    candidates: list[int] = []
    seen: set[int] = set()
    for key in gm_keys:
        for candidate in pool.get(key, []):
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def select_patch(
    program: int,
    is_drum: bool,
    *,
    pool_id: str | None = None,
    category: str | None = None,
    rng: random.Random | None = None,
) -> PatchAssignment:
    """Select a rendering patch for a track.

    When ``pool_id`` is set, pick a GM program at random from the pool. If
    ``category`` is set (listening category from tuning), sample across all GM
    classes for that category — e.g. wind draws from pipe + reed + brass lists.
    Otherwise fall back to the track's native GM class. Without a pool, pass
    through the original program unchanged.
    """
    gm_class = _gm_class(program, is_drum)
    chosen = program

    if pool_id is not None:
        candidates = pool_candidates(
            pool_id,
            program=program,
            is_drum=is_drum,
            category=category,
        )
        if candidates:
            chosen = rng.choice(candidates) if rng is not None else candidates[0]

    return PatchAssignment(
        program=chosen,
        is_drum=is_drum,
        pool_id=pool_id,
        gm_class=gm_class,
    )


def apply_patch_to_midi_track(track, assignment: PatchAssignment):
    """Rewrite program changes on a mido track to match the assignment."""
    for message in track:
        if message.type == "program_change":
            message.program = assignment.program


def _normalize_name(name: str | None) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def resolve_probe_category(
    *,
    program: int,
    is_drum: bool,
    track_name: str | None = None,
) -> str:
    """Map a track to a probe listening category (matches probe_stems.yaml)."""
    name = _normalize_name(track_name)
    if is_drum:
        return "drums"
    if "piano" in name:
        return "piano"
    if "organ" in name:
        return "organ"
    if any(k in name for k in ("marimba", "vibraphone", "xylophone", "glockenspiel")):
        return "mallet"
    if any(k in name for k in ("voice", "soprano", "alto", "tenor", "baritone", "choir", "vocal")):
        return "voice"
    if any(k in name for k in ("violin", "viola", "cello", "contrabass", "string")):
        return "strings"
    if any(k in name for k in ("flute", "clarinet", "saxophone", "sax", "oboe", "bassoon", "piccolo")):
        return "wind"
    gm = _gm_class(program, is_drum)
    if gm == "piano":
        return "piano"
    if gm in ("reed", "pipe"):
        return "wind"
    if gm == "strings":
        return "strings"
    if gm == "chromatic_percussion":
        return "mallet"
    if gm == "organ":
        return "organ"
    if gm == "ensemble":
        return "voice"
    return "polyphonic"


def slakh_render_for_category(category: str | None) -> dict:
    """Return locked render recipe for a probe category, with fallbacks."""
    recipe: dict = {"category": category}
    if category and category in SLAKH_CATEGORY_RENDER:
        recipe.update(SLAKH_CATEGORY_RENDER[category])
        return recipe
    if "default" in SLAKH_CATEGORY_RENDER:
        recipe.update(SLAKH_CATEGORY_RENDER["default"])
    return recipe


def slakh_render_for_track(
    *,
    program: int,
    is_drum: bool,
    track_name: str | None = None,
) -> dict:
    category = resolve_probe_category(
        program=program,
        is_drum=is_drum,
        track_name=track_name,
    )
    return slakh_render_for_category(category)
