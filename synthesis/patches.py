"""Slakh-style patch selection (stub).

Slakh randomly assigns a patch from a class-matched pool for each track's MIDI
program number (187 patches, 34 classes). That table is not implemented yet.

When implemented, `select_patch` will return a bank/program pair or soundfont
patch id used before rendering. For now it passes through the original program.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PatchAssignment:
    program: int
    is_drum: bool
    # Future: bank, patch_name, soundfont_path, effects flags, etc.


def select_patch(
    program: int,
    is_drum: bool,
    rng: random.Random | None = None,
) -> PatchAssignment:
    """Select a rendering patch for a track. Stub: use PDMX program unchanged."""
    _ = rng  # reserved for random class-matched patch pools
    return PatchAssignment(program=program, is_drum=is_drum)


def apply_patch_to_midi_track(track, assignment: PatchAssignment):
    """Rewrite program changes on a mido track. Stub: no-op until patch pools exist."""
    _ = track, assignment
