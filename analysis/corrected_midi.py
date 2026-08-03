"""Write dense corrected MIDI copies (register applied, empty tracks dropped)."""

from __future__ import annotations

from pathlib import Path

import mido
import pandas as pd

from synthesis.patches import PatchAssignment, apply_patch_to_midi_track

TRACK_MAP_COLUMNS = [
    "mid",
    "track",
    "original_track",
    "program",
    "is_drum",
    "name",
]

_CONDUCTOR_META_TYPES = frozenset({
    "set_tempo",
    "time_signature",
    "key_signature",
    "smpte_offset",
})


def note_on_count(track) -> int:
    return sum(
        1
        for message in track
        if getattr(message, "type", None) == "note_on"
        and getattr(message, "velocity", 0) > 0
    )


def track_map_path_for_midi(midi_path: str | Path) -> Path:
    path = Path(midi_path)
    return path.with_suffix(path.suffix + ".track_map.csv")


def _conductor_meta_prefix(midi: mido.MidiFile) -> list:
    """Tempo / meter meta to prepend onto each note-bearing track.

    Prefer messages from track 0 (standard MIDI conductor). Falls back to the
    first matching meta found anywhere so timing is preserved when track 0 is
    missing those events.
    """
    collected: dict[str, object] = {}
    sources: list = list(midi.tracks[:1]) + list(midi.tracks[1:])
    for track in sources:
        abs_time = 0
        for message in track:
            abs_time += message.time
            if not getattr(message, "is_meta", False):
                continue
            if message.type not in _CONDUCTOR_META_TYPES:
                continue
            # Keep the earliest event of each type.
            if message.type not in collected:
                collected[message.type] = message.copy(time=0)
        if len(collected) == len(_CONDUCTOR_META_TYPES):
            break
    # Stable order for readers.
    ordered = []
    for key in ("set_tempo", "time_signature", "key_signature", "smpte_offset"):
        if key in collected:
            ordered.append(collected[key])
    return ordered


def _track_program_and_drum(track) -> tuple[int, bool]:
    program = 0
    is_drum = False
    determined = False
    for message in track:
        if message.type == "program_change":
            program = int(message.program)
        if not determined and hasattr(message, "channel"):
            is_drum = message.channel == 9
            determined = True
    return program, is_drum


def _track_name(track) -> str | None:
    for message in track:
        if message.type == "track_name":
            name = " ".join(message.name.replace(",", " ").split())
            return name if name else None
    return None


def write_corrected_midi(
    src_mid: str | Path,
    dest_mid: str | Path,
    *,
    program_by_original_track: dict[int, int] | None = None,
    mid_rel: str | None = None,
) -> list[dict]:
    """Write a dense MIDI: note-bearing tracks only, register programs applied.

    Timing meta (tempo / time signature / key) is folded onto each kept track so
    single-track stem exports still have correct wall-clock duration. Empty
    grand-staff / conductor stubs are omitted.

    Returns sidecar rows (``TRACK_MAP_COLUMNS``) and writes
    ``{dest}.track_map.csv`` next to the MIDI.
    """
    src_mid = Path(src_mid)
    dest_mid = Path(dest_mid)
    program_by_original_track = program_by_original_track or {}
    rel = mid_rel if mid_rel is not None else str(src_mid)

    midi = mido.MidiFile(filename=str(src_mid), charset="utf8")
    meta_prefix = _conductor_meta_prefix(midi)
    out = mido.MidiFile(ticks_per_beat=midi.ticks_per_beat, charset="utf8")
    map_rows: list[dict] = []

    dense_idx = 0
    for original_idx, track in enumerate(midi.tracks):
        if note_on_count(track) == 0:
            continue
        program_orig, is_drum = _track_program_and_drum(track)
        program = int(program_by_original_track.get(original_idx, program_orig))
        name = _track_name(track)

        new_track = mido.MidiTrack()
        for meta in meta_prefix:
            new_track.append(meta.copy(time=0))
        for message in track:
            new_track.append(message.copy())
        apply_patch_to_midi_track(
            new_track,
            PatchAssignment(program=program, is_drum=is_drum),
        )
        out.tracks.append(new_track)
        map_rows.append({
            "mid": rel,
            "track": dense_idx,
            "original_track": original_idx,
            "program": program,
            "is_drum": bool(is_drum),
            "name": name,
        })
        dense_idx += 1

    if not out.tracks:
        # Degenerate all-empty file: keep a tempo-only stub so mido can save.
        stub = mido.MidiTrack()
        for meta in meta_prefix:
            stub.append(meta.copy(time=0))
        stub.append(mido.MetaMessage("end_of_track", time=0))
        out.tracks.append(stub)

    dest_mid.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(dest_mid))

    map_path = track_map_path_for_midi(dest_mid)
    pd.DataFrame(map_rows, columns=TRACK_MAP_COLUMNS).to_csv(map_path, index=False)
    return map_rows


def load_track_map(midi_path: str | Path) -> dict[int, dict]:
    """Map dense ``track`` → row dict from the sidecar next to ``midi_path``."""
    map_path = track_map_path_for_midi(midi_path)
    if not map_path.is_file():
        raise FileNotFoundError(
            f"Dense MIDI track map not found: {map_path}\n"
            "Re-run: uv run python -m analysis.prepare_synthesis --subset all_valid -j 8"
        )
    df = pd.read_csv(map_path)
    out: dict[int, dict] = {}
    for _, row in df.iterrows():
        out[int(row["track"])] = {
            "original_track": int(row["original_track"]),
            "program": int(row["program"]) if pd.notna(row.get("program")) else 0,
            "is_drum": bool(row.get("is_drum", False)),
            "name": None if pd.isna(row.get("name")) else str(row["name"]),
        }
    return out


def write_corrected_midis_from_register(
    register: pd.DataFrame,
    *,
    pdmx_root: str | Path,
    corrected_midi_dir: str | Path,
    jobs: int = 1,
) -> tuple[int, int]:
    """Write all corrected midis referenced by ``register``. Returns (ok, failed)."""
    from analysis.track_names import mid_path_for_row

    pdmx_root = Path(pdmx_root)
    corrected_midi_dir = Path(corrected_midi_dir)
    corrected_midi_dir.mkdir(parents=True, exist_ok=True)

    grouped = register.groupby("mid", sort=False)
    ok = 0
    failed = 0
    global_rows: list[dict] = []

    items = list(grouped)
    iterator = items
    if len(items) > 1:
        from tqdm import tqdm

        iterator = tqdm(items, total=len(items), desc="Writing corrected MIDI", unit="song")

    for mid_rel, group in iterator:
        src = mid_path_for_row(str(mid_rel), pdmx_root)
        if not src.is_file():
            failed += 1
            continue
        # Preserve relative layout under mid_corrected/ (strip leading ./).
        rel = str(mid_rel).lstrip("./")
        if rel.startswith("mid/"):
            dest_rel = rel[len("mid/"):]
        else:
            dest_rel = rel
        dest = corrected_midi_dir / dest_rel
        programs = {
            int(row["track"]): int(row["program_corrected"])
            for _, row in group.iterrows()
        }
        try:
            rows = write_corrected_midi(
                src,
                dest,
                program_by_original_track=programs,
                mid_rel=str(mid_rel),
            )
        except Exception:
            failed += 1
            continue
        global_rows.extend(rows)
        ok += 1

    if global_rows:
        pd.DataFrame(global_rows, columns=TRACK_MAP_COLUMNS).to_csv(
            corrected_midi_dir / "track_map.csv",
            index=False,
        )
    return ok, failed


def resolve_corrected_midi_path(
    pdmx_mid: str | Path,
    *,
    pdmx_root: str | Path,
    corrected_midi_dir: str | Path,
) -> Path:
    """Map a PDMX absolute/relative mid path to its corrected copy."""
    pdmx_mid = Path(pdmx_mid)
    pdmx_root = Path(pdmx_root)
    corrected_midi_dir = Path(corrected_midi_dir)
    try:
        rel = pdmx_mid.resolve().relative_to(pdmx_root.resolve())
    except ValueError:
        # Fall back: use filename layout mid/a/b/x.mid → a/b/x.mid
        parts = pdmx_mid.parts
        if "mid" in parts:
            idx = parts.index("mid")
            rel = Path(*parts[idx + 1 :])
        else:
            rel = Path(pdmx_mid.name)
    rel_s = str(rel).lstrip("./")
    if rel_s.startswith("mid/"):
        rel_s = rel_s[len("mid/"):]
    return corrected_midi_dir / rel_s
