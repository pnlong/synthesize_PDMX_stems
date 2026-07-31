"""Correct GM program ids from MIDI track names for the sPDMX register."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mido
import pandas as pd
import yaml

from analysis.gm_programs import GM_PROGRAM_NAMES, gm_program_name
from shared.csv_tables import sanitize_track_name
from synthesis.patches import _gm_class, _normalize_name

DEFAULT_ALIASES_PATH = Path(__file__).resolve().parent / "gm_register_aliases.yaml"

STATUS_KEEP = "keep"
STATUS_CORRECTED = "corrected"
STATUS_SKIPPED_DRUM = "skipped_drum"
STATUS_SKIPPED_UNNAMED = "skipped_unnamed"
STATUS_SKIPPED_GENERIC = "skipped_generic"
STATUS_SKIPPED_AMBIGUOUS = "skipped_ambiguous"
STATUS_SKIPPED_NO_MATCH = "skipped_no_match"

REGISTER_COLUMNS = [
    "mid",
    "track",
    "name",
    "is_drum",
    "program_original",
    "program_corrected",
    "gm_name_original",
    "gm_name_corrected",
    "status",
    "match_key",
]

# Tokens extracted from GM names for secondary substring matching (longest first).
# ``music`` is excluded so ``MusicXML Part`` does not become Music Box.
_GM_NAME_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "new",
        "age",
        "music",
        "lead",
        "pad",
        "synth",
        "fx",
        "square",
        "warm",
    }
)

# Short needles use word boundaries so ``chor`` ⊄ ``harpsichord``, ``arpa`` ⊄ ``nyckelharpa``.
_WORD_BOUNDARY_NEEDLE_LEN = 5


def _fold_accents(text: str) -> str:
    """Lowercase ASCII-fold for multilingual matching (ténor → tenor, flöte → flote)."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c)).lower()


def _match_normalize(name: str | None) -> str:
    base = _normalize_name(name)
    return _fold_accents(base) if base else ""


@dataclass(frozen=True)
class AliasConfig:
    aliases: tuple[tuple[str, int], ...]  # (needle, program), longest first
    family_defaults: tuple[tuple[str, int, str], ...]  # needle, program, gm_class
    generic_skip: frozenset[str]


@dataclass(frozen=True)
class ResolveResult:
    program: int
    status: str
    match_key: str


def _needle_in_name(needle: str, name: str) -> bool:
    """True if needle appears in name; short needles require word-ish boundaries."""
    if not needle:
        return False
    if len(needle) < _WORD_BOUNDARY_NEEDLE_LEN:
        return bool(
            re.search(rf"(^|[^a-z0-9]){re.escape(needle)}([^a-z0-9]|$)", name)
        )
    return needle in name


@lru_cache(maxsize=1)
def _gm_name_needles() -> tuple[tuple[str, int], ...]:
    """Distinctive substrings from GM_PROGRAM_NAMES → program, longest first.

    Only needles that map to a single GM program are kept (avoids bare
    ``bass`` / ``lead`` / ``pad`` silently remapping to the first hit).
    """
    from collections import defaultdict

    needle_programs: dict[str, set[int]] = defaultdict(set)
    for program, label in enumerate(GM_PROGRAM_NAMES):
        full = _match_normalize(label)
        cleaned = re.sub(r"[()+,/]", " ", full)
        cleaned = " ".join(cleaned.split())
        if cleaned and len(cleaned) >= 4:
            needle_programs[cleaned].add(program)
        for token in cleaned.split():
            if token in _GM_NAME_STOPWORDS or len(token) < 4:
                continue
            needle_programs[token].add(program)

    pairs: list[tuple[str, int]] = []
    for needle, programs in needle_programs.items():
        if len(programs) == 1:
            pairs.append((needle, next(iter(programs))))
    pairs.sort(key=lambda item: (-len(item[0]), item[0], item[1]))
    return tuple(pairs)


@lru_cache(maxsize=4)
def load_alias_config(path: str | Path | None = None) -> AliasConfig:
    aliases_path = Path(path) if path is not None else DEFAULT_ALIASES_PATH
    with open(aliases_path) as f:
        raw = yaml.safe_load(f) or {}

    aliases: list[tuple[str, int]] = []
    for entry in raw.get("aliases") or []:
        needle = _match_normalize(entry.get("needle"))
        program = int(entry["program"])
        if needle:
            aliases.append((needle, program))
    aliases.sort(key=lambda item: (-len(item[0]), item[0], item[1]))

    family: list[tuple[str, int, str]] = []
    for entry in raw.get("family_defaults") or []:
        needle = _match_normalize(entry.get("needle"))
        if not needle:
            continue
        family.append((needle, int(entry["program"]), str(entry["gm_class"])))
    family.sort(key=lambda item: (-len(item[0]), item[0]))

    skip = frozenset(
        _match_normalize(x) for x in (raw.get("generic_skip") or []) if _match_normalize(x)
    )
    return AliasConfig(tuple(aliases), tuple(family), skip)


def _is_generic_name(name: str, config: AliasConfig) -> bool:
    if not name or name == "(unnamed)":
        return True
    if name in config.generic_skip:
        return True
    if re.fullmatch(r"(track|part|staff|channel)[\s_\-]*\d+", name):
        return True
    if re.fullmatch(r"\d+", name):
        return True
    for skip in config.generic_skip:
        if " " in skip and name == skip:
            return True
    # Export / tooling labels that are not instruments.
    if "musicxml" in name:
        return True
    return False


def _current_agrees_with_name(name: str, program: int) -> bool:
    """True if the track name already looks like the current GM program."""
    gm_name = _match_normalize(gm_program_name(program))
    if not gm_name:
        return False
    if gm_name in name or name in gm_name:
        return True
    gm_tokens = [
        t
        for t in re.sub(r"[()+,/]", " ", gm_name).split()
        if t not in _GM_NAME_STOPWORDS and len(t) >= 4
    ]
    return any(_needle_in_name(t, name) for t in gm_tokens)


def _collect_needle_hits(
    name: str,
    needles: tuple[tuple[str, int], ...],
) -> list[tuple[str, int]]:
    """Return all (needle, program) hits at the maximum needle length."""
    hits: list[tuple[str, int]] = []
    best_len = 0
    for needle, program in needles:
        if len(needle) < best_len:
            break
        if _needle_in_name(needle, name):
            if len(needle) > best_len:
                best_len = len(needle)
                hits = [(needle, program)]
            elif len(needle) == best_len:
                hits.append((needle, program))
    return hits


def resolve_program(
    *,
    track_name: str | None,
    program: int,
    is_drum: bool,
    config: AliasConfig | None = None,
) -> ResolveResult:
    """Decide whether to keep or correct a track's GM program from its name."""
    if config is None:
        config = load_alias_config()
    original = int(program)

    if is_drum:
        return ResolveResult(original, STATUS_SKIPPED_DRUM, "")

    cleaned = sanitize_track_name(track_name)
    name = _match_normalize(cleaned) if cleaned else ""
    if not name:
        return ResolveResult(original, STATUS_SKIPPED_UNNAMED, "")

    if _is_generic_name(name, config):
        return ResolveResult(original, STATUS_SKIPPED_GENERIC, "")

    if _current_agrees_with_name(name, original):
        return ResolveResult(original, STATUS_KEEP, "")

    # Primary: curated aliases (longest needle). Specific program targets
    # correct even within the same GM class (e.g. piano 0 → harpsichord 6).
    alias_hits = _collect_needle_hits(name, config.aliases)
    if alias_hits:
        programs = {p for _, p in alias_hits}
        if len(programs) > 1:
            return ResolveResult(original, STATUS_SKIPPED_AMBIGUOUS, alias_hits[0][0])
        needle, candidate = alias_hits[0]
        if candidate == original:
            return ResolveResult(original, STATUS_KEEP, needle)
        return ResolveResult(candidate, STATUS_CORRECTED, needle)

    # Family defaults: only when current program is outside that class.
    family_hits = _collect_needle_hits(
        name,
        tuple((n, p) for n, p, _ in config.family_defaults),
    )
    if family_hits:
        # Map needle → gm_class from config
        class_by_needle = {n: c for n, _, c in config.family_defaults}
        programs = {p for _, p in family_hits}
        if len(programs) > 1:
            return ResolveResult(original, STATUS_SKIPPED_AMBIGUOUS, family_hits[0][0])
        needle, candidate = family_hits[0]
        target_class = class_by_needle.get(needle)
        if target_class and _gm_class(original, False) == target_class:
            return ResolveResult(original, STATUS_KEEP, needle)
        if candidate == original:
            return ResolveResult(original, STATUS_KEEP, needle)
        return ResolveResult(candidate, STATUS_CORRECTED, needle)

    # Secondary: GM program-name substrings (specific instrument names).
    gm_hits = _collect_needle_hits(name, _gm_name_needles())
    if gm_hits:
        programs = {p for _, p in gm_hits}
        if len(programs) > 1:
            return ResolveResult(original, STATUS_SKIPPED_AMBIGUOUS, gm_hits[0][0])
        needle, candidate = gm_hits[0]
        if candidate == original:
            return ResolveResult(original, STATUS_KEEP, needle)
        return ResolveResult(candidate, STATUS_CORRECTED, needle)

    return ResolveResult(original, STATUS_SKIPPED_NO_MATCH, "")


def extract_register_rows_from_mid(
    mid_path: str | Path,
    *,
    mid_rel: str,
    config: AliasConfig | None = None,
) -> list[dict] | None:
    """One register row per MIDI track index (including empty tracks)."""
    if config is None:
        config = load_alias_config()
    try:
        midi = mido.MidiFile(filename=str(mid_path), charset="utf8")
    except Exception:
        return None

    rows: list[dict] = []
    for j, track in enumerate(midi.tracks):
        program = 0
        is_drum = False
        track_name: str | None = None
        determined_whether_track_is_drum = False

        for message in track:
            if message.type == "program_change":
                program = message.program
            elif message.type == "track_name":
                track_name = sanitize_track_name(
                    " ".join(message.name.replace(",", " ").split())
                )
            if not determined_whether_track_is_drum and hasattr(message, "channel"):
                is_drum = message.channel == 9
                determined_whether_track_is_drum = True

        result = resolve_program(
            track_name=track_name,
            program=program,
            is_drum=is_drum,
            config=config,
        )
        rows.append(
            {
                "mid": mid_rel,
                "track": j,
                "name": track_name if track_name else None,
                "is_drum": bool(is_drum),
                "program_original": int(program),
                "program_corrected": int(result.program),
                "gm_name_original": gm_program_name(program),
                "gm_name_corrected": gm_program_name(result.program),
                "status": result.status,
                "match_key": result.match_key or None,
            }
        )
    return rows


def register_dataframe(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=REGISTER_COLUMNS)
    return pd.DataFrame(rows)[REGISTER_COLUMNS]


def build_register_report(
    register_df: pd.DataFrame,
    *,
    subset: str | None = None,
    n_songs: int | None = None,
    n_songs_failed: int = 0,
    top_n: int = 20,
) -> dict:
    """Aggregate corrected-vs-fine-as-is stats and top remaps."""
    df = register_df.copy()
    n_tracks = len(df)
    n_corrected = int((df["status"] == STATUS_CORRECTED).sum()) if n_tracks else 0
    n_fine = n_tracks - n_corrected
    pct = lambda n: round(100.0 * n / n_tracks, 2) if n_tracks else 0.0

    status_counts = (
        df["status"].value_counts().to_dict() if n_tracks else {}
    )
    status_counts = {str(k): int(v) for k, v in status_counts.items()}

    corrected = df[df["status"] == STATUS_CORRECTED] if n_tracks else df.iloc[0:0]
    top_corrections: list[dict] = []
    if len(corrected):
        grouped = (
            corrected.groupby(
                ["program_original", "program_corrected", "gm_name_original", "gm_name_corrected"],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        for _, row in grouped.head(top_n).iterrows():
            top_corrections.append(
                {
                    "program_original": int(row["program_original"]),
                    "program_corrected": int(row["program_corrected"]),
                    "gm_name_original": row["gm_name_original"],
                    "gm_name_corrected": row["gm_name_corrected"],
                    "count": int(row["count"]),
                    "share_of_corrections": round(100.0 * row["count"] / n_corrected, 2),
                    "label": (
                        f"{int(row['program_original'])} {row['gm_name_original']} → "
                        f"{int(row['program_corrected'])} {row['gm_name_corrected']}"
                    ),
                }
            )

    top_match_keys: list[dict] = []
    if len(corrected) and "match_key" in corrected.columns:
        mk = corrected["match_key"].dropna()
        if len(mk):
            for key, count in mk.value_counts().head(top_n).items():
                top_match_keys.append({"match_key": str(key), "count": int(count)})

    top_names: list[dict] = []
    if len(corrected) and "name" in corrected.columns:
        names = corrected["name"].dropna()
        if len(names):
            for name, count in names.value_counts().head(top_n).items():
                top_names.append({"name": str(name), "count": int(count)})

    top_original_programs: list[dict] = []
    if len(corrected):
        for prog, count in corrected["program_original"].value_counts().head(top_n).items():
            top_original_programs.append(
                {
                    "program_original": int(prog),
                    "gm_name_original": gm_program_name(int(prog)),
                    "count": int(count),
                }
            )

    # Named non-drum denominator (tracks that could plausibly be corrected).
    eligible = df[
        (~df["is_drum"].astype(bool))
        & (df["status"] != STATUS_SKIPPED_UNNAMED)
        & (df["status"] != STATUS_SKIPPED_GENERIC)
    ] if n_tracks else df.iloc[0:0]
    n_eligible = len(eligible)
    n_corrected_eligible = int((eligible["status"] == STATUS_CORRECTED).sum()) if n_eligible else 0

    return {
        "subset": subset,
        "n_songs": n_songs,
        "n_songs_failed": int(n_songs_failed),
        "n_tracks": n_tracks,
        "n_corrected": n_corrected,
        "pct_corrected": pct(n_corrected),
        "n_fine_as_is": n_fine,
        "pct_fine_as_is": pct(n_fine),
        "n_eligible_named_nondrum": n_eligible,
        "n_corrected_among_eligible": n_corrected_eligible,
        "pct_corrected_among_eligible": (
            round(100.0 * n_corrected_eligible / n_eligible, 2) if n_eligible else 0.0
        ),
        "status_counts": status_counts,
        "top_corrections": top_corrections,
        "top_match_keys": top_match_keys,
        "top_corrected_names": top_names,
        "top_original_programs_corrected": top_original_programs,
    }


def format_register_report(report: dict) -> str:
    """Human-readable summary (same text printed to the console)."""
    lines: list[str] = []
    subset = report.get("subset") or "?"
    n_songs = report.get("n_songs")
    songs_bit = f", {n_songs} songs" if n_songs is not None else ""
    failed = report.get("n_songs_failed") or 0
    failed_bit = f", {failed} failed" if failed else ""
    lines.append(f"GM register summary ({subset}{songs_bit}{failed_bit})")
    lines.append(f"  tracks: {report['n_tracks']}")
    lines.append(
        f"  corrected: {report['n_corrected']} ({report['pct_corrected']}%)"
    )
    lines.append(
        f"  fine as-is: {report['n_fine_as_is']} ({report['pct_fine_as_is']}%)"
    )
    lines.append(
        f"  among named non-drum: "
        f"{report['n_corrected_among_eligible']}/{report['n_eligible_named_nondrum']} "
        f"({report['pct_corrected_among_eligible']}%)"
    )
    status = report.get("status_counts") or {}
    if status:
        parts = " ".join(f"{k}={v}" for k, v in sorted(status.items()))
        lines.append(f"  by status: {parts}")
    top = report.get("top_corrections") or []
    if top:
        lines.append("  top corrections:")
        for i, row in enumerate(top[:15], start=1):
            lines.append(f"    {i}. {row['label']}  ({row['count']})")
    keys = report.get("top_match_keys") or []
    if keys:
        joined = ", ".join(f"{k['match_key']} ({k['count']})" for k in keys[:12])
        lines.append(f"  top match keys: {joined}")
    return "\n".join(lines) + "\n"


def print_register_report(report: dict) -> None:
    print(format_register_report(report), end="")


def top_corrections_dataframe(report: dict) -> pd.DataFrame:
    rows = report.get("top_corrections") or []
    if not rows:
        return pd.DataFrame(
            columns=[
                "program_original",
                "gm_name_original",
                "program_corrected",
                "gm_name_corrected",
                "count",
                "share_of_corrections",
                "label",
            ]
        )
    return pd.DataFrame(rows)


def load_register_lookup(
    register_path: str | Path,
    *,
    pdmx_root: str | Path | None = None,
) -> dict[tuple[str, int], int]:
    """Map (mid_key, track) → program_corrected.

    Keys include the CSV ``mid`` relative path and, when ``pdmx_root`` is set,
    the absolute resolved MIDI path string used by synthesize.
    """
    path = Path(register_path)
    if not path.is_file():
        raise FileNotFoundError(f"GM register not found: {path}")
    df = pd.read_csv(path)
    lookup: dict[tuple[str, int], int] = {}
    root = Path(pdmx_root) if pdmx_root is not None else None
    for _, row in df.iterrows():
        mid = str(row["mid"])
        track = int(row["track"])
        program = int(row["program_corrected"])
        lookup[(mid, track)] = program
        if root is not None:
            from analysis.track_names import mid_path_for_row

            abs_path = str(mid_path_for_row(mid, root))
            lookup[(abs_path, track)] = program
    return lookup


def lookup_corrected_program(
    lookup: dict[tuple[str, int], int] | None,
    *,
    mid: str,
    track: int,
    default: int,
) -> int:
    if not lookup:
        return default
    if (mid, track) in lookup:
        return lookup[(mid, track)]
    return default
