"""PDMX dataset filtering and sampling for synthesis."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from shared.config import (
    ABLATION_MIN_STEMS_PER_CATEGORY,
    ABLATION_SAMPLE_SEED,
    ABLATION_SAMPLE_SIZE,
    ABLATION_SUBSET_COLUMN,
    LISTENING_SAMPLE_FILE_NAME,
)
from synthesis.patches import LISTENING_CATEGORY_GM_CLASSES, resolve_probe_category
from synthesis.paths import ablations_root


def listening_sample_path(output_dir: str) -> Path:
    return Path(ablations_root(output_dir)) / LISTENING_SAMPLE_FILE_NAME


def prepare_full_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """All valid PDMX rows (caller should filter ``subset:all_valid`` first)."""
    return dataset.reset_index(drop=True)


def prepare_ablation_dataset(
    dataset: pd.DataFrame,
    sample_size: int = ABLATION_SAMPLE_SIZE,
    sample_seed: int = ABLATION_SAMPLE_SEED,
    subset_column: str = ABLATION_SUBSET_COLUMN,
    *,
    min_stems_per_category: int = ABLATION_MIN_STEMS_PER_CATEGORY,
    register_df: pd.DataFrame | None = None,
    listening_sample_file: Path | str | None = None,
    persist_sample: bool = True,
) -> pd.DataFrame:
    """Category-stratified sample from ``subset:rated_deduplicated`` for listening ablations.

    If ``listening_sample_file`` exists, reload that exact song set (shared across modes).
    Otherwise fill until each listening category has ``min_stems_per_category`` stems
    (via GM register metadata), capped at ``sample_size`` songs, and optionally persist.
    Falls back to a plain random sample when no register is provided.
    """
    if subset_column not in dataset.columns:
        raise KeyError(f"Missing subset column {subset_column!r}")
    dataset = dataset[dataset[subset_column]].reset_index(drop=True)

    sample_file = Path(listening_sample_file) if listening_sample_file else None
    if sample_file is not None and sample_file.is_file():
        return _filter_to_listening_sample(dataset, sample_file)

    if register_df is None or register_df.empty:
        if len(dataset) > sample_size:
            dataset = dataset.sample(n=sample_size, random_state=sample_seed).reset_index(drop=True)
        return dataset

    selected, stem_inventory = stratified_song_sample(
        dataset,
        register_df,
        min_stems_per_category=min_stems_per_category,
        max_songs=sample_size,
        sample_seed=sample_seed,
    )
    if persist_sample and sample_file is not None:
        write_listening_sample(
            sample_file,
            selected,
            stem_inventory,
            sample_seed=sample_seed,
            min_stems_per_category=min_stems_per_category,
            max_songs=sample_size,
        )
    return selected.reset_index(drop=True)


def stratified_song_sample(
    dataset: pd.DataFrame,
    register_df: pd.DataFrame,
    *,
    min_stems_per_category: int = ABLATION_MIN_STEMS_PER_CATEGORY,
    max_songs: int = ABLATION_SAMPLE_SIZE,
    sample_seed: int = ABLATION_SAMPLE_SEED,
) -> tuple[pd.DataFrame, list[dict]]:
    """Keep songs that fill listening-category stem quotas (metadata only)."""
    categories = tuple(LISTENING_CATEGORY_GM_CLASSES.keys())
    counts: dict[str, int] = {cat: 0 for cat in categories}
    by_mid = _stems_by_mid(register_df)
    # Only require quotas for categories that exist somewhere in the register pool.
    available_categories = {
        stem["category"]
        for stems in by_mid.values()
        for stem in stems
        if stem["category"] in counts
    }

    shuffled = dataset.sample(frac=1.0, random_state=sample_seed).reset_index(drop=True)
    keep_indices: list[int] = []
    stem_inventory: list[dict] = []

    for idx, row in shuffled.iterrows():
        if len(keep_indices) >= max_songs:
            break
        if available_categories and all(
            counts[c] >= min_stems_per_category for c in available_categories
        ):
            break

        mid_keys = _mid_lookup_keys(row)
        stems = None
        for key in mid_keys:
            if key in by_mid:
                stems = by_mid[key]
                break
        if not stems:
            continue

        contributes = False
        song_stem_rows: list[dict] = []
        song_id = _song_id_from_pdmx_path(str(row["path"]))
        for stem in stems:
            cat = stem["category"]
            if cat in counts and counts[cat] < min_stems_per_category:
                contributes = True
            song_stem_rows.append({
                "path": str(row["path"]),
                "mid": str(row["mid"]) if "mid" in row else mid_keys[0],
                "song_id": song_id,
                "track": stem["track"],
                "program": stem["program"],
                "is_drum": stem["is_drum"],
                "name": stem["name"],
                "category": cat,
            })
        if not contributes and keep_indices:
            # Still allow early songs; after we have some, only keep fillers.
            # First song always kept if it has any categorized stems.
            continue
        if not song_stem_rows:
            continue

        keep_indices.append(idx)
        for stem_row in song_stem_rows:
            cat = stem_row["category"]
            if cat in counts:
                counts[cat] += 1
            stem_inventory.append(stem_row)

    if not keep_indices:
        # Degenerate register join — fall back to random sample.
        fallback = shuffled.head(min(max_songs, len(shuffled))).reset_index(drop=True)
        return fallback, []

    selected = shuffled.loc[keep_indices].reset_index(drop=True)
    return selected, stem_inventory


def write_listening_sample(
    path: Path | str,
    songs: pd.DataFrame,
    stem_inventory: list[dict],
    *,
    sample_seed: int,
    min_stems_per_category: int,
    max_songs: int,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_seed": int(sample_seed),
        "min_stems_per_category": int(min_stems_per_category),
        "max_songs": int(max_songs),
        "n_songs": int(len(songs)),
        "songs": [
            {
                "path": str(row["path"]),
                "mid": str(row["mid"]) if "mid" in row and pd.notna(row["mid"]) else None,
            }
            for _, row in songs.iterrows()
        ],
        "stems": stem_inventory,
    }
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    return path


def load_listening_sample(path: Path | str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _filter_to_listening_sample(dataset: pd.DataFrame, sample_file: Path) -> pd.DataFrame:
    doc = load_listening_sample(sample_file)
    paths = {str(s["path"]) for s in doc.get("songs") or [] if s.get("path")}
    if not paths:
        raise ValueError(f"Listening sample has no songs: {sample_file}")
    filtered = dataset[dataset["path"].astype(str).isin(paths)].reset_index(drop=True)
    if len(filtered) == 0:
        raise ValueError(
            f"No PDMX rows matched listening sample paths in {sample_file}"
        )
    # Preserve sample order.
    order = {
        str(s["path"]): i
        for i, s in enumerate(doc.get("songs") or [])
        if s.get("path")
    }
    filtered = filtered.sort_values(
        by="path", key=lambda col: col.map(lambda p: order.get(str(p), 10**9))
    ).reset_index(drop=True)
    return filtered


def _song_id_from_pdmx_path(path: str) -> str:
    """Derive ``7/19/Qm…`` style id from a PDMX ``path`` / ``mid`` value."""
    text = path.replace("\\", "/").lstrip("./")
    if text.startswith("data/"):
        text = text[len("data/") :]
    # Drop file extension.
    if "." in Path(text).name:
        text = str(Path(text).with_suffix(""))
    return text


def _mid_lookup_keys(row: pd.Series) -> list[str]:
    keys: list[str] = []
    if "mid" in row and pd.notna(row["mid"]):
        keys.append(str(row["mid"]))
    if "path" in row and pd.notna(row["path"]):
        path = str(row["path"])
        keys.append(path)
        # Common PDMX layout: path is .json, mid is .mid with same stem.
        if path.endswith(".json"):
            keys.append(path[: -len(".json")] + ".mid")
    # Dedupe preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _stems_by_mid(register_df: pd.DataFrame) -> dict[str, list[dict]]:
    by_mid: dict[str, list[dict]] = defaultdict(list)
    for _, row in register_df.iterrows():
        mid = str(row["mid"])
        program = int(row["program_corrected"])
        is_drum = bool(row["is_drum"]) if "is_drum" in row and pd.notna(row["is_drum"]) else False
        name = None
        if "name" in row and pd.notna(row["name"]):
            text = str(row["name"]).strip()
            name = text if text and text != "NA" else None
        category = resolve_probe_category(program=program, is_drum=is_drum, track_name=name)
        by_mid[mid].append({
            "track": int(row["track"]),
            "program": program,
            "is_drum": is_drum,
            "name": name,
            "category": category,
        })
    return dict(by_mid)
