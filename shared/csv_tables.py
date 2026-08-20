"""Helpers for writing PDMX metadata CSV tables."""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from os.path import exists
from pathlib import Path

import pandas as pd

from shared.config import NA_STRING


def sanitize_track_name(name: str | None) -> str | None:
    """Remove characters that break CSV export (e.g. null bytes in PDMX MIDI track names)."""
    if name is None:
        return None
    cleaned = name.replace("\x00", "").replace(",", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned or None


@contextmanager
def _csv_exclusive_lock(csv_path: str):
    """Serialize read-modify-write so Fluidsynth and DDSP can upsert in parallel."""
    lock_path = Path(str(csv_path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def append_rows_deduped(
    csv_path: str,
    columns: list[str],
    new_rows: list[dict],
    *,
    key_col: str = "path",
    key_cols: list[str] | None = None,
) -> None:
    """Append rows to a CSV, replacing any existing rows with the same key value(s)."""
    if not new_rows:
        return

    cols = list(key_cols) if key_cols else [key_col]
    with _csv_exclusive_lock(csv_path):
        new_df = pd.DataFrame(new_rows, columns=columns)
        if exists(csv_path):
            existing = pd.read_csv(csv_path, sep=",", header=0, index_col=False)
            if len(existing) > 0:
                new_keys = _row_key_set(new_df, cols)
                keep = [_row_key(row, cols) not in new_keys for row in existing.to_dict("records")]
                existing = existing[keep]
                new_df = pd.concat([existing, new_df], ignore_index=True)

        new_df.to_csv(
            csv_path,
            sep=",",
            na_rep=NA_STRING,
            header=True,
            index=False,
            mode="w",
        )


def _row_key(row: dict, cols: list[str]) -> tuple:
    values = []
    for col in cols:
        value = row[col]
        if col == "track":
            values.append(int(value))
        else:
            values.append(str(value))
    return tuple(values)


def _row_key_set(df: pd.DataFrame, cols: list[str]) -> set[tuple]:
    return {_row_key(row, cols) for row in df.to_dict("records")}
