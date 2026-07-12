"""Build a JSON-serializable catalog for preset-sweep experiment outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from experiments.preset_sweep.sweep import (
    DEFAULT_PROBE_STEMS,
    MANIFEST_FILENAME,
    VARIANTS_DIR_NAME,
    default_output_dir,
    default_source_dir,
)
from shared.config import DATA_DIR_NAME, DEFAULT_AUDIO_FORMAT, PROTOTYPE_AUDIO_FORMAT
from shared.repo_symlinks import REPO_PRESET_SWEEP_OUTPUT_SYMLINK
from synthesis.audio import stem_filename
from synthesis.listening.catalog import song_id_from_path


def default_preset_sweep_dir() -> Path:
    if REPO_PRESET_SWEEP_OUTPUT_SYMLINK.is_dir():
        return REPO_PRESET_SWEEP_OUTPUT_SYMLINK.resolve()
    return default_output_dir()


class PresetSweepCatalog:
    def __init__(
        self,
        sweep_dir: Path,
        source_dir: Path | None = None,
        probe_stems_path: Path = DEFAULT_PROBE_STEMS,
    ):
        self.sweep_dir = sweep_dir.resolve()
        self.source_dir = (source_dir or default_source_dir()).resolve()
        self.probe_stems_path = probe_stems_path
        self._manifest = self._load_manifest()
        self._probe_by_id = self._load_probe_index()
        self._audio_format = self._detect_audio_format()

    def _load_manifest(self) -> pd.DataFrame:
        path = self.sweep_dir / MANIFEST_FILENAME
        if not path.is_file():
            return pd.DataFrame()
        return pd.read_csv(path)

    def _load_probe_index(self) -> dict[str, dict]:
        with open(self.probe_stems_path) as f:
            cfg = yaml.safe_load(f)
        return {entry["id"]: entry for entry in cfg.get("stems", [])}

    def _detect_audio_format(self) -> str:
        if self._manifest.empty:
            return DEFAULT_AUDIO_FORMAT
        out_path = Path(self._manifest.iloc[0]["out_path"])
        if out_path.suffix.lower() == f".{PROTOTYPE_AUDIO_FORMAT}":
            return PROTOTYPE_AUDIO_FORMAT
        return DEFAULT_AUDIO_FORMAT

    def available(self) -> bool:
        return (self.sweep_dir / MANIFEST_FILENAME).is_file()

    def variants(self) -> list[dict]:
        if self._manifest.empty:
            return []
        rows = (
            self._manifest[
                ["variant_id", "init_noise_level", "prompt_variant"]
            ]
            .drop_duplicates()
            .sort_values("variant_id")
        )
        return rows.to_dict(orient="records")

    def list_stems(self) -> list[dict]:
        if self._manifest.empty:
            return []
        stems = []
        for stem_id, group in self._manifest.groupby("stem_id"):
            first = group.iloc[0]
            probe = self._probe_by_id.get(stem_id, {})
            song_id = song_id_from_path(first["path"])
            stems.append({
                "id": stem_id,
                "category": first.get("category") or probe.get("category"),
                "song_id": song_id,
                "track": int(first["track"]),
                "note": probe.get("note"),
            })
        stems.sort(key=lambda row: (row.get("category") or "", row["id"]))
        return stems

    def get_stem(self, stem_id: str) -> dict | None:
        if self._manifest.empty:
            return None
        group = self._manifest[self._manifest["stem_id"] == stem_id]
        if group.empty:
            return None

        first = group.iloc[0]
        track = int(first["track"])
        song_id = song_id_from_path(first["path"])
        filename = stem_filename(track, self._audio_format)
        probe = self._probe_by_id.get(stem_id, {})

        reference = self._reference_cell(stem_id, filename)
        variant_cells = []
        for _, row in group.sort_values("variant_id").iterrows():
            variant_cells.append({
                "variant_id": row["variant_id"],
                "init_noise_level": float(row["init_noise_level"]),
                "prompt_variant": row["prompt_variant"],
                "prompt": row["prompt"],
                "audio": self._variant_cell(
                    row["variant_id"],
                    song_id,
                    filename,
                ),
            })

        return {
            "id": stem_id,
            "category": first.get("category") or probe.get("category"),
            "song_id": song_id,
            "track": track,
            "note": probe.get("note"),
            "filename": filename,
            "reference": reference,
            "variants": variant_cells,
        }

    def _reference_cell(self, stem_id: str, filename: str) -> dict:
        audio_path = self.resolve_reference_audio(stem_id, filename)
        available = audio_path is not None
        return {
            "available": available,
            "url": (
                f"/audio/preset-sweep/reference/{stem_id}/{filename}"
                if available
                else None
            ),
        }

    def _variant_cell(self, variant_id: str, song_id: str, filename: str) -> dict:
        audio_path = self.resolve_variant_audio(variant_id, song_id, filename)
        available = audio_path is not None
        return {
            "available": available,
            "url": (
                f"/audio/preset-sweep/variant/{variant_id}/{song_id}/{filename}"
                if available
                else None
            ),
        }

    def resolve_reference_audio(self, stem_id: str, filename: str) -> Path | None:
        stem = self._probe_by_id.get(stem_id)
        if stem is None:
            return None
        if "/" in filename or "\\" in filename or ".." in Path(filename).parts:
            return None
        song_id = stem["song_id"]
        audio_path = (self.source_dir / DATA_DIR_NAME / song_id / filename).resolve()
        if not str(audio_path).startswith(str(self.source_dir.resolve())):
            return None
        return audio_path if audio_path.is_file() else None

    def resolve_variant_audio(
        self,
        variant_id: str,
        song_id: str,
        filename: str,
    ) -> Path | None:
        if ".." in Path(song_id).parts or ".." in Path(filename).parts:
            return None
        if "/" in filename or "\\" in filename:
            return None
        audio_path = (
            self.sweep_dir
            / VARIANTS_DIR_NAME
            / variant_id
            / DATA_DIR_NAME
            / song_id
            / filename
        ).resolve()
        if not str(audio_path).startswith(str(self.sweep_dir.resolve())):
            return None
        return audio_path if audio_path.is_file() else None
