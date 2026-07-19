"""Build JSON-serializable catalogs for patch and preset sweep outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from experiments.paths import (
    DEFAULT_PROBE_STEMS,
    patch_sweep_output_root,
    preset_sweep_output_root,
)
from experiments.patch_sweep.sweep import MANIFEST_FILENAME as PATCH_MANIFEST
from experiments.patch_sweep.sweep import VARIANTS_DIR_NAME as PATCH_VARIANTS_DIR
from experiments.preset_sweep.clips_dir import resolve_sweep_clips_dir
from experiments.preset_sweep.sweep import MANIFEST_FILENAME as PRESET_MANIFEST
from experiments.preset_sweep.sweep import VARIANTS_DIR_NAME as PRESET_VARIANTS_DIR
from shared.config import DATA_DIR_NAME, DEFAULT_AUDIO_FORMAT, OUTPUT_DIR, PROTOTYPE_AUDIO_FORMAT, STEMS_FILE_NAME
from shared.repo_symlinks import (
    REPO_PATCH_SWEEP_OUTPUT_SYMLINK,
    REPO_PRESET_SWEEP_OUTPUT_SYMLINK,
)
from synthesis.audio import stem_filename
from synthesis.listening.catalog import default_ablations_dir, song_id_from_path
from synthesis.paths import ablation_raw_dir


def default_source_dir(output_root: str = OUTPUT_DIR) -> Path:
    ablations = default_ablations_dir()
    basic = ablations / "basic"
    if basic.is_dir():
        return basic
    return Path(ablation_raw_dir(output_root, "basic"))


def default_sweep_dir(sweep_type: str, output_root: str = OUTPUT_DIR) -> Path:
    if sweep_type == "preset":
        if REPO_PRESET_SWEEP_OUTPUT_SYMLINK.is_dir():
            return REPO_PRESET_SWEEP_OUTPUT_SYMLINK.resolve()
        return Path(preset_sweep_output_root(output_root))
    if sweep_type == "patch":
        if REPO_PATCH_SWEEP_OUTPUT_SYMLINK.is_dir():
            return REPO_PATCH_SWEEP_OUTPUT_SYMLINK.resolve()
        return Path(patch_sweep_output_root(output_root))
    raise ValueError(f"Unknown sweep type: {sweep_type}")


def resolve_sweep_catalog_dir(
    sweep_type: str,
    sweep_dir: Path,
    *,
    prefer_verification_phase: bool = False,
) -> Path:
    """Pick a phased sweep output directory that actually has a manifest."""
    sweep_dir = sweep_dir.resolve()
    manifest_name = (
        PRESET_MANIFEST if sweep_type == "preset" else PATCH_MANIFEST
    )

    if prefer_verification_phase:
        try:
            from experiments.listening.final_verify import final_sweep_dir, readiness_errors

            if not readiness_errors(sweep_type):
                phase_dir = final_sweep_dir(sweep_type)
                if (phase_dir / manifest_name).is_file():
                    return phase_dir.resolve()
        except Exception:
            pass

    if (sweep_dir / manifest_name).is_file():
        return sweep_dir

    candidates = sorted(
        (
            path
            for path in sweep_dir.iterdir()
            if path.is_dir() and (path / manifest_name).is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()
    return sweep_dir


class SweepCatalog:
    def __init__(
        self,
        sweep_type: str,
        sweep_dir: Path,
        source_dir: Path | None = None,
        probe_stems_path: Path = DEFAULT_PROBE_STEMS,
    ):
        if sweep_type not in ("preset", "patch"):
            raise ValueError(f"Unknown sweep type: {sweep_type}")
        self.sweep_type = sweep_type
        self.sweep_dir = sweep_dir.resolve()
        self.source_dir = (source_dir or default_source_dir()).resolve()
        clips_dir = resolve_sweep_clips_dir(self.sweep_dir)
        if clips_dir is not None:
            self.source_dir = clips_dir
        self.probe_stems_path = probe_stems_path
        self._manifest = self._load_manifest()
        self._probe_by_id = self._load_probe_index()
        self._audio_format = self._detect_audio_format()

    @property
    def _variants_dir_name(self) -> str:
        return PRESET_VARIANTS_DIR if self.sweep_type == "preset" else PATCH_VARIANTS_DIR

    @property
    def _manifest_filename(self) -> str:
        return PRESET_MANIFEST if self.sweep_type == "preset" else PATCH_MANIFEST

    def manifest_id(self) -> str:
        path = self.sweep_dir / self._manifest_filename
        if not path.is_file():
            return "missing"
        stat = path.stat()
        return f"{stat.st_mtime_ns}_{stat.st_size}"

    def _load_manifest(self) -> pd.DataFrame:
        path = self.sweep_dir / self._manifest_filename
        if not path.is_file():
            return pd.DataFrame()
        df = pd.read_csv(path)
        return self._normalize_manifest(df)

    def _normalize_manifest(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        for col in (
            "phase",
            "variant_id",
            "soundfont_id",
            "soundfont_file",
            "fx_profile",
            "pool_id",
            "gm_class",
            "stem_id",
            "category",
            "path",
            "out_path",
            "prompt_variant",
            "prompt",
        ):
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
        if "program" in df.columns:
            df["program"] = pd.to_numeric(df["program"], errors="coerce")
        if "track" in df.columns:
            df["track"] = pd.to_numeric(df["track"], errors="coerce").astype("Int64")
        if "init_noise_level" in df.columns:
            df["init_noise_level"] = pd.to_numeric(df["init_noise_level"], errors="coerce")
        return df

    def _load_probe_index(self) -> dict[str, dict]:
        diverse_path = self.sweep_dir / "diverse_stems.yaml"
        if diverse_path.is_file():
            from experiments.preset_sweep.diverse_stems import load_diverse_stems_manifest

            return {entry["id"]: entry for entry in load_diverse_stems_manifest(diverse_path)}
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
        return (self.sweep_dir / self._manifest_filename).is_file()

    def variants(self) -> list[dict]:
        if self._manifest.empty:
            return []
        if self.sweep_type == "preset":
            rows = (
                self._manifest[
                    ["variant_id", "init_noise_level", "prompt_variant"]
                ]
                .drop_duplicates(subset=["variant_id"])
                .sort_values("variant_id")
            )
        else:
            cols = ["variant_id"]
            for c in ("pool_id", "soundfont_id", "fx_profile", "phase"):
                if c in self._manifest.columns:
                    cols.append(c)
            rows = (
                self._manifest[cols]
                .drop_duplicates(subset=["variant_id"])
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

    def get_stem_test(
        self,
        stem_id: str,
        *,
        session_seed: int,
        blinded: bool = True,
    ) -> dict | None:
        if self._manifest.empty:
            return None
        group = self._manifest[self._manifest["stem_id"] == stem_id]
        if group.empty:
            return None

        from experiments.listening.session import blinded_variant_order

        first = group.iloc[0]
        track = int(first["track"])
        song_id = song_id_from_path(first["path"])
        filename = self._reference_filename(stem_id, track)
        probe = self._probe_by_id.get(stem_id, {})

        variant_ids = group.sort_values("variant_id")["variant_id"].tolist()
        order = blinded_variant_order(
            variant_ids,
            stem_id=stem_id,
            session_seed=session_seed,
        )

        samples = []
        for blind_label, variant_id in order:
            row = group[group["variant_id"] == variant_id].iloc[0]
            sample = {
                "blind_label": blind_label,
                "audio": self._variant_cell(variant_id, song_id, filename),
            }
            if not blinded:
                sample["variant_id"] = variant_id
                if self.sweep_type == "preset":
                    sample["init_noise_level"] = float(row["init_noise_level"])
                    sample["prompt_variant"] = row["prompt_variant"]
                    sample["prompt"] = row.get("prompt")
                else:
                    if "pool_id" in row:
                        sample["pool_id"] = row["pool_id"]
                    if "soundfont_id" in row:
                        sample["soundfont_id"] = row["soundfont_id"]
                    if "fx_profile" in row:
                        sample["fx_profile"] = row["fx_profile"]
            else:
                sample["variant_id"] = variant_id
            samples.append(sample)

        return {
            "id": stem_id,
            "category": first.get("category") or probe.get("category"),
            "song_id": song_id,
            "track": track,
            "note": probe.get("note"),
            "filename": filename,
            "reference": self._reference_cell(stem_id, filename),
            "samples": samples,
        }

    def _reference_cell(self, stem_id: str, filename: str) -> dict:
        audio_path = self.resolve_reference_audio(stem_id, filename)
        available = audio_path is not None
        return {
            "available": available,
            "url": (
                f"/audio/{self.sweep_type}/reference/{stem_id}/{filename}"
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
                f"/audio/{self.sweep_type}/variant/{variant_id}/{song_id}/{filename}"
                if available
                else None
            ),
        }

    def _reference_filename(self, stem_id: str, track: int) -> str:
        probe = self._probe_by_id.get(stem_id, {})
        audio_format = str(probe.get("audio_format") or self._audio_format)
        return stem_filename(track, audio_format)

    def _manifest_reference_path(self, stem_id: str) -> Path | None:
        if self._manifest.empty:
            return None
        group = self._manifest[self._manifest["stem_id"] == stem_id]
        if group.empty:
            return None

        row = group.iloc[0]
        track = int(row["track"])
        song_dir = Path(str(row["path"])).resolve()
        filename = self._reference_filename(stem_id, track)
        audio_path = (song_dir / filename).resolve()

        allowed_roots = {
            self.source_dir.resolve(),
            self.sweep_dir.resolve(),
        }
        clips_dir = resolve_sweep_clips_dir(self.sweep_dir)
        if clips_dir is not None:
            allowed_roots.add(clips_dir.resolve())

        if not any(str(audio_path).startswith(str(root)) for root in allowed_roots):
            return None
        return audio_path if audio_path.is_file() else None

    def resolve_reference_audio(self, stem_id: str, filename: str) -> Path | None:
        stem = self._probe_by_id.get(stem_id)
        if stem is None and self._manifest.empty:
            return None
        if "/" in filename or "\\" in filename or ".." in Path(filename).parts:
            return None

        manifest_path = self._manifest_reference_path(stem_id)
        if manifest_path is not None:
            return manifest_path

        if stem is None:
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
            / self._variants_dir_name
            / variant_id
            / DATA_DIR_NAME
            / song_id
            / filename
        ).resolve()
        if not str(audio_path).startswith(str(self.sweep_dir.resolve())):
            return None
        return audio_path if audio_path.is_file() else None

    def responses_dir(self) -> Path:
        path = self.sweep_dir / "responses"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def session_responses_path(self) -> Path:
        return self.responses_dir() / "responses_in_progress.json"
