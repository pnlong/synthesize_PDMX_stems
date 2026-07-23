"""Catalog for model listening test trials."""

from __future__ import annotations

from pathlib import Path

import yaml

from experiments.ablation_listening.session import blinded_condition_order, trial_order
from experiments.model_listening.paths import DEFAULT_CLIPS_DIR, DEFAULT_MANIFEST, DEFAULT_RESPONSES_DIR


class ModelListeningCatalog:
    """Same trial/blinding pattern as ablation listening, different condition IDs."""

    def __init__(
        self,
        manifest_path: Path = DEFAULT_MANIFEST,
        clips_dir: Path = DEFAULT_CLIPS_DIR,
    ):
        self.manifest_path = manifest_path.resolve()
        self.clips_dir = clips_dir.resolve()
        with open(self.manifest_path) as f:
            self._doc = yaml.safe_load(f) or {}
        self.test_id = str(self._doc.get("test_id") or "model_listening_v1")
        self.trials = list(self._doc.get("trials") or [])
        self.trial_by_id = {trial["id"]: trial for trial in self.trials}
        self.condition_ids = self._condition_ids()

    def _condition_ids(self) -> list[str]:
        ids: list[str] = []
        for model in self._doc.get("models") or []:
            if isinstance(model, dict) and model.get("id"):
                ids.append(str(model["id"]))
            elif isinstance(model, str):
                ids.append(model)
        if ids:
            return ids
        for trial in self.trials:
            for key in (trial.get("conditions") or {}):
                if key not in ids:
                    ids.append(key)
        return ids

    def meta(self, session_seed: int) -> dict:
        trial_ids = trial_order([trial["id"] for trial in self.trials], session_seed)
        return {
            "test_id": self.test_id,
            "session_seed": session_seed,
            "n_trials": len(trial_ids),
            "trial_order": trial_ids,
            "models": self.condition_ids,
            "rubrics": {
                "content": {
                    "label": "Content",
                    "help": "Does the output match the requested musical content?",
                },
                "realism": {
                    "label": "Realism",
                    "help": "Sounds like a realistic recording?",
                },
            },
            "scale": {"min": 0, "max": 100},
        }

    def get_trial(self, trial_id: str, session_seed: int) -> dict | None:
        trial = self.trial_by_id.get(trial_id)
        if trial is None or not self.condition_ids:
            return None

        ordered = blinded_condition_order(
            self.condition_ids,
            trial_id=trial_id,
            session_seed=session_seed,
        )
        audio_format = trial.get("audio_format") or "mp3"
        samples = []
        for blind_label, condition_id in ordered:
            rel_path = (trial.get("conditions") or {}).get(condition_id)
            audio_path = self.clips_dir / rel_path if rel_path else None
            samples.append({
                "blind_label": blind_label,
                "condition_id": condition_id,
                "available": bool(audio_path and audio_path.is_file()),
                "url": f"/audio/{trial_id}/{condition_id}.{audio_format}",
            })

        return {
            "id": trial_id,
            "type": trial.get("type"),
            "note": trial.get("note"),
            "clip_seconds": trial.get("clip_seconds"),
            "samples": samples,
            "realism_rubric": {
                "label": "Realism",
                "help": "Sounds like a realistic recording?",
            },
        }

    def resolve_audio_path(self, trial_id: str, filename: str) -> Path | None:
        if ".." in Path(filename).parts or "/" in filename or "\\" in filename:
            return None
        trial = self.trial_by_id.get(trial_id)
        if trial is None:
            return None
        condition_id = Path(filename).stem
        rel_path = (trial.get("conditions") or {}).get(condition_id)
        if not rel_path:
            return None
        audio_path = (self.clips_dir / rel_path).resolve()
        if not str(audio_path).startswith(str(self.clips_dir)):
            return None
        return audio_path if audio_path.is_file() else None

    def responses_dir(self) -> Path:
        return DEFAULT_RESPONSES_DIR

    def session_responses_path(self) -> Path:
        out = self.responses_dir()
        out.mkdir(parents=True, exist_ok=True)
        return out / "responses_in_progress.json"
