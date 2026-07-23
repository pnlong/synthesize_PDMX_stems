"""Catalog for ablation listening test trials and audio."""

from __future__ import annotations

from pathlib import Path

import yaml

from experiments.ablation_listening.paths import (
    DEFAULT_CLIPS_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_RESPONSES_DIR,
)
from experiments.ablation_listening.session import (
    REFERENCE_CONDITION,
    blinded_condition_order,
    realism_rubric,
    trial_order,
    variant_condition_ids,
)
from synthesis.listening.catalog import CONDITION_LABELS


class AblationListeningCatalog:
    def __init__(
        self,
        manifest_path: Path = DEFAULT_MANIFEST,
        clips_dir: Path = DEFAULT_CLIPS_DIR,
    ):
        self.manifest_path = manifest_path.resolve()
        self.clips_dir = clips_dir.resolve()
        with open(self.manifest_path) as f:
            self._doc = yaml.safe_load(f) or {}
        self.test_id = str(self._doc.get("test_id") or "ablation_listening_v1")
        self.trials = list(self._doc.get("trials") or [])
        self.trial_by_id = {trial["id"]: trial for trial in self.trials}

    def meta(self, session_seed: int) -> dict:
        trial_ids = trial_order([trial["id"] for trial in self.trials], session_seed)
        return {
            "test_id": self.test_id,
            "session_seed": session_seed,
            "n_trials": len(trial_ids),
            "trial_order": trial_ids,
            "rubrics": {
                "content": {
                    "label": "Content",
                    "help": "Same melody, rhythm, and timing as the reference (A1)?",
                },
                "reference": {
                    "label": "Reference (A1)",
                    "help": "Basic Fluidsynth synthesis — content ground truth; rate realism only.",
                },
                "realism_stem": realism_rubric("stem"),
                "realism_mix": realism_rubric("mixture"),
            },
            "scale": {
                "min": 0,
                "max": 100,
                "bands": [
                    {"range": "0–20", "content": "Very different", "realism": "Very synthetic"},
                    {"range": "20–40", "content": "Different", "realism": "Synthetic"},
                    {"range": "40–60", "content": "Mostly same", "realism": "Mixed"},
                    {"range": "60–80", "content": "Same", "realism": "Realistic"},
                    {"range": "80–100", "content": "Identical", "realism": "Very realistic"},
                ],
            },
        }

    def get_trial(self, trial_id: str, session_seed: int) -> dict | None:
        trial = self.trial_by_id.get(trial_id)
        if trial is None:
            return None

        audio_format = trial.get("audio_format") or "mp3"
        conditions = trial.get("conditions") or {}

        ref_path = conditions.get(REFERENCE_CONDITION)
        if not ref_path:
            return None
        ref_audio = self.clips_dir / ref_path
        reference = {
            "condition_id": REFERENCE_CONDITION,
            "condition_label": CONDITION_LABELS.get(REFERENCE_CONDITION, "A1"),
            "available": ref_audio.is_file(),
            "url": f"/audio/{trial_id}/{REFERENCE_CONDITION}.{audio_format}",
        }

        ordered = blinded_condition_order(
            variant_condition_ids(),
            trial_id=trial_id,
            session_seed=session_seed,
        )
        samples = []
        for blind_label, condition_id in ordered:
            rel_path = conditions.get(condition_id)
            if not rel_path:
                return None
            audio_path = self.clips_dir / rel_path
            samples.append({
                "blind_label": blind_label,
                "condition_id": condition_id,
                "available": audio_path.is_file(),
                "url": f"/audio/{trial_id}/{condition_id}.{audio_format}",
            })

        return {
            "id": trial_id,
            "type": trial.get("type"),
            "category": trial.get("category"),
            "song_id": trial.get("song_id"),
            "track": trial.get("track"),
            "note": trial.get("note"),
            "clip_seconds": trial.get("clip_seconds"),
            "reference": reference,
            "samples": samples,
            "realism_rubric": realism_rubric(str(trial.get("type") or "stem")),
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
        if (self.manifest_path.parent / "output" / "responses").is_dir():
            return self.manifest_path.parent / "output" / "responses"
        return DEFAULT_RESPONSES_DIR

    def session_responses_path(self) -> Path:
        out = self.responses_dir()
        out.mkdir(parents=True, exist_ok=True)
        return out / "responses_in_progress.json"
