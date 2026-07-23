"""webMUSHRA config generation and WAV export for ablation listening."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import soundfile as sf
import yaml

from experiments.ablation_listening.paths import (
    DEFAULT_CLIPS_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_WEBMUSHRA_ROOT,
    WEBMUSHRA_CONFIG_NAME,
    WEBMUSHRA_STIMULI_DIR,
    WEBMUSHRA_TEST_ID,
)
from synthesis.listening.catalog import CONDITION_LABELS, CONDITION_ORDER

REFERENCE_CONDITION = "basic"

MUSHRA_INTRO_HTML = """
<p><strong>sPDMX ablation listening test.</strong> Use wired headphones in a quiet room.</p>
<p>Each trial plays four versions of the same musical excerpt (A1–B2 rendering ablations).
The <strong>Reference</strong> button plays A1 (basic Fluidsynth synthesis). One of the four
blind conditions is identical to the Reference; the labels are shuffled so you cannot tell
which blind slot is the Reference.</p>
<p>Rate <strong>Basic Audio Quality (BAQ)</strong> from 0–100 for each blind condition
relative to the Reference (100 = transparent / identical to Reference).</p>
<p>You may loop excerpts and switch between Reference and conditions while rating.</p>
"""

VOLUME_PAGE_HTML = """
<p>Set a comfortable listening level using the Reference excerpt below, then continue.</p>
"""

FINISH_PAGE_HTML = """
<p>Thank you for participating. Results are saved automatically when you submit this page.</p>
"""


def default_webmushra_root() -> Path:
    return DEFAULT_WEBMUSHRA_ROOT


def ensure_webmushra(root: Path) -> Path:
    root = root.resolve()
    if not (root / "index.html").is_file():
        raise FileNotFoundError(
            f"webMUSHRA not found at {root}. Clone it:\n"
            f"  git clone https://github.com/audiolabs/webMUSHRA.git {root}"
        )
    return root


def load_manifest(manifest_path: Path) -> dict:
    with open(manifest_path) as f:
        return yaml.safe_load(f) or {}


def _read_audio(path: Path):
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return audio, sample_rate


def convert_to_wav(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".wav" and source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
        return destination
    audio, sample_rate = _read_audio(source)
    sf.write(str(destination), audio, sample_rate, subtype="PCM_16")
    return destination


def export_trial_wavs(
    trial: dict,
    clips_dir: Path,
    webmushra_root: Path,
    *,
    stimuli_subdir: str = WEBMUSHRA_STIMULI_DIR,
) -> dict[str, str]:
    """Copy/convert trial clips to webMUSHRA stimuli tree. Returns rel paths for YAML."""
    trial_id = trial["id"]
    conditions = trial.get("conditions") or {}
    rel_paths: dict[str, str] = {}

    for condition_id in CONDITION_ORDER:
        rel_clip = conditions.get(condition_id)
        if not rel_clip:
            raise FileNotFoundError(f"Trial {trial_id} missing condition {condition_id}")
        source = clips_dir / rel_clip
        if not source.is_file():
            raise FileNotFoundError(f"Missing clip: {source}")

        dest = webmushra_root / stimuli_subdir / trial_id / f"{condition_id}.wav"
        convert_to_wav(source, dest)
        rel_paths[condition_id] = f"{stimuli_subdir}/{trial_id}/{condition_id}.wav"

    return rel_paths


def trial_page_content(trial: dict) -> str:
    trial_type = trial.get("type") or "stem"
    if trial_type == "mixture":
        label = f"Mixture — {trial.get('song_id', trial['id'])}"
    else:
        category = trial.get("category") or "stem"
        label = f"Stem ({category}) — track {trial.get('track')}"
    note = trial.get("note")
    note_html = f"<p><em>{note}</em></p>" if note else ""
    return (
        f"<p><strong>{label}</strong></p>"
        f"{note_html}"
        "<p>Rate each blind condition vs the Reference (A1). "
        "One blind condition matches the Reference.</p>"
    )


def build_mushra_trial_page(trial: dict, wav_paths: dict[str, str]) -> dict:
    reference = wav_paths[REFERENCE_CONDITION]
    return {
        "type": "mushra",
        "id": trial["id"],
        "name": trial["id"].replace("_", " ").title(),
        "content": trial_page_content(trial),
        "showWaveform": True,
        "enableLooping": True,
        "switchBack": True,
        "strict": False,
        "randomize": True,
        "showConditionNames": False,
        "createAnchor35": False,
        "createAnchor70": False,
        "reference": reference,
        "stimuli": {condition_id: wav_paths[condition_id] for condition_id in CONDITION_ORDER},
    }


def build_webmushra_config(
    manifest: dict,
    *,
    volume_stimulus: str,
    test_id: str = WEBMUSHRA_TEST_ID,
    test_name: str = "sPDMX Ablation Listening Test",
) -> dict:
    trials = list(manifest.get("trials") or [])
    mushra_pages = []
    for trial in trials:
        wav_paths = trial.get("webmushra_wav_paths")
        if not wav_paths:
            raise ValueError(f"Trial {trial['id']} missing webmushra_wav_paths")
        mushra_pages.append(build_mushra_trial_page(trial, wav_paths))

    pages: list = [
        {
            "type": "generic",
            "id": "intro",
            "name": "Welcome",
            "content": MUSHRA_INTRO_HTML,
        },
        {
            "type": "volume",
            "id": "volume",
            "name": "Volume",
            "content": VOLUME_PAGE_HTML,
            "stimulus": volume_stimulus,
            "defaultVolume": 0.75,
        },
        "random",
        *mushra_pages,
        {
            "type": "finish",
            "name": "Thank you",
            "content": FINISH_PAGE_HTML,
            "showResults": False,
            "writeResults": True,
            "questionnaire": [
                {
                    "type": "text",
                    "label": "Listener ID",
                    "name": "listener_id",
                },
                {
                    "type": "text",
                    "label": "Headphones (optional)",
                    "name": "headphones",
                },
            ],
        },
    ]

    return {
        "testname": test_name,
        "testId": test_id,
        "bufferSize": 2048,
        "stopOnErrors": True,
        "showButtonPreviousPage": True,
        "remoteService": "service/write.php",
        "pages": pages,
    }


def write_webmushra_config(config: dict, webmushra_root: Path, config_name: str) -> Path:
    config_dir = webmushra_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / config_name
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return config_path


def prepare_webmushra(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    clips_dir: Path = DEFAULT_CLIPS_DIR,
    webmushra_root: Path = DEFAULT_WEBMUSHRA_ROOT,
    config_name: str = WEBMUSHRA_CONFIG_NAME,
    test_id: str = WEBMUSHRA_TEST_ID,
) -> tuple[Path, Path]:
    webmushra_root = ensure_webmushra(webmushra_root)
    manifest_path = manifest_path.resolve()
    clips_dir = clips_dir.resolve()

    manifest = load_manifest(manifest_path)
    trials = list(manifest.get("trials") or [])
    if not trials:
        raise RuntimeError(f"No trials in manifest: {manifest_path}")

    for trial in trials:
        trial["webmushra_wav_paths"] = export_trial_wavs(
            trial,
            clips_dir,
            webmushra_root,
        )

    volume_stimulus = trials[0]["webmushra_wav_paths"][REFERENCE_CONDITION]
    config = build_webmushra_config(manifest, volume_stimulus=volume_stimulus, test_id=test_id)
    config_path = write_webmushra_config(config, webmushra_root, config_name)

    # Save manifest copy with wav paths for aggregation/debugging.
    manifest_out = manifest_path.parent / "trial_manifest_webmushra.yaml"
    with open(manifest_out, "w") as f:
        yaml.safe_dump(
            {
                **manifest,
                "webmushra_root": str(webmushra_root),
                "webmushra_config": config_name,
                "trials": trials,
            },
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    return config_path, webmushra_root


def webmushra_url(host: str, port: int, config_name: str) -> str:
    return f"http://{host}:{port}/?config={config_name}"


def serve_webmushra(
    *,
    webmushra_root: Path = DEFAULT_WEBMUSHRA_ROOT,
    host: str = "127.0.0.1",
    port: int = 8767,
    config_name: str = WEBMUSHRA_CONFIG_NAME,
) -> subprocess.Popen:
    webmushra_root = ensure_webmushra(webmushra_root)
    url = webmushra_url(host, port, config_name)
    print(f"webMUSHRA: {url}")
    print("For remote listeners: ngrok http", port)
    print(f"Results CSV: third_party/webMUSHRA/results/{WEBMUSHRA_TEST_ID}/mushra.csv")
    return subprocess.Popen(
        ["php", "-S", f"{host}:{port}"],
        cwd=str(webmushra_root),
    )
