"""HTTP server tests for ablation listening."""

import json
from http import HTTPStatus
from io import BytesIO
from pathlib import Path

import yaml

from experiments.ablation_listening.catalog import AblationListeningCatalog
from experiments.ablation_listening.serve import (
    SHARED_STATIC_DIR,
    STATIC_DIR,
    AblationListeningHandler,
    make_handler,
)


def _write_manifest(tmp_path: Path) -> AblationListeningCatalog:
    clips_dir = tmp_path / "clips"
    trial_dir = clips_dir / "mix_01"
    trial_dir.mkdir(parents=True)
    for cond in ("basic", "basic_realify", "slakh", "slakh_realify"):
        (trial_dir / f"{cond}.mp3").write_bytes(b"\x00" * 64)

    manifest = tmp_path / "trial_manifest.yaml"
    with open(manifest, "w") as f:
        yaml.safe_dump({
            "test_id": "test_v1",
            "trials": [{
                "id": "mix_01",
                "type": "mixture",
                "song_id": "1/2/QmTest",
                "clip_seconds": 10.0,
                "audio_format": "mp3",
                "conditions": {
                    c: f"mix_01/{c}.mp3"
                    for c in ("basic", "basic_realify", "slakh", "slakh_realify")
                },
            }],
        }, f)
    return AblationListeningCatalog(manifest, clips_dir)


def _handler(catalog: AblationListeningCatalog) -> AblationListeningHandler:
    factory = make_handler(catalog)
    handler = factory.__wrapped__ if hasattr(factory, "__wrapped__") else None
    AblationListeningHandler.catalog = catalog
    AblationListeningHandler.static_dir = STATIC_DIR.resolve()
    AblationListeningHandler.shared_static_dir = SHARED_STATIC_DIR.resolve()
    handler = AblationListeningHandler.__new__(AblationListeningHandler)
    handler._last_status = 0
    handler.request_version = "HTTP/1.1"
    handler.headers = {"Content-Length": "0"}
    handler.rfile = BytesIO(b"{}")
    handler.wfile = BytesIO()

    def send_response(code, message=None):
        handler._last_status = code

    handler.send_response = send_response
    handler.send_header = lambda *args, **kwargs: None
    handler.end_headers = lambda: None
    return handler


def test_serves_meta(tmp_path: Path):
    catalog = _write_manifest(tmp_path)
    handler = _handler(catalog)
    handler.path = "/api/meta?seed=42"
    handler.do_GET()
    assert handler._last_status == HTTPStatus.OK
    payload = json.loads(handler.wfile.getvalue())
    assert payload["n_trials"] == 1


def test_post_responses(tmp_path: Path, monkeypatch):
    catalog = _write_manifest(tmp_path)
    responses_dir = tmp_path / "responses"
    monkeypatch.setattr(catalog, "responses_dir", lambda: responses_dir)
    monkeypatch.setattr(
        catalog,
        "session_responses_path",
        lambda: responses_dir / "responses_in_progress.json",
    )

    handler = _handler(catalog)
    handler.catalog = catalog
    body = json.dumps({
        "listener_id": "tester",
        "checkpoint": True,
        "ratings": [],
    }).encode()
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = BytesIO(body)
    handler.path = "/api/responses"
    handler.do_POST()
    assert handler._last_status == HTTPStatus.OK
    assert (responses_dir / "responses_in_progress.json").is_file()
