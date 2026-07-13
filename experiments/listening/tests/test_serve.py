"""Tests for sweep listening HTTP server."""

from http import HTTPStatus
from io import BytesIO
from pathlib import Path

import pandas as pd
import yaml

from experiments.listening.catalog import SweepCatalog
from experiments.listening.serve import (
    STATIC_DIR,
    SweepListeningHandler,
    parse_sweep_audio,
)


def _write_sweep(tmp_path: Path) -> SweepCatalog:
    sweep_dir = tmp_path / "sweep"
    source_dir = tmp_path / "basic"
    song_id = "0/13/QmTest"
    song_dir = source_dir / "data" / song_id
    song_dir.mkdir(parents=True)
    (song_dir / "stem_0.flac").write_bytes(b"fake")

    variant_dir = sweep_dir / "variants" / "noise0.25_current" / "data" / song_id
    variant_dir.mkdir(parents=True)
    (variant_dir / "stem_0.flac").write_bytes(b"fake")

    pd.DataFrame([{
        "variant_id": "noise0.25_current",
        "init_noise_level": 0.25,
        "prompt_variant": "current",
        "prompt": "solo piano",
        "stem_id": "piano_test",
        "category": "piano",
        "path": str(song_dir),
        "track": 0,
        "out_path": str(variant_dir / "stem_0.flac"),
    }]).to_csv(sweep_dir / "manifest.csv", index=False)

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [{
            "id": "piano_test",
            "category": "piano",
            "song_id": song_id,
            "track": 0,
        }],
    }))
    return SweepCatalog("preset", sweep_dir, source_dir, probe_stems_path=probe_path)


def _handler(catalog: SweepCatalog) -> SweepListeningHandler:
    SweepListeningHandler.catalogs = {"preset": catalog}
    SweepListeningHandler.default_sweep = "preset"
    SweepListeningHandler.static_dir = STATIC_DIR.resolve()
    handler = SweepListeningHandler.__new__(SweepListeningHandler)
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


def test_serves_test_page(tmp_path: Path):
    catalog = _write_sweep(tmp_path)
    handler = _handler(catalog)
    handler.path = "/test"
    handler.do_GET()
    assert handler._last_status == HTTPStatus.OK
    assert b"Rate each sample" in handler.wfile.getvalue()


def test_api_meta(tmp_path: Path):
    catalog = _write_sweep(tmp_path)
    handler = _handler(catalog)
    handler.path = "/api/preset/meta?session_seed=42"
    handler.do_GET()
    assert handler._last_status == HTTPStatus.OK
    payload = __import__("json").loads(handler.wfile.getvalue())
    assert payload["sweep_type"] == "preset"
    assert len(payload["stems"]) == 1


def _write_patch_sweep(tmp_path: Path) -> SweepCatalog:
    sweep_dir = tmp_path / "sweep"
    source_dir = tmp_path / "basic"
    song_id = "0/13/QmTest"
    song_dir = source_dir / "data" / song_id
    song_dir.mkdir(parents=True)
    (song_dir / "stem_0.flac").write_bytes(b"fake")

    variant_dir = sweep_dir / "variants" / "sgm_v2" / "data" / song_id
    variant_dir.mkdir(parents=True)
    (variant_dir / "stem_0.flac").write_bytes(b"fake")

    pd.DataFrame([{
        "phase": "phase1_soundfonts",
        "variant_id": "sgm_v2",
        "soundfont_id": "sgm_v2",
        "soundfont_file": "SGM-V2.01.sf2",
        "fx_profile": "dry",
        "pool_id": "",
        "program": 0,
        "gm_class": "piano",
        "stem_id": "piano_test",
        "category": "piano",
        "path": str(song_dir),
        "track": 0,
        "out_path": str(variant_dir / "stem_0.flac"),
    }]).to_csv(sweep_dir / "manifest.csv", index=False)

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [{
            "id": "piano_test",
            "category": "piano",
            "song_id": song_id,
            "track": 0,
        }],
    }))
    return SweepCatalog("patch", sweep_dir, source_dir, probe_stems_path=probe_path)


def test_api_meta_patch_phase1_json(tmp_path: Path):
    catalog = _write_patch_sweep(tmp_path)
    handler = _handler(catalog)
    handler.catalogs = {"patch": catalog}
    handler.path = "/api/patch/meta?session_seed=42"
    handler.do_GET()
    assert handler._last_status == HTTPStatus.OK
    body = handler.wfile.getvalue()
    assert b": NaN" not in body
    payload = __import__("json").loads(body)
    assert payload["sweep_type"] == "patch"
    assert payload["variants"][0]["pool_id"] in ("", None)


def test_parse_sweep_audio_paths():
    ref = parse_sweep_audio("/audio/preset/reference/piano_test/stem_0.flac")
    assert ref == ("preset", "reference", "piano_test", "stem_0.flac")

    var = parse_sweep_audio("/audio/preset/variant/noise0.25_current/0/13/QmTest/stem_0.flac")
    assert var == ("preset", "variant", "noise0.25_current|0/13/QmTest", "stem_0.flac")


def test_api_responses_session_and_checkpoint(tmp_path: Path):
    catalog = _write_sweep(tmp_path)
    handler = _handler(catalog)

    handler.path = "/api/preset/responses/session"
    handler.do_GET()
    assert handler._last_status == HTTPStatus.OK
    payload = __import__("json").loads(handler.wfile.getvalue())
    assert payload["ratings"] == []

    body = __import__("json").dumps({
        "checkpoint": True,
        "ratings": [{
            "stem_id": "piano_test",
            "category": "piano",
            "samples": [{
                "variant_id": "noise0.25_current",
                "blind_label": "A",
                "content": 4,
                "realism": 5,
            }],
        }],
    }).encode("utf-8")
    handler.path = "/api/preset/responses"
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = BytesIO(body)
    handler.wfile = BytesIO()
    handler.do_POST()
    assert handler._last_status == HTTPStatus.OK
    result = __import__("json").loads(handler.wfile.getvalue())
    assert result["checkpoint"] is True
    session_path = catalog.session_responses_path()
    assert session_path.is_file()
    saved = __import__("json").loads(session_path.read_text())
    assert saved["ratings"][0]["stem_id"] == "piano_test"
    assert "checkpoint" not in saved

    handler.path = "/api/preset/responses/session"
    handler.wfile = BytesIO()
    handler.do_GET()
    reloaded = __import__("json").loads(handler.wfile.getvalue())
    assert reloaded["ratings"][0]["stem_id"] == "piano_test"
