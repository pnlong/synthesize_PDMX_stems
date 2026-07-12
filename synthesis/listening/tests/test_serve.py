"""Tests for listening viewer HTTP helpers."""

from http import HTTPStatus
from io import BytesIO

from synthesis.listening.preset_sweep_catalog import PresetSweepCatalog
from synthesis.listening.serve import (
    STATIC_DIR,
    ListeningHandler,
    parse_audio_request,
    parse_preset_sweep_reference_audio,
    parse_preset_sweep_variant_audio,
)


def _handler_without_init(catalog, preset_sweep_catalog) -> ListeningHandler:
    ListeningHandler.catalog = catalog
    ListeningHandler.preset_sweep_catalog = preset_sweep_catalog
    ListeningHandler.static_dir = STATIC_DIR.resolve()
    handler = ListeningHandler.__new__(ListeningHandler)
    handler._last_status = 0
    handler.request_version = "HTTP/1.1"
    handler.wfile = BytesIO()

    def send_response(code, message=None):
        handler._last_status = code

    handler.send_response = send_response
    handler.send_header = lambda *args, **kwargs: None
    handler.end_headers = lambda: None
    return handler


def test_serves_static_assets_without_ablation_catalog(tmp_path):
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    (sweep_dir / "manifest.csv").write_text(
        "variant_id,init_noise_level,prompt_variant,prompt,stem_id,category,path,track,out_path\n"
    )

    handler = _handler_without_init(None, PresetSweepCatalog(sweep_dir))
    handler.path = "/static/preset_sweep.js"
    handler.do_GET()
    assert handler._last_status == HTTPStatus.OK
    assert b"fetchJson" in handler.wfile.getvalue()


def test_root_serves_preset_sweep_when_ablation_missing(tmp_path):
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    (sweep_dir / "manifest.csv").write_text(
        "variant_id,init_noise_level,prompt_variant,prompt,stem_id,category,path,track,out_path\n"
    )

    handler = _handler_without_init(None, PresetSweepCatalog(sweep_dir))
    handler.path = "/"
    handler.do_GET()
    assert handler._last_status == HTTPStatus.OK
    assert b"Preset Sweep Viewer" in handler.wfile.getvalue()


def test_parse_audio_request_with_sharded_song_id():
    path = "/audio/basic/13/35/QmVoyUuajQJjJ5qUDGVm5sibbWrx1YqN9kNFH295NnRiqM/mixture.mp3"
    assert parse_audio_request(path) == (
        "basic",
        "13/35/QmVoyUuajQJjJ5qUDGVm5sibbWrx1YqN9kNFH295NnRiqM",
        "mixture.mp3",
    )


def test_parse_audio_request_rejects_traversal():
    assert parse_audio_request("/audio/basic/../evil/mixture.mp3") is None
    assert parse_audio_request("/audio/basic/7/19/QmTest/stem_0.mp3") == (
        "basic",
        "7/19/QmTest",
        "stem_0.mp3",
    )


def test_parse_audio_request_rejects_invalid_condition():
    assert parse_audio_request("/audio/unknown/7/19/QmTest/mixture.mp3") is None


def test_parse_preset_sweep_reference_audio():
    path = "/audio/preset-sweep/reference/piano_test/stem_0.mp3"
    assert parse_preset_sweep_reference_audio(path) == ("piano_test", "stem_0.mp3")


def test_parse_preset_sweep_variant_audio():
    path = "/audio/preset-sweep/variant/noise0.45_minimal/0/13/QmTest/stem_0.mp3"
    assert parse_preset_sweep_variant_audio(path) == (
        "noise0.45_minimal",
        "0/13/QmTest",
        "stem_0.mp3",
    )

