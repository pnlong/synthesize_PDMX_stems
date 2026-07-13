"""Tests for listening viewer HTTP helpers."""

from http import HTTPStatus
from io import BytesIO

from synthesis.listening.serve import (
    STATIC_DIR,
    ListeningHandler,
    parse_audio_request,
)


def _handler(catalog) -> ListeningHandler:
    ListeningHandler.catalog = catalog
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


def test_serves_static_assets(tmp_path, monkeypatch):
    from synthesis.listening.catalog import AblationCatalog

    ablations = tmp_path / "ablations"
    ablations.mkdir()
    (ablations / "basic").mkdir()
    (ablations / "basic" / "data.csv").write_text("path,n_tracks,title\n")
    (ablations / "basic" / "stems.csv").write_text("path,track,program,is_drum,name,has_lyrics\n")

    monkeypatch.setattr(
        "synthesis.listening.catalog.default_ablations_dir",
        lambda: ablations,
    )
    catalog = AblationCatalog(ablations)
    handler = _handler(catalog)
    handler.path = "/static/app.js"
    handler.do_GET()
    assert handler._last_status == HTTPStatus.OK


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
