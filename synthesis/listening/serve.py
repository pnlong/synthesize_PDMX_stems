"""Localhost HTTP server for ablation listening viewer (A1–B2)."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from synthesis.listening.catalog import (
    AblationCatalog,
    CONDITION_ORDER,
    default_ablations_dir,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

AUDIO_MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
}


def parse_audio_request(path: str) -> tuple[str, str, str] | None:
    """Parse /audio/{condition}/{song_id...}/{filename} where song_id may contain slashes."""
    parts = path.strip("/").split("/")
    if len(parts) < 4 or parts[0] != "audio":
        return None
    condition = parts[1]
    if condition not in CONDITION_ORDER:
        return None
    filename = parts[-1]
    if "/" in filename or "\\" in filename or ".." in Path(filename).parts:
        return None
    song_id = "/".join(parts[2:-1])
    if ".." in Path(song_id).parts:
        return None
    return condition, song_id, filename


class ListeningHandler(BaseHTTPRequestHandler):
    catalog: AblationCatalog
    static_dir: Path

    def log_message(self, format: str, *args) -> None:
        print(f"[listening] {self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path.startswith("/static/"):
            self._serve_static(path[len("/static/") :])
            return

        if path == "/api/conditions":
            self._send_json(self.catalog.conditions())
            return
        if path == "/api/songs":
            self._send_json(self.catalog.list_songs())
            return
        if path.startswith("/api/songs/"):
            song_id = path[len("/api/songs/") :].strip("/")
            if not song_id:
                self._send_error(HTTPStatus.BAD_REQUEST, "Missing song id")
                return
            detail = self.catalog.get_song(song_id)
            if detail is None:
                self._send_error(HTTPStatus.NOT_FOUND, f"Song not found: {song_id}")
                return
            self._send_json(detail)
            return
        if path.startswith("/audio/"):
            self._serve_ablation_audio(path)
            return

        if path in ("", "/"):
            self._serve_static("index.html")
            return

        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _serve_ablation_audio(self, path: str) -> None:
        parsed = parse_audio_request(path)
        if parsed is None:
            self._send_error(HTTPStatus.BAD_REQUEST, "Invalid audio path")
            return
        condition, song_id, filename = parsed
        audio_path = self.catalog.resolve_audio_path(condition, song_id, filename)
        if audio_path is None:
            self._send_error(HTTPStatus.NOT_FOUND, "Audio not found")
            return
        self._send_file(audio_path)

    def _send_file(self, audio_path: Path) -> None:
        mime = AUDIO_MIME_TYPES.get(audio_path.suffix.lower()) or mimetypes.guess_type(
            str(audio_path)
        )[0] or "application/octet-stream"
        data = audio_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, relative_path: str) -> None:
        file_path = (self.static_dir / relative_path).resolve()
        self._serve_static_file(file_path)

    def _serve_static_file(self, file_path: Path) -> None:
        file_path = file_path.resolve()
        if not str(file_path).startswith(str(self.static_dir.resolve())):
            self._send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not file_path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, code: HTTPStatus, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_handler(catalog: AblationCatalog, static_dir: Path = STATIC_DIR):
    def handler(*args, **kwargs):
        ListeningHandler.catalog = catalog
        ListeningHandler.static_dir = static_dir.resolve()
        return ListeningHandler(*args, **kwargs)

    return handler


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Serve ablation listening viewer on localhost (A1–B2).",
    )
    parser.add_argument(
        "--ablations-dir",
        default=None,
        type=Path,
        help="Ablation root (default: synthesis/ablations_output symlink).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    return parser.parse_args(args)


def main(args=None) -> None:
    opts = parse_args(args)
    catalog = AblationCatalog(opts.ablations_dir or default_ablations_dir())

    handler = make_handler(catalog)
    server = ThreadingHTTPServer((opts.host, opts.port), handler)
    url = f"http://{opts.host}:{opts.port}"
    print(f"Serving ablation listening viewer at {url}")
    print(f"Ablation root: {catalog.ablations_dir}")
    print(f"Songs: {len(catalog.list_songs())}")
    print("Sweep listening tests: uv run python -m experiments.listening.serve (port 8766)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
