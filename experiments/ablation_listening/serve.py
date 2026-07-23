"""HTTP server for the ablation listening test (A1–B2)."""

from __future__ import annotations

import argparse
import json
import mimetypes
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from experiments.ablation_listening.catalog import AblationListeningCatalog
from experiments.ablation_listening.paths import (
    DEFAULT_CLIPS_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_RESPONSES_DIR,
)
from experiments.ablation_listening.session import storage_key
from experiments.listening.json_util import json_safe

MODULE_DIR = Path(__file__).resolve().parent
STATIC_DIR = MODULE_DIR / "static"
SHARED_STATIC_DIR = Path(__file__).resolve().parents[1] / "listening_shared" / "static"

AUDIO_MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
}


class AblationListeningHandler(BaseHTTPRequestHandler):
    catalog: AblationListeningCatalog
    static_dir: Path
    shared_static_dir: Path

    def log_message(self, format: str, *args) -> None:
        print(f"[ablation-listening] {self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)

        if path.startswith("/shared/"):
            self._serve_static(path[len("/shared/") :], self.shared_static_dir)
            return
        if path.startswith("/static/"):
            self._serve_static(path[len("/static/") :], self.static_dir)
            return
        if path in ("", "/", "/index.html"):
            self._serve_static_file(self.static_dir / "index.html")
            return
        if path in ("/test", "/test.html"):
            self._serve_static_file(self.static_dir / "test.html")
            return
        if path.startswith("/api/"):
            self._handle_api(path, query)
            return
        if path.startswith("/audio/"):
            self._serve_audio(path)
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path != "/api/responses":
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON body")
            return

        checkpoint = bool(payload.pop("checkpoint", False))
        out_dir = self.catalog.responses_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        if checkpoint:
            out_path = self.catalog.session_responses_path()
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            listener = payload.get("listener_id") or "anonymous"
            safe_listener = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(listener))
            out_path = out_dir / f"responses_{safe_listener}_{timestamp}.json"
        out_path.write_text(json.dumps(json_safe(payload), indent=2))
        self._send_json({"saved": str(out_path), "checkpoint": checkpoint})

    def _handle_api(self, path: str, query: dict) -> None:
        session_seed = int((query.get("seed") or ["42"])[0])
        if path == "/api/meta":
            payload = self.catalog.meta(session_seed)
            payload["storage_key"] = storage_key(self.catalog.test_id, session_seed)
            self._send_json(payload)
            return
        if path == "/api/responses/session":
            session_path = self.catalog.session_responses_path()
            if session_path.is_file():
                self._send_json(json.loads(session_path.read_text()))
            else:
                self._send_json({"ratings": []})
            return
        if path.startswith("/api/trials/"):
            trial_id = path.split("/api/trials/", 1)[1]
            detail = self.catalog.get_trial(trial_id, session_seed)
            if detail is None:
                self._send_error(HTTPStatus.NOT_FOUND, f"Unknown trial: {trial_id}")
                return
            self._send_json(detail)
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _serve_audio(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 3 or parts[0] != "audio":
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        trial_id, filename = parts[1], parts[2]
        audio_path = self.catalog.resolve_audio_path(trial_id, filename)
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

    def _serve_static(self, relative_path: str, root: Path) -> None:
        file_path = (root / relative_path).resolve()
        self._serve_static_file(file_path, root)

    def _serve_static_file(self, file_path: Path, root: Path | None = None) -> None:
        root = (root or self.static_dir).resolve()
        file_path = file_path.resolve()
        if not str(file_path).startswith(str(root)):
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
        data = json.dumps(json_safe(payload), allow_nan=False).encode("utf-8")
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


def make_handler(catalog: AblationListeningCatalog):
    def handler(*args, **kwargs):
        AblationListeningHandler.catalog = catalog
        AblationListeningHandler.static_dir = STATIC_DIR.resolve()
        AblationListeningHandler.shared_static_dir = SHARED_STATIC_DIR.resolve()
        return AblationListeningHandler(*args, **kwargs)

    return handler


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Serve ablation listening test.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8767, type=int)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument("--clips-dir", default=DEFAULT_CLIPS_DIR, type=Path)
    return parser.parse_args(args)


def main(args=None) -> None:
    opts = parse_args(args)
    manifest = opts.manifest.resolve()
    if not manifest.is_file():
        raise RuntimeError(
            f"Missing trial manifest: {manifest}. "
            "Run: uv run python -m experiments.ablation_listening.prepare_clips"
        )
    catalog = AblationListeningCatalog(manifest, opts.clips_dir.resolve())
    handler = make_handler(catalog)
    server = ThreadingHTTPServer((opts.host, opts.port), handler)
    url = f"http://{opts.host}:{opts.port}"
    print(f"Serving ablation listening test at {url}")
    print("For remote listeners: ngrok http", opts.port)
    print("Open", f"{url}/test?seed=42")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
