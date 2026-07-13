"""Localhost HTTP server for sweep listening tests (patch + preset)."""

from __future__ import annotations

import argparse
import json
import mimetypes
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from experiments.listening.catalog import SweepCatalog, default_sweep_dir
from experiments.listening.json_util import json_safe
from experiments.listening.session import RUBRICS, stem_order, storage_key

STATIC_DIR = Path(__file__).resolve().parent / "static"

AUDIO_MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
}


def parse_sweep_audio(path: str) -> tuple[str, str, str, str] | None:
    """Parse /audio/{sweep}/reference|variant/..."""
    parts = path.strip("/").split("/")
    if len(parts) < 4 or parts[0] != "audio":
        return None
    sweep_type = parts[1]
    if sweep_type not in ("preset", "patch"):
        return None
    kind = parts[2]
    if kind not in ("reference", "variant"):
        return None

    if kind == "reference":
        if len(parts) != 5:
            return None
        stem_id, filename = parts[3], parts[4]
        if ".." in Path(stem_id).parts or "/" in filename or "\\" in filename:
            return None
        return sweep_type, kind, stem_id, filename

    if len(parts) < 6:
        return None
    variant_id = parts[3]
    filename = parts[-1]
    song_id = "/".join(parts[4:-1])
    if ".." in Path(song_id).parts or "/" in filename or "\\" in filename:
        return None
    return sweep_type, kind, f"{variant_id}|{song_id}", filename


class SweepListeningHandler(BaseHTTPRequestHandler):
    catalogs: dict[str, SweepCatalog]
    static_dir: Path
    default_sweep: str | None

    def log_message(self, format: str, *args) -> None:
        print(f"[sweep-listening] {self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)

        if path.startswith("/static/"):
            self._serve_static(path[len("/static/") :])
            return

        if path in ("", "/", "/index.html"):
            self._serve_static_file(STATIC_DIR / "index.html")
            return

        if path in ("/test", "/test.html"):
            self._serve_static_file(STATIC_DIR / "test.html")
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
        if not path.startswith("/api/"):
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        parts = path.strip("/").split("/")
        if len(parts) != 3 or parts[0] != "api" or parts[2] != "responses":
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        sweep_type = parts[1]
        catalog = self.catalogs.get(sweep_type)
        if catalog is None:
            self._send_error(HTTPStatus.NOT_FOUND, f"Sweep not configured: {sweep_type}")
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON body")
            return

        checkpoint = bool(payload.pop("checkpoint", False))
        if checkpoint:
            out_path = catalog.session_responses_path()
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_path = catalog.responses_dir() / f"responses_{timestamp}.json"
        out_path.write_text(json.dumps(json_safe(payload), indent=2))
        self._send_json({"saved": str(out_path), "checkpoint": checkpoint})

    def _catalog_for_path(self, path: str) -> SweepCatalog | None:
        parts = path.strip("/").split("/")
        if len(parts) < 3 or parts[0] != "api":
            return None
        return self.catalogs.get(parts[1])

    def _handle_api(self, path: str, query: dict) -> None:
        parts = path.strip("/").split("/")
        if len(parts) < 3:
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        sweep_type = parts[1]
        catalog = self.catalogs.get(sweep_type)
        if catalog is None:
            self._send_error(HTTPStatus.NOT_FOUND, f"Sweep not configured: {sweep_type}")
            return

        if path == f"/api/{sweep_type}/meta":
            session_seed = int(query.get("session_seed", ["42"])[0])
            stem_ids = [s["id"] for s in catalog.list_stems()]
            self._send_json({
                "sweep_type": sweep_type,
                "available": catalog.available(),
                "manifest_id": catalog.manifest_id(),
                "storage_key": storage_key(sweep_type, catalog.manifest_id()),
                "rubric": RUBRICS[sweep_type],
                "variants": catalog.variants(),
                "stems": catalog.list_stems(),
                "stem_order": stem_order(stem_ids, session_seed),
                "session_seed": session_seed,
            })
            return

        if path == f"/api/{sweep_type}/responses/session":
            session_path = catalog.session_responses_path()
            if session_path.is_file():
                with open(session_path) as f:
                    payload = json.load(f)
            else:
                payload = {"ratings": []}
            self._send_json(payload)
            return

        if path == f"/api/{sweep_type}/stems":
            self._send_json(catalog.list_stems())
            return

        if path.startswith(f"/api/{sweep_type}/stems/"):
            stem_id = path[len(f"/api/{sweep_type}/stems/") :].strip("/")
            if not stem_id:
                self._send_error(HTTPStatus.BAD_REQUEST, "Missing stem id")
                return
            session_seed = int(query.get("session_seed", ["42"])[0])
            detail = catalog.get_stem_test(stem_id, session_seed=session_seed)
            if detail is None:
                self._send_error(HTTPStatus.NOT_FOUND, f"Stem not found: {stem_id}")
                return
            self._send_json(detail)
            return

        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _serve_audio(self, path: str) -> None:
        parsed = parse_sweep_audio(path)
        if parsed is None:
            self._send_error(HTTPStatus.BAD_REQUEST, "Invalid audio path")
            return

        sweep_type, kind, key, filename = parsed
        catalog = self.catalogs.get(sweep_type)
        if catalog is None:
            self._send_error(HTTPStatus.NOT_FOUND, "Sweep not configured")
            return

        if kind == "reference":
            audio_path = catalog.resolve_reference_audio(key, filename)
        else:
            variant_id, song_id = key.split("|", 1)
            audio_path = catalog.resolve_variant_audio(variant_id, song_id, filename)

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


def make_handler(
    catalogs: dict[str, SweepCatalog],
    default_sweep: str | None = None,
    static_dir: Path = STATIC_DIR,
):
    def handler(*args, **kwargs):
        SweepListeningHandler.catalogs = catalogs
        SweepListeningHandler.default_sweep = default_sweep
        SweepListeningHandler.static_dir = static_dir.resolve()
        return SweepListeningHandler(*args, **kwargs)

    return handler


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Serve sweep listening tests on localhost.",
    )
    parser.add_argument(
        "--sweep",
        default=None,
        choices=["preset", "patch"],
        help="Enable a single sweep (default: both when manifests exist).",
    )
    parser.add_argument(
        "--preset-sweep-dir",
        default=None,
        type=Path,
        help="Preset sweep output root.",
    )
    parser.add_argument(
        "--patch-sweep-dir",
        default=None,
        type=Path,
        help="Patch sweep output root.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8766, type=int)
    return parser.parse_args(args)


def main(args=None) -> None:
    opts = parse_args(args)
    catalogs: dict[str, SweepCatalog] = {}

    sweep_types = [opts.sweep] if opts.sweep else ["preset", "patch"]
    dir_overrides = {
        "preset": opts.preset_sweep_dir,
        "patch": opts.patch_sweep_dir,
    }

    for sweep_type in sweep_types:
        sweep_dir = dir_overrides[sweep_type] or default_sweep_dir(sweep_type)
        if sweep_type == opts.sweep or (sweep_dir / "manifest.csv").is_file():
            catalogs[sweep_type] = SweepCatalog(sweep_type, sweep_dir)

    if not catalogs:
        raise RuntimeError(
            "No sweep manifests found. Run a sweep first or pass --sweep preset|patch."
        )

    handler = make_handler(catalogs, default_sweep=opts.sweep)
    server = ThreadingHTTPServer((opts.host, opts.port), handler)
    url = f"http://{opts.host}:{opts.port}"
    print(f"Serving sweep listening test at {url}")
    for sweep_type, catalog in catalogs.items():
        print(f"  {sweep_type}: {catalog.sweep_dir} ({len(catalog.list_stems())} stems)")
    print(f"  Test UI: {url}/test")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
