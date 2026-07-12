"""Localhost HTTP server for ablation and preset-sweep listening viewers."""

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
from synthesis.listening.preset_sweep_catalog import (
    PresetSweepCatalog,
    default_preset_sweep_dir,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
PRESET_SWEEP_STATIC = STATIC_DIR / "preset_sweep.html"

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


def parse_preset_sweep_reference_audio(path: str) -> tuple[str, str] | None:
    parts = path.strip("/").split("/")
    if len(parts) != 5 or parts[:3] != ["audio", "preset-sweep", "reference"]:
        return None
    stem_id, filename = parts[3], parts[4]
    if ".." in Path(stem_id).parts or "/" in filename or "\\" in filename:
        return None
    return stem_id, filename


def parse_preset_sweep_variant_audio(path: str) -> tuple[str, str, str] | None:
    parts = path.strip("/").split("/")
    if len(parts) < 6 or parts[:3] != ["audio", "preset-sweep", "variant"]:
        return None
    variant_id = parts[3]
    filename = parts[-1]
    song_id = "/".join(parts[4:-1])
    if ".." in Path(song_id).parts or "/" in filename or "\\" in filename:
        return None
    return variant_id, song_id, filename


class ListeningHandler(BaseHTTPRequestHandler):
    catalog: AblationCatalog | None
    preset_sweep_catalog: PresetSweepCatalog | None
    static_dir: Path

    def log_message(self, format: str, *args) -> None:
        print(f"[listening] {self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/preset-sweep":
            self._serve_static_file(PRESET_SWEEP_STATIC)
            return

        if path.startswith("/api/preset-sweep/"):
            self._handle_preset_sweep_api(path)
            return

        if path.startswith("/audio/preset-sweep/"):
            self._serve_preset_sweep_audio(path)
            return

        if path.startswith("/static/"):
            self._serve_static(path[len("/static/") :])
            return

        if self.catalog is None:
            if path in ("", "/") and self.preset_sweep_catalog is not None:
                self._serve_static_file(PRESET_SWEEP_STATIC)
                return
            if self.preset_sweep_catalog is not None:
                self._send_error(
                    HTTPStatus.NOT_FOUND,
                    "Ablation catalog not configured; open /preset-sweep",
                )
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Ablation catalog not configured")
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

    def _handle_preset_sweep_api(self, path: str) -> None:
        if self.preset_sweep_catalog is None:
            self._send_error(HTTPStatus.NOT_FOUND, "Preset sweep catalog not configured")
            return

        if path == "/api/preset-sweep/meta":
            self._send_json({
                "available": self.preset_sweep_catalog.available(),
                "variants": self.preset_sweep_catalog.variants(),
            })
            return
        if path == "/api/preset-sweep/stems":
            self._send_json(self.preset_sweep_catalog.list_stems())
            return
        if path.startswith("/api/preset-sweep/stems/"):
            stem_id = path[len("/api/preset-sweep/stems/") :].strip("/")
            if not stem_id:
                self._send_error(HTTPStatus.BAD_REQUEST, "Missing stem id")
                return
            detail = self.preset_sweep_catalog.get_stem(stem_id)
            if detail is None:
                self._send_error(HTTPStatus.NOT_FOUND, f"Stem not found: {stem_id}")
                return
            self._send_json(detail)
            return

        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _serve_preset_sweep_audio(self, path: str) -> None:
        if self.preset_sweep_catalog is None:
            self._send_error(HTTPStatus.NOT_FOUND, "Preset sweep catalog not configured")
            return

        reference = parse_preset_sweep_reference_audio(path)
        if reference is not None:
            stem_id, filename = reference
            audio_path = self.preset_sweep_catalog.resolve_reference_audio(stem_id, filename)
            if audio_path is None:
                self._send_error(HTTPStatus.NOT_FOUND, "Audio not found")
                return
            self._send_file(audio_path)
            return

        variant = parse_preset_sweep_variant_audio(path)
        if variant is not None:
            variant_id, song_id, filename = variant
            audio_path = self.preset_sweep_catalog.resolve_variant_audio(
                variant_id,
                song_id,
                filename,
            )
            if audio_path is None:
                self._send_error(HTTPStatus.NOT_FOUND, "Audio not found")
                return
            self._send_file(audio_path)
            return

        self._send_error(HTTPStatus.BAD_REQUEST, "Invalid preset-sweep audio path")

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


def make_handler(
    catalog: AblationCatalog | None,
    preset_sweep_catalog: PresetSweepCatalog | None,
    static_dir: Path = STATIC_DIR,
):
    def handler(*args, **kwargs):
        ListeningHandler.catalog = catalog
        ListeningHandler.preset_sweep_catalog = preset_sweep_catalog
        ListeningHandler.static_dir = static_dir.resolve()
        return ListeningHandler(*args, **kwargs)

    return handler


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Serve ablation and preset-sweep listening viewers on localhost.",
    )
    parser.add_argument(
        "--ablations-dir",
        default=None,
        type=Path,
        help="Ablation root (default: synthesis/ablations_output symlink).",
    )
    parser.add_argument(
        "--preset-sweep-dir",
        default=None,
        type=Path,
        help="Preset sweep output root (default: experiments/preset_sweep/output symlink).",
    )
    parser.add_argument(
        "--preset-sweep",
        action="store_true",
        help="Enable preset-sweep API routes (always on when sweep manifest exists).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    return parser.parse_args(args)


def main(args=None) -> None:
    opts = parse_args(args)

    catalog = None
    ablations_dir = opts.ablations_dir
    if ablations_dir is not None or not opts.preset_sweep:
        try:
            catalog = AblationCatalog(ablations_dir or default_ablations_dir())
        except FileNotFoundError as exc:
            if opts.preset_sweep:
                print(f"Ablation catalog unavailable: {exc}")
            else:
                raise

    preset_sweep_catalog = None
    sweep_dir = opts.preset_sweep_dir or default_preset_sweep_dir()
    if opts.preset_sweep or (sweep_dir / "manifest.csv").is_file():
        preset_sweep_catalog = PresetSweepCatalog(sweep_dir)

    handler = make_handler(catalog, preset_sweep_catalog)
    server = ThreadingHTTPServer((opts.host, opts.port), handler)
    url = f"http://{opts.host}:{opts.port}"
    print(f"Serving listening viewer at {url}")
    if catalog is not None:
        print(f"Ablation root: {catalog.ablations_dir}")
        print(f"Songs: {len(catalog.list_songs())}")
    if preset_sweep_catalog is not None:
        print(f"Preset sweep: {sweep_dir}")
        print(f"Preset sweep stems: {len(preset_sweep_catalog.list_stems())}")
        print(f"Preset sweep UI: {url}/preset-sweep")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
