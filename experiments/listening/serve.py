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

from experiments.listening.catalog import SweepCatalog, default_sweep_dir, resolve_sweep_catalog_dir
from experiments.listening.json_util import json_safe
from experiments.listening.session import RUBRICS, rubric_for_catalog, stem_order, storage_key

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

        if path in ("/verify", "/verify.html"):
            self._serve_static_file(STATIC_DIR / "verify.html")
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

        if payload.get("mode") == "verification":
            self._save_verification(sweep_type, payload)
            return

        checkpoint = bool(payload.pop("checkpoint", False))
        if checkpoint:
            out_path = catalog.session_responses_path()
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_path = catalog.responses_dir() / f"responses_{timestamp}.json"
        out_path.write_text(json.dumps(json_safe(payload), indent=2))
        self._send_json({"saved": str(out_path), "checkpoint": checkpoint})

    def _save_verification(self, sweep_type: str, payload: dict) -> None:
        from experiments.listening.final_verify import final_catalog
        from experiments.listening.verification import (
            validate_verification,
            verification_in_progress_path,
        )

        source_responses = payload.get("source_responses")
        if not source_responses:
            self._send_error(HTTPStatus.BAD_REQUEST, "Missing source_responses")
            return

        try:
            catalog, _ = final_catalog(sweep_type)
        except RuntimeError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        errors = validate_verification(payload)
        checkpoint = bool(payload.get("checkpoint", False))
        if not checkpoint and errors:
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                "; ".join(errors),
            )
            return

        if checkpoint:
            out_path = verification_in_progress_path(catalog, source_responses)
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            safe = source_responses.replace(".json", "")
            out_path = catalog.responses_dir() / f"verification_final_{safe}_{timestamp}.json"
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
                "rubric": rubric_for_catalog(catalog),
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

        if path.startswith(f"/api/{sweep_type}/verify/"):
            self._handle_verify_api(sweep_type, catalog, path, query)
            return

        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _handle_verify_api(
        self,
        sweep_type: str,
        catalog: SweepCatalog,
        path: str,
        query: dict,
    ) -> None:
        from experiments.listening.aggregate import load_responses
        from experiments.listening.final_verify import (
            composed_config,
            final_catalog,
            readiness_errors,
            winners_path_for,
        )
        from experiments.listening.verification import (
            PRESET_VERIFY_SOURCE,
            build_category_verification,
            build_patch_shortlist_verification_meta,
            build_preset_realify_verification_meta,
            build_verification_meta,
            list_response_files,
            resolve_responses_path,
            verification_in_progress_path,
        )

        errors = readiness_errors(sweep_type)
        if errors:
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                "Final verification requires all tuning phases complete: "
                + "; ".join(errors),
            )
            return

        try:
            verify_catalog, verify_phase = final_catalog(sweep_type)
        except RuntimeError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        config_fn = lambda category, variant_id: composed_config(
            sweep_type, category, variant_id
        )
        winners_path = winners_path_for(sweep_type)

        if path == f"/api/{sweep_type}/verify/responses":
            if sweep_type == "preset":
                self._send_json({"files": [], "source": PRESET_VERIFY_SOURCE})
            else:
                self._send_json({"files": list_response_files(verify_catalog)})
            return

        responses_name = query.get("responses", [""])[0]
        if sweep_type == "preset":
            responses_name = responses_name or PRESET_VERIFY_SOURCE
        elif not responses_name:
            self._send_error(HTTPStatus.BAD_REQUEST, "Missing responses query param")
            return

        use_preset_winners = (
            sweep_type == "preset" and responses_name == PRESET_VERIFY_SOURCE
        )
        if not use_preset_winners:
            responses_path = resolve_responses_path(verify_catalog, responses_name)
            if responses_path is None:
                self._send_error(
                    HTTPStatus.NOT_FOUND,
                    f"Responses not found: {responses_name}",
                )
                return

        if path == f"/api/{sweep_type}/verify/meta":
            if sweep_type == "patch":
                from experiments.patch_sweep.winners import phase_winners
                from experiments.patch_sweep.config import PHASE1 as PATCH_PHASE1

                responses = load_responses(responses_path)
                shortlists = phase_winners(PATCH_PHASE1, winners_path)
                meta = build_patch_shortlist_verification_meta(
                    verify_catalog,
                    responses,
                    source_responses=responses_name,
                    shortlists=shortlists,
                    verification_phase=verify_phase,
                )
            elif use_preset_winners:
                from experiments.listening.final_verify import final_phase_winners

                meta = build_preset_realify_verification_meta(
                    verify_catalog,
                    category_winners=final_phase_winners(sweep_type, winners_path),
                    source_responses=PRESET_VERIFY_SOURCE,
                    composed_config_fn=config_fn,
                    verification_phase=verify_phase,
                )
            else:
                responses = load_responses(responses_path)
                meta = build_verification_meta(
                    verify_catalog,
                    responses,
                    source_responses=responses_name,
                    composed_config_fn=config_fn,
                    verification_phase=verify_phase,
                )
            self._send_json(meta)
            return

        if path == f"/api/{sweep_type}/verify/session":
            session_path = verification_in_progress_path(verify_catalog, responses_name)
            if session_path.is_file():
                with open(session_path) as f:
                    payload = json.load(f)
            else:
                payload = {"categories": []}
            self._send_json(payload)
            return

        prefix = f"/api/{sweep_type}/verify/categories/"
        if path.startswith(prefix):
            category = path[len(prefix) :].strip("/")
            if not category:
                self._send_error(HTTPStatus.BAD_REQUEST, "Missing category")
                return
            variant_ids = query.get("variant_id", [])
            if not variant_ids:
                if sweep_type == "patch":
                    from experiments.patch_sweep.winners import phase1_soundfont_ids

                    variant_ids = phase1_soundfont_ids(category, winners_path)
                elif use_preset_winners:
                    from experiments.listening.final_verify import final_phase_winners

                    locked = final_phase_winners(sweep_type, winners_path)
                    if category in locked:
                        variant_ids = [locked[category]]
                else:
                    responses = load_responses(responses_path)
                    meta = build_verification_meta(
                        verify_catalog,
                        responses,
                        source_responses=responses_name,
                        composed_config_fn=config_fn,
                        verification_phase=verify_phase,
                    )
                    for entry in meta.get("categories", []):
                        if entry["category"] == category:
                            variant_ids = [
                                v["variant_id"]
                                for v in entry.get("variants", [])
                                if v.get("passed_filter")
                            ]
                            break
            detail = build_category_verification(
                verify_catalog,
                category,
                variant_ids,
                composed_config_fn=config_fn,
            )
            if detail is None:
                self._send_error(HTTPStatus.NOT_FOUND, f"Category not found: {category}")
                return
            self._send_json(detail)
            return

        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _resolve_audio_path(
        self,
        sweep_type: str,
        kind: str,
        key: str,
        filename: str,
    ) -> Path | None:
        catalog = self.catalogs.get(sweep_type)
        if catalog is None:
            return None

        catalogs_to_try = [catalog]
        try:
            from experiments.listening.final_verify import final_catalog, readiness_errors

            if not readiness_errors(sweep_type):
                verify_catalog, _ = final_catalog(sweep_type)
                if verify_catalog.sweep_dir != catalog.sweep_dir:
                    if kind == "reference":
                        catalogs_to_try = [verify_catalog, catalog]
                    else:
                        catalogs_to_try.append(verify_catalog)
        except RuntimeError:
            pass

        for cat in catalogs_to_try:
            if kind == "reference":
                audio_path = cat.resolve_reference_audio(key, filename)
            else:
                variant_id, song_id = key.split("|", 1)
                audio_path = cat.resolve_variant_audio(variant_id, song_id, filename)
            if audio_path is not None:
                return audio_path
        return None

    def _serve_audio(self, path: str) -> None:
        parsed = parse_sweep_audio(path)
        if parsed is None:
            self._send_error(HTTPStatus.BAD_REQUEST, "Invalid audio path")
            return

        sweep_type, kind, key, filename = parsed
        audio_path = self._resolve_audio_path(sweep_type, kind, key, filename)

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
        sweep_root = dir_overrides[sweep_type] or default_sweep_dir(sweep_type)
        sweep_dir = resolve_sweep_catalog_dir(
            sweep_type,
            sweep_root,
            prefer_verification_phase=True,
        )
        manifest_name = "manifest.csv"
        if sweep_type == opts.sweep or (sweep_dir / manifest_name).is_file():
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
    print(f"  Verification UI: {url}/verify")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
