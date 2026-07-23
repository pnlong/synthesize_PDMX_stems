"""HTTP server for model listening test (Test 2)."""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path

from experiments.ablation_listening.catalog import AblationListeningCatalog
from experiments.ablation_listening.serve import (
    SHARED_STATIC_DIR,
    AblationListeningHandler,
)
from experiments.model_listening.catalog import ModelListeningCatalog
from experiments.model_listening.paths import DEFAULT_CLIPS_DIR, DEFAULT_MANIFEST

ABLATION_STATIC = Path(__file__).resolve().parents[1] / "ablation_listening" / "static"
MODULE_STATIC = Path(__file__).resolve().parent / "static"


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Serve model listening test.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8768, type=int)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument("--clips-dir", default=DEFAULT_CLIPS_DIR, type=Path)
    return parser.parse_args(args)


def main(args=None) -> None:
    opts = parse_args(args)
    manifest = opts.manifest.resolve()
    if not manifest.is_file():
        raise RuntimeError(f"Missing manifest: {manifest}")
    catalog: AblationListeningCatalog | ModelListeningCatalog = ModelListeningCatalog(
        manifest,
        opts.clips_dir.resolve(),
    )
    if not catalog.trials:
        print("Warning: trial_manifest.yaml has no trials yet (scaffold mode).")

    static_dir = MODULE_STATIC if MODULE_STATIC.is_dir() else ABLATION_STATIC

    def factory(*args, **kwargs):
        AblationListeningHandler.catalog = catalog  # type: ignore[assignment]
        AblationListeningHandler.static_dir = static_dir.resolve()
        AblationListeningHandler.shared_static_dir = SHARED_STATIC_DIR.resolve()
        return AblationListeningHandler(*args, **kwargs)

    server = ThreadingHTTPServer((opts.host, opts.port), factory)
    url = f"http://{opts.host}:{opts.port}"
    print(f"Serving model listening test at {url}")
    print("ngrok http", opts.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
