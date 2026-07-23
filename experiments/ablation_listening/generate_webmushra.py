"""CLI: export WAV stimuli and generate webMUSHRA config."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.ablation_listening.paths import (
    DEFAULT_CLIPS_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_WEBMUSHRA_ROOT,
    WEBMUSHRA_CONFIG_NAME,
    WEBMUSHRA_TEST_ID,
)
from experiments.ablation_listening.webmushra import prepare_webmushra, webmushra_url


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Export ablation clips to webMUSHRA WAV stimuli and write config YAML.",
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=str)
    parser.add_argument("--clips-dir", default=DEFAULT_CLIPS_DIR, type=str)
    parser.add_argument("--webmushra-root", default=DEFAULT_WEBMUSHRA_ROOT, type=str)
    parser.add_argument("--config-name", default=WEBMUSHRA_CONFIG_NAME)
    parser.add_argument("--test-id", default=WEBMUSHRA_TEST_ID)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8767, type=int)
    return parser.parse_args(args)


def main(args=None) -> None:
    opts = parse_args(args)
    config_path, root = prepare_webmushra(
        manifest_path=Path(opts.manifest),
        clips_dir=Path(opts.clips_dir),
        webmushra_root=Path(opts.webmushra_root),
        config_name=opts.config_name,
        test_id=opts.test_id,
    )
    print(f"Wrote config: {config_path}")
    print(f"Stimuli under: {root / 'stimuli/spdmx_ablation'}")
    print(f"Open: {webmushra_url(opts.host, opts.port, opts.config_name)}")
    print("Serve with: uv run python -m experiments.ablation_listening.serve_webmushra")


if __name__ == "__main__":
    main()
