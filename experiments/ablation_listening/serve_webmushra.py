"""Serve webMUSHRA via PHP built-in server."""

from __future__ import annotations

import argparse
import signal
import sys
import time

from experiments.ablation_listening.paths import DEFAULT_WEBMUSHRA_ROOT, WEBMUSHRA_CONFIG_NAME
from experiments.ablation_listening.webmushra import ensure_webmushra, serve_webmushra


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Serve ablation test via webMUSHRA (PHP).")
    parser.add_argument("--webmushra-root", default=DEFAULT_WEBMUSHRA_ROOT, type=str)
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (0.0.0.0 for ngrok).")
    parser.add_argument("--port", default=8767, type=int)
    parser.add_argument("--config-name", default=WEBMUSHRA_CONFIG_NAME)
    return parser.parse_args(args)


def main(args=None) -> None:
    opts = parse_args(args)
    root = ensure_webmushra(opts.webmushra_root)
    config_path = root / "configs" / opts.config_name
    if not config_path.is_file():
        raise RuntimeError(
            f"Missing {config_path}. Run:\n"
            "  uv run python -m experiments.ablation_listening.generate_webmushra"
        )

    process = serve_webmushra(
        webmushra_root=root,
        host=opts.host,
        port=opts.port,
        config_name=opts.config_name,
    )

    def _stop(*_args):
        process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        while process.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        _stop()


if __name__ == "__main__":
    main()
