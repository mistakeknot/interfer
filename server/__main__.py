"""Entry point: ``python -m server`` or ``uv run python -m server``."""

from __future__ import annotations

import argparse
import sys

import uvicorn

from .main import create_app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="interfere",
        description="Local MLX-LM inference server for Apple Silicon",
    )
    parser.add_argument(
        "--port", type=int, default=8421, help="Port to listen on (default: 8421)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Start in dry-run mode (fake tokens, no MLX)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    app = create_app(dry_run=args.dry_run)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
