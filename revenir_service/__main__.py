#!/usr/bin/env python3
"""
Revenir Delegation Service — entry point.

Usage:
    python -m revenir_service                  # Start with defaults (port 7782)
    python -m revenir_service --port 8080      # Custom port
    python -m revenir_service --token abc123   # Pre-set API token
    python -m revenir_service --work-dir /tmp  # Custom work directory
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Revenir Delegation Service — distributed worker for R3v3n!R"
    )
    parser.add_argument(
        "--port", type=int,
        default=int(os.getenv("REVENIR_PORT", "7782")),
        help="Port to listen on (default: 7782)",
    )
    parser.add_argument(
        "--token", type=str,
        default=os.getenv("REVENIR_TOKEN", ""),
        help="API token for authentication (or set via pairing)",
    )
    parser.add_argument(
        "--callback-url", type=str,
        default=os.getenv("REVENIR_CALLBACK_URL", ""),
        help="URL to POST results back to the main system",
    )
    parser.add_argument(
        "--work-dir", type=str,
        default=os.getenv("REVENIR_WORK_DIR", ""),
        help="Working directory for task data (default: ~/.revenir)",
    )
    parser.add_argument(
        "--log-level", type=str,
        default=os.getenv("REVENIR_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    work_dir = Path(args.work_dir) if args.work_dir else None

    from .server import RevenirServer
    server = RevenirServer(
        port=args.port,
        work_dir=work_dir,
        api_token=args.token,
        callback_url=args.callback_url,
    )

    print(f"""
  ==============================================
    Revenir Delegation Service v0.1.0
    Port: {args.port}
    Work: {server.work_dir}
    Mode: {"PAIRED" if server.api_token else "PAIRING"}
  ==============================================
""")

    server.start()


if __name__ == "__main__":
    main()
