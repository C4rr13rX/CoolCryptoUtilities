#!/usr/bin/env python
"""Standalone OHLCV→brain feeder.

Runs the wizard_trainer.brain_feeder loop in a long-lived process so
the W1z4rD brain keeps learning even when the main production manager
process is stuck or its TF training cycle is broken.  Safe to run
alongside the production manager — the feeder is idempotent and the
brain's /brain/observe endpoint is multi-writer safe.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Sensible defaults; override via env before launch.
os.environ.setdefault("WIZARD_BRAIN_FEEDER_ENABLED", "1")
os.environ.setdefault("WIZARD_BRAIN_FEEDER_INTERVAL", "120")
os.environ.setdefault("WIZARD_BRAIN_FEEDER_TAIL", "16")

from trading.wizard_trainer import (  # noqa: E402
    get_trainer,
    start_brain_feeder,
    brain_feeder_status,
)


def main() -> int:
    trainer = get_trainer()
    if not trainer.is_online():
        print("[feeder] brain endpoint unreachable; exiting")
        return 1
    started = start_brain_feeder(chains=("base", "arbitrum", "optimism", "polygon"))
    print(f"[feeder] started={started} interval={os.getenv('WIZARD_BRAIN_FEEDER_INTERVAL')}s "
          f"tail={os.getenv('WIZARD_BRAIN_FEEDER_TAIL')} candles")
    last_log = 0.0
    while True:
        time.sleep(60)
        status = brain_feeder_status()
        if status["last_run"] > last_log:
            print(f"[feeder] cycle@{int(status['last_run'])} pushed={status['last_pushed']}")
            last_log = status["last_run"]


if __name__ == "__main__":
    sys.exit(main())
