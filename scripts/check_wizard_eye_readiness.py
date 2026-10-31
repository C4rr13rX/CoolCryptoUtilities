"""Inspect the deployed Wizard before observation training; never changes model state."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web.wizard_chat.eye_readiness import probe_eye_readiness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8090")
    parser.add_argument("--chat-path", choices=("/brain/chat", "/chat"), default="/brain/chat")
    parser.add_argument("--repo", type=Path, default=ROOT.parent / "W1z4rDV1510n")
    parser.add_argument("--timeout", type=float, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = probe_eye_readiness(args.endpoint, args.chat_path, timeout=args.timeout)
    def git(*command: str) -> str:
        return subprocess.check_output(["git", "-C", str(args.repo), *command], text=True).strip()
    try:
        head, upstream = git("rev-parse", "HEAD"), git("rev-parse", "origin/main")
        report["source"] = {"head": head, "fetched_origin_main": upstream,
                            "matches_fetched_main": head == upstream,
                            "clean": not bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError) as error:
        report["source"] = {"error": str(error)}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["training_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
