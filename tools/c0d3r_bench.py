#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


PROMPTS = [
    ("django", "Create a minimal Django project skeleton with a single app and a passing unit test."),
    ("fastapi", "Create a minimal FastAPI project skeleton with a health endpoint and a passing unit test."),
    ("flask", "Create a minimal Flask project skeleton with a hello route and a passing unit test."),
    ("react_vite", "Create a minimal React (Vite) project skeleton with a passing unit test."),
    ("vue_vite", "Create a minimal Vue (Vite) project skeleton with a passing unit test."),
    ("svelte_vite", "Create a minimal Svelte (Vite) project skeleton with a passing unit test."),
    ("nextjs", "Create a minimal Next.js project skeleton with a passing unit test."),
    ("astro", "Create a minimal Astro project skeleton with a passing unit test."),
    ("angular", "Create a minimal Angular project skeleton with a passing unit test."),
    ("nuxt", "Create a minimal Nuxt project skeleton with a passing unit test."),
]


def run_prompt(name: str, prompt: str, timeout_s: int = 300) -> bool:
    env = os.environ.copy()
    env.setdefault("C0D3R_TOOL_STEPS", "4")
    env.setdefault("C0D3R_MODEL_TIMEOUT_S", "50")
    env.setdefault("C0D3R_CMD_TIMEOUT_S", "180")
    env.setdefault("C0D3R_READ_TIMEOUT_S", "60")
    env.setdefault("C0D3R_CONNECT_TIMEOUT_S", "10")
    cmd = ["cmd", "/c", "c0d3r", prompt]
    start = time.time()
    try:
        out = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"[{name}] timeout after {timeout_s}s")
        return False
    duration = time.time() - start
    ok = "Success:" in (out.stdout or "")
    print(f"[{name}] {'ok' if ok else 'fail'} in {duration:.1f}s")
    if not ok:
        print(out.stdout[-1200:])
        print(out.stderr[-1200:])
    return ok


def main() -> int:
    total = len(PROMPTS)
    passed = 0
    for name, prompt in PROMPTS:
        ok = run_prompt(name, prompt)
        if ok:
            passed += 1
    print(f"Bench complete: {passed}/{total} succeeded")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
