#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH_ROOT = Path(os.getenv("C0D3R_BENCH_ROOT", "C:/Users/Adam/Projects/c0d3r_benchmarks")).resolve()

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


def run_prompt(name: str, prompt: str, timeout_s: int = 1800) -> tuple[bool, float, str]:
    """
    Run each benchmark in its own elevated PowerShell window and wait for completion.
    """
    env = os.environ.copy()
    env.setdefault("C0D3R_TOOL_STEPS", "5")
    env.setdefault("C0D3R_MODEL_TIMEOUT_S", "60")
    env.setdefault("C0D3R_CMD_TIMEOUT_S", "240")
    env.setdefault("C0D3R_READ_TIMEOUT_S", "60")
    env.setdefault("C0D3R_CONNECT_TIMEOUT_S", "10")

    scripts_dir = ROOT / "runtime" / "c0d3r" / "bench_scripts"
    logs_dir = ROOT / "runtime" / "c0d3r" / "bench_logs"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / f"{name}.ps1"
    log_path = logs_dir / f"{name}.out.log"
    err_path = logs_dir / f"{name}.err.log"
    bench_root = DEFAULT_BENCH_ROOT / name
    bench_root.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        f'cd /d "{bench_root}"\n'
        f'c0d3r "{prompt}"\n'
        "exit $LASTEXITCODE\n",
        encoding="utf-8",
    )
    ps_cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Start-Process",
        "PowerShell",
        "-ArgumentList",
        f"'-NoProfile','-ExecutionPolicy','Bypass','-File','{script_path}'",
        "-Verb",
        "RunAs",
        "-Wait",
        "-RedirectStandardOutput",
        str(log_path),
        "-RedirectStandardError",
        str(err_path),
    ]
    start = time.time()
    try:
        out = subprocess.run(
            ps_cmd,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, timeout_s, "timeout"
    duration = time.time() - start
    stdout = ""
    stderr = ""
    if log_path.exists():
        stdout = log_path.read_text(encoding="utf-8", errors="ignore")
    if err_path.exists():
        stderr = err_path.read_text(encoding="utf-8", errors="ignore")
    ok = "Success:" in stdout
    tail = (stdout + "\n" + stderr + "\n" + (out.stderr or "")).strip()[-1200:]
    return ok, duration, tail


def main() -> int:
    results = []
    for name, prompt in PROMPTS:
        ok, duration, tail = run_prompt(name, prompt)
        results.append((name, ok, duration, tail))
        status = "ok" if ok else "fail"
        print(f"[{name}] {status} in {duration:.1f}s")
        if not ok:
            print(tail)
            print("-" * 60)
    passed = sum(1 for _, ok, _, _ in results if ok)
    total = len(results)
    print(f"Bench complete: {passed}/{total} succeeded")
    # write log
    log_path = ROOT / "runtime" / "c0d3r" / "bench_results.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        for name, ok, duration, tail in results:
            fh.write(f"[{name}] {'ok' if ok else 'fail'} in {duration:.1f}s\n")
            if tail:
                fh.write(tail + "\n")
            fh.write("-" * 60 + "\n")
        fh.write(f"Bench complete: {passed}/{total} succeeded\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
