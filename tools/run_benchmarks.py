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


def _build_master_script() -> Path:
    scripts_dir = ROOT / "runtime" / "c0d3r" / "bench_scripts"
    logs_dir = ROOT / "runtime" / "c0d3r" / "bench_logs"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    master_path = scripts_dir / "run_all.ps1"
    summary_path = logs_dir / "summary.log"
    lines = [
        '$ErrorActionPreference = "Continue"',
        f'$summary = "{summary_path}"',
        'if (Test-Path $summary) { Remove-Item $summary -Force }',
        'function Write-Result($name, $status, $seconds) {',
        '  $line = "[$name] $status in ${seconds}s"',
        '  $line | Tee-Object -FilePath $summary -Append',
        '}',
        '',
    ]
    for name, prompt in PROMPTS:
        bench_root = DEFAULT_BENCH_ROOT / name
        try:
            if str(bench_root).startswith(str(ROOT)):
                bench_root = Path("C:/Users/Adam/Projects/c0d3r_benchmarks") / name
        except Exception:
            bench_root = Path("C:/Users/Adam/Projects/c0d3r_benchmarks") / name
        out_path = logs_dir / f"{name}.out.log"
        err_path = logs_dir / f"{name}.err.log"
        safe_prompt = prompt.replace('"', '`"')
        lines.extend(
            [
                f'Write-Host "=== {name} ==="',
                f'New-Item -ItemType Directory -Force "{bench_root}" | Out-Null',
                f'Set-Location -Path "{bench_root}"',
                f'Write-Host "Starting benchmark in {bench_root}..."',
                '$start = Get-Date',
                '$env:C0D3R_ONESHOT="1"',
                '$env:C0D3R_MINIMAL_CONTEXT="1"',
                '$env:C0D3R_TOOL_LOOP="1"',
                '$env:C0D3R_SCIENTIFIC_MODE="0"',
                '$env:C0D3R_ENABLE_RESEARCH="0"',
                f'$log = "{out_path}"',
                f'$err = "{err_path}"',
                'if (Test-Path $log) { Clear-Content $log }',
                'if (Test-Path $err) { Clear-Content $err }',
                f'$proc = Start-Process -FilePath "c0d3r" -ArgumentList @("{safe_prompt}") -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError $err',
                '$spinner = @("|","/","-","\\")',
                '$spin = 0',
                '$lastSize = -1',
                '$stale = 0',
                'while (-not $proc.HasExited) {',
                '  Start-Sleep -Seconds 2',
                '  $size = 0',
                '  if (Test-Path $log) { $size = (Get-Item $log).Length }',
                '  if ($size -eq $lastSize) { $stale += 5 } else { $stale = 0; $lastSize = $size }',
                '  $char = $spinner[$spin % $spinner.Length]',
                '  $spin += 1',
                '  $diag = "runtime/c0d3r/diagnostics.log"',
                '  $diagTail = ""',
                '  if (Test-Path $diag) { $diagTail = (Get-Content $diag -Tail 2 | Out-String).Trim() }',
                '  Write-Host ("[{0}] running... output bytes={1} stale={2}s" -f $char, $size, $stale)',
                '  if ($diagTail) { Write-Host ("[diag] {0}" -f $diagTail) }',
                '  if ($stale -ge 60) { Write-Host "No output for 60s; stopping."; $proc.Kill(); break }',
                '}',
                '$code = $proc.ExitCode',
                '$elapsed = [int]((Get-Date) - $start).TotalSeconds',
                'if ($code -eq 0) { Write-Result "%s" "ok" $elapsed } else { Write-Result "%s" "fail" $elapsed }'
                % (name, name),
                '',
            ]
        )
    lines.append('Write-Host "Benchmarks complete. Summary:"')
    lines.append('Get-Content $summary')
    master_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return master_path


def run_all(timeout_s: int = 7200) -> tuple[bool, float, str]:
    env = os.environ.copy()
    env.setdefault("C0D3R_TOOL_STEPS", "5")
    env.setdefault("C0D3R_MODEL_TIMEOUT_S", "60")
    env.setdefault("C0D3R_CMD_TIMEOUT_S", "240")
    env.setdefault("C0D3R_READ_TIMEOUT_S", "60")
    env.setdefault("C0D3R_CONNECT_TIMEOUT_S", "10")
    script_path = _build_master_script()
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
    tail = (out.stdout or "") + "\n" + (out.stderr or "")
    return out.returncode == 0, duration, tail.strip()[-1200:]


def main() -> int:
    ok, duration, tail = run_all()
    print(f"Benchmarks finished in {duration:.1f}s")
    if tail:
        print(tail)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
