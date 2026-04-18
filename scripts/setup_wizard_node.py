#!/usr/bin/env python3
"""scripts/setup_wizard_node.py — Bootstrap the W1z4rD Vision Node for CoolCryptoUtilities.

Usage:
    python scripts/setup_wizard_node.py [--start] [--check] [--train]

Options:
    --check   Only probe the node and report status (default if no flags).
    --start   Build and start the node if it's offline.
    --train   Print training instructions and suggested script order.

The W1z4rD Vision Node is the default AI backend for C0d3rV2.  It runs locally
at http://localhost:8090 and answers queries via Hebbian neural associations
trained on your own data.  Unlike a transformer API, responses improve as the
node trains on more data — run the training scripts to teach it your domain.

Node repo: https://github.com/C4rr13rX/W1z4rDV1510n
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

WIZARD_ENDPOINT = os.getenv("WIZARD_NODE_URL", "http://localhost:8090")
WIZARD_REPO_URL = "https://github.com/C4rr13rX/W1z4rDV1510n.git"

# Where to look for (or clone) the node repo, in priority order.
_SEARCH_PATHS = [
    Path(os.getenv("WIZARD_NODE_DIR", "")).expanduser() if os.getenv("WIZARD_NODE_DIR") else None,
    Path(__file__).resolve().parent.parent.parent / "W1z4rDV1510n",
    Path.home() / "W1z4rDV1510n",
    Path("D:/Projects/W1z4rDV1510n") if platform.system() == "Windows" else None,
    Path("/opt/W1z4rDV1510n"),
]

# W1z4rDV1510n data directory (Hebbian pool lives here).
WIZARD_DATA_DIR = os.getenv("W1Z4RDV1510N_DATA_DIR", "D:\\w1z4rdv1510n-data" if platform.system() == "Windows" else "/var/w1z4rdv1510n-data")

TRAINING_SCRIPTS_NOTE = """\
Training Scripts (run from the W1z4rDV1510n project root to teach the node):
--------------------------------------------------------------------------
The node learns via Hebbian association from text corpora.  More training =
better answers.  Training scripts live in W1z4rDV1510n/scripts/.

Suggested order:
  Stage  0: build_base_corpus.py        — language fundamentals, grammar
  Stage  1: build_book_corpus.py        — general knowledge books (Gutenberg)
  Stage 30: build_medical_corpus.py     — medical/neuroscience terminology
  Stage 34: build_math_corpus.py        — math symbology, calculus, topology
  Stage 35: build_pedagogy_corpus.py    — learning science, curriculum design
  Stage  2: build_cow_dataset.py        — video/image training data (optional)

Run them all at once:
  bash scripts/run_all_training.sh

Or individually:
  python scripts/build_math_corpus.py
  python scripts/build_pedagogy_corpus.py

Training progress is logged to: {data_dir}/training/

IMPORTANT: The node must be running while training scripts execute so they can
push data to the /neuro/train endpoint.
""".format(data_dir=WIZARD_DATA_DIR)


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def probe_node(endpoint: str = WIZARD_ENDPOINT) -> dict:
    url = f"{endpoint.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        return {"online": True, "endpoint": endpoint, "version": data.get("version", ""), "error": ""}
    except urllib.error.URLError as exc:
        return {"online": False, "endpoint": endpoint, "version": "", "error": str(exc)}
    except Exception as exc:
        return {"online": False, "endpoint": endpoint, "version": "", "error": str(exc)}


# ---------------------------------------------------------------------------
# Locate / clone repo
# ---------------------------------------------------------------------------

def find_repo() -> Path | None:
    for candidate in _SEARCH_PATHS:
        if candidate is None:
            continue
        cargo = candidate / "Cargo.toml"
        if cargo.exists():
            return candidate
    return None


def clone_repo(target: Path) -> bool:
    if not shutil.which("git"):
        print("[setup] ERROR: git not found in PATH — cannot clone W1z4rDV1510n repo.")
        return False
    print(f"[setup] Cloning {WIZARD_REPO_URL} → {target} …")
    result = subprocess.run(["git", "clone", "--depth", "1", WIZARD_REPO_URL, str(target)], check=False)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _cargo_env(repo: Path) -> dict:
    env = dict(os.environ)
    # WinLibs MinGW-w64 needed for dlltool on Windows
    if platform.system() == "Windows":
        winlibs_paths = [
            r"C:\Users\Node\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin",
            r"C:\msys64\mingw64\bin",
        ]
        for wl in winlibs_paths:
            if Path(wl).exists():
                env["PATH"] = wl + os.pathsep + env.get("PATH", "")
                break
        cargo_bin = Path.home() / ".cargo" / "bin"
        if cargo_bin.exists():
            env["PATH"] = str(cargo_bin) + os.pathsep + env["PATH"]
    env["W1Z4RDV1510N_DATA_DIR"] = WIZARD_DATA_DIR
    return env


def build_node(repo: Path) -> bool:
    if not shutil.which("cargo"):
        cargo_bin = Path.home() / ".cargo" / "bin" / "cargo"
        if not cargo_bin.exists():
            print("[setup] ERROR: Rust/cargo not found.  Install from https://rustup.rs/")
            return False
    print(f"[setup] Building W1z4rDV1510n (release, workspace) in {repo} …")
    print("[setup] This may take several minutes on first build.")
    env = _cargo_env(repo)
    result = subprocess.run(
        ["cargo", "build", "--release", "--workspace"],
        cwd=str(repo),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        print("[setup] Build FAILED.  Check output above for errors.")
        return False
    # Copy node binary to repo/bin/
    src = repo / "target" / "release" / ("w1z4rdv1510n-node.exe" if platform.system() == "Windows" else "w1z4rdv1510n-node")
    dst_dir = repo / "bin"
    dst_dir.mkdir(exist_ok=True)
    dst = dst_dir / ("w1z4rd_node.exe" if platform.system() == "Windows" else "w1z4rd_node")
    if src.exists():
        shutil.copy2(str(src), str(dst))
        print(f"[setup] Binary copied to {dst}")
    return True


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def start_node(repo: Path) -> bool:
    binary = repo / "bin" / ("w1z4rd_node.exe" if platform.system() == "Windows" else "w1z4rd_node")
    if not binary.exists():
        print(f"[setup] Node binary not found at {binary} — build first.")
        return False

    Path(WIZARD_DATA_DIR).mkdir(parents=True, exist_ok=True)
    env = _cargo_env(repo)

    print(f"[setup] Starting W1z4rD node from {repo} …")
    log_path = Path(WIZARD_DATA_DIR) / "node.log"
    with open(log_path, "a") as log_fh:
        subprocess.Popen(
            [str(binary)],
            cwd=str(repo),
            env=env,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
        )

    # Wait up to 15s for the node to come online.
    for attempt in range(1, 16):
        time.sleep(1)
        probe = probe_node()
        if probe["online"]:
            print(f"[setup] Node is online at {probe['endpoint']} (v{probe['version'] or 'unknown'}).")
            print(f"[setup] Logs: {log_path}")
            return True
        if attempt % 5 == 0:
            print(f"[setup] Waiting for node… ({attempt}s)")

    print(f"[setup] Node did not come online within 15s.  Check {log_path} for errors.")
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_check() -> int:
    probe = probe_node()
    if probe["online"]:
        print(f"[setup] W1z4rD node is ONLINE at {probe['endpoint']}  version={probe['version'] or 'unknown'}")
        return 0
    print(f"[setup] W1z4rD node is OFFLINE ({probe['error']})")
    print(f"[setup] Start with:  python scripts/setup_wizard_node.py --start")
    return 1


def cmd_start() -> int:
    probe = probe_node()
    if probe["online"]:
        print(f"[setup] Node already online at {probe['endpoint']}  version={probe['version'] or 'unknown'}")
        return 0

    print("[setup] Node is offline — locating repo …")
    repo = find_repo()
    if repo is None:
        target = Path.home() / "W1z4rDV1510n"
        if not clone_repo(target):
            print("[setup] Could not clone repo.  Set WIZARD_NODE_DIR to the repo path and retry.")
            return 1
        repo = target

    binary = repo / "bin" / ("w1z4rd_node.exe" if platform.system() == "Windows" else "w1z4rd_node")
    if not binary.exists():
        if not build_node(repo):
            return 1

    if not start_node(repo):
        return 1

    print("\n" + TRAINING_SCRIPTS_NOTE)
    return 0


def cmd_train() -> int:
    repo = find_repo()
    print(TRAINING_SCRIPTS_NOTE)
    if repo:
        print(f"[setup] Repo found at: {repo}")
        training_dir = repo / "scripts"
        if training_dir.exists():
            scripts = sorted(training_dir.glob("build_*.py"))
            if scripts:
                print(f"\n[setup] Available training scripts in {training_dir}:")
                for s in scripts:
                    print(f"  {s.name}")
    else:
        print("[setup] Repo not found locally.  Run --start to clone and build.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap the W1z4rD Vision Node.")
    parser.add_argument("--check", action="store_true", help="Probe node and report status")
    parser.add_argument("--start", action="store_true", help="Build and start node if offline")
    parser.add_argument("--train", action="store_true", help="Print training instructions")
    args = parser.parse_args()

    if args.start:
        return cmd_start()
    if args.train:
        return cmd_train()
    return cmd_check()


if __name__ == "__main__":
    sys.exit(main())
