from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional


DEV_BRANCH = "branddozer/development"
STABLE_BRANCH = "main"
_PREVIEWS: Dict[str, subprocess.Popen] = {}
_PREVIEW_META: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.RLock()


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(root), text=True, capture_output=True,
        timeout=60, check=check,
    )


def _has_head(root: Path) -> bool:
    return _git(root, "rev-parse", "--verify", "HEAD", check=False).returncode == 0


def ensure_lifecycle(project: Dict[str, Any]) -> Dict[str, Any]:
    root = Path(project["root_path"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    control = root / ".branddozer"
    control.mkdir(exist_ok=True)
    config = {
        "project_id": project.get("id"),
        "workflow_kind": project.get("workflow_kind") or "generic",
        "stable_branch": STABLE_BRANCH,
        "development_branch": DEV_BRANCH,
        "license": project.get("license_key") or "unlicensed",
        "preview": detect_preview(root),
    }
    (control / "project.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _write_launcher(root, config["preview"])
    _apply_license(root, project.get("license_key") or "unlicensed", project.get("name") or root.name)

    if not (root / ".git").exists():
        _git(root, "init", "-b", STABLE_BRANCH)
    _git(root, "config", "user.name", "BrandDozer")
    _git(root, "config", "user.email", "branddozer@localhost")
    if not _has_head(root):
        _git(root, "add", "-A")
        _git(root, "commit", "-m", "BrandDozer initial working baseline", "--allow-empty")
    branches = _git(root, "branch", "--format=%(refname:short)").stdout.splitlines()
    current = _git(root, "branch", "--show-current").stdout.strip()
    if STABLE_BRANCH not in branches:
        _git(root, "branch", STABLE_BRANCH, current or "HEAD")
    if DEV_BRANCH not in branches:
        _git(root, "branch", DEV_BRANCH, STABLE_BRANCH)
    return lifecycle_status(project)


def prepare_cycle(project: Dict[str, Any]) -> Dict[str, Any]:
    ensure_lifecycle(project)
    root = Path(project["root_path"]).resolve()
    _git(root, "checkout", DEV_BRANCH)
    if _git(root, "status", "--porcelain").stdout.strip():
        _git(root, "add", "-A")
        _git(root, "commit", "-m", "BrandDozer pre-cycle checkpoint")
    return lifecycle_status(project)


def finalize_cycle(project: Dict[str, Any], *, success: bool, message: str = "") -> Dict[str, Any]:
    root = Path(project["root_path"]).resolve()
    _git(root, "add", "-A")
    if _git(root, "status", "--porcelain").stdout.strip():
        _git(root, "commit", "-m", message[:180] or "BrandDozer refinement cycle")
    promoted = False
    if success and bool(project.get("git_auto_promote", True)):
        _git(root, "checkout", STABLE_BRANCH)
        merged = _git(root, "merge", "--ff-only", DEV_BRANCH, check=False)
        if merged.returncode == 0:
            promoted = True
        _git(root, "checkout", DEV_BRANCH)
    status = lifecycle_status(project)
    status["promoted"] = promoted
    return status


def detect_preview(root: Path) -> Dict[str, Any]:
    candidates = [root / "site" / "index.html", root / "dist" / "index.html", root / "build" / "index.html", root / "index.html"]
    for index in candidates:
        if index.exists():
            # Serve the workspace root so storefront links can reach sibling
            # product artifacts instead of being trapped under site/.
            return {"kind": "static_web", "cwd": str(root), "entry": index.relative_to(root).as_posix()}
    if (root / "package.json").exists():
        try:
            scripts = json.loads((root / "package.json").read_text(encoding="utf-8")).get("scripts", {})
        except Exception:
            scripts = {}
        for name in ("dev", "start", "preview"):
            if name in scripts:
                return {"kind": "node_script", "cwd": str(root), "script": name}
    for exe in root.glob("*.exe"):
        return {"kind": "executable", "cwd": str(root), "entry": str(exe)}
    for pattern, kind in (("*.pdf", "document"), ("*.docx", "document"), ("*.html", "static_web")):
        item = next(root.glob(pattern), None)
        if item:
            if kind == "static_web":
                return {"kind": kind, "cwd": str(root), "entry": item.name}
            return {"kind": kind, "cwd": str(root), "entry": str(item)}
    return {"kind": "workspace", "cwd": str(root)}


def _write_launcher(root: Path, preview: Dict[str, Any]) -> None:
    launcher = root / ".branddozer" / "start.ps1"
    kind = preview.get("kind")
    if kind == "static_web":
        body = f"Set-Location -LiteralPath '{preview['cwd'].replace("'", "''")}'\nStart-Process 'http://127.0.0.1:8765/{preview['entry']}'\npython -m http.server 8765\n"
    elif kind == "node_script":
        body = f"Set-Location -LiteralPath '{str(root).replace("'", "''")}'\nnpm run {preview['script']}\n"
    elif kind == "executable":
        body = f"Start-Process -FilePath '{preview['entry'].replace("'", "''")}'\n"
    elif kind == "document":
        body = f"Start-Process -FilePath '{preview['entry'].replace("'", "''")}'\n"
    else:
        body = f"Start-Process explorer.exe -ArgumentList '{str(root).replace("'", "''")}'\n"
    launcher.write_text(body, encoding="utf-8")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def launch_preview(project: Dict[str, Any], *, open_on_pc: bool = True) -> Dict[str, Any]:
    status = ensure_lifecycle(project)
    preview = status["preview"]
    key = str(project["id"])
    kind = preview.get("kind")
    with _LOCK:
        old = _PREVIEWS.get(key)
        if old and old.poll() is None:
            return _PREVIEW_META[key]
        if kind == "static_web":
            port = _free_port()
            proc = subprocess.Popen(
                [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                cwd=preview["cwd"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            result = {"running": True, "kind": kind, "url": f"http://127.0.0.1:{port}/{preview.get('entry', '')}", "pid": proc.pid}
        elif kind == "executable":
            proc = subprocess.Popen([preview["entry"]], cwd=preview["cwd"])
            result = {"running": True, "kind": kind, "url": "", "pid": proc.pid}
        elif kind == "document":
            os.startfile(preview["entry"])
            return {"running": True, "kind": kind, "url": "", "opened": preview["entry"]}
        elif kind == "node_script":
            port = _free_port()
            env = dict(os.environ)
            env.setdefault("PORT", str(port))
            npm = shutil.which("npm.cmd") or shutil.which("npm")
            if not npm:
                raise RuntimeError("npm is required to preview this project")
            proc = subprocess.Popen(
                [npm, "run", preview["script"], "--", "--host", "127.0.0.1", "--port", str(port)],
                cwd=preview["cwd"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            result = {"running": True, "kind": kind, "url": f"http://127.0.0.1:{port}/", "pid": proc.pid}
        else:
            subprocess.Popen(["explorer.exe", preview["cwd"]])
            return {"running": True, "kind": kind, "url": "", "opened": preview["cwd"], "launcher": status["launcher"]}
        _PREVIEWS[key], _PREVIEW_META[key] = proc, result
    if open_on_pc and result.get("url"):
        webbrowser.open(result["url"])
    return result


def lifecycle_status(project: Dict[str, Any]) -> Dict[str, Any]:
    root = Path(project["root_path"]).expanduser().resolve()
    preview = detect_preview(root)
    git = {"initialized": (root / ".git").exists(), "stable_branch": STABLE_BRANCH, "development_branch": DEV_BRANCH}
    if git["initialized"]:
        git["current_branch"] = _git(root, "branch", "--show-current", check=False).stdout.strip()
        git["head"] = _git(root, "rev-parse", "--short", "HEAD", check=False).stdout.strip()
        git["dirty"] = bool(_git(root, "status", "--porcelain", check=False).stdout.strip())
    key = str(project.get("id"))
    proc = _PREVIEWS.get(key)
    running = bool(proc and proc.poll() is None)
    active_work: Dict[str, Any] = {}
    active_path = root / "runtime" / "active-product.json"
    try:
        active = json.loads(active_path.read_text(encoding="utf-8"))
        spec = active.get("spec") if isinstance(active.get("spec"), dict) else {}
        active_work = {
            "name": spec.get("name") or active.get("slug"),
            "status": active.get("status"),
            "inner_iteration": active.get("inner_iteration", 0),
            "validation_passed": bool((active.get("last_validation") or {}).get("passed")),
            "validation_error": (active.get("last_validation") or {}).get("error", ""),
            "primary_artifact": active.get("primary_artifact", ""),
        }
    except Exception:
        active_work = {}
    return {
        "working_on": active_work.get("name") or project.get("workflow_config", {}).get("mission") or project.get("default_prompt") or project.get("name"),
        "active_work": active_work,
        "workflow_kind": project.get("workflow_kind") or "generic",
        "preview": preview,
        "preview_running": running,
        "preview_url": (_PREVIEW_META.get(key) or {}).get("url", "") if running else "",
        "launcher": str(root / ".branddozer" / "start.ps1"),
        "license": project.get("license_key") or "unlicensed",
        "git": git,
    }


def _apply_license(root: Path, key: str, name: str) -> None:
    path = root / "LICENSE"
    marker = root / ".branddozer" / "license-managed"
    # Never replace or remove a license that BrandDozer did not create.
    if path.exists() and not marker.exists():
        return
    if key == "mit":
        year = time.gmtime().tm_year
        path.write_text(f"MIT License\n\nCopyright (c) {year} {name}\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n", encoding="utf-8")
        marker.write_text("mit", encoding="utf-8")
    elif key == "proprietary":
        path.write_text(f"Copyright {time.gmtime().tm_year} {name}. All rights reserved.\nNo permission is granted to copy, modify, distribute, sublicense, or sell this work.\n", encoding="utf-8")
        marker.write_text("proprietary", encoding="utf-8")
    elif path.exists() and key == "unlicensed" and marker.exists():
        path.unlink()
        marker.unlink(missing_ok=True)
