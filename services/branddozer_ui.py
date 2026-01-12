from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass
class UISnapshotResult:
    stdout: str
    stderr: str
    exit_code: int
    screenshots: List[Any]
    base_url: str
    server_started: bool
    server_log: Optional[Path]
    meta: Dict[str, Any]


DEFAULT_ROUTES: List[Dict[str, str]] = [
    {"name": "dashboard", "path": "/"},
    {"name": "pipeline", "path": "/pipeline"},
    {"name": "streams", "path": "/streams"},
    {"name": "telemetry", "path": "/telemetry"},
    {"name": "organism", "path": "/organism"},
    {"name": "wallet", "path": "/wallet"},
    {"name": "advisories", "path": "/advisories"},
    {"name": "model-lab", "path": "/lab"},
    {"name": "data-lab", "path": "/datalab"},
    {"name": "guardian", "path": "/guardian"},
    {"name": "code-graph", "path": "/codegraph"},
    {"name": "integrations", "path": "/integrations"},
    {"name": "settings", "path": "/settings"},
    {"name": "branddozer", "path": "/branddozer"},
    {"name": "branddozer-solo", "path": "/branddozer/solo"},
    {"name": "u53rxr080t", "path": "/u53rxr080t"},
]


def _safe_run(
    cmd: List[str],
    *,
    cwd: Path,
    timeout: int,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[str, str, int]:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env or os.environ.copy(),
        )
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        return "", f"command not found: {cmd[0]}", 127
    except subprocess.TimeoutExpired:
        return "", "timeout", 124
    except Exception as exc:
        return "", f"error: {exc}", 1


def _url_ready(url: str, timeout: float = 1.5) -> bool:
    try:
        req = Request(url, headers={"User-Agent": "BrandDozerUI/1.0"})
        with urlopen(req, timeout=timeout) as resp:  # nosec - internal URL check
            return 200 <= resp.status < 500
    except Exception:
        return False


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _parse_host_port(base_url: str) -> Tuple[str, int]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return host, port


def _ensure_frontend_build(root: Path) -> Tuple[bool, str]:
    frontend_dir = root / "web" / "frontend"
    dist_dir = frontend_dir / "dist"
    build_mode = (os.getenv("BRANDDOZER_UI_BUILD") or "auto").strip().lower()
    if dist_dir.exists() and any(dist_dir.iterdir()):
        return True, ""
    if build_mode in {"0", "false", "no", "off"}:
        return False, "frontend build missing; set BRANDDOZER_UI_BUILD=1 or run npm install && npm run build"
    if not shutil.which("npm"):
        return False, "npm not found; cannot build frontend"
    stdout, stderr, code = _safe_run(["npm", "install"], cwd=frontend_dir, timeout=1200)
    if code != 0:
        return False, stderr or stdout or "npm install failed"
    stdout, stderr, code = _safe_run(["npm", "run", "build"], cwd=frontend_dir, timeout=1200)
    if code != 0:
        return False, stderr or stdout or "npm run build failed"
    return True, ""


def _generate_password() -> str:
    import secrets

    return f"bdz_{secrets.token_urlsafe(12)}"


def _ensure_admin_user(root: Path, username: str, password: str, email: str, reset: bool) -> Tuple[bool, str]:
    manage_py = root / "web" / "manage.py"
    if not manage_py.exists():
        return False, "manage.py not found; cannot create admin user"
    cmd = [
        "python",
        str(manage_py),
        "ensure_ui_admin",
        "--username",
        username,
        "--password",
        password,
        "--email",
        email,
    ]
    if reset:
        cmd.append("--reset")
    stdout, stderr, code = _safe_run(cmd, cwd=root, timeout=60)
    if code != 0:
        return False, stderr or stdout or "Failed to create admin user"
    return True, ""


def _playwright_browsers_ready() -> bool:
    path_override = os.getenv("PLAYWRIGHT_BROWSERS_PATH")
    candidates = []
    if path_override:
        candidates.append(Path(path_override).expanduser())
    candidates.append(Path.home() / ".cache" / "ms-playwright")
    candidates.append(Path.cwd() / ".cache" / "ms-playwright")
    candidates.append(Path.cwd() / "web" / "frontend" / ".cache" / "ms-playwright")
    for base in candidates:
        if not base.exists():
            continue
        try:
            if any(child.name.startswith("chromium") for child in base.iterdir()):
                return True
        except Exception:
            continue
    return False


def _start_django_server(root: Path, host: str, port: int, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("GUARDIAN_AUTO_DISABLED", "1")
    env.setdefault("PRODUCTION_AUTO_DISABLED", "1")
    cmd = ["python", str(root / "web" / "manage.py"), "runserver", f"{host}:{port}", "--noreload"]
    return subprocess.Popen(cmd, cwd=str(root), stdout=log_handle, stderr=log_handle, env=env)


def _wait_for_ready(url: str, timeout_s: int = 60) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _url_ready(url):
            return True
        time.sleep(1.0)
    return False


def _parse_capture_payload(stdout: str) -> Tuple[List[Path], Dict[str, Any]]:
    stdout = stdout.strip()
    if not stdout:
        return [], {}
    try:
        payload = json.loads(stdout)
    except Exception:
        return [], {}
    shots = payload.get("shots") if isinstance(payload, dict) else None
    normalized: List[Dict[str, Any]] = []
    if isinstance(shots, list):
        for entry in shots:
            if isinstance(entry, dict):
                path = entry.get("path")
                if not path:
                    continue
                kind = entry.get("kind") or entry.get("tag")
                viewport = entry.get("viewport") or entry.get("name")
                meta = {k: v for k, v in entry.items() if k not in {"path", "kind", "tag"}}
                if viewport and "viewport" not in meta:
                    meta["viewport"] = viewport
                normalized.append({"path": str(path), "kind": kind, "meta": meta})
            elif isinstance(entry, str):
                normalized.append({"path": entry, "kind": None, "meta": {}})
    return normalized, payload if isinstance(payload, dict) else {}


def capture_ui_screenshots(
    root: Path,
    *,
    output_dir: Path,
    base_url: Optional[str] = None,
    routes: Optional[List[Dict[str, str]]] = None,
    auto_start_server: bool = True,
) -> UISnapshotResult:
    base_url = (base_url or os.getenv("BRANDDOZER_UI_BASE_URL") or "http://127.0.0.1:8000").strip()
    routes = routes or DEFAULT_ROUTES
    offline_mode = (os.getenv("BRANDDOZER_UI_OFFLINE") or "").strip().lower() in {"1", "true", "yes", "on"}
    output_dir.mkdir(parents=True, exist_ok=True)
    frontend_dir = root / "web" / "frontend"
    script_path = frontend_dir / "scripts" / "branddozer_capture.mjs"
    if not script_path.exists():
        return UISnapshotResult(
            stdout="",
            stderr=f"capture script missing: {script_path}",
            exit_code=1,
            screenshots=[],
            base_url=base_url,
            server_started=False,
            server_log=None,
            meta={"dependency_missing": True, "reason": "capture_script_missing"},
        )
    build_ok, build_err = _ensure_frontend_build(root)
    if not build_ok:
        return UISnapshotResult(
            stdout="",
            stderr=build_err,
            exit_code=1,
            screenshots=[],
            base_url=base_url,
            server_started=False,
            server_log=None,
            meta={"dependency_missing": True, "reason": "frontend_build_missing", "error": build_err},
        )
    playwright_dir = frontend_dir / "node_modules" / "playwright"
    if not playwright_dir.exists():
        return UISnapshotResult(
            stdout="",
            stderr="playwright not installed; run `npm install --save-dev playwright` and `npx playwright install` in web/frontend",
            exit_code=1,
            screenshots=[],
            base_url=base_url,
            server_started=False,
            server_log=None,
            meta={"dependency_missing": True, "reason": "playwright_missing"},
        )
    if not _playwright_browsers_ready():
        return UISnapshotResult(
            stdout="",
            stderr="playwright browsers missing; run `npx playwright install chromium` in web/frontend",
            exit_code=1,
            screenshots=[],
            base_url=base_url,
            server_started=False,
            server_log=None,
            meta={"dependency_missing": True, "reason": "playwright_browsers_missing"},
        )
    server_proc = None
    server_log = None
    server_started = False
    host, port = _parse_host_port(base_url)
    if not offline_mode and not _url_ready(base_url):
        start_mode = (os.getenv("BRANDDOZER_UI_START_SERVER") or "auto").strip().lower()
        if start_mode in {"0", "false", "no", "off"} or not auto_start_server:
            return UISnapshotResult(
                stdout="",
                stderr=f"UI not reachable at {base_url}",
                exit_code=1,
                screenshots=[],
                base_url=base_url,
                server_started=False,
                server_log=None,
                meta={},
            )
        if not _port_open(host, port):
            server_log = output_dir / "django-ui.log"
            server_proc = _start_django_server(root, host, port, server_log)
            server_started = True
        if not _wait_for_ready(base_url, timeout_s=60):
            if server_proc:
                server_proc.terminate()
            return UISnapshotResult(
                stdout="",
                stderr=f"UI did not become ready at {base_url}",
                exit_code=1,
                screenshots=[],
                base_url=base_url,
                server_started=server_started,
                server_log=server_log,
                meta={},
            )
    seed_flag = (os.getenv("BRANDDOZER_UI_SEED_ADMIN") or "1").strip().lower()
    if offline_mode:
        seed_flag = "0"
    auth_user = (os.getenv("BRANDDOZER_UI_ADMIN_USER") or "branddozer_qa").strip()
    auth_pass = (os.getenv("BRANDDOZER_UI_ADMIN_PASS") or "").strip()
    auth_email = (os.getenv("BRANDDOZER_UI_ADMIN_EMAIL") or f"{auth_user}@local.test").strip()
    reset_flag = (os.getenv("BRANDDOZER_UI_ADMIN_RESET") or "").strip().lower() in {"1", "true", "yes", "on"}
    if seed_flag not in {"0", "false", "no", "off"}:
        if not auth_pass:
            auth_pass = _generate_password()
            reset_flag = True
        ok, err = _ensure_admin_user(root, auth_user, auth_pass, auth_email, reset_flag)
        if not ok:
            return UISnapshotResult(
                stdout="",
                stderr=err,
                exit_code=1,
                screenshots=[],
                base_url=base_url,
                server_started=server_started,
                server_log=server_log,
                meta={},
            )
    env = os.environ.copy()
    env["BRANDDOZER_BASE_URL"] = base_url
    env["BRANDDOZER_SCREENSHOT_DIR"] = str(output_dir)
    if routes:
        env["BRANDDOZER_ROUTES"] = json.dumps(routes)
    if auth_user and auth_pass:
        env["BRANDDOZER_AUTH_USER"] = auth_user
        env["BRANDDOZER_AUTH_PASS"] = auth_pass
        env["BRANDDOZER_AUTH_LOGIN_PATH"] = os.getenv("BRANDDOZER_AUTH_LOGIN_PATH", "/")
    stdout, stderr, code = _safe_run(
        ["node", str(script_path)],
        cwd=frontend_dir,
        timeout=900,
        env=env,
    )
    screenshots, meta = _parse_capture_payload(stdout)
    tagged = []
    for entry in screenshots:
        if isinstance(entry, dict):
            raw_path = entry.get("path") or ""
            meta_entry = entry.get("meta") or {}
            kind = entry.get("kind")
            viewport = meta_entry.get("viewport") or entry.get("viewport")
        else:
            raw_path = str(entry)
            meta_entry = {}
            kind = None
            viewport = None
        if not raw_path:
            continue
        path_obj = Path(raw_path)
        name = path_obj.name.lower()
        if not kind:
            if viewport:
                if "mobile" in str(viewport).lower():
                    kind = "ui_screenshot_mobile"
                elif "desktop" in str(viewport).lower():
                    kind = "ui_screenshot_desktop"
            if not kind:
                if "mobile" in name:
                    kind = "ui_screenshot_mobile"
                elif "desktop" in name:
                    kind = "ui_screenshot_desktop"
                else:
                    kind = "ui_screenshot"
        tagged.append({"path": str(path_obj), "kind": kind, "meta": meta_entry})
    meta_payload = meta if isinstance(meta, dict) else {}
    if offline_mode:
        meta_payload = {"offline_mode": True, **meta_payload}
    if server_proc:
        server_proc.terminate()
    return UISnapshotResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=code,
        screenshots=tagged,
        base_url=base_url,
        server_started=server_started,
        server_log=server_log,
        meta=meta_payload,
    )
