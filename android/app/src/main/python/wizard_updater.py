"""
Fetch wizard-node builds published by the C4rr13rX repo.

Runs from ``WizardUpdateWorker`` (WorkManager) and also from the Django route
``/api/wizard-chat/node/update/`` so the node can be updated on demand from the
GUI as well as on a schedule.

The update flow, and why each step is there:

1. **Read a manifest** describing the current build (version, arch, sha256,
   URL). A manifest rather than a bare download lets the device decide whether
   it needs the bytes at all -- most checks end here, costing one small GET.
2. **Compare against the installed version.** No version change, no download.
3. **Download to a temp file and verify the sha256 before installing.** A
   truncated or tampered binary must never replace a working node.
4. **Install atomically** (``os.replace``), keeping one backup.

Step 3 is the important one. This project has already been bitten by a node
process running a stale binary whose routes 404'd -- see the
wizard-node-deployment notes. Verifying the hash and recording the version is
what makes "which build is actually running?" answerable instead of a guess.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger("android.wizard_updater")

# Where C4rr13rX publishes node builds. Overridable so a local checkout or a
# staging bucket can be pointed at without a rebuild.
MANIFEST_URL = os.getenv(
    "WIZARD_UPDATE_MANIFEST",
    "https://c4rr13rx.com/wizard-node/manifest.json",
)

# Android is arm64 in practice; the manifest may carry several builds.
TARGET_ARCH = os.getenv("WIZARD_NODE_ARCH", "aarch64-linux-android")

DOWNLOAD_TIMEOUT = int(os.getenv("WIZARD_UPDATE_TIMEOUT", "120"))
# A node binary is tens of MB. Anything far larger is a misconfigured URL
# (an HTML error page, a redirect loop) rather than a real build.
MAX_BYTES = int(os.getenv("WIZARD_UPDATE_MAX_BYTES", str(256 * 1024 * 1024)))


def _paths(files_dir: str) -> dict[str, Path]:
    home = Path(files_dir) / "wizard"
    return {
        "home": home,
        "bin": home / "bin",
        "binary": home / "bin" / "w1z4rd_node",
        "state": home / "installed.json",
    }


def installed_version(files_dir: str) -> dict[str, Any]:
    """What is on disk right now, per our own record."""
    state = _paths(files_dir)["state"]
    if not state.exists():
        return {"version": None, "sha256": None}
    try:
        return json.loads(state.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": None, "sha256": None}


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "coolcrypto-android/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _select_build(manifest: dict) -> dict | None:
    """Pick the build matching this device's architecture."""
    builds = manifest.get("builds") or []
    for build in builds:
        if build.get("arch") == TARGET_ARCH:
            return build
    # A single-build manifest with no arch field is treated as ours; anything
    # else would silently install a binary for the wrong architecture.
    if len(builds) == 1 and not builds[0].get("arch"):
        return builds[0]
    return None


def _download_verified(url: str, expected_sha: str) -> Path:
    """Download to a temp file and verify before it can replace anything."""
    digest = hashlib.sha256()
    total = 0
    handle, tmp_name = tempfile.mkstemp(prefix="w1z4rd-node-")
    tmp = Path(tmp_name)

    req = urllib.request.Request(url, headers={"User-Agent": "coolcrypto-android/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp, \
                os.fdopen(handle, "wb") as out:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ValueError(f"download exceeded {MAX_BYTES} bytes")
                digest.update(chunk)
                out.write(chunk)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    actual = digest.hexdigest()
    if expected_sha and actual.lower() != expected_sha.lower():
        tmp.unlink(missing_ok=True)
        raise ValueError(f"sha256 mismatch: expected {expected_sha}, got {actual}")
    return tmp


def check_and_stage(files_dir: str, force: bool = False) -> dict:
    """
    Check for a newer node build and install it.

    Returns ``{"status": "unchanged" | "installed" | "error", ...}``. Never
    raises: the caller is a WorkManager job and a crash there is retried
    blindly, which is worse than a reported error.
    """
    paths = _paths(files_dir)
    try:
        manifest = _fetch_json(MANIFEST_URL)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning("manifest fetch failed: %s", exc)
        return {"status": "error", "error": f"manifest: {exc}"}

    build = _select_build(manifest)
    if not build:
        return {"status": "error", "error": f"no build for {TARGET_ARCH}"}

    version = str(build.get("version") or manifest.get("version") or "")
    current = installed_version(files_dir)
    if not force and version and version == current.get("version"):
        return {"status": "unchanged", "version": version}

    url = build.get("url")
    if not url:
        return {"status": "error", "error": "build has no url"}

    try:
        tmp = _download_verified(url, str(build.get("sha256") or ""))
    except Exception as exc:  # noqa: BLE001
        logger.warning("download failed: %s", exc)
        return {"status": "error", "error": f"download: {exc}"}

    try:
        paths["bin"].mkdir(parents=True, exist_ok=True)
        target = paths["binary"]

        # Keep exactly one backup: enough to roll back a bad build, not enough
        # to fill a phone with old binaries.
        if target.exists():
            shutil.copy2(target, target.with_suffix(".bak"))

        # os.replace is atomic on the same filesystem: the node is never
        # observed half-written, even if the process dies here.
        os.replace(tmp, target)
        target.chmod(0o755)

        paths["state"].write_text(json.dumps({
            "version": version,
            "sha256": build.get("sha256"),
            "installed_at": time.time(),
            "url": url,
        }, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        Path(tmp).unlink(missing_ok=True)
        logger.exception("install failed")
        return {"status": "error", "error": f"install: {exc}"}

    logger.info("installed wizard node %s", version)
    return {"status": "installed", "version": version,
            "path": str(paths["binary"])}


def status(files_dir: str) -> dict:
    """Current install state, for the Django route and the UI."""
    paths = _paths(files_dir)
    current = installed_version(files_dir)
    staged = paths["binary"]
    return {
        "installed_version": current.get("version"),
        "installed_at": current.get("installed_at"),
        "staged_binary": str(staged) if staged.exists() else None,
        "manifest_url": MANIFEST_URL,
        "arch": TARGET_ARCH,
    }
