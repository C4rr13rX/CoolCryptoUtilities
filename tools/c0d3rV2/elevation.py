"""tools/c0d3rV2/elevation.py — cross-OS privilege elevation.

Every function in this module spawns a command via the host OS's NATIVE
authentication dialog: UAC on Windows, polkit/pkexec or sudo+graphical
askpass on Linux, AuthorizationServices via osascript on macOS.

This module never sees the user's password.  It hands off authentication
to the OS, the OS prompts the user, and only after the OS approves does
the elevated command run.  No credentials are cached — every elevation
requires fresh user authentication.

Returns (return_code, stdout, stderr) tuples just like the regular
Executor so calling code can swap between elevated/non-elevated.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


DEFAULT_TIMEOUT_S: int = 300


# ── Error sentinels ─────────────────────────────────────────────────────────

class ElevationUnavailable(RuntimeError):
    """No usable elevation mechanism on this host (no UAC, no polkit agent,
    no ssh-askpass, no osascript).  Caller should surface this to the user."""


class ElevationDenied(RuntimeError):
    """User cancelled the OS auth dialog or supplied wrong credentials."""


# ── Public entry point ──────────────────────────────────────────────────────

def run_elevated(command: str, *, cwd: Path | None = None,
                  timeout_s: int = DEFAULT_TIMEOUT_S) -> tuple[int, str, str]:
    """Run `command` with elevated privileges via OS-native auth dialog.

    Returns (return_code, stdout, stderr).  Raises ElevationUnavailable if
    no elevation mechanism exists, or returns rc=126 if the user cancelled.
    """
    if not command or not command.strip():
        return 1, "", "empty command"
    if os.name == "nt":
        return _run_windows_uac(command, cwd=cwd, timeout_s=timeout_s)
    if sys.platform == "darwin":
        return _run_macos_osascript(command, cwd=cwd, timeout_s=timeout_s)
    return _run_linux(command, cwd=cwd, timeout_s=timeout_s)


def elevation_method() -> str:
    """Return a short human-readable name for the elevation mechanism that
    will be used on this host.  Useful for the audit log and UI hints."""
    if os.name == "nt":
        return "Windows UAC (Start-Process -Verb RunAs)"
    if sys.platform == "darwin":
        return "macOS AuthorizationServices (osascript)"
    if shutil.which("pkexec"):
        return "Linux polkit (pkexec)"
    if shutil.which("sudo") and _has_graphical_askpass():
        return "Linux sudo + graphical askpass"
    if shutil.which("sudo"):
        return "Linux sudo (terminal — may not prompt in web context)"
    return "unavailable"


# ── Windows ─────────────────────────────────────────────────────────────────

def _run_windows_uac(command: str, *, cwd: Path | None,
                       timeout_s: int) -> tuple[int, str, str]:
    """Trigger UAC via Start-Process -Verb RunAs.  The elevated process runs
    detached, so we redirect its stdout/stderr/exitcode to temp files and
    read them back after Wait-Process exits."""
    out_file = Path(tempfile.mkstemp(prefix="c0d3rv2_elev_out_", suffix=".log")[1])
    err_file = Path(tempfile.mkstemp(prefix="c0d3rv2_elev_err_", suffix=".log")[1])
    try:
        # Inner script: cd to cwd, run the user command, capture rc.  Quoted
        # so PowerShell sees it as one literal argument.
        cwd_str = str(cwd) if cwd else os.getcwd()
        inner = (
            f"Set-Location -Path '{_ps_escape(cwd_str)}'; "
            f"& {{ {command} }} *> '{_ps_escape(str(out_file))}'; "
            f"$LASTEXITCODE | Out-File -FilePath "
            f"'{_ps_escape(str(err_file))}' -Encoding ascii"
        )
        # Outer script: launch the inner script elevated, wait for completion.
        outer = (
            f"$p = Start-Process -FilePath powershell.exe "
            f"-ArgumentList '-NoProfile','-NonInteractive','-Command',"
            f"\"{_ps_escape(inner)}\" -Verb RunAs -PassThru -WindowStyle Hidden; "
            f"if ($null -ne $p) {{ $p.WaitForExit({timeout_s * 1000}); $p.ExitCode }}"
        )
        try:
            launch = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", outer],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout_s + 30,
            )
        except subprocess.TimeoutExpired:
            return 124, "", f"elevation timed out after {timeout_s}s"
        # If the user cancelled the UAC prompt, Start-Process raises and the
        # outer script returns a non-zero exit with a message in stderr.
        if "was canceled by the user" in (launch.stderr or "") or \
           "was cancelled by the user" in (launch.stderr or ""):
            return 126, "", "user cancelled UAC prompt"
        # Read captured output.
        try:
            stdout = out_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            stdout = ""
        try:
            rc_text = err_file.read_text(encoding="utf-8", errors="replace").strip()
            inner_rc = int(rc_text) if rc_text else 0
        except (OSError, ValueError):
            inner_rc = 0
        return inner_rc, stdout, launch.stderr or ""
    finally:
        for f in (out_file, err_file):
            try: f.unlink(missing_ok=True)
            except OSError: pass


def _ps_escape(s: str) -> str:
    """Escape a string for embedding inside a single-quoted PowerShell literal.
    PowerShell only requires doubling embedded single quotes."""
    return s.replace("'", "''")


# ── macOS ───────────────────────────────────────────────────────────────────

def _run_macos_osascript(command: str, *, cwd: Path | None,
                           timeout_s: int) -> tuple[int, str, str]:
    """Use AppleScript's `do shell script ... with administrator privileges`
    to trigger the macOS authentication dialog.  osascript itself returns
    the elevated process's stdout."""
    cwd_str = str(cwd) if cwd else os.getcwd()
    inner = f"cd {shlex.quote(cwd_str)} && ({command})"
    script = f'do shell script {_as_escape(inner)} with administrator privileges'
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"elevation timed out after {timeout_s}s"
    except FileNotFoundError:
        raise ElevationUnavailable("osascript not found") from None
    # User cancelled → osascript returns "User canceled. (-128)" on stderr.
    if "User canceled" in result.stderr or result.returncode == 1 and "-128" in result.stderr:
        return 126, "", "user cancelled auth dialog"
    return result.returncode, result.stdout, result.stderr


def _as_escape(s: str) -> str:
    """Quote a string for AppleScript: wrap in double quotes, escape \\ and \"."""
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


# ── Linux ───────────────────────────────────────────────────────────────────

def _run_linux(command: str, *, cwd: Path | None,
                timeout_s: int) -> tuple[int, str, str]:
    """Try pkexec (graphical polkit) first, fall back to sudo with a
    graphical askpass program.  If neither is usable, raise."""
    cwd_str = str(cwd) if cwd else os.getcwd()
    # We always wrap the user command in a `sh -c "cd <cwd> && <cmd>"` shim
    # so the elevated process inherits the right working directory.
    wrapped = ["/bin/sh", "-c", f"cd {shlex.quote(cwd_str)} && ({command})"]

    if shutil.which("pkexec"):
        # pkexec inherits stdout/stderr to us; auth dialog comes from polkit.
        try:
            result = subprocess.run(
                ["pkexec", *wrapped],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return 124, "", f"elevation timed out after {timeout_s}s"
        # pkexec exit codes: 126 = not authorised, 127 = auth dialog dismissed.
        if result.returncode in (126, 127):
            return 126, result.stdout, result.stderr or "user cancelled polkit prompt"
        return result.returncode, result.stdout, result.stderr

    if shutil.which("sudo") and _has_graphical_askpass():
        env = os.environ.copy()
        env["SUDO_ASKPASS"] = _find_askpass()
        try:
            result = subprocess.run(
                ["sudo", "--askpass", *wrapped],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout_s, env=env,
            )
        except subprocess.TimeoutExpired:
            return 124, "", f"elevation timed out after {timeout_s}s"
        if result.returncode == 1 and "askpass" in (result.stderr or "").lower():
            return 126, result.stdout, result.stderr or "askpass cancelled"
        return result.returncode, result.stdout, result.stderr

    raise ElevationUnavailable(
        "no usable elevation mechanism on this host: install polkit (pkexec) "
        "or a graphical askpass program (ssh-askpass, x11-ssh-askpass, "
        "lxqt-openssh-askpass) to enable agent admin operations"
    )


_ASKPASS_CANDIDATES = [
    "/usr/lib/ssh/x11-ssh-askpass",
    "/usr/libexec/openssh/ssh-askpass",
    "/usr/bin/ssh-askpass",
    "/usr/bin/lxqt-openssh-askpass",
    "/usr/bin/ksshaskpass",
    "/usr/bin/x11-ssh-askpass",
]


def _has_graphical_askpass() -> bool:
    return _find_askpass() != ""


def _find_askpass() -> str:
    if env := os.environ.get("SUDO_ASKPASS"):
        if Path(env).exists():
            return env
    for path in _ASKPASS_CANDIDATES:
        if Path(path).exists():
            return path
    for name in ("ssh-askpass", "ksshaskpass", "lxqt-openssh-askpass"):
        if found := shutil.which(name):
            return found
    return ""
