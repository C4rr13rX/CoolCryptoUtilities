#!/usr/bin/env python3
"""
Revenir Delegation Service — Cross-platform Installer

Run this single file on any target machine to set up the delegation service.
Supports: Windows, macOS, Linux, Userland Ubuntu (Android), Termux.

Usage:
    python install.py                      # Interactive install
    python install.py --port 7782          # Specify port
    python install.py --token <token>      # Pre-pair with API token
    python install.py --uninstall          # Remove the service
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SERVICE_NAME = "revenir-delegation"
DEFAULT_PORT = 7782
REQUIRED_PIP_PACKAGES = []  # stdlib-only core; psutil is optional but recommended
OPTIONAL_PIP_PACKAGES = ["psutil"]  # Better resource monitoring
ML_PIP_PACKAGES = ["numpy"]  # For training capability


def main() -> None:
    parser = argparse.ArgumentParser(description="Install Revenir Delegation Service")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--token", type=str, default="")
    parser.add_argument("--callback-url", type=str, default="")
    parser.add_argument("--install-dir", type=str, default="")
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--skip-deps", action="store_true")
    args = parser.parse_args()

    env = detect_environment()
    print_banner(env)

    if args.uninstall:
        uninstall(env)
        return

    install_dir = Path(args.install_dir) if args.install_dir else default_install_dir(env)
    install(env, install_dir, args.port, args.token, args.callback_url, args.skip_deps)


def detect_environment() -> dict:
    """Detect what kind of system we're running on."""
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "is_userland": os.path.exists("/data/data/tech.ula") or bool(os.getenv("USERLAND")),
        "is_termux": os.path.exists("/data/data/com.termux") or bool(os.getenv("TERMUX_VERSION")),
        "is_android": False,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "has_systemd": shutil.which("systemctl") is not None,
        "has_launchctl": shutil.which("launchctl") is not None,
        "is_root": False,
    }

    # Android detection
    if info["is_userland"] or info["is_termux"]:
        info["is_android"] = True
    elif os.path.exists("/system/build.prop"):
        info["is_android"] = True

    # Root check
    try:
        info["is_root"] = os.geteuid() == 0
    except AttributeError:
        pass  # Windows

    return info


def default_install_dir(env: dict) -> Path:
    if env["os"] == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "RevenirService"
    elif env["os"] == "Darwin":
        return Path.home() / "Library" / "RevenirService"
    elif env["is_userland"] or env["is_termux"]:
        return Path.home() / ".revenir-service"
    else:
        return Path.home() / ".revenir-service"


def print_banner(env: dict) -> None:
    device = "Android (Userland)" if env["is_userland"] else \
             "Android (Termux)" if env["is_termux"] else \
             env["os"]
    print(f"""
  =============================================
   Revenir Delegation Service - Installer
  =============================================
   OS:      {device} ({env['arch']})
   Python:  {env['python_version']}
  =============================================
""")


def install(env: dict, install_dir: Path, port: int, token: str, callback_url: str, skip_deps: bool) -> None:
    print(f"[1/5] Creating install directory: {install_dir}")
    install_dir.mkdir(parents=True, exist_ok=True)
    service_dir = install_dir / "revenir_service"
    service_dir.mkdir(exist_ok=True)

    # Copy service files
    print("[2/5] Copying service files...")
    src_dir = Path(__file__).parent
    for f in src_dir.glob("*.py"):
        shutil.copy2(f, service_dir / f.name)

    # Write config
    import json
    config = {"port": port, "api_token": token, "callback_url": callback_url}
    work_dir = install_dir / "data"
    work_dir.mkdir(exist_ok=True)
    (work_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Install dependencies
    if not skip_deps:
        print("[3/5] Installing dependencies...")
        install_dependencies(env)
    else:
        print("[3/5] Skipping dependency installation")

    # Create startup script
    print("[4/5] Creating startup script...")
    create_startup_script(env, install_dir, port)

    # Register as system service (optional)
    print("[5/5] Registering service...")
    register_service(env, install_dir, port)

    print(f"""
  =============================================
   Installation complete!
  =============================================
   Install dir: {install_dir}
   Port:        {port}
   Token:       {"configured" if token else "NOT SET (use pairing mode)"}

   To start manually:
     {_start_command(env, install_dir)}

   To pair with R3v3n!R:
     1. Open the Pipeline page in the R3v3n!R GUI
     2. Click "+ Add Host"
     3. Enter this machine's IP and port {port}
     4. Copy the API token shown and pass it with --token
        OR: the service will accept the first pairing request
  =============================================
""")
    if env["is_userland"] or env["is_termux"]:
        print("""
  ** ANDROID IMPORTANT **
  To prevent Android from killing the service when backgrounded:
    1. Go to Settings > Apps > UserLand (or Termux)
    2. Battery > "Don't optimize" (or "Unrestricted")
    3. Also disable "Remove permissions if app is unused"
  Without this, Android will kill the service within minutes
  of switching to another app.
""")


def install_dependencies(env: dict) -> None:
    pip = [sys.executable, "-m", "pip", "install", "--user", "--quiet"]

    # On Userland Ubuntu, some packages need special handling
    if env["is_userland"]:
        print("  (Userland Ubuntu detected — using compatible packages)")
        # psutil may fail on Userland due to missing /proc access
        for pkg in OPTIONAL_PIP_PACKAGES:
            try:
                _run(pip + [pkg])
                print(f"  + {pkg}")
            except Exception:
                print(f"  ~ {pkg} (skipped — fallback /proc monitoring will be used)")
        # numpy: pip wheels don't work on aarch64 Android, use apt instead
        try:
            _run(["apt", "install", "-y", "python3-numpy"])
            print("  + numpy (via apt)")
        except Exception:
            try:
                _run(pip + ["numpy"])
                print("  + numpy (via pip)")
            except Exception:
                print("  ~ numpy (skipped — training will not be available)")

    elif env["is_termux"]:
        print("  (Termux detected — using pkg where needed)")
        # Termux has its own package manager for native libs
        try:
            _run(["pkg", "install", "-y", "python-numpy"])
        except Exception:
            try:
                _run(pip + ["numpy"])
            except Exception:
                print("  ~ numpy (skipped)")
        for pkg in OPTIONAL_PIP_PACKAGES:
            try:
                _run(pip + [pkg])
                print(f"  + {pkg}")
            except Exception:
                print(f"  ~ {pkg} (skipped)")

    else:
        # Standard install
        for pkg in OPTIONAL_PIP_PACKAGES + ML_PIP_PACKAGES:
            try:
                _run(pip + [pkg])
                print(f"  + {pkg}")
            except Exception:
                print(f"  ~ {pkg} (optional, skipped)")


def create_startup_script(env: dict, install_dir: Path, port: int) -> None:
    if env["os"] == "Windows":
        # .bat file
        bat = install_dir / "start_revenir.bat"
        bat.write_text(
            f'@echo off\r\n'
            f'echo Starting Revenir Delegation Service on port {port}...\r\n'
            f'"{sys.executable}" -m revenir_service --port {port} --work-dir "{install_dir / "data"}"\r\n'
            f'pause\r\n',
            encoding="utf-8",
        )
        # .ps1 for PowerShell
        ps1 = install_dir / "start_revenir.ps1"
        ps1.write_text(
            f'Write-Host "Starting Revenir Delegation Service on port {port}..."\n'
            f'& "{sys.executable}" -m revenir_service --port {port} --work-dir "{install_dir / "data"}"\n',
            encoding="utf-8",
        )
        print(f"  Created: {bat}")
    else:
        # Shell script
        sh = install_dir / "start_revenir.sh"
        sh.write_text(
            f'#!/bin/bash\n'
            f'echo "Starting Revenir Delegation Service on port {port}..."\n'
            f'exec "{sys.executable}" -m revenir_service --port {port} --work-dir "{install_dir / "data"}"\n',
            encoding="utf-8",
        )
        sh.chmod(0o755)
        print(f"  Created: {sh}")


def register_service(env: dict, install_dir: Path, port: int) -> None:
    work_dir = install_dir / "data"
    python = sys.executable

    if env["os"] == "Windows":
        _register_windows_service(install_dir, python, port, work_dir)
    elif env["os"] == "Darwin" and env["has_launchctl"]:
        _register_launchd(install_dir, python, port, work_dir)
    elif env["has_systemd"] and not env["is_android"]:
        _register_systemd(install_dir, python, port, work_dir, env["is_root"])
    elif env["is_userland"] or env["is_termux"]:
        # No systemd on Android — use a cron-based keepalive
        _register_cron_keepalive(install_dir, python, port, work_dir)
    else:
        print("  No service manager detected — start manually with the startup script")


def _register_windows_service(install_dir: Path, python: str, port: int, work_dir: Path) -> None:
    """Create a Windows Task Scheduler entry to auto-start on login."""
    xml = f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Revenir Delegation Service</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger><Enabled>true</Enabled></LogonTrigger>
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
  </Settings>
  <Actions>
    <Exec>
      <Command>{python}</Command>
      <Arguments>-m revenir_service --port {port} --work-dir "{work_dir}"</Arguments>
      <WorkingDirectory>{install_dir}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>'''
    xml_path = install_dir / "revenir_task.xml"
    xml_path.write_text(xml, encoding="utf-16")
    try:
        _run(["schtasks", "/Create", "/TN", SERVICE_NAME, "/XML", str(xml_path), "/F"])
        print(f"  Registered Windows Task Scheduler entry: {SERVICE_NAME}")
    except Exception as exc:
        print(f"  Could not register scheduled task (run as admin): {exc}")
        print(f"  You can import manually: schtasks /Create /TN {SERVICE_NAME} /XML \"{xml_path}\"")


def _register_launchd(install_dir: Path, python: str, port: int, work_dir: Path) -> None:
    """Create a macOS launchd plist for auto-start."""
    plist_dir = Path.home() / "Library" / "LaunchAgents"
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path = plist_dir / "com.revenir.delegation.plist"
    plist = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.revenir.delegation</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>-m</string>
        <string>revenir_service</string>
        <string>--port</string>
        <string>{port}</string>
        <string>--work-dir</string>
        <string>{work_dir}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{install_dir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{work_dir / "service.log"}</string>
    <key>StandardErrorPath</key>
    <string>{work_dir / "service.err.log"}</string>
</dict>
</plist>'''
    plist_path.write_text(plist)
    try:
        _run(["launchctl", "load", str(plist_path)])
        print(f"  Registered launchd service: com.revenir.delegation")
    except Exception as exc:
        print(f"  launchctl load failed: {exc}")
        print(f"  Load manually: launchctl load {plist_path}")


def _register_systemd(install_dir: Path, python: str, port: int, work_dir: Path, is_root: bool) -> None:
    """Create a systemd user service."""
    if is_root:
        unit_dir = Path("/etc/systemd/system")
    else:
        unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path = unit_dir / f"{SERVICE_NAME}.service"
    unit = f'''[Unit]
Description=Revenir Delegation Service
After=network.target

[Service]
Type=simple
ExecStart={python} -m revenir_service --port {port} --work-dir {work_dir}
WorkingDirectory={install_dir}
Restart=on-failure
RestartSec=10

[Install]
WantedBy={"multi-user.target" if is_root else "default.target"}
'''
    unit_path.write_text(unit)
    scope = [] if is_root else ["--user"]
    try:
        _run(["systemctl"] + scope + ["daemon-reload"])
        _run(["systemctl"] + scope + ["enable", SERVICE_NAME])
        _run(["systemctl"] + scope + ["start", SERVICE_NAME])
        print(f"  Registered systemd service: {SERVICE_NAME}")
    except Exception as exc:
        print(f"  systemd registration failed: {exc}")
        print(f"  Enable manually: systemctl {'--user ' if not is_root else ''}enable --now {SERVICE_NAME}")


def _register_cron_keepalive(install_dir: Path, python: str, port: int, work_dir: Path) -> None:
    """For Android Userland/Termux — use cron to keep the service alive."""
    script = install_dir / "start_revenir.sh"
    check_script = install_dir / "keepalive.sh"
    check_script.write_text(
        f'#!/bin/bash\n'
        f'pgrep -f "revenir_service" > /dev/null || nohup "{script}" > "{work_dir}/service.log" 2>&1 &\n',
        encoding="utf-8",
    )
    check_script.chmod(0o755)

    # Add to crontab
    try:
        existing = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
        if "keepalive.sh" not in existing:
            new_crontab = existing.rstrip() + f"\n*/5 * * * * {check_script}\n"
            proc = subprocess.run(["crontab", "-"], input=new_crontab, text=True)
            if proc.returncode == 0:
                print("  Added cron keepalive (checks every 5 minutes)")
            else:
                print("  crontab update failed — add manually:")
                print(f"    */5 * * * * {check_script}")
    except Exception as exc:
        print(f"  cron setup failed: {exc}")
        print(f"  Start manually: {script}")


def _start_command(env: dict, install_dir: Path) -> str:
    if env["os"] == "Windows":
        return str(install_dir / "start_revenir.bat")
    return str(install_dir / "start_revenir.sh")


def uninstall(env: dict) -> None:
    print("Uninstalling Revenir Delegation Service...")

    if env["os"] == "Windows":
        try:
            _run(["schtasks", "/Delete", "/TN", SERVICE_NAME, "/F"])
            print("  Removed Windows scheduled task")
        except Exception:
            pass
    elif env["os"] == "Darwin":
        plist = Path.home() / "Library" / "LaunchAgents" / "com.revenir.delegation.plist"
        if plist.exists():
            try:
                _run(["launchctl", "unload", str(plist)])
                plist.unlink()
                print("  Removed launchd service")
            except Exception:
                pass
    elif env["has_systemd"] and not env["is_android"]:
        scope = ["--user"]
        try:
            _run(["systemctl"] + scope + ["stop", SERVICE_NAME])
            _run(["systemctl"] + scope + ["disable", SERVICE_NAME])
            unit = Path.home() / ".config" / "systemd" / "user" / f"{SERVICE_NAME}.service"
            if unit.exists():
                unit.unlink()
            _run(["systemctl"] + scope + ["daemon-reload"])
            print("  Removed systemd service")
        except Exception:
            pass

    install_dir = default_install_dir(env)
    if install_dir.exists():
        confirm = input(f"  Remove {install_dir}? [y/N] ").strip().lower()
        if confirm == "y":
            shutil.rmtree(install_dir)
            print(f"  Removed {install_dir}")

    print("  Uninstall complete.")


def _run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)


if __name__ == "__main__":
    main()
