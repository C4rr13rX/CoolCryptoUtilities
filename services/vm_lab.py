from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VM_ROOT = (PROJECT_ROOT / "storage" / "vm_lab").resolve()
VBOX_HOME = VM_ROOT / "vbox_home"
VBOX_PORTABLE = VM_ROOT / "virtualbox"
IMAGES_DIR = VM_ROOT / "images"
VMS_DIR = VM_ROOT / "vms"
LOGS_DIR = VM_ROOT / "logs"
DOWNLOADS_DIR = VM_ROOT / "downloads"
SSH_DIR = VM_ROOT / "ssh_keys"
STATE_PATH = VM_ROOT / "vm_state.json"
LOG_PATH = LOGS_DIR / "vm_lab.log"
CATALOG_PATH = PROJECT_ROOT / "config" / "vm_lab_catalog.json"


def _ensure_dirs() -> None:
    for path in (VM_ROOT, VBOX_HOME, VBOX_PORTABLE, IMAGES_DIR, VMS_DIR, LOGS_DIR, DOWNLOADS_DIR, SSH_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _log_line(message: str) -> None:
    _ensure_dirs()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _read_text(url: str, timeout: int = 20) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _url_exists(url: str, timeout: int = 10) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def _load_catalog() -> dict:
    if CATALOG_PATH.exists():
        try:
            return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "images": [
            {
                "id": "ubuntu",
                "label": "Ubuntu Desktop (LTS)",
                "os_type": "Ubuntu_64",
                "arch": "amd64",
                "resolver": "ubuntu_lts",
                "url": "",
                "notes": "Auto-resolves latest LTS from releases.ubuntu.com.",
            },
            {
                "id": "kali",
                "label": "Kali Linux",
                "os_type": "Debian_64",
                "arch": "amd64",
                "resolver": "kali_current",
                "url": "",
                "notes": "Auto-resolves current ISO from cdimage.kali.org.",
            },
            {
                "id": "parrot",
                "label": "ParrotOS",
                "os_type": "Debian_64",
                "arch": "amd64",
                "resolver": "parrot_latest",
                "url": "",
                "notes": "Auto-resolves latest ISO from deb.parrot.sh.",
            },
        ]
    }


def _save_catalog(catalog: dict) -> None:
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2), encoding="utf-8")


def _vboxmanage_candidates() -> List[Path]:
    candidates: List[Path] = []
    env_override = os.getenv("C0D3R_VBOXMANAGE")
    if env_override:
        candidates.append(Path(env_override).expanduser())
    if os.name == "nt":
        candidates.extend(
            [
                Path("C:/Program Files/Oracle/VirtualBox/VBoxManage.exe"),
                Path("C:/Program Files (x86)/Oracle/VirtualBox/VBoxManage.exe"),
                VBOX_PORTABLE / "VBoxManage.exe",
            ]
        )
    else:
        candidates.append(VBOX_PORTABLE / "VBoxManage")
    path = shutil.which("VBoxManage")
    if path:
        candidates.append(Path(path))
    return candidates


def _vboxmanage_path() -> Optional[Path]:
    for candidate in _vboxmanage_candidates():
        try:
            if candidate and candidate.exists():
                return candidate.resolve()
        except Exception:
            continue
    return None


def _run_vboxmanage(args: List[str], *, timeout: int = 120) -> Tuple[int, str, str]:
    exe = _vboxmanage_path()
    if not exe:
        return 1, "", "VBoxManage not found"
    _ensure_dirs()
    env = os.environ.copy()
    env["VBOX_USER_HOME"] = str(VBOX_HOME)
    cmd = [str(exe)] + args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        _log_line(f"VBoxManage {' '.join(args)} -> rc={proc.returncode}")
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        _log_line(f"VBoxManage {' '.join(args)} -> timeout")
        return 1, "", f"VBoxManage timeout after {timeout}s"
    except Exception as exc:
        _log_line(f"VBoxManage {' '.join(args)} -> error: {exc}")
        return 1, "", str(exc)


def _version_tuple(text: str) -> Tuple[int, int, int]:
    parts = re.findall(r"\d+", text)
    vals = [int(p) for p in parts[:3]]
    while len(vals) < 3:
        vals.append(0)
    return tuple(vals)  # type: ignore[return-value]


def _guest_additions_path() -> Optional[Path]:
    candidates = []
    if os.name == "nt":
        candidates.append(Path("C:/Program Files/Oracle/VirtualBox/VBoxGuestAdditions.iso"))
        candidates.append(Path("C:/Program Files (x86)/Oracle/VirtualBox/VBoxGuestAdditions.iso"))
    else:
        candidates.append(Path("/usr/share/virtualbox/VBoxGuestAdditions.iso"))
        candidates.append(Path("/usr/lib/virtualbox/additions/VBoxGuestAdditions.iso"))
        candidates.append(Path("/usr/lib/virtualbox/VBoxGuestAdditions.iso"))
    for path in candidates:
        if path.exists():
            return path
    return None


def _guest_os_for_image(image_id: str) -> str:
    image_id = (image_id or "").lower()
    if image_id in {"ubuntu", "kali", "parrot"}:
        return "linux"
    if image_id in {"windows", "win10", "win11", "windows10", "windows11"}:
        return "windows"
    return "unknown"


def _guest_os_for_vm(name: str) -> str:
    state = _load_state().get(name, {})
    guest_os = (state.get("guest_os") or "").lower().strip()
    if guest_os:
        return guest_os
    image_id = state.get("image_id") or ""
    return _guest_os_for_image(str(image_id))


def vm_check_disk(min_free_gb: float | None = None) -> dict:
    _ensure_dirs()
    usage = shutil.disk_usage(str(VM_ROOT))
    free_gb = usage.free / (1024 ** 3)
    min_req = float(min_free_gb) if min_free_gb is not None else float(os.getenv("C0D3R_VM_MIN_FREE_GB", "40") or "40")
    return {
        "ok": free_gb >= min_req,
        "free_gb": round(free_gb, 2),
        "min_required_gb": round(min_req, 2),
        "path": str(VM_ROOT),
    }


def vm_latest_virtualbox() -> dict:
    info: dict = {"ok": False}
    try:
        version = _read_text("https://download.virtualbox.org/virtualbox/LATEST.TXT").strip()
        if not re.match(r"\d+\.\d+\.\d+", version):
            return {"ok": False, "error": "unable to parse VirtualBox version"}
        base = f"https://download.virtualbox.org/virtualbox/{version}"
        info.update(
            {
                "ok": True,
                "version": version,
                "base_url": base,
                "windows_installer": f"{base}/VirtualBox-{version}-Win.exe",
                "guest_additions_iso": f"{base}/VBoxGuestAdditions_{version}.iso",
                "extension_pack": f"{base}/Oracle_VirtualBox_Extension_Pack-{version}.vbox-extpack",
            }
        )
        return info
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def vm_status() -> dict:
    _ensure_dirs()
    exe = _vboxmanage_path()
    status = {
        "virtualbox": {
            "available": bool(exe),
            "path": str(exe) if exe else "",
            "vbox_home": str(VBOX_HOME),
            "vm_dir": str(VMS_DIR),
        },
        "images_dir": str(IMAGES_DIR),
        "downloads_dir": str(DOWNLOADS_DIR),
        "catalog": _load_catalog(),
        "disk": vm_check_disk(),
        "guest_additions": {"path": str(_guest_additions_path() or "")},
        "logs": {"path": str(LOG_PATH)},
    }
    if exe:
        rc, out, err = _run_vboxmanage(["list", "vms"])
        status["virtualbox"]["list_rc"] = rc
        status["virtualbox"]["list_error"] = err
        status["virtualbox"]["vms"] = [line.strip() for line in out.splitlines() if line.strip()] if out else []
        rc, out, _ = _run_vboxmanage(["-v"])
        if rc == 0:
            status["virtualbox"]["version"] = out.strip()
    return status


def vm_catalog() -> dict:
    catalog = _load_catalog()
    if not CATALOG_PATH.exists():
        _save_catalog(catalog)
    return catalog


def vm_tail_logs(lines: int = 200) -> dict:
    try:
        if not LOG_PATH.exists():
            return {"ok": True, "lines": []}
        data = LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
        return {"ok": True, "lines": data[-max(1, min(lines, 2000)) :], "path": str(LOG_PATH)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _download(url: str, dest: Path, *, timeout: int = 30, label: str = "") -> Tuple[bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, dest.open("wb") as fh:
            total = resp.headers.get("Content-Length")
            remaining = int(total) if total and total.isdigit() else None
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                if remaining is not None:
                    remaining -= len(chunk)
                    if remaining % (50 * 1024 * 1024) < len(chunk):
                        _log_line(f"download {label or dest.name}: {max(0, remaining) / (1024 * 1024):.1f}MB remaining")
        return True, f"downloaded {url} -> {dest}"
    except Exception as exc:
        return False, f"download failed: {exc}"


def _resolve_kali_iso() -> Tuple[Optional[str], str]:
    url = "https://cdimage.kali.org/kali-images/current/"
    text = _read_text(url)
    matches = re.findall(r'href="(kali-linux-[^"]+-installer-amd64\.iso)"', text)
    if not matches:
        matches = re.findall(r'href="(kali-linux-[^"]+-live-amd64\.iso)"', text)
    if not matches:
        return None, "No Kali ISO found"
    return url + matches[0], "resolved"


def _resolve_ubuntu_iso(*, prefer_latest: bool = True) -> Tuple[Optional[str], str]:
    meta = _read_text("https://changelogs.ubuntu.com/meta-release-lts")
    versions = re.findall(r"Version:\s*(\d+\.\d+(?:\.\d+)?)\s+LTS", meta)
    if not versions:
        return None, "No Ubuntu LTS found"
    sorted_versions = sorted(versions, key=_version_tuple, reverse=True)
    pick = sorted_versions[0] if prefer_latest else (sorted_versions[1] if len(sorted_versions) > 1 else sorted_versions[0])
    base = f"https://releases.ubuntu.com/{pick}/"
    desktop = f"{base}ubuntu-{pick}-desktop-amd64.iso"
    server = f"{base}ubuntu-{pick}-live-server-amd64.iso"
    if _url_exists(desktop):
        return desktop, "resolved"
    if _url_exists(server):
        return server, "resolved"
    return None, "No Ubuntu ISO found"


def _resolve_parrot_iso() -> Tuple[Optional[str], str]:
    base = "https://deb.parrot.sh/direct/parrot/iso/"
    text = _read_text(base)
    dirs = re.findall(r'href="(\d+\.\d+(?:\.\d+)?)/"', text)
    if not dirs:
        return None, "No Parrot ISO directories found"
    dirs_sorted = sorted(dirs, key=_version_tuple, reverse=True)
    latest = dirs_sorted[0]
    listing = _read_text(base + latest + "/")
    match = re.search(r'href="(Parrot-security-.*amd64\.iso)"', listing, re.IGNORECASE)
    if not match:
        match = re.search(r'href="(Parrot-home-.*amd64\.iso)"', listing, re.IGNORECASE)
    if not match:
        return None, "No Parrot ISO found"
    return base + latest + "/" + match.group(1), "resolved"


def _resolve_image_url(image: dict) -> Tuple[Optional[str], str]:
    url = str(image.get("url") or "").strip()
    if url:
        return url, "configured"
    resolver = str(image.get("resolver") or "").strip().lower()
    if resolver == "kali_current":
        return _resolve_kali_iso()
    if resolver == "ubuntu_lts":
        return _resolve_ubuntu_iso(prefer_latest=True)
    if resolver == "parrot_latest":
        return _resolve_parrot_iso()
    return None, "no resolver"


def vm_fetch_image(image_id: str, *, url: Optional[str] = None, overwrite: bool = False) -> dict:
    catalog = vm_catalog()
    image = next((img for img in catalog.get("images", []) if img.get("id") == image_id), None)
    if not image:
        return {"ok": False, "error": f"unknown image id {image_id}"}
    if url:
        fetch_url = url.strip()
        resolve_note = "override"
    else:
        fetch_url, resolve_note = _resolve_image_url(image)
    if not fetch_url:
        return {"ok": False, "error": "no url resolved for image", "resolver": image.get("resolver")}
    filename = image.get("filename")
    if not filename:
        filename = fetch_url.split("/")[-1] or f"{image_id}.iso"
    dest = IMAGES_DIR / filename
    if dest.exists() and not overwrite:
        return {"ok": True, "image": image_id, "path": str(dest), "note": "already exists", "resolver": resolve_note}
    ok, msg = _download(fetch_url, dest, timeout=60, label=image_id)
    _log_line(f"fetch image {image_id} -> {msg}")
    return {"ok": ok, "image": image_id, "path": str(dest), "message": msg, "resolver": resolve_note}


def vm_download_guest_additions(version: Optional[str] = None) -> dict:
    if not version:
        latest = vm_latest_virtualbox()
        if not latest.get("ok"):
            return {"ok": False, "error": latest.get("error", "no virtualbox info")}
        version = str(latest.get("version"))
        url = str(latest.get("guest_additions_iso"))
    else:
        url = f"https://download.virtualbox.org/virtualbox/{version}/VBoxGuestAdditions_{version}.iso"
    dest = DOWNLOADS_DIR / f"VBoxGuestAdditions_{version}.iso"
    if dest.exists():
        return {"ok": True, "path": str(dest), "note": "already downloaded"}
    ok, msg = _download(url, dest, timeout=60, label="guest-additions")
    _log_line(f"guest additions -> {msg}")
    return {"ok": ok, "path": str(dest), "message": msg}


def _state_bump_counter(name: str, key: str) -> int:
    state = _load_state()
    entry = state.get(name, {}) if name else {}
    current = int(entry.get(key) or 0) + 1
    entry[key] = current
    if name:
        state[name] = entry
        _save_state(state)
    return current


def vm_attach_guest_additions_iso(name: str, iso_path: Optional[str] = None) -> dict:
    if not name:
        return {"ok": False, "error": "vm name required"}
    if not iso_path:
        iso_path = str(_guest_additions_path() or "")
        if not iso_path:
            download = vm_download_guest_additions()
            iso_path = str(download.get("path") or "")
    if not iso_path:
        return {"ok": False, "error": "guest additions iso not available"}
    rc, out, err = _run_vboxmanage(
        [
            "storageattach",
            name,
            "--storagectl",
            "IDE",
            "--port",
            "1",
            "--device",
            "0",
            "--type",
            "dvddrive",
            "--medium",
            iso_path,
        ],
        timeout=60,
    )
    _log_line(f"attach guest additions iso {name} -> rc={rc}")
    return {"ok": rc == 0, "message": out or err, "path": iso_path}


def vm_repair_guest_additions(
    name: str,
    *,
    user: str = "c0d3r",
    password: Optional[str] = None,
    timeout_s: float = 900.0,
) -> dict:
    logs: List[str] = []
    if not name:
        return {"ok": False, "error": "vm name required"}
    if vm_wait_guest_additions(name, timeout_s=5).get("ok"):
        return {"ok": True, "message": "guest additions already ready"}
    attempt = _state_bump_counter(name, "ga_repair_attempts")
    max_attempts = int(os.getenv("C0D3R_VM_GA_REPAIR_MAX", "3") or "3")
    logs.append(f"ga repair attempt {attempt}/{max_attempts}")
    if attempt > max_attempts:
        return {"ok": False, "error": "guest additions repair maxed", "logs": logs, "action": "rebuild"}
    attach = vm_attach_guest_additions_iso(name)
    if attach.get("ok"):
        logs.append("guest additions iso attached")
    else:
        logs.append(f"guest additions iso attach failed: {attach.get('error') or attach.get('message')}")
    if vm_wait_ssh(name, timeout_s=30).get("ok"):
        cmd = (
            "sudo apt-get update -y"
            " && sudo apt-get install -y build-essential dkms linux-headers-$(uname -r) "
            "virtualbox-guest-dkms virtualbox-guest-utils virtualbox-guest-x11"
            " && sudo systemctl enable --now vboxservice || true"
            " && sudo modprobe vboxguest vboxsf vboxvideo || true"
        )
        exec_result = vm_exec_ssh(name, cmd, timeout_s=timeout_s)
        logs.append(exec_result.get("stderr") or exec_result.get("stdout") or "ssh install attempted")
    else:
        logs.append("ssh not ready; falling back to gui recovery")
        gui = vm_gui_recover(name, user=user, password=password)
        logs.append(gui.get("message") or gui.get("error") or "")
    wait = vm_wait_guest_additions(name, timeout_s=timeout_s)
    logs.append(wait.get("message") or wait.get("error") or "")
    return {"ok": wait.get("ok", False), "logs": logs, "message": "guest additions repair attempted"}


def vm_bootstrap(auto_install: bool = True) -> dict:
    _ensure_dirs()
    exe = _vboxmanage_path()
    if exe:
        _log_line("virtualbox already available")
        return {"ok": True, "message": "VirtualBox available", "path": str(exe)}
    if not auto_install:
        return {
            "ok": False,
            "error": "VirtualBox not installed",
            "hint": "Set C0D3R_VM_AUTO_INSTALL=1 or provide C0D3R_VBOXMANAGE to enable auto-install.",
        }
    _log_line("virtualbox install start")
    install_attempts: List[dict] = []
    if os.name == "nt":
        for tool, args in (
            ("winget", ["install", "--id", "Oracle.VirtualBox", "-e", "--silent"]),
            ("choco", ["install", "virtualbox", "-y"]),
            ("scoop", ["install", "virtualbox"]),
        ):
            path = shutil.which(tool)
            if not path:
                continue
            proc = subprocess.run([path] + args, capture_output=True, text=True)
            install_attempts.append(
                {"tool": tool, "rc": proc.returncode, "out": proc.stdout.strip(), "err": proc.stderr.strip()}
            )
            _log_line(f"virtualbox install {tool} rc={proc.returncode}")
            if proc.returncode == 0:
                break
    elif platform.system().lower() == "darwin":
        path = shutil.which("brew")
        if path:
            proc = subprocess.run([path, "install", "--cask", "virtualbox"], capture_output=True, text=True)
            install_attempts.append(
                {"tool": "brew", "rc": proc.returncode, "out": proc.stdout.strip(), "err": proc.stderr.strip()}
            )
            _log_line(f"virtualbox install brew rc={proc.returncode}")
    else:
        for tool, args in (
            ("apt-get", ["install", "-y", "virtualbox"]),
            ("dnf", ["install", "-y", "VirtualBox"]),
            ("yum", ["install", "-y", "VirtualBox"]),
            ("pacman", ["-S", "--noconfirm", "virtualbox"]),
        ):
            path = shutil.which(tool)
            if not path:
                continue
            proc = subprocess.run([path] + args, capture_output=True, text=True)
            install_attempts.append(
                {"tool": tool, "rc": proc.returncode, "out": proc.stdout.strip(), "err": proc.stderr.strip()}
            )
            _log_line(f"virtualbox install {tool} rc={proc.returncode}")
            if proc.returncode == 0:
                break
    exe = _vboxmanage_path()
    _log_line(f"virtualbox install complete ok={bool(exe)}")
    return {
        "ok": bool(exe),
        "message": "VirtualBox install attempted",
        "path": str(exe) if exe else "",
        "attempts": install_attempts,
    }


def vm_update_virtualbox(auto_update: bool = True) -> dict:
    if not auto_update:
        return {"ok": True, "note": "auto update disabled"}
    _log_line("virtualbox update start")
    latest = vm_latest_virtualbox()
    if not latest.get("ok"):
        _log_line(f"virtualbox update failed: {latest.get('error')}")
        return {"ok": False, "error": latest.get("error", "unknown")}
    installed = ""
    rc, out, _ = _run_vboxmanage(["-v"])
    if rc == 0:
        installed = out.strip().split("r")[0]
    target = str(latest.get("version"))
    if installed and _version_tuple(installed) >= _version_tuple(target):
        _log_line(f"virtualbox already latest {installed}")
        return {"ok": True, "note": "already latest", "version": installed}
    if os.name == "nt":
        url = latest.get("windows_installer")
        if not url:
            return {"ok": False, "error": "no windows installer url"}
        dest = DOWNLOADS_DIR / f"VirtualBox-{target}-Win.exe"
        if not dest.exists():
            ok, msg = _download(str(url), dest, timeout=120, label="virtualbox")
            if not ok:
                return {"ok": False, "error": msg}
        proc = subprocess.run([str(dest), "--silent"], capture_output=True, text=True)
        _log_line(f"virtualbox update win rc={proc.returncode}")
        return {"ok": proc.returncode == 0, "rc": proc.returncode, "out": proc.stdout.strip(), "err": proc.stderr.strip()}
    if platform.system().lower() == "darwin":
        tool = shutil.which("brew")
        if tool:
            proc = subprocess.run([tool, "upgrade", "--cask", "virtualbox"], capture_output=True, text=True)
            _log_line(f"virtualbox update brew rc={proc.returncode}")
            return {"ok": proc.returncode == 0, "rc": proc.returncode, "out": proc.stdout.strip(), "err": proc.stderr.strip()}
        return {"ok": False, "error": "brew not available"}
    for tool, args in (
        ("apt-get", ["install", "-y", "virtualbox"]),
        ("dnf", ["upgrade", "-y", "VirtualBox"]),
        ("yum", ["upgrade", "-y", "VirtualBox"]),
        ("pacman", ["-Syu", "--noconfirm", "virtualbox"]),
    ):
        path = shutil.which(tool)
        if not path:
            continue
        proc = subprocess.run([path] + args, capture_output=True, text=True)
        _log_line(f"virtualbox update {tool} rc={proc.returncode}")
        return {"ok": proc.returncode == 0, "rc": proc.returncode, "out": proc.stdout.strip(), "err": proc.stderr.strip()}
    return {"ok": False, "error": "no supported package manager found"}


def vm_create(
    name: str,
    *,
    image_path: Optional[str] = None,
    os_type: str = "Ubuntu_64",
    memory_mb: int = 4096,
    cpus: int = 2,
    vram_mb: int = 64,
    disk_gb: int = 40,
    efi: bool = False,
) -> dict:
    if not name:
        return {"ok": False, "error": "vm name required"}
    _ensure_dirs()
    rc, out, err = _run_vboxmanage(["list", "vms"])
    if rc == 0 and name in out:
        return {"ok": True, "message": "vm already exists", "name": name}
    rc, out, err = _run_vboxmanage(["createvm", "--name", name, "--register", "--ostype", os_type])
    if rc != 0:
        return {"ok": False, "error": err or out}
    _run_vboxmanage(["modifyvm", name, "--memory", str(memory_mb), "--cpus", str(cpus), "--vram", str(vram_mb)])
    if efi:
        _run_vboxmanage(["modifyvm", name, "--firmware", "efi"])
    _run_vboxmanage(["modifyvm", name, "--nic1", "nat"])
    _run_vboxmanage(["setproperty", "machinefolder", str(VMS_DIR)])
    if image_path:
        img = Path(image_path).expanduser()
        if img.suffix.lower() in {".ova", ".ovf"}:
            rc, out, err = _run_vboxmanage(["import", str(img), "--vsys", "0", "--vmname", name], timeout=600)
            return {"ok": rc == 0, "message": out or err, "name": name}
        if img.suffix.lower() in {".vdi", ".vhd", ".vmdk", ".qcow2"}:
            _run_vboxmanage(["storagectl", name, "--name", "SATA", "--add", "sata", "--controller", "IntelAhci"])
            _run_vboxmanage(
                ["storageattach", name, "--storagectl", "SATA", "--port", "0", "--device", "0", "--type", "hdd", "--medium", str(img)]
            )
            return {"ok": True, "message": "disk attached", "name": name}
        if img.suffix.lower() == ".iso":
            disk_path = VMS_DIR / f"{name}.vdi"
            _run_vboxmanage(["createhd", "--filename", str(disk_path), "--size", str(disk_gb * 1024)])
            _run_vboxmanage(["storagectl", name, "--name", "SATA", "--add", "sata", "--controller", "IntelAhci"])
            _run_vboxmanage(
                ["storageattach", name, "--storagectl", "SATA", "--port", "0", "--device", "0", "--type", "hdd", "--medium", str(disk_path)]
            )
            _run_vboxmanage(["storagectl", name, "--name", "IDE", "--add", "ide"])
            _run_vboxmanage(
                ["storageattach", name, "--storagectl", "IDE", "--port", "0", "--device", "0", "--type", "dvddrive", "--medium", str(img)]
            )
            return {"ok": True, "message": "iso attached", "name": name}
    _log_line(f"vm created {name}")
    return {"ok": True, "message": "vm created", "name": name}


def vm_delete(name: str, *, delete_files: bool = True) -> dict:
    if not name:
        return {"ok": False, "error": "vm name required"}
    args = ["unregistervm", name]
    if delete_files:
        args.append("--delete")
    rc, out, err = _run_vboxmanage(args, timeout=300)
    _log_line(f"vm delete {name} -> rc={rc}")
    state = _load_state()
    if name in state:
        state.pop(name, None)
        _save_state(state)
    return {"ok": rc == 0, "message": out or err}


def vm_enable_ssh(name: str, port: int = 2222) -> dict:
    if not name:
        return {"ok": False, "error": "vm name required"}
    rule = f"ssh,tcp,,{port},,22"
    rc, out, err = _run_vboxmanage(["modifyvm", name, "--natpf1", rule])
    if rc != 0 and err and "already exists" in err.lower():
        return {"ok": True, "message": "ssh rule already exists", "rule": rule}
    return {"ok": rc == 0, "message": out or err, "rule": rule}


def _ssh_key_paths(name: str) -> Tuple[Path, Path]:
    key_path = SSH_DIR / f"{name}_id_rsa"
    pub_path = SSH_DIR / f"{name}_id_rsa.pub"
    return key_path, pub_path


def vm_ensure_ssh_key(name: str) -> dict:
    _ensure_dirs()
    key_path, pub_path = _ssh_key_paths(name)
    if key_path.exists() and pub_path.exists():
        return {"ok": True, "private_key": str(key_path), "public_key": str(pub_path)}
    ssh_keygen = shutil.which("ssh-keygen")
    if not ssh_keygen:
        return {"ok": False, "error": "ssh-keygen not available"}
    try:
        proc = subprocess.run(
            [ssh_keygen, "-t", "rsa", "-b", "4096", "-N", "", "-f", str(key_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return {"ok": False, "error": proc.stderr.strip() or proc.stdout.strip()}
        return {"ok": True, "private_key": str(key_path), "public_key": str(pub_path)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def vm_unattended_install(
    name: str,
    *,
    iso_path: str,
    user: str,
    password: str,
    full_name: str = "c0d3r",
    hostname: Optional[str] = None,
    locale: str = "en_US",
    timezone: str = "UTC",
    install_additions: bool = True,
    additions_iso: Optional[str] = None,
    post_install: Optional[str] = None,
) -> dict:
    if not name or not iso_path:
        return {"ok": False, "error": "name and iso_path required"}
    if not hostname:
        hostname = name
    if "." not in hostname:
        hostname = hostname + ".local"
    args = [
        "unattended",
        "install",
        name,
        "--iso",
        str(iso_path),
        "--user",
        user,
        "--password",
        password,
        "--full-user-name",
        full_name,
        "--hostname",
        hostname,
        "--locale",
        locale,
        "--time-zone",
        timezone,
    ]
    if install_additions:
        args.append("--install-additions")
    if additions_iso:
        args += ["--additions-iso", str(additions_iso)]
    if post_install:
        args += ["--post-install-command", post_install]
    rc, out, err = _run_vboxmanage(args, timeout=600)
    if rc != 0 and additions_iso and re.search(r"unknown option|invalid option|unrecognized", err, re.I):
        args = [arg for arg in args if arg != "--additions-iso" and arg != str(additions_iso)]
        rc, out, err = _run_vboxmanage(args, timeout=600)
    _log_line(f"unattended install {name} -> rc={rc}")
    return {"ok": rc == 0, "message": out or err}


def vm_wait_port(host: str, port: int, timeout_s: float = 120.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=3):
                return {"ok": True, "message": f"port {port} open"}
        except Exception:
            time.sleep(3)
    return {"ok": False, "error": f"timeout waiting for port {port}"}


def vm_wait_guest_additions(name: str, timeout_s: float = 300.0) -> dict:
    if not name:
        return {"ok": False, "error": "vm name required"}
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rc, out, err = _run_vboxmanage(["guestproperty", "get", name, "/VirtualBox/GuestAdd/RunLevel"])
        if rc == 0 and out and "No value set!" not in out:
            return {"ok": True, "message": out.strip()}
        rc, out, err = _run_vboxmanage(["guestproperty", "get", name, "/VirtualBox/GuestAdd/Version"])
        if rc == 0 and out and "No value set!" not in out:
            # Some guests never publish RunLevel but do publish Version.
            return {"ok": True, "message": out.strip()}
        time.sleep(5)
    return {"ok": False, "error": "timeout waiting for guest additions"}


def vm_wait_guest_user(name: str, user: str, timeout_s: float = 600.0) -> dict:
    if not name or not user:
        return {"ok": False, "error": "vm name and user required"}
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rc, out, err = _run_vboxmanage(["guestproperty", "get", name, "/VirtualBox/GuestInfo/OS/LoggedInUsersList"])
        if rc == 0 and out and "No value set!" not in out:
            if user in out:
                return {"ok": True, "message": out.strip()}
        time.sleep(5)
    return {"ok": False, "error": "timeout waiting for guest user login"}


def vm_wait_ssh(name: str, timeout_s: float = 300.0) -> dict:
    if not name:
        return {"ok": False, "error": "vm name required"}
    state = _load_state().get(name, {})
    port = int(state.get("ssh_port") or 2222)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        exec_result = vm_exec_ssh(name, "true", timeout_s=20)
        if exec_result.get("ok"):
            return {"ok": True, "message": "ssh ready"}
        err = (exec_result.get("stderr") or exec_result.get("error") or "").lower()
        if any(sig in err for sig in ("connection reset", "connection refused", "kex_exchange", "no route", "timed out", "connection aborted")):
            vm_wait_port("127.0.0.1", port, timeout_s=30)
            time.sleep(5)
            continue
        time.sleep(5)
    return {"ok": False, "error": "timeout waiting for ssh ready"}


def vm_wait_ready(
    name: str,
    *,
    timeout_s: float = 1800.0,
    poll_s: float = 5.0,
    require_user: Optional[str] = None,
    require_guest_additions: bool = True,
    vbox_timeout_s: float = 15.0,
) -> dict:
    if not name:
        return {"ok": False, "error": "vm name required"}
    state = _load_state().get(name, {})
    user = require_user or (state.get("user") or "c0d3r")
    port = int(state.get("ssh_port") or 2222)
    deadline = time.time() + timeout_s
    last_status = ""
    vbox_timeout = max(5, int(vbox_timeout_s))
    while time.time() < deadline:
        rc1, out1, _ = _run_vboxmanage(
            ["guestproperty", "get", name, "/VirtualBox/GuestAdd/RunLevel"],
            timeout=vbox_timeout,
        )
        ga_present = rc1 == 0 and out1 and "No value set!" not in out1
        if not ga_present:
            rcv, outv, _ = _run_vboxmanage(
                ["guestproperty", "get", name, "/VirtualBox/GuestAdd/Version"],
                timeout=vbox_timeout,
            )
            ga_present = rcv == 0 and outv and "No value set!" not in outv
        rc2, out2, _ = _run_vboxmanage(
            ["guestproperty", "get", name, "/VirtualBox/GuestInfo/OS/LoggedInUsersList"],
            timeout=vbox_timeout,
        )
        user_ok = rc2 == 0 and out2 and user in out2
        port_ok = vm_wait_port("127.0.0.1", port, timeout_s=1).get("ok")
        ga_ok = ga_present or not require_guest_additions
        status = f"ga={ga_present} user={user_ok} port={port_ok}"
        if ga_ok and user_ok and port_ok:
            ssh_ok = vm_exec_ssh(name, "true", timeout_s=10).get("ok")
            if ssh_ok:
                return {"ok": True, "message": "ready"}
            status = f"ga={ga_present} user={user_ok} port={port_ok} ssh=False"
        last_status = status
        time.sleep(poll_s)
    return {"ok": False, "error": "timeout waiting for vm ready", "last": last_status}


def vm_exists(name: str) -> bool:
    if not name:
        return False
    rc, out, _ = _run_vboxmanage(["list", "vms"])
    if rc != 0 or not out:
        return False
    return f"\"{name}\"" in out


def vm_info(name: str) -> dict:
    if not name:
        return {}
    rc, out, _ = _run_vboxmanage(["showvminfo", name, "--machinereadable"])
    if rc != 0 or not out:
        return {}
    info: dict = {}
    for line in out.splitlines():
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        info[key.strip()] = val.strip().strip('"')
    return info


def vm_state(name: str) -> str:
    info = vm_info(name)
    return (info.get("VMState") or "").strip().lower()


def _read_vbox_log_tail(name: str, lines: int = 80) -> str:
    try:
        log_path = VMS_DIR / name / "Logs" / "VBox.log"
        if not log_path.exists():
            return ""
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        parts = text.splitlines()
        tail = "\n".join(parts[-lines:]) if parts else ""
        return tail
    except Exception:
        return ""


def vm_health_snapshot(name: str, *, user: str = "c0d3r") -> dict:
    snapshot: dict = {"vm": name}
    snapshot["state"] = vm_state(name)
    rc1, out1, err1 = _run_vboxmanage(["guestproperty", "get", name, "/VirtualBox/GuestAdd/RunLevel"], timeout=10)
    snapshot["guest_additions"] = out1.strip() if out1 else err1
    rc2, out2, err2 = _run_vboxmanage(["guestproperty", "get", name, "/VirtualBox/GuestInfo/OS/LoggedInUsersList"], timeout=10)
    snapshot["logged_in_users"] = out2.strip() if out2 else err2
    snapshot["user_present"] = bool(out2 and user in out2)
    snapshot["vbox_log_tail"] = _read_vbox_log_tail(name, lines=40)
    return snapshot


def _health_evidence(snapshot: dict) -> list[str]:
    evidence: list[str] = []
    state = (snapshot.get("state") or "").lower()
    if state in {"aborted", "stopping", "stopped"}:
        evidence.append(f"state={state}")
    guest_add = snapshot.get("guest_additions") or ""
    if "No value set!" in guest_add:
        evidence.append("guest_additions_unset")
    if "not found" in guest_add.lower():
        evidence.append("guest_additions_unavailable")
    if not snapshot.get("user_present"):
        evidence.append("user_not_logged_in")
    log_tail = (snapshot.get("vbox_log_tail") or "").lower()
    for sig in ("guru meditation", "vt-x", "vbox_e_invalid_object_state", "aborted", "e_fail", "err"):
        if sig in log_tail:
            evidence.append(f"log:{sig}")
    return evidence


def vm_resume_or_recover(
    name: str,
    *,
    user: str = "c0d3r",
    timeout_s: float = 1800.0,
    poll_s: float = 5.0,
    recovery_retries: int = 2,
) -> dict:
    logs: List[str] = []
    if not vm_exists(name):
        return {"ok": False, "action": "missing", "error": "vm not found", "logs": logs}
    snapshot = vm_health_snapshot(name, user=user)
    evidence = _health_evidence(snapshot)
    state = snapshot.get("state") or ""
    logs.append(f"vm state: {state or 'unknown'}")
    if evidence:
        ev_line = ", ".join(sorted(set(evidence)))
        logs.append("evidence: " + ev_line)
        _log_line(f"vm resume evidence {name}: {ev_line}")
    if state == "paused":
        _run_vboxmanage(["controlvm", name, "resume"])
        time.sleep(2)
    elif state in {"poweroff", "saved", "aborted", "stopping", "stopped"}:
        start = vm_start(name, headless=True, timeout_s=30)
        logs.append(f"vm start: {start.get('message')}")
    if "state=aborted" in evidence or "log:guru meditation" in evidence or "log:vt-x" in evidence:
        return {
            "ok": False,
            "action": "rebuild",
            "error": "critical vm error evidence",
            "logs": logs,
            "evidence": evidence,
        }
    if "guest_additions_unset" in evidence and "user_not_logged_in" in evidence:
        logs.append("early gui recovery (login + guest additions)")
        _log_line(f"vm resume: gui recovery triggered for {name}")
        gui = vm_gui_recover(name, user=user)
        logs.append(gui.get("message") or gui.get("error") or "")
    if "guest_additions_unset" in evidence and "user_not_logged_in" not in evidence:
        logs.append("guest additions repair")
        _log_line(f"vm resume: guest additions repair triggered for {name}")
        ga_repair = vm_repair_guest_additions(name, user=user)
        logs.append(ga_repair.get("message") or ga_repair.get("error") or "")
    ready = vm_wait_ready(name, timeout_s=timeout_s, poll_s=poll_s, require_user=user)
    logs.append(ready.get("message") or ready.get("error") or "")
    if ready.get("ok"):
        return {"ok": True, "logs": logs, "message": "ready"}
    if "guest_additions_unset" in evidence or "user_not_logged_in" in evidence:
        logs.append("attempting gui recovery")
        gui = vm_gui_recover(name, user=user)
        logs.append(gui.get("message") or gui.get("error") or "")
        ready = vm_wait_ready(name, timeout_s=min(600.0, timeout_s), poll_s=poll_s, require_user=user)
        logs.append(ready.get("message") or ready.get("error") or "")
        if ready.get("ok"):
            return {"ok": True, "logs": logs, "message": "ready after gui recovery"}
    for attempt in range(max(0, recovery_retries)):
        logs.append(f"recovery attempt {attempt + 1}/{recovery_retries}")
        vm_stop(name, force=True)
        vm_start(name, headless=True, timeout_s=30)
        ready = vm_wait_ready(name, timeout_s=min(900.0, timeout_s), poll_s=poll_s, require_user=user)
        logs.append(ready.get("message") or ready.get("error") or "")
        if ready.get("ok"):
            return {"ok": True, "logs": logs, "message": "ready after recovery"}
    evidence.append("resume_timeout")
    return {
        "ok": False,
        "action": "rebuild",
        "error": "resume failed",
        "logs": logs,
        "evidence": evidence,
        "last": ready.get("last"),
    }


def vm_start(name: str, *, headless: bool = True, timeout_s: int = 120) -> dict:
    if not name:
        return {"ok": False, "error": "vm name required"}
    args = ["startvm", name]
    if headless:
        args += ["--type", "headless"]
    rc, out, err = _run_vboxmanage(args, timeout=timeout_s)
    _log_line(f"vm start {name} -> rc={rc}")
    msg = out or err
    if rc != 0 and msg:
        lowered = msg.lower()
        if "already locked" in lowered or "already running" in lowered or "busy" in lowered:
            return {"ok": True, "message": msg}
    return {"ok": rc == 0, "message": msg}


def vm_stop(name: str, *, force: bool = False) -> dict:
    if not name:
        return {"ok": False, "error": "vm name required"}
    cmd = ["controlvm", name, "poweroff" if force else "acpipowerbutton"]
    rc, out, err = _run_vboxmanage(cmd)
    _log_line(f"vm stop {name} -> rc={rc}")
    return {"ok": rc == 0, "message": out or err}


def vm_reset(name: str) -> dict:
    if not name:
        return {"ok": False, "error": "vm name required"}
    rc, out, err = _run_vboxmanage(["controlvm", name, "reset"])
    _log_line(f"vm reset {name} -> rc={rc}")
    return {"ok": rc == 0, "message": out or err}


def vm_screenshot(name: str, path: Optional[str] = None) -> dict:
    if not name:
        return {"ok": False, "error": "vm name required"}
    _ensure_dirs()
    if not path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = str(LOGS_DIR / f"{name}_screenshot_{ts}.png")
    rc, out, err = _run_vboxmanage(["controlvm", name, "screenshotpng", path])
    _log_line(f"vm screenshot {name} -> {path}")
    return {"ok": rc == 0, "path": path, "message": out or err}


def vm_type(name: str, text: str) -> dict:
    if not name or text is None:
        return {"ok": False, "error": "vm name and text required"}
    rc, out, err = _run_vboxmanage(["controlvm", name, "keyboardputstring", text])
    _log_line(f"vm type {name} -> {len(text)} chars")
    return {"ok": rc == 0, "message": out or err}


def vm_mouse(
    name: str,
    *,
    x: Optional[int] = None,
    y: Optional[int] = None,
    buttons: int = 0,
    screen_w: Optional[int] = None,
    screen_h: Optional[int] = None,
) -> dict:
    if not name:
        return {"ok": False, "error": "vm name required"}
    if x is None or y is None:
        return {"ok": False, "error": "x and y required"}
    if screen_w and screen_h:
        abs_x = int(max(0, min(65535, x / max(1, screen_w) * 65535)))
        abs_y = int(max(0, min(65535, y / max(1, screen_h) * 65535)))
    else:
        abs_x = int(max(0, min(65535, x)))
        abs_y = int(max(0, min(65535, y)))
    rc, out, err = _run_vboxmanage(["controlvm", name, "mouseputstate", str(abs_x), str(abs_y), str(buttons), "0"])
    _log_line(f"vm mouse {name} -> {abs_x},{abs_y} buttons={buttons}")
    msg = out or err
    if rc == 0:
        return {"ok": True, "message": msg}
    if msg and "mouseputstate" in msg.lower():
        guest = vm_guest_mouse(name, x=x, y=y, buttons=buttons)
        if guest.get("ok"):
            return {"ok": True, "message": "mouse via guestcontrol"}
        return {"ok": False, "error": "mouse input not supported by VBoxManage on this host", "message": msg}
    return {"ok": False, "message": msg}


def vm_guest_prepare_input(name: str) -> dict:
    guest_os = _guest_os_for_vm(name)
    if guest_os != "linux":
        return {"ok": False, "error": f"guest input prepare not supported for {guest_os}"}
    state = _load_state().get(name, {})
    password = state.get("password") or ""
    if not password:
        return {"ok": False, "error": "no guest password available"}
    sudo_prefix = f"echo '{password}' | sudo -S "
    user = state.get("user") or "c0d3r"
    checks = [
        f"{sudo_prefix}apt-get update -y",
        f"{sudo_prefix}apt-get install -y xdotool x11-utils",
        f"{sudo_prefix}sed -i 's/^#\\?WaylandEnable=.*/WaylandEnable=false/' /etc/gdm3/custom.conf",
        f"{sudo_prefix}sed -i 's/^#\\?AutomaticLoginEnable=.*/AutomaticLoginEnable=true/' /etc/gdm3/custom.conf",
        f"{sudo_prefix}sed -i 's/^#\\?AutomaticLogin=.*/AutomaticLogin={user}/' /etc/gdm3/custom.conf",
    ]
    for cmd in checks:
        vm_guest_exec(name, cmd, timeout_s=600)
    return {"ok": True, "message": "guest input prepared"}


def vm_guest_mouse(name: str, *, x: int, y: int, buttons: int = 0) -> dict:
    guest_os = _guest_os_for_vm(name)
    if guest_os == "windows":
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "Add-Type -AssemblyName System.Drawing; "
            "Add-Type -TypeDefinition @'\n"
            "using System;\n"
            "using System.Runtime.InteropServices;\n"
            "public static class Win32Mouse {\n"
            "  [DllImport(\"user32.dll\", CallingConvention=CallingConvention.StdCall)]\n"
            "  public static extern void mouse_event(int dwFlags, int dx, int dy, int cButtons, int dwExtraInfo);\n"
            "}\n"
            "'@; "
            f"[System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point({x},{y}); "
        )
        if buttons & 1:
            script += "[Win32Mouse]::mouse_event(2,0,0,0,0); [Win32Mouse]::mouse_event(4,0,0,0,0);"
        return vm_guest_exec(name, script, timeout_s=30)
    if guest_os != "linux":
        return {"ok": False, "error": f"guest mouse not supported for {guest_os}"}
    prep = vm_guest_prepare_input(name)
    if not prep.get("ok"):
        return prep
    cmd = f"DISPLAY=:0 xdotool mousemove {int(x)} {int(y)}"
    if buttons & 1:
        cmd += " && DISPLAY=:0 xdotool click 1"
    return vm_guest_exec(name, cmd, timeout_s=30)


_SCAN_CODES = {
    "ctrl_down": "1d",
    "ctrl_up": "9d",
    "alt_down": "38",
    "alt_up": "b8",
    "shift_down": "2a",
    "shift_up": "aa",
    "enter_down": "1c",
    "enter_up": "9c",
    "esc_down": "01",
    "esc_up": "81",
    "tab_down": "0f",
    "tab_up": "8f",
    "up_down": "e0 48",
    "up_up": "e0 c8",
    "down_down": "e0 50",
    "down_up": "e0 d0",
    "left_down": "e0 4b",
    "left_up": "e0 cb",
    "right_down": "e0 4d",
    "right_up": "e0 cd",
    "t_down": "14",
    "t_up": "94",
    "a_down": "1e",
    "a_up": "9e",
    "f1_down": "3b",
    "f1_up": "bb",
    "f3_down": "3d",
    "f3_up": "bd",
}


def vm_key_combo(name: str, combo: str) -> dict:
    combo = (combo or "").lower().strip()
    if combo == "ctrl+alt+t":
        sc = f"{_SCAN_CODES['ctrl_down']} {_SCAN_CODES['alt_down']} {_SCAN_CODES['t_down']} {_SCAN_CODES['t_up']} {_SCAN_CODES['alt_up']} {_SCAN_CODES['ctrl_up']}"
    elif combo == "ctrl+alt+f3":
        sc = f"{_SCAN_CODES['ctrl_down']} {_SCAN_CODES['alt_down']} {_SCAN_CODES['f3_down']} {_SCAN_CODES['f3_up']} {_SCAN_CODES['alt_up']} {_SCAN_CODES['ctrl_up']}"
    elif combo == "ctrl+alt+f1":
        sc = f"{_SCAN_CODES['ctrl_down']} {_SCAN_CODES['alt_down']} {_SCAN_CODES['f1_down']} {_SCAN_CODES['f1_up']} {_SCAN_CODES['alt_up']} {_SCAN_CODES['ctrl_up']}"
    elif combo == "ctrl+a":
        sc = f"{_SCAN_CODES['ctrl_down']} {_SCAN_CODES['a_down']} {_SCAN_CODES['a_up']} {_SCAN_CODES['ctrl_up']}"
    elif combo == "enter":
        sc = f"{_SCAN_CODES['enter_down']} {_SCAN_CODES['enter_up']}"
    elif combo == "tab":
        sc = f"{_SCAN_CODES['tab_down']} {_SCAN_CODES['tab_up']}"
    elif combo == "esc":
        sc = f"{_SCAN_CODES['esc_down']} {_SCAN_CODES['esc_up']}"
    elif combo in {"up", "down", "left", "right"}:
        sc = f"{_SCAN_CODES[f'{combo}_down']} {_SCAN_CODES[f'{combo}_up']}"
    else:
        return {"ok": False, "error": f"unsupported combo {combo}"}
    rc, out, err = _run_vboxmanage(["controlvm", name, "keyboardputscancode"] + sc.split())
    _log_line(f"vm key combo {name} -> {combo} rc={rc}")
    return {"ok": rc == 0, "message": out or err}


def vm_keys(name: str, sequence: List[str]) -> dict:
    for item in sequence:
        result = vm_key_combo(name, item)
        if not result.get("ok"):
            return result
        time.sleep(0.2)
    return {"ok": True, "message": "keys sent"}


def _vm_type_line(name: str, text: str, *, delay_s: float = 0.4) -> None:
    vm_type(name, text)
    vm_key_combo(name, "enter")
    time.sleep(delay_s)


def vm_gui_recover(name: str, *, user: Optional[str] = None, password: Optional[str] = None) -> dict:
    state = _load_state().get(name, {})
    user = user or state.get("user") or "c0d3r"
    password = password or state.get("password") or ""
    if not password:
        return {"ok": False, "error": "missing password for gui recovery"}
    safe_pw = password.replace("'", "")
    sudo = f"echo '{safe_pw}' | sudo -S "
    attach = vm_attach_guest_additions_iso(name)
    if attach.get("ok"):
        _log_line(f"vm gui recover attached additions iso {name}")
    _log_line(f"vm gui recover start {name}")
    # Switch to TTY for deterministic login.
    vm_key_combo(name, "ctrl+alt+f3")
    time.sleep(2.0)
    vm_key_combo(name, "enter")
    time.sleep(1.0)
    _vm_type_line(name, user, delay_s=1.0)
    _vm_type_line(name, password, delay_s=2.0)
    # Install guest additions + ssh, enable autologin.
    _vm_type_line(name, f"{sudo}apt-get update -y", delay_s=3.0)
    _vm_type_line(
        name,
        f"{sudo}apt-get install -y openssh-server xdotool build-essential dkms linux-headers-$(uname -r) "
        "virtualbox-guest-x11 virtualbox-guest-utils virtualbox-guest-dkms",
        delay_s=3.0,
    )
    _vm_type_line(name, f"{sudo}systemctl enable --now ssh", delay_s=1.0)
    _vm_type_line(
        name,
        "{sudo}bash -lc \"for s in $(systemctl list-unit-files | awk '/vbox/ {{print $1}}'); do systemctl enable --now $s || true; done\"".format(
            sudo=sudo
        ),
        delay_s=2.0,
    )
    _vm_type_line(
        name,
        f"{sudo}bash -lc \"mkdir -p /mnt/vboxadd; mount /dev/cdrom /mnt/vboxadd 2>/dev/null || mount /dev/sr0 /mnt/vboxadd 2>/dev/null || true; "
        "if [ -x /mnt/vboxadd/VBoxLinuxAdditions.run ]; then /mnt/vboxadd/VBoxLinuxAdditions.run --nox11 || true; fi\"",
        delay_s=2.0,
    )
    _vm_type_line(name, f"{sudo}systemctl enable --now vboxservice || true", delay_s=1.0)
    _vm_type_line(name, f"{sudo}modprobe vboxguest vboxsf vboxvideo || true", delay_s=1.0)
    _vm_type_line(name, f"{sudo}mkdir -p /etc/gdm3", delay_s=1.0)
    gdm_conf = (
        "[daemon]\\n"
        "WaylandEnable=false\\n"
        "AutomaticLoginEnable=true\\n"
        f"AutomaticLogin={user}\\n"
    )
    _vm_type_line(
        name,
        f"{sudo}bash -lc \"printf '%s' '{gdm_conf}' > /etc/gdm3/custom.conf\"",
        delay_s=1.0,
    )
    _vm_type_line(
        name,
        f"{sudo}bash -lc \"if systemctl list-unit-files | grep -q lightdm; then printf '[Seat:*]\\nautologin-user={user}\\n' > /etc/lightdm/lightdm.conf; systemctl restart lightdm; fi\"",
        delay_s=1.0,
    )
    _vm_type_line(name, f"{sudo}systemctl restart gdm3", delay_s=2.0)
    _vm_type_line(name, f"{sudo}reboot", delay_s=1.0)
    # Return to GUI session.
    vm_key_combo(name, "ctrl+alt+f1")
    time.sleep(2.0)
    # Attempt GUI login if auto-login did not trigger.
    vm_key_combo(name, "enter")
    time.sleep(1.5)
    vm_type(name, password)
    vm_key_combo(name, "enter")
    time.sleep(3.0)
    # Open terminal in GUI and re-assert installs if needed.
    vm_key_combo(name, "ctrl+alt+t")
    time.sleep(2.0)
    _vm_type_line(name, f"{sudo}apt-get update -y", delay_s=2.0)
    _vm_type_line(
        name,
        f"{sudo}apt-get install -y openssh-server xdotool build-essential dkms linux-headers-$(uname -r) "
        "virtualbox-guest-x11 virtualbox-guest-utils virtualbox-guest-dkms",
        delay_s=2.0,
    )
    _vm_type_line(name, f"{sudo}systemctl enable ssh", delay_s=1.0)
    _vm_type_line(name, f"{sudo}systemctl start ssh", delay_s=1.0)
    _vm_type_line(name, f"{sudo}systemctl enable --now vboxservice || true", delay_s=1.0)
    _vm_type_line(name, f"{sudo}modprobe vboxguest vboxsf vboxvideo || true", delay_s=1.0)
    _vm_type_line(
        name,
        f"{sudo}bash -lc \"mkdir -p /mnt/vboxadd; mount /dev/cdrom /mnt/vboxadd 2>/dev/null || mount /dev/sr0 /mnt/vboxadd 2>/dev/null || true; "
        "if [ -x /mnt/vboxadd/VBoxLinuxAdditions.run ]; then /mnt/vboxadd/VBoxLinuxAdditions.run --nox11 || true; fi\"",
        delay_s=2.0,
    )
    _log_line(f"vm gui recover end {name}")
    return {"ok": True, "message": "gui recovery attempted"}


def vm_obstacle_course(steps: List[dict]) -> dict:
    results: List[dict] = []
    for idx, step in enumerate(steps, start=1):
        action = str(step.get("action") or "").strip().lower()
        result: dict = {"step": idx, "action": action}
        if action == "sleep":
            duration = float(step.get("seconds") or step.get("duration") or 1.0)
            time.sleep(duration)
            result["ok"] = True
        elif action == "start":
            result.update(vm_start(str(step.get("name") or step.get("vm") or "")))
        elif action == "stop":
            result.update(vm_stop(str(step.get("name") or step.get("vm") or ""), force=bool(step.get("force"))))
        elif action == "wait_ready":
            vm_name = str(step.get("name") or step.get("vm") or "")
            user = step.get("user") or None
            timeout_s = float(step.get("timeout_s") or 1800.0)
            result.update(vm_wait_ready(vm_name, timeout_s=timeout_s, require_user=user))
        elif action == "ssh":
            vm_name = str(step.get("name") or step.get("vm") or "")
            command = str(step.get("command") or "")
            timeout_s = float(step.get("timeout_s") or 120.0)
            result.update(vm_exec_ssh(vm_name, command, timeout_s=timeout_s))
        elif action == "screenshot":
            result.update(vm_screenshot(str(step.get("name") or step.get("vm") or ""), path=step.get("path")))
        elif action == "type":
            result.update(vm_type(str(step.get("name") or step.get("vm") or ""), str(step.get("text") or "")))
        elif action == "mouse":
            result.update(
                vm_mouse(
                    str(step.get("name") or step.get("vm") or ""),
                    x=step.get("x"),
                    y=step.get("y"),
                    buttons=int(step.get("buttons") or 0),
                    screen_w=step.get("screen_w"),
                    screen_h=step.get("screen_h"),
                )
            )
        elif action == "exec":
            name = str(step.get("name") or step.get("vm") or "")
            cmd = str(step.get("command") or "")
            retries = int(step.get("retries") or 6)
            timeout_s = float(step.get("timeout_s") or 120)
            retry_sleep_s = float(step.get("retry_sleep_s") or 10)
            attempt = 0
            last = {}
            while attempt <= retries:
                exec_result = vm_exec_ssh(name, cmd, timeout_s=timeout_s)
                last = exec_result
                if exec_result.get("ok"):
                    break
                err = (exec_result.get("stderr") or exec_result.get("error") or "").lower()
                if any(sig in err for sig in ("connection reset", "connection refused", "kex_exchange", "no route", "timed out", "connection aborted")):
                    wait = vm_wait_ssh(name, timeout_s=180)
                    exec_result["retry_wait"] = wait
                    time.sleep(retry_sleep_s)
                    attempt += 1
                    continue
                break
            last["attempts"] = attempt + 1
            result.update(last)
        else:
            result.update({"ok": False, "error": "unknown action"})
        results.append(result)
        if not result.get("ok", False):
            break
    return {"ok": all(r.get("ok", False) for r in results), "steps": results}


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def vm_autopilot(
    *,
    image_id: str,
    vm_name: str,
    auto_install: bool = True,
    auto_update: bool = True,
    min_free_gb: float = 20.0,
    ssh_port: int = 2222,
    user: str = "c0d3r",
    password: Optional[str] = None,
    force_recreate: bool | None = None,
    image_url_override: Optional[str] = None,
    allow_fallback: bool = True,
) -> dict:
    _log_line(f"autopilot start image={image_id} vm={vm_name}")
    logs: List[str] = []
    disk = vm_check_disk(min_free_gb)
    logs.append(f"disk free {disk.get('free_gb')}GB (min {disk.get('min_required_gb')}GB)")
    if not disk.get("ok"):
        if not force_recreate and vm_exists(vm_name):
            logs.append("disk below threshold; attempting resume with existing VM")
            resume = vm_resume_or_recover(vm_name, user=user, timeout_s=1200, recovery_retries=1)
            logs.extend(resume.get("logs") or [])
            if resume.get("ok"):
                state = _load_state().get(vm_name, {})
                return {
                    "ok": True,
                    "logs": logs,
                    "vm": vm_name,
                    "ssh_port": int(state.get("ssh_port") or ssh_port),
                    "user": state.get("user") or user,
                    "password": state.get("password") or password,
                    "ssh_private_key": str(state.get("ssh_private_key") or ""),
                }
        return {"ok": False, "logs": logs, "error": "insufficient disk space"}

    boot = vm_bootstrap(auto_install=auto_install)
    logs.append(f"virtualbox: {boot.get('message') or boot.get('error')}")
    if not boot.get("ok"):
        return {"ok": False, "logs": logs, "error": boot.get("error", "virtualbox unavailable")}

    update = vm_update_virtualbox(auto_update=auto_update)
    if update.get("ok"):
        logs.append(f"virtualbox update: {update.get('note') or 'ok'}")
    else:
        logs.append(f"virtualbox update failed: {update.get('error')}")

    if force_recreate is None:
        force_recreate = os.getenv("C0D3R_VM_FORCE_RECREATE", "0") in {"1", "true", "yes", "on"}
    if not force_recreate and vm_exists(vm_name):
        resume_timeout = float(os.getenv("C0D3R_VM_RESUME_TIMEOUT", "1800") or "1800")
        recovery_retries = int(os.getenv("C0D3R_VM_RECOVERY_RETRIES", "2") or "2")
        resume = vm_resume_or_recover(
            vm_name,
            user=user,
            timeout_s=resume_timeout,
            recovery_retries=recovery_retries,
        )
        logs.extend(resume.get("logs") or [])
        if resume.get("ok"):
            state = _load_state().get(vm_name, {})
            return {
                "ok": True,
                "logs": logs,
                "vm": vm_name,
                "ssh_port": int(state.get("ssh_port") or ssh_port),
                "user": state.get("user") or user,
                "password": state.get("password") or password,
                "ssh_private_key": str(state.get("ssh_private_key") or ""),
            }
        if resume.get("action") == "rebuild":
            logs.append("resume failed; recreating vm")
            vm_stop(vm_name, force=True)
            vm_delete(vm_name, delete_files=True)

    if not _guest_additions_path():
        ga = vm_download_guest_additions()
        logs.append(f"guest additions: {ga.get('message') or ga.get('error')}")
        additions_iso = ga.get("path")
    else:
        additions_iso = str(_guest_additions_path())

    fetch = vm_fetch_image(image_id, url=image_url_override)
    logs.append(f"image: {fetch.get('message') or fetch.get('note') or fetch.get('error')}")
    if not fetch.get("ok"):
        return {"ok": False, "logs": logs, "error": fetch.get("error", "image fetch failed")}

    image_path = fetch.get("path")
    if force_recreate:
        rc, out, err = _run_vboxmanage(["list", "vms"])
        if rc == 0 and vm_name in (out or ""):
            vm_stop(vm_name, force=True)
            vm_delete(vm_name, delete_files=True)
    create = vm_create(vm_name, image_path=image_path)
    logs.append(f"vm create: {create.get('message')}")
    if not create.get("ok"):
        return {"ok": False, "logs": logs, "error": create.get("error", "create failed")}

    enable = vm_enable_ssh(vm_name, port=ssh_port)
    logs.append(f"ssh port forward: {enable.get('message')}")

    ssh_key = vm_ensure_ssh_key(vm_name)
    if not ssh_key.get("ok"):
        logs.append(f"ssh key error: {ssh_key.get('error')}")
    if not password:
        password = f"c0d3r-{int(time.time())}"
    pub_key = ""
    try:
        pub_key = Path(ssh_key.get("public_key") or "").read_text(encoding="utf-8").strip()
    except Exception:
        pub_key = ""
    post_install = (
        "sudo apt-get update"
        " && sudo apt-get install -y openssh-server xdotool x11-utils build-essential dkms linux-headers-$(uname -r) "
        "virtualbox-guest-x11 virtualbox-guest-utils virtualbox-guest-dkms"
        " && sudo systemctl enable --now vboxservice || true"
        " && sudo modprobe vboxguest vboxsf vboxvideo || true"
        " && sudo mkdir -p /etc/gdm3"
        " && sudo touch /etc/gdm3/custom.conf"
        " && sudo sed -i 's/^#\\?WaylandEnable=.*/WaylandEnable=false/' /etc/gdm3/custom.conf"
        " && sudo sed -i 's/^#\\?AutomaticLoginEnable=.*/AutomaticLoginEnable=true/' /etc/gdm3/custom.conf"
        " && sudo sed -i 's/^#\\?AutomaticLogin=.*/AutomaticLogin={u}/' /etc/gdm3/custom.conf"
    ).format(u=user)
    if pub_key:
        post_install += (
            " && mkdir -p /home/{u}/.ssh && echo '{k}' >> /home/{u}/.ssh/authorized_keys"
            " && chown -R {u}:{u} /home/{u}/.ssh"
        ).format(u=user, k=pub_key.replace("'", ""))
    unattended = vm_unattended_install(
        vm_name,
        iso_path=str(image_path),
        user=user,
        password=password,
        full_name=user,
        hostname=vm_name,
        post_install=post_install,
        additions_iso=additions_iso,
    )
    logs.append(f"unattended install: {unattended.get('message')}")
    if not unattended.get("ok"):
        return {"ok": False, "logs": logs, "error": unattended.get('message', 'unattended failed')}

    start = vm_start(vm_name, headless=True)
    logs.append(f"vm start: {start.get('message')}")
    if not start.get("ok"):
        return {"ok": False, "logs": logs, "error": start.get("message", "start failed")}

    ga_wait = vm_wait_guest_additions(vm_name, timeout_s=600)
    logs.append(ga_wait.get("message") or ga_wait.get("error"))
    if not ga_wait.get("ok"):
        logs.append("guest additions missing; attempting repair")
        _log_line(f"autopilot guest additions repair triggered for {vm_name}")
        ga_repair = vm_repair_guest_additions(vm_name, user=user, password=password)
        logs.extend(ga_repair.get("logs") or [])
        ga_wait = vm_wait_guest_additions(vm_name, timeout_s=600)
        logs.append(ga_wait.get("message") or ga_wait.get("error"))
    if not ga_wait.get("ok"):
        if allow_fallback and image_id == "ubuntu" and not image_url_override:
            fallback_url, note = _resolve_ubuntu_iso(prefer_latest=False)
            if fallback_url:
                logs.append(f"fallback ubuntu LTS for guest additions: {note}")
                vm_stop(vm_name, force=True)
                vm_delete(vm_name, delete_files=True)
                return vm_autopilot(
                    image_id=image_id,
                    vm_name=vm_name,
                    auto_install=auto_install,
                    auto_update=auto_update,
                    min_free_gb=min_free_gb,
                    ssh_port=ssh_port,
                    user=user,
                    password=password,
                    force_recreate=True,
                    image_url_override=fallback_url,
                    allow_fallback=False,
                )
        return {"ok": False, "logs": logs, "error": "guest additions not ready"}

    user_wait = vm_wait_guest_user(vm_name, user, timeout_s=1200)
    logs.append(user_wait.get("message") or user_wait.get("error"))
    if not user_wait.get("ok"):
        logs.append("user not logged in; attempting gui recovery")
        _log_line(f"autopilot gui recovery triggered for {vm_name}")
        gui = vm_gui_recover(vm_name, user=user, password=password)
        logs.append(gui.get("message") or gui.get("error") or "")
        user_wait = vm_wait_guest_user(vm_name, user, timeout_s=300)
        logs.append(user_wait.get("message") or user_wait.get("error"))
    if not user_wait.get("ok") and allow_fallback and image_id == "ubuntu" and not image_url_override:
        fallback_url, note = _resolve_ubuntu_iso(prefer_latest=False)
        if fallback_url:
            logs.append(f"fallback ubuntu LTS: {note}")
            vm_stop(vm_name, force=True)
            vm_delete(vm_name, delete_files=True)
            return vm_autopilot(
                image_id=image_id,
                vm_name=vm_name,
                auto_install=auto_install,
                auto_update=auto_update,
                min_free_gb=min_free_gb,
                ssh_port=ssh_port,
                user=user,
                password=password,
                force_recreate=True,
                image_url_override=fallback_url,
                allow_fallback=False,
            )

    wait = vm_wait_ssh(vm_name, timeout_s=600)
    logs.append(wait.get("message") or wait.get("error"))
    if not wait.get("ok"):
        _log_line("ssh not ready; attempting guestcontrol repair")
        sudo_prefix = f"echo '{password}' | sudo -S "
        safe_key = pub_key.replace("'", "") if pub_key else ""
        vm_guest_exec(vm_name, f"{sudo_prefix}apt-get update -y", timeout_s=600)
        vm_guest_exec(vm_name, f"{sudo_prefix}apt-get install -y openssh-server", timeout_s=600)
        if safe_key:
            vm_guest_exec(
                vm_name,
                f"{sudo_prefix}mkdir -p /home/{user}/.ssh && echo '{safe_key}' >> /home/{user}/.ssh/authorized_keys && chown -R {user}:{user} /home/{user}/.ssh",
                timeout_s=120,
            )
        vm_guest_exec(vm_name, f"{sudo_prefix}systemctl enable ssh && {sudo_prefix}systemctl start ssh", timeout_s=120)
        wait = vm_wait_ssh(vm_name, timeout_s=300)
        logs.append(wait.get("message") or wait.get("error"))

    state = _load_state()
    state[vm_name] = {
        "image_id": image_id,
        "image_path": image_path,
        "guest_os": _guest_os_for_image(image_id),
        "user": user,
        "password": password,
        "ssh_port": ssh_port,
        "ssh_private_key": str(ssh_key.get("private_key") or ""),
        "ssh_public_key": str(ssh_key.get("public_key") or ""),
        "last_bootstrap": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_state(state)

    return {
        "ok": True,
        "logs": logs,
        "vm": vm_name,
        "ssh_port": ssh_port,
        "user": user,
        "password": password,
        "ssh_private_key": str(ssh_key.get("private_key") or ""),
    }


def vm_exec_ssh(name: str, command: str, timeout_s: float = 120.0) -> dict:
    if not name or not command:
        return {"ok": False, "error": "name and command required"}
    state = _load_state().get(name, {})
    user = state.get("user") or "c0d3r"
    port = int(state.get("ssh_port") or 2222)
    key_path = state.get("ssh_private_key") or ""
    password = state.get("password") or ""
    ssh = shutil.which("ssh")
    if not ssh:
        return {"ok": False, "error": "ssh client not available"}
    cmd = command.strip()
    if password and cmd.startswith("sudo ") and " -S" not in cmd and "--stdin" not in cmd:
        cmd = f"echo '{password}' | sudo -S {cmd[5:]}"
    args = [
        ssh,
        "-p",
        str(port),
        "-o",
        "ConnectTimeout=5",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]
    if key_path:
        args += ["-i", key_path]
    args.append(f"{user}@127.0.0.1")
    args.append(cmd)
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
        snippet = (proc.stdout.strip() or proc.stderr.strip() or "")[:400]
        _log_line(f"ssh exec {name}: {cmd} -> rc={proc.returncode} {snippet}")
        return {
            "ok": proc.returncode == 0,
            "rc": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        _log_line(f"ssh exec {name}: timeout {command}")
        return {"ok": False, "error": "ssh command timeout"}
    except Exception as exc:
        _log_line(f"ssh exec {name}: error {exc}")
        return {"ok": False, "error": str(exc)}


def vm_guest_exec(name: str, command: str, timeout_s: float = 120.0) -> dict:
    if not name or not command:
        return {"ok": False, "error": "name and command required"}
    state = _load_state().get(name, {})
    user = state.get("user") or "c0d3r"
    password = state.get("password") or ""
    if not password:
        return {"ok": False, "error": "no guest password available"}
    exe = _vboxmanage_path()
    if not exe:
        return {"ok": False, "error": "VBoxManage not found"}

    guest_os = _guest_os_for_vm(name)
    if guest_os == "windows":
        shell_exe = "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"
        args = [
            str(exe),
            "guestcontrol",
            name,
            "run",
            "--username",
            user,
            "--password",
            password,
            "--exe",
            shell_exe,
            "--",
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ]
    else:
        args = [
            str(exe),
            "guestcontrol",
            name,
            "run",
            "--username",
            user,
            "--password",
            password,
            "--exe",
            "/bin/bash",
            "--",
            "bash",
            "-lc",
            command,
        ]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
        _log_line(f"guest exec {name}: {command} -> rc={proc.returncode}")
        return {
            "ok": proc.returncode == 0,
            "rc": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        _log_line(f"guest exec {name}: timeout {command}")
        return {"ok": False, "error": "guest command timeout"}
    except Exception as exc:
        _log_line(f"guest exec {name}: error {exc}")
        return {"ok": False, "error": str(exc)}
