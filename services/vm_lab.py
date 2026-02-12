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


def _resolve_ubuntu_iso() -> Tuple[Optional[str], str]:
    meta = _read_text("https://changelogs.ubuntu.com/meta-release-lts")
    versions = re.findall(r"Version:\s*(\d+\.\d+(?:\.\d+)?)\s+LTS", meta)
    if not versions:
        return None, "No Ubuntu LTS found"
    latest = sorted(versions, key=_version_tuple, reverse=True)[0]
    base = f"https://releases.ubuntu.com/{latest}/"
    desktop = f"{base}ubuntu-{latest}-desktop-amd64.iso"
    server = f"{base}ubuntu-{latest}-live-server-amd64.iso"
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
        return _resolve_ubuntu_iso()
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


def vm_start(name: str, *, headless: bool = True) -> dict:
    if not name:
        return {"ok": False, "error": "vm name required"}
    args = ["startvm", name]
    if headless:
        args += ["--type", "headless"]
    rc, out, err = _run_vboxmanage(args)
    _log_line(f"vm start {name} -> rc={rc}")
    return {"ok": rc == 0, "message": out or err}


def vm_stop(name: str, *, force: bool = False) -> dict:
    if not name:
        return {"ok": False, "error": "vm name required"}
    cmd = ["controlvm", name, "poweroff" if force else "acpipowerbutton"]
    rc, out, err = _run_vboxmanage(cmd)
    _log_line(f"vm stop {name} -> rc={rc}")
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
    return {"ok": rc == 0, "message": out or err}


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
    "up_down": "e048",
    "up_up": "e0c8",
    "down_down": "e050",
    "down_up": "e0d0",
    "left_down": "e04b",
    "left_up": "e0cb",
    "right_down": "e04d",
    "right_up": "e0cd",
    "t_down": "14",
    "t_up": "94",
}


def vm_key_combo(name: str, combo: str) -> dict:
    combo = (combo or "").lower().strip()
    if combo == "ctrl+alt+t":
        sc = f"{_SCAN_CODES['ctrl_down']} {_SCAN_CODES['alt_down']} {_SCAN_CODES['t_down']} {_SCAN_CODES['t_up']} {_SCAN_CODES['alt_up']} {_SCAN_CODES['ctrl_up']}"
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
            result.update(vm_exec_ssh(name, cmd, timeout_s=float(step.get("timeout_s") or 120)))
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
) -> dict:
    _log_line(f"autopilot start image={image_id} vm={vm_name}")
    logs: List[str] = []
    disk = vm_check_disk(min_free_gb)
    logs.append(f"disk free {disk.get('free_gb')}GB (min {disk.get('min_required_gb')}GB)")
    if not disk.get("ok"):
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

    if not _guest_additions_path():
        ga = vm_download_guest_additions()
        logs.append(f"guest additions: {ga.get('message') or ga.get('error')}")
        additions_iso = ga.get("path")
    else:
        additions_iso = str(_guest_additions_path())

    fetch = vm_fetch_image(image_id)
    logs.append(f"image: {fetch.get('message') or fetch.get('note') or fetch.get('error')}")
    if not fetch.get("ok"):
        return {"ok": False, "logs": logs, "error": fetch.get("error", "image fetch failed")}

    image_path = fetch.get("path")
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
    post_install = "sudo apt-get update && sudo apt-get install -y openssh-server"
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

    wait = vm_wait_port("127.0.0.1", ssh_port, timeout_s=180)
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
        wait = vm_wait_port("127.0.0.1", ssh_port, timeout_s=180)
        logs.append(wait.get("message") or wait.get("error"))

    state = _load_state()
    state[vm_name] = {
        "image_id": image_id,
        "image_path": image_path,
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
    ssh = shutil.which("ssh")
    if not ssh:
        return {"ok": False, "error": "ssh client not available"}
    args = [
        ssh,
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]
    if key_path:
        args += ["-i", key_path]
    args.append(f"{user}@127.0.0.1")
    args.append(command)
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
        snippet = (proc.stdout.strip() or proc.stderr.strip() or "")[:400]
        _log_line(f"ssh exec {name}: {command} -> rc={proc.returncode} {snippet}")
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
