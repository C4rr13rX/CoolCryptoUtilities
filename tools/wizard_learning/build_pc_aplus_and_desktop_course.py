"""tools/wizard_learning/build_pc_aplus_and_desktop_course.py

Builds the Permissibly Licensed (`CC0 1.0 Universal` / Open Standards) Multi-OS PC
Basics, Maintenance, Diagnostics, CompTIA A+-Scope & Multimodal Desktop Navigation
Course, captures REAL Linux X11 desktop frames inside WSL2 Ubuntu (`Xvfb` + `Tkinter`
+ ImageMagick `import -window root`), extracts OpenCV + `POOL_IMAGE=2` visual
features, and trains W1z4rD's multi-pool & cross-modal substrates.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.wizard_learning.multimodal_desktop_navigator import (
    cofire_desktop_episode_to_wizard,
    extract_desktop_visual_features,
    plan_and_mutate_desktop_action,
)
from tools.wizard_learning.pc_diagnostics_and_os_engine import (
    LICENSE_TAG,
    PC_OS_APLUS_CONCEPTS,
)

COURSE_DIR = _PROJECT_ROOT / "data" / "wizard_learning" / "pc_aplus_and_desktop_course"


def _write_license_provenance() -> Path:
    COURSE_DIR.mkdir(parents=True, exist_ok=True)
    md = (
        "# PC Maintenance, Multi-OS Diagnostics, CompTIA A+-Scope & Multimodal Desktop Course — License & Provenance\n\n"
        "- **Course Compilation License**: `CC0 1.0 Universal (Public Domain Dedication)`\n"
        "- **Proprietary / Copyrighted Exam Dump Usage**: **0 bytes** (strictly zero CompTIA proprietary exam items or copyrighted textbook prose).\n"
        "- **Standards & Technical Provenance**:\n"
        "  1. **Hardware, Firmware & Storage**: UEFI Specification 2.10, ISO/IEC 11889 (TPM 2.0), JEDEC DDR4/DDR5, ATX 3.0, INCITS T13 ATA/S.M.A.R.T., NVM Express Base Spec 2.0, SNIA RAID DDF.\n"
        "  2. **Operating Systems**: IEEE Std 1003.1 (POSIX.1-2017), Linux `proc(5)` / `PSI` / `systemd(1)` / `freedesktop.org` XDG & `os-release`, Microsoft public PowerShell/DISM/SFC/CIM command interfaces, FreeBSD/Darwin `man` interfaces.\n"
        "  3. **Networking & Security**: IETF RFC 791 (IPv4), RFC 793 (TCP), RFC 826 (ARP), RFC 1035 (DNS), RFC 2131 (DHCP), RFC 3927 (Link-Local `169.254.0.0/16`), NIST SP 800-61r2 Incident Handling & SP 800-88 Media Sanitization.\n"
        "  4. **Virtualization & Empirical X11 Desktop Captures**: Oracle VirtualBox 7.2 `VBoxManage` CLI interface and original synthetic GUI layouts rendered live inside WSL2 Ubuntu `Xvfb` Xorg Virtual Framebuffer.\n"
    )
    out = COURSE_DIR / "LICENSE_AND_PROVENANCE.md"
    out.write_text(md, encoding="utf-8")
    return out


def _build_foundations_jsonl() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    # 1. Expand every PcOsConcept into general, Windows, Linux, macOS/BSD, and observation-mutation Q&A items
    for c in PC_OS_APLUS_CONCEPTS:
        # General + Multi-OS unified lesson
        unified_ans = (
            f"**{c.title}** *(General Principle + Multi-OS Realization)*\n\n"
            f"- **General Computer Principle**: {c.general_principle}\n"
            f"- **Windows 10/11**: {c.windows_specific}\n"
            f"- **Linux (Ubuntu/Fedora/Arch/Alpine)**: {c.linux_specific}\n"
            f"- **macOS / FreeBSD**: {c.macos_bsd_specific}\n"
            f"- **Harmless Empirical Probes**: {' | '.join(c.harmless_probes)}\n"
            f"- **Observation-Driven Action Mutation**: {c.remediation_and_mutation}\n"
            f"*(License: {LICENSE_TAG} · Ref: {c.standards_ref})*"
        )
        records.append({
            "id": f"pc_aplus_{c.concept_id}_unified",
            "category": f"pc_aplus_{c.domain}",
            "question": f"Explain {c.title} across Windows, Linux, and macOS.",
            "paraphrases": [
                f"What is the general principle and cross-OS behavior of {c.title}?",
                f"How does {c.title} work on Windows, Linux, and macOS/BSD?",
                f"PC basics and A+ guide: {c.title}",
            ],
            "answer": unified_ans,
            "license": LICENSE_TAG,
            "standards_ref": c.standards_ref,
        })

        # Diagnostic / Remediation focused lesson
        diag_ans = (
            f"**Diagnostic & Observation-Mutation Protocol — {c.title}**\n\n"
            f"1. **Harmless Read-Only Probes First**: {' ; '.join(c.harmless_probes)}\n"
            f"2. **Empirical Principle**: {c.general_principle}\n"
            f"3. **How Observations Mutate Action**: {c.remediation_and_mutation}"
        )
        records.append({
            "id": f"pc_aplus_{c.concept_id}_triage",
            "category": f"pc_aplus_{c.domain}",
            "question": f"How do you safely diagnose and troubleshoot {c.title}?",
            "paraphrases": [
                f"What harmless diagnostic commands check {c.title}?",
                f"CompTIA A+ troubleshooting protocol for {c.title}",
            ],
            "answer": diag_ans,
            "license": LICENSE_TAG,
            "standards_ref": c.standards_ref,
        })

    # 2. Additional Targeted CompTIA A+ Core 1 & Core 2 Concrete Essentials
    extra_lessons = [
        (
            "pc_ports_protocols_aplus",
            "networking_and_security",
            "What are the standard TCP and UDP port numbers for SSH, DNS, DHCP, HTTP, HTTPS, SMB, RDP, IMAP, and SMTP?",
            [
                "List essential CompTIA A+ network port numbers and transport protocols",
                "Standard well-known TCP/UDP ports for PC diagnostics and firewall rules",
            ],
            "**Essential IETF Well-Known Ports for Multi-OS Diagnostics & Firewall Triage**:\n"
            "- **`20/21` TCP**: FTP (File Transfer — legacy cleartext; prefer SFTP over `22`)\n"
            "- **`22` TCP**: **SSH / SCP / SFTP** (encrypted remote shell & file transfer)\n"
            "- **`23` TCP**: Telnet (unencrypted legacy console — disable in production)\n"
            "- **`25` TCP (`587` TLS submission / `465` SMTPS)**: **SMTP** outbound mail relay\n"
            "- **`53` UDP & TCP**: **DNS** (UDP for standard queries; TCP for zone transfers & large DNSSEC responses)\n"
            "- **`67/68` UDP**: **DHCP** (`67` server, `68` client — DORA lease negotiation)\n"
            "- **`80` TCP / `443` TCP & UDP (QUIC)**: **HTTP** / **HTTPS (TLS)**\n"
            "- **`110`/`995` TCP**: POP3 / POP3S · **`143`/`993` TCP**: **IMAP / IMAPS**\n"
            "- **`123` UDP**: **NTP** (clock sync — critical for Kerberos & TLS validity!)\n"
            "- **`389`/`636` TCP**: LDAP / LDAPS · **`445` TCP**: **SMB** (Windows/Samba file shares — block at WAN perimeter!)\n"
            "- **`3389` TCP & UDP**: **RDP** (Windows Remote Desktop — protect behind VPN/SSH tunnel)",
        ),
        (
            "pc_safe_mode_recovery_cross_os",
            "multi_os_administration",
            "How do you boot into Safe Mode or single-user recovery environment on Windows, Linux, and macOS?",
            [
                "Cross-OS recovery boot modes: Windows Safe Mode, Linux rescue target, macOS Recovery",
                "How to recover an unbootable desktop after a bad driver or fstab edit across operating systems",
            ],
            "**Cross-OS Minimal Recovery Environments (Isolating Kernel/Core OS from Faulty Drivers & Startup Services)**:\n"
            "- **General Principle**: Booting with minimal core drivers and supervisor units isolates whether a crash/black-screen stems from a third-party kernel module, GPU driver, or broken startup mount.\n"
            "- **Windows 10/11**: Hold `Shift` while clicking *Restart* -> *Troubleshoot -> Advanced options -> Startup Settings -> F4 (Safe Mode) / F5 (Safe Mode with Networking)*, or from WinRE run `bcdedit /set {default} safeboot minimal` (revert via `bcdedit /deletevalue {default} safeboot`).\n"
            "- **Linux (`systemd` / GRUB)**: Press `e` in GRUB and append `systemd.unit=rescue.target` (single-user root shell with mounts) or `systemd.unit=emergency.target` (minimal root before `/etc/fstab` — ideal when a bad `/etc/fstab` UUID blocks boot!).\n"
            "- **macOS**: On Apple Silicon, press and hold the **Power button** until *Loading startup options* appears (Safe Mode: hold `Shift` while clicking *Continue in Safe Mode*); on Intel Macs, hold `Cmd+R` for macOS Recovery or `Shift` at boot for Safe Mode.",
        ),
        (
            "pc_cidr_private_ipv4_ipv6",
            "networking_and_security",
            "What are the RFC 1918 private IPv4 ranges, loopback, APIPA link-local, and IPv6 equivalents?",
            [
                "Explain RFC 1918 private IP addresses, loopback 127.0.0.1, APIPA 169.254.x.x, and IPv6 fe80::",
                "IPv4 CIDR subnet mask and private address space reference for A+ networking",
            ],
            "**IPv4 & IPv6 Special Address Blocks (IETF RFC 1918 / RFC 3927 / RFC 4291)**:\n"
            "- **RFC 1918 Private LAN Ranges** (non-routable on public internet; require NAT): `10.0.0.0/8` (`255.0.0.0`), `172.16.0.0/12` (`172.16.0.0`–`172.31.255.255`), and `192.168.0.0/16` (`255.255.0.0`).\n"
            "- **Loopback (Localhost Stack Verification)**: IPv4 `127.0.0.0/8` (`127.0.0.1`) · IPv6 `::1/128`.\n"
            "- **Link-Local / APIPA (No DHCP Server Reached)**: IPv4 `169.254.0.0/16` · IPv6 `fe80::/10` (mandatory on every IPv6 interface for Neighbor Discovery Protocol!).\n"
            "- **Global Unicast IPv6**: `2000::/3` · **Unique Local IPv6 (ULA)**: `fc00::/7` (`fd00::/8`).\n"
            "- **Usable Host Count Formula**: For IPv4 prefix `/p`, total addresses $= 2^{32-p}$ and usable hosts $= 2^{32-p} - 2$ (subtracting network and broadcast addresses; e.g., `/24` = `254` hosts, `/26` = `62` hosts, `/30` = `2` point-to-point hosts).",
        ),
    ]

    for lid, dom, q, paras, ans in extra_lessons:
        records.append({
            "id": lid,
            "category": f"pc_aplus_{dom}",
            "question": q,
            "paraphrases": paras,
            "answer": ans,
            "license": LICENSE_TAG,
            "standards_ref": "IETF RFCs & Cross-OS Vendor Public CLI Documentation",
        })

    out_path = COURSE_DIR / "pc_aplus_foundations_and_multi_os.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


# ── Live Empirical WSL2 Ubuntu X11 (`Xvfb`) Desktop Scene Capture ────────────

_WSL_X11_CAPTURE_SCRIPT = r"""
import base64, json, os, subprocess, time, tkinter as tk

SCENES = [
    ("ubuntu_gnome_normal", "open_launcher_or_search", "windows11"),
    ("win11_centered_normal", "open_launcher_or_search", "windows10"),
    ("macos_topleft_controls", "close_active_window", "windows11"),
    ("win_kde_topright_controls", "close_active_window", "macos"),
    ("blocking_red_error_modal", "activate_primary_button", "windows11"),
    ("blocking_amber_auth_modal", "open_or_focus_terminal", "ubuntu"),
    ("disabled_btn_unchecked_prereq", "activate_primary_button", "windows11"),
    ("enabled_btn_checked_prereq", "activate_primary_button", "windows11"),
    ("open_terminal_pane_top_panel", "open_or_focus_terminal", "windows11"),
    ("uefi_blue_tui_console", "open_launcher_or_search", "windows11"),
]

def draw_scene(canvas, kind, W=1024, H=768):
    canvas.delete("all")
    if kind == "uefi_blue_tui_console":
        canvas.configure(bg="#001b7a")
        canvas.create_rectangle(0, 0, W, H, fill="#001b7a", outline="")
        canvas.create_text(W//2, 36, text="UEFI Platform Setup Utility — Boot & Security", fill="#ffffff", font=("Courier", 16, "bold"))
        canvas.create_rectangle(60, 80, W-60, H-80, outline="#b8c8ff", width=2)
        canvas.create_text(120, 140, text="[Boot Mode: UEFI Only]   [Secure Boot: Enabled]   [VT-x: Enabled]", fill="#ffff66", anchor="w", font=("Courier", 13))
        canvas.create_text(120, 210, text="Use Arrow Keys / Tab / F10 to Save & Exit (No Mouse Cursor)", fill="#dce4ff", anchor="w", font=("Courier", 12))
        return

    # Desktop wallpaper
    canvas.create_rectangle(0, 0, W, H, fill="#243447", outline="")

    # Panels / Docks
    if kind in ("ubuntu_gnome_normal", "macos_topleft_controls", "open_terminal_pane_top_panel"):
        canvas.create_rectangle(0, 0, W, 36, fill="#14181f", outline="#4a5568", width=2)
        canvas.create_text(52, 18, text="Activities", fill="#f7fafc", font=("Sans", 10, "bold"))
        canvas.create_text(W//2, 18, text="11:45", fill="#e2e8f0", font=("Sans", 10))
    if kind == "ubuntu_gnome_normal":
        canvas.create_rectangle(0, 36, 64, H, fill="#1a202c", outline="#4a5568", width=2)
        for idx, col in enumerate(("#dd4814", "#3182ce", "#38a169", "#805ad5")):
            y0 = 54 + idx * 56
            canvas.create_rectangle(14, y0, 50, y0+36, fill=col, outline="#edf2f7", width=1)

    if kind in ("win11_centered_normal", "win_kde_topright_controls", "blocking_red_error_modal", "blocking_amber_auth_modal", "disabled_btn_unchecked_prereq", "enabled_btn_checked_prereq"):
        canvas.create_rectangle(0, H-48, W, H, fill="#11161d", outline="#4a5568", width=2)
        if kind == "win11_centered_normal":
            for idx, col in enumerate(("#00a4ef", "#ecc94b", "#48bb78", "#ed64a6")):
                x0 = 430 + idx * 44
                canvas.create_rectangle(x0, H-38, x0+28, H-10, fill=col, outline="#ffffff", width=1)
        else:
            canvas.create_rectangle(14, H-38, 46, H-10, fill="#00a4ef", outline="#ffffff", width=1)
            canvas.create_rectangle(58, H-38, 90, H-10, fill="#4a5568", outline="#cbd5e0", width=1)

    # Application Window
    wx, wy, ww, wh = 130, 80, 760, 540
    canvas.create_rectangle(wx, wy, wx+ww, wy+wh, fill="#f7fafc", outline="#2d3748", width=2)
    canvas.create_rectangle(wx, wy, wx+ww, wy+36, fill="#e2e8f0", outline="#cbd5e0", width=2)
    if kind == "macos_topleft_controls":
        canvas.create_oval(wx+12, wy+11, wx+26, wy+25, fill="#ff5f56", outline="#c53030")
        canvas.create_oval(wx+34, wy+11, wx+48, wy+25, fill="#ffbd2e", outline="#b7791f")
        canvas.create_oval(wx+56, wy+11, wx+70, wy+25, fill="#27c93f", outline="#276749")
        canvas.create_text(wx+ww//2, wy+18, text="System Preferences — Network", fill="#1a202c", font=("Sans", 11, "bold"))
    else:
        canvas.create_text(wx+18, wy+18, text="System Maintenance & Package Manager", fill="#1a202c", anchor="w", font=("Sans", 11, "bold"))
        canvas.create_rectangle(wx+ww-38, wy+6, wx+ww-8, wy+30, fill="#e53e3e", outline="#9b2c2c", width=1)
        canvas.create_text(wx+ww-23, wy+18, text="X", fill="#ffffff", font=("Sans", 10, "bold"))

    # Interior widgets by scene
    if kind == "open_terminal_pane_top_panel":
        canvas.create_rectangle(wx+30, wy+60, wx+ww-30, wy+wh-40, fill="#0d1117", outline="#4a5568", width=2)
        canvas.create_text(wx+50, wy+95, text="adam@ubuntu-vm:~$ uname -sr && df -h /", fill="#3fb950", anchor="w", font=("Courier", 12, "bold"))
        canvas.create_text(wx+50, wy+125, text="Linux 6.18.58-WSL2   42G free of 250G (18%)", fill="#c9d1d9", anchor="w", font=("Courier", 11))
    elif kind == "disabled_btn_unchecked_prereq":
        canvas.create_text(wx+50, wy+110, text="Driver & Firmware Repair Wizard", fill="#1a202c", anchor="w", font=("Sans", 14, "bold"))
        canvas.create_rectangle(wx+56, wy+396, wx+78, wy+418, fill="#ffffff", outline="#2d3748", width=2)
        canvas.create_text(wx+92, wy+407, text="I have verified backup & accept CC0 repair terms", fill="#2d3748", anchor="w", font=("Sans", 11))
        canvas.create_rectangle(wx+ww-190, wy+450, wx+ww-50, wy+492, fill="#b0b7c3", outline="#8c96a6", width=2)
        canvas.create_text(wx+ww-120, wy+471, text="Continue", fill="#6b7280", font=("Sans", 11, "bold"))
    elif kind == "enabled_btn_checked_prereq":
        canvas.create_text(wx+50, wy+110, text="Driver & Firmware Repair Wizard", fill="#1a202c", anchor="w", font=("Sans", 14, "bold"))
        canvas.create_rectangle(wx+56, wy+396, wx+78, wy+418, fill="#2b6cb0", outline="#1a365d", width=2)
        canvas.create_line(wx+60, wy+407, wx+66, wy+414, wx+75, wy+400, fill="#ffffff", width=2)
        canvas.create_text(wx+92, wy+407, text="I have verified backup & accept CC0 repair terms", fill="#2d3748", anchor="w", font=("Sans", 11))
        canvas.create_rectangle(wx+ww-190, wy+450, wx+ww-50, wy+492, fill="#0078d4", outline="#004578", width=2)
        canvas.create_text(wx+ww-120, wy+471, text="Continue", fill="#ffffff", font=("Sans", 11, "bold"))
    else:
        canvas.create_rectangle(wx+40, wy+60, wx+ww-40, wy+96, fill="#ffffff", outline="#718096", width=2)
        canvas.create_text(wx+56, wy+78, text="Search settings, disks, network...", fill="#4a5568", anchor="w", font=("Sans", 10))
        canvas.create_rectangle(wx+ww-190, wy+450, wx+ww-50, wy+492, fill="#0078d4", outline="#004578", width=2)
        canvas.create_text(wx+ww-120, wy+471, text="Apply", fill="#ffffff", font=("Sans", 11, "bold"))

    # Blocking Centered Modals
    if kind in ("blocking_red_error_modal", "blocking_amber_auth_modal"):
        mx, my, mw, mh = 310, 235, 404, 250
        canvas.create_rectangle(mx, my, mx+mw, my+mh, fill="#ffffff", outline="#1a202c", width=3)
        hdr_col = "#d93025" if kind == "blocking_red_error_modal" else "#dd6b20"
        hdr_txt = "Critical Mount Lock — Action Blocked" if kind == "blocking_red_error_modal" else "Polkit / UAC Authentication Required"
        canvas.create_rectangle(mx, my, mx+mw, my+42, fill=hdr_col, outline="#1a202c", width=2)
        canvas.create_text(mx+18, my+21, text=hdr_txt, fill="#ffffff", anchor="w", font=("Sans", 11, "bold"))
        canvas.create_text(mx+24, my+105, text="Underlying window controls are occluded.\nResolve or dismiss this modal first.", fill="#1a202c", anchor="w", font=("Sans", 10))
        canvas.create_rectangle(mx+mw-148, my+mh-56, mx+mw-28, my+mh-18, fill="#2b6cb0", outline="#1a365d", width=2)
        canvas.create_text(mx+mw-88, my+mh-37, text="Acknowledge", fill="#ffffff", font=("Sans", 10, "bold"))

root = tk.Tk()
root.overrideredirect(True)
root.geometry("1024x768+0+0")
canvas = tk.Canvas(root, width=1024, height=768, highlightthickness=0)
canvas.pack(fill="both", expand=True)

results = []
for kind, goal_kind, assumed_os in SCENES:
    draw_scene(canvas, kind)
    root.update_idletasks()
    root.update()
    time.sleep(0.08)
    png_path = f"/tmp/w1z_x11_{kind}.png"
    subprocess.run(["import", "-window", "root", png_path], check=True)
    with open(png_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    os.remove(png_path)
    results.append({"scene_id": kind, "goal_kind": goal_kind, "assumed_os": assumed_os, "png_b64": b64})

print("X11_JSON_BEGIN")
print(json.dumps(results))
print("X11_JSON_END")
root.destroy()
"""


def _capture_real_wsl_x11_desktop_lessons(endpoint: str = "http://127.0.0.1:8090") -> List[Dict[str, Any]]:
    wsl_bin = shutil.which("wsl")
    if not wsl_bin:
        raise RuntimeError("WSL binary not found on host")

    cp = subprocess.run(
        [wsl_bin, "-d", "Ubuntu", "--", "xvfb-run", "-a", "-s", "-screen 0 1024x768x24", "python3", "-c", _WSL_X11_CAPTURE_SCRIPT],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    if cp.returncode != 0 or "X11_JSON_BEGIN" not in cp.stdout:
        raise RuntimeError(f"WSL Xvfb capture failed (rc={cp.returncode}): {cp.stderr[:500]}")

    payload_s = cp.stdout.split("X11_JSON_BEGIN", 1)[1].split("X11_JSON_END", 1)[0].strip()
    raw_scenes = json.loads(payload_s)

    goal_prompts = {
        "open_launcher_or_search": "Open the system launcher menu to search for System Monitor",
        "close_active_window": "Close the active window on the desktop",
        "activate_primary_button": "Click Continue / Apply primary button to proceed",
        "open_or_focus_terminal": "Open or focus the terminal shell to run diagnostics",
    }

    lessons: List[Dict[str, Any]] = []
    cofired = 0
    for sc in raw_scenes:
        png_bytes = base64.b64decode(sc["png_b64"])
        sha = hashlib.sha256(png_bytes).hexdigest()[:16]
        vf = extract_desktop_visual_features(png_bytes)
        goal_text = goal_prompts[sc["goal_kind"]]
        plan = plan_and_mutate_desktop_action(goal_text, vf, assumed_os=sc["assumed_os"])
        if cofire_desktop_episode_to_wizard(goal_text, vf, plan, endpoint=endpoint, timeout=0.6):
            cofired += 1

        act = plan["selected_action"]
        ans = (
            f"**Empirical X11 Multimodal Desktop Lesson (`{sc['scene_id']}`, sha256=`{sha}`)**\n\n"
            f"- **Goal**: {goal_text} *(prior OS assumption: `{sc['assumed_os']}`)*\n"
            f"- **Observed Visual Features (`POOL_IMAGE=2`)**: `{vf['layout']['inferred_desktop_family']}`, "
            f"`panels={vf['layout']['panel_edges']}`, `controls={vf['layout']['window_controls_side']}`, `anomalies={vf['anomalies']}`\n"
            f"- **Mutated by Live Observation**: `{plan['mutated']}` ({'; '.join(plan['observation_overrides']) or 'baseline preconditions verified'})\n"
            f"- **Selected Action**: **`{act['action_type']}`** at `({act.get('x')}, {act.get('y')})` — {act['rationale']}"
        )
        lessons.append({
            "id": f"x11_multimodal_{sc['scene_id']}",
            "category": "multimodal_desktop_navigation",
            "scene_id": sc["scene_id"],
            "screenshot_sha256_16": sha,
            "captured_via": "WSL2 Ubuntu Xvfb 1024x768x24 + ImageMagick import -window root",
            "question": f"How should desktop navigation act on scene {sc['scene_id']} when trying to '{goal_text}'?",
            "paraphrases": [
                f"Multimodal desktop action plan for {sc['scene_id']} ({vf['layout']['inferred_desktop_family']})",
            ],
            "answer": ans,
            "visual_features": vf,
            "desktop_plan": plan,
            "license": LICENSE_TAG,
        })

    out_path = COURSE_DIR / "multimodal_desktop_observation_lessons.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in lessons:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[X11 Capture] Rendered, analyzed & co-fired {len(lessons)} live Linux X11 desktop scenes (cofired_ok={cofired}).")
    return lessons


def _train_wizard_chat_pool(records: List[Dict[str, Any]], endpoint: str = "http://127.0.0.1:8090") -> int:
    ep = endpoint.rstrip("/")
    trained = 0
    for pid in ("chat_in", "chat_out"):
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{ep}/multi_pool/register",
                    data=json.dumps({"pool_id": pid}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=3.0,
            ).read()
        except Exception:
            pass

    consecutive_timeouts = 0
    for r in records:
        if consecutive_timeouts >= 2:
            break
        q = str(r.get("question") or "")
        a = str(r.get("answer") or "")[:1800]
        if not q or not a:
            continue
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{ep}/multi_pool/train_pair",
                    data=json.dumps({
                        "src_pool": "chat_in",
                        "src": q,
                        "tgt_pool": "chat_out",
                        "tgt": a,
                        "passes": 18,
                        "lr": 0.45,
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=1.2,
            ).read()
            trained += 1
            consecutive_timeouts = 0
        except Exception:
            consecutive_timeouts += 1
            continue
    return trained


def main() -> None:
    t0 = time.perf_counter()
    lic = _write_license_provenance()
    foundations = _build_foundations_jsonl()
    x11_lessons = _capture_real_wsl_x11_desktop_lessons()
    trained = _train_wizard_chat_pool(foundations + x11_lessons)
    dt = time.perf_counter() - t0
    summary = {
        "license": LICENSE_TAG,
        "license_doc": str(lic.relative_to(_PROJECT_ROOT)),
        "foundations_records": len(foundations),
        "live_x11_multimodal_scenes": len(x11_lessons),
        "wizard_multi_pool_trained": trained,
        "elapsed_s": round(dt, 2),
        "x11_scene_verifications": [
            {
                "scene_id": x["scene_id"],
                "inferred_family": x["visual_features"]["layout"]["inferred_desktop_family"],
                "anomalies": x["visual_features"]["anomalies"],
                "mutated": x["desktop_plan"]["mutated"],
                "action_type": x["desktop_plan"]["selected_action"]["action_type"],
                "coords": [x["desktop_plan"]["selected_action"].get("x"), x["desktop_plan"]["selected_action"].get("y")],
            }
            for x in x11_lessons
        ],
    }
    (COURSE_DIR / "course_build_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
