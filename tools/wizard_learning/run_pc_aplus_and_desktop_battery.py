"""tools/wizard_learning/run_pc_aplus_and_desktop_battery.py

Comprehensive Empirical Verification Battery for:
  1. Live Multi-OS Host Diagnostics (`Pegasus` Windows 11 + Live `WSL2 Ubuntu` + `VirtualBox 7.2.22`)
  2. Live Disposable VirtualBox VM (`VBoxManage`) Observation-Mutation Repair & Clean Teardown
  3. Live Closed-Loop X11 GUI Desktop Navigation inside `WSL2 Ubuntu` (`Xvfb` + `Tkinter` + `import`)
     driven PURELY by raw screenshot pixels sent over HTTP to `:8001/api/wizard-chat/message/`!
  4. Real Screenshot Upload Verification via `:8001/api/wizard-chat/upload/`
  5. 2,000 Diverse Untrained Multi-OS PC Maintenance, CompTIA A+ 7-Step Triage, RAID Math,
     Bottom-Up Networking, Security Quarantine, and Observation-Override Desktop Mutation Prompts!
"""
from __future__ import annotations

import base64
import concurrent.futures
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

WEB_BASE = os.environ.get("WIZARD_WEB_BASE", "http://127.0.0.1:8001").rstrip("/")
COURSE_DIR = _PROJECT_ROOT / "data" / "wizard_learning" / "pc_aplus_and_desktop_course"


def _post_json(path: str, payload: Dict[str, Any], timeout: float = 20.0) -> Dict[str, Any]:
    req = urllib.request.Request(
        f"{WEB_BASE}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── Suite 1: Live Host + WSL2 + VirtualBox Diagnostics ───────────────────────

def verify_live_host_diagnostics() -> Dict[str, Any]:
    r_msg = _post_json("/api/wizard-chat/message/", {
        "text": "Run a live harmless multi-OS PC diagnostic on this machine.",
        "allow_web_lookup": False,
    })
    ans = str(r_msg.get("answer") or "")
    r_agent = _post_json("/api/wizard-chat/agent/", {
        "prompt": "Diagnose this PC and check WSL2 Linux and VirtualBox status.",
        "session_key": "pc-diag-verify",
    })
    agent_ans = str(r_agent.get("reply") or r_agent.get("answer") or "")
    checks = {
        "msg_non_hypothesis": not bool(r_msg.get("is_hypothesis", True)),
        "has_windows_host": "Windows" in ans,
        "has_memory_gb": "GB available" in ans,
        "has_wsl2_ubuntu": "WSL2 Linux" in ans and "Linux" in ans,
        "has_virtualbox": "VirtualBox" in ans and "7.2" in ans,
        "agent_matches": ("Windows" in agent_ans and "VirtualBox" in agent_ans),
    }
    return {"passed": all(checks.values()), "checks": checks, "sample_excerpt": ans[:420]}


# ── Suite 2: Live Disposable VirtualBox 7.2.22 Observation-Mutation Lab ──────

def verify_live_virtualbox_disposable_mutation() -> Dict[str, Any]:
    vbox = Path(r"C:\Program Files\Oracle\VirtualBox\VBoxManage.exe")
    if not vbox.exists():
        return {"passed": False, "error": "VBoxManage.exe not found"}

    vm_name = f"w1z4rd_disposable_aplus_{uuid.uuid4().hex[:8]}"
    tmp_base = Path(tempfile.mkdtemp(prefix="w1z_vbox_"))
    created = False
    try:
        # 1. Create disposable VM in isolated temp folder
        cp_create = subprocess.run(
            [str(vbox), "createvm", "--name", vm_name, "--ostype", "Ubuntu_64", "--basefolder", str(tmp_base), "--register"],
            capture_output=True, text=True, timeout=15,
        )
        if cp_create.returncode != 0:
            return {"passed": False, "error": f"createvm failed: {cp_create.stderr}"}
        created = True

        # 2. Configure intentionally broken WAN topology (`--nic1 hostonly`)
        subprocess.run(
            [str(vbox), "modifyvm", vm_name, "--memory", "512", "--cpus", "1", "--nic1", "hostonly"],
            capture_output=True, text=True, timeout=15, check=True,
        )

        # 3. Empirically observe live VM machine-readable config
        cp_info1 = subprocess.run(
            [str(vbox), "showvminfo", vm_name, "--machinereadable"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        observed_before = "nic1=\"hostonly\"" in cp_info1.stdout

        # 4. Feed empirical observation to W1z4rD and verify it mutates NIC1 to `nat`!
        q = (
            f"VirtualBox VM `{vm_name}` has no internet access inside the guest. "
            f"Empirical `VBoxManage showvminfo` observation shows `nic1=\"hostonly\"`. "
            f"Why does this happen and how should we mutate the VM configuration?"
        )
        wiz = _post_json("/api/wizard-chat/message/", {"text": q, "allow_web_lookup": False})
        w_ans = str(wiz.get("answer") or "").lower()
        mutated_to_nat = ("nat" in w_ans and ("modifyvm" in w_ans or "host-only" in w_ans or "hostonly" in w_ans))

        # 5. Apply W1z4rD's recommended `--nic1 nat` mutation and verify empirically
        if mutated_to_nat:
            subprocess.run(
                [str(vbox), "modifyvm", vm_name, "--nic1", "nat"],
                capture_output=True, text=True, timeout=15, check=True,
            )
        cp_info2 = subprocess.run(
            [str(vbox), "showvminfo", vm_name, "--machinereadable"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        verified_after = "nic1=\"nat\"" in cp_info2.stdout

        return {
            "passed": bool(observed_before and mutated_to_nat and verified_after),
            "vm_name": vm_name,
            "observed_nic1_hostonly_before": observed_before,
            "wizard_recommended_nat_mutation": mutated_to_nat,
            "verified_nic1_nat_after": verified_after,
        }
    finally:
        if created:
            subprocess.run([str(vbox), "unregistervm", vm_name, "--delete"], capture_output=True, text=True, timeout=15)
        shutil.rmtree(tmp_base, ignore_errors=True)


# ── Suite 3: Live Closed-Loop Interactive X11 GUI Obstacle Course in WSL2 ────

_CLOSED_LOOP_X11_AGENT_SCRIPT = r"""
import base64, json, os, subprocess, sys, time, tkinter as tk, urllib.request

class LiveX11ObstacleApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.geometry("1024x768+0+0")
        self.canvas = tk.Canvas(self.root, width=1024, height=768, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.state = {}
        self.hitboxes = []

    def reset_obstacle(self, obs_id):
        self.state = {
            "obs_id": obs_id,
            "modal_open": obs_id == "modal_blocks_apply",
            "prereq_checked": False,
            "goal_completed": False,
            "steps_taken": 0,
            "actions_log": [],
        }
        self.render()

    def add_hitbox(self, name, x0, y0, x1, y1, fn):
        self.hitboxes.append((name, x0, y0, x1, y1, fn))

    def click_at(self, x, y):
        self.state["steps_taken"] += 1
        # Top-most hitboxes are registered last -> iterate in reverse!
        for name, x0, y0, x1, y1, fn in reversed(self.hitboxes):
            if x0 <= x <= x1 and y0 <= y <= y1:
                self.state["actions_log"].append(f"hit:{name}@({x},{y})")
                fn()
                self.render()
                return name
        self.state["actions_log"].append(f"miss@({x},{y})")
        return "miss"

    def render(self):
        c = self.canvas
        c.delete("all")
        self.hitboxes = []
        W, H = 1024, 768
        oid = self.state["obs_id"]

        # Wallpaper
        c.create_rectangle(0, 0, W, H, fill="#243447", outline="")

        # Shell panels by obstacle
        if oid in ("ubuntu_dock_launcher", "relocated_top_panel", "open_terminal_focus"):
            c.create_rectangle(0, 0, W, 36, fill="#14181f", outline="#4a5568", width=2)
            c.create_text(52, 18, text="Activities", fill="#f7fafc", font=("Sans", 10, "bold"))
            if oid == "relocated_top_panel":
                self.add_hitbox("top_launcher", 0, 0, 120, 36, lambda: self.state.__setitem__("goal_completed", True))
            if oid == "ubuntu_dock_launcher":
                c.create_rectangle(0, 36, 64, H, fill="#1a202c", outline="#4a5568", width=2)
                for idx, col in enumerate(("#dd4814", "#3182ce", "#38a169")):
                    y0 = 54 + idx * 56
                    c.create_rectangle(14, y0, 50, y0+36, fill=col, outline="#edf2f7", width=1)
                self.add_hitbox("left_dock_launcher", 0, 36, 68, 220, lambda: self.state.__setitem__("goal_completed", True))
        elif oid == "macos_topleft_close":
            c.create_rectangle(0, 0, W, 34, fill="#1a202c", outline="#4a5568", width=2)
            c.create_text(64, 17, text="Finder   File   Edit", fill="#f7fafc", font=("Sans", 10))
        else:
            c.create_rectangle(0, H-48, W, H, fill="#11161d", outline="#4a5568", width=2)
            c.create_rectangle(14, H-38, 46, H-10, fill="#00a4ef", outline="#ffffff", width=1)

        # App Window
        wx, wy, ww, wh = 130, 80, 760, 540
        c.create_rectangle(wx, wy, wx+ww, wy+wh, fill="#f7fafc", outline="#2d3748", width=2)
        c.create_rectangle(wx, wy, wx+ww, wy+36, fill="#e2e8f0", outline="#cbd5e0", width=2)

        if oid == "macos_topleft_close":
            c.create_oval(wx+12, wy+11, wx+26, wy+25, fill="#ff5f56", outline="#c53030")
            c.create_oval(wx+34, wy+11, wx+48, wy+25, fill="#ffbd2e", outline="#b7791f")
            c.create_oval(wx+56, wy+11, wx+70, wy+25, fill="#27c93f", outline="#276749")
            self.add_hitbox("topleft_close", wx+4, wy+4, wx+32, wy+32, lambda: self.state.__setitem__("goal_completed", True))
        else:
            c.create_rectangle(wx+ww-38, wy+6, wx+ww-8, wy+30, fill="#e53e3e", outline="#9b2c2c", width=1)
            c.create_text(wx+ww-23, wy+18, text="X", fill="#ffffff", font=("Sans", 10, "bold"))

        if oid == "open_terminal_focus":
            c.create_rectangle(wx+30, wy+60, wx+ww-30, wy+wh-40, fill="#0d1117", outline="#4a5568", width=2)
            c.create_text(wx+50, wy+95, text="adam@x11-lab:/tmp$ ready", fill="#3fb950", anchor="w", font=("Courier", 12, "bold"))
            self.add_hitbox("terminal_pane", wx+30, wy+60, wx+ww-30, wy+wh-40, lambda: self.state.__setitem__("goal_completed", True))
        elif oid == "disabled_prereq_flow":
            chk = self.state["prereq_checked"]
            c.create_rectangle(wx+56, wy+396, wx+78, wy+418, fill=("#2b6cb0" if chk else "#ffffff"), outline="#2d3748", width=2)
            if chk:
                c.create_line(wx+60, wy+407, wx+66, wy+414, wx+75, wy+400, fill="#ffffff", width=2)
            c.create_text(wx+92, wy+407, text="Verify disk backup before Continue", fill="#2d3748", anchor="w", font=("Sans", 11))
            self.add_hitbox("prereq_checkbox", wx+48, wy+388, wx+86, wy+426, lambda: self.state.__setitem__("prereq_checked", True))

            btn_fill = "#0078d4" if chk else "#b0b7c3"
            txt_fill = "#ffffff" if chk else "#6b7280"
            c.create_rectangle(wx+ww-190, wy+450, wx+ww-50, wy+492, fill=btn_fill, outline="#4a5568", width=2)
            c.create_text(wx+ww-120, wy+471, text="Continue", fill=txt_fill, font=("Sans", 11, "bold"))
            if chk:
                self.add_hitbox("enabled_continue", wx+ww-190, wy+450, wx+ww-50, wy+492, lambda: self.state.__setitem__("goal_completed", True))
            else:
                self.add_hitbox("disabled_continue_noop", wx+ww-190, wy+450, wx+ww-50, wy+492, lambda: None)
        else:
            c.create_rectangle(wx+ww-190, wy+450, wx+ww-50, wy+492, fill="#0078d4", outline="#004578", width=2)
            c.create_text(wx+ww-120, wy+471, text="Apply", fill="#ffffff", font=("Sans", 11, "bold"))
            if oid == "modal_blocks_apply" and not self.state["modal_open"]:
                self.add_hitbox("apply_button", wx+ww-190, wy+450, wx+ww-50, wy+492, lambda: self.state.__setitem__("goal_completed", True))
            elif oid == "modal_blocks_apply" and self.state["modal_open"]:
                # Blocked while modal is open!
                self.add_hitbox("occluded_apply_blocked", wx+ww-190, wy+450, wx+ww-50, wy+492, lambda: None)

        if self.state["modal_open"]:
            mx, my, mw, mh = 310, 235, 404, 250
            c.create_rectangle(mx, my, mx+mw, my+mh, fill="#ffffff", outline="#1a202c", width=3)
            c.create_rectangle(mx, my, mx+mw, my+42, fill="#d93025", outline="#1a202c", width=2)
            c.create_text(mx+18, my+21, text="Lock Conflict — Dismiss First", fill="#ffffff", anchor="w", font=("Sans", 11, "bold"))
            c.create_rectangle(mx+mw-148, my+mh-56, mx+mw-28, my+mh-18, fill="#2b6cb0", outline="#1a365d", width=2)
            c.create_text(mx+mw-88, my+mh-37, text="Acknowledge", fill="#ffffff", font=("Sans", 10, "bold"))
            # Modal hitbox blocks entire modal area + dismiss button
            self.add_hitbox("modal_body_block", mx, my, mx+mw, my+mh, lambda: None)
            self.add_hitbox("modal_ack_btn", mx+mw-152, my+mh-60, mx+mw-24, my+mh-14, lambda: self.state.__setitem__("modal_open", False))

        self.root.update_idletasks()
        self.root.update()
        time.sleep(0.05)

    def grab_png_b64(self):
        p = "/tmp/w1z_closed_loop.png"
        subprocess.run(["import", "-window", "root", p], check=True)
        with open(p, "rb") as fh:
            b = base64.b64encode(fh.read()).decode("ascii")
        os.remove(p)
        return b

app = LiveX11ObstacleApp()
OBSTACLES = [
    ("modal_blocks_apply", "Click Apply primary button to save configuration", "windows11", 3),
    ("disabled_prereq_flow", "Click Continue primary button to install update", "windows11", 3),
    ("macos_topleft_close", "Close the active window", "windows11", 2),
    ("ubuntu_dock_launcher", "Open the launcher menu to find System Monitor", "windows11", 2),
    ("relocated_top_panel", "Open the launcher menu on this desktop", "windows10", 2),
    ("open_terminal_focus", "Open or focus the terminal shell", "windows11", 2),
]

outcomes = []
for oid, goal, assumed_os, max_steps in OBSTACLES:
    app.reset_obstacle(oid)
    mutated_any = False
    for step in range(max_steps):
        if app.state["goal_completed"]:
            break
        b64 = app.grab_png_b64()
        req = urllib.request.Request(
            f"{WEB_BASE}/api/wizard-chat/message/",
            data=json.dumps({
                "text": goal,
                "image_b64": b64,
                "context": {"assumed_os": assumed_os},
                "env_vars": {"fs_readonly": (oid == "open_terminal_focus")},
                "allow_web_lookup": False,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        plan = res.get("desktop_plan") or {}
        if plan.get("mutated"):
            mutated_any = True
        act = plan.get("selected_action") or {}
        ax, ay = act.get("x"), act.get("y")
        if ax is not None and ay is not None:
            app.click_at(int(ax), int(ay))
    outcomes.append({
        "obstacle_id": oid,
        "goal_completed": app.state["goal_completed"],
        "steps_taken": app.state["steps_taken"],
        "mutated_by_observation": mutated_any,
        "actions_log": app.state["actions_log"],
    })

app.root.destroy()
print("CLOSED_LOOP_BEGIN")
print(json.dumps(outcomes))
print("CLOSED_LOOP_END")
"""


def verify_live_wsl_x11_closed_loop_navigation() -> Dict[str, Any]:
    wsl_bin = shutil.which("wsl")
    if not wsl_bin:
        return {"passed": False, "error": "WSL missing"}

    # Spawn live interactive X11 GUI server inside WSL2 Ubuntu `Xvfb` (`import -window root` real PNG captures)
    # and post each raw X11 screenshot frame to Windows `127.0.0.1:8001/api/wizard-chat/message/`!
    pipe_x11_script = _CLOSED_LOOP_X11_AGENT_SCRIPT.split("app = LiveX11ObstacleApp()", 1)[0] + r"""
app = LiveX11ObstacleApp()
for line in sys.stdin:
    cmd = json.loads(line.strip())
    op = cmd["op"]
    if op == "reset":
        app.reset_obstacle(cmd["obs_id"])
        print(json.dumps({"ok": True, "png_b64": app.grab_png_b64(), "state": app.state}), flush=True)
    elif op == "click":
        hit = app.click_at(int(cmd["x"]), int(cmd["y"]))
        print(json.dumps({"hit": hit, "png_b64": app.grab_png_b64(), "state": app.state}), flush=True)
    elif op == "quit":
        break
app.root.destroy()
"""
    proc = subprocess.Popen(
        [wsl_bin, "-d", "Ubuntu", "--", "xvfb-run", "-a", "-s", "-screen 0 1024x768x24", "python3", "-u", "-c", pipe_x11_script],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
    )
    obstacles = [
        ("modal_blocks_apply", "Click Apply primary button to save configuration", "windows11", 3),
        ("disabled_prereq_flow", "Click Continue primary button to install update", "windows11", 3),
        ("macos_topleft_close", "Close the active window", "windows11", 2),
        ("ubuntu_dock_launcher", "Open the launcher menu to find System Monitor", "windows11", 2),
        ("relocated_top_panel", "Open the launcher menu on this desktop", "windows10", 2),
        ("open_terminal_focus", "Open or focus the terminal shell", "windows11", 2),
    ]
    outcomes = []
    try:
        assert proc.stdin and proc.stdout
        for oid, goal, assumed_os, max_steps in obstacles:
            proc.stdin.write(json.dumps({"op": "reset", "obs_id": oid}) + "\n")
            proc.stdin.flush()
            st_pkt = json.loads(proc.stdout.readline())
            mutated_any = False
            for _ in range(max_steps):
                if st_pkt["state"]["goal_completed"]:
                    break
                res = _post_json("/api/wizard-chat/message/", {
                    "text": goal,
                    "image_b64": st_pkt["png_b64"],
                    "context": {"assumed_os": assumed_os},
                    "env_vars": {"fs_readonly": (oid == "open_terminal_focus")},
                    "allow_web_lookup": False,
                })
                plan = res.get("desktop_plan") or {}
                if plan.get("mutated"):
                    mutated_any = True
                act = plan.get("selected_action") or {}
                ax, ay = act.get("x"), act.get("y")
                if ax is not None and ay is not None:
                    proc.stdin.write(json.dumps({"op": "click", "x": int(ax), "y": int(ay)}) + "\n")
                    proc.stdin.flush()
                    st_pkt = json.loads(proc.stdout.readline())
            outcomes.append({
                "obstacle_id": oid,
                "goal_completed": st_pkt["state"]["goal_completed"],
                "steps_taken": st_pkt["state"]["steps_taken"],
                "mutated_by_observation": mutated_any,
                "actions_log": st_pkt["state"]["actions_log"],
            })
        proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
        proc.stdin.flush()
    finally:
        proc.terminate()

    passed_cnt = sum(1 for o in outcomes if o["goal_completed"] and o["mutated_by_observation"])
    return {
        "passed": passed_cnt == len(outcomes),
        "completed": f"{passed_cnt}/{len(outcomes)}",
        "transport": "WSL2 Xvfb Live X11 Pipe <-> Windows :8001 HTTP",
        "outcomes": outcomes,
    }


# ── Suite 4: 2,000 Untrained Multi-OS, A+ & Observation-Override Scenarios ───

def build_2000_untrained_pc_and_desktop_cases() -> List[Tuple[str, str, Callable[[str], bool]]]:
    rng = random.Random(20261121)
    cases: List[Tuple[str, str, Callable[[str], bool]]] = []

    hosts = [f"Node-{w}-{i}" for w in ("Atlas", "Borealis", "Cygnus", "Draco", "Eridanus", "Fornax", "Gemini", "Hydra", "Indus", "Juno") for i in range(1, 26)]

    # 1. 500 Hardware, S.M.A.R.T., RAID Math (+ Broken Preconditions), Boot & Peripheral Physics
    raid_specs = [
        ("0", 2, 4, 8, False), ("0", 4, 2, 8, False), ("1", 2, 8, 8, False),
        ("5", 4, 4, 12, False), ("5", 5, 6, 24, False), ("5", 2, 4, 0, True),
        ("6", 6, 8, 32, False), ("6", 3, 10, 0, True), ("10", 4, 6, 12, False), ("10", 5, 4, 0, True),
    ]
    for i in range(500):
        h = hosts[i % len(hosts)]
        mode = i % 5
        if mode == 0:
            lvl, nd, sz, exp_u, broken = raid_specs[(i // 5) % len(raid_specs)]
            # Vary drive size dynamically so combinations are genuinely untrained
            mult = ((i // 50) % 4) + 1
            sz_v = sz * mult
            exp_v = exp_u * mult
            q = f"On storage server `{h}`, an admin configures {nd} x {sz_v} TB NVMe drives in RAID {lvl}. What is the usable capacity and fault tolerance?"
            if broken:
                cases.append(("hardware_raid_aplus", q, lambda a: "precondition" in a.lower() or "requires at least" in a.lower() or "even number" in a.lower()))
            else:
                cases.append(("hardware_raid_aplus", q, lambda a, ev=exp_v: f"{ev} tb" in a.lower()))
        elif mode == 1:
            pend = rng.randint(14, 420)
            q = f"Workstation `{h}` is slow opening files. You plan to run `chkdsk /r` overnight, but `smartctl` observation reveals `Current_Pending_Sector = {pend}` and `Media and Data Integrity Errors = 9`. How must this observation mutate your plan?"
            cases.append(("hardware_raid_aplus", q, lambda a: ("ddrescue" in a.lower() or "read-only" in a.lower() or "backup" in a.lower()) and "chkdsk" in a.lower()))
        elif mode == 2:
            q = f"After a BIOS battery replacement on `{h}`, firmware reports `No Bootable Device`, and you observe CSM Legacy mode got enabled on a GPT NVMe drive. Should you wipe and reinstall the OS or mutate firmware settings?"
            cases.append(("hardware_raid_aplus", q, lambda a: "uefi" in a.lower() and ("reformat" in a.lower() or "preserve" in a.lower() or "gpt" in a.lower())))
        elif mode == 3:
            q = f"Laser printer `{h}-PRN` prints sharp text, but toner rubs off easily onto the user's fingers when touched. Which stage and component of the 7-step laser imaging process failed?"
            cases.append(("hardware_raid_aplus", q, lambda a: "fuser" in a.lower()))
        else:
            q = f"Laptop `{h}` appears to have a black screen, so a tech assumes the GPU died. However, shining a flashlight closely reveals a faint desktop image and an external HDMI monitor works fine. What is the real fault?"
            cases.append(("hardware_raid_aplus", q, lambda a: "backlight" in a.lower()))

    # 2. 500 Multi-OS Administration & Observation-Driven Command Mutation
    os_variants = [
        ("Alpine Linux 3.20", "apk"),
        ("Fedora Workstation 41", "dnf"),
        ("Arch Linux", "pacman"),
        ("FreeBSD 14.1", "pkg"),
        ("macOS Sequoia (Darwin)", "brew"),
    ]
    for i in range(500):
        h = hosts[(i + 37) % len(hosts)]
        mode = i % 5
        if mode == 0:
            os_nm, exp_pkg = os_variants[(i // 5) % len(os_variants)]
            q = f"On host `{h}`, a runbook says `apt install htop`, which fails with `command not found`, and `/etc/os-release` / `uname` shows `{os_nm}`. How do you mutate the command?"
            cases.append(("multi_os_mutation", q, lambda a, ep=exp_pkg: ep in a.lower()))
        elif mode == 1:
            q = f"Deployment on `{h}` fails with `EROFS: Read-only file system` when writing `/var/log/app.log`. Should you immediately force `mount -o remount,rw /` or what empirical checks come first?"
            cases.append(("multi_os_mutation", q, lambda a: "dmesg" in a.lower() and ("fsck" in a.lower() or "findmnt" in a.lower())))
        elif mode == 2:
            pid = rng.randint(2100, 9800)
            q = f"On Linux server `{h}`, PID `{pid}` appears as `Z` (`<defunct>` zombie process). A junior admin runs `kill -9 {pid}` repeatedly with no effect. Why, and what is the right action?"
            cases.append(("multi_os_mutation", q, lambda a: ("parent" in a.lower() or "ppid" in a.lower()) and "already" in a.lower()))
        elif mode == 3:
            free_mb = rng.randint(180, 420)
            avail_gb = rng.randint(18, 48)
            q = f"Monitoring on `{h}` shows `MemFree = {free_mb} MB` out of 64 GB, causing someone to suspect an OOM leak, yet `MemAvailable = {avail_gb} GB` in `buff/cache` with zero swap activity. Is the host out of memory?"
            cases.append(("multi_os_mutation", q, lambda a: "cache" in a.lower() and "memavailable" in a.lower()))
        else:
            if (i // 5) % 2 == 0:
                q = f"SSH login from `{h}` fails with `WARNING: UNPROTECTED PRIVATE KEY FILE! Permissions 0644 are too open`, and someone suggests `chmod 777`. What is the ethical, secure fix?"
                cases.append(("multi_os_mutation", q, lambda a: "600" in a and "700" in a))
            else:
                q = f"Running `pip install scipy` on `{h}` returns `error: externally-managed-environment (PEP 668)`. How should you mutate the workflow cleanly without breaking OS packages?"
                cases.append(("multi_os_mutation", q, lambda a: "venv" in a.lower()))

    # 3. 500 Networking Bottom-Up L1->L7, Malware 7-Step Containment & VirtualBox
    for i in range(500):
        h = hosts[(i + 91) % len(hosts)]
        mode = i % 5
        if mode == 0:
            oct3 = rng.randint(1, 253)
            oct4 = rng.randint(2, 253)
            q = f"Workstation `{h}` cannot load internal web apps and the user blames DNS, but `ipconfig` / `ip addr` shows IPv4 `169.254.{oct3}.{oct4}`. What layer actually failed?"
            cases.append(("network_security_vm", q, lambda a: "dhcp" in a.lower() and ("dora" in a.lower() or "link-local" in a.lower() or "l2" in a.lower())))
        elif mode == 1:
            q = f"Host `{h}` reports 'websites won't open — Wi-Fi internet is dead!', yet `ping 1.1.1.1` succeeds with 0% packet loss while `nslookup` fails with `NXDOMAIN`/timeout. Mutate the diagnosis."
            cases.append(("network_security_vm", q, lambda a: "dns" in a.lower() and ("healthy" in a.lower() or "resolv" in a.lower())))
        elif mode == 2:
            port = rng.choice([5432, 6379, 8080, 9090, 27017])
            q = f"Connecting from `{h}` to backend port `{port}` fails in 1 ms with `Connection refused (ECONNREFUSED)`. Does this indicate a silent firewall DROP or a closed/loopback socket?"
            cases.append(("network_security_vm", q, lambda a: "rst" in a.lower() and ("127.0.0.1" in a.lower() or "listen" in a.lower())))
        elif mode == 3:
            q = f"Workstation `{h}` has mounted company SMB shares and suddenly begins ransomware-encrypting files! Before running full antivirus scans or rebooting, what is the mandatory immediate action?"
            cases.append(("network_security_vm", q, lambda a: ("quarantine" in a.lower() or "isolat" in a.lower() or "sever" in a.lower()) and ("network" in a.lower() or "smb" in a.lower() or "lan" in a.lower())))
        else:
            if (i // 5) % 2 == 0:
                q = f"In step 3 of the 7-step CompTIA A+ troubleshooting methodology on `{h}`, your empirical test proves your initial theory is falsified (not confirmed). What must you do next?"
                cases.append(("network_security_vm", q, lambda a: "new theory" in a.lower() or "step 2" in a.lower()))
            else:
                q = f"A Linux VM on `{h}` created with `VBoxManage` has `nic1` set to `host-only` and cannot reach external package mirrors. Why, and how do you mutate its VirtualBox NIC layout?"
                cases.append(("network_security_vm", q, lambda a: "nat" in a.lower()))

    # 4. 500 Untrained Multimodal Desktop Layout & Observation-Override Action Mutations
    for i in range(500):
        h = hosts[(i + 143) % len(hosts)]
        mode = i % 5
        if mode == 0:
            q = f"On desktop `{h}`, you intend to click Continue to apply settings, but the screenshot shows a red error modal dialog blocking the center of the screen. How must your action plan mutate?"
            cases.append(("desktop_multimodal_mutation", q, lambda a: "mutated=true" in a.lower() and "click_modal_dismiss_or_ack" in a.lower()))
        elif mode == 1:
            q = f"On workstation `{h}`, you want to click the Install button, but on screen the button is greyed-out disabled alongside an unchecked prerequisite checkbox. What action should you take first?"
            cases.append(("desktop_multimodal_mutation", q, lambda a: "mutated=true" in a.lower() and "click_prerequisite_checkbox" in a.lower()))
        elif mode == 2:
            q = f"Navigating `{h}` (assumed Windows 11), you need to close the active window, but the screenshot reveals traffic-light close controls on the top-left corner and a top bar. How does your click mutate?"
            cases.append(("desktop_multimodal_mutation", q, lambda a: "mutated=true" in a.lower() and "top_left" in a.lower() and "click_window_close" in a.lower()))
        elif mode == 3:
            if (i // 5) % 2 == 0:
                q = f"On `{h}`, you expect a bottom taskbar to open the launcher, but the screenshot shows a top bar and a left dock with no bottom taskbar. Where does your action mutate?"
                cases.append(("desktop_multimodal_mutation", q, lambda a: "mutated=true" in a.lower() and "click_dock_launcher_left" in a.lower()))
            else:
                q = f"On `{h}`, you need to open a terminal to inspect logs, and the screenshot shows a Minimal Tiling Window Manager (`i3wm`) with zero panels and no taskbar. How do you mutate?"
                cases.append(("desktop_multimodal_mutation", q, lambda a: "mutated=true" in a.lower() and "keyboard_tiling_hotkey" in a.lower()))
        else:
            q = f"Booting `{h}`, you plan to click the Start menu, but the screen shows a fullscreen UEFI BIOS setup screen with no mouse pointer or GUI taskbar. How does your navigation mutate?"
            cases.append(("desktop_multimodal_mutation", q, lambda a: "mutated=true" in a.lower() and "keyboard_tui_navigation" in a.lower()))

    return cases


def run_2000_battery() -> Dict[str, Any]:
    cases = build_2000_untrained_pc_and_desktop_cases()
    latencies: List[float] = []
    by_cat: Dict[str, List[int]] = {}
    failures: List[Dict[str, Any]] = []

    def _eval_one(idx_case: Tuple[int, Tuple[str, str, Callable[[str], bool]]]):
        idx, (cat, q, checker) = idx_case
        t0 = time.perf_counter()
        res = _post_json("/api/wizard-chat/message/", {"text": q, "allow_web_lookup": False})
        dt_ms = (time.perf_counter() - t0) * 1000.0
        ans = str(res.get("answer") or "")
        ok = bool(ans and not res.get("is_hypothesis", False) and checker(ans))
        return idx, cat, q, ans, ok, dt_ms

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for idx, cat, q, ans, ok, dt_ms in pool.map(_eval_one, enumerate(cases)):
            latencies.append(dt_ms)
            pair = by_cat.setdefault(cat, [0, 0])
            pair[1] += 1
            if ok:
                pair[0] += 1
            elif len(failures) < 15:
                failures.append({"idx": idx, "category": cat, "question": q, "answer": ans[:260]})

    elapsed = time.perf_counter() - t_start
    total_ok = sum(v[0] for v in by_cat.values())
    total_n = sum(v[1] for v in by_cat.values())
    latencies.sort()
    return {
        "passed": total_ok,
        "total": total_n,
        "accuracy_pct": round(100.0 * total_ok / max(1, total_n), 2),
        "elapsed_s": round(elapsed, 2),
        "p50_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "p95_ms": round(latencies[int(0.95 * (len(latencies) - 1))], 2) if latencies else 0.0,
        "by_category": {k: f"{v[0]}/{v[1]} ({100.0*v[0]/max(1,v[1]):.2f}%)" for k, v in by_cat.items()},
        "sample_failures": failures,
    }


def main() -> None:
    for _ in range(30):
        try:
            if urllib.request.urlopen(f"{WEB_BASE}/healthz", timeout=2).status == 200:
                break
        except Exception:
            time.sleep(1.0)
    print("1/4 Verifying live Windows 11 + WSL2 Linux + VirtualBox host diagnostics...")
    s1 = verify_live_host_diagnostics()
    print(f"    -> passed={s1['passed']} checks={s1['checks']}")

    print("2/4 Running live disposable VirtualBox 7.2.22 VM observation-mutation & teardown lab...")
    s2 = verify_live_virtualbox_disposable_mutation()
    print(f"    -> passed={s2['passed']} details={s2}")

    print("3/4 Running live closed-loop X11 GUI desktop obstacle course inside WSL2 Ubuntu (Xvfb)...")
    s3 = verify_live_wsl_x11_closed_loop_navigation()
    print(f"    -> passed={s3['passed']} completed={s3.get('completed')} transport={s3.get('transport')}")

    print("4/4 Running 2,000-question untrained Multi-OS, CompTIA A+ & Multimodal Desktop Mutation battery...")
    s4 = run_2000_battery()
    print(f"    -> {s4['passed']}/{s4['total']} ({s4['accuracy_pct']}%) in {s4['elapsed_s']}s (p50={s4['p50_ms']}ms, p95={s4['p95_ms']}ms)")

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "live_host_diagnostics": s1,
        "live_virtualbox_disposable_lab": s2,
        "live_wsl_x11_closed_loop_navigation": s3,
        "untrained_2000_battery": s4,
    }
    COURSE_DIR.mkdir(parents=True, exist_ok=True)
    out_p = COURSE_DIR / "pc_aplus_and_desktop_battery_report.json"
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved full empirical report to {out_p}")
    if s4["sample_failures"]:
        print("Sample failures:", json.dumps(s4["sample_failures"][:5], indent=2))


if __name__ == "__main__":
    main()
