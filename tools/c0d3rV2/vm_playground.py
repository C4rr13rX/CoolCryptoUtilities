"""
VM Playground — AI-driven experiment subsystem using VirtualBox VMs.

Wraps the ``services.vm_lab`` backend in an OOP interface that the
Orchestrator dispatches via the ToolRegistry.  Every public method returns
a JSON-ready dict so results flow into the accumulated context layer.

Capabilities ported from V1:
  - Status / catalog / logs inspection
  - VirtualBox bootstrap (auto-install) and update
  - Image fetching (Ubuntu, Kali, Parrot — auto-resolved)
  - VM create / delete / start / stop / reset
  - Unattended OS install with SSH + guest additions
  - Autopilot: one-call end-to-end VM provisioning
  - SSH and guest-control command execution
  - Screenshot, keyboard input, mouse input
  - Obstacle course (scripted multi-step sequences)
  - Wait helpers (port, SSH, guest additions, ready)
  - Health snapshot / resume / recover / GUI recovery
  - AI-driven experiment loop (model decides commands, iterates)
"""
from __future__ import annotations

import json
from typing import Any


class VMPlayground:
    """
    AI-driven subsystem for running experiments inside virtual machines.

    Delegates to ``services.vm_lab`` for all VirtualBox operations.  The
    Orchestrator never imports vm_lab directly — this class is the single
    integration point.

    The AI (via the orchestrator) can:
      - Boot a VM from scratch with ``autopilot``.
      - Run shell commands via SSH or guest-control.
      - Capture screenshots and feed them back into context.
      - Drive multi-step experiment loops with model-directed iteration.
    """

    def __init__(self, session: Any, executor: Any) -> None:
        self.session = session
        self.executor = executor
        self._lab: Any | None = None

    # ------------------------------------------------------------------
    # Lazy import of the backend
    # ------------------------------------------------------------------

    @property
    def lab(self) -> Any:
        """Import services.vm_lab on first use so startup stays fast."""
        if self._lab is None:
            from services import vm_lab
            self._lab = vm_lab
        return self._lab

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return VirtualBox install status, VM list, disk, catalog, paths."""
        return self.lab.vm_status()

    def catalog(self) -> dict:
        """Return the image catalog (Ubuntu, Kali, Parrot, etc.)."""
        return self.lab.vm_catalog()

    def latest_virtualbox(self) -> dict:
        """Fetch the latest VirtualBox version info from Oracle."""
        return self.lab.vm_latest_virtualbox()

    def tail_logs(self, lines: int = 200) -> dict:
        """Return the tail of the VM lab log file."""
        return self.lab.vm_tail_logs(lines=lines)

    def health_snapshot(self, name: str, *, user: str = "c0d3r") -> dict:
        """Capture a health snapshot of a named VM."""
        return self.lab.vm_health_snapshot(name, user=user)

    def vm_info(self, name: str) -> dict:
        """Return VBoxManage showvminfo --machinereadable as a dict."""
        return self.lab.vm_info(name)

    # ------------------------------------------------------------------
    # VirtualBox bootstrap / update
    # ------------------------------------------------------------------

    def bootstrap(self, spec: dict) -> dict:
        """
        Bootstrap VirtualBox (auto-install if missing).

        If ``spec`` contains ``image_id`` and ``vm_name``, runs the full
        autopilot sequence.  Otherwise just ensures VirtualBox is available.
        """
        image_id = str(spec.get("image_id") or spec.get("image") or "").strip()
        vm_name = str(spec.get("vm_name") or spec.get("name") or "").strip()

        if image_id and vm_name:
            return self.autopilot(spec)

        auto_install = bool(spec.get("auto_install", True))
        return self.lab.vm_bootstrap(auto_install=auto_install)

    def update_virtualbox(self, *, auto_update: bool = True) -> dict:
        """Update VirtualBox to the latest version."""
        return self.lab.vm_update_virtualbox(auto_update=auto_update)

    # ------------------------------------------------------------------
    # Image management
    # ------------------------------------------------------------------

    def fetch_image(
        self,
        image_id: str,
        *,
        url: str | None = None,
        overwrite: bool = False,
    ) -> dict:
        """
        Download an OS image by catalog id (ubuntu, kali, parrot, ...).

        URLs are auto-resolved from the catalog.  Pass ``url`` to override.
        """
        return self.lab.vm_fetch_image(image_id, url=url, overwrite=overwrite)

    # ------------------------------------------------------------------
    # VM lifecycle
    # ------------------------------------------------------------------

    def create(self, spec: dict) -> dict:
        """
        Create a new VM.

        spec keys: name, image_path/image, os_type, memory_mb, cpus,
                   vram_mb, disk_gb, efi
        """
        name = str(spec.get("name") or "").strip()
        return self.lab.vm_create(
            name,
            image_path=spec.get("image_path") or spec.get("image"),
            os_type=str(spec.get("os_type") or "Ubuntu_64"),
            memory_mb=int(spec.get("memory_mb") or 4096),
            cpus=int(spec.get("cpus") or 2),
            vram_mb=int(spec.get("vram_mb") or 64),
            disk_gb=int(spec.get("disk_gb") or 40),
            efi=bool(spec.get("efi", False)),
        )

    def delete(self, name: str, *, delete_files: bool = True) -> dict:
        """Delete a VM and optionally its disk files."""
        return self.lab.vm_delete(name, delete_files=delete_files)

    def start(self, name: str, *, headless: bool = True) -> dict:
        """Start a VM (headless by default)."""
        return self.lab.vm_start(name, headless=headless)

    def stop(self, name: str, *, force: bool = False) -> dict:
        """Stop a VM (ACPI graceful by default, force=True for poweroff)."""
        return self.lab.vm_stop(name, force=force)

    def reset(self, name: str) -> dict:
        """Hard-reset a running VM."""
        return self.lab.vm_reset(name)

    # ------------------------------------------------------------------
    # Unattended install
    # ------------------------------------------------------------------

    def unattended_install(self, spec: dict) -> dict:
        """
        Start an unattended OS install.

        Required spec keys: name, iso_path, password
        Optional: user, full_name, hostname, locale, timezone,
                  install_additions, additions_iso, post_install
        """
        name = str(spec.get("name") or "").strip()
        iso_path = str(spec.get("iso_path") or spec.get("iso") or "").strip()
        password = str(spec.get("password") or "").strip()
        if not name or not iso_path or not password:
            return {"ok": False, "error": "unattended requires name, iso_path, password"}
        return self.lab.vm_unattended_install(
            name,
            iso_path=iso_path,
            user=str(spec.get("user") or "c0d3r").strip(),
            password=password,
            full_name=str(spec.get("full_name") or spec.get("user") or "c0d3r"),
            hostname=spec.get("hostname"),
            locale=str(spec.get("locale") or "en_US"),
            timezone=str(spec.get("timezone") or "UTC"),
            install_additions=bool(spec.get("install_additions", True)),
            additions_iso=spec.get("additions_iso"),
            post_install=spec.get("post_install"),
        )

    # ------------------------------------------------------------------
    # Autopilot (end-to-end provisioning)
    # ------------------------------------------------------------------

    def autopilot(self, spec: dict) -> dict:
        """
        One-call end-to-end VM provisioning.

        Handles: disk check → VirtualBox install/update → image fetch →
        VM create → unattended install → SSH setup → guest additions →
        wait-for-ready.  Includes resume/recover if the VM already exists.

        Required spec keys: image_id, vm_name
        Optional: auto_install, auto_update, min_free_gb, ssh_port,
                  user, password, force_recreate
        """
        image_id = str(spec.get("image_id") or spec.get("image") or "").strip()
        vm_name = str(spec.get("vm_name") or spec.get("name") or "").strip()
        if not image_id or not vm_name:
            return {"ok": False, "error": "autopilot requires image_id and vm_name"}
        return self.lab.vm_autopilot(
            image_id=image_id,
            vm_name=vm_name,
            auto_install=bool(spec.get("auto_install", True)),
            auto_update=bool(spec.get("auto_update", True)),
            min_free_gb=float(spec.get("min_free_gb") or 20.0),
            ssh_port=int(spec.get("ssh_port") or 2222),
            user=str(spec.get("user") or "c0d3r"),
            password=spec.get("password"),
            force_recreate=spec.get("force_recreate"),
        )

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def exec(self, name: str, command: str, *, timeout_s: float = 120.0) -> dict:
        """Run a command inside the VM via SSH."""
        if not name or not command:
            return {"ok": False, "error": "name and command required"}
        return self.lab.vm_exec_ssh(name, command, timeout_s=timeout_s)

    def guest_exec(self, name: str, command: str, *, timeout_s: float = 120.0) -> dict:
        """Run a command via VBoxManage guestcontrol (no SSH needed)."""
        if not name or not command:
            return {"ok": False, "error": "name and command required"}
        return self.lab.vm_guest_exec(name, command, timeout_s=timeout_s)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def screenshot(self, name: str, *, path: str | None = None) -> dict:
        """Capture a PNG screenshot of the VM display."""
        if not name:
            return {"ok": False, "error": "vm name required"}
        return self.lab.vm_screenshot(name, path=path)

    # ------------------------------------------------------------------
    # Input (keyboard / mouse)
    # ------------------------------------------------------------------

    def type_text(self, name: str, text: str) -> dict:
        """Type text into the VM via keyboard string injection."""
        if not name:
            return {"ok": False, "error": "vm name required"}
        return self.lab.vm_type(name, text)

    def send_keys(self, name: str, sequence: list[str]) -> dict:
        """
        Send a sequence of key combos (e.g. ["ctrl+alt+t", "enter"]).

        Supported combos: ctrl+alt+t, ctrl+alt+f1, ctrl+alt+f3, ctrl+a,
        enter, tab, esc, up, down, left, right.
        """
        if not name:
            return {"ok": False, "error": "vm name required"}
        if not isinstance(sequence, list):
            return {"ok": False, "error": "sequence must be a list"}
        return self.lab.vm_keys(name, [str(k) for k in sequence])

    def mouse(self, name: str, spec: dict) -> dict:
        """
        Send a mouse event.

        spec keys: x, y, buttons (1=left click), screen_w, screen_h
        """
        if not name:
            return {"ok": False, "error": "vm name required"}
        return self.lab.vm_mouse(
            name,
            x=spec.get("x"),
            y=spec.get("y"),
            buttons=int(spec.get("buttons") or 0),
            screen_w=spec.get("screen_w"),
            screen_h=spec.get("screen_h"),
        )

    # ------------------------------------------------------------------
    # Wait helpers
    # ------------------------------------------------------------------

    def wait_port(self, host: str, port: int, *, timeout_s: float = 120.0) -> dict:
        """Wait for a TCP port to become reachable."""
        return self.lab.vm_wait_port(host, port, timeout_s=timeout_s)

    def wait_ssh(self, name: str, *, timeout_s: float = 300.0) -> dict:
        """Wait for SSH to be ready inside the VM."""
        return self.lab.vm_wait_ssh(name, timeout_s=timeout_s)

    def wait_guest_additions(self, name: str, *, timeout_s: float = 300.0) -> dict:
        """Wait for VirtualBox guest additions to report ready."""
        return self.lab.vm_wait_guest_additions(name, timeout_s=timeout_s)

    def wait_ready(self, name: str, spec: dict | None = None) -> dict:
        """
        Wait for a VM to be fully ready (guest additions + user + SSH).

        Optional spec keys: timeout_s, poll_s, require_user,
                            require_guest_additions
        """
        spec = spec or {}
        return self.lab.vm_wait_ready(
            name,
            timeout_s=float(spec.get("timeout_s") or 1800),
            poll_s=float(spec.get("poll_s") or 5),
            require_user=spec.get("require_user"),
            require_guest_additions=bool(spec.get("require_guest_additions", True)),
        )

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def resume_or_recover(self, name: str, spec: dict | None = None) -> dict:
        """
        Resume a stopped/paused/aborted VM with automatic recovery.

        Handles: pause resume, cold start, guest additions repair,
        GUI recovery (TTY login), SSH repair, and rebuild signaling.
        """
        spec = spec or {}
        return self.lab.vm_resume_or_recover(
            name,
            user=str(spec.get("user") or "c0d3r"),
            timeout_s=float(spec.get("timeout_s") or 1800),
            poll_s=float(spec.get("poll_s") or 5),
            recovery_retries=int(spec.get("recovery_retries") or 2),
        )

    def gui_recover(self, name: str, spec: dict | None = None) -> dict:
        """
        Attempt GUI-based recovery (TTY login + guest additions install).

        Used when SSH is unavailable.  Types credentials via keyboard
        input and reinstalls guest additions from a TTY session.
        """
        spec = spec or {}
        return self.lab.vm_gui_recover(
            name,
            user=spec.get("user"),
            password=spec.get("password"),
        )

    def repair_guest_additions(self, name: str, spec: dict | None = None) -> dict:
        """Repair guest additions (SSH-based, falls back to GUI)."""
        spec = spec or {}
        return self.lab.vm_repair_guest_additions(
            name,
            user=str(spec.get("user") or "c0d3r"),
            password=spec.get("password"),
            timeout_s=float(spec.get("timeout_s") or 900),
        )

    # ------------------------------------------------------------------
    # Obstacle course (scripted multi-step sequences)
    # ------------------------------------------------------------------

    def obstacle_course(self, steps: list[dict]) -> dict:
        """
        Execute a scripted sequence of VM actions.

        Each step: {action, name/vm, ...action-specific keys}
        Supported actions: sleep, start, stop, wait_ready, ssh,
        screenshot, type, mouse, exec.

        Stops on first failure and returns all step results.
        """
        if not isinstance(steps, list):
            return {"ok": False, "error": "steps must be a list"}
        return self.lab.vm_obstacle_course(steps)

    # ------------------------------------------------------------------
    # AI-driven experiment loop
    # ------------------------------------------------------------------

    def run_experiment(
        self,
        name: str,
        task: str,
        *,
        max_steps: int = 10,
    ) -> dict:
        """
        Drive an experiment inside the VM using the AI session.

        The model iterates in a loop:
          1. See VM state + task + prior outputs.
          2. Decide: run a command, take a screenshot, or finish.
          3. Results accumulate and feed back into the next iteration.

        Returns a dict with all step outputs and a final summary.
        """
        if not name or not task:
            return {"ok": False, "error": "name and task required"}
        if not self.session:
            return {"ok": False, "error": "no AI session available"}

        accumulated: list[dict] = []
        system = (
            "You are an AI experiment controller inside a VirtualBox VM. "
            "You have access to these actions:\n"
            '  {"action": "exec", "command": "..."} — run a shell command via SSH\n'
            '  {"action": "screenshot"} — capture the VM screen\n'
            '  {"action": "type", "text": "..."} — type text into the VM\n'
            '  {"action": "keys", "sequence": ["ctrl+alt+t", ...]} — send key combos\n'
            '  {"action": "mouse", "x": int, "y": int, "buttons": 0|1} — mouse event\n'
            '  {"action": "done", "summary": "..."} — finish the experiment\n'
            "Respond with EXACTLY ONE JSON object per step. "
            "No markdown fences, no prose outside the JSON."
        )

        for step_num in range(1, max_steps + 1):
            prompt_parts = [
                f"Task: {task}",
                f"VM: {name}",
                f"Step: {step_num}/{max_steps}",
            ]
            if accumulated:
                recent = accumulated[-5:]
                prompt_parts.append(
                    f"Prior outputs:\n{json.dumps(recent, indent=2, default=str)[:3000]}"
                )
            prompt_parts.append("Decide the next action.")

            try:
                raw = self.session.send(
                    prompt="\n".join(prompt_parts),
                    stream=False,
                    system=system,
                )
                action = self._safe_json(raw or "")
            except Exception:
                action = None

            if not action or not isinstance(action, dict):
                accumulated.append({"step": step_num, "error": "no valid action"})
                break

            action_type = str(action.get("action") or "done")
            step_result: dict = {"step": step_num, "action": action_type}

            if action_type == "exec":
                cmd = str(action.get("command") or "")
                result = self.exec(name, cmd)
                step_result.update(result)

            elif action_type == "screenshot":
                result = self.screenshot(name)
                step_result.update(result)

            elif action_type == "type":
                text = str(action.get("text") or "")
                result = self.type_text(name, text)
                step_result.update(result)

            elif action_type == "keys":
                seq = action.get("sequence") or []
                result = self.send_keys(name, seq)
                step_result.update(result)

            elif action_type == "mouse":
                result = self.mouse(name, action)
                step_result.update(result)

            elif action_type == "done":
                step_result["summary"] = str(action.get("summary") or "")
                accumulated.append(step_result)
                break

            else:
                step_result["error"] = f"unknown action: {action_type}"

            accumulated.append(step_result)

        return {
            "ok": True,
            "vm": name,
            "task": task,
            "steps": accumulated,
            "total_steps": len(accumulated),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_json(text: str) -> Any:
        """Extract JSON from model output (tolerates surrounding prose)."""
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None
