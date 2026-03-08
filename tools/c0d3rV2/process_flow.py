from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from context_builder import ContextBuilder
from orchestrator import Orchestrator, StepResult
from petal_system import PetalManager
from task_tree import TaskTree
from tool_registry import ToolRegistry


class ProcessFlow:
    """
    Main coordinator for c0d3r V2.

    Implements the three-step process described in the architecture outline:

      Step 1  User sends a request via the CLI input field.
      Step 2  Context is injected (local + memory + transcript + tools +
              accumulated results).
      Step 3  The Orchestrator reformulates the request in scientific /
              engineering vernacular, plans branches in a TaskTree, then
              executes each branch recursively.  Every AI call within the
              tree sees ALL tool descriptions and ALL accumulated tool
              outputs — this is the cross-tool feedback loop.
      Step 3A Each step has a self-regulatory validation loop.

    After each turn the rolling summary is updated via the model and the
    transcript is stored in long-term memory.
    """

    def __init__(
        self,
        session: Any,
        workdir: Path,
        tools: ToolRegistry,
        *,
        session_id: str | None = None,
        lt_memory: Any | None = None,
        st_memory: Any | None = None,
        lt_side_memory: Any | None = None,
        usage_tracker: Any | None = None,
        header: Any | None = None,
        tui: Any | None = None,
        petals: PetalManager | None = None,
    ) -> None:
        self.session = session
        self.workdir = workdir
        self.tools = tools
        self.session_id = session_id
        self.lt_memory = lt_memory
        self.st_memory = st_memory
        self.lt_side_memory = lt_side_memory
        self.usage = usage_tracker
        self.header = header
        self.tui = tui
        self.petals = petals or PetalManager()
        self._context: str = ""
        # If a modular STMemory is provided, it owns summary + transcript.
        # Otherwise fall back to the embedded summary bundle.
        self._st_mem = None
        if st_memory and hasattr(st_memory, "build_memory_section"):
            # This is the new modular STMemory (st_memory.py).
            self._st_mem = st_memory
            self._summary_bundle = st_memory.summary_bundle
        else:
            self._summary_bundle = self._load_summary_bundle()

    # ------------------------------------------------------------------
    # Step 1: User input
    # ------------------------------------------------------------------

    def step_1_read_input(self, prompt: str | None = None) -> str:
        """
        Read the user's request.

        If *prompt* is provided it is returned immediately.
        Otherwise we read from the TUI, stdin pipe, or interactive input.
        """
        if prompt is not None:
            return prompt.strip()

        # TUI input queue
        if self.tui:
            return self.tui.read_input(f"[{self.workdir.name}]> ")

        # Non-interactive stdin (pipe / file redirect)
        if not sys.stdin.isatty():
            return sys.stdin.read().strip()

        # Interactive REPL
        try:
            line = input(f"[{self.workdir.name}]> ")
            return (line or "").strip()
        except (EOFError, KeyboardInterrupt):
            return "/exit"

    # ------------------------------------------------------------------
    # Step 2: Context injection
    # ------------------------------------------------------------------

    def step_2_inject_context(
        self,
        request: str,
        accumulated_results: str = "",
        task_tree_summary: str = "",
    ) -> str:
        """
        Build the full context block and prepend it to the user request.

        Stores the raw context on *self._context* so the Orchestrator can
        receive system context separately from the user input.
        """
        builder = ContextBuilder(
            self.workdir,
            session_id=self.session_id,
            lt_memory=self.lt_memory,
            st_memory=self._st_mem,
            tool_descriptions=self.tools.tool_descriptions(),
            summary_bundle=self._summary_bundle,
            accumulated_results=accumulated_results,
            task_tree_summary=task_tree_summary,
        )
        self._context = builder.build()
        return f"{self._context}\n\nUser request:\n{request}"

    # ------------------------------------------------------------------
    # Step 3 + 3A: Orchestration
    # ------------------------------------------------------------------

    def step_3_orchestrate(
        self, user_request: str,
    ) -> tuple[list[StepResult], TaskTree]:
        """
        Pass the user request to the Orchestrator.

        The Orchestrator:
          1. Reformulates the request in scientific / engineering vernacular.
          2. Plans branches in a TaskTree.
          3. Executes each branch recursively with an inner agent loop.
          4. Every AI call sees ALL tool descriptions + ALL accumulated
             results (the feedback loop).

        Returns (flat results list, TaskTree).
        """
        orchestrator = Orchestrator(
            session=self.session,
            tools=self.tools,
            context=self._context,
            petals=self.petals,
        )
        return orchestrator.run(user_request)

    # ------------------------------------------------------------------
    # Post-step: memory update
    # ------------------------------------------------------------------

    def _update_memory(
        self,
        user_input: str,
        results: list[StepResult],
        tree: TaskTree,
    ) -> None:
        """Store the turn in LT memory and update the rolling summary."""
        output_text = "\n".join(r.output for r in results if r.output)

        # Persist transcript
        if self.lt_memory:
            self.lt_memory.append(
                user_input,
                output_text[:8000],
                workdir=str(self.workdir),
                model_id=getattr(
                    self.session, "get_model_id", lambda: ""
                )(),
                session_id=self.session_id or "",
            )

        # Record file paths in side-loaded memory (ST + LT)
        for entry in tree.accumulated_results():
            result = entry.get("result") or {}
            # Collect paths from file_locate results, executor outputs, etc.
            paths = result.get("paths") or []
            # Also look for file paths in stdout (executor results)
            stdout = result.get("stdout", "")
            if not paths and stdout:
                import re as _re
                # Extract plausible file paths from stdout
                found = _re.findall(
                    r'[A-Za-z]:[/\\][\w./\\-]+|/[\w./\\-]{5,}',
                    stdout,
                )
                paths = [p for p in found if "." in p.split("/")[-1].split("\\")[-1]]

            if paths:
                if self.st_memory:
                    self.st_memory.record_paths(
                        user_input[:200],
                        paths,
                        cwd=str(self.workdir),
                        project_root=str(self.workdir),
                    )
                if self.lt_side_memory and hasattr(self.lt_side_memory, "record_paths"):
                    self.lt_side_memory.record_paths(
                        user_input[:200],
                        paths,
                        cwd=str(self.workdir),
                        project_root=str(self.workdir),
                        session_id=self.session_id or "",
                    )

        # Delegate rolling summary to modular STMemory if available.
        if self._st_mem and hasattr(self._st_mem, "record_turn"):
            has_error = any(not r.success for r in results)
            has_tool_calls = any(r.step_id for r in results)
            self._st_mem.record_turn(
                user_input,
                output_text,
                has_error=has_error,
                has_tool_calls=has_tool_calls,
            )
            self._summary_bundle = self._st_mem.summary_bundle
        else:
            self._update_rolling_summary(user_input, output_text)

    def _on_session_exit(self) -> None:
        """Promote ST side-loaded memory entries to LT on session end."""
        try:
            if (
                self.st_memory
                and self.lt_side_memory
                and hasattr(self.st_memory, "hazy_hash")
                and hasattr(self.lt_side_memory, "absorb_from_session")
            ):
                promoted = self.lt_side_memory.absorb_from_session(
                    self.st_memory.hazy_hash._scope,
                )
                if promoted and self.tui:
                    self.tui.write_line(
                        f"[memory] promoted {promoted} location entries to long-term memory"
                    )
        except Exception:
            pass

    def _update_rolling_summary(self, user_input: str, output: str) -> None:
        """
        Rolling summary: summarize the new turn together with the previous
        summary.  The model produces the updated summary and the 10 most
        important points to keep track of.
        """
        prev_summary = self._summary_bundle.get("summary", "")
        prompt = (
            f"Previous summary:\n{prev_summary[:2000]}\n\n"
            f"New user input:\n{user_input[:1000]}\n\n"
            f"New system output:\n{output[:2000]}\n\n"
            "Create an updated rolling summary (max 200 words) and list the "
            "10 most important points to keep track of.\n"
            'Return JSON only: {"summary": str, "key_points": [str]}'
        )
        try:
            raw = self.session.send(prompt=prompt, stream=False)
            m = re.search(r"\{[\s\S]*\}", raw or "")
            if m:
                payload = json.loads(m.group(0))
                self._summary_bundle = {
                    "summary": str(payload.get("summary", ""))[:1000],
                    "key_points": [
                        str(p)
                        for p in (payload.get("key_points") or [])[:10]
                    ],
                }
                self._save_summary_bundle()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Summary persistence
    # ------------------------------------------------------------------

    def _load_summary_bundle(self) -> dict:
        path = self._summary_path()
        if path.exists():
            try:
                payload = json.loads(
                    path.read_text(encoding="utf-8", errors="ignore")
                )
                stored_id = str(payload.get("session_id", "")).strip()
                if (
                    self.session_id
                    and stored_id
                    and stored_id != self.session_id
                ):
                    return {"summary": "", "key_points": []}
                return {
                    "summary": str(payload.get("summary", ""))[:1000],
                    "key_points": [
                        str(p).strip()
                        for p in (payload.get("key_points") or [])[:10]
                        if str(p).strip()
                    ],
                }
            except Exception:
                pass
        return {"summary": "", "key_points": []}

    def _save_summary_bundle(self) -> None:
        path = self._summary_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **self._summary_bundle,
            "session_id": self.session_id or "",
        }
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _summary_path(self) -> Path:
        from helpers import _runtime_path
        if self.session_id:
            return _runtime_path(f"summary_{self.session_id}.json")
        return _runtime_path("summary.json")

    # ------------------------------------------------------------------
    # Main REPL loop
    # ------------------------------------------------------------------

    def run(self, initial_prompt: str | None = None) -> int:
        """
        Drive the full step 1-3 loop.

        initial_prompt: if provided, runs once (single-shot) when stdin is
                        not interactive, otherwise enters the REPL after
                        the first turn.
        Returns an OS exit code (0 = clean exit).
        """
        single_shot = (
            initial_prompt is not None and not sys.stdin.isatty()
        )

        while True:
            # Step 1 ──────────────────────────────────────────────────
            raw = self.step_1_read_input(initial_prompt)
            initial_prompt = None  # only use the seed prompt once

            if not raw or raw.lower() in {"/exit", "/quit", "exit", "quit"}:
                self._on_session_exit()
                if self.tui:
                    self.tui.stop()
                return 0

            # Show in TUI
            if self.tui:
                self.tui.write_user(raw)

            # Step 2 ──────────────────────────────────────────────────
            if self.usage:
                self.usage.set_status("planning", "building context")
            augmented = self.step_2_inject_context(raw)
            if self.usage:
                self.usage.add_input(augmented)

            # Step 3 / 3A ─────────────────────────────────────────────
            if self.usage:
                self.usage.set_status("executing", "orchestrating")
            results, tree = self.step_3_orchestrate(raw)

            # Show task tree in TUI
            if self.tui:
                self.tui.write_line(f"\n{tree.context_summary()}\n")

            # Output ──────────────────────────────────────────────────
            output_parts: list[str] = []
            for r in results:
                if r.output:
                    output_parts.append(r.output)
                    if self.tui:
                        self.tui.write_final(r.output)
                    else:
                        print(r.output)
                if not r.success and r.error:
                    err = f"[error in {r.step_id}] {r.error}"
                    if self.tui:
                        self.tui.write_line(err)
                    else:
                        print(err, file=sys.stderr)

            # Track output tokens
            if self.usage:
                self.usage.add_output("\n".join(output_parts))

            # Update memory
            self._update_memory(raw, results, tree)

            # Header refresh
            if self.usage:
                self.usage.set_status("idle")
            if self.header:
                self.header.update()

            if single_shot:
                if self.tui:
                    self.tui.stop()
                return 0
