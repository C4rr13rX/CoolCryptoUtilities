from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Tool:
    """Base interface for tools available to the Orchestrator."""

    name: str = ""
    description: str = ""

    def execute(self, params: dict) -> dict:
        """Run the tool with the given params and return a result dict."""
        raise NotImplementedError(f"{type(self).__name__}.execute not implemented")

    # Subclasses set these for structured context injection.
    use_when: str = ""
    params_schema: dict = {}

    def schema(self) -> dict:
        """Return a structured description dict for injection into model context."""
        d: dict = {"name": self.name, "description": self.description}
        if self.use_when:
            d["use_when"] = self.use_when
        if self.params_schema:
            d["params"] = self.params_schema
        return d


# ------------------------------------------------------------------
# Concrete tool wrappers
# ------------------------------------------------------------------


class ExecutorTool(Tool):
    """Run shell / PowerShell / cmd commands."""

    name = "executor"
    description = (
        "Run a terminal command (PowerShell, cmd, or bash).  Use this to "
        "execute code, install packages, run tests, inspect files, or perform "
        "any OS-level operation.  Returns stdout, stderr, and return code."
    )
    use_when = (
        "Use for: running scripts, installing packages, running tests/linters, "
        "git operations, building projects, starting/stopping services, "
        "or any OS-level task.  Prefer file_read/file_write for reading/editing "
        "source files — reserve executor for running things.  When you need a "
        "path first, call file_locate before this."
    )
    params_schema = {"command": "str — the shell command to execute"}

    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def execute(self, params: dict) -> dict:
        command = str(params.get("command", ""))
        if not command:
            return {"error": "No command provided"}
        code, stdout, stderr = self._executor.run(command)
        return {"return_code": code, "stdout": stdout, "stderr": stderr}


class WebSearchTool(Tool):
    """Ethically search the web and get AI-summarized results."""

    name = "web_search"
    description = (
        "Search the web at human pace using DuckDuckGo and return "
        "AI-summarized results.  Results feed back into context so "
        "other tools (e.g. equation_matrix) can build on them."
    )
    use_when = (
        "Use for: current documentation, API references, research papers, "
        "news, prices, package versions, or any information newer than training "
        "data.  Call this BEFORE equation_matrix or unbounded_solver when you "
        "need up-to-date source material to fill knowledge gaps.  Also use to "
        "verify assumptions before writing code."
    )
    params_schema = {"query": "str — the search query"}

    def __init__(self, web_search: Any) -> None:
        self._ws = web_search

    def execute(self, params: dict) -> dict:
        query = str(params.get("query", ""))
        if not query:
            return {"error": "No query provided"}
        return self._ws.search(query)


class MemorySearchTool(Tool):
    """Search long-term memory for relevant past interactions."""

    name = "memory_search"
    description = (
        "Search the long-term memory store for past interactions, code, "
        "and decisions matching a keyword query."
    )
    use_when = (
        "Use FIRST at the start of any task to check whether similar work was "
        "done in a prior session — avoids repeating research or re-solving "
        "already-solved problems.  Also use when the user references 'last time' "
        "or 'the version we built' or any prior work.  Call before web_search "
        "to exhaust local knowledge first."
    )
    params_schema = {"query": "str — keyword or phrase to search past sessions"}

    def __init__(self, lt_memory: Any) -> None:
        self._mem = lt_memory

    def execute(self, params: dict) -> dict:
        query = str(params.get("query", ""))
        if not query:
            return {"error": "No query provided"}
        results = self._mem.search(query, limit=10)
        return {"results": results}


class FileLocateTool(Tool):
    """Find file locations using Hazy Hash contextual approximation."""

    name = "file_locate"
    description = (
        "Find likely file and directory locations using Hazy Hash — a "
        "Kuzu-backed contextual approximation system.  Searches both this "
        "session (ST) and all prior sessions (LT).  Returns ranked candidate "
        "paths even with approximate or misspelled names."
    )
    use_when = (
        "Use BEFORE file_read, file_write, or executor whenever you do not "
        "have an exact confirmed file path.  Works with approximate names "
        "(e.g. 'No Mans Land' finds 'N0M4n5L4nD').  Always prefer this over "
        "running 'find' or 'ls -r' — it is faster and context-aware.  If "
        "detailed=True you get scores and reasons per candidate."
    )
    params_schema = {
        "query": "str — file or directory name (approximate OK)",
        "cwd": "str — current working directory (optional)",
        "project_root": "str — project root hint (optional)",
        "detailed": "bool — return scored candidates with reasons (default false)",
    }

    def __init__(self, st_memory: Any, lt_memory: Any | None = None) -> None:
        self._st = st_memory
        self._lt = lt_memory

    def execute(self, params: dict) -> dict:
        query = str(params.get("query", ""))
        cwd = str(params.get("cwd", ""))
        project_root = str(params.get("project_root", ""))
        detailed = bool(params.get("detailed", False))
        if not query:
            return {"error": "No query provided"}

        if detailed:
            st_detail = self._st.lookup_detailed(query, cwd=cwd) if self._st and hasattr(self._st, "lookup_detailed") else []
            lt_detail = self._lt.lookup_detailed(query, cwd=cwd, project_root=project_root) if self._lt and hasattr(self._lt, "lookup_detailed") else []
            # Merge and dedupe by path, keeping highest score
            seen: dict[str, dict] = {}
            for item in st_detail + lt_detail:
                path = item.get("path", "")
                if not path:
                    continue
                if path not in seen or item.get("score", 0) > seen[path].get("score", 0):
                    item["source"] = "st" if item in st_detail else "lt"
                    seen[path] = item
            candidates = sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)
            return {"candidates": candidates[:20], "query": query}

        # Simple mode: return flat path list
        st_hits = self._st.lookup(query, cwd=cwd) if self._st else []
        lt_hits = self._lt.lookup(query, cwd=cwd, project_root=project_root) if self._lt else []
        seen_paths: set[str] = set()
        merged: list[str] = []
        for p in st_hits + lt_hits:
            if p not in seen_paths:
                seen_paths.add(p)
                merged.append(p)
        return {"paths": merged}


class MatrixSearchTool(Tool):
    """Search the environmental equation matrix."""

    name = "equation_matrix"
    description = (
        "Search the environmental equation matrix — a graph of equations "
        "across physics, engineering, and mathematics with plain-English "
        "labels, domain tags, variable lists, confidence scores, and "
        "cross-equation links.  Accelerated by Kuzu graph traversal."
    )
    use_when = (
        "Use when: facing a technical/scientific/mathematical problem, "
        "verifying a formula, finding what equations govern a domain, or "
        "discovering gaps between two disciplines (gaps = where new physics "
        "or novel solutions are needed).  Call AFTER web_search has pulled "
        "source material so the matrix has been recently enriched.  For "
        "truly unknown problems use unbounded_solver instead — it drives "
        "this tool automatically."
    )
    params_schema = {
        "action": "str — one of: search | by_discipline | by_variables | find_gaps | linked",
        "query": "str — text/label/variable search (action=search)",
        "discipline": "str — domain name e.g. thermodynamics (action=by_discipline)",
        "variables": "[str] — variable symbols e.g. ['E','m','c'] (action=by_variables)",
        "discipline_a": "str — first domain (action=find_gaps)",
        "discipline_b": "str — second domain (action=find_gaps)",
        "eq_id": "int — equation id (action=linked)",
        "limit": "int — max results (default 12)",
    }

    def execute(self, params: dict) -> dict:
        action = str(params.get("action", "search")).strip()
        try:
            from matrix_helpers import (
                _matrix_search,
                _matrix_search_by_discipline,
                _matrix_search_by_variables,
                _matrix_find_gaps,
                _matrix_get_linked,
            )
        except Exception as exc:
            return {"error": str(exc), "hits": [], "missing": []}

        if action == "by_discipline":
            discipline = str(params.get("discipline", ""))
            if not discipline:
                return {"error": "No discipline provided"}
            return {"hits": _matrix_search_by_discipline(discipline)}

        if action == "by_variables":
            variables = params.get("variables") or []
            if not variables:
                return {"error": "No variables provided"}
            return {"hits": _matrix_search_by_variables(variables)}

        if action == "find_gaps":
            a = str(params.get("discipline_a", ""))
            b = str(params.get("discipline_b", ""))
            if not a or not b:
                return {"error": "Need discipline_a and discipline_b"}
            return {"gaps": _matrix_find_gaps(a, b)}

        if action == "linked":
            eq_id = params.get("eq_id")
            if eq_id is None:
                return {"error": "No eq_id provided"}
            return {"linked": _matrix_get_linked(int(eq_id))}

        # Default: text search.
        query = str(params.get("query", ""))
        if not query:
            return {"error": "No query provided"}
        return _matrix_search(query, limit=int(params.get("limit", 12)))


class FileReadTool(Tool):
    """Read a file from the workspace."""

    name = "file_read"
    description = (
        "Read the contents of a file at the given path.  Returns file "
        "contents, total line count, and the starting offset."
    )
    use_when = (
        "Use BEFORE file_write or any code edit — always read first so "
        "you have the current content.  Use for inspecting source code, "
        "configs, logs, or data files.  If you don't know the exact path, "
        "call file_locate first.  Use offset+limit to read large files in "
        "chunks rather than loading everything."
    )
    params_schema = {
        "path": "str — absolute or workdir-relative file path",
        "offset": "int — line number to start from (optional, default 0)",
        "limit": "int — number of lines to read (optional, 0 = all)",
    }

    def __init__(self, workdir: str | Path) -> None:
        self._workdir = Path(workdir)

    def execute(self, params: dict) -> dict:
        raw = str(params.get("path", ""))
        if not raw:
            return {"error": "No path provided"}
        path = Path(raw) if Path(raw).is_absolute() else self._workdir / raw
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        except FileNotFoundError:
            return {"error": f"File not found: {path}"}
        except Exception as exc:
            return {"error": str(exc)}
        offset = int(params.get("offset") or 0)
        limit = int(params.get("limit") or 0)
        chunk = lines[offset:offset + limit] if limit else lines[offset:]
        return {"content": "".join(chunk), "total_lines": len(lines), "offset": offset}


class FileWriteTool(Tool):
    """Write or patch a file in the workspace."""

    name = "file_write"
    description = (
        "Write content to a file, or apply a targeted patch by replacing "
        "old_string with new_string.  Creates parent directories automatically."
    )
    use_when = (
        "Use for all source code edits, config changes, and new file creation.  "
        "ALWAYS call file_read first so you have the exact current content before "
        "patching.  Prefer patch mode (old_string + new_string) over full rewrites "
        "for existing files — it is safer and produces a clear diff.  "
        "old_string must be unique in the file."
    )
    params_schema = {
        "path": "str — absolute or workdir-relative file path",
        "content": "str — full file content (full-write mode)",
        "old_string": "str — exact text to replace (patch mode)",
        "new_string": "str — replacement text (patch mode)",
        "create_dirs": "bool — create parent dirs if missing (default true)",
    }

    def __init__(self, workdir: str | Path) -> None:
        self._workdir = Path(workdir)

    def execute(self, params: dict) -> dict:
        raw = str(params.get("path", ""))
        if not raw:
            return {"error": "No path provided"}
        path = Path(raw) if Path(raw).is_absolute() else self._workdir / raw
        if params.get("create_dirs", True):
            path.parent.mkdir(parents=True, exist_ok=True)

        # Patch mode: replace old_string with new_string
        old = params.get("old_string")
        new = params.get("new_string")
        if old is not None and new is not None:
            if not path.exists():
                return {"error": f"File not found for patching: {path}"}
            text = path.read_text(encoding="utf-8", errors="replace")
            if old not in text:
                return {"error": f"old_string not found in {path}", "preview": text[:500]}
            patched = text.replace(str(old), str(new), 1)
            path.write_text(patched, encoding="utf-8")
            return {"status": "patched", "path": str(path)}

        # Full write mode
        content = params.get("content")
        if content is None:
            return {"error": "Provide content (full write) or old_string+new_string (patch)"}
        path.write_text(str(content), encoding="utf-8")
        return {"status": "written", "path": str(path), "bytes": len(str(content).encode())}


class UnboundedSolverTool(Tool):
    """Solve unbounded problems via the environmental equation matrix."""

    name = "unbounded_solver"
    description = (
        "Resolve problems the AI would normally declare 'impossible' or "
        "'out of scope' by recursively decomposing them into sub-questions, "
        "researching each, converting findings to equations, and propagating "
        "answers back up until the root question is answered.  No discipline "
        "caps, no cycle limits — it runs until solved."
    )
    use_when = (
        "Use when: the task involves novel physics, cross-disciplinary synthesis, "
        "or any domain where the model would normally say 'I don't know' or "
        "'that is impossible'.  Pass the original prompt AND the AI's uncertain/"
        "refusing response as ai_response — the solver treats that refusal as a "
        "map of knowledge gaps to fill.  DO NOT use for straightforward coding "
        "or file tasks — use executor/file_write for those.  This tool is for "
        "research-grade problems that require recursive equation-backed reasoning."
    )
    params_schema = {
        "prompt": "str — the original user question or problem statement",
        "ai_response": "str — the AI's uncertain or refusing response (the gap map)",
    }

    def __init__(self, solver: Any) -> None:
        self._solver = solver

    def execute(self, params: dict) -> dict:
        prompt = str(params.get("prompt", ""))
        ai_response = str(params.get("ai_response", ""))
        if not prompt:
            return {"error": "No prompt provided"}
        result = self._solver.solve(prompt, ai_response)
        return {
            "answered": result.answered,
            "answer": result.answer,
            "questions_total": result.questions_total,
            "questions_answered": result.questions_answered,
            "equations_added": result.equations_added,
            "hypotheses": [
                {"statement": h.statement, "equation": h.equation, "score": h.score}
                for h in result.hypotheses
            ],
            "anomalies": result.anomalies,
            "question_tree": result.question_tree,
            "context_block": self._solver.format_context_block(result),
        }


class MathGroundingTool(Tool):
    """Convert a request into mathematical equations and solve."""

    name = "math_grounding"
    description = (
        "Convert a natural language request into mathematical form: extract "
        "variables, unknowns, equations, and constraints; research missing "
        "constants via web search; solve with SymPy.  Returns a grounding "
        "block that scopes the problem mathematically."
    )
    use_when = (
        "Use at the START of any task involving measurement, optimization, "
        "simulation, engineering calculation, physics, finance modeling, or "
        "anything with numeric relationships.  Call this BEFORE attempting "
        "a solution — it identifies what is known, what is unknown, and "
        "what equations govern the system.  The grounding block it returns "
        "should be included in the context for all subsequent tool calls "
        "on the same task.  Not needed for pure text/code tasks with no "
        "numeric or scientific component."
    )
    params_schema = {
        "prompt": "str — the problem statement in plain English",
    }

    def __init__(self, solver: Any) -> None:
        self._solver = solver

    def execute(self, params: dict) -> dict:
        prompt = str(params.get("prompt", ""))
        if not prompt:
            return {"error": "No prompt provided"}
        record = self._solver.math_grounding(prompt)
        return {
            "grounding_block": self._solver.format_grounding_block(record),
            **record,
        }


class VMPlaygroundTool(Tool):
    """Run experiments in isolated virtual machines."""

    name = "vm_playground"
    description = (
        "Boot, control, and run experiments inside isolated VirtualBox VMs.  "
        "Test applications, run sandboxed commands, validate GUI changes, "
        "and run AI-driven experiment loops — all without touching the host."
    )
    use_when = (
        "Use when: you need to test something destructive or risky without "
        "touching the host system; when validating GUI or OS-level changes; "
        "when running untrusted code; when the task says 'test in a clean "
        "environment'; or when running a multi-step experiment that could "
        "corrupt system state.  For simple script execution use executor.  "
        "Start with action=status to see what VMs exist, then action=start "
        "or action=autopilot for a fresh OS."
    )
    params_schema = {
        "action": (
            "str — status | catalog | bootstrap | autopilot | fetch_image | "
            "create | delete | start | stop | reset | exec | guest_exec | "
            "screenshot | type | keys | mouse | wait_ready | wait_ssh | "
            "resume_or_recover | obstacle_course | run_experiment | health | "
            "tail_logs | unattended"
        ),
        "name": "str — VM name",
        "...": "action-specific keys — see description",
    }

    def __init__(self, vm_playground: Any) -> None:
        self._vm = vm_playground

    def execute(self, params: dict) -> dict:
        action = str(params.get("action", "")).strip()
        name = str(params.get("name") or params.get("vm_id") or params.get("vm") or "").strip()

        # --- Inspection ---
        if action == "status":
            return self._vm.status()
        if action == "catalog":
            return self._vm.catalog()
        if action == "latest_virtualbox":
            return self._vm.latest_virtualbox()
        if action == "tail_logs":
            return self._vm.tail_logs(lines=int(params.get("lines") or 200))
        if action == "health":
            return self._vm.health_snapshot(name, user=str(params.get("user") or "c0d3r"))
        if action == "info":
            return self._vm.vm_info(name)

        # --- Bootstrap / Update ---
        if action == "bootstrap":
            return self._vm.bootstrap(params)
        if action == "update_virtualbox":
            return self._vm.update_virtualbox(
                auto_update=bool(params.get("auto_update", True)),
            )

        # --- Image management ---
        if action == "fetch_image":
            image_id = str(params.get("image_id") or params.get("image") or "").strip()
            return self._vm.fetch_image(
                image_id,
                url=params.get("url"),
                overwrite=bool(params.get("overwrite", False)),
            )

        # --- VM lifecycle ---
        if action == "create":
            return self._vm.create(params)
        if action == "delete":
            return self._vm.delete(name, delete_files=bool(params.get("delete_files", True)))
        if action == "start":
            return self._vm.start(name, headless=bool(params.get("headless", True)))
        if action == "stop":
            return self._vm.stop(name, force=bool(params.get("force", False)))
        if action == "reset":
            return self._vm.reset(name)

        # --- Unattended install ---
        if action == "unattended":
            return self._vm.unattended_install(params)

        # --- Autopilot ---
        if action == "autopilot":
            return self._vm.autopilot(params)

        # --- Command execution ---
        if action == "exec":
            return self._vm.exec(
                name,
                str(params.get("command") or ""),
                timeout_s=float(params.get("timeout_s") or 120),
            )
        if action == "guest_exec":
            return self._vm.guest_exec(
                name,
                str(params.get("command") or ""),
                timeout_s=float(params.get("timeout_s") or 120),
            )

        # --- Observation ---
        if action == "screenshot":
            return self._vm.screenshot(name, path=params.get("path"))

        # --- Input ---
        if action == "type":
            return self._vm.type_text(name, str(params.get("text") or ""))
        if action == "keys":
            seq = params.get("sequence") or params.get("keys") or []
            return self._vm.send_keys(name, seq)
        if action == "mouse":
            return self._vm.mouse(name, params)

        # --- Wait helpers ---
        if action == "wait_port":
            return self._vm.wait_port(
                str(params.get("host") or "127.0.0.1"),
                int(params.get("port") or 22),
                timeout_s=float(params.get("timeout_s") or 120),
            )
        if action == "wait_ssh":
            return self._vm.wait_ssh(name, timeout_s=float(params.get("timeout_s") or 300))
        if action == "wait_guest_additions":
            return self._vm.wait_guest_additions(
                name, timeout_s=float(params.get("timeout_s") or 300),
            )
        if action == "wait_ready":
            return self._vm.wait_ready(name, params)

        # --- Recovery ---
        if action == "resume_or_recover":
            return self._vm.resume_or_recover(name, params)
        if action == "gui_recover":
            return self._vm.gui_recover(name, params)
        if action == "repair_guest_additions":
            return self._vm.repair_guest_additions(name, params)

        # --- Scripted sequences ---
        if action == "obstacle_course":
            steps = params.get("steps") or []
            return self._vm.obstacle_course(steps)

        # --- AI-driven experiment ---
        if action == "run_experiment":
            return self._vm.run_experiment(
                name,
                str(params.get("task") or ""),
                max_steps=int(params.get("max_steps") or 10),
            )

        return {"error": f"Unknown VM action: {action}"}


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------


class ToolRegistry:
    """
    Central registry of all tools available to the Orchestrator.

    The Orchestrator injects tool_descriptions() into every AI call so
    the model always knows which tools are available.  The model decides
    when and how to chain tools — tool results flow back through the
    accumulated context, creating feedback loops between them.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def dispatch(self, name: str, params: dict) -> dict:
        """Dispatch a tool call by name; returns a result dict."""
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"Unknown tool: {name}"}
        try:
            return tool.execute(params)
        except NotImplementedError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"{name} failed: {exc}"}

    def tool_descriptions(self) -> list[dict]:
        """Return schemas for all registered tools (for model context)."""
        return [t.schema() for t in self._tools.values()]

    def tool_names(self) -> list[str]:
        return list(self._tools.keys())
