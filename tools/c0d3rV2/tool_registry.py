from __future__ import annotations

import json
from typing import Any


class Tool:
    """Base interface for tools available to the Orchestrator."""

    name: str = ""
    description: str = ""

    def execute(self, params: dict) -> dict:
        """Run the tool with the given params and return a result dict."""
        raise NotImplementedError(f"{type(self).__name__}.execute not implemented")

    def schema(self) -> dict:
        """Return a description dict for injection into model context."""
        return {"name": self.name, "description": self.description}


# ------------------------------------------------------------------
# Concrete tool wrappers
# ------------------------------------------------------------------


class ExecutorTool(Tool):
    """Run shell / PowerShell / cmd commands."""

    name = "executor"
    description = (
        "Run a terminal command (PowerShell, cmd, or bash).  Use this to "
        "execute code, install packages, run tests, inspect files, or perform "
        "any OS-level operation.  Returns stdout, stderr, and return code.  "
        "Params: {command: str}"
    )

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
        "AI-summarized results.  Use this when you need current information, "
        "documentation, research papers, or authoritative references that "
        "go beyond your training data.  Results feed back into context so "
        "other tools (e.g. equation_matrix) can build on them.  "
        "Params: {query: str}"
    )

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
        "and decisions matching a keyword query.  Use this to recall prior "
        "context, avoid repeating work, or build on previous findings.  "
        "Params: {query: str}"
    )

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
        "Kuzu-backed contextual approximation system inspired by how "
        "human brains recall locations.  Instead of scanning the entire "
        "file system, this tool narrows the search using context: which "
        "machine, which user, which project area, past session history, "
        "and query similarity.  It returns ranked candidate paths.  "
        "Use this FIRST before running expensive file searches or directory "
        "traversals.  If a project or file was accessed in any past session, "
        "this tool can approximate where it is even without an exact name "
        "match (e.g. finding 'N0M4n5L4nD' when searching for 'No Mans Land').  "
        "The tool searches both session-scoped (ST) memory for files found "
        "this session, and cross-session (LT) memory for historical patterns.  "
        "You make the final disambiguation — the tool provides candidates "
        "with scores and reasons.  "
        "Params: {query: str, cwd: str, project_root: str, detailed: bool}"
    )

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
        "Search the environmental equation matrix for relevant equations "
        "across physics, engineering, and mathematics domains.  Equations "
        "have plain-English labels, domain tags, variable lists, confidence "
        "scores, and connections to other equations.  Use this when facing "
        "unbounded problems: research equations that govern the problem "
        "domain, find which equations integrate and which don't (gaps "
        "represent where new physics is needed).  "
        "Accelerated by Kuzu graph traversal when available.\n"
        "Actions:\n"
        "  search — search by text, label, or variable (query: str)\n"
        "  by_discipline — list equations in a discipline (discipline: str)\n"
        "  by_variables — find equations using specific variables (variables: [str])\n"
        "  find_gaps — find unbridged gaps between two disciplines "
        "(discipline_a: str, discipline_b: str)\n"
        "  linked — get all equations linked to one (eq_id: int)\n"
        "Params: {action: str, query: str, discipline: str, ...}"
    )

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
        "Read the contents of a file at the given path.  Use this to inspect "
        "source code, config files, logs, or any text file before making changes.  "
        "Returns the file contents and line count.  "
        "Params: {path: str, offset?: int, limit?: int}"
    )

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
        "Write content to a file in the workspace.  Use this to create new files, "
        "overwrite existing files, or apply a targeted patch (replace old_string "
        "with new_string).  All paths are relative to the project workdir unless "
        "absolute.  For patches, provide old_string + new_string.  For full writes, "
        "provide content only.  "
        "Params: {path: str, content?: str, old_string?: str, new_string?: str, "
        "create_dirs?: bool}"
    )

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
        "Resolve problems the AI declares 'impossible' or 'out of scope' "
        "by recursively decomposing them into answerable sub-questions "
        "and filling the environmental equation matrix.\n"
        "The solver has NO predefined disciplines, NO cycle caps, NO "
        "equation limits.  The problem defines the disciplines.  It runs "
        "until the original question is answered — even if that means "
        "answering questions to questions to questions, then propagating "
        "answers back up.\n"
        "Process:\n"
        "  1. Can we answer the question? If yes, done.\n"
        "  2. If no: what sub-questions must be answered first?\n"
        "  3. Research each, convert to equations, ingest into matrix.\n"
        "  4. Recurse into each sub-question.\n"
        "  5. When all children answered, attempt parent again.\n"
        "  6. If still stuck, generate hypotheses to bridge the gap.\n"
        "  7. Continue until root question is answered.\n"
        "Use this for unbounded problems: FTL travel, novel physics, "
        "cross-disciplinary synthesis.  It does NOT give up.\n"
        "Params: {prompt: str, ai_response: str}"
    )

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
        "Convert a natural language request into mathematical form: "
        "extract variables, unknowns, equations, and constraints.  "
        "Research missing constants via web search, then solve with SymPy.  "
        "Returns a grounding block with equations, solutions, and gap "
        "fill steps that can be injected into subsequent requests.  "
        "Use this to mathematically scope any problem before attempting "
        "a solution.\n"
        "Params: {prompt: str}"
    )

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
        "Boot, control, and run experiments inside isolated VirtualBox "
        "virtual machines.  Use this to test applications, run commands in "
        "a sandboxed environment, test GUI redesigns on free operating "
        "systems (Ubuntu, Kali, Parrot), and validate changes before "
        "applying them to the host.  "
        "Actions (pass as 'action' param):\n"
        "  status — VirtualBox install info, VM list, disk, catalog\n"
        "  catalog — list available OS images\n"
        "  bootstrap — install VirtualBox if missing; or with image_id + vm_name runs full autopilot\n"
        "  autopilot — end-to-end: fetch image → create VM → unattended install → SSH → ready\n"
        "  fetch_image — download an OS ISO (image_id, url?)\n"
        "  create — create a VM (name, image_path, os_type, memory_mb, cpus, disk_gb)\n"
        "  delete — delete a VM (name)\n"
        "  start — start a VM (name, headless?)\n"
        "  stop — stop a VM (name, force?)\n"
        "  reset — hard-reset a VM (name)\n"
        "  exec — run a command via SSH (name, command, timeout_s?)\n"
        "  guest_exec — run via VBoxManage guestcontrol, no SSH needed (name, command)\n"
        "  screenshot — capture VM display as PNG (name, path?)\n"
        "  type — type text into VM keyboard (name, text)\n"
        "  keys — send key combos like ctrl+alt+t (name, sequence list)\n"
        "  mouse — send mouse event (name, x, y, buttons, screen_w?, screen_h?)\n"
        "  wait_ready — wait for VM to be fully ready (name, timeout_s?, require_user?)\n"
        "  wait_ssh — wait for SSH access (name, timeout_s?)\n"
        "  resume_or_recover — resume stopped/aborted VM with auto-recovery (name)\n"
        "  obstacle_course — run a scripted multi-step sequence (steps list)\n"
        "  run_experiment — AI-driven experiment loop (name, task, max_steps?)\n"
        "  health — health snapshot (name)\n"
        "  tail_logs — VM lab log tail (lines?)\n"
        "  unattended — start unattended OS install (name, iso_path, password, ...)\n"
        "Params: {action: str, name: str, ...action-specific keys}"
    )

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
