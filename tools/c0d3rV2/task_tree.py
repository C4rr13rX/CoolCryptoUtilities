"""
Tree-based task tracking for recursive orchestration.

The user's request becomes the root node.  The Orchestrator reformulates it
in scientific/engineering vernacular, plans top-level branches, and each
branch can itself spawn sub-branches as the AI decides more work is needed.

Every node accumulates tool outputs so that the full history of what every
tool discovered is available as context for subsequent AI calls — this is
the mechanism by which tools "send feedback loops to each other."
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskNode:
    """A single node in the task tree (branch or leaf)."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    scientific_form: str = ""          # reformulated in scientific vernacular
    status: str = "pending"            # pending | in_progress | completed | failed
    children: list[TaskNode] = field(default_factory=list)
    tool_outputs: list[dict] = field(default_factory=list)
    MAX_TOOL_OUTPUTS: int = field(default=50, repr=False)
    parent_id: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    error: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.status = "in_progress"

    def complete(self) -> None:
        self.status = "completed"
        self.completed_at = time.time()

    def fail(self, error: str = "") -> None:
        self.status = "failed"
        self.error = error
        self.completed_at = time.time()

    @property
    def is_done(self) -> bool:
        return self.status in {"completed", "failed"}

    @property
    def all_children_done(self) -> bool:
        return all(c.is_done for c in self.children)

    # ------------------------------------------------------------------
    # Tree operations
    # ------------------------------------------------------------------

    def add_child(
        self,
        description: str,
        scientific_form: str = "",
    ) -> TaskNode:
        child = TaskNode(
            description=description,
            scientific_form=scientific_form,
            parent_id=self.id,
        )
        self.children.append(child)
        return child

    def add_tool_output(self, tool_name: str, result: dict) -> None:
        self.tool_outputs.append({
            "tool": tool_name,
            "result": result,
            "ts": time.time(),
        })
        if len(self.tool_outputs) > self.MAX_TOOL_OUTPUTS:
            self.tool_outputs = self.tool_outputs[-self.MAX_TOOL_OUTPUTS:]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "scientific_form": self.scientific_form,
            "status": self.status,
            "error": self.error,
            "tool_outputs_count": len(self.tool_outputs),
            "children": [c.to_dict() for c in self.children],
        }

    def summary(self, depth: int = 0) -> str:
        """Human-readable tree summary for context injection."""
        indent = "  " * depth
        icon = {"pending": "○", "in_progress": "►", "completed": "✓", "failed": "✗"}.get(self.status, "?")
        label = self.scientific_form or self.description
        line = f"{indent}{icon} [{self.id}] {label}"
        parts = [line]
        for child in self.children:
            parts.append(child.summary(depth + 1))
        return "\n".join(parts)


class TaskTree:
    """
    Manages the full task tree for a single user request.

    Provides:
      - Accumulated results across all nodes (the cross-tool feedback loop).
      - A structured summary for context injection.
      - Branch-level completion tracking.
    """

    def __init__(self, root_description: str, scientific_form: str = "") -> None:
        self.root = TaskNode(
            description=root_description,
            scientific_form=scientific_form,
        )
        self.root.start()

    # ------------------------------------------------------------------
    # Accumulated context (the feedback loop)
    # ------------------------------------------------------------------

    def accumulated_results(self) -> list[dict]:
        """
        Collect every tool output from every node in the tree.

        This is what gets injected into context so that each tool call
        can see what every other tool has already discovered.
        """
        results: list[dict] = []
        self._collect_outputs(self.root, results)
        return results

    def _collect_outputs(self, node: TaskNode, out: list[dict]) -> None:
        for entry in node.tool_outputs:
            out.append({
                "branch": node.id,
                "branch_desc": node.scientific_form or node.description,
                **entry,
            })
        for child in node.children:
            self._collect_outputs(child, out)

    # ------------------------------------------------------------------
    # Context summary
    # ------------------------------------------------------------------

    def context_summary(self) -> str:
        """Full tree summary for injection into AI context."""
        return self.root.summary()

    def accumulated_results_summary(self, max_chars: int = 6000) -> str:
        """
        Condensed text block of all tool outputs for context injection.
        Truncates per-entry to stay within budget.
        """
        entries = self.accumulated_results()
        if not entries:
            return ""
        lines = ["[Accumulated Tool Results]"]
        char_count = 0
        per_entry_limit = max(200, max_chars // max(len(entries), 1))
        for entry in entries:
            tool = entry.get("tool", "?")
            branch = entry.get("branch_desc", entry.get("branch", ""))
            result = entry.get("result", {})
            # Compact serialisation
            result_str = json.dumps(result, default=str)
            if len(result_str) > per_entry_limit:
                result_str = result_str[:per_entry_limit] + "..."
            line = f"- [{tool}] (branch: {branch[:80]}): {result_str}"
            if char_count + len(line) > max_chars:
                lines.append(f"  ... ({len(entries) - len(lines) + 1} more entries truncated)")
                break
            lines.append(line)
            char_count += len(line)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def find(self, node_id: str) -> TaskNode | None:
        return self._find(self.root, node_id)

    def _find(self, node: TaskNode, node_id: str) -> TaskNode | None:
        if node.id == node_id:
            return node
        for child in node.children:
            found = self._find(child, node_id)
            if found:
                return found
        return None

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def pending_branches(self) -> list[TaskNode]:
        """Return all nodes that still need work."""
        pending: list[TaskNode] = []
        self._collect_pending(self.root, pending)
        return pending

    def _collect_pending(self, node: TaskNode, out: list[TaskNode]) -> None:
        if not node.is_done and not node.children:
            out.append(node)
        for child in node.children:
            self._collect_pending(child, out)

    @property
    def is_complete(self) -> bool:
        return self.root.is_done

    def mark_root_complete(self) -> None:
        if self.root.all_children_done:
            self.root.complete()

    def __repr__(self) -> str:
        total = self._count(self.root)
        done = self._count_done(self.root)
        return f"TaskTree(total={total}, done={done})"

    def _count(self, node: TaskNode) -> int:
        return 1 + sum(self._count(c) for c in node.children)

    def _count_done(self, node: TaskNode) -> int:
        n = 1 if node.is_done else 0
        return n + sum(self._count_done(c) for c in node.children)
