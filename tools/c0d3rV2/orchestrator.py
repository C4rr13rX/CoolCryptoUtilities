"""
Orchestrator — recursive agent loop with scientific reformulation.

Flow for every user request:

  1. Reformulate the request in scientific / engineering vernacular so the
     AI draws on its strongest training.
  2. Plan top-level branches (TaskTree).
  3. For each branch:
     a. Reformulate the branch in scientific / engineering vernacular.
     b. Send to AI with ALL tool descriptions + ALL accumulated results
        from every other branch so far (the feedback loop).
     c. AI responds with tool calls, sub-branch creation, or completion.
     d. Loop until the branch (and any sub-branches) are done.
  4. Validate overall completion.

Tools never call each other directly — the AI sees every tool's prior
output in the accumulated context and decides when to chain them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from task_tree import TaskNode, TaskTree
from tool_registry import ToolRegistry


@dataclass
class StepResult:
    """Result of executing one orchestration step."""

    step_id: str
    description: str
    output: str
    success: bool
    attempts: int = 1
    error: str = ""
    tool_outputs: list[dict] = field(default_factory=list)


class Orchestrator:
    """
    Recursive orchestration engine.

    Every AI call receives:
      - Full system context (no raw user input after planning).
      - Descriptions of ALL available tools.
      - ALL accumulated tool results from the current task tree
        (this is the cross-tool feedback loop).
      - The current branch's scientific reformulation.
    """

    MAX_PLAN_BRANCHES: int = 15
    MAX_BRANCH_DEPTH: int = 5
    MAX_AGENT_ITERATIONS: int = 12
    MAX_STEP_ATTEMPTS: int = 3

    CONTROL_PREFIX: str = (
        "You are a closed-loop systems-engineering control system. "
        "Frame every decision as a hypothesis with measurable acceptance "
        "criteria.  Return deterministic, schema-compliant JSON only. "
        "No markdown fences, no prose outside the JSON object.\n\n"
        "TOOL SELECTION RULES (apply in order):\n"
        "  1. Start every task with memory_search to check prior session work.\n"
        "  2. Use file_locate before any file_read/file_write/executor call "
        "when you do not have a confirmed exact path.\n"
        "  3. Always call file_read before file_write on existing files.\n"
        "  4. For science/math/engineering problems: call math_grounding first, "
        "then web_search for current data, then equation_matrix to find/fill gaps.\n"
        "  5. Use unbounded_solver only when the problem would normally be declared "
        "impossible or out of scope — it runs until the question is answered.\n"
        "  6. Use vm_playground for sandboxed execution, GUI testing, risky "
        "operations, or clean-environment experiments.\n"
        "  7. Use executor for running scripts, builds, tests, git, and installs. "
        "Do NOT use executor for file edits — use file_write instead.\n"
        "  8. Chain tools: the output of one tool is visible to all subsequent "
        "tool calls in the same task tree — use this feedback loop deliberately."
    )

    REFORMULATION_SYSTEM: str = (
        "You are a senior research scientist and engineer.  Restate the "
        "following task in precise scientific and engineering vernacular. "
        "Use correct domain terminology, reference relevant physical laws, "
        "mathematical frameworks, and engineering standards where applicable. "
        "The reformulation must preserve the original intent but scope it "
        "into language that draws on authoritative scientific and engineering "
        "knowledge.  Return ONLY the reformulated text, nothing else."
    )

    def __init__(
        self,
        session: Any,
        tools: ToolRegistry,
        context: str,
        *,
        petals: Any | None = None,
        max_step_attempts: int = MAX_STEP_ATTEMPTS,
    ) -> None:
        self.session = session
        self.tools = tools
        self.context = context
        self.petals = petals
        self.max_step_attempts = max_step_attempts

    # ------------------------------------------------------------------
    # Scientific reformulation
    # ------------------------------------------------------------------

    def reformulate(self, text: str) -> str:
        """
        Restate *text* in scientific / engineering vernacular.

        This scopes the request into terminology that draws on the model's
        strongest training from scientific and engineering literature.
        """
        try:
            result = self.session.send(
                prompt=text,
                stream=False,
                system=self.REFORMULATION_SYSTEM,
            )
            return (result or "").strip() or text
        except Exception:
            return text

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, request: str) -> tuple[list[StepResult], TaskTree]:
        """
        Full recursive orchestration.

        Returns (flat list of StepResults, the TaskTree).
        """
        # Run petals (dynamic pre-steps).
        if self.petals:
            self.petals.execute_all(request)
            self.petals.prune_wilted()

        # Step 0 — reformulate the user request.
        scientific_request = self.reformulate(request)

        # Build the task tree.
        tree = TaskTree(
            root_description=request,
            scientific_form=scientific_request,
        )

        # Step 1 — plan top-level branches.
        branches = self._plan_branches(scientific_request, tree)
        for branch_def in branches:
            desc = str(branch_def.get("description", ""))
            sci = self.reformulate(desc)
            tree.root.add_child(description=desc, scientific_form=sci)

        # Step 2 — execute each branch recursively.
        all_results: list[StepResult] = []
        for child in tree.root.children:
            branch_results = self._execute_branch(child, tree, depth=1)
            all_results.extend(branch_results)

        # Finalize tree.
        tree.mark_root_complete()

        # Evaluate petal effectiveness.
        if self.petals:
            feedback = [
                {"petal": r.step_id, "success": r.success} for r in all_results
            ]
            self.petals.evaluate_effectiveness(feedback)

        return all_results, tree

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan_branches(
        self, scientific_request: str, tree: TaskTree,
    ) -> list[dict]:
        """Ask AI to decompose the request into branches."""
        tool_desc = json.dumps(self.tools.tool_descriptions(), indent=2)
        accumulated = tree.accumulated_results_summary()

        system = (
            self.CONTROL_PREFIX
            + " Return ONLY a JSON object with key 'branches' (list)."
            " Each branch: {description: str, rationale: str}."
            f" Maximum {self.MAX_PLAN_BRANCHES} branches."
            f"\n\nAvailable tools:\n{tool_desc}"
        )
        prompt = (
            f"{self.context}\n\n"
            f"Task (scientific reformulation):\n{scientific_request}\n\n"
        )
        if accumulated:
            prompt += f"{accumulated}\n\n"
        prompt += (
            "Decompose this task into sequential branches.  Each branch "
            "should be a coherent sub-task.  For each branch, explain the "
            "rationale for why it is needed."
        )
        try:
            raw = self.session.send(prompt=prompt, stream=False, system=system)
        except Exception:
            return [{"description": scientific_request, "rationale": "fallback"}]

        payload = self._safe_json(raw or "")
        if isinstance(payload, dict):
            branches = payload.get("branches") or payload.get("steps") or []
            if isinstance(branches, list) and branches:
                return branches[: self.MAX_PLAN_BRANCHES]
        return [{"description": scientific_request, "rationale": "fallback"}]

    # ------------------------------------------------------------------
    # Recursive branch execution (the agent loop)
    # ------------------------------------------------------------------

    def _execute_branch(
        self,
        node: TaskNode,
        tree: TaskTree,
        depth: int = 1,
    ) -> list[StepResult]:
        """
        Execute a single branch via an inner agent loop.

        The AI is called iteratively with:
          - The branch's scientific reformulation.
          - ALL tool descriptions.
          - ALL accumulated results from the entire tree.

        The AI can:
          a) Request tool calls  → dispatched, results added to tree context.
          b) Spawn sub-branches  → child TaskNodes created and recursed into.
          c) Declare completion   → branch marked done.
        """
        node.start()
        results: list[StepResult] = []

        for iteration in range(1, self.MAX_AGENT_ITERATIONS + 1):
            # Build the per-iteration prompt with full accumulated context.
            action = self._agent_step(node, tree, depth)

            if action is None:
                # AI call failed — mark done with what we have.
                node.complete()
                break

            action_type = action.get("action", "complete")

            # --- Tool calls -------------------------------------------
            if action_type == "tool_calls":
                calls = action.get("tool_calls") or []
                step_result = self._dispatch_tool_calls(
                    node, calls, tree, attempt=iteration,
                )
                results.append(step_result)
                if not step_result.success:
                    # Try to fix via the validation loop (Step 3A).
                    fixed = self._attempt_fix(node, step_result, tree)
                    if fixed:
                        results.append(fixed)

            # --- Sub-branches -----------------------------------------
            elif action_type == "sub_branches":
                if depth >= self.MAX_BRANCH_DEPTH:
                    node.complete()
                    break
                sub_defs = action.get("sub_branches") or []
                for sub_def in sub_defs[: self.MAX_PLAN_BRANCHES]:
                    desc = str(sub_def.get("description", ""))
                    sci = self.reformulate(desc)
                    child = node.add_child(description=desc, scientific_form=sci)
                    child_results = self._execute_branch(child, tree, depth + 1)
                    results.extend(child_results)
                # After all sub-branches, check if parent is done.
                if node.all_children_done:
                    node.complete()
                break

            # --- Completion -------------------------------------------
            elif action_type == "complete":
                output = action.get("output", "")
                if output:
                    node.add_tool_output("synthesis", {"summary": output})
                node.complete()
                results.append(StepResult(
                    step_id=node.id,
                    description=node.scientific_form or node.description,
                    output=output,
                    success=True,
                    attempts=iteration,
                ))
                break

        # Safety net: if we exhausted iterations, mark done.
        if not node.is_done:
            node.complete()

        return results

    def _agent_step(
        self,
        node: TaskNode,
        tree: TaskTree,
        depth: int,
    ) -> dict | None:
        """
        Single iteration of the inner agent loop.

        Returns a dict with:
          {"action": "tool_calls", "tool_calls": [...]}
          {"action": "sub_branches", "sub_branches": [...]}
          {"action": "complete", "output": "..."}
        """
        tool_desc = json.dumps(self.tools.tool_descriptions(), indent=2)
        accumulated = tree.accumulated_results_summary()
        tree_summary = tree.context_summary()

        system = (
            self.CONTROL_PREFIX
            + "\n\nYou are executing one branch of a task tree.  "
            "You have access to the tools listed below, each with a Scope "
            "field that tells you WHEN to use it.  Read the Scope before "
            "choosing a tool.  You can see every result that every tool "
            "has produced so far across all branches — use this to avoid "
            "redundant work and build on prior discoveries.\n\n"
            "Respond with EXACTLY ONE of these JSON shapes:\n"
            '1. {"action": "tool_calls", "tool_calls": [{"tool": "<name>", "params": {...}}, ...]}\n'
            '   — Call one or more tools.  Use each tool\'s Params schema exactly.\n'
            '   — You may batch multiple independent tool calls in one response.\n'
            '   — Results feed back into context for the next iteration.\n'
            '2. {"action": "sub_branches", "sub_branches": [{"description": "<task>"}]}\n'
            '   — Decompose this branch when it contains multiple distinct sub-tasks.\n'
            '3. {"action": "complete", "output": "<summary of what was accomplished>"}\n'
            '   — Only when this branch is fully resolved with evidence.\n\n'
            "NEVER respond with prose.  NEVER use markdown fences.  "
            "The JSON must be parseable with json.loads().\n\n"
            f"Available tools:\n{tool_desc}"
        )

        prompt_parts = [
            f"System context:\n{self.context}",
            f"\nCurrent task tree:\n{tree_summary}",
        ]
        if accumulated:
            prompt_parts.append(f"\n{accumulated}")
        prompt_parts.append(
            f"\nCurrent branch [{node.id}]:\n"
            f"  Description: {node.description}\n"
            f"  Scientific form: {node.scientific_form}\n"
            f"  Status: {node.status}\n"
            f"  Depth: {depth}/{self.MAX_BRANCH_DEPTH}\n"
            f"  Prior tool outputs on this branch: {len(node.tool_outputs)}\n"
        )
        if node.tool_outputs:
            # Show this branch's recent outputs so the AI knows what it
            # already tried.
            recent = node.tool_outputs[-5:]
            recent_str = json.dumps(recent, indent=2, default=str)[:3000]
            prompt_parts.append(f"\nRecent outputs on this branch:\n{recent_str}")

        prompt_parts.append(
            "\nDecide the next action for this branch.  "
            "If you need information, call a tool.  "
            "If the task is too complex, break it into sub-branches.  "
            "If the branch is resolved, complete it with a summary."
        )

        prompt = "\n".join(prompt_parts)

        try:
            raw = self.session.send(prompt=prompt, stream=False, system=system)
            return self._safe_json(raw or "")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool_calls(
        self,
        node: TaskNode,
        calls: list[dict],
        tree: TaskTree,
        attempt: int = 1,
    ) -> StepResult:
        """Dispatch tool calls, record results on the node and tree."""
        tool_outputs: list[dict] = []
        errors: list[str] = []
        stdout_parts: list[str] = []

        for call in calls:
            tool_name = str(call.get("tool", ""))
            params = call.get("params") or {}
            if not tool_name:
                continue
            result = self.tools.dispatch(tool_name, params)
            tool_outputs.append({"tool": tool_name, "result": result})

            # Record on the node so accumulated context grows.
            node.add_tool_output(tool_name, result)

            if result.get("error"):
                errors.append(f"{tool_name}: {result['error']}")

            # Collect readable output.
            for key in ("stdout", "summary", "result"):
                val = result.get(key)
                if val and isinstance(val, str) and val.strip():
                    stdout_parts.append(val.strip())
            for key in ("paths", "results", "hits"):
                val = result.get(key)
                if val and isinstance(val, (list, dict)):
                    stdout_parts.append(
                        json.dumps(val, indent=2, default=str)[:2000]
                    )

        return StepResult(
            step_id=node.id,
            description=node.scientific_form or node.description,
            output="\n".join(stdout_parts),
            success=not errors,
            attempts=attempt,
            error="\n".join(errors),
            tool_outputs=tool_outputs,
        )

    # ------------------------------------------------------------------
    # Step 3A: Validation / fix loop
    # ------------------------------------------------------------------

    def _attempt_fix(
        self,
        node: TaskNode,
        failed_result: StepResult,
        tree: TaskTree,
    ) -> StepResult | None:
        """
        Ask AI to diagnose the failure and provide fix tool calls.
        Returns a new StepResult if the fix succeeded, else None.
        """
        tool_desc = json.dumps(self.tools.tool_descriptions(), indent=2)
        accumulated = tree.accumulated_results_summary()

        system = (
            self.CONTROL_PREFIX
            + " A tool call failed.  Diagnose the issue and provide "
            "corrective tool calls.\n"
            'Return ONLY JSON: {"fix_tool_calls": [{"tool": str, "params": dict}], "reasoning": str}'
            f"\n\nAvailable tools:\n{tool_desc}"
        )
        prompt = (
            f"Branch: {node.scientific_form or node.description}\n\n"
            f"Failed output:\n{failed_result.output[:2000]}\n\n"
            f"Errors:\n{failed_result.error[:1000]}\n\n"
        )
        if accumulated:
            prompt += f"{accumulated}\n\n"
        prompt += "Provide fix_tool_calls to resolve the issue."

        try:
            raw = self.session.send(prompt=prompt, stream=False, system=system)
            payload = self._safe_json(raw or "")
            if isinstance(payload, dict):
                fix_calls = payload.get("fix_tool_calls") or []
                if isinstance(fix_calls, list) and fix_calls:
                    return self._dispatch_tool_calls(
                        node, fix_calls, tree, attempt=2,
                    )
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_json(text: str) -> Any:
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None
