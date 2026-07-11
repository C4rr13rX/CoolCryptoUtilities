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
import os
import re
from dataclasses import dataclass, field
from typing import Any

from task_tree import TaskNode, TaskTree
from tool_registry import ToolRegistry


class ModelCallBudgetExceeded(RuntimeError):
    pass


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


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
    MAX_TOTAL_AGENT_ITERATIONS: int = 64
    MAX_TOTAL_MODEL_CALLS: int = 64
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
        "then scientific_method when the claim is uncertain or experimentally "
        "testable, then web_search/equation_matrix to find/fill gaps.\n"
        "  5. Use unbounded_solver only when the problem would normally be declared "
        "impossible or out of scope — it runs until the question is answered.\n"
        "  6. Use vm_playground for sandboxed execution, GUI testing, risky "
        "operations, or clean-environment experiments.\n"
        "  7. Use executor for running scripts, builds, tests, git, and installs. "
        "Do NOT use executor for file edits — use file_write instead.\n"
        "  8. In the Django web app, use c0d3r_native_os when the user asks "
        "for full-PC project work, OS commands, or file operations outside the "
        "current workspace; it is the authenticated native Windows service bridge.\n"
        "  9. For React/TypeScript SPA/PWA vertical-slice builds, use "
        "react_pwa_scaffold first; then validate or customize with native OS commands.\n"
        "  10. For natural-language sandboxed file organization, sorting, "
        "renaming, copying, moving, manifests, flattening, or deduplication, "
        "use sandbox_file_ops instead of handwritten shell commands.\n"
        "  11. For hardware-backed concepts that need software-first virtual "
        "components, virtual drivers, device simulation, radio/sensor/actuator "
        "abstractions, or a hardware replacement path, use "
        "virtual_hardware_sim_scaffold first.\n"
        "  12. For directory creation use directory_ensure, not executor. For "
        "multi-framework or multi-language workspace setup use workspace_scaffold, "
        "not handwritten shell loops. Use compact presets such as "
        "preset=major_app_frameworks when they match the task.\n"
        "  13. For turning common scaffolds into installed/runnable environments, "
        "use environment_bootstrap presets before handwritten executor sequences.\n"
        "  14. If you are stuck, unsure, or facing a likely hallucination in "
        "science/engineering/math, call scientific_method instead of guessing.\n"
        "  15. Chain tools: the output of one tool is visible to all subsequent "
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

    SCRUTINY_SYSTEM: str = (
        "You are C0d3rV2's first-call scrutiny gate. Decide the minimum work "
        "needed and return strict JSON only. For greetings, acknowledgements, "
        "small talk, or questions answerable immediately, return "
        '{"decision":"direct","answer":"<complete user-facing answer>"}. '
        "For work requiring tools, research, files, validation, or multiple "
        "steps, return {\"decision\":\"execute\",\"scientific_request\":\"<precise "
        "request>\",\"branches\":[...]}. Each branch contains id, description, "
        "rationale, dependencies, constraints, acceptance_criteria, and "
        "recovery_policy. Do not use decision=direct merely to acknowledge the "
        "request. The direct answer is final and will be shown verbatim."
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
        self._atomic_workday = "unattended atomic workday job" in context.lower()
        self._corrective_retry = "corrective retry" in context.lower()
        self._total_agent_iterations = 0
        self._total_model_calls = 0
        self._max_total_agent_iterations = max(
            8,
            int(os.getenv("C0D3R_MAX_TOTAL_ITERATIONS", str(self.MAX_TOTAL_AGENT_ITERATIONS))),
        )
        self._max_total_model_calls = max(
            4,
            int(os.getenv("C0D3R_MAX_MODEL_CALLS", str(self.MAX_TOTAL_MODEL_CALLS))),
        )

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
            result = self._send(
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
        self._total_agent_iterations = 0
        self._total_model_calls = 0
        # Run petals (dynamic pre-steps).
        if self.petals:
            self.petals.execute_all(request)
            self.petals.prune_wilted()

        corrective_retry = "corrective retry" in request.lower() or "corrective retry" in self.context.lower()

        scrutiny: dict[str, Any] = {}
        if not corrective_retry:
            scrutiny = self._scrutinize(request)
            if scrutiny.get("decision") == "direct":
                answer = str(scrutiny.get("answer") or "").strip()
                root = TaskTree(root_description=request, scientific_form=request)
                root.mark_root_complete()
                return [StepResult(
                    step_id=root.root.id, description=request, output=answer,
                    success=True, attempts=1,
                )], root

        # The scrutiny call combines classification, scientific scoping, and
        # planning. Corrective retries already carry their exact validator plan.
        scientific_request = (
            request if corrective_retry
            else str(scrutiny.get("scientific_request") or request).strip()
        )

        # Build the task tree.
        tree = TaskTree(
            root_description=request,
            scientific_form=scientific_request,
        )

        # Step 1 — plan top-level branches. A failed external validation is
        # already a precise plan: inspect, repair, and rerun that validator.
        branches = ([{
            "id": "correct-validation",
            "description": request,
            "rationale": "Repair the existing artifact against exact external validation evidence.",
            "dependencies": [],
            "constraints": [
                "Preserve working behavior", "Do not redesign or restart the project",
                "Use the supplied validation error as ground truth",
            ],
            "acceptance_criteria": ["The supplied acceptance command exits zero"],
            "recovery_policy": "Inspect the failing file or test, make the smallest evidence-backed repair, and rerun validation.",
        }] if corrective_retry else list(scrutiny.get("branches") or []) or [{
            "id": "execute-request", "description": scientific_request,
            "rationale": "The scrutiny gate determined that execution is required.",
        }])
        for index, branch_def in enumerate(branches, start=1):
            desc = str(branch_def.get("description", ""))
            sci = desc
            alias = str(branch_def.get("id") or f"step-{index}")
            tree.root.add_child(
                description=f"[{alias}] {desc}",
                scientific_form=sci,
                rationale=str(branch_def.get("rationale") or ""),
                dependencies=_as_str_list(branch_def.get("dependencies")),
                constraints=_as_str_list(branch_def.get("constraints")),
                acceptance_criteria=_as_str_list(branch_def.get("acceptance_criteria")),
                recovery_policy=str(branch_def.get("recovery_policy") or "Diverge only to resolve evidence-backed blockers; then return to this step's acceptance criteria."),
            )

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

    @staticmethod
    def _deterministic_direct_answer(request: str) -> str:
        """Deprecated: direct answers must come from the selected model."""
        return ""

    @staticmethod
    def _is_social_turn(request: str) -> bool:
        """Return true for prompts where a conversational reply is the task."""
        text = " ".join(str(request or "").strip().lower().split())
        text = re.sub(r"[.!?,;:]+$", "", text).strip()
        social_patterns = (
            r"(hi|hello|hey|hiya|howdy|good (morning|afternoon|evening))(\b.*)?",
            r"(how are you|how are things|what's up|whats up|how's it going|hows it going)(\b.*)?",
            r"(thanks|thank you|thank you very much|thx)(\b.*)?",
            r"(bye|goodbye|see you|see you later)(\b.*)?",
        )
        return any(re.fullmatch(pattern, text) for pattern in social_patterns)

    def _scrutinize(self, request: str) -> dict[str, Any]:
        """Use one call to either answer directly or produce the execution plan."""
        prompt = f"{self.context}\n\nUser request:\n{request}"
        try:
            raw = self._send(prompt=prompt, stream=False, system=self.SCRUTINY_SYSTEM)
        except Exception:
            return {"decision": "execute", "scientific_request": request, "branches": []}
        payload = self._safe_json(raw or "")
        if isinstance(payload, dict):
            decision = str(payload.get("decision") or "").lower().strip()
            answer = str(payload.get("answer") or "").strip()
            if (
                decision == "direct"
                and not self._requires_tool_execution(request)
                and not self._non_answer_validation_error(
                TaskNode(description=request), answer,
                )
            ):
                return {"decision": "direct", "answer": answer}
            if decision == "execute":
                branches = payload.get("branches")
                return {
                    "decision": "execute",
                    "scientific_request": str(payload.get("scientific_request") or request),
                    "branches": branches if isinstance(branches, list) else [],
                }
        raw_text = str(raw or "").strip()
        if (
            raw_text
            and len(raw_text) <= 2000
            and not raw_text.startswith(("{", "["))
            and not self._requires_tool_execution(request)
        ):
            # Some free models ignore the JSON envelope but still provide a
            # usable direct answer. Preserve that one-call result rather than
            # launching an expensive execution loop.
            if not self._non_answer_validation_error(TaskNode(description=request), raw_text):
                return {"decision": "direct", "answer": raw_text}
        # A malformed gate response must not trigger another classification
        # call. Execute the original request conservatively.
        return {"decision": "execute", "scientific_request": request, "branches": []}

    @staticmethod
    def _requires_tool_execution(request: str) -> bool:
        """Return true when prose-only direct answers are not acceptable."""
        text = " ".join(str(request or "").lower().split())
        markers = (
            "build", "create", "implement", "write", "edit", "patch", "fix",
            "repair", "install", "run", "execute", "test", "validate",
            "benchmark", "scaffold", "set up", "setup", "organize",
            "delete", "move", "copy", "rename", "file", "folder", "directory",
            "project", "app", "spa", "pwa", "website", "workspace",
        )
        return any(marker in text for marker in markers)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan_branches(
        self, scientific_request: str, tree: TaskTree,
    ) -> list[dict]:
        """Ask AI to decompose the request into branches."""
        accumulated = tree.accumulated_results_summary()
        context_lower = self.context.lower()
        corrective = "corrective retry" in context_lower
        atomic = self._atomic_workday
        max_branches = 3 if corrective else 4 if atomic else self.MAX_PLAN_BRANCHES

        system = (
            "You are a systems-engineering planner. Do not call tools and do not return an action object."
            " Return ONLY a JSON object with key 'branches' (list)."
            " Each branch must be: {id: str, description: str, rationale: str, "
            "dependencies: [prior branch ids], constraints: [str], "
            "acceptance_criteria: [measurable str], recovery_policy: str}."
            " Every acceptance criterion must state a test or measurement with a numeric threshold, "
            "pass/fail condition, p95 bound, percentage, zero-error condition, or minimum count."
            " Example criterion form: 'Load test passes with p95 latency <= 300 ms, zero critical "
            "errors, and >= 99.9% availability across 10,000 simulated users.' A document or report "
            "by itself is not measurable acceptance evidence."
            " Every recovery_policy must explicitly state rollback conditions, evidence required for "
            "acceptance revalidation, and how execution resumes or reconverges on later branches."
            " Preserve cross-cutting scientific, safety, interface, data-shape, "
            "performance, licensing, and validation constraints in every affected branch."
            f" Maximum {max_branches} branches."
        )
        prompt = (
            f"{self.context}\n\n"
            f"Original task (preserve every explicit term and constraint verbatim somewhere in the plan):\n"
            f"{tree.root.description}\n\n"
            f"Task (scientific reformulation):\n{scientific_request}\n\n"
        )
        if accumulated:
            prompt += f"{accumulated}\n\n"
        prompt += (
            "Decompose this task into sequential branches.  Each branch "
            "should be a coherent sub-task.  For each branch, explain the "
            "rationale for why it is needed."
        )
        if atomic:
            prompt += (
                " This is one atomic implementation job. Use no more than four direct branches: "
                "computational core, GUI/application integration, tests/documentation, and final "
                "validation/repair as applicable. Do not create planning-only or recursive branches."
            )
        try:
            raw = self._send(prompt=prompt, stream=False, system=system)
        except Exception:
            return [{"description": scientific_request, "rationale": "fallback"}]

        def extract_branches(payload: Any) -> list[dict]:
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)][:max_branches]
            if isinstance(payload, dict):
                candidates = payload.get("branches") or payload.get("steps") or payload.get("plan") or []
                if isinstance(candidates, dict):
                    candidates = candidates.get("branches") or candidates.get("steps") or []
                if isinstance(candidates, list):
                    return [item for item in candidates if isinstance(item, dict)][:max_branches]
            return []

        branches = extract_branches(self._safe_json(raw or ""))
        if branches:
            return branches
        repair_prompt = (
            "Repair the planning response below into strict JSON. Return only "
            "{\"branches\":[...]}; no markdown or commentary. Every branch needs nonempty "
            "id, description, rationale, dependencies, constraints, acceptance_criteria, "
            "and recovery_policy. Dependencies may reference only earlier branch ids. "
            f"Use at most {max_branches} branches and preserve every constraint from this task:\n"
            f"{scientific_request}\n\nInvalid prior response:\n{str(raw)[:5000]}"
        )
        try:
            repaired_raw = self._send(prompt=repair_prompt, stream=False, system=system)
            branches = extract_branches(self._safe_json(repaired_raw or ""))
            if branches:
                return branches
        except Exception:
            pass
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
            if self._total_agent_iterations >= self._max_total_agent_iterations:
                node.fail(
                    f"global agent-iteration budget exhausted ({self._max_total_agent_iterations})"
                )
                results.append(StepResult(
                    step_id=node.id,
                    description=node.scientific_form or node.description,
                    output="",
                    success=False,
                    attempts=iteration - 1,
                    error=node.error,
                ))
                break
            self._total_agent_iterations += 1
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
                    attribution = self._capture_attribution()
                    fixed = self._attempt_fix(node, step_result, tree)
                    correction_attribution = self._failed_fix_attribution(fixed, attribution)
                    self._report_correction(
                        classification="hallucination" if self._looks_model_caused(step_result.error) else "tool_failure",
                        trigger=step_result.error,
                        failed_output=step_result.output,
                        correction=(fixed.output if fixed else ""),
                        resolved=bool(fixed and fixed.success),
                        is_hallucination=self._looks_model_caused(step_result.error),
                        attribution=correction_attribution,
                        metadata={"branch": node.description, "tool_outputs": step_result.tool_outputs},
                    )
                    if fixed:
                        results.append(fixed)

            # --- Sub-branches -----------------------------------------
            elif action_type == "sub_branches":
                if "unattended atomic workday job" in self.context.lower():
                    rejection = (
                        "Atomic workday branches may not recursively decompose. "
                        "Implement this branch directly with file_write/executor evidence."
                    )
                    node.add_tool_output("sub_branch_rejected", {"error": rejection})
                    self._report_correction(
                        classification="protocol_hallucination", trigger=rejection,
                        failed_output=json.dumps(action, default=str)[:4000],
                        resolved=False, is_hallucination=True,
                        attribution=self._capture_attribution(),
                        metadata={"branch": node.description},
                    )
                    continue
                if depth >= self.MAX_BRANCH_DEPTH:
                    node.complete()
                    break
                sub_defs = action.get("sub_branches") or []
                for sub_def in sub_defs[: self.MAX_PLAN_BRANCHES]:
                    if isinstance(sub_def, str):
                        desc = sub_def
                        sub_def = {}
                    elif isinstance(sub_def, dict):
                        desc = str(sub_def.get("description", ""))
                    else:
                        continue
                    sci = self.reformulate(desc)
                    child = node.add_child(
                        description=desc,
                        scientific_form=sci,
                        rationale=str(sub_def.get("rationale") or ""),
                        dependencies=_as_str_list(sub_def.get("dependencies")),
                        constraints=_as_str_list(sub_def.get("constraints") or node.constraints),
                        acceptance_criteria=_as_str_list(sub_def.get("acceptance_criteria")),
                        recovery_policy=str(sub_def.get("recovery_policy") or node.recovery_policy),
                    )
                    child_results = self._execute_branch(child, tree, depth + 1)
                    results.extend(child_results)
                # After all sub-branches, check if parent is done.
                if node.all_children_done:
                    node.complete()
                break

            # --- Direct conversational answer -------------------------
            # Used when the branch is a question the model can answer
            # without any tool, OR when the model explicitly declines to
            # use a tool and wants to explain why.  Either way we surface
            # the text instead of swallowing it.
            elif action_type == "answer":
                output = (action.get("output") or "").strip()
                reason = (action.get("reason") or "").strip()
                if self._corrective_retry:
                    rejection = (
                        "Corrective retry rejected action=answer before validator evidence; "
                        "use tool_calls to inspect, patch, and validate."
                    )
                    node.add_tool_output("answer_rejected", {"error": rejection, "output": output})
                    self._report_correction(
                        classification="protocol_hallucination", trigger=rejection,
                        failed_output=output, resolved=False, is_hallucination=True,
                        attribution=self._capture_attribution(),
                        metadata={"branch": node.description},
                    )
                    continue
                non_answer_error = self._non_answer_validation_error(node, output)
                if non_answer_error:
                    node.add_tool_output("answer_rejected", {
                        "error": non_answer_error, "output": output,
                    })
                    self._report_correction(
                        classification="non_answer_hallucination",
                        trigger=non_answer_error, failed_output=output,
                        resolved=False, is_hallucination=True,
                        attribution=self._capture_attribution(),
                        metadata={"branch": node.description},
                    )
                    continue
                if not output and reason:
                    output = reason
                elif output and reason:
                    output = f"{output}\n\n[reason: {reason}]"
                if output:
                    node.add_tool_output("answer", {"text": output, "reason": reason})
                node.complete()
                results.append(StepResult(
                    step_id=node.id,
                    description=node.scientific_form or node.description,
                    output=output or "[no answer text returned]",
                    success=True,
                    attempts=iteration,
                ))
                break

            # --- Completion -------------------------------------------
            elif action_type == "complete":
                output = action.get("output", "")
                completion_error = self._completion_validation_error(node)
                if completion_error:
                    self._report_correction(
                        classification="premature_completion",
                        trigger=completion_error,
                        failed_output=str(output),
                        resolved=False,
                        is_hallucination=True,
                        attribution=self._capture_attribution(),
                        metadata={"branch": node.description},
                    )
                    node.add_tool_output("completion_validation", {
                        "error": completion_error,
                    })
                    continue
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

            # --- Unknown / malformed action ---------------------------
            # The model emitted something we don't recognise.  Surface
            # whatever text it returned rather than silently dropping
            # the turn — at minimum the user should see what the model
            # tried to say.
            else:
                raw_value = (action.get("output") or action.get("text")
                             or action.get("message") or "")
                if isinstance(raw_value, str):
                    raw_text = raw_value.strip()
                elif raw_value:
                    raw_text = json.dumps(raw_value, default=str).strip()
                else:
                    raw_text = ""
                if raw_text:
                    node.add_tool_output("fallback_text", {"text": raw_text,
                                                              "action": action_type})
                    results.append(StepResult(
                        step_id=node.id,
                        description=node.scientific_form or node.description,
                        output=raw_text,
                        success=True,
                        attempts=iteration,
                    ))
                node.complete()
                break

        # Safety net: if we exhausted iterations, mark done.
        if not node.is_done:
            node.complete()

        return results

    @staticmethod
    def _non_answer_validation_error(node: TaskNode, output: str) -> str:
        """Reject conversational acknowledgements presented as task answers."""
        text = " ".join(str(output or "").lower().split())
        acknowledgements = (
            "how can i assist", "how may i assist", "how can i help",
            "how may i help", "what would you like", "ready to help",
            "please provide the task", "please provide more details",
        )
        if any(marker in text for marker in acknowledgements):
            if Orchestrator._is_social_turn(node.description) and len(text.split()) >= 4:
                return ""
            return (
                "Answer rejected: the model returned a conversational acknowledgement "
                "instead of addressing the current branch. Execute or answer the branch."
            )
        if not text:
            return "Answer rejected: the model returned no substantive answer."
        return ""

    @staticmethod
    def _completion_validation_error(node: TaskNode) -> str:
        """Return an evidence error when a branch claims completion too early."""
        successful_tools = {
            str(entry.get("tool") or "")
            for entry in node.tool_outputs
            if isinstance(entry.get("result"), dict)
            and not entry["result"].get("error")
        }
        if not successful_tools:
            return (
                "Completion rejected: this branch has no successful tool-produced "
                "evidence. Execute the required tool call before completing."
            )

        task = f"{node.description}\n{node.scientific_form}".lower()
        mutation_markers = (
            "write", "replace", "implement", "create", "edit", "patch",
            "repair", "fix", "build", "update",
        )
        if any(marker in task for marker in mutation_markers) and "file_write" not in successful_tools:
            return (
                "Completion rejected: this is a code/file mutation branch but no "
                "successful file_write evidence exists. Write the requested file first."
            )

        validation_markers = ("node --check", "syntax-check", "syntax check", "validate with executor")
        if any(marker in task for marker in validation_markers) and "executor" not in successful_tools:
            return (
                "Completion rejected: this branch explicitly requires executor "
                "validation, but no successful executor evidence exists."
            )
        return ""

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
        tool_descriptions = self.tools.tool_descriptions()
        if self._corrective_retry:
            required = self._corrective_required_tool(node)
            navigation_allowed = self._corrective_navigation_allowed(node, required)
            if required == "file_write" and not navigation_allowed:
                allowed_names = {"file_write"}
            elif required == "executor":
                allowed_names = {"executor"}
            elif required == "file_read":
                allowed_names = {"file_locate", "file_read"}
            else:
                allowed_names = {"file_locate", "file_read", "file_write", "executor"}
            tool_descriptions = [
                item for item in tool_descriptions if item.get("name") in allowed_names
            ]
        tool_desc = json.dumps(tool_descriptions, indent=2)
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
            '3. {"action": "answer", "output": "<direct answer text>", "reason": "<why no tool was needed>"}\n'
            '   — Use when the branch is a question you can answer from context (rolling chat history, your own knowledge) with no tool required, OR when you have considered the tools and none is appropriate.  Always include BOTH output (the actual answer the user should see) and reason (one short sentence on why no tool was used).  Never leave both empty.\n'
            '4. {"action": "complete", "output": "<summary of what was accomplished>"}\n'
            '   — Only when this branch is fully resolved with tool-produced evidence.  If you have no evidence and only opinions, use action=answer instead.\n\n'
            "NEVER respond with prose outside the JSON.  NEVER use markdown fences.  "
            "The JSON must be parseable with json.loads().  "
            "If you genuinely have nothing to say, still emit action=answer with output explaining that, rather than action=complete with empty output.\n\n"
            f"Available tools:\n{tool_desc}"
        )
        if self._atomic_workday:
            system += (
                "\n\nATOMIC IMPLEMENTATION POLICY: Execute this branch directly; do not request "
                "sub-branches. Batch multiple independent file_write calls in one response. Write "
                "complete functional files, not empty placeholders. Use executor after writing to "
                "validate concrete behavior. Conserve model calls for implementation and repair. "
                "On a corrective retry, inspect every failing call site and the full implementation "
                "signature before editing; reconcile the complete interface contract in one patch, "
                "not only the first parameter or first traceback."
            )
        if self._corrective_retry:
            system += (
                "\nCORRECTIVE OVERRIDE: The validator evidence supplies exact local paths and errors. "
                "Skip memory_search and web_search unless the error is version/API-documentation dependent. "
                "Use file_locate and additional file_read calls whenever paths or interfaces remain unclear. "
                "First batch-read the failing tests "
                "and implementation, then patch the complete contract, then run the supplied validator. "
                "If an import target or required file does not exist, create it with file_write content "
                "(full-write mode); never send old_string/new_string patch parameters for a missing file. "
                "Until that validator succeeds, action=sub_branches, action=answer, and action=complete "
                "are invalid; respond only with action=tool_calls. When hidden/reference invariants pass "
                "but generated tests conflict with those invariants or their own configured limits, treat "
                "the reference evidence as ground truth and repair the faulty generated test fixtures; "
                "do not weaken a working implementation merely to satisfy a contradictory assertion. "
                "When any hidden/reference invariant is failing, test files are immutable: do not edit "
                "tests, validators, or fixtures; repair the production implementation only."
            )
        correction_guidance = getattr(self.session, "correction_guidance", lambda _context="": "")(
            f"{node.description}\n{node.scientific_form}"
        )
        if correction_guidance:
            system += f"\n\n{correction_guidance}"

        corrective_tools = {
            str(entry.get("tool") or "")
            for entry in node.tool_outputs
            if isinstance(entry.get("result"), dict) and not entry["result"].get("error")
        }
        failed_executor = next((
            entry.get("result") or {}
            for entry in reversed(node.tool_outputs)
            if entry.get("tool") == "executor"
            and isinstance(entry.get("result"), dict)
            and (
                entry["result"].get("error")
                or int(entry["result"].get("return_code") or 0) != 0
            )
        ), None)
        if self._corrective_retry:
            if failed_executor:
                system += (
                    "\nCORRECTIVE STATE: The last executor validation failed. Its stderr/stdout is in "
                    "accumulated results. Do not rerun the same command unchanged. Inspect any still-unclear "
                    "path or call site, then patch the reported error with file_write."
                )
            elif "file_write" in corrective_tools and "executor" not in corrective_tools:
                system += (
                    "\nCORRECTIVE STATE: A patch was already written. Run the supplied validator with "
                    "executor now, unless another necessary file in the same repair still needs inspection "
                    "or modification."
                )
            elif "file_read" in corrective_tools and "file_write" not in corrective_tools:
                system += (
                    "\nCORRECTIVE STATE: At least one failing file was read successfully. Locate/read "
                    "any other required call sites, then use file_write for the complete interface/test repair."
                )
            elif not corrective_tools:
                system += (
                    "\nCORRECTIVE STATE: Batch all necessary file_read calls in this first action only."
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
            raw = self._send(prompt=prompt, stream=False, system=system)
        except Exception:
            return None
        parsed = self._safe_json(raw or "")
        if parsed is not None:
            return self._normalize_action(parsed)
        # Hosted/free models occasionally describe the tool call in prose even
        # though the control prompt requires JSON. Give the same model one
        # explicit protocol-repair opportunity before treating the text as a
        # conversational answer; otherwise coding branches silently terminate.
        text = (raw or "").strip()
        if text:
            repair_prompt = (
                prompt
                + "\n\nYour previous response violated the required JSON protocol:\n"
                + text[:1500]
                + "\n\nDo not explain or plan. Emit exactly one parseable JSON object "
                  "using one of the allowed action shapes now."
            )
            try:
                repaired_raw = self._send(
                    prompt=repair_prompt,
                    stream=False,
                    system=system,
                )
                repaired = self._safe_json(repaired_raw or "")
                if repaired is not None:
                    return self._normalize_action(repaired)
            except Exception:
                pass
        # Hebbian recall backends (W1z4rD brain) often return prose rather
        # than JSON.  Surface that text as a direct answer instead of
        # silently dropping the turn.  Sentinels like
        # "[wizard-brain unavailable: ...]" / "[wizard-brain OOG]" /
        # "[wizard-brain hypothesis]" indicate the brain has nothing
        # useful — flag as no_recall so the caller can escalate to tools.
        if not text:
            return None
        low = text.lower()
        if low.startswith(("[wizard-brain unavailable", "[wizard-brain error",
                           "[wizard-brain oog", "[wizard-brain hypothesis")):
            return {"action": "answer", "output": text,
                    "reason": "brain returned no grounded recall"}
        return {"action": "answer", "output": text,
                "reason": "non-JSON response surfaced verbatim"}

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _hidden_invariants_failing(self) -> bool:
        """Conservatively protect tests unless embedded validator evidence says hidden checks pass."""
        if "hidden" not in self.context.lower():
            return False
        statuses = re.findall(
            r'"command"\s*:\s*(?:\[[^\]]*<hidden[^\]]*\]|"[^"]*<hidden[^"]*")'
            r'[\s\S]{0,2000}?"ok"\s*:\s*(true|false)',
            self.context,
            flags=re.IGNORECASE,
        )
        return not statuses or any(status.lower() != "true" for status in statuses)

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
        attribution = self._capture_attribution()
        wrote_in_batch = False
        for call in calls:
            tool_name = str(call.get("tool", ""))
            params = call.get("params") or {}
            if not tool_name:
                continue
            # Re-evaluate after every call so a batched read -> write -> executor
            # response can advance through the corrective state machine. File
            # navigation remains legal at every stage; it does not satisfy the
            # required mutation/validation transition by itself.
            required_tool = self._corrective_required_tool(node) if self._corrective_retry else ""
            navigation_tools = {"file_locate", "file_read"}
            navigation_allowed = self._corrective_navigation_allowed(node, required_tool)
            allowed = (
                not required_tool
                or tool_name == required_tool
                or (tool_name in navigation_tools and navigation_allowed)
                or (required_tool == "executor" and tool_name == "file_write" and wrote_in_batch)
            )
            path_parts = re.split(r"[\\/]", str(params.get("path") or "").lower())
            protected_test_write = (
                self._corrective_retry and tool_name == "file_write"
                and "tests" in path_parts and self._hidden_invariants_failing()
            )
            if (
                self._corrective_retry and tool_name == "file_write"
                and self._hidden_invariants_failing() and "tests" not in path_parts
            ):
                params = {**params, "require_semantic_change": True}
            if protected_test_write:
                result = {
                    "error": (
                        "Corrective test write rejected while hidden/reference invariants fail; "
                        "repair the production implementation instead."
                    )
                }
            elif not allowed:
                result = {
                    "error": (
                        f"Corrective state requires {required_tool}; rejected {tool_name}. "
                        "The navigation allowance is exhausted; advance the validator-driven "
                        "read -> write -> executor sequence now."
                    )
                }
            else:
                result = self.tools.dispatch(tool_name, params)
            if attribution:
                result = {**result, "_attribution": attribution}
            tool_outputs.append({"tool": tool_name, "result": result})
            if tool_name == "file_write" and not result.get("error"):
                wrote_in_batch = True

            # Record on the node so accumulated context grows.
            node.add_tool_output(tool_name, result)

            if result.get("error"):
                errors.append(f"{tool_name}: {result['error']}")

            # Collect readable output.
            for key in ("stdout", "summary", "result", "content", "preview"):
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

    @staticmethod
    def _corrective_required_tool(node: TaskNode) -> str:
        """Return the only valid next tool for a validator-driven repair."""
        successful_read = any(
            entry.get("tool") == "file_read"
            and isinstance(entry.get("result"), dict)
            and not entry["result"].get("error")
            for entry in node.tool_outputs
        )
        if not successful_read:
            return "file_read"

        last_write = -1
        last_executor = -1
        executor_ok = False
        for index, entry in enumerate(node.tool_outputs):
            result = entry.get("result") or {}
            if entry.get("tool") == "file_write" and not result.get("error"):
                last_write = index
            elif entry.get("tool") == "executor":
                last_executor = index
                executor_ok = not result.get("error") and int(result.get("return_code") or 0) == 0

        if last_write < 0:
            return "file_write"
        if last_executor < last_write:
            return "executor"
        if executor_ok:
            return ""
        return "file_write"

    @staticmethod
    def _corrective_navigation_allowed(node: TaskNode, required_tool: str) -> bool:
        """Allow enough inspection for context without permitting read-only loops."""
        if required_tool in {"", "file_read"}:
            return True
        anchor = -1
        if required_tool == "executor":
            anchor = max(
                (index for index, entry in enumerate(node.tool_outputs)
                 if entry.get("tool") == "file_write"
                 and not (entry.get("result") or {}).get("error")),
                default=-1,
            )
            limit = 2
        else:
            anchor = max(
                (index for index, entry in enumerate(node.tool_outputs)
                 if entry.get("tool") == "executor"),
                default=-1,
            )
            limit = 6
        navigation = sum(
            1 for entry in node.tool_outputs[anchor + 1:]
            if entry.get("tool") in {"file_locate", "file_read"}
            and not (entry.get("result") or {}).get("error")
        )
        return navigation < limit

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
        tool_descriptions = self.tools.tool_descriptions()
        if self._corrective_retry:
            required = self._corrective_required_tool(node)
            navigation_allowed = self._corrective_navigation_allowed(node, required)
            if required == "file_write" and not navigation_allowed:
                allowed_names = {"file_write"}
            elif required == "executor":
                allowed_names = {"executor"}
            elif required == "file_read":
                allowed_names = {"file_locate", "file_read"}
            else:
                allowed_names = {"file_locate", "file_read", "file_write", "executor"}
            tool_descriptions = [
                item for item in tool_descriptions if item.get("name") in allowed_names
            ]
        tool_desc = json.dumps(tool_descriptions, indent=2)
        accumulated = tree.accumulated_results_summary()

        if self._needs_web_verification(failed_result.error):
            research = self.tools.dispatch("web_search", {
                "query": f"official documentation {node.description} {failed_result.error[:500]}"
            })
            if not research.get("error"):
                node.add_tool_output("web_search", research)
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
            raw = self._send(prompt=prompt, stream=False, system=system)
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

    def _send(self, *args: Any, **kwargs: Any) -> str:
        if self._total_model_calls >= self._max_total_model_calls:
            raise ModelCallBudgetExceeded(
                f"global model-call budget exhausted ({self._max_total_model_calls})"
            )
        self._total_model_calls += 1
        return self.session.send(*args, **kwargs)

    @staticmethod
    def _safe_json(text: str) -> Any:
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\[{]", text):
            try:
                value, _ = decoder.raw_decode(text[match.start():])
                return value
            except Exception:
                continue
        return None

    def _capture_attribution(self) -> dict:
        route = getattr(self.session, "last_route", [])
        selected = [item for item in route if isinstance(item, dict) and item.get("outcome") == "selected"]
        return dict(selected[-1]) if selected else {}

    @staticmethod
    def _failed_fix_attribution(fixed: StepResult | None, fallback: dict) -> dict:
        if fixed and not fixed.success:
            for tool_output in reversed(fixed.tool_outputs):
                candidate = (tool_output.get("result") or {}).get("_attribution") or {}
                if candidate:
                    return dict(candidate)
        return dict(fallback)

    def _report_correction(self, **payload: Any) -> None:
        reporter = getattr(self.session, "report_correction", None)
        if not callable(reporter):
            return
        attribution = payload.pop("attribution", {})
        metadata = dict(payload.pop("metadata", {}) or {})
        if attribution:
            metadata["origin_attribution"] = attribution
        try:
            reporter(metadata=metadata, origin_attribution=attribution, **payload)
        except Exception:
            pass

    @staticmethod
    def _looks_model_caused(error: str) -> bool:
        lowered = (error or "").lower()
        markers = (
            "unknown tool", "no command provided", "no query provided",
            "no path provided", "file not found", "old_string not found",
            "provide content", "syntaxerror", "syntax error", "parse error",
            "unexpected token", "cannot find module", "modulenotfounderror",
            "no such file", "unrecognized argument", "invalid parameter",
            "corrective state requires", "navigation allowance is exhausted",
            "test write rejected", "patch rejected;",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _needs_web_verification(error: str) -> bool:
        lowered = (error or "").lower()
        if any(marker in lowered for marker in (
            "old_string not found", "file not found", "no such file", "path escapes workdir",
            "syntaxerror", "syntax error",
        )):
            return False
        markers = (
            "module", "package", "import", "version", "deprecated", "api",
            "argument", "parameter", "attribute", "unknown tool", "syntax",
            "not found", "does not exist", "unsupported",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _normalize_action(payload: Any) -> Any:
        """Accept common function-call dialects emitted by hosted models.

        C0d3rV2's canonical protocol uses ``action=tool_calls`` and a list of
        ``{tool, params}`` objects.  OpenAI-compatible models also commonly
        produce a singular ``tool_call``, nest that call under ``tool_call``,
        or call the argument object ``args``.  Normalizing at this boundary
        keeps the tool registry strict while allowing ATF to rotate models.
        """
        if isinstance(payload, list):
            if len(payload) == 1 and isinstance(payload[0], dict):
                payload = payload[0]
            elif payload and all(isinstance(item, dict) for item in payload):
                if any(item.get("action") for item in payload):
                    payload = next((item for item in payload if item.get("action")), payload[0])
                else:
                    payload = {"action": "tool_calls", "tool_calls": payload}
            else:
                return {"action": "answer", "output": json.dumps(payload, default=str),
                        "reason": "unsupported top-level list response surfaced verbatim"}

        if not isinstance(payload, dict):
            return payload

        action = str(payload.get("action") or "").strip()
        if action == "tool_call":
            nested = payload.get("tool_call")
            call = nested if isinstance(nested, dict) else payload
            payload = {"action": "tool_calls", "tool_calls": [call]}
        elif not action and (payload.get("tool") or payload.get("name")):
            payload = {"action": "tool_calls", "tool_calls": [payload]}

        if payload.get("action") != "tool_calls":
            return payload

        raw_calls = payload.get("tool_calls") or []
        if isinstance(raw_calls, dict):
            raw_calls = [raw_calls]
        calls: list[dict] = []
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function")
            if isinstance(function, dict):
                raw_call = {**raw_call, **function}
            tool = raw_call.get("tool") or raw_call.get("name")
            params = raw_call.get("params")
            if params is None:
                params = raw_call.get("args")
            if params is None:
                params = raw_call.get("arguments")
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except Exception:
                    params = {}
            if not isinstance(params, dict):
                params = {}
            if "filename" in params and "path" not in params:
                params["path"] = params.pop("filename")
            if tool:
                calls.append({"tool": str(tool), "params": params})
        return {**payload, "action": "tool_calls", "tool_calls": calls}
