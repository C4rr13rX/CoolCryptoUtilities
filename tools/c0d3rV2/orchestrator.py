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
from pathlib import Path
from typing import Any

from task_tree import TaskNode, TaskTree
from tool_registry import ToolRegistry
from outline_refiner import OutlineRefiner, is_creation_request
try:
    from model_response_normalizer import ModelResponseNormalizer
except ModuleNotFoundError:  # package import in tests/library consumers
    from .model_response_normalizer import ModelResponseNormalizer


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
        "  3a. Before a cross-file change or validator repair, use dependency_traversal. "
        "If a dependency/regression injection packet is already present, follow its ordered "
        "regression_route and hashed evidence instead of rediscovering unrelated files.\n"
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

    def _control_prefix(self) -> str:
        """Expose routing rules only for tools that this flow actually owns.

        Delivery flows intentionally register a smaller tool surface than the
        interactive desktop flow. Advertising unavailable tools makes smaller
        models repeatedly request impossible branches before correcting.
        """
        available = set(self.tools.tool_names())
        blocks = re.split(r"(?=  \d+\. )", self.CONTROL_PREFIX)
        known = {
            "memory_search", "file_locate", "file_read", "file_write",
            "math_grounding", "scientific_method", "web_search",
            "equation_matrix", "unbounded_solver", "vm_playground",
            "executor", "c0d3r_native_os", "directory_ensure",
        }
        kept: list[str] = []
        for block in blocks:
            mentioned = {name for name in known if re.search(rf"\b{re.escape(name)}\b", block)}
            if mentioned and not mentioned.issubset(available):
                continue
            kept.append(block)
        guidance: list[str] = []
        if {"scientific_method", "web_search", "equation_matrix"}.issubset(available):
            guidance.append(
                "For uncertain scientific claims, use web_search for archival evidence, "
                "scientific_method for falsifiable evaluation, and equation_matrix for mathematical grounding."
            )
        return (
            "".join(kept)
            + "\nOnly call tools present in Available tools; never invent or substitute an unregistered tool.\n"
            + "\n".join(guidance)
            + "\n"
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
        self._bounded_read_only = (
            "bounded read-only archival-research role" in context.lower()
        )
        self._total_agent_iterations = 0
        self._total_model_calls = 0
        self.refined_outline: dict[str, Any] = {}
        self.response_normalizer = ModelResponseNormalizer.from_tool_descriptions(
            self.tools.tool_descriptions()
        )
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
        if self._atomic_workday:
            # BrandDozer already persisted a scope-locked plan, input/output
            # contract, and exact next work package. Reclassifying it with a
            # scarce model call cannot make the task smaller and gives weak
            # providers another opportunity to derail before execution.
            scrutiny = {
                "decision": "execute",
                "scientific_request": request,
                "branches": [],
            }
        elif not corrective_retry:
            scrutiny = self._scrutinize(request)
            if scrutiny.get("decision") == "direct" and (
                self._bounded_read_only or not is_creation_request(request)
            ):
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

        refined_branches: list[dict[str, Any]] = []
        # BrandDozer atomic work packages are already derived from a persisted,
        # scope-locked outline and project map. Refining them again wastes
        # several scarce calls and commonly expands one class/scaffold into
        # redundant branches before any file is written.
        if (
            not corrective_retry
            and not self._atomic_workday
            and not self._bounded_read_only
            and is_creation_request(request)
        ):
            self.refined_outline, refined_branches = self._refine_creation_request(
                request, scientific_request,
            )
            if self.refined_outline:
                scientific_request = str(
                    self.refined_outline.get("scientific_request") or scientific_request
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
        }] if corrective_retry else ([{
            "id": "atomic-work-package",
            "description": scientific_request,
            "rationale": "Execute the persisted BrandDozer work package without replanning it.",
            "dependencies": [],
            "constraints": ["Stay inside the named work package and project root"],
            "acceptance_criteria": ["Produce mutation evidence and run the package validation command"],
            "recovery_policy": "Use validator evidence to repair the same package, then rerun validation.",
        }] if self._atomic_workday else refined_branches or list(scrutiny.get("branches") or []) or [{
            "id": "execute-request", "description": scientific_request,
            "rationale": "The scrutiny gate determined that execution is required.",
        }]))
        mapper_task_by_node: dict[str, str] = {}
        for index, branch_def in enumerate(branches, start=1):
            desc = str(branch_def.get("description", ""))
            sci = desc
            alias = str(branch_def.get("id") or f"step-{index}")
            child = tree.root.add_child(
                description=f"[{alias}] {desc}",
                scientific_form=sci,
                rationale=str(branch_def.get("rationale") or ""),
                dependencies=_as_str_list(branch_def.get("dependencies")),
                constraints=_as_str_list(branch_def.get("constraints")),
                acceptance_criteria=_as_str_list(branch_def.get("acceptance_criteria")),
                recovery_policy=str(branch_def.get("recovery_policy") or "Diverge only to resolve evidence-backed blockers; then return to this step's acceptance criteria."),
            )
            if refined_branches:
                mapper_task_by_node[child.id] = alias

        # Step 2 — execute each branch recursively.
        all_results: list[StepResult] = []
        completed_aliases: set[str] = set()
        planned_aliases = set(mapper_task_by_node.values())
        for child in tree.root.children:
            mapper_task = mapper_task_by_node.get(child.id)
            required_planned = set(child.dependencies) & planned_aliases
            if mapper_task and not required_planned.issubset(completed_aliases):
                child.fail("dependency contract did not complete successfully")
                all_results.append(StepResult(
                    step_id=child.id,
                    description=child.scientific_form or child.description,
                    output="",
                    success=False,
                    attempts=0,
                    error=child.error,
                ))
                continue
            branch_results = self._execute_branch(child, tree, depth=1)
            all_results.extend(branch_results)
            successful = child.status == "completed" and any(
                result.success for result in branch_results
            )
            if mapper_task and successful and self.tools.get("project_work_mapper"):
                evidence = {
                    "branch_id": child.id,
                    "tool_outputs": child.tool_outputs[-10:],
                    "result_count": len(branch_results),
                }
                self.tools.dispatch("project_work_mapper", {
                    "action": "complete", "task_id": mapper_task,
                    "evidence": evidence,
                })
                completed_aliases.add(mapper_task)

        # Finalize tree.
        tree.mark_root_complete()

        # Evaluate petal effectiveness.
        if self.petals:
            feedback = [
                {"petal": r.step_id, "success": r.success} for r in all_results
            ]
            self.petals.evaluate_effectiveness(feedback)

        return all_results, tree

    def _refine_creation_request(
        self, request: str, scientific_request: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Refine scope before any implementation branch can execute."""
        def market_search(query: str) -> dict:
            result = self.tools.dispatch("web_search", {"query": query})
            return result if isinstance(result, dict) and not result.get("error") else {}

        refiner = OutlineRefiner(
            send=self._send,
            market_search=market_search if self.tools.get("web_search") else None,
        )
        outline = refiner.refine(request, scientific_request)
        if not (outline.get("quality") or {}).get("passed"):
            # Deterministic passes normally clear this gate. If they do not,
            # refuse to disguise a thin plan as implementation readiness.
            return outline, [{
                "id": "outline-quality-blocker",
                "description": "Refine the persisted outline until its planning quality gate passes",
                "rationale": "Implementation is prohibited before the outline is complete and scope-locked.",
                "dependencies": [],
                "constraints": list((outline.get("scope_boundary") or {}).get("forbidden") or []),
                "acceptance_criteria": ["Outline quality score is >= 92/100 with zero scope violations"],
                "recovery_policy": "Add missing contracts or validation detail without adding a new user goal, then rescore.",
            }]
        mapped = self.tools.dispatch("project_work_mapper", {
            "action": "map",
            "request": request,
            "acceptance": {
                "quality": outline.get("quality"),
                "deliverables": outline.get("deliverables"),
                "requirements": outline.get("functional_requirements"),
                "validation": outline.get("validation"),
            },
            "outline": outline,
        }) if self.tools.get("project_work_mapper") else {}
        tasks = mapped.get("tasks") if isinstance(mapped, dict) else []
        branches: list[dict[str, Any]] = []
        for task in tasks or []:
            if task.get("status") == "complete":
                continue
            inputs = task.get("inputs") or {}
            outputs = task.get("outputs") or {}
            branches.append({
                "id": str(task.get("id") or f"contract-{len(branches)+1}"),
                "description": (
                    f"{task.get('title')}. INPUT CONTRACT: {json.dumps(inputs, ensure_ascii=True)}. "
                    f"OUTPUT CONTRACT: {json.dumps(outputs, ensure_ascii=True)}."
                ),
                "rationale": "Atomic contract produced by the scope-locked project mapper.",
                "dependencies": _as_str_list(task.get("depends_on")),
                "constraints": [
                    *list((mapped.get("scope") or {}).get("allowed_roots") or []),
                    *_as_str_list(task.get("forbidden")),
                    "Do not redesign or expand the original request",
                ],
                "acceptance_criteria": _as_str_list(task.get("acceptance")),
                "recovery_policy": "Use observed validation evidence only; make the smallest in-scope repair and rerun this contract's checks.",
            })
        return outline, branches[:self.MAX_PLAN_BRANCHES]

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
        normalized = self.response_normalizer.normalize_scrutiny(raw or "")
        payload = normalized.value if normalized.valid else None
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
        self._seed_validator_context(node)

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
                if self._atomic_workday:
                    # Read-only context gathering is safe to batch within its
                    # deterministic budget. Mutations and command execution
                    # remain serialized so each has its own validation
                    # boundary and truncated tails cannot partially apply.
                    progress_tools = {
                        "file_write", "workspace_scaffold",
                        "environment_bootstrap", "executor",
                    }
                    progress_calls = [
                        call for call in calls
                        if str(call.get("tool") or "") in progress_tools
                    ]
                    if progress_calls:
                        calls = progress_calls[:1]
                    else:
                        validator_directed = "validator-directed repair paths" in (
                            f"{node.description}\n{node.scientific_form}"
                        ).lower()
                        navigation_limit = self._atomic_navigation_limit(node)
                        calls = self._prioritize_repair_calls(node, calls)
                        successful_navigation = sum(
                            entry.get("tool") in {"memory_search", "file_locate", "file_read"}
                            and isinstance(entry.get("result"), dict)
                            and not entry["result"].get("error")
                            for entry in node.tool_outputs
                        )
                        calls = calls[:max(1, navigation_limit - successful_navigation)]
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
        mutation_tools = {"file_write", "workspace_scaffold", "environment_bootstrap"}
        if Orchestrator._task_requests_mutation(task) and not (successful_tools & mutation_tools):
            return (
                "Completion rejected: this is a code/file mutation branch but no "
                "successful file_write/scaffold/bootstrap evidence exists. Mutate the requested workspace first."
            )

        validation_markers = ("node --check", "syntax-check", "syntax check", "validate with executor")
        if any(marker in task for marker in validation_markers) and "executor" not in successful_tools:
            return (
                "Completion rejected: this branch explicitly requires executor "
                "validation, but no successful executor evidence exists."
            )
        return ""

    @staticmethod
    def _task_requests_mutation(task: str) -> bool:
        """Distinguish requested mutations from negated/read-only vocabulary."""
        text = " ".join(str(task or "").lower().split())
        if any(marker in text for marker in ("read-only", "read only", "without modifying")):
            # Explicit evidence/review tasks often mention forbidden writes in
            # their safety contract; those words must not turn into a write gate.
            text = re.sub(
                r"\b(?:do not|don't|never|without)\s+(?:write|modify|edit|change|mutate|create|update)\b",
                "", text,
            )
            if not re.search(
                r"\b(?:implement|repair|fix|build|patch|replace|refactor|add|remove|rename|move)\b",
                text,
            ):
                return False
        text = re.sub(
            r"\b(?:do not|don't|never|without)\s+(?:write|modify|edit|change|mutate|create|update)\b",
            "", text,
        )
        return bool(re.search(
            r"\b(?:write|replace|implement|create|edit|patch|repair|fix|build|update|"
            r"refactor|add|remove|rename|move|modify|mutate)\b",
            text,
        ))

    @staticmethod
    def _compact_atomic_task_text(text: str, *, max_chars: int = 15000) -> str:
        """Retain the active contract while dropping duplicated validator history."""
        value = str(text or "")
        if len(value) <= max_chars:
            return value
        anchors: list[str] = []
        for pattern in (
            r"Current plan summary:\s*[^\n]+",
            r"Next step:\s*[^\n]+",
            r"Atomic work package:\s*[^\n]+",
            r"Expected outputs:\s*[^\n]+",
            r"Acceptance checks:\s*[^\n]+",
            r"Validator-directed repair paths:\s*\[[^\]]*\][^.]*\.?",
        ):
            match = re.search(pattern, value, flags=re.IGNORECASE)
            if match:
                anchors.append(match.group(0))
        marker = value.rfind("Persisted deterministic repair packet")
        if marker < 0:
            marker = value.rfind("Deterministic repair packet")
        tail_budget = max(2000, max_chars - sum(len(item) + 1 for item in anchors))
        tail = value[marker:] if marker >= 0 else value[-tail_budget:]
        compact = "\n".join(anchors + [tail[-tail_budget:]])
        return compact[-max_chars:]

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
        atomic_required = ""
        if self._atomic_workday and not self._corrective_retry:
            validator_directed = "validator-directed repair paths" in (
                f"{node.description}\n{node.scientific_form}"
            ).lower()
            navigation_limit = self._atomic_navigation_limit(node)
            successful = [
                entry for entry in node.tool_outputs
                if isinstance(entry.get("result"), dict) and not entry["result"].get("error")
            ]
            progress_tools = {"file_write", "workspace_scaffold", "environment_bootstrap", "executor"}
            mutation_tools = {"file_write", "workspace_scaffold", "environment_bootstrap"}
            mutation_seen = any(entry.get("tool") in mutation_tools for entry in successful)
            last_mutation = max(
                (index for index, entry in enumerate(node.tool_outputs)
                 if entry.get("tool") in mutation_tools
                 and isinstance(entry.get("result"), dict)
                 and not entry["result"].get("error")),
                default=-1,
            )
            last_executor = max(
                (index for index, entry in enumerate(node.tool_outputs)
                 if entry.get("tool") == "executor"),
                default=-1,
            )
            last_executor_failed = False
            if last_executor >= 0:
                executor_result = node.tool_outputs[last_executor].get("result") or {}
                last_executor_failed = bool(executor_result.get("error")) or int(
                    executor_result.get("return_code") or 0
                ) != 0
            last_progress = max(
                (index for index, entry in enumerate(successful) if entry.get("tool") in progress_tools),
                default=-1,
            )
            navigation_total = sum(
                entry.get("tool") in {"memory_search", "file_locate", "file_read"}
                for entry in successful
            )
            navigation_since = sum(
                entry.get("tool") in {"memory_search", "file_locate", "file_read"}
                for entry in successful[last_progress + 1:]
            )
            allowed_names: set[str] | None = None
            if not mutation_seen and navigation_total >= navigation_limit:
                # Inspection has established enough context. A single write
                # is the universally valid transition for both initialized
                # and empty workspaces and cannot destructively re-bootstrap
                # an established project.
                allowed_names = {"file_write", "executor"}
                atomic_required = (
                    "Navigation is exhausted. Use exactly one file_write for a necessary "
                    "repair, or run executor validation if the inspected contract is already correct."
                )
            elif mutation_seen and validator_directed:
                if last_executor > last_mutation and last_executor_failed:
                    # Validation is fresh evidence.  Reopen exactly one edit
                    # rather than trapping small models in an executor loop.
                    allowed_names = {"file_write"}
                    atomic_required = (
                        "Fresh executor validation failed after the last mutation. Apply exactly one "
                        "evidence-directed file_write within the declared repair paths; do not validate "
                        "again until that correction lands."
                    )
                else:
                    # Every mutation must cross an external validation boundary
                    # before another mutation is permitted.
                    allowed_names = {"executor"}
                    atomic_required = (
                        "A validator-directed mutation already landed. Run executor validation "
                        "now; do not repeat or extend the edit before collecting fresh evidence."
                    )
            elif mutation_seen and navigation_since >= 3:
                allowed_names = {"file_write", "executor"}
                atomic_required = "Write the next cohesive unit or run validation now; navigation is exhausted."
            if allowed_names is not None:
                tool_descriptions = [
                    item for item in tool_descriptions if item.get("name") in allowed_names
                ]
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
        accumulated = tree.accumulated_results_summary(max_chars=7000 if self._atomic_workday else 7000)
        tree_summary = tree.context_summary()

        system = (
            self._control_prefix()
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
                "sub-branches. Emit exactly one complete file_write call per response, then continue in the "
                "next iteration; never attempt to emit an entire application in one JSON response. Write "
                "complete functional files, not empty placeholders. Use executor after writing to "
                "validate concrete behavior. Conserve model calls for implementation and repair. "
                "For a new application foundation, prefer workspace_scaffold or environment_bootstrap, "
                "then upgrade the generated files with file_write and validate the build/tests. "
                "The host shell is PowerShell; never use Unix-only flags such as `ls -la`. "
                "On a corrective retry, inspect every failing call site and the full implementation "
                "signature before editing; reconcile the complete interface contract in one patch, "
                "not only the first parameter or first traceback. Never add a counterfeit production "
                "fallback solely to satisfy a headless test; separate pure domain construction from "
                "platform adapters or inject the platform dependency and use an explicit test double."
            )
            if atomic_required:
                system += f"\nATOMIC REQUIRED TRANSITION: {atomic_required}"
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
        guidance_fn = getattr(self.session, "correction_guidance", lambda _context="", **_kwargs: "")
        guidance_context = f"{node.description}\n{node.scientific_form}"
        try:
            correction_guidance = guidance_fn(
                guidance_context, limit=3 if self._atomic_workday else 8,
            )
        except TypeError:
            correction_guidance = guidance_fn(guidance_context)
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

        if self._atomic_workday:
            task_text = self._compact_atomic_task_text(node.description)
            if node.scientific_form and node.scientific_form not in node.description:
                scientific_tail = self._compact_atomic_task_text(node.scientific_form, max_chars=2500)
                task_text = (task_text + "\nScientific constraints:\n" + scientific_tail)[-18000:]
            prompt_parts = [f"System context:\n{str(self.context)[:4000]}"]
        else:
            task_text = node.description
            prompt_parts = [
                f"System context:\n{self.context}",
                f"\nCurrent task tree:\n{tree_summary}",
            ]
        if accumulated:
            prompt_parts.append(f"\n{accumulated}")
        prompt_parts.append(
            f"\nCurrent branch [{node.id}]:\n"
            f"  Description: {task_text}\n"
            + (f"  Scientific form: {node.scientific_form}\n" if not self._atomic_workday else "")
            +
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

        if atomic_required:
            prompt_parts.append(
                "\nMANDATORY NEXT ACTION: " + atomic_required + " "
                "Respond with action=tool_calls and exactly one call using only a tool "
                "listed in Available tools. Do not request more context, discovery, reads, "
                "sub-branches, an answer, or completion."
            )
        else:
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
        normalized = self.response_normalizer.normalize_action(raw or "")
        if normalized.valid:
            if normalized.transformations:
                node.add_tool_output("response_normalizer", {
                    "status": "normalized",
                    "transformations": normalized.transformations[-20:],
                })
            return normalized.value
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
                repaired = self.response_normalizer.normalize_action(repaired_raw or "")
                if repaired.valid:
                    node.add_tool_output("response_normalizer", {
                        "status": "protocol_repaired",
                        "transformations": repaired.transformations[-20:],
                    })
                    return repaired.value
            except Exception:
                pass
        if self._atomic_workday:
            trigger = (
                "Atomic implementation response did not match any action schema and supplied "
                "no tool evidence; prose claims of edits, tests, or completion are unsupported."
            )
            self._report_correction(
                classification="protocol_hallucination",
                trigger=trigger,
                failed_output=text[:4000],
                correction="Emit one schema-valid tool call and rely on validator evidence.",
                resolved=False,
                is_hallucination=True,
                attribution=self._capture_attribution(),
                metadata={"branch": node.description},
            )
            node.add_tool_output("response_normalizer", {"error": trigger})
            return {"action": "tool_calls", "tool_calls": []}
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

    @staticmethod
    def _evidence_forbidden_write_error(node: TaskNode, params: dict) -> str:
        """Reject a validator-named anti-pattern before it reaches disk."""
        evidence = f"{node.description}\n{node.scientific_form}".lower()
        path = str(params.get("path") or "").replace("\\", "/").lower()
        existing = ""
        for item in reversed(node.tool_outputs):
            if item.get("tool") != "file_read" or not isinstance(item.get("result"), dict):
                continue
            result = item["result"]
            read_path = str(result.get("path") or "").replace("\\", "/").lower()
            if read_path == path or read_path.endswith(f"/{path}") or path.endswith(f"/{read_path}"):
                existing = str(result.get("content") or "")
                break
        proposed = str(params.get("content") or "")
        if not proposed and params.get("new_string") is not None:
            # Judge the effective file, not merely the replacement fragment. A
            # superficially harmless one-line patch must not leave a validator-
            # identified counterfeit implementation in the rest of the file.
            old_string = str(params.get("old_string") or "")
            proposed = (
                existing.replace(old_string, str(params.get("new_string") or ""), 1)
                if existing and old_string and old_string in existing
                else str(params.get("new_string") or "")
            )
        if Path(path).name.lower() == "package.json" and existing and proposed:
            try:
                before_manifest = json.loads(existing)
                after_manifest = json.loads(proposed)
            except (TypeError, ValueError, json.JSONDecodeError):
                before_manifest = after_manifest = {}
            removed: list[str] = []
            for section in ("scripts", "dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
                before_items = before_manifest.get(section) or {}
                after_items = after_manifest.get(section) or {}
                if isinstance(before_items, dict) and isinstance(after_items, dict):
                    removed.extend(f"{section}.{key}" for key in before_items if key not in after_items)
            if removed and not any(marker in evidence for marker in (
                "remove dependency", "remove script", "obsolete dependency", "conflicting dependency",
            )):
                return (
                    "Evidence-directed write rejected: corrective manifest rewrite removes established entries "
                    + ", ".join(removed[:12])
                    + "; apply an additive/targeted manifest patch."
                )
        if existing and proposed and Path(path).suffix.lower() in {".ts", ".tsx", ".js", ".jsx", ".py"}:
            export_pattern = re.compile(
                r"\bexport\s+(?:default\s+)?(?:async\s+)?"
                r"(?:function|class|const|let|var|type|interface|enum)\s+([A-Za-z_$][\w$]*)"
            )
            before_exports = set(export_pattern.findall(existing))
            after_exports = set(export_pattern.findall(proposed))
            removed_exports = sorted(before_exports - after_exports)
            if removed_exports and not any(marker in evidence for marker in (
                "remove export", "rename export", "breaking interface", "deprecate export",
                # Preserve the more specific security/quality diagnostic when a
                # rewrite is also attempting a counterfeit platform object.
                "counterfeit", "unsafe double",
            )):
                return (
                    "Evidence-directed write rejected: corrective rewrite removes established exports "
                    + ", ".join(removed_exports[:16])
                    + "; preserve the public contract or provide an explicit migration."
                )
        is_test_path = "/tests/" in f"/{path}" or path.startswith("tests/") or re.search(
            r"(?:^|/)[^/]+\.(?:test|spec)\.[^/]+$", path,
        ) is not None
        if is_test_path and "inject" in evidence:
            injectable_functions = set(re.findall(
                r"function\s+([a-z_$][\w$]*)\s*\([^)]*(?:factory|adapter|port)",
                evidence,
                flags=re.IGNORECASE,
            ))
            for function_name in injectable_functions:
                calls = re.findall(
                    rf"\b{re.escape(function_name)}\s*\(([^()]*)\)", proposed,
                    flags=re.IGNORECASE,
                )
                if any(call.strip() for call in calls) and any(not call.strip() for call in calls):
                    return (
                        "Evidence-directed write rejected: partial dependency-injection migration; "
                        f"{function_name} still has uninjected call sites in the same regression file."
                    )
        if is_test_path:
            asserted_double_types = set(re.findall(
                r"\bas\s+unknown\s+as\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)",
                proposed,
            ))
            for type_name in asserted_double_types:
                short_name = type_name.rsplit(".", 1)[-1]
                if re.search(
                    rf"\.toBeInstanceOf\(\s*(?:[A-Za-z_$][\w$]*\.)?{re.escape(short_name)}\s*\)",
                    proposed,
                ):
                    return (
                        "Evidence-directed write rejected: a structural test double cannot satisfy an "
                        f"instanceof assertion for {type_name}; assert the injected port behavior instead."
                    )
        rules: list[tuple[bool, str, str]] = [
            (
                any(term in evidence for term in ("unsafe double assertion", "counterfeit")) and not is_test_path,
                r"\bas\s+unknown\s+as\s+(?:THREE\.)?WebGLRenderer\b",
                "validator forbids counterfeit WebGLRenderer assertions; define/inject an honest renderer port",
            ),
            (
                any(term in evidence for term in ("platform mock", "test-environment shim"))
                and not is_test_path,
                r"(?:HTMLCanvasElement|globalThis|window)[^\n]{0,160}\.prototype\.|\b(?:mock|fake|dummy|stub)\w*(?:renderer|context|canvas)\b",
                "validator forbids production test shims; inject the platform dependency and keep doubles in tests",
            ),
            (
                "placeholder" in evidence,
                r"\bplaceholder\b",
                "validator forbids placeholder source; implement observable behavior and invariants",
            ),
        ]
        for enabled, pattern, message in rules:
            if enabled and re.search(pattern, proposed, re.IGNORECASE):
                return f"Evidence-directed write rejected: {message}."
        return ""

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
        repair_paths = self._validator_repair_paths(node)
        calls = self._prioritize_repair_calls(node, calls)
        for call in calls:
            tool_name = str(call.get("tool", ""))
            params = call.get("params") or {}
            if not tool_name:
                continue
            if tool_name == "file_locate" and repair_paths:
                query = str(params.get("query") or "").replace("\\", "/").strip().lower()
                matches = [
                    path for path in repair_paths
                    if path.lower() == query or Path(path).name.lower() == Path(query).name.lower()
                ]
                if query and len(matches) == 1:
                    tool_name = "file_read"
                    params = {"path": matches[0]}
            if self._atomic_workday and tool_name == "file_read":
                requested = str(params.get("path") or "").replace("\\", "/").lower()
                last_mutation_index = max(
                    (index for index, entry in enumerate(node.tool_outputs)
                     if entry.get("tool") in {"file_write", "workspace_scaffold", "environment_bootstrap"}
                     and not (entry.get("result") or {}).get("error")),
                    default=-1,
                )
                duplicate = any(
                    entry.get("tool") == "file_read"
                    and not (entry.get("result") or {}).get("error")
                    and str((entry.get("result") or {}).get("path") or "").replace("\\", "/").lower() == requested
                    for entry in node.tool_outputs[last_mutation_index + 1:]
                )
                if duplicate:
                    result = {"error": "File is already preloaded and unchanged; use its current content to write or validate now."}
                    tool_outputs.append({"tool": tool_name, "result": result})
                    node.add_tool_output(tool_name, result)
                    errors.append(f"{tool_name}: {result['error']}")
                    continue
            if self._atomic_workday and not self._corrective_retry:
                validator_directed = "validator-directed repair paths" in (
                    f"{node.description}\n{node.scientific_form}"
                ).lower()
                navigation_limit = self._atomic_navigation_limit(node)
                successful = [
                    entry for entry in node.tool_outputs
                    if isinstance(entry.get("result"), dict)
                    and not entry["result"].get("error")
                ]
                navigation_count = sum(
                    entry.get("tool") in {"memory_search", "file_locate", "file_read"}
                    for entry in successful
                )
                mutation_seen = any(
                    entry.get("tool") in {"file_write", "workspace_scaffold", "environment_bootstrap"}
                    for entry in successful
                )
                last_mutation = max(
                    (index for index, entry in enumerate(node.tool_outputs)
                     if entry.get("tool") in {"file_write", "workspace_scaffold", "environment_bootstrap"}
                     and not (entry.get("result") or {}).get("error")),
                    default=-1,
                )
                last_executor = max(
                    (index for index, entry in enumerate(node.tool_outputs)
                     if entry.get("tool") == "executor"),
                    default=-1,
                )
                executor_failed_after_mutation = False
                if last_executor > last_mutation:
                    prior_validation = node.tool_outputs[last_executor].get("result") or {}
                    executor_failed_after_mutation = bool(prior_validation.get("error")) or int(
                        prior_validation.get("return_code") or 0
                    ) != 0
                requested_read = str(params.get("path") or "").replace("\\", "/").lower()
                read_is_fresh_validator_path = False
                if executor_failed_after_mutation and tool_name == "file_read" and requested_read:
                    within_surface = any(
                        requested_read == allowed.lower()
                        or requested_read.endswith("/" + allowed.lower())
                        for allowed in repair_paths
                    )
                    read_after_validation = any(
                        index > last_executor
                        and entry.get("tool") == "file_read"
                        and not (entry.get("result") or {}).get("error")
                        and (
                            str((entry.get("result") or {}).get("path") or "").replace("\\", "/").lower()
                            == requested_read
                            or str((entry.get("result") or {}).get("path") or "").replace("\\", "/").lower().endswith("/" + requested_read)
                            or requested_read.endswith("/" + str((entry.get("result") or {}).get("path") or "").replace("\\", "/").lower())
                        )
                        for index, entry in enumerate(node.tool_outputs)
                    )
                    read_is_fresh_validator_path = within_surface and not read_after_validation
                if (
                    validator_directed and mutation_seen and tool_name != "executor"
                    and not (executor_failed_after_mutation and tool_name == "file_write")
                    and not read_is_fresh_validator_path
                ):
                    result = {
                        "error": (
                            "Validator-directed mutation already landed; validate it with executor "
                            "before any further discovery or mutation."
                        )
                    }
                    tool_outputs.append({"tool": tool_name, "result": result})
                    node.add_tool_output(tool_name, result)
                    errors.append(f"{tool_name}: {result['error']}")
                    continue
                if not mutation_seen and navigation_count >= navigation_limit and tool_name not in {
                    "file_write", "workspace_scaffold", "environment_bootstrap", "executor",
                }:
                    result = {
                        "error": (
                            "Atomic navigation budget exhausted. Existing context is sufficient; "
                            "the next call must create concrete project files using file_write, "
                            "workspace_scaffold, or environment_bootstrap."
                        )
                    }
                    tool_outputs.append({"tool": tool_name, "result": result})
                    node.add_tool_output(tool_name, result)
                    errors.append(f"{tool_name}: {result['error']}")
                    continue
                progress_tools = {"file_write", "workspace_scaffold", "environment_bootstrap", "executor"}
                last_progress = max(
                    (index for index, entry in enumerate(successful) if entry.get("tool") in progress_tools),
                    default=-1,
                )
                navigation_since_progress = sum(
                    entry.get("tool") in {"memory_search", "file_locate", "file_read"}
                    for entry in successful[last_progress + 1:]
                )
                if mutation_seen and navigation_since_progress >= 3 and tool_name in {
                    "memory_search", "file_locate", "file_read",
                }:
                    result = {
                        "error": (
                            "Atomic phase navigation budget exhausted. Use the gathered evidence to "
                            "write the next cohesive implementation unit or run executor validation."
                        )
                    }
                    tool_outputs.append({"tool": tool_name, "result": result})
                    node.add_tool_output(tool_name, result)
                    errors.append(f"{tool_name}: {result['error']}")
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
            write_outside_repair_surface = False
            if tool_name == "file_write" and repair_paths:
                requested_path = str(params.get("path") or "").replace("\\", "/").lower()
                try:
                    requested_relative = str(Path(requested_path).resolve().relative_to(self.tools.workdir.resolve())).replace("\\", "/").lower()
                except (AttributeError, ValueError, OSError):
                    requested_relative = requested_path
                write_outside_repair_surface = not any(
                    requested_relative == allowed.lower()
                    or requested_relative.endswith("/" + allowed.lower())
                    for allowed in repair_paths
                )
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
            elif write_outside_repair_surface:
                requested_label = str(params.get("path") or "<missing path>")
                result = {
                    "error": (
                        f"Validator-directed write rejected: attempted path {requested_label!r} "
                        "is outside the declared repair paths: " + ", ".join(repair_paths)
                    )
                }
            elif tool_name == "executor" and self._atomic_workday and (
                executor_mutation_error := self._atomic_executor_mutation_error(params)
            ):
                result = {"error": executor_mutation_error}
            elif tool_name == "file_write" and self._atomic_workday and (
                forbidden := self._evidence_forbidden_write_error(node, params)
            ):
                result = {"error": forbidden}
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
    def _atomic_executor_mutation_error(params: dict) -> str:
        """Keep atomic source/config mutations on the auditable file-write path."""
        command = str(params.get("command") or "")
        forbidden = (
            (r"(?:^|[;&|]\s*)npm(?:\.cmd)?\s+init\b", "npm init"),
            (r"(?:^|[;&|]\s*)(?:Set-Content|Add-Content|Out-File)\b", "PowerShell content writer"),
            (r"(?:^|[;&|]\s*)sed\s+-i\b", "in-place sed edit"),
        )
        for pattern, label in forbidden:
            if re.search(pattern, command, flags=re.IGNORECASE):
                return (
                    f"Atomic executor rejected {label}; use file_write for source/config changes "
                    "so repair-surface, schema, and anti-regression gates remain enforceable."
                )
        return ""

    @staticmethod
    def _validator_repair_paths(node: TaskNode) -> list[str]:
        text = f"{node.description}\n{node.scientific_form}"
        match = re.search(
            r"Validator-directed repair paths:\s*(\[[^\]]*\])",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return []
        try:
            value = json.loads(match.group(1))
        except Exception:
            return []
        paths = [str(item).replace("\\", "/") for item in value if str(item).strip()]
        # A truthful interface repair often moves the next failure into a
        # consumer. Expand the bounded surface only from fresh, failed executor
        # evidence, so the agent can migrate that caller without regaining
        # arbitrary workspace write access.
        latest_failed_validation = ""
        for entry in reversed(node.tool_outputs):
            if entry.get("tool") != "executor" or not isinstance(entry.get("result"), dict):
                continue
            result = entry["result"]
            failed = bool(result.get("error")) or int(result.get("return_code") or 0) != 0
            if failed:
                latest_failed_validation = "\n".join(
                    str(result.get(key) or "") for key in ("stdout", "stderr", "error")
                )
            break
        if latest_failed_validation:
            normalized = latest_failed_validation.replace("\\\\", "/").replace("\\", "/")
            discovered = re.findall(
                r"(?:src|tests|app|lib)/[A-Za-z0-9_./-]+\.(?:ts|tsx|js|jsx|py|rs|go|java|cpp|c|h|php|pl)",
                normalized,
                flags=re.IGNORECASE,
            )
            for path in discovered:
                canonical = path.replace("\\", "/")
                if canonical.lower() not in {item.lower() for item in paths}:
                    paths.append(canonical)
        return paths

    @classmethod
    def _atomic_navigation_limit(cls, node: TaskNode) -> int:
        """Size bounded inspection to the explicit validator contract."""
        repair_paths = cls._validator_repair_paths(node)
        if not repair_paths:
            text = f"{node.description}\n{node.scientific_form}".lower()
            return 3 if "validator-directed repair paths" in text else 4
        return min(8, max(3, len(repair_paths)))

    @classmethod
    def _prioritize_repair_calls(cls, node: TaskNode, calls: list[dict]) -> list[dict]:
        """Stable-sort validator evidence before incidental model discovery."""
        repair_paths = cls._validator_repair_paths(node)
        if not repair_paths:
            return list(calls)

        def priority(call: dict) -> int:
            tool = str(call.get("tool") or "")
            params = call.get("params") or {}
            candidate = str(
                params.get("path") if tool == "file_read" else params.get("query")
                if tool == "file_locate" else ""
            ).replace("\\", "/").strip().lower()
            if not candidate:
                return 1
            matches = [
                path for path in repair_paths
                if path.lower() == candidate
                or Path(path).name.lower() == Path(candidate).name.lower()
            ]
            return 0 if len(matches) == 1 else 1

        return sorted(calls, key=priority)

    def _seed_validator_context(self, node: TaskNode) -> None:
        """Preload exact validator files so the first model turn can act."""
        if not self._atomic_workday:
            return
        paths = self._validator_repair_paths(node)[: self._atomic_navigation_limit(node)]
        if not paths:
            return
        already_read = {
            str((entry.get("result") or {}).get("path") or "").replace("\\", "/").lower()
            for entry in node.tool_outputs
            if entry.get("tool") == "file_read"
        }
        for path in paths:
            normalized = path.replace("\\", "/").lower()
            if normalized in already_read or any(
                item.endswith("/" + normalized) for item in already_read if item
            ):
                continue
            result = self.tools.dispatch("file_read", {"path": path})
            node.add_tool_output("file_read", result)

    @classmethod
    def _validator_read_context(cls, node: TaskNode, *, max_chars: int = 16000) -> str:
        """Return current validator-named files even when generic history is compacted.

        Atomic repair turns may legitimately exhaust their navigation allowance
        before every causal file survives the accumulated-results summary.  A
        corrective prompt that advertises only ``file_write`` while omitting the
        exact current file creates an impossible read-before-write contract.  Keep
        the latest successful read for each bounded repair path in the prompt.
        """
        repair_paths = cls._validator_repair_paths(node)
        if not repair_paths:
            return ""
        latest: dict[str, dict] = {}
        for entry in node.tool_outputs:
            if entry.get("tool") != "file_read" or not isinstance(entry.get("result"), dict):
                continue
            result = entry["result"]
            if result.get("error"):
                continue
            read_path = str(result.get("path") or "").replace("\\", "/").lower()
            for repair_path in repair_paths:
                normalized = repair_path.replace("\\", "/").lower()
                if read_path == normalized or read_path.endswith("/" + normalized):
                    latest[normalized] = result
                    break
        if not latest:
            return ""
        sections: list[str] = []
        remaining = max(1000, int(max_chars))
        for repair_path in repair_paths:
            result = latest.get(repair_path.replace("\\", "/").lower())
            if not result or remaining <= 0:
                continue
            content = str(result.get("content") or "")
            # Preserve complete small files; bounded truncation is explicit so a
            # model will not mistake a fragment for the whole source file.
            allowance = min(6000, remaining)
            clipped = content[:allowance]
            marker = "\n...[validator file truncated]" if len(content) > allowance else ""
            section = f"--- {repair_path} (current validator context) ---\n{clipped}{marker}"
            sections.append(section)
            remaining -= len(section)
        return "\n\n".join(sections)

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
        allowed_names: set[str] | None = None
        if self._atomic_workday and "navigation budget exhausted" in failed_result.error.lower():
            allowed_names = {"file_write", "executor"}
            tool_descriptions = [
                item for item in tool_descriptions if item.get("name") in allowed_names
            ]
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
        validator_context = self._validator_read_context(node)

        if self._needs_web_verification(failed_result.error):
            research = self.tools.dispatch("web_search", {
                "query": f"official documentation {node.description} {failed_result.error[:500]}"
            })
            if not research.get("error"):
                node.add_tool_output("web_search", research)
                accumulated = tree.accumulated_results_summary()

        system = (
            self._control_prefix()
            + " A tool call failed.  Diagnose the issue and provide "
            "corrective tool calls.\n"
            'Return ONLY JSON: {"action":"tool_calls","tool_calls": [{"tool": str, "params": dict}]}. '
            'Return exactly one small, complete corrective call so the JSON cannot be truncated. '
            'Use only registered tool names such as file_read, file_write, or executor; '
            'never emit XML/tool_call tags, markdown, shell-style pseudo-calls, or prose.'
            f"\n\nAvailable tools:\n{tool_desc}"
        )
        prompt = (
            f"Branch: {node.scientific_form or node.description}\n\n"
            f"Failed output:\n{failed_result.output[:2000]}\n\n"
            f"Errors:\n{failed_result.error[:1000]}\n\n"
        )
        if accumulated:
            prompt += f"{accumulated}\n\n"
        if validator_context:
            prompt += f"[Current bounded validator files]\n{validator_context}\n\n"
        prompt += "Provide normalized tool_calls to resolve the issue."

        try:
            raw = self._send(prompt=prompt, stream=False, system=system)
            normalized = self.response_normalizer.normalize_action(raw or "")
            if normalized.valid and normalized.value.get("action") == "tool_calls":
                if normalized.transformations:
                    node.add_tool_output("response_normalizer", {
                        "status": "corrective_normalized",
                        "transformations": normalized.transformations[-20:],
                    })
                fix_calls = normalized.value.get("tool_calls") or []
                if allowed_names is not None:
                    requested = [str(call.get("tool") or "") for call in fix_calls]
                    fix_calls = [call for call in fix_calls if call.get("tool") in allowed_names]
                    if requested and not fix_calls:
                        violation = (
                            "Corrective model violated the advertised tool schema; "
                            f"allowed={sorted(allowed_names)}, requested={requested}."
                        )
                        # Attribute the corrective provider here. The caller
                        # separately records the original failed action, so
                        # ATF can avoid both noncompliant models this turn.
                        self._report_correction(
                            classification="protocol_hallucination",
                            trigger=violation,
                            failed_output=str(raw or "")[:8000],
                            resolved=False,
                            is_hallucination=True,
                            attribution=self._capture_attribution(),
                            metadata={"branch": node.description, "allowed_tools": sorted(allowed_names)},
                        )
                if fix_calls:
                    return self._dispatch_tool_calls(node, fix_calls[:1], tree, attempt=2)
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
            "atomic navigation budget exhausted",
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
        """Backward-compatible boundary normalizer for library consumers.

        Runtime paths use the instance normalizer populated with registered
        tool schemas.  This helper remains schema-neutral because callers have
        historically invoked it directly on the class.
        """
        normalized = ModelResponseNormalizer().normalize_action(payload)
        return normalized.value if normalized.valid else payload
