from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional


WorkflowRunner = Callable[[Dict[str, Any]], Any]
_WORKFLOWS: Dict[str, WorkflowRunner] = {}


def register_workflow(kind: str, runner: WorkflowRunner) -> None:
    """Register a reusable continuous-refinement workflow implementation."""
    normalized = str(kind or "").strip().lower()
    if not normalized:
        raise ValueError("workflow kind is required")
    _WORKFLOWS[normalized] = runner


def _load_builtin_workflows() -> None:
    if "digital_product_business" not in _WORKFLOWS:
        from services.branddozer_product_loop import run_product_loop_cycle

        register_workflow(
            "digital_product_business",
            lambda project: run_product_loop_cycle(
                root=Path(project.get("root_path") or "."), max_cycles=1
            ),
        )


def run_project_workflow(project: Dict[str, Any]) -> Optional[Any]:
    """Run the configured workflow, or return None for the generic prompt loop."""
    kind = str(project.get("workflow_kind") or "").strip().lower()
    if not kind:
        return None
    _load_builtin_workflows()
    runner = _WORKFLOWS.get(kind)
    if runner is None:
        raise RuntimeError(f"Unknown BrandDozer workflow: {kind}")
    return runner(project)
