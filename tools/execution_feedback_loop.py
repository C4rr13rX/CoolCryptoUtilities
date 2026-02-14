from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Callable
import logging
import time

from adaptive_tool_selector import AdaptiveToolSelector
from continuous_learning_system import ContinuousLearningSystem
from tool_capability_mapper import ToolCapabilityMapper


@dataclass
class ExecutionResult:
    success: bool
    output: str
    error: str
    execution_time: float
    tool_used: str
    task_type: str
    validation_score: float
    artifacts_created: List[str]


class ExecutionFeedbackLoop:
    def __init__(
        self,
        capability_mapper: ToolCapabilityMapper,
        tool_selector: AdaptiveToolSelector,
        learning_system: ContinuousLearningSystem | None = None,
    ):
        self.capability_mapper = capability_mapper
        self.tool_selector = tool_selector
        self.learning_system = learning_system
        self.logger = logging.getLogger(__name__)

    def execute_with_feedback(self, task: str, context: Dict[str, Any] | None = None) -> ExecutionResult:
        context = context or {}
        task_types = self.capability_mapper.analyze_task_requirements(task)
        task_type = task_types[0] if task_types else self._classify_task(task)
        ranked_tools = self.tool_selector.select_tools_for_command(task, context)

        if not ranked_tools:
            return ExecutionResult(
                success=False,
                output="",
                error="No tools available",
                execution_time=0.0,
                tool_used="none",
                task_type=task_type,
                validation_score=0.0,
                artifacts_created=[],
            )

        executor: Callable[[str, str, Dict[str, Any]], Dict[str, Any]] = context.get("execute_tool") or self._execute_tool
        validator: Callable[[Dict[str, Any], str, Dict[str, Any]], float] = context.get("validate_result") or self._validate_result
        min_score = float(context.get("min_validation_score", 0.7))
        max_attempts = int(context.get("max_attempts", len(ranked_tools)) or len(ranked_tools))

        for tool_info in ranked_tools[:max_attempts]:
            tool_name = tool_info.get("tool_name") or tool_info.get("tool") or "unknown"
            start_time = time.time()
            try:
                result = executor(tool_name, task, context)
                execution_time = time.time() - start_time
                validation_score = validator(result, task_type, context)
                error = str(result.get("error") or "")
                success = not error and validation_score >= 0.0

                exec_result = ExecutionResult(
                    success=success,
                    output=str(result.get("output") or ""),
                    error=error,
                    execution_time=execution_time,
                    tool_used=tool_name,
                    task_type=task_type,
                    validation_score=validation_score,
                    artifacts_created=list(result.get("artifacts") or []),
                )

                self._record_feedback(exec_result, context)

                if exec_result.success and validation_score >= min_score:
                    return exec_result

            except Exception as exc:
                execution_time = time.time() - start_time
                exec_result = ExecutionResult(
                    success=False,
                    output="",
                    error=str(exc),
                    execution_time=execution_time,
                    tool_used=tool_name,
                    task_type=task_type,
                    validation_score=0.0,
                    artifacts_created=[],
                )
                self._record_feedback(exec_result, context)
                continue

        return ExecutionResult(
            success=False,
            output="",
            error="All tools failed",
            execution_time=0.0,
            tool_used="none",
            task_type=task_type,
            validation_score=0.0,
            artifacts_created=[],
        )

    def _classify_task(self, task: str) -> str:
        task_lower = task.lower()

        if any(word in task_lower for word in ["file", "create", "write", "read"]):
            return "file_operations"
        if any(word in task_lower for word in ["git", "commit", "push", "pull"]):
            return "version_control"
        if any(word in task_lower for word in ["search", "find", "grep"]):
            return "text_search"
        if any(word in task_lower for word in ["test", "verify", "check"]):
            return "testing"
        if any(word in task_lower for word in ["install", "package", "dependency"]):
            return "package_management"
        return "general"

    def _execute_tool(self, tool_name: str, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": f"Executed {task} with {tool_name}", "artifacts": []}

    def _validate_result(self, result: Dict[str, Any], task_type: str, context: Dict[str, Any] | None = None) -> float:
        score = 0.0
        if result.get("output"):
            score += 0.3
        if not result.get("error"):
            score += 0.3
        if result.get("artifacts"):
            score += 0.2

        output_text = str(result.get("output") or "").lower()
        if task_type == "file_operations" and "created" in output_text:
            score += 0.2
        elif task_type == "testing" and "passed" in output_text:
            score += 0.2
        else:
            score += 0.1

        return min(score, 1.0)

    def _record_feedback(self, result: ExecutionResult, context: Dict[str, Any] | None = None) -> None:
        self.capability_mapper.update_effectiveness(
            result.tool_used,
            result.task_type,
            result.success,
            result.execution_time,
            result.validation_score,
        )
        if self.learning_system:
            self.learning_system.learn_from_execution(context or {}, result.tool_used, result.success)
