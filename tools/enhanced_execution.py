from __future__ import annotations

from typing import Any, Dict

from adaptive_tool_selector import AdaptiveToolSelector
from continuous_learning_system import ContinuousLearningSystem
from execution_feedback_loop import ExecutionFeedbackLoop, ExecutionResult
from tool_capability_mapper import ToolCapabilityMapper


class EnhancedExecutionEngine:
    def __init__(
        self,
        capability_file: str = "tool_capabilities.json",
        learning_file: str = "learning_patterns.json",
    ):
        self.capability_mapper = ToolCapabilityMapper(capability_file)
        self.learning_system = ContinuousLearningSystem(learning_file)
        self.tool_selector = AdaptiveToolSelector(self.capability_mapper, self.learning_system)
        self.feedback_loop = ExecutionFeedbackLoop(
            self.capability_mapper,
            self.tool_selector,
            self.learning_system,
        )

    def build_context(self, base: Dict[str, Any] | None = None, **overrides: Any) -> Dict[str, Any]:
        context = dict(base or {})
        context.update(overrides)
        return context

    def run_task(self, task: str, context: Dict[str, Any] | None = None) -> ExecutionResult:
        return self.feedback_loop.execute_with_feedback(task, context or {})
