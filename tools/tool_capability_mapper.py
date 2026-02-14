from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging
import time


@dataclass
class ToolCapability:
    tool_name: str
    task_type: str
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    total_executions: int = 0
    last_used: float = 0.0
    capabilities: List[str] = field(default_factory=list)


class ToolCapabilityMapper:
    def __init__(self, data_file: str = "tool_capabilities.json"):
        self.data_file = Path(data_file)
        self.capabilities: Dict[str, ToolCapability] = {}
        self.logger = logging.getLogger(__name__)
        self._load_capabilities()

        self.system_tools = {
            "python": ["script_execution", "data_processing", "file_operations", "testing"],
            "git": ["version_control", "repository_management", "branch_operations"],
            "rg": ["text_search", "pattern_matching", "file_content_analysis"],
        }

        self.meta_tools = {
            "datalab": ["data_analysis", "news_retrieval", "web_queries", "table_operations"],
            "wallet": ["crypto_operations", "transaction_management", "balance_queries"],
            "vm": ["virtualization", "system_automation", "screenshot_capture", "remote_execution"],
        }

        self.shell_tools = {
            "powershell": ["windows_administration", "file_operations", "system_management"],
            "bash": ["unix_operations", "scripting", "text_processing"],
            "cmd": ["basic_windows_operations", "batch_processing"],
        }

        self._initialize_base_capabilities()

    def _load_capabilities(self) -> None:
        if not self.data_file.exists():
            return
        try:
            data = json.loads(self.data_file.read_text(encoding="utf-8"))
        except Exception as exc:
            self.logger.warning("Failed to load capabilities: %s", exc)
            return
        if not isinstance(data, dict):
            return
        for key, cap_data in data.items():
            if not isinstance(cap_data, dict):
                continue
            try:
                self.capabilities[key] = ToolCapability(**cap_data)
            except Exception:
                continue

    def _save_capabilities(self) -> None:
        try:
            data = {key: asdict(cap) for key, cap in self.capabilities.items()}
            self.data_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            self.logger.error("Failed to save capabilities: %s", exc)

    def _initialize_base_capabilities(self) -> None:
        all_tools = {**self.system_tools, **self.meta_tools, **self.shell_tools}
        for tool_name, task_types in all_tools.items():
            for task_type in task_types:
                key = f"{tool_name}:{task_type}"
                if key not in self.capabilities:
                    self.capabilities[key] = ToolCapability(
                        tool_name=tool_name,
                        task_type=task_type,
                        success_rate=0.5,
                        capabilities=list(task_types),
                    )

    def get_tool_effectiveness(self, tool_name: str, task_type: str) -> float:
        key = f"{tool_name}:{task_type}"
        if key not in self.capabilities:
            return 0.5

        cap = self.capabilities[key]
        if cap.total_executions == 0:
            return max(0.1, cap.success_rate or 0.5)

        success_score = cap.success_rate
        speed_score = max(0.1, 1.0 / (1.0 + cap.avg_execution_time / 10.0))
        recency_score = max(0.5, 1.0 / (1.0 + (time.time() - cap.last_used) / 86400))

        effectiveness = success_score * 0.6 + speed_score * 0.3 + recency_score * 0.1
        return min(1.0, max(0.0, effectiveness))

    def update_effectiveness(
        self,
        tool_name: str,
        task_type: str,
        success: bool,
        execution_time: float,
        validation_score: float | None = None,
    ) -> None:
        key = f"{tool_name}:{task_type}"
        if key not in self.capabilities:
            self._initialize_base_capabilities()
        cap = self.capabilities[key]

        cap.total_executions += 1
        cap.last_used = time.time()

        score = 1.0 if success else 0.0
        if validation_score is not None:
            score = max(0.0, min(1.0, float(validation_score)))
            if not success:
                score = 0.0

        if cap.total_executions == 1:
            cap.success_rate = score
            cap.avg_execution_time = execution_time
        else:
            cap.success_rate = ((cap.success_rate * (cap.total_executions - 1)) + score) / cap.total_executions
            cap.avg_execution_time = (
                (cap.avg_execution_time * (cap.total_executions - 1)) + execution_time
            ) / cap.total_executions

        self._save_capabilities()

    def record_execution(self, tool_name: str, task_type: str, success: bool, execution_time: float) -> None:
        self.update_effectiveness(
            tool_name=tool_name,
            task_type=task_type,
            success=success,
            execution_time=execution_time,
            validation_score=1.0 if success else 0.0,
        )

    def get_best_tools_for_task(self, task_type: str, limit: int = 3) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = {}
        for cap in self.capabilities.values():
            if cap.task_type == task_type or task_type in cap.capabilities:
                effectiveness = self.get_tool_effectiveness(cap.tool_name, cap.task_type)
                scores[cap.tool_name] = max(scores.get(cap.tool_name, 0.0), effectiveness)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:limit]

    def analyze_task_requirements(self, command: str) -> List[str]:
        task_types: List[str] = []
        command_lower = command.lower()

        if any(word in command_lower for word in ["file", "directory", "folder", "copy", "move", "delete"]):
            task_types.append("file_operations")

        if any(word in command_lower for word in ["search", "find", "grep", "pattern", "text"]):
            task_types.append("text_search")

        if any(word in command_lower for word in ["data", "json", "csv", "parse", "process"]):
            task_types.append("data_processing")

        if any(word in command_lower for word in ["system", "process", "service", "registry"]):
            task_types.append("system_management")

        if any(word in command_lower for word in ["git", "commit", "branch", "repository"]):
            task_types.append("version_control")

        if any(word in command_lower for word in ["test", "verify", "check"]):
            task_types.append("testing")

        if not task_types:
            task_types.append("script_execution")

        return task_types
