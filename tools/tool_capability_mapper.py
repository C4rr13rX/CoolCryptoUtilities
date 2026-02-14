from typing import Dict, List, Any, Optional, Tuple
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

@dataclass
class ToolCapability:
    tool_name: str
    task_type: str
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    total_executions: int = 0
    last_used: float = 0.0
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []

class ToolCapabilityMapper:
    def __init__(self, data_file: str = "tool_capabilities.json"):
        self.data_file = Path(data_file)
        self.capabilities: Dict[str, ToolCapability] = {}
        self.logger = logging.getLogger(__name__)
        self._load_capabilities()
        
        self.system_tools = {
            "python": ["script_execution", "data_processing", "file_manipulation", "testing"],
            "git": ["version_control", "repository_management", "branch_operations"],
            "rg": ["text_search", "pattern_matching", "file_content_analysis"]
        }
        
        self.meta_tools = {
            "datalab": ["data_analysis", "news_retrieval", "web_queries", "table_operations"],
            "wallet": ["crypto_operations", "transaction_management", "balance_queries"],
            "vm": ["virtualization", "system_automation", "screenshot_capture", "remote_execution"]
        }
        
        self.shell_tools = {
            "powershell": ["windows_administration", "file_operations", "system_management"],
            "bash": ["unix_operations", "scripting", "text_processing"],
            "cmd": ["basic_windows_operations", "batch_processing"]
        }
        
        self._initialize_base_capabilities()
    
    def _load_capabilities(self):
        if self.data_file.exists():
            try:
                with open(self.data_file, "r") as f:
                    data = json.load(f)
                    for key, cap_data in data.items():
                        self.capabilities[key] = ToolCapability(**cap_data)
            except Exception as e:
                self.logger.warning(f"Failed to load capabilities: {e}")
    
    def _save_capabilities(self):
        try:
            data = {key: asdict(cap) for key, cap in self.capabilities.items()}
            with open(self.data_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save capabilities: {e}")
    
    def _initialize_base_capabilities(self):
        all_tools = {**self.system_tools, **self.meta_tools, **self.shell_tools}
        for tool_name, task_types in all_tools.items():
            for task_type in task_types:
                key = f"{tool_name}:{task_type}"
                if key not in self.capabilities:
                    self.capabilities[key] = ToolCapability(
                        tool_name=tool_name,
                        task_type=task_type,
                        success_rate=0.5,
                        capabilities=task_types
                    )
    
    def get_tool_effectiveness(self, tool_name: str, task_type: str) -> float:
        key = f"{tool_name}:{task_type}"
        if key not in self.capabilities:
            return 0.5
        
        cap = self.capabilities[key]
        if cap.total_executions == 0:
            return 0.5
        
        success_score = cap.success_rate
        speed_score = max(0.1, 1.0 / (1.0 + cap.avg_execution_time / 10.0))
        recency_score = max(0.5, 1.0 / (1.0 + (time.time() - cap.last_used) / 86400))
        
        effectiveness = (success_score * 0.6 + speed_score * 0.3 + recency_score * 0.1)
        return min(1.0, max(0.0, effectiveness))
    
    def record_execution(self, tool_name: str, task_type: str, success: bool, execution_time: float):
        key = f"{tool_name}:{task_type}"
        if key not in self.capabilities:
            self._initialize_base_capabilities()
        
        cap = self.capabilities[key]
        cap.total_executions += 1
        cap.last_used = time.time()
        
        if cap.total_executions == 1:
            cap.success_rate = 1.0 if success else 0.0
            cap.avg_execution_time = execution_time
        else:
            cap.success_rate = ((cap.success_rate * (cap.total_executions - 1)) + (1.0 if success else 0.0)) / cap.total_executions
            cap.avg_execution_time = ((cap.avg_execution_time * (cap.total_executions - 1)) + execution_time) / cap.total_executions
        
        self._save_capabilities()
    
    def get_best_tools_for_task(self, task_type: str, limit: int = 3) -> List[Tuple[str, float]]:
        candidates = []
        for key, cap in self.capabilities.items():
            if cap.task_type == task_type or task_type in cap.capabilities:
                effectiveness = self.get_tool_effectiveness(cap.tool_name, cap.task_type)
                candidates.append((cap.tool_name, effectiveness))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:limit]
        if key not in self.capabilities:
            self.capabilities[key] = ToolCapability(
                tool_name=tool_name,
                task_type=task_type,
                success_rate=0.5 if success else 0.0,
                avg_execution_time=execution_time,
                total_executions=1,
                last_used=time.time()
            )
        else:
            cap = self.capabilities[key]
            old_total = cap.total_executions
            new_total = old_total + 1
            
            # Update success rate
            old_success_count = cap.success_rate * old_total
            new_success_count = old_success_count + (1 if success else 0)
            cap.success_rate = new_success_count / new_total
            
            # Update average execution time
            cap.avg_execution_time = ((cap.avg_execution_time * old_total) + execution_time) / new_total
            cap.total_executions = new_total
            cap.last_used = time.time()
        
        self._save_capabilities()
    
    def get_best_tools_for_task(self, task_type: str, limit: int = 3) -> List[Tuple[str, float]]:
        tool_scores = []
        
        for tool_name in list(self.system_tools.keys()) + list(self.meta_tools.keys()) + list(self.shell_tools.keys()):
            effectiveness = self.get_tool_effectiveness(tool_name, task_type)
            tool_scores.append((tool_name, effectiveness))
        
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        return tool_scores[:limit]
    
    def analyze_task_requirements(self, command: str) -> List[str]:
        task_types = []
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
        
        if not task_types:
            task_types.append("script_execution")
        
        return task_types
