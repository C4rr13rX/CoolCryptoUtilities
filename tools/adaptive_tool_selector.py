from typing import Dict, List, Any, Optional, Tuple
import re
import logging
from tool_capability_mapper import ToolCapabilityMapper, ToolCapability

class AdaptiveToolSelector:
    def __init__(self, capability_mapper: ToolCapabilityMapper):
        self.capability_mapper = capability_mapper
        self.logger = logging.getLogger(__name__)
    
    def select_tools_for_command(self, command: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if context is None:
            context = {}
        
        task_types = self.capability_mapper.analyze_task_requirements(command)
        tool_recommendations = []
        
        for task_type in task_types:
            best_tools = self.capability_mapper.get_best_tools_for_task(task_type, limit=3)
            
            for tool_name, effectiveness in best_tools:
                adjusted_effectiveness = self._apply_context_adjustments(
                    tool_name, effectiveness, context, command
                )
                
                tool_recommendations.append({
                    "tool_name": tool_name,
                    "task_type": task_type,
                    "effectiveness": adjusted_effectiveness,
                    "original_effectiveness": effectiveness,
                    "reason": self._get_selection_reason(tool_name, task_type, command)
                })
        
        unique_tools = {}
        for rec in tool_recommendations:
            key = rec["tool_name"]
            if key not in unique_tools or rec["effectiveness"] > unique_tools[key]["effectiveness"]:
                unique_tools[key] = rec
        
        sorted_recommendations = sorted(
            unique_tools.values(),
            key=lambda x: x["effectiveness"],
            reverse=True
        )
        
        return sorted_recommendations
    
    def _apply_context_adjustments(self, tool_name: str, effectiveness: float, 
                                 context: Dict[str, Any], command: str) -> float:
        adjusted = effectiveness
        
        platform = context.get("platform", "win32")
        if platform == "win32":
            if tool_name == "powershell":
                adjusted *= 1.2
            elif tool_name == "bash":
                adjusted *= 0.8
        
        command_complexity = len(command.split()) / 10.0
        if tool_name == "python" and command_complexity > 0.5:
            adjusted *= 1.1
        
        if context.get("network_available", True):
            if tool_name in ["datalab", "wallet"]:
                adjusted *= 1.1
        else:
            if tool_name in ["datalab", "wallet"]:
                adjusted *= 0.3
        
        if context.get("is_admin", False):
            if tool_name in ["powershell", "cmd"]:
                adjusted *= 1.1
        
        return min(1.0, max(0.0, adjusted))
    
    def _get_selection_reason(self, tool_name: str, task_type: str, command: str) -> str:
        reasons = {
            "python": "Excellent for data processing and complex logic",
            "powershell": "Native Windows administration and file operations",
            "git": "Version control and repository management",
            "rg": "Fast text search and pattern matching",
            "datalab": "Data analysis and web queries",
            "wallet": "Cryptocurrency operations",
            "vm": "Virtualization and system automation",
            "bash": "Unix-style scripting and text processing",
            "cmd": "Basic Windows command operations"
        }
        
        base_reason = reasons.get(tool_name, f"Suitable for {task_type}")
        
        if "test" in command.lower() and tool_name == "python":
            return f"{base_reason} - ideal for testing workflows"
        elif "search" in command.lower() and tool_name == "rg":
            return f"{base_reason} - optimized for fast search operations"
        
        return base_reason
ons"
        
        return base_reason
    
    def get_fallback_tools(self, primary_tool: str, task_type: str) -> List[str]:
        fallback_map = {
            "python": ["powershell", "bash"],
            "powershell": ["cmd", "python"],
            "bash": ["powershell", "python"],
            "git": ["powershell", "python"],
            "rg": ["powershell", "python"],
            "datalab": ["python", "powershell"],
            "wallet": ["python", "powershell"],
            "vm": ["powershell", "python"]
        }
        
        return fallback_map.get(primary_tool, ["python", "powershell"])
        return base_reason
    
    def get_fallback_tools(self, primary_tool: str, task_type: str) -> List[str]:
        fallback_map = {
            "python": ["powershell", "bash"],
            "powershell": ["cmd", "python"],
            "bash": ["powershell", "python"],
            "git": ["powershell", "python"],
            "rg": ["powershell", "python"],
            "datalab": ["python", "powershell"],
            "wallet": ["python", "powershell"],
            "vm": ["powershell", "python"]
        }
        
        return fallback_map.get(primary_tool, ["python", "powershell"])
        return base_reason
    
    def get_fallback_tools(self, primary_tool: str, task_type: str) -> List[str]:
        fallback_map = {
            "python": ["powershell", "bash"],
            "powershell": ["cmd", "python"],
            "bash": ["powershell", "python"],
            "git": ["powershell", "python"],
            "rg": ["powershell", "python"],
            "datalab": ["python", "powershell"],
            "wallet": ["python", "powershell"],
            "vm": ["powershell", "python"]
        }
        
        return fallback_map.get(primary_tool, ["python", "powershell"])
