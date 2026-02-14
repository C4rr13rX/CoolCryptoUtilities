from typing import Dict, List, Any, Optional, Tuple, Callable
import time
import json
import logging
from dataclasses import dataclass
from tool_capability_mapper import ToolCapabilityMapper
from adaptive_tool_selector import AdaptiveToolSelector

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
    def __init__(self, capability_mapper: ToolCapabilityMapper, tool_selector: AdaptiveToolSelector):
        self.capability_mapper = capability_mapper
        self.tool_selector = tool_selector
        self.logger = logging.getLogger(__name__)
    
    def execute_with_feedback(self, task: str, context: Dict[str, Any]) -> ExecutionResult:
        task_type = self._classify_task(task)
        ranked_tools = self.tool_selector.select_tools(task, context)
        
        for tool_info in ranked_tools:
            tool_name = tool_info["tool"]
            start_time = time.time()
            
            try:
                result = self._execute_tool(tool_name, task, context)
                execution_time = time.time() - start_time
                
                validation_score = self._validate_result(result, task_type)
                
                exec_result = ExecutionResult(
                    success=True,
                    output=result.get("output", ""),
                    error=result.get("error", ""),
                    execution_time=execution_time,
                    tool_used=tool_name,
                    task_type=task_type,
                    validation_score=validation_score,
                    artifacts_created=result.get("artifacts", [])
                )
                
                self._record_feedback(exec_result)
                
                if validation_score > 0.7:
                    return exec_result
                    
            except Exception as e:
                execution_time = time.time() - start_time
                exec_result = ExecutionResult(
                    success=False,
                    output="",
                    error=str(e),
                    execution_time=execution_time,
                    tool_used=tool_name,
                    task_type=task_type,
                    validation_score=0.0,
                    artifacts_created=[]
                )
                
                self._record_feedback(exec_result)
                continue
        
        return ExecutionResult(
            success=False,
            output="",
            error="All tools failed",
            execution_time=0.0,
            tool_used="none",
            task_type=task_type,
            validation_score=0.0,
            artifacts_created=[]
        )
    
    def _classify_task(self, task: str) -> str:
        task_lower = task.lower()
        
        if any(word in task_lower for word in ["file", "create", "write", "read"]):
            return "file_operation"
        elif any(word in task_lower for word in ["git", "commit", "push", "pull"]):
            return "version_control"
        elif any(word in task_lower for word in ["search", "find", "grep"]):
            return "search"
        elif any(word in task_lower for word in ["test", "verify", "check"]):
            return "testing"
        elif any(word in task_lower for word in ["install", "package", "dependency"]):
            return "package_management"
        else:
            return "general"
    
    def _execute_tool(self, tool_name: str, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": f"Executed {task} with {tool_name}", "artifacts": []}
    
    def _validate_result(self, result: Dict[str, Any], task_type: str) -> float:
        score = 0.0
        
        if result.get("output"):
            score += 0.3
        
        if not result.get("error"):
            score += 0.3
        
        if result.get("artifacts"):
            score += 0.2
        
        if task_type == "file_operation" and "created" in result.get("output", "").lower():
            score += 0.2
        elif task_type == "testing" and "passed" in result.get("output", "").lower():
            score += 0.2
        else:
            score += 0.1
        
        return min(score, 1.0)
    
    def _record_feedback(self, result: ExecutionResult):
        self.capability_mapper.update_effectiveness(
            result.tool_used,
            result.task_type,
            result.success,
            result.execution_time,
            result.validation_score
        )
