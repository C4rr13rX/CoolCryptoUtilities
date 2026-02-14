from typing import Dict, List, Any, Optional
import json
import time
from dataclasses import dataclass

@dataclass
class LearningPattern:
    pattern_id: str
    context_features: Dict[str, Any]
    successful_tools: List[str]
    failed_tools: List[str]
    success_rate: float
    confidence: float
    last_updated: float

class ContinuousLearningSystem:
    def __init__(self, persistence_file: str = "learning_patterns.json"):
        self.persistence_file = persistence_file
        self.patterns = {}
        self.execution_history = []
    
    def learn_from_execution(self, context: Dict[str, Any], tool_used: str, success: bool):
        pattern_id = hash(str(context)) % 1000
        
        if pattern_id not in self.patterns:
            self.patterns[pattern_id] = LearningPattern(
                pattern_id=str(pattern_id),
                context_features=context,
                successful_tools=[],
                failed_tools=[],
                success_rate=0.0,
                confidence=0.0,
                last_updated=time.time()
            )
        
        pattern = self.patterns[pattern_id]
        
        if success and tool_used not in pattern.successful_tools:
            pattern.successful_tools.append(tool_used)
        elif not success and tool_used not in pattern.failed_tools:
            pattern.failed_tools.append(tool_used)
        
        total_attempts = len(pattern.successful_tools) + len(pattern.failed_tools)
        pattern.success_rate = len(pattern.successful_tools) / max(total_attempts, 1)
        pattern.confidence = min(total_attempts / 10.0, 1.0)
        pattern.last_updated = time.time()
