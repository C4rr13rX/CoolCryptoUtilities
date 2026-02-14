from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import time

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
        self.persistence_file = Path(persistence_file)
        self.patterns: Dict[str, LearningPattern] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self._load_patterns()

    def _context_signature(self, context: Dict[str, Any]) -> str:
        try:
            return json.dumps(context, sort_keys=True, default=str)
        except Exception:
            return str(context)

    def _pattern_key(self, context: Dict[str, Any]) -> str:
        signature = self._context_signature(context)
        return str(abs(hash(signature)) % 10000)

    def _load_patterns(self) -> None:
        if not self.persistence_file.exists():
            return
        try:
            payload = json.loads(self.persistence_file.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            try:
                self.patterns[key] = LearningPattern(**value)
            except Exception:
                continue

    def _save_patterns(self) -> None:
        try:
            payload = {key: pattern.__dict__ for key, pattern in self.patterns.items()}
            self.persistence_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            return
    
    def learn_from_execution(self, context: Dict[str, Any], tool_used: str, success: bool):
        pattern_id = self._pattern_key(context)
        
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
        self.execution_history.append(
            {"context": context, "tool": tool_used, "success": success, "timestamp": pattern.last_updated}
        )
        self._save_patterns()

    def suggest_tools(self, context: Dict[str, Any]) -> Dict[str, float]:
        pattern_id = self._pattern_key(context)
        pattern = self.patterns.get(pattern_id)
        if not pattern:
            return {}
        bonus = 0.05 + 0.15 * pattern.success_rate * pattern.confidence
        suggestions: Dict[str, float] = {}
        for tool in pattern.successful_tools:
            suggestions[tool] = max(suggestions.get(tool, 0.0), min(0.2, bonus))
        for tool in pattern.failed_tools:
            suggestions[tool] = min(suggestions.get(tool, 0.0), -0.1)
        return suggestions
