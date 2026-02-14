from typing import Dict, List, Any, Optional, Tuple
import re
import json
from dataclasses import dataclass
from enum import Enum

class DecompositionStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    ITERATIVE = "iterative"

@dataclass
class CommandStep:
    id: str
    command: str
    dependencies: List[str]
    expected_output: str
    validation_criteria: List[str]
    estimated_time: float
    priority: int

class AdaptiveCommandDecomposer:
    def __init__(self):
        self.strategy_history = {}
        self.complexity_patterns = {
            "simple": r"^[a-zA-Z0-9_\-\.\s]{1,50}$",
            "medium": r".*[;&|].*",
            "complex": r".*(for|while|if|function|class).*"
        }
    
    def decompose_command(self, command: str, context: Dict[str, Any]) -> Tuple[List[CommandStep], DecompositionStrategy]:
        complexity = context.get("complexity") or self._assess_complexity(command)
        strategy_hint = context.get("strategy")
        if strategy_hint:
            try:
                strategy = DecompositionStrategy(strategy_hint)
            except Exception:
                strategy = self._select_strategy(command, complexity, context)
        else:
            strategy = self._select_strategy(command, complexity, context)
        
        if complexity == "simple":
            return self._simple_decomposition(command), strategy
        elif complexity == "medium":
            return self._medium_decomposition(command, strategy), strategy
        else:
            return self._complex_decomposition(command, strategy), strategy
    
    def _assess_complexity(self, command: str) -> str:
        if re.match(self.complexity_patterns["complex"], command):
            return "complex"
        elif re.match(self.complexity_patterns["medium"], command):
            return "medium"
        else:
            return "simple"
    
    def _select_strategy(self, command: str, complexity: str, context: Dict[str, Any]) -> DecompositionStrategy:
        command_hash = hash(command) % 1000
        
        if command_hash in self.strategy_history:
            best_strategy = max(self.strategy_history[command_hash].items(), key=lambda x: x[1])
            return DecompositionStrategy(best_strategy[0])
        
        if complexity == "simple":
            return DecompositionStrategy.SEQUENTIAL
        elif complexity == "medium":
            return DecompositionStrategy.PARALLEL
        else:
            return DecompositionStrategy.HIERARCHICAL

    def record_outcome(self, command: str, strategy: DecompositionStrategy, score: float) -> None:
        command_hash = hash(command) % 1000
        history = self.strategy_history.setdefault(command_hash, {})
        prior = history.get(strategy.value, 0.0)
        blended = prior * 0.7 + max(0.0, min(1.0, score)) * 0.3
        history[strategy.value] = blended
    
    def _simple_decomposition(self, command: str) -> List[CommandStep]:
        return [CommandStep(
            id="step_1",
            command=command,
            dependencies=[],
            expected_output="Command executed successfully",
            validation_criteria=["no_errors", "output_present"],
            estimated_time=1.0,
            priority=1
        )]
    
    def _medium_decomposition(self, command: str, strategy: DecompositionStrategy) -> List[CommandStep]:
        parts = re.split(r"[;&|]", command)
        steps = []
        
        for i, part in enumerate(parts):
            step = CommandStep(
                id=f"step_{i+1}",
                command=part.strip(),
                dependencies=[f"step_{i}"] if i > 0 and strategy == DecompositionStrategy.SEQUENTIAL else [],
                expected_output=f"Part {i+1} completed",
                validation_criteria=["no_errors"],
                estimated_time=1.0,
                priority=i+1
            )
            steps.append(step)
        
        return steps
