from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Petal:
    """
    A dynamic step injected into the process flow.

    Petals wilt (are permanently removed) when the system determines
    they are ineffective — e.g., repeated failures or a signal from the
    orchestrator that the work is not producing useful output.
    """

    name: str
    fn: Callable[..., Any]
    priority: int = 0
    max_failures: int = 3
    _failure_count: int = field(default=0, init=False, repr=False)
    _wilted: bool = field(default=False, init=False, repr=False)
    _created_at: float = field(default_factory=time.time, init=False, repr=False)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        if self._wilted:
            return None
        try:
            return self.fn(*args, **kwargs)
        except Exception:
            self._failure_count += 1
            if self._failure_count >= self.max_failures:
                self.wilt()
            return None

    def wilt(self) -> None:
        """Mark this petal as ineffective; it will be pruned on next cycle."""
        self._wilted = True

    @property
    def is_wilted(self) -> bool:
        return self._wilted


class PetalManager:
    """
    Manages the set of active petals in the process flow.

    Petals are sorted by priority (descending) and pruned automatically
    when they wilt.  The orchestrator calls evaluate_effectiveness() after
    each step batch to let the system decide which petals to drop.
    """

    def __init__(self) -> None:
        self._petals: list[Petal] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def add(self, petal: Petal) -> None:
        self._petals.append(petal)
        self._petals.sort(key=lambda p: p.priority, reverse=True)

    def remove(self, name: str) -> None:
        self._petals = [p for p in self._petals if p.name != name]

    def prune_wilted(self) -> list[str]:
        """Remove wilted petals and return their names."""
        removed = [p.name for p in self._petals if p.is_wilted]
        self._petals = [p for p in self._petals if not p.is_wilted]
        return removed

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_all(self, *args: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        """Run all active petals in priority order; returns (name, result) pairs."""
        results: list[tuple[str, Any]] = []
        for petal in self.active_petals():
            result = petal.execute(*args, **kwargs)
            results.append((petal.name, result))
        self.prune_wilted()
        return results

    # ------------------------------------------------------------------
    # Effectiveness evaluation
    # ------------------------------------------------------------------

    def evaluate_effectiveness(self, step_results: list[dict]) -> None:
        """
        Review step results and wilt petals that are consistently failing.

        step_results: list of {"petal": name, "success": bool}
        """
        fail_counts: dict[str, int] = {}
        for result in step_results:
            name = result.get("petal", "")
            if not result.get("success"):
                fail_counts[name] = fail_counts.get(name, 0) + 1
        for petal in self._petals:
            if fail_counts.get(petal.name, 0) >= petal.max_failures:
                petal.wilt()
        self.prune_wilted()

    def update_from_directives(self, directives: list[dict]) -> None:
        """Apply petal add/remove directives extracted from a user prompt."""
        for directive in directives:
            name = directive.get("name", "")
            action = directive.get("action", "")
            if action == "add" and name:
                petal = Petal(
                    name=name,
                    fn=lambda: None,
                    priority=int(directive.get("priority", 0)),
                )
                self.add(petal)
            elif action in {"remove", "wilt"} and name:
                for p in self._petals:
                    if p.name == name:
                        p.wilt()
                self.prune_wilted()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def active_petals(self) -> list[Petal]:
        return [p for p in self._petals if not p.is_wilted]

    def __len__(self) -> int:
        return len(self.active_petals())

    def __repr__(self) -> str:
        names = [p.name for p in self.active_petals()]
        return f"PetalManager(active={names})"
