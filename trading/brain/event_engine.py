from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class ReflexRule:
    name: str
    condition: Callable[[Dict[str, float]], bool]
    action: Callable[[Dict[str, float]], None]
    cooldown: float = 30.0
    last_trigger: float = 0.0


class EventEngine:
    """
    Reflex wiring: when high-risk events occur, the engine activates actions immediately.

    The engine is deliberately simple so it can run on every market tick without adding
    noticeable latency.
    """

    def __init__(self) -> None:
        self.rules: List[ReflexRule] = []

    def register(self, rule: ReflexRule) -> None:
        self.rules.append(rule)

    def process(self, context: Dict[str, float], now: float) -> List[str]:
        triggered: List[str] = []
        for rule in self.rules:
            if now - rule.last_trigger < rule.cooldown:
                continue
            try:
                if rule.condition(context):
                    context.setdefault("cooldown", rule.cooldown)
                    context["reflex_rule"] = rule.name
                    rule.action(context)
                    rule.last_trigger = now
                    triggered.append(rule.name)
                    context.pop("reflex_rule", None)
            except Exception:
                continue
        return triggered


def make_default_engine(block_trade_cb: Callable[[Dict[str, float]], None]) -> EventEngine:
    engine = EventEngine()

    def large_drawdown(ctx: Dict[str, float]) -> bool:
        return ctx.get("drawdown", 0.0) <= -0.05

    engine.register(
        ReflexRule(
            name="stop_loss_reflex",
            condition=large_drawdown,
            action=block_trade_cb,
            cooldown=60.0,
        )
    )

    def volatility_spike(ctx: Dict[str, float]) -> bool:
        sigma = ctx.get("volatility", 0.0)
        return sigma > 4.0 * ctx.get("volatility_avg", 1.0)

    engine.register(
        ReflexRule(
            name="volatility_ceiling",
            condition=volatility_spike,
            action=block_trade_cb,
            cooldown=90.0,
        )
    )

    return engine
