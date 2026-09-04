from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class ReflexRule:
    name: str
    condition: Callable[[Dict[str, float]], bool]
    action: Callable[[Dict[str, float]], None]
    cooldown: float = 30.0
    last_trigger: float = 0.0
    #: Cooldown is counted per SCOPE, read from ``context["reflex_scope"]``.
    #: A rule that measures one symbol must not have its next detection
    #: suppressed by a different symbol having tripped it moments ago. Callers
    #: that pass no scope share the single "" bucket, which is exactly the old
    #: behaviour.
    last_trigger_by_scope: Dict[str, float] = field(default_factory=dict)


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
        scope = str(context.get("reflex_scope") or "")
        for rule in self.rules:
            last = rule.last_trigger_by_scope.get(scope, 0.0)
            if now - last < rule.cooldown:
                continue
            try:
                if rule.condition(context):
                    context.setdefault("cooldown", rule.cooldown)
                    context["reflex_rule"] = rule.name
                    rule.action(context)
                    rule.last_trigger_by_scope[scope] = now
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

    #: Minimum RELATIVE dispersion before a symbol counts as spiking at all, as
    #: a fraction of price. Below this the "spike" is quantisation noise on a
    #: nearly-flat feed: measured over 6h of base ticks on 2026-09-04, the p99
    #: relative tick dispersion is 0.0022 for CBBTC and 0.0045 for AERO, while
    #: BSTONK sits at 0.0365 and BASECAT at 0.0240.
    SPIKE_FLOOR = 0.005
    #: The symbol must be this much noisier than IT usually is.
    SPIKE_MULTIPLE = 4.0
    #: ...measured against a baseline with at least this many observations, so
    #: a symbol's first ticks cannot trip a rule whose average is still cold.
    SPIKE_WARMUP = 10

    def volatility_spike(ctx: Dict[str, float]) -> bool:
        # SCALE-FREE by construction. The absolute-price version this replaces
        # compared a $79,698 coin's price-diff std against a global average
        # dominated by a $0.000025 one, so it ranked symbols by price rather
        # than by risk: 49 of 61 triggers in a 6h replay were CBBTC, and the
        # global block they set refused 23.6% of every symbol's decisions.
        if ctx.get("volatility_rel_samples", 0.0) < SPIKE_WARMUP:
            return False
        sigma = float(ctx.get("volatility_rel", 0.0) or 0.0)
        baseline = float(ctx.get("volatility_rel_avg", 0.0) or 0.0)
        if not (sigma > SPIKE_FLOOR):
            return False
        return sigma > SPIKE_MULTIPLE * baseline

    engine.register(
        ReflexRule(
            name="volatility_ceiling",
            condition=volatility_spike,
            action=block_trade_cb,
            cooldown=90.0,
        )
    )

    return engine
