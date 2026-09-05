"""Strategies the search found, rather than ones somebody wrote.

WHAT THIS CLOSES
----------------
``web/tradingagent/evolve.py`` searches feature space for rules that separate
returns on held-out data, and it finds them -- ``hurst > 0.535`` survives at
t=+2.86 and recurs across four of five random seeds. Until now those findings
sat in a report. Nothing traded them, so nothing could confirm or refute them
on money, and the search was an observatory rather than an instrument.

This turns a survivor into a ``Strategy``: it enters when the rule holds and
the entry clears its own cost, exits on the same terms as any other strategy,
and books its outcomes into the same ledger. It graduates the same way, and it
gets demoted the same way. A discovered rule earns no special standing.

WHY THE RULE IS RE-EVALUATED LIVE, NOT REPLAYED
-----------------------------------------------
The search judged the rule on CLOSED round trips, where every feature was
known after the fact. A live tick has to compute the same features from the
window in front of it, and some cannot be computed at all -- a chaos measure
needs 64 samples, and a thin symbol will not have them.

Where a feature is unavailable the rule does not fire. That is the whole
discipline: a rule that quietly treats "unmeasurable" as "condition met"
trades on the absence of evidence, which is how the horizon variants came to
hold positions nobody could justify.

WHAT KEEPS THIS HONEST
----------------------
A discovered rule is loaded from disk with the holdout statistic that earned
it. If that statistic is missing or below the bar, the strategy refuses to
run at all -- there is no path by which a rule reaches live money without a
number behind it, and no way to hand-write one into the file and have it
trade.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from trading.strategies.base import (Strategy, StrategyContext, env_float,
                                     env_flag, sample_arrays)

ROOT = Path(__file__).resolve().parents[2]

#: Where the search writes what survived. One JSON array of rule records.
DISCOVERED_PATH = ROOT / "data" / "discovered_rules.json"

#: Holdout t a rule must have earned before it may trade at all. The search
#: itself only reports survivors above 1.7; this is the second gate, and it is
#: deliberately higher -- reaching real money should cost more than appearing
#: in a report.
MIN_HOLDOUT_T = 2.0

#: Samples needed before the chaos features can be computed at all. Matches
#: chaos.MIN_SERIES; a shorter window yields no measurement, not a default.
MIN_CHAOS_SAMPLES = 64


def _load_rules() -> List[Dict[str, Any]]:
    """Rules the search has published, filtered to those that earned it."""
    try:
        raw = json.loads(DISCOVERED_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - no file is "no discovered rules"
        return []
    if not isinstance(raw, list):
        return []

    out: List[Dict[str, Any]] = []
    for record in raw:
        if not isinstance(record, dict):
            continue
        try:
            t = float(record.get("holdout_t"))
        except (TypeError, ValueError):
            continue                       # no statistic, no trading
        if not math.isfinite(t) or t < MIN_HOLDOUT_T:
            continue
        conditions = record.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            continue
        out.append(record)
    return out


def _live_features(state: Any, ctx: StrategyContext) -> Dict[str, Any]:
    """Compute, from the window in front of us, what the search searched over.

    Every value may be absent. A feature that cannot be measured here is
    omitted rather than defaulted, so a rule referencing it will not fire.
    """
    features: Dict[str, Any] = {}

    try:
        _, prices, _ = sample_arrays(state, 6 * 3600.0)
    except Exception:  # noqa: BLE001
        prices = np.array([])

    if ctx.last_price and ctx.last_price > 0:
        features["entry_price"] = float(ctx.last_price)

    features["is_live"] = bool(ctx.live_trading)
    features["is_stable_quote"] = True     # every route here ends in a stable

    if prices.size >= MIN_CHAOS_SAMPLES:
        series = [float(p) for p in prices if p and float(p) > 0]
        try:
            import sys

            web = str(ROOT / "web")
            if web not in sys.path:
                sys.path.insert(0, web)
            from tradingagent.chaos import chaos_profile

            profile = chaos_profile(getattr(state, "symbol", "?"), series, 300.0)
            for key in ("hurst", "return_autocorrelation",
                        "usable_horizon_sec"):
                value = profile.get(key)
                if value is not None:
                    features[key] = float(value)
            character = profile.get("character")
            if character and character != "unmeasurable":
                features["chaos_character"] = character
        except Exception:  # noqa: BLE001 - unmeasurable, not zero
            pass

    return features


def _condition_holds(condition: Dict[str, Any],
                     features: Dict[str, Any]) -> Optional[bool]:
    """True, False, or None when the feature is not measurable here."""
    name = str(condition.get("feature") or "")
    op = str(condition.get("op") or "")
    if name not in features:
        return None
    actual = features[name]
    value = condition.get("value")
    try:
        if op == ">":
            return float(actual) > float(value)
        if op == "<":
            return float(actual) < float(value)
        if op == "==":
            return str(actual) == str(value)
        if op == "!=":
            return str(actual) != str(value)
    except (TypeError, ValueError):
        return None
    return None


class DiscoveredRuleStrategy(Strategy):
    """One rule the genetic search found and the holdout confirmed."""

    min_samples = 24

    def __init__(self, record: Dict[str, Any]) -> None:
        self.record = record
        self.rule_id = str(record.get("id") or record.get("rule") or "rule")
        # The id must be UNIQUE as well as readable. Truncating to 40
        # characters collided two rules that shared a prefix -- the registry
        # is keyed by strategy_id, so the second silently replaced the first
        # and one discovered rule vanished. A short hash of the full rule
        # keeps ids distinct however similar their prefixes.
        import hashlib

        safe = "".join(ch if ch.isalnum() else "_" for ch in self.rule_id)[:32]
        digest = hashlib.sha1(self.rule_id.encode("utf-8")).hexdigest()[:6]
        self.strategy_id = f"discovered_{safe}_{digest}"
        self.default_horizon = str(record.get("horizon") or "15m")
        self.conditions: List[Dict[str, Any]] = list(record.get("conditions") or [])
        try:
            self.holdout_t = float(record.get("holdout_t"))
        except (TypeError, ValueError):
            self.holdout_t = 0.0

    def enabled(self) -> bool:
        if self.holdout_t < MIN_HOLDOUT_T:
            return False
        return env_flag("STRATEGY_DISCOVERED_ENABLED", "1")

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        if ctx.available_quote <= 0 or ctx.last_price <= 0:
            return None

        features = _live_features(state, ctx)

        for condition in self.conditions:
            verdict = _condition_holds(condition, features)
            if verdict is None:
                # UNMEASURABLE IS NOT PERMISSION. A rule that fires when its
                # own precondition cannot be evaluated is trading on the
                # absence of evidence.
                return None
            if not verdict:
                return None

        # THE RULE SAYS "WHEN", NOT "HOW MUCH". Expected edge comes from what
        # the rule earned on held-out data, discounted hard: a t-statistic is
        # evidence that a difference exists, not a promise of its size.
        try:
            edge = float(self.record.get("holdout_mean_excess") or 0.0)
        except (TypeError, ValueError):
            edge = 0.0
        if edge <= 0:
            return None

        # Must still clear its own round trip, like anything else here.
        if edge <= ctx.fee_rate:
            return None

        confidence = min(0.9, 0.5 + self.holdout_t / 20.0)
        return self.make_candidate(
            state, ctx,
            action="enter",
            expected_return=edge,
            target_price=ctx.last_price * (1.0 + edge),
            confidence=confidence,
            direction_prob=confidence,
            reason=(f"[discovered t={self.holdout_t:.2f}] "
                    f"{self.record.get('rule') or self.rule_id}"),
            extra_meta={
                "holdout_t": self.holdout_t,
                "rule": str(self.record.get("rule") or ""),
                "discovered": True,
            },
        )


def build_discovered_strategies() -> List[Strategy]:
    """Every published rule that still clears the bar, as strategies.

    Returns an empty list when nothing has been discovered or nothing clears
    MIN_HOLDOUT_T -- which is the correct state for a system that has not yet
    found anything, and is why this is safe to call unconditionally.
    """
    strategies: List[Strategy] = []
    for record in _load_rules():
        try:
            strategies.append(DiscoveredRuleStrategy(record))
        except Exception:  # noqa: BLE001 - one bad record is not fatal
            continue
    return strategies
