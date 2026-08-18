"""Market-evolution genome strategy.

Trades the current GA champion so an evolved genome can earn a real ghost
record instead of only a backtest profit factor. This closes rung 2 of the
promotion ladder:

    GA backtest (profit factor) -> ghost trading -> live promotion

Design mirrors SwarmConsensusStrategy: this is a lookup plus arithmetic, not
a computation over raw windows. Building the genome's 41+ features is
cross-sectional (market breadth is a median across the whole universe at one
timestamp), so it cannot be done per-pair inside evaluate(). The publisher
computes signals once per tick for every asset and drops them into
``ctx.extras["genome_signals"]``; this strategy reads its own symbol out.

Two refusals are deliberate, because both failure modes produce a confident
signal from a model nobody validated:

  * a genome below the objective, or measured on a partial walk-forward,
    never trades at all;
  * a signal whose feature vector was incomplete is skipped rather than
    scored against defaulted zeros.

The strategy_id embeds the genome id, so a new champion earns its OWN ghost
record rather than inheriting the incumbent's graduation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from trading.strategies.base import Strategy, StrategyContext, env_float, env_flag


class GenomeChampionStrategy(Strategy):
    #: Base identity. The live ledger id is per-genome; see strategy_id below.
    strategy_id = "genome_champion"
    default_horizon = "12h"
    min_samples = 12

    def __init__(self) -> None:
        self._active_id = ""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    def _ledger_id(self, signal: Dict[str, Any]) -> str:
        """Per-genome ledger identity.

        A champion that has not proven itself in ghost must not inherit the
        previous champion's trade count, win rate or graduation.
        """
        genome_id = str(signal.get("genome_id") or "").strip()
        return f"genome_{genome_id[:12]}" if genome_id else self.strategy_id

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        if not env_flag("STRATEGY_GENOME_CHAMPION_ENABLED", "1"):
            return None

        signals = (ctx.extras or {}).get("genome_signals")
        if not isinstance(signals, dict):
            return None
        signal = signals.get(str(getattr(state, "base_token", "")).upper())
        if not isinstance(signal, dict):
            return None

        # Refuse anything the publisher could not fully score.
        if not signal.get("scorable"):
            return None
        if not signal.get("meets_objective"):
            return None

        try:
            direction = int(signal.get("direction", 0))
            confidence = float(signal.get("confidence", 0.0) or 0.0)
            expected = float(signal.get("expected_return", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None

        if direction == 0 or ctx.last_price <= 0:
            return None

        # The genome's own selectivity: it abstains on most bars by design,
        # and forcing it to act outside its confidence band is exactly the
        # coverage-chasing that made the search unprofitable. The publisher
        # already zeroes direction below the genome's confidence_quantile, so
        # that validated band is the gate. A fixed 0.55 default sat well above
        # the champion's own quantile (0.271) and above every confidence it
        # actually emits (max 0.568), so it only ever discarded signals the
        # genome had already qualified.
        # Default to the genome's own validated quantile when the signal
        # carries one, so the gate tracks what the search actually qualified
        # rather than a number chosen independently of it.
        try:
            genome_band = float(signal.get("confidence_quantile") or 0.0)
        except (TypeError, ValueError):
            genome_band = 0.0
        min_conf = env_float("GENOME_STRATEGY_MIN_CONFIDENCE",
                             max(0.25, genome_band), lo=0.0, hi=1.0)
        if confidence < min_conf:
            return None

        # Never trade an edge smaller than the round trip costs.
        #
        # `expected_return` is anchored on the genome's measured expectancy,
        # which the GA computes NET of its OBJECTIVE_COST_BPS (25bps = 0.0025)
        # round trip. Defaulting this floor to 0.0025 therefore charged that
        # cost a second time and demanded ~2.7x an edge the genome can ever
        # report: with the champion at expectancy 0.00091, zero of 33 live
        # signals cleared it, so the strategy could never have traded no
        # matter how profitable the genome was. Compare against the fee this
        # venue actually charges instead, which the next line already does.
        min_net = env_float("GENOME_STRATEGY_MIN_NET_RETURN", 0.0, lo=0.0, hi=0.5)
        edge = abs(expected)
        if edge <= ctx.fee_rate or edge < min_net:
            return None
        edge = min(edge, 0.05)

        ledger_id = self._ledger_id(signal)
        # `strategy_id` is read by make_candidate() for both the directive and
        # the ledger, so bind it for this evaluation.
        self.strategy_id = ledger_id

        action = "enter" if direction > 0 else "exit"
        if action == "enter" and ctx.available_quote <= 0:
            return None
        if action == "exit" and ctx.available_base <= 0:
            return None

        target = (ctx.last_price * (1.0 + edge) if action == "enter"
                  else ctx.last_price * (1.0 - edge))
        reason = (f"genome {str(signal.get('genome_id') or '')[:12]} "
                  f"pf={float(signal.get('profit_factor') or 0.0):.3f} "
                  f"dir={direction:+d} conf={confidence:.2f}")

        return self.make_candidate(
            state, ctx,
            action=action,
            expected_return=edge,
            target_price=target,
            confidence=confidence,
            reason=reason,
            direction_prob=float(signal.get("direction_prob", 0.5) or 0.5),
            extra_meta={
                "genome_id": signal.get("genome_id"),
                "genome_profit_factor": signal.get("profit_factor"),
                "genome_coverage": signal.get("coverage"),
                "genome_folds": signal.get("evaluated_folds"),
                "source": "market_evolution",
            },
        )
