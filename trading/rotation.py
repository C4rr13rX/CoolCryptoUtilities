"""Portfolio-level cross-token rotation — the missing "jump around the market"
layer above the per-pair bots.

Each TradingBot is bound to one pair; its BusScheduler only ever re-enters the
same token after an exit. The PortfolioRotator sits above all bots: when any
bot closes a position at a profit (sell high), the rotator collects the
freshest buy-low candidates from EVERY other streamed pair and runs the same
CDCL SAT/UNSAT machinery with portfolio clauses:

  - gas affordability on the chain (global clause, reused from the solver)
  - expected return must clear round-trip fees × a safety factor
  - at most one open position per token (skip pairs already holding)
  - candidates must be fresh (a stale dip signal is a knife, not a dip)

SAT → the chosen enter directive is queued on the target pair's bot, which
consumes it on its next tick instead of its own scheduler output ("sell high
on A, jump into B"). UNSAT → stay parked in stablecoin and wait; that is the
correct answer, not a failure. Profit skimming to the stable bank runs before
this hook, so savings are never re-risked.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from services.logging_utils import log_message
from trading.cdcl_solver import CDCLTradingSolver, Clause


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _rotation_fee_safety(cand: Dict, ctx: Dict) -> bool:
    d = cand.get("directive")
    if d is None:
        return False
    ret = float(getattr(d, "expected_return", 0.0))
    fee = float(ctx.get("fee_rate", 0.0))
    safety = float(ctx.get("rotation_fee_safety", 2.0))
    return math.isfinite(ret) and ret >= fee * safety


def _rotation_freshness(cand: Dict, ctx: Dict) -> bool:
    ts = float((cand.get("meta") or {}).get("rotation_candidate_ts", 0.0))
    ttl = float(ctx.get("rotation_candidate_ttl", 180.0))
    return ts > 0.0 and (time.time() - ts) <= ttl


def _rotation_no_open_position(cand: Dict, ctx: Dict) -> bool:
    open_symbols = ctx.get("open_symbols") or set()
    d = cand.get("directive")
    return getattr(d, "symbol", None) not in open_symbols


class PortfolioRotator:
    """Owns the cross-pair rotation decision; one instance per supervisor."""

    def __init__(self) -> None:
        self._bots: Dict[str, Any] = {}
        self._solver = CDCLTradingSolver(
            nogood_ttl_s=_env_float("ROTATION_NOGOOD_TTL_S", 180.0)
        )
        for clause in (
            Clause(
                name="rotation_fee_safety",
                test=_rotation_fee_safety,
                priority=65,
                resolution="Expected return below fee-safety multiple — hold stable",
            ),
            Clause(
                name="rotation_freshness",
                test=_rotation_freshness,
                priority=75,
                resolution="Candidate too old — market moved on",
            ),
            Clause(
                name="rotation_no_open_position",
                test=_rotation_no_open_position,
                priority=85,
                resolution="Target token already holds a position",
            ),
        ):
            self._solver._clauses.add_clause(clause)
        self.last_rotation: Optional[Dict[str, Any]] = None
        self.rotations_scheduled = 0
        self.rotations_unsat = 0

    # ------------------------------------------------------------------

    def enabled(self) -> bool:
        return os.getenv("ROTATION_ENABLED", "1").lower() in {"1", "true", "yes", "on"}

    def register_bot(self, bot: Any) -> None:
        symbol = str(getattr(bot, "primary_symbol", "") or "")
        if symbol:
            self._bots[symbol] = bot
        setattr(bot, "rotator", self)

    def status(self) -> Dict[str, Any]:
        return {
            "bots": list(self._bots.keys()),
            "scheduled": self.rotations_scheduled,
            "unsat": self.rotations_unsat,
            "last": self.last_rotation,
            "solver": self._solver.status() if hasattr(self._solver, "status") else {},
        }

    # ------------------------------------------------------------------

    def on_exit(
        self,
        source_bot: Any,
        *,
        symbol: str,
        chain: str,
        freed_quote: float,
        profit: float,
    ) -> Optional[Dict[str, Any]]:
        """Called by a bot right after a profitable exit closes.

        Returns a summary of the scheduled rotation, or None on UNSAT/skip.
        """
        if not self.enabled() or profit <= 0:
            return None
        now = time.time()
        ttl = _env_float("ROTATION_CANDIDATE_TTL_S", 180.0)

        open_symbols = set()
        pool: List[Tuple[Any, Dict[str, Any]]] = []
        for sym, bot in self._bots.items():
            try:
                if getattr(bot, "positions", {}).get(sym):
                    open_symbols.add(sym)
            except Exception:
                pass
        for sym, bot in self._bots.items():
            if sym == symbol or sym in open_symbols:
                continue
            scheduler = getattr(bot, "scheduler", None)
            entry = getattr(scheduler, "last_enter_candidates", {}).get(sym) if scheduler else None
            if not entry:
                continue
            entry_ts = float(entry.get("ts", 0.0))
            if now - entry_ts > ttl:
                continue
            for cand in entry.get("candidates", []):
                meta = dict(cand.get("meta") or {})
                meta["rotation_candidate_ts"] = entry_ts
                pool.append((bot, {**cand, "meta": meta}))

        if not pool:
            self.rotations_unsat += 1
            self.last_rotation = {
                "ts": now, "source": symbol, "result": "unsat",
                "reason": "no_fresh_candidates", "freed_quote": freed_quote,
            }
            return None

        try:
            native_balance = float(source_bot.portfolio.get_native_balance(chain))
        except Exception:
            native_balance = 0.0
        context = {
            "native_balance": native_balance,
            "min_native": _env_float("GAS_ALERT_MIN_NATIVE", 0.01),
            "fee_rate": _env_float("ROTATION_FEE_RATE", 0.0075),
            "risk_budget": 1.0,
            "rotation_fee_safety": _env_float("ROTATION_FEE_SAFETY", 2.0),
            "rotation_candidate_ttl": ttl,
            "open_symbols": open_symbols,
        }
        chosen = self._solver.select([cand for _, cand in pool], context)
        if chosen is None:
            self.rotations_unsat += 1
            unsat = getattr(self._solver, "last_unsat", None)
            self.last_rotation = {
                "ts": now, "source": symbol, "result": "unsat",
                "reason": getattr(unsat, "clause", "unknown") if unsat else "unknown",
                "candidates": len(pool), "freed_quote": freed_quote,
            }
            return None

        target_bot = None
        for bot, cand in pool:
            if cand.get("directive") is chosen:
                target_bot = bot
                break
        if target_bot is None:
            for bot, cand in pool:
                if getattr(cand.get("directive"), "symbol", None) == getattr(chosen, "symbol", None):
                    target_bot = bot
                    break
        if target_bot is None:
            return None

        chosen.reason = f"{chosen.reason} [rotated from {symbol}]"
        target_bot.pending_rotation_directive = {
            "directive": chosen,
            "ts": now,
            "source_symbol": symbol,
            "freed_quote": float(freed_quote),
        }
        self.rotations_scheduled += 1
        self.last_rotation = {
            "ts": now,
            "source": symbol,
            "target": getattr(chosen, "symbol", "?"),
            "strategy": getattr(chosen, "strategy_id", ""),
            "expected_return": float(getattr(chosen, "expected_return", 0.0)),
            "result": "sat",
            "candidates": len(pool),
            "freed_quote": freed_quote,
        }
        log_message(
            "rotation",
            f"sell-high on {symbol} -> buy-low on {getattr(chosen, 'symbol', '?')} "
            f"(strategy={getattr(chosen, 'strategy_id', '') or 'legacy'}, "
            f"expected={float(getattr(chosen, 'expected_return', 0.0)):.2%}, "
            f"pool={len(pool)})",
        )
        return self.last_rotation
