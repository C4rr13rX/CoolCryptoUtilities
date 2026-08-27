"""Per-strategy performance ledger with ghost→live graduation.

Every closed trade is attributed to the strategy that generated its entry
directive (``TradeDirective.strategy_id``). Each strategy accumulates its own
ghost/live stats and graduates to live independently: a strategy that proves
itself in ghost gets ``live_approved``; strategies that haven't stay in ghost
simulation even while the bot itself trades live (dual-track).

Thresholds (env-tunable):
  STRATEGY_GRADUATION_MIN_TRADES   (default 20)  ghost trades required
  STRATEGY_GRADUATION_MIN_WINRATE  (default 0.55)
  STRATEGY_GRADUATION_MIN_PROFIT   (default 0.0) net ghost profit floor
Demotion (mirrors the bot's live circuit breaker at strategy granularity):
  STRATEGY_DEMOTE_MAX_LIVE_LOSSES  (default 4) consecutive live losses
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from services.logging_utils import log_message


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


#: Absolute ceiling on a single ghost outcome, in quote units. The account
#: this runs against holds single-digit dollars, so any individual trade
#: clearing this is a bookkeeping artifact rather than a fill.
_ABSOLUTE_MAX_OUTCOME = _env_float("STRATEGY_MAX_TRADE_PROFIT", 2.0)

#: ...and a relative one: an outcome this many times the strategy's own recent
#: average magnitude is treated as an artifact even when it is small in
#: absolute terms. Scaled per strategy so a large-size strategy is not
#: penalised for trading larger.
_RELATIVE_MAX_MULTIPLE = _env_float("STRATEGY_MAX_TRADE_PROFIT_MULTIPLE", 25.0)


def _is_implausible(profit: float, *, relative_to: Optional[float]) -> bool:
    """Is this outcome too good to have actually happened?

    **Only outsized GAINS are filtered.** A large loss is entirely plausible --
    a stop-loss, a rug, a crash -- and discarding one is actively dangerous:
    it removes the evidence that a strategy is losing money. An earlier version
    of this guard rejected a -10.0 loss, leaving only wins behind, and
    graduated a strategy that should have been blocked. A filter meant to stop
    fiction reaching the promotion gate had become a way to launder a losing
    record, which is worse than the problem it was written for.

    Deliberately conservative in the direction it does filter: a gain must
    clear BOTH an absolute floor and the strategy's own recent scale before
    being rejected. A strategy with no history is judged on the absolute bound.
    """
    try:
        value = float(profit)
    except (TypeError, ValueError):
        return True                      # unparseable is not recordable
    if value != value or value in (float("inf"), float("-inf")):   # NaN / inf
        return True
    if value <= 0.0:
        # Losses are believable -- with one exception.
        #
        # The original rule was "never filter a loss", on the reasoning that
        # discarding one hides the fact that a strategy is losing money. That
        # is right for ordinary losses and wrong for one specific case: a
        # repricing artifact can be a LOSS as well as a gain.
        #
        # Observed 2026-08-26: AERO-USDC entered at 0.5122 -- a frozen
        # pre-repair price -- and marked against the corrected 1.14, booking
        # profit -1.1115 while the same record carried net_pnl 0.0. That is
        # not a strategy losing money, it is a position measured against a
        # price that did not exist when it was opened, and counting it
        # punishes a strategy for a data bug.
        #
        # BUT: filtering a loss is only ever safe against a strategy that has
        # already shown its own scale. With no history, `relative_to` is None
        # and an absolute-bound rule would discard the FIRST big loss a
        # strategy takes -- leaving only its wins and graduating it. That is
        # not hypothetical: this exact change was attempted and immediately
        # tripped `test_unprofitable_never_graduates`, which is the same test
        # that caught it the first time.
        #
        # So a loss is filtered only when the strategy has enough history to
        # say the loss is out of character. A strategy with no track record
        # keeps every loss it takes, which is the conservative direction.
        magnitude = -value
        if relative_to is None or relative_to <= 0.0:
            return False
        return (
            magnitude >= _ABSOLUTE_MAX_OUTCOME
            and magnitude > relative_to * _RELATIVE_MAX_MULTIPLE
        )
    if value < _ABSOLUTE_MAX_OUTCOME:
        return False
    if relative_to is None or relative_to <= 0.0:
        return True                      # no history: absolute bound governs
    return value > relative_to * _RELATIVE_MAX_MULTIPLE


def _blank_mode() -> Dict[str, float]:
    return {
        "trades": 0,
        "wins": 0,
        "total_profit": 0.0,
        "conf_ema": 0.0,
        "peak_profit": 0.0,
        "max_drawdown": 0.0,
        "consecutive_losses": 0,
        "last_ts": 0.0,
    }


class StrategyLedger:
    """JSON-persisted per-strategy × per-mode (ghost/live) trade stats."""

    DEFAULT_PATH = Path("data") / "strategy_ledger.json"

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = Path(path) if path else self.DEFAULT_PATH
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data = raw
        except Exception:
            self._data = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except Exception:
            pass

    def _entry(self, strategy_id: str) -> Dict[str, Any]:
        ent = self._data.setdefault(strategy_id, {})
        ent.setdefault("ghost", _blank_mode())
        ent.setdefault("live", _blank_mode())
        ent.setdefault("live_approved", False)
        ent.setdefault("demotions", 0)
        ent.setdefault("demote_reason", None)
        return ent

    # ------------------------------------------------------------------
    # Recording + graduation
    # ------------------------------------------------------------------

    def _recent_scale(self, sid: str, mode_key: str) -> Optional[float]:
        """Average magnitude of this strategy's outcomes so far, or None.

        Uses total_profit/trades rather than a rolling window because the
        ledger does not retain individual outcomes; it is only ever used as an
        order-of-magnitude reference, so the approximation is adequate.
        """
        try:
            stats = (self._data.get(sid) or {}).get(mode_key) or {}
            trades = int(stats.get("trades", 0) or 0)
            if trades <= 0:
                return None
            return abs(float(stats.get("total_profit", 0.0) or 0.0)) / trades
        except Exception:
            return None

    def record(
        self,
        strategy_id: str,
        *,
        profit: float,
        mode: str,
        confidence: Optional[float] = None,
    ) -> None:
        """Record a closed trade outcome and re-evaluate graduation/demotion."""
        sid = (strategy_id or "unclassified").strip() or "unclassified"
        mode_key = "live" if str(mode).lower() == "live" else "ghost"

        # Reject outcomes too large to be real.
        #
        # A ghost position entered while its feed was frozen and closed after
        # the feed was repaired books the entire repricing as profit. Observed
        # 2026-08-26: AERO-USDC entered at a stale 0.436805, exited at the
        # corrected 1.14, and recorded +3.21 on a 4.58-unit position -- a
        # +161% "gain" that never happened in the market. The same record
        # carried net_pnl 0.0, so the trade both did and did not make money.
        #
        # Left unchecked this is how a strategy graduates on fiction: it is
        # the same failure that put 969 trades at a 0% win rate into the
        # ledger and forced a reset. A discarded outcome costs one data point;
        # an accepted fantasy costs the integrity of the promotion gate.
        if _is_implausible(profit, relative_to=self._recent_scale(sid, mode_key)):
            log_message(
                "strategy-ledger",
                f"rejected implausible {mode_key} outcome for {sid}: "
                f"{profit:+.6f} (likely a stale-entry repricing artifact, "
                "not a real fill)",
                severity="warning",
            )
            return
        with self._lock:
            ent = self._entry(sid)
            stats = ent[mode_key]
            stats["trades"] = int(stats.get("trades", 0)) + 1
            if profit > 0:
                stats["wins"] = int(stats.get("wins", 0)) + 1
                stats["consecutive_losses"] = 0
            else:
                stats["consecutive_losses"] = int(stats.get("consecutive_losses", 0)) + 1
            stats["total_profit"] = float(stats.get("total_profit", 0.0)) + float(profit)
            stats["peak_profit"] = max(float(stats.get("peak_profit", 0.0)), stats["total_profit"])
            stats["max_drawdown"] = max(
                float(stats.get("max_drawdown", 0.0)),
                stats["peak_profit"] - stats["total_profit"],
            )
            if confidence is not None:
                alpha = 0.1
                prev = float(stats.get("conf_ema", 0.0))
                stats["conf_ema"] = (1.0 - alpha) * prev + alpha * max(0.0, min(1.0, float(confidence)))
            stats["last_ts"] = time.time()

            self._evaluate_graduation_locked(sid)
            if mode_key == "live":
                self._evaluate_demotion_locked(sid)
            self._save()

    def _evaluate_graduation_locked(self, sid: str) -> None:
        ent = self._entry(sid)
        if ent.get("live_approved"):
            return
        ghost = ent["ghost"]
        trades = int(ghost.get("trades", 0))
        wins = int(ghost.get("wins", 0))
        profit = float(ghost.get("total_profit", 0.0))
        min_trades = _env_int("STRATEGY_GRADUATION_MIN_TRADES", 20)
        min_winrate = _env_float("STRATEGY_GRADUATION_MIN_WINRATE", 0.55)
        min_profit = _env_float("STRATEGY_GRADUATION_MIN_PROFIT", 0.0)
        if trades >= min_trades and (wins / max(trades, 1)) >= min_winrate and profit > min_profit:
            ent["live_approved"] = True
            ent["graduated_ts"] = time.time()

    def _evaluate_demotion_locked(self, sid: str) -> None:
        ent = self._entry(sid)
        if not ent.get("live_approved"):
            return
        live = ent["live"]
        max_losses = _env_int("STRATEGY_DEMOTE_MAX_LIVE_LOSSES", 4)
        if int(live.get("consecutive_losses", 0)) >= max_losses:
            self._demote_locked(sid, f"{live['consecutive_losses']} consecutive live losses")
            return

        # Live profitability is the metric that decides, above all others.
        #
        # Consecutive losses alone are not enough. A strategy that alternates
        # win/loss/win/loss while paying a ~1.0% round trip each time never
        # reaches four in a row, so it could drain real funds indefinitely and
        # never be demoted -- and that alternating pattern is exactly what was
        # measured on this feed, where a rising price continues rising only
        # 44-50% of the time.
        #
        # So: once a strategy has a fair sample of LIVE trades, it must be net
        # positive on real money. Win rate, ghost record and consecutive-loss
        # counts are all secondary to whether the account grew.
        live_trades = int(live.get("trades", 0))
        min_sample = _env_int("STRATEGY_DEMOTE_MIN_LIVE_TRADES", 8)
        if live_trades >= min_sample:
            live_profit = float(live.get("total_profit", 0.0))
            floor = _env_float("STRATEGY_DEMOTE_MIN_LIVE_PROFIT", 0.0)
            if live_profit <= floor:
                self._demote_locked(
                    sid,
                    f"live P/L {live_profit:+.4f} over {live_trades} trades "
                    f"is not profitable",
                )
                return

        # Drawdown brake: give back too much of the peak and stop, even while
        # still net positive. A strategy that made money and is now handing it
        # back is not one to keep funding.
        peak = float(live.get("peak_profit", 0.0))
        current = float(live.get("total_profit", 0.0))
        max_dd = _env_float("STRATEGY_DEMOTE_MAX_LIVE_DRAWDOWN", 0.5)
        if peak > 0 and current < peak * (1.0 - max_dd):
            self._demote_locked(
                sid,
                f"live drawdown: {current:+.4f} from peak {peak:+.4f}",
            )

    def _demote_locked(self, sid: str, reason: str) -> None:
        ent = self._entry(sid)
        ent["live_approved"] = False
        ent["demotions"] = int(ent.get("demotions", 0)) + 1
        ent["demote_reason"] = reason
        ent["demoted_ts"] = time.time()
        # Demotion resets the ghost proving-ground so re-graduation requires
        # fresh evidence, not stale pre-demotion stats.
        ent["ghost"] = _blank_mode()
        ent["live"]["consecutive_losses"] = 0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def demote(self, strategy_id: str, reason: str) -> None:
        with self._lock:
            self._demote_locked((strategy_id or "unclassified").strip() or "unclassified", reason)
            self._save()

    def is_live_approved(self, strategy_id: str) -> bool:
        with self._lock:
            ent = self._data.get((strategy_id or "").strip() or "unclassified")
            return bool(ent and ent.get("live_approved"))

    def any_live_approved(self) -> bool:
        with self._lock:
            return any(bool(ent.get("live_approved")) for ent in self._data.values())

    def approved_ids(self) -> list[str]:
        with self._lock:
            return [sid for sid, ent in self._data.items() if ent.get("live_approved")]

    def stats(self, strategy_id: str) -> Dict[str, Any]:
        with self._lock:
            ent = self._data.get((strategy_id or "").strip() or "unclassified")
            return json.loads(json.dumps(ent)) if ent else {}

    def snapshot(self) -> Dict[str, Any]:
        """Full copy for readiness reports / dashboards."""
        with self._lock:
            return json.loads(json.dumps(self._data))
