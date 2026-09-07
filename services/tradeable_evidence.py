"""Reconstruct each strategy's live-tradeable evidence from recorded history.

WHY THIS EXISTS
---------------

Graduation and re-arm do not read a strategy's ghost book. They read the
*live-tradeable subset* of it -- ``_tradeable_of(ghost)`` at
``trading/strategies/ledger.py`` in ``_evaluate_graduation_locked``, and
``_fresh_tradeable_delta`` in ``_maybe_rearm_locked``. That subset is the right
population to judge: a record earned on symbols the live lane refuses on sight
is not evidence that real money could have been spent.

The subset counter landed in commit ``dcb7517`` (2026-09-07 02:03:15) and was
never backfilled, because ``record()`` only maintains it going forward. Measured
2026-09-07 06:45, 4.7 hours after that commit:

    ledger ghost trades, all strategies          394
    ledger `tradeable` counters, all strategies    7
    ledger entries with no `tradeable` key at all 33 of 37

So the gate that decides whether real money gets spent was judging the only
live-capable executor on a sample of ONE. Replaying the 524 recorded
``ghost-exit`` rows through the same predicates the ledger uses gives the real
population for ``atf_static``:

    ghost exits                                  298
    live-tradeable AND inside the evidence horizon 174
    wins                                          78  (44.8%)
    net                                       +0.0674

That number is the point of this module, and it is not the number anyone was
acting on. "Needs more evidence" and "has 174 round trips and wins 44.8% of
them against a 55% bar" are different findings that call for different work:
the first says wait, the second says this strategy does not have an edge and
waiting will not give it one.

WHAT THIS MODULE IS NOT
-----------------------

It is READ-ONLY and it does not promote anything. It does not write the ledger,
and deliberately so -- see ``reconcile_window`` for the measurement that says a
backfill cannot be done honestly for the two strategies it would matter for.

Every predicate is IMPORTED from the ledger rather than reimplemented here. A
second copy of "is this symbol tradeable" or "is this hold too long" would drift
from the one that decides, and then this module would report a population the
gate does not use -- which is the exact failure it was written to expose.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Where the recorded ghost round trips live. ``trading_ops`` is append-only and
#: keeps pre-fix artifacts forever, so it is read here ONLY for sequence and the
#: fields of an individual exit -- never summed as a book. See the project note
#: "Live P/L has three sources".
DEFAULT_DB = PROJECT_ROOT / "storage" / "trading_cache.db"
DEFAULT_LEDGER = PROJECT_ROOT / "data" / "strategy_ledger.json"


@dataclass
class Evidence:
    """One strategy's reconstructed record, split the way the gate splits it."""

    strategy_id: str
    exits: int = 0
    #: Passed every evidence predicate AND the live lane could have placed it.
    trades: int = 0
    wins: int = 0
    losses: int = 0
    net: float = 0.0
    #: Why the other exits did not count, so the drop is auditable rather than
    #: silent. A funnel that only reports its output hides its own filter.
    dropped_untradeable: int = 0
    dropped_out_of_horizon: int = 0
    dropped_implausible: int = 0
    dropped_no_profit_field: int = 0
    #: Exit-reason histogram, split by outcome. This is not decoration: it is
    #: how ``atf_static_scout``'s 92.5% headline was shown to be an artifact of
    #: exit ROUTING rather than a win rate -- 78 of its 107 exits are
    #: ``max_hold`` and all 78 are wins, because underwater positions are routed
    #: out of ``max_hold`` into ``stale_underwater`` before they can land there.
    exit_reasons: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0


def _bounds() -> Tuple[float, float]:
    """The absolute outcome cap and the evidence horizon, from the ledger.

    Read at call time, not at import: both are env-tuned against a running
    production process, and a value captured at import would report against a
    threshold the gate is no longer using.
    """
    from trading.strategies.ledger import (
        _ABSOLUTE_MAX_OUTCOME,
        _max_evidence_hold_sec,
    )

    return float(_ABSOLUTE_MAX_OUTCOME), float(_max_evidence_hold_sec())


def iter_ghost_exits(db_path: Optional[Path | str] = None) -> Iterable[Dict[str, Any]]:
    """Yield every recorded ghost round trip, oldest first.

    Each row carries what ``record()`` is given: ``strategy_id``, ``symbol``,
    ``profit`` in quote units, and the hold in seconds. ``age_sec`` is absent on
    one exit path, so the hold falls back to ``exit_ts - entry_ts``; a round trip
    with neither is yielded with ``held_sec=None``, which ``record()`` treats as
    "not known" and passes.
    """
    path = Path(db_path or DEFAULT_DB)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT ts, symbol, details FROM trading_ops "
            "WHERE action = 'exit' AND status = 'ghost-exit' ORDER BY ts"
        )
        for row in cur:
            try:
                detail = json.loads(row["details"] or "{}")
            except (TypeError, ValueError):
                continue
            if not isinstance(detail, dict):
                continue
            held = detail.get("age_sec")
            if held is None:
                entry_ts = detail.get("entry_ts")
                exit_ts = detail.get("exit_ts")
                if entry_ts is not None and exit_ts is not None:
                    try:
                        held = float(exit_ts) - float(entry_ts)
                    except (TypeError, ValueError):
                        held = None
            yield {
                "ts": float(row["ts"] or 0.0),
                "strategy_id": (
                    detail.get("strategy_id") or detail.get("strategy") or "unclassified"
                ),
                "symbol": detail.get("symbol") or row["symbol"] or "",
                "profit": detail.get("profit"),
                "held_sec": held,
                "exit_reason": detail.get("exit_reason") or detail.get("reason") or "",
            }
    finally:
        conn.close()


def reconstruct(db_path: Optional[Path | str] = None) -> Dict[str, Evidence]:
    """Per-strategy live-tradeable, in-horizon evidence from recorded history.

    Applies the ledger's own three filters, in the order ``record()`` applies
    them, so a row counted here is a row the ledger would have counted:

      1. an outcome at or above the absolute implausibility cap is an artifact,
      2. an outcome held past the evidence horizon describes market drift rather
         than this strategy's decision,
      3. an outcome on a symbol the live lane cannot trade is not evidence that
         real money could have been placed.

    The relative half of the implausibility test is NOT applied. It scales
    against the strategy's own recent history at the moment of the write, which
    is not recoverable after the fact, and guessing at it would let this module
    report a population the gate never saw. Omitting it can only ADMIT rows the
    ledger might have refused, so every count here is an upper bound -- which is
    the safe direction for a number used to argue a strategy lacks an edge.
    """
    from trading.strategies.ledger import _exceeds_evidence_horizon, _live_tradeable

    abs_cap, _horizon = _bounds()
    out: Dict[str, Evidence] = {}
    # `_live_tradeable` reaches into trading.pipeline per call; the symbol set is
    # tiny and the history is not, so memoise rather than re-ask hundreds of
    # times for the same ticker.
    tradeable_cache: Dict[str, bool] = {}

    for row in iter_ghost_exits(db_path):
        sid = str(row["strategy_id"])
        ev = out.setdefault(sid, Evidence(strategy_id=sid))
        ev.exits += 1

        profit = row["profit"]
        if profit is None:
            ev.dropped_no_profit_field += 1
            continue
        try:
            profit = float(profit)
        except (TypeError, ValueError):
            ev.dropped_no_profit_field += 1
            continue

        if abs(profit) >= abs_cap:
            ev.dropped_implausible += 1
            continue
        if _exceeds_evidence_horizon(row["held_sec"]):
            ev.dropped_out_of_horizon += 1
            continue

        symbol = str(row["symbol"] or "")
        if symbol not in tradeable_cache:
            try:
                tradeable_cache[symbol] = bool(_live_tradeable(symbol))
            except Exception:  # noqa: BLE001
                # Same direction the ledger takes: a symbol whose tradeability
                # cannot be established is not proof that it was tradeable.
                tradeable_cache[symbol] = False
        if not tradeable_cache[symbol]:
            ev.dropped_untradeable += 1
            continue

        ev.trades += 1
        ev.net += profit
        # A flat exit books as a loss, matching `record()`: it still paid the
        # round trip, and the two books have to stay reconcilable.
        bucket = "wins" if profit > 0 else "losses"
        if profit > 0:
            ev.wins += 1
        else:
            ev.losses += 1
        reason = str(row["exit_reason"] or "(none)")
        ev.exit_reasons.setdefault(reason, {"wins": 0, "losses": 0})[bucket] += 1

    return out


def graduation_verdict(ev: Evidence) -> Dict[str, Any]:
    """Does this reconstructed record clear the FIRST-licence bar, and if not, why.

    Reads the same three thresholds ``_evaluate_graduation_locked`` reads, from
    the same env vars, so the verdict here and the gate's decision cannot use
    different numbers. This reports; it does not promote.
    """
    from trading.strategies.ledger import _env_float, _env_int

    min_trades = _env_int("STRATEGY_GRADUATION_MIN_TRADES", 20)
    min_winrate = _env_float("STRATEGY_GRADUATION_MIN_WINRATE", 0.55)
    min_profit = _env_float("STRATEGY_GRADUATION_MIN_PROFIT", 0.0)

    blockers: List[str] = []
    if ev.trades < min_trades:
        blockers.append(
            f"{ev.trades} live-tradeable round trips against a {min_trades} bar"
        )
    if ev.trades and ev.win_rate < min_winrate:
        blockers.append(
            f"win rate {ev.win_rate:.1%} against a {min_winrate:.0%} bar"
        )
    if ev.net <= min_profit:
        blockers.append(f"net {ev.net:+.4f} is not above {min_profit:+.4f}")

    return {
        "clears_bar": not blockers,
        "blockers": blockers,
        "thresholds": {
            "min_trades": min_trades,
            "min_winrate": min_winrate,
            "min_profit": min_profit,
        },
    }


def reconcile_window(
    ledger_path: Optional[Path | str] = None,
    db_path: Optional[Path | str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Can the ledger's rolling window be located in the recorded history?

    The ledger is a rolling promotion window that gets RESET, so its
    ``ghost.trades`` is a suffix of the full history rather than all of it. To
    write a ``tradeable`` sub-book INTO an entry, that suffix has to be located
    exactly -- and the only available check is the one the project note "ledger
    window forensics" records: match the suffix by profit sum, not by timestamp.

    This function exists to say when that check FAILS, because it fails for
    exactly the strategies a backfill would matter for. Measured 2026-09-07:

        33 of 37 entries reconcile exactly
        atf_static        ledger 49 trades / +1.5777, last-49 suffix +1.1342
        atf_static_scout  ledger 235 trades, history holds only 107 exits
        obv_accumulation@1w / @3d   ledger holds one more trade than history

    ``atf_static_scout``'s gap is structural: its exits are written by
    ``services/atf_static_strategy.py`` on a path that does not log every one to
    ``trading_ops``. So for both live-relevant strategies the window boundary is
    unrecoverable, and a backfill would be inventing it. Fail closed: report the
    reconstruction as an independent measurement, and do not write it into the
    gate's own state.
    """
    ledger = json.loads(Path(ledger_path or DEFAULT_LEDGER).read_text("utf-8"))
    history: Dict[str, List[float]] = {}
    for row in iter_ghost_exits(db_path):
        profit = row["profit"]
        if profit is None:
            continue
        try:
            history.setdefault(str(row["strategy_id"]), []).append(float(profit))
        except (TypeError, ValueError):
            continue

    out: Dict[str, Dict[str, Any]] = {}
    for sid, entry in ledger.items():
        ghost = entry.get("ghost") or {}
        n = int(ghost.get("trades", 0) or 0)
        if n <= 0:
            continue
        booked = float(ghost.get("total_profit", 0.0) or 0.0)
        rows = history.get(sid, [])
        if len(rows) < n:
            out[sid] = {
                "reconciles": False,
                "reason": (
                    f"ledger holds {n} ghost trades but only {len(rows)} exits "
                    "are recorded; the window cannot be located"
                ),
                "ledger_trades": n,
                "history_exits": len(rows),
            }
            continue
        suffix = sum(rows[-n:])
        # Proportional with an absolute floor: the books are single-digit
        # dollars, so a pure percentage tolerance is meaningless near zero.
        tolerance = max(0.01, abs(booked) * 0.02)
        ok = abs(suffix - booked) <= tolerance
        out[sid] = {
            "reconciles": ok,
            "reason": (
                ""
                if ok
                else (
                    f"last-{n} suffix sums {suffix:+.4f} against a booked "
                    f"{booked:+.4f} (diff {suffix - booked:+.4f})"
                )
            ),
            "ledger_trades": n,
            "history_exits": len(rows),
            "ledger_profit": booked,
            "suffix_profit": suffix,
        }
    return out


def report(
    db_path: Optional[Path | str] = None,
    ledger_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """The whole measurement, shaped for an API response or a console dump."""
    evidence = reconstruct(db_path)
    ledger: Dict[str, Any] = {}
    try:
        ledger = json.loads(Path(ledger_path or DEFAULT_LEDGER).read_text("utf-8"))
    except Exception:  # noqa: BLE001
        ledger = {}

    strategies = []
    for sid, ev in sorted(evidence.items(), key=lambda kv: -kv[1].trades):
        counter = ((ledger.get(sid) or {}).get("ghost") or {}).get("tradeable") or {}
        row = asdict(ev)
        row["win_rate"] = ev.win_rate
        row["verdict"] = graduation_verdict(ev)
        # The gap between what the gate is reading and what actually happened.
        # This pair is the finding; neither number alone is.
        row["ledger_counter_trades"] = int(counter.get("trades", 0) or 0)
        row["uncounted_trades"] = ev.trades - row["ledger_counter_trades"]
        strategies.append(row)

    return {
        "strategies": strategies,
        "totals": {
            "strategies_with_evidence": sum(1 for s in strategies if s["trades"]),
            "tradeable_round_trips": sum(s["trades"] for s in strategies),
            "ledger_counter_total": sum(s["ledger_counter_trades"] for s in strategies),
        },
        "reconciliation": reconcile_window(ledger_path, db_path),
    }


if __name__ == "__main__":  # pragma: no cover - console entry point
    data = report()
    tot = data["totals"]
    print(
        f"{tot['tradeable_round_trips']} live-tradeable in-horizon round trips "
        f"in the recorded history; the ledger's own counters hold "
        f"{tot['ledger_counter_total']}."
    )
    print()
    print(
        "%-30s %6s %7s %6s %7s %10s  %s"
        % ("strategy", "exits", "counted", "wins", "win%", "net", "blocking promotion")
    )
    for s in data["strategies"]:
        if not s["exits"]:
            continue
        print(
            "%-30s %6d %7d %6d %6.1f%% %+10.4f  %s"
            % (
                s["strategy_id"],
                s["exits"],
                s["trades"],
                s["wins"],
                s["win_rate"] * 100.0,
                s["net"],
                "; ".join(s["verdict"]["blockers"]) or "CLEARS THE BAR",
            )
        )
