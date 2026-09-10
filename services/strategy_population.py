"""The whole strategy population, in one read, with the stage each one is at.

Why this exists
---------------
Until this module landed there was no way to see the population. The pipeline
page rendered aggregate stage dots and a metrics table; the readiness endpoint
(``scripts/readiness_report.py``) listed the 37 strategies that appear in the
LEDGER and nothing else. Measured 2026-09-07 that hid two whole groups:

  * 4 strategies are in ``data/strategy_registry.json`` and NOT in the ledger
    -- mean_reversion, momentum_breakout, volume_spike, vwap_reversion. They
    were commissioned 10.9 days ago and have never closed a single ghost round
    trip, so nothing that reads the ledger can see them at all. A strategy
    that produces no evidence is the one most worth looking at, and it was the
    one guaranteed to be invisible.
  * every strategy that was tried and permanently barred. ``atf_static_scout``
    carries 235 ghost round trips and ``graduation_blocked=True``; the page
    showed neither the block nor its reason.

So this reads registry UNION ledger, never the intersection.

The bar it reports against is the real one
------------------------------------------
The thresholds and the evidence population are IMPORTED from
``trading.strategies.ledger`` rather than restated here. That is deliberate.
A dashboard that keeps its own copy of a promotion rule is a dashboard that
eventually disagrees with the promoter, and the disagreement is invisible
precisely because both look authoritative.

``scripts/readiness_report.py`` is the worked example of that failure, and it
is live today: it reports ``atf_static`` as ``ready: true`` with 49 ghost
trades and an empty ``blockers`` list. ``atf_static`` cannot graduate. It was
demoted, so ``_evaluate_graduation_locked`` hands it to ``_maybe_rearm_locked``,
which reads the FRESH LIVE-TRADEABLE delta since ``ghost_at_demotion`` -- 1
round trip against a bar of 20. The readiness row is off by a factor of ~49
because it read the pooled ghost book, which is not the population any bar
consults. This module asks the ledger which basis applies and reports that
one.

Two things follow, and both are reported per row rather than averaged away:

  * ``basis`` says WHICH population the bar reads: ``first-licence`` (the
    live-tradeable ghost subset) or ``re-arm`` (only what was gathered since
    the demotion). They differ by an order of magnitude for the one strategy
    that has ever been live.
  * ``tradeable`` is broken out beside ``ghost`` on every row. Pooled ghost
    counts are what make a strategy look ready; the live lane can only spend
    on the tradeable subset. Population-wide that gap is the finding: 37
    ledger strategies hold 336 pooled ghost round trips and 7 live-tradeable
    ones.

Read-only. Nothing here promotes, demotes or writes: the ledger decides
graduation as evidence arrives, and this module reports what it decided.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

# Imported for their VALUES, not re-implemented. See the module docstring:
# a second copy of a promotion rule is a rule that silently drifts.
from trading.strategies.ledger import (
    _env_float,
    _env_int,
    _fresh_tradeable_delta,
    _ghost_only_ids,
    _tradeable_of,
    StrategyLedger,
)

#: Stage order, broadest-to-live, plus the terminal one. The page renders the
#: funnel in this order and an empty stage still gets a row -- an empty stage
#: is a finding ("nothing is being backtested"), not something to hide.
STAGE_ORDER = ("candidate", "backtest", "ghost", "live", "rejected")

STAGE_LABELS = {
    "candidate": "Candidate",
    "backtest": "Backtest",
    "ghost": "Ghost",
    "live": "Live",
    "rejected": "Rejected",
}


def populations() -> Dict[str, Any]:
    """The THREE strategy populations, side by side, in one read.

    WHY THIS EXISTS. Measured pass 106-111: three different counts of "the
    strategies" were quoted in a single pass -- 72, 43 and 39 -- and at most
    one of them can be the denominator of any given share. They are all
    correct; they answer different questions, and nothing said which:

      ``offered``      72   every strategy ``StrategyRegistry.evaluate_all``
                            is asked for a candidate on EVERY tick. Built from
                            code (``build_default_registry``), so it exists
                            whether or not anything has been written to disk.
      ``commissioned`` 43   ``data/strategy_registry.json``, the append-only
                            lifetime record.
      ``evidenced``    39   ``data/strategy_ledger.json``, the rolling
                            promotion window. A strict SUBSET of
                            ``commissioned``.

    AND THEY DO NOT NEST THE WAY EVERYONE ASSUMES. ``offered`` is NOT a
    superset of ``commissioned``: 35 plugin ids (every ``@5h``/``@12h``/
    ``@1d``/``@3d``/``@5d``/``@1w`` horizon variant, plus dust_micro_swing,
    genome_champion, obv_accumulation, omen_reversion, swarm_consensus) have
    no registry row until they first record an outcome, so they are invisible
    to the status command and the population page while being asked on every
    tick. And 6 registry ids are not plugins at all -- atf_static_scout,
    bus_schedule, unclassified and three discovered_hurst rules -- because
    they are non-plugin executors. The union is 78.

    THE DEFECT THIS NAMES, and it is worse than a moving denominator: A SHARE
    WHOSE DENOMINATOR IS ``evidenced`` IS SELF-REFERENTIAL, because a strategy
    enters that population BY PRODUCING THE NUMERATOR. "strategies with >=5
    tradeable trades must rise from 11 of 38" can be satisfied by strategies
    leaving the ledger. Measured against the population that COULD produce
    evidence it is 1 of 78, not 11 of 38.

    So: use ``evidenced`` only for "of the strategies that have traded", and
    ``known`` for any statement about coverage or starvation. Whichever is
    used, name it -- that is what nothing did.

    Ruled out while measuring this, so nobody re-checks it: the registry is
    NOT written non-atomically. ``services/strategy_registry._save`` goes
    through ``services.atomic_json.write_json``, which writes a PID+uuid
    unique temp and ``os.replace``s it under an O_EXCL lock with retries, so a
    reader cannot observe a partial file. A torn read would fail json parsing
    and yield 0, never a plausible smaller count.
    """
    offered: List[str] = []
    try:
        from trading.strategies import build_default_registry

        offered = sorted(build_default_registry().ids())
    except Exception:  # noqa: BLE001
        # Reported as an empty set with ``offered_ok`` False rather than as
        # zero strategies: "the code would not import" and "nothing is
        # offered a tick" are opposite findings and must not look alike.
        offered_ok = False
    else:
        offered_ok = True

    commissioned: List[str] = []
    try:
        from services import strategy_registry as _registry

        commissioned = sorted(
            str(row.get("strategy_id") or row.get("name") or "")
            for row in (_registry.list_strategies() or [])
            if isinstance(row, dict)
        )
        commissioned = [s for s in commissioned if s]
    except Exception:  # noqa: BLE001
        commissioned_ok = False
    else:
        commissioned_ok = True

    evidenced: List[str] = []
    try:
        snap = StrategyLedger().snapshot()
        evidenced = sorted(str(k) for k in (snap or {}))
    except Exception:  # noqa: BLE001
        evidenced_ok = False
    else:
        evidenced_ok = True

    o, c, e = set(offered), set(commissioned), set(evidenced)
    return {
        "offered": offered,
        "commissioned": commissioned,
        "evidenced": evidenced,
        "known": sorted(o | c | e),
        "counts": {
            "offered": len(o),
            "commissioned": len(c),
            "evidenced": len(e),
            "known": len(o | c | e),
        },
        "sources": {
            "offered": "trading.strategies.build_default_registry().ids()",
            "commissioned": "data/strategy_registry.json",
            "evidenced": "data/strategy_ledger.json",
            "known": "union of all three",
        },
        "ok": {
            "offered": offered_ok,
            "commissioned": commissioned_ok,
            "evidenced": evidenced_ok,
        },
        # The differences, because "why are these numbers not equal" is the
        # question that cost two passes, and an answer that requires the
        # reader to re-derive a set difference is not an answer.
        "offered_not_commissioned": sorted(o - c),
        "commissioned_not_offered": sorted(c - o),
        "evidenced_not_commissioned": sorted(e - c),
        "evidence_is_a_subset_of_commissioned": e <= c,
    }


def _rate(wins: Any, trades: Any) -> Optional[float]:
    """Win rate, or None when there is nothing to divide.

    None rather than 0.0 on an empty book. A strategy with no trades has an
    UNKNOWN win rate, and rendering that as 0% puts it at the bottom of a
    sorted column beside strategies that genuinely lose -- which is how the
    four never-traded strategies would read as the worst in the system rather
    than as the untested ones.
    """
    try:
        n = int(trades or 0)
        w = int(wins or 0)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    return w / n


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ts(value: Any) -> Optional[float]:
    """An epoch timestamp, or None. Zero is treated as absent."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _mode_block(stats: Any) -> Dict[str, Any]:
    stats = stats if isinstance(stats, dict) else {}
    trades = _i(stats.get("trades"))
    wins = _i(stats.get("wins"))
    return {
        "trades": trades,
        "wins": wins,
        "losses": _i(stats.get("losses")),
        "win_rate": _rate(wins, trades),
        "profit": _f(stats.get("total_profit")),
        "last_ts": _ts(stats.get("last_ts")),
    }


def _tradeable_block(stats: Any) -> Dict[str, Any]:
    sub = _tradeable_of(stats)
    trades = _i(sub.get("trades"))
    wins = _i(sub.get("wins"))
    return {
        "trades": trades,
        "wins": wins,
        "losses": _i(sub.get("losses")),
        "win_rate": _rate(wins, trades),
        "profit": _f(sub.get("total_profit")),
    }


def _historical_tradeable(reg_row: Dict[str, Any]) -> Optional[int]:
    """Live-tradeable ghost round trips in the strategy's LIFETIME record.

    The graduation bar reads ``ghost.tradeable.trades`` in the ledger, and that
    sub-counter was introduced by ``dcb7517`` at 2026-09-07 02:03:15. It starts
    at zero for every strategy and nothing backfilled it, so a bar that asks
    for 20 live-tradeable round trips is currently reading a counter that is
    hours old rather than a book that is days old.

    Measured 2026-09-07 06:45, re-deriving tradeability from the registry's own
    per-symbol lifetime record through the same ``stop_is_unenforceable``
    predicate ``ledger._live_tradeable`` uses:

        population   275 historical live-tradeable round trips, bar sees 7
        atf_static    35 historical, counter 1, needs 20
        scout        180 historical, counter 2
        obv_accumulation@5d  11 historical, counter 0

    So "no strategy is armed" is, right now, a counter that was reset 4.6 hours
    ago rather than a market condition. That belongs on the page: a stage
    reading "0 live" is a different problem depending on whether the evidence
    does not exist or merely is not being counted.

    Returns None when the registry has no symbol record to derive from --
    unknown, rather than zero, because a zero here would read as "this
    strategy has no tradeable history" which is a much stronger claim than
    "there is nothing to check".
    """
    lifetime = reg_row.get("lifetime")
    if not isinstance(lifetime, dict):
        return None
    ghost = lifetime.get("ghost")
    if not isinstance(ghost, dict):
        return None
    symbols = ghost.get("symbols")
    if not isinstance(symbols, dict) or not symbols:
        return None
    try:
        from trading.pipeline import stop_is_unenforceable
    except Exception:  # noqa: BLE001
        return None

    total = 0
    for sym, count in symbols.items():
        try:
            if not stop_is_unenforceable(str(sym)):
                total += _i(count)
        except Exception:  # noqa: BLE001
            # One unjudgeable symbol must not discard the whole count; it is
            # simply not counted as tradeable, which matches _live_tradeable.
            continue
    return total


def _criteria() -> Dict[str, Any]:
    """The live thresholds, read the same way the ledger reads them.

    Read at call time rather than at import: they are env-tunable, and a
    module-level snapshot would show the page a bar that production is no
    longer using.
    """
    return {
        "min_trades": _env_int("STRATEGY_GRADUATION_MIN_TRADES", 20),
        "min_winrate": _env_float("STRATEGY_GRADUATION_MIN_WINRATE", 0.55),
        "min_profit": _env_float("STRATEGY_GRADUATION_MIN_PROFIT", 0.0),
        "rearm_min_live_trades": _env_int("STRATEGY_REARM_MIN_LIVE_TRADES", 3),
    }


def _progress(entry: Dict[str, Any], crit: Dict[str, Any]) -> Dict[str, Any]:
    """How far this strategy is from its NEXT licence, on the right population.

    Mirrors the branch in ``_evaluate_graduation_locked``: a strategy carrying
    a ``demote_reason`` is judged by ``_maybe_rearm_locked`` on evidence
    gathered SINCE the demotion, and every other strategy is judged on its
    live-tradeable ghost subset. Reporting the pooled ghost book for a demoted
    strategy is the exact error in ``scripts/readiness_report.py``.
    """
    ghost = entry.get("ghost") or {}
    demoted = bool(entry.get("demote_reason"))
    if demoted:
        basis = "re-arm"
        sub = _fresh_tradeable_delta(ghost, entry.get("ghost_at_demotion"))
        trades = _i(sub.get("trades"))
        wins = _i(sub.get("wins"))
        profit = _f(sub.get("total_profit"))
    else:
        basis = "first-licence"
        sub = _tradeable_of(ghost)
        trades = _i(sub.get("trades"))
        wins = _i(sub.get("wins"))
        profit = _f(sub.get("total_profit"))

    need = _i(crit["min_trades"])
    return {
        "basis": basis,
        "trades_have": trades,
        "trades_need": need,
        # Clamped to 1.0 so a strategy with the sample does not render a bar
        # past the end of its track. The bar is "sample gathered", and the
        # win-rate/profit conditions are reported beside it rather than folded
        # in -- they are pass/fail, not progress.
        "sample_fraction": min(1.0, trades / need) if need > 0 else 1.0,
        "winrate_have": _rate(wins, trades),
        "winrate_need": _f(crit["min_winrate"]),
        "profit_have": profit,
        "profit_need": _f(crit["min_profit"]),
        "meets_trades": trades >= need,
        "meets_winrate": (_rate(wins, trades) or 0.0) >= _f(crit["min_winrate"]),
        "meets_profit": profit > _f(crit["min_profit"]),
    }


def _blockers(entry: Dict[str, Any], prog: Dict[str, Any]) -> List[str]:
    """Plain-language reasons this strategy is not live, most binding first."""
    out: List[str] = []
    if entry.get("live_approved"):
        return out
    if entry.get("graduation_blocked"):
        reason = entry.get("graduation_blocked_reason") or "graduation blocked"
        out.append(str(reason))
        # A permanent bar is never re-litigated against a fresh book
        # (_evaluate_graduation_locked returns before reading one), so listing
        # sample shortfalls underneath it would imply a route that does not
        # exist.
        return out
    if not prog["meets_trades"]:
        out.append(
            f"{prog['trades_have']}/{prog['trades_need']} live-tradeable ghost "
            f"round trips ({prog['basis']} basis)"
        )
    if prog["trades_have"] > 0 and not prog["meets_winrate"]:
        out.append(
            f"win rate {prog['winrate_have']:.1%} below {prog['winrate_need']:.0%}"
        )
    if prog["trades_have"] > 0 and not prog["meets_profit"]:
        out.append(f"net {prog['profit_have']:+.4f} not above {prog['profit_need']:+.4f}")
    return out


def _classify(entry: Dict[str, Any], reg_row: Dict[str, Any], prog: Dict[str, Any]):
    """(stage, status, status_reason, stage_since_ts) for one strategy.

    Precedence matters and follows the ledger's own precedence: a live licence
    outranks everything, a permanent bar outranks any record, and a demotion
    outranks the ordinary ghost path.
    """
    ghost = entry.get("ghost") or {}
    live = entry.get("live") or {}
    ghost_trades = _i(ghost.get("trades"))
    live_trades = _i(live.get("trades"))

    if entry.get("live_approved"):
        since = _ts(entry.get("reinstated_ts")) or _ts(entry.get("graduated_ts"))
        return "live", "live-armed", "approved to spend real money", since

    if entry.get("graduation_blocked"):
        reason = entry.get("graduation_blocked_reason") or "graduation blocked"
        # A block recorded after a demotion dates from the demotion; a
        # structural block (a ghost-only executor) has no event of its own, so
        # fall back to when the strategy was created rather than inventing one.
        since = _ts(entry.get("demoted_ts")) or _ts(reg_row.get("created_at"))
        return "rejected", "blocked-permanently", str(reason), since

    if entry.get("demote_reason"):
        return (
            "ghost",
            "demoted-rearming",
            str(entry.get("demote_reason")),
            _ts(entry.get("demoted_ts")),
        )

    if ghost_trades > 0 or live_trades > 0:
        if prog["meets_trades"] and prog["meets_winrate"] and prog["meets_profit"]:
            # The ledger evaluates on every record(), so this should be
            # transient. If a row sits here it means evidence arrived while
            # something upstream of record() was down -- worth seeing, not
            # worth hiding.
            status, reason = "ready-to-graduate", "clears the bar; awaiting evaluation"
        else:
            status, reason = "collecting-evidence", "gathering ghost round trips"
        return "ghost", status, reason, _ts(ghost.get("first_ts"))

    # No evidence at all. Distinguish "has a recorded backtest" from "has
    # never been measured": both are pre-ghost, and only the second one is a
    # strategy nothing has ever run.
    if reg_row.get("metrics") or reg_row.get("experiments"):
        return (
            "backtest",
            "backtested",
            "backtest recorded; no ghost round trip yet",
            _ts(reg_row.get("created_at")),
        )
    return (
        "candidate",
        "never-run",
        "commissioned but has never closed a ghost round trip",
        _ts(reg_row.get("created_at")),
    )


def collect(now: Optional[float] = None, ledger_path: Any = None) -> Dict[str, Any]:
    """Every strategy the system knows about, with its stage and its numbers.

    Registry UNION ledger. A strategy present in only one of them is exactly
    the kind this is meant to surface.

    ``ledger_path`` is an injection point for tests. Production passes nothing
    and gets ``StrategyLedger.DEFAULT_PATH``.
    """
    now = float(now if now is not None else time.time())
    crit = _criteria()

    try:
        from services import strategy_registry as _registry

        reg_rows = {
            str(row.get("strategy_id") or row.get("name") or ""): row
            for row in (_registry.list_strategies() or [])
            if isinstance(row, dict)
        }
        reg_rows.pop("", None)
    except Exception:  # noqa: BLE001
        # The ledger half is still worth serving. Reported on the payload so a
        # short population reads as a degraded source rather than as strategies
        # having disappeared.
        reg_rows = {}
        registry_ok = False
    else:
        registry_ok = True

    # Constructed per call, never cached: __init__ runs _load(), and
    # production rewrites this file under a lock while the page is open. A
    # module-level instance would serve a snapshot frozen at first import.
    ledger = StrategyLedger(path=ledger_path) if ledger_path else StrategyLedger()
    ledger_rows = ledger.snapshot() if hasattr(ledger, "snapshot") else {}
    if not isinstance(ledger_rows, dict):
        ledger_rows = {}

    ghost_only = _ghost_only_ids()

    rows: List[Dict[str, Any]] = []
    for sid in sorted(set(reg_rows) | set(ledger_rows)):
        entry = ledger_rows.get(sid)
        entry = dict(entry) if isinstance(entry, dict) else {}
        reg_row = reg_rows.get(sid) or {}

        # A ghost-only id is barred whether or not the ledger file has caught
        # up with it: the ledger applies the same set at read time
        # (_entry/_revoke_ghost_only_approval), so mirroring it here keeps the
        # page and the promoter in agreement on a strategy the ledger has not
        # yet rewritten.
        if sid in ghost_only and not entry.get("graduation_blocked"):
            entry["graduation_blocked"] = True
            entry.setdefault(
                "graduation_blocked_reason",
                "listed in GHOST_ONLY_STRATEGY_IDS: no live execution branch",
            )

        prog = _progress(entry, crit)
        hist = _historical_tradeable(reg_row)
        stage, status, status_reason, since = _classify(entry, reg_row, prog)
        ghost = _mode_block(entry.get("ghost"))
        live = _mode_block(entry.get("live"))
        last_ts = max(
            [t for t in (ghost["last_ts"], live["last_ts"]) if t is not None],
            default=None,
        )

        rows.append(
            {
                "id": sid,
                "name": str(reg_row.get("name") or sid),
                "kind": str(reg_row.get("kind") or ("ledger-only" if not reg_row else "")),
                "in_registry": sid in reg_rows,
                "in_ledger": sid in ledger_rows,
                "commissioned": bool(reg_row.get("commissioned", False)),
                "stage": stage,
                "status": status,
                "status_reason": status_reason,
                "stage_since_ts": since,
                "stage_age_sec": (now - since) if since is not None else None,
                "ghost": ghost,
                "tradeable": _tradeable_block(entry.get("ghost")),
                # What the bar WOULD see if the counter had been backfilled.
                # Reported beside the counter, never instead of it: the ledger
                # decides on the counter, and a page that quietly substituted
                # this number would show licences that do not exist.
                "tradeable_historical": hist,
                "tradeable_uncounted": (
                    max(0, hist - _tradeable_block(entry.get("ghost"))["trades"])
                    if hist is not None
                    else None
                ),
                "live": live,
                "live_approved": bool(entry.get("live_approved")),
                "demotions": _i(entry.get("demotions")),
                "progress": prog,
                "blockers": _blockers(entry, prog),
                "last_trade_ts": last_ts,
                "last_trade_age_sec": (now - last_ts) if last_ts is not None else None,
            }
        )

    stages = [
        {
            "stage": name,
            "label": STAGE_LABELS[name],
            "count": sum(1 for r in rows if r["stage"] == name),
        }
        for name in STAGE_ORDER
    ]

    # Population-wide totals. The pooled/tradeable pair is the headline: the
    # first number is what makes the population look busy and the second is
    # what any licence is actually granted on.
    return {
        "generated_at": now,
        "registry_ok": registry_ok,
        # Carried on every payload so a share computed from `strategies` below
        # can be checked against the population it should have used. `rows` is
        # registry UNION ledger, which is NOT the population offered a tick.
        "populations": populations(),
        "criteria": crit,
        "stages": stages,
        "totals": {
            "strategies": len(rows),
            "ghost_trades": sum(r["ghost"]["trades"] for r in rows),
            "tradeable_trades": sum(r["tradeable"]["trades"] for r in rows),
            "tradeable_profit": sum(r["tradeable"]["profit"] for r in rows),
            # The headline gap. `tradeable_trades` is what every graduation
            # bar in the system can currently see; `tradeable_historical` is
            # what the same predicate finds in the lifetime record. Measured
            # 2026-09-07: 7 against 275.
            "tradeable_historical": sum(
                r["tradeable_historical"] or 0 for r in rows
            ),
            "live_trades": sum(r["live"]["trades"] for r in rows),
            "live_profit": sum(r["live"]["profit"] for r in rows),
            "live_approved": sum(1 for r in rows if r["live_approved"]),
        },
        "strategies": rows,
    }
