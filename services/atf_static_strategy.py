from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from db import get_db
from services.watchlists import load_watchlists, save_watchlists
from trading.portfolio import PortfolioState, STABLE_TOKENS


REPO_ROOT = Path(__file__).resolve().parents[1]
SIGNAL_KEY = "atf_static_strategy:signals"
LATEST_KEY = "atf_static_strategy:latest"
PENDING_BUS_KEY = "atf_static_strategy:pending_bus_actions"
FEEDBACK_KEY = "atf_static_strategy:feedback"
GHOST_POSITIONS_KEY = "atf_static_strategy:ghost_positions"
SOURCE = "c0d3rv2_atf_static"

#: Ledger identity for trades this module opens and closes ITSELF.
#:
#: Deliberately NOT "atf_static". Two different executors trade the ATF
#: signals: this scout, and ``trading/strategies/atf_static.py`` running
#: inside the bot. They share a signal source and nothing else -- the scout
#: enters on its own corroborated quote and exits on an 8% stop, a 1h hold
#: or its target, while the bot enters through the CDCL solver and exits on
#: triggers, a 2% stop, confidence drops and timed exits. Same entry idea,
#: entirely different realised P/L.
#:
#: They were reporting into ONE ledger id, and that is the root cause of
#: link 9. Measured 2026-09-02 over the whole database: of 376 closed
#: ``atf_static`` trades, **368 (97.9%) were taken by this scout** and 8 by
#: the bot. The scout hardcodes ``wallet="ghost"`` and ``mode="ghost"`` and
#: publishes ``live_execution_enabled: False`` -- it has no live branch and
#: cannot spend money at all. So ``atf_static`` was granted ``live_approved``
#: on the record of an executor that can never place a live trade, while the
#: executor that CAN place one had 8 trades against the 20 promotion needs.
#:
#: The bot then reached its live entry gate 264 times in 24h and the swap
#: guard PASSED 94 of them -- every one downgraded to ghost, because the only
#: graduated strategy never produces the directives that arrive there. The
#: one strategy allowed to spend could not, and the ones that could were not
#: allowed to. Splitting the id is what makes the ledger mean what the live
#: gate reads it to mean.
SCOUT_STRATEGY_ID = "atf_static_scout"

#: The signal identity. The bot-side plugin publishes under this and it is
#: what the live gate consults, so nothing this module executes may claim it.
SIGNAL_STRATEGY_ID = "atf_static"


def _record_ghost_outcome(strategy_id: str, profit: float, symbol: str = "") -> None:
    """
    Report a closed ghost trade to the strategy ledger.

    This loop runs its own ghost cycle rather than going through
    ``bot.py``'s exit path, which is the only other place that calls
    ``StrategyLedger.record()``. Without this the outcomes were written to
    ``trading_ops`` and nowhere else: 196 closed trades over four days that
    the graduation gate never saw, so the ledger sat unchanged and no
    strategy could ever accumulate the 20 trades promotion requires.

    Callers must pass ``SCOUT_STRATEGY_ID``. See its docstring for why an
    outcome this module produced must never be filed under the id the live
    gate reads.

    Deliberately best-effort. A ledger write must never abort a trading
    cycle -- losing one outcome is recoverable, stalling the loop is not.
    """
    try:
        from trading.strategies.ledger import StrategyLedger

        StrategyLedger().record(
            strategy_id, profit=float(profit), mode="ghost", symbol=symbol
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[atf-static] ledger record failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _now() -> float:
    return time.time()


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _bool_env(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or default).strip().lower() in {"1", "true", "yes", "on"}


def _feed_price(db: Any, symbol: str, chain: str, max_age_sec: float) -> Optional[float]:
    """Representative streamed price for ``symbol``, or None if the feed is silent.

    Uses the MEDIAN of recent ticks rather than the single latest one.
    ``market_stream`` carries interleaved sources, and when one of them
    publishes a different denomination the series alternates between correct
    and wrong values on a scale of minutes. Observed 2026-08-27 against
    DexScreener ($29M liquidity) as ground truth:

        AERO-USDC  truth 0.5153  feed alternating 0.5138 and 1.14   (2.2x)
        MAMO-USDC  truth 0.01055 feed alternating 0.0105 and 0.1723 (16x)

    Comparing a good quote against whichever tick happened to land last made
    corroboration a coin flip -- it refused 100% of live signals, so no
    position could open at all. The median ignores a minority of bad ticks
    while still going silent when the whole series is wrong.
    """
    window = max(max_age_sec, 0.0)
    try:
        rows = db.recent_market_prices(symbol, chain, since_ts=_now() - window, limit=25)
    except Exception:
        rows = None
    prices: List[float] = []
    if rows:
        for row in rows:
            try:
                value = _float(row[0] if isinstance(row, (list, tuple)) else row, 0.0)
            except Exception:
                continue
            if value > 0.0:
                prices.append(value)
    if not prices:
        # Fall back to the single-tick lookup when the batch helper is absent.
        try:
            row = db.get_market_price(symbol, chain, ts=_now() - window, after=True)
        except Exception:
            return None
        if not row:
            return None
        price = _float(row[0], 0.0)
        return price if price > 0.0 else None
    prices.sort()
    mid = len(prices) // 2
    if len(prices) % 2:
        return prices[mid]
    return (prices[mid - 1] + prices[mid]) / 2.0


def _feed_is_dense_enough(db: Any, symbol: str, chain: str) -> bool:
    """Can a stop-loss actually be enforced on this symbol?

    The stop is evaluated when a tick arrives. If ticks are minutes or hours
    apart, price gaps past the stop and the realised loss is unbounded --
    measured 2026-08-27, 4 of 6 stop_loss exits breached an 8% stop (worst
    -22.2% on SOL-USDC, which had a 663-minute hole between ticks).

    Requires BOTH a minimum sample count and a recent median gap below the
    ceiling. Median rather than max: one long outage should not disqualify a
    symbol that is otherwise well covered, but a consistently thin feed should.
    """
    if not _bool_env("ATF_STATIC_REQUIRE_DENSE_FEED", "1"):
        return True
    window = _float_env("ATF_STATIC_FEED_DENSITY_WINDOW_SEC", 3600.0)
    max_gap = _float_env("ATF_STATIC_MAX_MEDIAN_TICK_GAP_SEC", 300.0)
    min_ticks = int(_float_env("ATF_STATIC_MIN_TICKS_FOR_ENTRY", 6))
    try:
        rows = db.recent_market_prices(
            symbol, chain, since_ts=_now() - window, limit=200
        )
    except Exception:
        return False
    stamps = sorted(float(r[1]) for r in (rows or []) if len(r) > 1)
    if len(stamps) < max(2, min_ticks):
        return False
    gaps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    if not gaps:
        return False
    gaps.sort()
    mid = len(gaps) // 2
    median_gap = gaps[mid] if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2.0
    if median_gap > max_gap:
        return False
    # A healthy median is not enough: the stop is breached by the WORST gap,
    # not the typical one.
    #
    # Measured 2026-08-28 on every stop_loss exit in the ledger -- each breach
    # coincided with a single long hole in the feed while the position was
    # open, even though entry-time density looked fine:
    #
    #   BSTONK-USDC    -8.39%  20 ticks @ 42s median at entry, then a 10.9min
    #                          hole during the hold (2 ticks total)
    #   BASEJUICE-USDC -8.10%  29.8min hole
    #   BASECAT-USDC   -8.52%   9.6min hole
    #
    # Those three breaches are the whole of the tail: they held ES95 at 0.0834
    # against a 0.08 guardrail and blocked live trading entirely. The median
    # test passed all three, because one long hole barely moves a median.
    #
    # So bound the tail directly. A feed that has recently gone quiet for
    # longer than the stop can survive is a feed that cannot enforce the stop,
    # regardless of how good it looks on average.
    #
    # The budget is calibrated, not guessed. Sampled across the 1h windows the
    # gate actually reads, on 2026-08-27 12:00-20:00 (an uninterrupted
    # stretch), healthy symbols carried these median max-gaps:
    #
    #   CBBTC 624s   AERO 1052s   MAMO 770s   BASECAT 567s   BSTONK 983s
    #
    # Normal jitter therefore reaches ~1000s even on feeds that are fine. A
    # 600s budget would refuse every symbol permanently -- trading one bug for
    # a worse one. 1200s clears that jitter while still catching the holes
    # that actually breached the stop (BASEJUICE 1788s, and the hold-time
    # holes that produced all three breaches).
    max_hole = _float_env("ATF_STATIC_MAX_TICK_HOLE_SEC", 1200.0)
    if max_hole > 0.0 and gaps[-1] > max_hole:
        return False
    # The feed must also be live NOW, not merely dense in aggregate: a window
    # that ended twenty minutes ago describes a feed that has already stopped.
    # Budgeted separately so disabling the hole check does not also disable
    # the staleness check -- they answer different questions.
    max_stale = _float_env("ATF_STATIC_MAX_FEED_STALENESS_SEC", 1200.0)
    if max_stale > 0.0 and stamps and (_now() - stamps[-1]) > max_stale:
        return False
    return True


def _corroborated_price(
    db: Any,
    symbol: str,
    chain: str,
    quoted: float,
) -> Optional[float]:
    """Validate a DexScreener quote against the streamed feed.

    Positions were being opened and marked purely from ``signal["price_usd"]``,
    which no guard had ever checked. Observed 2026-08-27: BASELIFE-USDC entered
    at 2.05e-07 and "exited" at 4.39e-06 for +2038% -- on a symbol with ZERO
    rows in ``market_stream``. That single fabricated fill was 81% of the
    strategy's entire net profit and carried its ghost record to graduation.

    A quote is only usable when the feed both (a) has a recent tick for the
    symbol at all, and (b) agrees with the quote to within
    ``ATF_STATIC_MAX_FEED_DEV``. Anything else is priced from nothing and must
    not become a trade -- the same rule the synthetic-tick guard already
    applies to the stream itself.

    Returns the price to use, or None to skip the symbol entirely.
    """
    if quoted <= 0.0:
        return None
    if not _bool_env("ATF_STATIC_REQUIRE_FEED_PRICE", "1"):
        return quoted
    max_age = _float_env("ATF_STATIC_FEED_MAX_AGE_SEC", 900.0)
    feed = _feed_price(db, symbol, chain, max_age)
    if feed is None:
        return None
    max_dev = _float_env("ATF_STATIC_MAX_FEED_DEV", 0.35)
    if max_dev > 0.0:
        deviation = abs(quoted - feed) / feed
        if deviation > max_dev:
            return None
    # Prefer the feed: it is the corroborated number, and marking against it
    # keeps entry and exit on the same price basis.
    return feed


def refresh_feedback_scores(*, max_age_sec: float = 6 * 3600.0) -> Dict[str, Any]:
    """
    Feed ghost/live outcomes back into ATF's next candidate scoring pass.

    This is intentionally pair-level and model-agnostic: C0D3R/ATF publishes
    candidates, the existing ghost/live machinery produces outcomes, and this
    function converts those outcomes into small scheduler knobs instead of
    hiding failures.
    """
    db = get_db()
    since = _now() - max(300.0, float(max_age_sec))
    rows = db.fetch_trades(limit=int(os.getenv("ATF_STATIC_FEEDBACK_TRADE_LIMIT", "500")), since_ts=since)
    by_symbol: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        details = row.get("details") if isinstance(row.get("details"), dict) else {}
        status = str(row.get("status") or details.get("status") or "").lower()
        action = str(row.get("action") or details.get("action") or "").lower()
        reason = str(details.get("reason") or "")
        sid = str(details.get("strategy_id") or details.get("strategy") or "")
        # Both executors of the ATF signals feed this pair-level scoring: the
        # scout's own round trips and the bot's. Splitting the ledger ids kept
        # the two apart where it decides who may spend money; here the
        # question is "how has this PAIR behaved", and both are evidence of
        # that. Matching only the bot's id would have silently emptied the
        # feedback loop the moment the scout was renamed -- 368 of the 376
        # closed trades are the scout's.
        if (
            sid not in {SIGNAL_STRATEGY_ID, SCOUT_STRATEGY_ID}
            and "ATF researched candidate" not in reason
        ):
            continue
        if action != "exit" and not status.endswith("-exit"):
            continue
        symbol = str(row.get("symbol") or details.get("symbol") or "").upper()
        if not symbol:
            continue
        try:
            profit = float(details.get("profit") or 0.0)
        except Exception:
            profit = 0.0
        ent = by_symbol.setdefault(symbol, {"symbol": symbol, "trades": 0, "wins": 0, "losses": 0, "profit": 0.0})
        ent["trades"] += 1
        ent["profit"] += profit
        if profit > 0:
            ent["wins"] += 1
        else:
            ent["losses"] += 1

    scores: Dict[str, Dict[str, Any]] = {}
    for symbol, ent in by_symbol.items():
        trades = max(1, int(ent["trades"]))
        wins = int(ent["wins"])
        win_rate = wins / trades
        profit = float(ent["profit"])
        # Conservative until there is a sample. Positive performers get more
        # allocation/priority; losers get throttled but remain visible.
        multiplier = 1.0
        priority = 0
        if trades >= 3:
            if win_rate >= 0.58 and profit > 0:
                multiplier = min(1.75, 1.0 + (win_rate - 0.5) + min(profit / 10.0, 0.5))
                priority = 8
            elif win_rate <= 0.42 or profit < 0:
                multiplier = max(0.25, 1.0 - (0.5 - win_rate) - min(abs(profit) / 10.0, 0.5))
                priority = -8
        scores[symbol] = {
            **ent,
            "win_rate": round(win_rate, 6),
            "profit": round(profit, 8),
            "allocation_multiplier": round(multiplier, 6),
            "priority": priority,
            "updated": _now(),
        }
        try:
            db.upsert_pair_adjustment(
                symbol,
                allocation_multiplier=multiplier,
                size_multiplier=max(0.25, min(1.75, multiplier)),
                priority=priority,
                details={"source": SOURCE, "feedback": scores[symbol]},
            )
        except Exception:
            pass

    payload = {"source": SOURCE, "ts": _now(), "max_age_sec": max_age_sec, "scores": scores}
    db.set_json(FEEDBACK_KEY, payload)
    return payload


def _feedback_for(symbol: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
    scores = feedback.get("scores") if isinstance(feedback, dict) else {}
    if not isinstance(scores, dict):
        return {}
    return scores.get(symbol.upper()) or {}


def _stable_source(portfolio: PortfolioState, chain: str) -> tuple[str, float]:
    portfolio.refresh(force=True)
    best_symbol = "USDC"
    best_qty = 0.0
    for (holding_chain, symbol), holding in portfolio.holdings.items():
        if holding_chain != chain.lower():
            continue
        if symbol.upper() not in STABLE_TOKENS and symbol.upper() not in {"USDBC", "USDC.E"}:
            continue
        qty = float(holding.quantity or 0.0)
        usd = float(holding.usd or qty)
        if usd > best_qty:
            best_symbol = symbol.upper()
            best_qty = qty
    return best_symbol, best_qty


def _quote_probe(
    *,
    chain: str,
    sell_token: str,
    buy_token: str,
    amount: float,
    from_address: str,
    slippage_bps: int,
    timeout_sec: int = 45,
) -> Dict[str, Any]:
    payload = {
        "chain": chain,
        "sell_token": sell_token,
        "buy_token": buy_token,
        "amount": f"{amount:.8f}",
        "from_address": from_address,
        "slippage_bps": slippage_bps,
    }
    cmd = [sys.executable, "-u", "main.py", "--action", "swap_quote", "--payload", json.dumps(payload)]
    started = _now()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
        )
        output = proc.stdout or ""
        return {
            "ok": proc.returncode == 0 and "No quote providers available" not in output,
            "returncode": proc.returncode,
            "duration_sec": round(_now() - started, 3),
            "payload": payload,
            "output_tail": output[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "duration_sec": round(_now() - started, 3),
            "payload": payload,
            "error": f"quote_timeout:{exc}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": None,
            "duration_sec": round(_now() - started, 3),
            "payload": payload,
            "error": str(exc),
        }


def _run_ghost_quote_scout(
    *,
    db: Any,
    signals: List[Dict[str, Any]],
    chain: str,
    quote_token: str,
    max_positions: int,
) -> Dict[str, Any]:
    """Ghost-only ATF entries/exits from verified quote-probed candidates."""
    if not _bool_env("ATF_STATIC_QUOTE_GHOST_SCOUT_ENABLED", "1"):
        return {"enabled": False}
    now = _now()
    try:
        positions = db.get_json(GHOST_POSITIONS_KEY) or {}
    except Exception:
        positions = {}
    if not isinstance(positions, dict):
        positions = {}
    try:
        max_hold_sec = max(60.0, min(24 * 3600.0, float(os.getenv("ATF_STATIC_GHOST_MAX_HOLD_SEC", "3600"))))
    except Exception:
        max_hold_sec = 3600.0
    try:
        stop_loss = max(0.001, min(0.50, float(os.getenv("ATF_STATIC_GHOST_STOP_LOSS", "0.08"))))
    except Exception:
        stop_loss = 0.08
    try:
        min_profit = max(0.0, min(0.50, float(os.getenv("ATF_STATIC_GHOST_MIN_EXIT_PROFIT", "0.005"))))
    except Exception:
        min_profit = 0.005

    by_symbol = {
        str(sig.get("symbol") or "").upper(): sig
        for sig in signals
        if isinstance(sig, dict) and str(sig.get("symbol") or "").strip()
    }
    events: List[Dict[str, Any]] = []
    skipped_unpriced: List[str] = []
    skipped_sparse_feed: List[str] = []

    for symbol, pos in list(positions.items()):
        if not isinstance(pos, dict):
            positions.pop(symbol, None)
            continue
        sig = by_symbol.get(str(symbol).upper())
        entry = _float(pos.get("entry_price"), 0.0)
        # An exit is only meaningful if the ENTRY was real too.
        #
        # Corroborating just the exit still books fiction when the position was
        # opened before the feed was trustworthy. Observed 2026-08-27 after the
        # cross-chain fix landed: BASENOUN-USDC held an entry of 3.08e-05 while
        # the feed's entire history spans 1.5e-04..3.5e-04, and exiting it
        # against the now-correct price booked +402% -- and SOL-USDC booked a
        # -22% "stop_loss" it never took. Ten of twelve open positions carried
        # entries no tick could support.
        #
        # Such a position is not a trade, it is a stale record. Drop it without
        # recording an outcome rather than let it reach the ledger.
        if _corroborated_price(db, symbol, chain, entry) is None:
            try:
                from services.logging_utils import log_message

                log_message(
                    "atf-static",
                    "dropped stale position %s: entry %.10g has no corroborating tick"
                    % (symbol, entry),
                    severity="warning",
                )
            except Exception:  # noqa: BLE001
                pass
            positions.pop(symbol, None)
            continue
        quoted_mark = _float(
            (sig or {}).get("price_usd"),
            _float(pos.get("last_price"), _float(pos.get("entry_price"), 0.0)),
        )
        # An exit mark decides realised P/L, so it needs the same corroboration
        # the entry did. Marking against an unchecked quote is what booked a
        # +2038% "target_hit" on a symbol the feed had never carried.
        mark = _corroborated_price(db, symbol, chain, quoted_mark)
        if mark is None:
            # Hold the position and keep the last good mark. A silent feed is
            # not a reason to realise a price nothing can confirm.
            pos["last_seen_ts"] = now
            continue
        if entry <= 0 or mark <= 0:
            continue
        age = now - _float(pos.get("entry_ts"), now)
        profit = (mark / entry) - 1.0
        target_return = _float(pos.get("target_return"), _float((sig or {}).get("expected_return"), 0.0))
        reason = ""
        if profit >= max(min_profit, target_return):
            reason = "target_hit"
        elif profit <= -stop_loss:
            reason = "stop_loss"
        elif age >= max_hold_sec:
            # The hold timer must not realise a loss.
            #
            # Closing on the clock sells at whatever the price happens to be
            # when the timer expires. Measured on the last 20 atf_static
            # trades: target_hit won 4/4 (100%) while max_hold won only 9/16
            # (56%) -- so the timer was the direct source of every losing
            # trade that was not a stop-loss.
            #
            # A position that is merely slow is not a position that is wrong.
            # The stop-loss above already bounds the downside; letting the
            # clock crystallise a small loss converts a recoverable position
            # into a realised one for no reason.
            #
            # So the timer only closes a WINNER. An underwater position keeps
            # running until it either recovers past the profit floor or hits
            # its stop. ATF_STATIC_HOLD_FORCES_EXIT=1 restores the old
            # unconditional behaviour.
            hold_forces_exit = _bool_env("ATF_STATIC_HOLD_FORCES_EXIT", "0")
            if hold_forces_exit or profit > 0.0:
                reason = "max_hold"
            else:
                # Bound how long a losing position may be carried, so a dead
                # token cannot occupy a slot indefinitely. Beyond this it is
                # closed as a stop even if the stop threshold was never hit.
                stale_sec = max(
                    max_hold_sec * 2.0,
                    _float_env("ATF_STATIC_MAX_UNDERWATER_SEC", 4.0 * 3600.0),
                )
                if age >= stale_sec:
                    reason = "stale_underwater"
        if not reason:
            pos["last_price"] = mark
            pos["last_seen_ts"] = now
            continue
        entry_ts = _float(pos.get("entry_ts"), now - age)
        details = {
            "source": SOURCE,
            "strategy_id": SCOUT_STRATEGY_ID,
            "symbol": symbol,
            "chain": chain,
            "entry_price": entry,
            "exit_price": mark,
            "profit": profit,
            "age_sec": age,
            "reason": reason,
            # The risk layer reads exits through MetricsCollector, which keys on
            # entry_ts/exit_ts and reads the exit label from "exit_reason".
            # Publishing only "reason" and burying the timestamp inside
            # "position" meant 58 of 60 exits reported as "unspecified" and none
            # of them could be paired to their own entry. Emit the field names
            # the reader actually uses; "reason" stays for existing consumers.
            "exit_reason": reason,
            "entry_ts": entry_ts,
            "exit_ts": now,
            "timestamp": now,
            "position": pos,
            "signal": sig,
        }
        db.log_trade(wallet="ghost", chain=chain, symbol=symbol, action="exit", status="ghost-exit", details=details)
        _record_ghost_outcome(SCOUT_STRATEGY_ID, profit, symbol=symbol)
        events.append({"symbol": symbol, "action": "exit", "profit": profit, "reason": reason})
        positions.pop(symbol, None)

    open_count = len(positions)
    for sig in signals:
        symbol = str(sig.get("symbol") or "").upper()
        if not symbol or symbol in positions:
            continue
        if open_count >= max(1, int(max_positions)):
            break
        quote_probe = sig.get("quote_probe") if isinstance(sig.get("quote_probe"), dict) else {}
        if not quote_probe.get("ok"):
            continue
        entry_price = _corroborated_price(
            db, symbol, chain, _float(sig.get("price_usd"), 0.0)
        )
        if not entry_price or entry_price <= 0:
            # No streamed tick to confirm the quote: refuse the entry rather
            # than open a position priced from a source nothing can check.
            skipped_unpriced.append(symbol)
            continue
        # A stop-loss is only as good as the feed that triggers it.
        #
        # The stop is checked when a tick ARRIVES, so on a sparse feed the
        # price gaps straight past it. Measured 2026-08-27: 4 of 6 stop_loss
        # exits breached the 8% stop, losing 11.2%, 11.3% and 22.2%, and
        # SOL-USDC had a 663-minute hole between ticks (14 ticks total). That
        # single -22% exit pushed tail risk to 0.095 against a 0.08 guardrail
        # and blocked live trading entirely.
        #
        # Entering a position we cannot bound the downside on is not a risk we
        # are choosing -- it is one we cannot see. Refuse it.
        if not _feed_is_dense_enough(db, symbol, chain):
            skipped_sparse_feed.append(symbol)
            continue
        target_return = max(min_profit, _float(sig.get("expected_return"), 0.0))
        position = {
            "source": SOURCE,
            "strategy_id": SCOUT_STRATEGY_ID,
            "symbol": symbol,
            "chain": chain,
            "quote_token": quote_token.upper(),
            "entry_ts": now,
            "entry_price": entry_price,
            "last_price": entry_price,
            "target_return": target_return,
            "target_price": entry_price * (1.0 + target_return),
            "confidence": sig.get("confidence"),
            "score": sig.get("score"),
            "token_address": sig.get("token_address"),
            "pair_address": sig.get("pair_address"),
            "quote_probe": quote_probe,
        }
        positions[symbol] = position
        db.log_trade(
            wallet="ghost",
            chain=chain,
            symbol=symbol,
            action="enter",
            status="ghost-entry",
            details={
                **position,
                "reason": f"ATF researched candidate quote_ok=True target={target_return:.2%}",
                "signal": sig,
            },
        )
        events.append({"symbol": symbol, "action": "enter", "target_return": target_return})
        open_count += 1

    try:
        db.set_json(GHOST_POSITIONS_KEY, positions)
    except Exception:
        pass
    if skipped_unpriced:
        # Surface the refusal instead of silently trading less. A signal the
        # feed cannot corroborate is a data problem to fix, not a candidate to
        # quietly drop.
        try:
            from services.logging_utils import log_message

            log_message(
                "atf-static",
                "refused %d uncorroborated candidate(s): %s"
                % (len(skipped_unpriced), ", ".join(sorted(set(skipped_unpriced))[:8])),
                severity="warning",
            )
        except Exception:
            pass
    if skipped_sparse_feed:
        # Surface it: a symbol we cannot stop out of is a coverage problem to
        # fix, not a candidate to silently drop.
        try:
            from services.logging_utils import log_message

            log_message(
                "atf-static",
                "refused %d candidate(s) with a feed too sparse to enforce a stop: %s"
                % (len(skipped_sparse_feed),
                   ", ".join(sorted(set(skipped_sparse_feed))[:8])),
                severity="warning",
            )
        except Exception:
            pass
    return {
        "enabled": True,
        "open": len(positions),
        "events": events,
        "skipped_unpriced": sorted(set(skipped_unpriced)),
        "skipped_sparse_feed": sorted(set(skipped_sparse_feed)),
    }


def build_static_strategy_signals(
    *,
    budget_usd: float = 20.0,
    max_positions: int = 3,
    chain: str = "base",
    quote_token: str = "USDC",
    slippage_bps: int = 100,
    probe_quotes: bool = True,
) -> Dict[str, Any]:
    """
    Research Base candidates and publish them as normal scheduler-readable
    strategy signals.

    This does not broadcast transactions. It writes:
      * watchlists.stream / watchlists.ghost entries
      * trading_ops audit rows
      * kv_store persistent ATF strategy signals
      * quote/readiness probe results when possible
    """
    chain = (chain or "base").lower()
    db = get_db()
    started = _now()
    try:
        from tools.c0d3rV2.crypto_paper_trade import select_candidates
    except Exception as exc:
        raise RuntimeError(f"Unable to load C0D3R/ATF candidate selector: {exc}") from exc

    try:
        portfolio = PortfolioState(chains=(chain,))
        wallet = portfolio.wallet
        stable_symbol, stable_qty = _stable_source(portfolio, chain)
    except Exception as exc:
        portfolio = None  # type: ignore[assignment]
        wallet = os.getenv("PRIMARY_WALLET", "")
        stable_symbol, stable_qty = quote_token.upper(), 0.0
        db.log_trade(
            wallet="ghost",
            chain=chain,
            symbol="ATF-STATIC",
            action="wallet_read",
            status="warning",
            details={"source": SOURCE, "error": str(exc)},
        )

    effective_budget = max(1.0, float(budget_usd))
    if stable_qty > 0:
        effective_budget = min(effective_budget, max(1.0, stable_qty))
    per_position_usd = effective_budget / max(1, int(max_positions))
    probe_amount = max(0.01, min(float(os.getenv("ATF_STATIC_QUOTE_PROBE_USD", "0.25")), per_position_usd))

    candidates = select_candidates(budget_usd=effective_budget, max_positions=max_positions)
    feedback = refresh_feedback_scores() if _bool_env("ATF_STATIC_FEEDBACK_ENABLED", "1") else {}
    signals: List[Dict[str, Any]] = []
    bus_actions: List[Dict[str, Any]] = []
    pairs: List[str] = []

    for idx, candidate in enumerate(candidates, start=1):
        symbol = str(candidate.symbol or "").upper()
        if not symbol or not candidate.address:
            continue
        pair_symbol = f"{symbol}-{quote_token.upper()}"
        pairs.append(pair_symbol)
        outcome = _feedback_for(pair_symbol, feedback)
        feedback_multiplier = max(0.25, min(1.75, _float(outcome.get("allocation_multiplier"), 1.0)))
        expected_return = max(0.0, min(0.15, (_float(candidate.price_change_m5) / 100.0) * 0.35 + (_float(candidate.score) * 0.04)))
        expected_return = max(0.0, min(0.15, expected_return * feedback_multiplier))
        confidence = max(0.05, min(0.9, _float(candidate.score) * feedback_multiplier))
        target_floor = max(0.005, min(0.10, _float(os.getenv("ATF_STATIC_TARGET_RETURN", "0.05"), 0.05)))
        target_return = max(0.015, min(0.15, max(expected_return, target_floor if confidence >= 0.35 else 0.015)))
        target_price = _float(candidate.price_usd) * (1.0 + target_return)
        signal = {
            "source": SOURCE,
            "ts": _now(),
            "chain": chain,
            "symbol": pair_symbol,
            "base_token": symbol,
            "quote_token": quote_token.upper(),
            "token_address": candidate.address,
            "pair_address": candidate.pair_address,
            "action": "enter",
            "strategy_id": "atf_static",
            "expected_return": round(target_return, 6),
            "target_price": target_price,
            "confidence": round(confidence, 6),
            "score": candidate.score,
            "budget_usd": round(per_position_usd, 6),
            "rationale": candidate.rationale,
            "feedback": outcome,
            "url": candidate.url,
            "liquidity_usd": candidate.liquidity_usd,
            "volume_h1": candidate.volume_h1,
            "price_usd": candidate.price_usd,
            "quote_probe": None,
        }
        if probe_quotes and wallet:
            signal["quote_probe"] = _quote_probe(
                chain=chain,
                sell_token=stable_symbol,
                buy_token=candidate.address,
                amount=min(probe_amount, max(stable_qty, probe_amount)),
                from_address=wallet,
                slippage_bps=slippage_bps,
            )
        status = "ghost_candidate_quote_ok" if (signal.get("quote_probe") or {}).get("ok") else "ghost_candidate"
        db.log_trade(
            wallet="ghost",
            chain=chain,
            symbol=pair_symbol,
            action="enter",
            status=status,
            details=signal,
        )
        signals.append(signal)
        bus_actions.append(
            {
                "action": "evaluate_atf_static_entry",
                "reason": "c0d3rv2_atf_candidate",
                "priority": 2,
                "chain": chain,
                "symbol": pair_symbol,
                "token_address": candidate.address,
                "target_usd": round(per_position_usd, 6),
                "quote_token": quote_token.upper(),
                "strategy_id": "atf_static",
                "window_sec": int(os.getenv("ATF_STATIC_BUS_WINDOW_SEC", "900")),
            }
        )

    if pairs:
        current = load_watchlists(db)
        pair_set = [p.upper() for p in pairs]
        current["stream"] = pair_set + [p for p in current.get("stream", []) if p not in pair_set]
        current["ghost"] = pair_set + [p for p in current.get("ghost", []) if p not in pair_set]
        save_watchlists(current, db=db)

    ghost_scout = _run_ghost_quote_scout(
        db=db,
        signals=signals,
        chain=chain,
        quote_token=quote_token,
        max_positions=max_positions,
    )

    payload = {
        "source": SOURCE,
        "ts": _now(),
        "duration_sec": round(_now() - started, 3),
        "chain": chain,
        "wallet": wallet,
        "quote_token": quote_token.upper(),
        "stable_source": stable_symbol,
        "stable_quantity": stable_qty,
        "budget_usd": budget_usd,
        "effective_budget_usd": effective_budget,
        "signals": signals,
        "bus_actions": bus_actions,
        "ghost_scout": ghost_scout,
        "live_execution_enabled": False,
        "live_execution_note": "Signals are ghost/scheduler inputs. Real swaps remain controlled by existing live readiness and dry-run gates.",
    }
    db.set_json(SIGNAL_KEY, signals)
    db.set_json(LATEST_KEY, payload)
    db.set_json(PENDING_BUS_KEY, bus_actions)
    db.log_trade(
        wallet="ghost",
        chain=chain,
        symbol="ATF-STATIC",
        action="strategy_publish",
        status="published" if signals else "no_candidates",
        details={k: v for k, v in payload.items() if k != "signals"},
    )
    return payload


def latest_signals(max_age_sec: float = 1800.0) -> List[Dict[str, Any]]:
    db = get_db()
    rows = db.get_json(SIGNAL_KEY) or []
    if not isinstance(rows, list):
        return []
    cutoff = _now() - max(30.0, float(max_age_sec))
    fresh = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if _float(row.get("ts")) < cutoff:
            continue
        fresh.append(row)
    return fresh


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Publish C0D3R/ATF static strategy signals into ghost trading.")
    parser.add_argument("--budget-usd", type=float, default=float(os.getenv("ATF_STATIC_BUDGET_USD", "20")))
    parser.add_argument("--max-positions", type=int, default=int(os.getenv("ATF_STATIC_MAX_POSITIONS", "3")))
    parser.add_argument("--chain", default=os.getenv("ATF_STATIC_CHAIN", "base"))
    parser.add_argument("--quote-token", default=os.getenv("ATF_STATIC_QUOTE_TOKEN", "USDC"))
    parser.add_argument("--slippage-bps", type=int, default=int(os.getenv("ATF_STATIC_SLIPPAGE_BPS", "100")))
    parser.add_argument("--no-probe-quotes", action="store_true")
    args = parser.parse_args(argv)
    payload = build_static_strategy_signals(
        budget_usd=args.budget_usd,
        max_positions=args.max_positions,
        chain=args.chain,
        quote_token=args.quote_token,
        slippage_bps=args.slippage_bps,
        probe_quotes=not args.no_probe_quotes,
    )
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
