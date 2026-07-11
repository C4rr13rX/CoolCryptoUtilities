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


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _now() -> float:
    return time.time()


def _bool_env(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or default).strip().lower() in {"1", "true", "yes", "on"}


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
        if sid != "atf_static" and "ATF researched candidate" not in reason:
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

    for symbol, pos in list(positions.items()):
        if not isinstance(pos, dict):
            positions.pop(symbol, None)
            continue
        sig = by_symbol.get(str(symbol).upper())
        mark = _float((sig or {}).get("price_usd"), _float(pos.get("last_price"), _float(pos.get("entry_price"), 0.0)))
        entry = _float(pos.get("entry_price"), 0.0)
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
            reason = "max_hold"
        if not reason:
            pos["last_price"] = mark
            pos["last_seen_ts"] = now
            continue
        details = {
            "source": SOURCE,
            "strategy_id": "atf_static",
            "symbol": symbol,
            "chain": chain,
            "entry_price": entry,
            "exit_price": mark,
            "profit": profit,
            "age_sec": age,
            "reason": reason,
            "position": pos,
            "signal": sig,
        }
        db.log_trade(wallet="ghost", chain=chain, symbol=symbol, action="exit", status="ghost-exit", details=details)
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
        entry_price = _float(sig.get("price_usd"), 0.0)
        if entry_price <= 0:
            continue
        target_return = max(min_profit, _float(sig.get("expected_return"), 0.0))
        position = {
            "source": SOURCE,
            "strategy_id": "atf_static",
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
    return {"enabled": True, "open": len(positions), "events": events}


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
