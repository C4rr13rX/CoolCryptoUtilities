from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _env_float(name: str, default: float, *, lo: float, hi: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except Exception:
        value = default
    return max(lo, min(hi, float(value)))


@dataclass
class TriggerDecision:
    should_exit: bool
    reason: str
    state: Dict[str, Any]


def evaluate_long_triggers(
    position: Dict[str, Any],
    *,
    price: float,
    fee_rate: float,
    now_ts: float,
    live: bool,
) -> TriggerDecision:
    """
    Deterministic bracket/OCO trigger stack for a long position.

    This does not promise market profit. It enforces the trading discipline:
    exit at target, cap losses, lock break-even after enough unrealized edge,
    and trail winners so profitable moves do not fully round-trip.
    """
    if price <= 0:
        return TriggerDecision(False, "", dict(position.get("trigger_state") or {}))

    state = dict(position.get("trigger_state") or {})
    entry = float(position.get("entry_price") or 0.0)
    if entry <= 0:
        return TriggerDecision(False, "", state)

    high = max(float(state.get("high_watermark") or entry), price)
    state["high_watermark"] = high
    held = max(0.0, now_ts - float(position.get("entry_ts", position.get("ts", now_ts)) or now_ts))
    pnl_pct = (price - entry) / entry
    high_pnl_pct = (high - entry) / entry

    target_price = float(position.get("target_price") or 0.0)
    if target_price > 0 and price >= target_price:
        return TriggerDecision(True, "take_profit_limit", state)

    stop_loss_default = 0.02 if not live else 0.015
    stop_loss = _env_float(
        "LIVE_STOP_LOSS_PCT" if live else "GHOST_STOP_LOSS_PCT",
        stop_loss_default,
        lo=0.001,
        hi=0.25,
    )
    if pnl_pct <= -stop_loss:
        return TriggerDecision(True, f"stop_loss:{pnl_pct:.4f}", state)

    # Once unrealized PnL clears fees by enough, forbid a winner from becoming
    # a fee-loss. This is a synthetic stop-limit policy in ghost, and a live
    # precondition before real swap execution.
    break_even_arm = _env_float("TRIGGER_BREAK_EVEN_ARM_PCT", 0.012, lo=0.0, hi=0.20)
    break_even_floor = fee_rate + _env_float("TRIGGER_BREAK_EVEN_BUFFER_PCT", 0.001, lo=0.0, hi=0.05)
    if high_pnl_pct >= break_even_arm:
        state["break_even_armed"] = True
    if state.get("break_even_armed") and pnl_pct <= break_even_floor:
        return TriggerDecision(True, f"break_even_lock:{pnl_pct:.4f}", state)

    # Profit lock: after a meaningful move, cash out if too much of the move
    # is given back. This is the practical stock-market trailing-stop behavior.
    profit_lock_arm = _env_float("TRIGGER_PROFIT_LOCK_ARM_PCT", 0.025, lo=0.0, hi=0.50)
    giveback = _env_float("TRIGGER_PROFIT_LOCK_GIVEBACK", 0.45, lo=0.05, hi=0.95)
    if high_pnl_pct >= profit_lock_arm:
        lock_floor = max(break_even_floor, high_pnl_pct * (1.0 - giveback))
        state["profit_lock_floor_pct"] = lock_floor
        if pnl_pct <= lock_floor:
            return TriggerDecision(True, f"profit_lock:{pnl_pct:.4f}<={lock_floor:.4f}", state)

    trailing_arm = _env_float("TRIGGER_TRAILING_ARM_PCT", 0.018, lo=0.0, hi=0.50)
    trailing_pct = _env_float("TRIGGER_TRAILING_STOP_PCT", 0.012, lo=0.001, hi=0.25)
    if high_pnl_pct >= trailing_arm:
        trailing_stop_price = high * (1.0 - trailing_pct)
        state["trailing_stop_price"] = trailing_stop_price
        if price <= trailing_stop_price and pnl_pct > fee_rate:
            return TriggerDecision(True, f"trailing_stop:{pnl_pct:.4f}", state)

    max_hold_winner = _env_float("TRIGGER_WINNER_MAX_HOLD_SEC", 3600.0, lo=60.0, hi=24 * 3600.0)
    min_net = _env_float("TRIGGER_MIN_NET_PROFIT_PCT", 0.004, lo=0.0, hi=0.20)
    if held >= max_hold_winner and pnl_pct >= fee_rate + min_net:
        return TriggerDecision(True, f"time_take_profit:{pnl_pct:.4f}", state)

    return TriggerDecision(False, "", state)

