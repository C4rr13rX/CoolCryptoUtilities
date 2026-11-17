"""Authoritative, idempotent trading accounting for dashboard and gates."""
from __future__ import annotations

import math
import os
from typing import Any, Dict

ACCOUNTING_VERSION = 2
USD_STABLE_QUOTES = frozenset({
    "USDC", "USDT", "DAI", "USDBC", "USDC.E", "BUSD", "EURC", "TUSD", "FDUSD",
})


def is_usd_accounting_pair(base_token: str, quote_token: str) -> bool:
    """True only when the existing P&L formula produces USD-denominated output."""
    base = str(base_token or "").strip().upper()
    quote = str(quote_token or "").strip().upper()
    return bool(base and quote and base not in USD_STABLE_QUOTES and quote in USD_STABLE_QUOTES)


def is_usd_accounting_symbol(symbol: str) -> bool:
    """Same question asked of a ``BASE-QUOTE`` string.

    The pair form above is what ``trading/bot.py`` calls once a decision has
    already been computed; this is the form the SELECTOR needs, so a pair that
    can never produce a USD P&L is refused before it is given a price stream.

    Split raw rather than through ``trading/data_stream.py:_split_symbol``:
    that applies TOKEN_NORMALIZATION (WETH->ETH, USDBC->USDC), and normalising
    USDBC to USDC would turn a pair this predicate must judge on its literal
    tokens into a different pair. ``USD_STABLE_QUOTES`` already lists both
    spellings.

    A malformed symbol (no quote leg, or more than two legs) is refused. It
    cannot be judged, and ``is_usd_accounting_pair`` would refuse it at
    decision time anyway -- returning True here would only move the refusal
    later and spend a stream slot reaching it.
    """
    parts = [part for part in str(symbol or "").strip().upper().split("-") if part]
    if len(parts) != 2:
        return False
    return is_usd_accounting_pair(parts[0], parts[1])


def validate_outcome_math(
    *,
    entry_price: float,
    exit_price: float,
    quantity: float,
    gross_profit: float,
    fee_cost: float,
    net_profit: float,
    base_token: str,
    quote_token: str,
    tolerance: float = 1e-8,
) -> tuple[bool, str]:
    values = [entry_price, exit_price, quantity, gross_profit, fee_cost, net_profit]
    if not all(math.isfinite(float(value)) for value in values):
        return False, "non_finite_value"
    if entry_price <= 0.0 or exit_price <= 0.0 or quantity <= 0.0:
        return False, "non_positive_price_or_quantity"
    if fee_cost < 0.0:
        return False, "negative_fee"
    if not is_usd_accounting_pair(base_token, quote_token):
        return False, "pnl_currency_not_usd"
    expected_gross = (float(exit_price) - float(entry_price)) * float(quantity)
    scale = max(1.0, abs(expected_gross), abs(float(gross_profit)))
    if abs(expected_gross - float(gross_profit)) > max(tolerance, tolerance * scale):
        return False, "gross_profit_mismatch"
    expected_net = float(gross_profit) - float(fee_cost)
    scale = max(1.0, abs(expected_net), abs(float(net_profit)))
    if abs(expected_net - float(net_profit)) > max(tolerance, tolerance * scale):
        return False, "net_profit_mismatch"
    return True, "valid"


def _safe_float(value: Any) -> float:
    try:
        result = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def trading_accounting_snapshot(db: Any) -> Dict[str, Any]:
    """Return one population for P&L, wins, losses, and open positions."""
    state = db.load_state() or {}
    ghost_state = state.get("ghost_trading") if isinstance(state, dict) else {}
    ghost_state = ghost_state if isinstance(ghost_state, dict) else {}
    positions = ghost_state.get("positions") if isinstance(ghost_state.get("positions"), dict) else {}

    ghost = dict(db.trade_outcome_summary("ghost"))
    live = dict(db.trade_outcome_summary("live"))
    ghost_open_exposure: Dict[str, float] = {}
    live_open_exposure: Dict[str, float] = {}
    for raw_sym, pos in positions.items():
        if not isinstance(pos, dict):
            continue
        sym = str(pos.get("symbol") or raw_sym or "").upper()
        notional = max(0.0, _safe_float(pos.get("size")) * _safe_float(pos.get("entry_price")))
        if str(pos.get("mode") or "ghost").lower() == "live":
            if sym and notional > 0.0:
                live_open_exposure[sym] = live_open_exposure.get(sym, 0.0) + notional
        else:
            if sym and notional > 0.0:
                ghost_open_exposure[sym] = ghost_open_exposure.get(sym, 0.0) + notional

    ghost_open_count = sum(
        1 for pos in positions.values()
        if not isinstance(pos, dict) or str(pos.get("mode") or "ghost").lower() != "live"
    )
    live_open_count = sum(
        1 for pos in positions.values()
        if isinstance(pos, dict) and str(pos.get("mode") or "").lower() == "live"
    )
    ghost["open"] = ghost_open_count
    ghost["total"] = int(ghost["closed"]) + int(ghost["open"])
    live["open"] = live_open_count
    live["total"] = int(live["closed"]) + int(live["open"])

    ghost_gross = _safe_float(ghost.get("gross_profit"))
    ghost_fee_cost = _safe_float(ghost.get("fee_cost", ghost.get("fees")))
    verified_profit = _safe_float(ghost.get("net_profit"))
    ghost_exposure = float(sum(ghost_open_exposure.values()))
    default_bankroll = _safe_float(os.getenv("GHOST_SIM_PER_CHAIN_USD", "25.0")) or 25.0
    initial_bankroll = _safe_float(ghost_state.get("initial_bankroll"))
    if initial_bankroll <= 0.0:
        initial_bankroll = default_bankroll
    ghost_balance = max(0.0, initial_bankroll + verified_profit - ghost_exposure)
    focus_chain = (os.getenv("LIVE_FOCUS_CHAIN") or "base").strip().lower() or "base"

    ghost["gross_profit"] = ghost_gross
    ghost["fee_cost"] = ghost_fee_cost
    ghost["fees"] = ghost_fee_cost
    ghost["initial_bankroll"] = initial_bankroll
    ghost["exposure"] = ghost_exposure
    ghost["active_exposure"] = ghost_open_exposure
    ghost["balance"] = ghost_balance
    ghost["sim_quote_balances"] = {f"{focus_chain}:USDC": ghost_balance}
    live["exposure"] = float(sum(live_open_exposure.values()))
    live["active_exposure"] = live_open_exposure

    version = int(ghost_state.get("accounting_version") or 0)
    cached_profit = _safe_float(ghost_state.get("total_profit"))
    cached_bank = _safe_float(ghost_state.get("stable_bank"))
    cache_consistent = version >= ACCOUNTING_VERSION and math.isclose(
        cached_profit + cached_bank,
        verified_profit,
        rel_tol=1e-9,
        abs_tol=1e-8,
    )
    quarantine = ghost_state.get("legacy_quarantine")
    return {
        "version": ACCOUNTING_VERSION,
        "source": "trade_outcomes",
        "ghost": ghost,
        "live": live,
        "recent_outcomes": db.fetch_trade_outcomes(limit=20),
        "integrity": {
            "valid": cache_consistent,
            "cache_consistent": cache_consistent,
            "state_accounting_version": version,
            "legacy_quarantined": isinstance(quarantine, dict),
            "legacy_cached_profit": cached_profit,
            "legacy_cached_bank": cached_bank,
            "verified_profit": verified_profit,
            "reason": "valid" if cache_consistent else "legacy_or_divergent_aggregate_excluded",
        },
    }
