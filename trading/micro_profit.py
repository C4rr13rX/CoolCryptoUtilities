"""Unit-safe viability checks for very small crypto trades."""
from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class MicroProfitDecision:
    viable: bool
    notional_usd: float
    gross_profit_usd: float
    estimated_cost_usd: float
    net_profit_usd: float
    minimum_net_profit_usd: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_micro_profit(
    *,
    notional_usd: float,
    gross_return: float,
    variable_cost_rate: float,
    fixed_cost_usd: float = 0.0,
    minimum_net_profit_usd: float = 0.02,
) -> MicroProfitDecision:
    """Return whether an opportunity earns the requested dollars after costs.

    Rates and dollar amounts deliberately have separate arguments.  This keeps
    a two-cent profit floor from accidentally becoming a 2% or 20% margin.
    """
    notional = max(0.0, float(notional_usd))
    gross = max(0.0, float(gross_return)) * notional
    costs = max(0.0, float(variable_cost_rate)) * notional + max(0.0, float(fixed_cost_usd))
    net = gross - costs
    floor = max(0.0, float(minimum_net_profit_usd))
    if notional <= 0.0:
        reason = "no_notional"
    elif gross_return <= variable_cost_rate:
        reason = "edge_does_not_cover_variable_costs"
    elif net + 1e-12 < floor:
        reason = "net_profit_below_dollar_floor"
    else:
        reason = "profitable_after_costs"
    return MicroProfitDecision(
        viable=reason == "profitable_after_costs",
        notional_usd=notional,
        gross_profit_usd=gross,
        estimated_cost_usd=costs,
        net_profit_usd=net,
        minimum_net_profit_usd=floor,
        reason=reason,
    )


#: chain -> (measured_at_ts, usd). Small, and keyed by chain, so a bot trading
#: two chains does not charge one of them the other's gas.
_GAS_CACHE: Dict[str, Tuple[float, float]] = {}
_GAS_CACHE_TTL_SEC = 300.0


def _clear_roundtrip_gas_cache() -> None:
    """Test hook. Production never needs this -- the TTL handles staleness."""
    _GAS_CACHE.clear()


def roundtrip_gas_usd(
    db: Any,
    chain: str,
    *,
    samples: int = 40,
    now: Optional[float] = None,
) -> float:
    """USD of gas a full round trip broadcasts: the entry swap plus the exit.

    Gas is a FIXED dollar cost. Every cost gate in this bot charged a
    variable RATE (``fees`` = 0.0065 of notional) and nothing else, so the
    one cost that does not shrink with the clip was the one nobody modelled.

    Measured on the five settled live round trips of 2026-09-03/04
    (``trade_outcomes``, mode=live, status=closed, gas re-read from the
    receipts in commit a5a3385):

        AERO  $0.75 notional   gross +0.001240   gas 0.004319   net -0.003079
        CBETH $0.44 notional   gross +0.014984   gas 0.004590   net +0.010394
        CBETH $0.32 notional   gross -0.002397   gas 0.003046   net -0.005443
        AERO  $0.75 notional   gross -0.001045   gas 0.003872   net -0.004917
        CBBTC $3.00 notional   gross -0.004102   gas 0.012776   net -0.016878

    Gross over all five is **+0.008680**; gas is **-0.028603**. The direction
    calls were right and the round trips still lost, because gas is 0.43% to
    1.03% of these notionals and no gate subtracted it.

    Sources, in order:

    1. MEASUREMENT -- the median ``fee_cost`` of recent settled live outcomes
       on this chain. ``fee_cost`` on a live row is realized gas in USD and
       nothing else (the DEX fee and slippage are inside the fill prices, so
       they land in ``gross_profit``). This self-calibrates as gas moves.
    2. ``ROUNDTRIP_GAS_USD_<CHAIN>`` then ``ROUNDTRIP_GAS_USD`` -- the
       bootstrap for a chain that has never settled a live trade, because a
       ghost book that charges no gas graduates strategies into a game that
       does.
    3. **0.0** -- and deliberately so. A missing measurement must degrade to
       exactly today's behaviour. The other plausible fallback, the existing
       ``_estimate_gas_cost`` (ESTIMATED_GAS_NATIVE=0.001 ETH = $2.50, a
       mainnet-shaped number), would price a base round trip at $5 and refuse
       every exit on the chain -- stranding positions instead of costing them.

    Returns USD (float, >= 0, finite). Never raises: a cost estimate must not
    be what stops the bot from closing a position.
    """
    chain_key = str(chain or "").strip().lower() or "unknown"
    stamp = float(now if now is not None else time.time())
    cached = _GAS_CACHE.get(chain_key)
    if cached and (stamp - cached[0]) < _GAS_CACHE_TTL_SEC:
        return cached[1]

    value = _measure_roundtrip_gas_usd(db, chain_key, samples=samples)
    if value <= 0.0:
        value = _env_roundtrip_gas_usd(chain_key)
    value = value if math.isfinite(value) and value > 0.0 else 0.0
    _GAS_CACHE[chain_key] = (stamp, value)
    return value


def _env_roundtrip_gas_usd(chain_key: str) -> float:
    suffix = "".join(ch if ch.isalnum() else "_" for ch in chain_key).upper()
    for name in (f"ROUNDTRIP_GAS_USD_{suffix}", "ROUNDTRIP_GAS_USD"):
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    return 0.0


def _measure_roundtrip_gas_usd(db: Any, chain_key: str, *, samples: int) -> float:
    """Median realized round-trip gas, or 0.0 when it cannot be measured."""
    try:
        rows = db.fetch_trade_outcomes(wallet="live", limit=max(1, int(samples)))
    except Exception:  # noqa: BLE001 - see the docstring: never raise
        return 0.0
    if not rows:
        return 0.0
    # A gas cost larger than a tenth of the trade is not a gas reading, it is
    # a repricing bug. The CBBTC exit of 2026-09-04 booked $0.4136 of "gas" on
    # a $3.00 trade (13.8%) because it valued ETH at the cbBTC price; that row
    # has since been corrected, but the guard is what stops the next one from
    # poisoning every cost gate at once.
    max_fraction = _positive_env_float("ROUNDTRIP_GAS_MAX_FRACTION", 0.10)
    observed = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "").lower() != "closed":
            continue
        if str(row.get("chain") or "").strip().lower() != chain_key:
            continue
        fee = _finite_positive(row.get("fee_cost"))
        if fee <= 0.0:
            continue
        notional = _finite_positive(row.get("entry_price")) * _finite_positive(row.get("quantity"))
        if notional > 0.0 and fee > notional * max_fraction:
            continue
        observed.append(fee)
    if not observed:
        return 0.0
    observed.sort()
    return observed[len(observed) // 2]


def _positive_env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0.0 else default


def _finite_positive(value: Any) -> float:
    try:
        result = float(value if value is not None else 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result) or result <= 0.0:
        return 0.0
    return result
