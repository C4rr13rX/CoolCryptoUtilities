"""What a fill row's cost fields actually hold, per leg, and which are measured.

THE DEFECT THIS MODULE EXISTS TO CLOSE. ``trade_fills.details`` looked like a
cost record and was not one. Measured 2026-09-11 over all 1394 fill rows:

    field              rows     distinct   what it really was
    fee_rate           1350        7       CONFIGURED default, ghost legs only
    slippage_bps         26        1       CONFIGURED tolerance (75.0 on 26/26)
    gas_spent_native     38       32       MEASURED, both live legs
    gas_price_usd        18        7       the NATIVE TOKEN's USD price, and on
                                           5 of 18 rows the TRADED PAIR's price
    fee_cost            198      121       ghost exit only, one scalar for the
                                           whole round trip
    fee_cost_usd         18       18       live exit only, same scalar problem

So a round trip could not be split into its buy and its sell leg, for one
mechanical reason: ``live_entry`` recorded ``gas_spent_native`` but no native
USD price and no fee field at all, so the BUY leg carried no priced cost. Every
per-leg number had to be invented, and the 0.3187%-plus-$0.004047 round-trip
constant the whole horizon table rests on was derived from the 38 rows that did
carry a receipt -- five of which priced ETH gas at the price of AERO ($0.4877),
cbETH ($2836.06) or cbBTC ($80884.98).

WHAT IS MEASURABLE PER LEG, AND WHAT IS NOT. A swap receipt gives two things:
the gas actually burned, and the amount actually received. So:

  * GAS is measurable per leg, in native units from the receipt and in USD once
    the native token's own price is recorded beside it.
  * The DEX FEE is NOT separately measurable. An AMM takes its fee out of the
    output amount, so it arrives already inside ``executed_price`` and cannot be
    separated from spread or price impact without the pool's fee tier and its
    reserves at that block. Recording a configured 0.65% and calling it the fee
    is the thing this module refuses to keep doing.
  * What IS measurable, and is the whole economic cost of the fee plus spread
    plus impact together, is the REALISED SLIPPAGE: how far the price we filled
    at sat from the price we were quoted, signed so that adverse is positive.
    Both legs have recorded ``expected_price`` and ``executed_price`` on every
    row since the table existed, so this is measurable retroactively.

A leg's cost is therefore ``gas_usd + slippage_usd``, both measured, with the
DEX fee documented as inseparable rather than defaulted to a constant.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

#: A field holding a number read off a receipt or a fill.
MEASURED = "measured"
#: A field holding a number a config or an env var chose, not one we observed.
CONFIGURED = "configured"
#: A field computed from measured ones by this module.
DERIVED = "derived"
#: A field that cannot be filled, with the reason it cannot.
UNMEASURABLE = "unmeasurable"

#: Every cost-bearing field a fill row can carry: what it holds, whether it is
#: measured, and -- when it is not -- why not. The writer in ``trading/bot.py``
#: cites this module, and ``scripts/roundtrip_cost_census.py`` reads it to
#: decide which fields it is allowed to treat as evidence.
FILL_COST_FIELDS: Dict[str, Dict[str, str]] = {
    "leg": {
        "status": MEASURED,
        "holds": "'buy' or 'sell' -- which half of the round trip this row is.",
    },
    "notional_usd": {
        "status": MEASURED,
        "holds": "USD the leg moved: quote spent on a buy, quote received on a sell.",
    },
    "gas_spent_native": {
        "status": MEASURED,
        "holds": "gas burned by THIS leg, in the chain's native token, from the receipt.",
    },
    "native_token_price_usd": {
        "status": MEASURED,
        "holds": (
            "USD price of the CHAIN'S NATIVE TOKEN (ETH on base), so that "
            "gas_spent_native can be valued. Was named gas_price_usd, which "
            "described neither the asset nor the quantity: it is not the gas "
            "price (wei per unit of gas) and never was."
        ),
    },
    "native_token_price_source": {
        "status": MEASURED,
        "holds": (
            "where native_token_price_usd came from -- 'price_book', "
            "'route_native', 'fallback_env' or 'fallback_constant'. 'route_native' "
            "means it was taken off the traded route, which is only valid when "
            "native is what the route sells; a row reading 'route_native' on any "
            "other pair carries the 2026-09-04 defect and is not evidence."
        ),
    },
    "gas_cost_usd": {
        "status": DERIVED,
        "holds": "gas_spent_native * native_token_price_usd, for THIS leg.",
    },
    "realised_slippage_bps": {
        "status": MEASURED,
        "holds": (
            "(executed_price - expected_price) / expected_price in basis points, "
            "signed so ADVERSE is POSITIVE on both legs: a buy that filled above "
            "its quote and a sell that filled below it both read positive. This "
            "is the fee, the spread and the price impact together, which is the "
            "only form they can be observed in."
        ),
    },
    "slippage_cost_usd": {
        "status": DERIVED,
        "holds": "notional_usd * realised_slippage_bps / 10000, for THIS leg.",
    },
    "slippage_tolerance_bps": {
        "status": CONFIGURED,
        "holds": (
            "the worst fill the router was ALLOWED to accept "
            "(LIVE_TRADE_SLIPPAGE_BPS, default 75). It was called slippage_bps, "
            "which read as what we paid; it is what we permitted, and it was "
            "75.0 on 26 of 26 rows because nobody ever changed the default."
        ),
    },
    "dex_fee_usd": {
        "status": UNMEASURABLE,
        "holds": (
            "NOT RECORDED, and cannot be. An AMM deducts its fee from the output "
            "amount, so it is already inside executed_price and cannot be "
            "separated from spread or impact without the pool's fee tier and its "
            "reserves at that block. realised_slippage_bps carries it, bundled."
        ),
    },
    "fee_rate": {
        "status": CONFIGURED,
        "holds": (
            "the round-trip cost RATE the ghost lane charged itself, from "
            "services.roundtrip_cost. Not an observed fill: 0.0065 on 1253 of "
            "1350 rows. Kept because the ghost book's P/L was computed with it "
            "and rewriting it would change the book retroactively."
        ),
    },
    "fee_cost": {
        "status": CONFIGURED,
        "holds": (
            "ghost exit only: notional * fee_rate for the WHOLE round trip, "
            "booked against the sell leg. Not splittable -- it never was two "
            "numbers."
        ),
    },
    "fee_cost_usd": {
        "status": MEASURED,
        "holds": (
            "live exit only: the gas the exit leg burned, in USD. Despite the "
            "name it is gas, not a DEX fee, and it is equal to gas_cost_usd on "
            "the sell leg."
        ),
    },
}

#: The USD band the chain's native token must fall in for a recorded price to be
#: believable. This is the guard whose absence let ETH gas be valued at the
#: traded pair's price for five live round trips. It is deliberately wide -- it
#: is a nonsense filter, not a price oracle.
NATIVE_PRICE_BANDS: Dict[str, Tuple[str, float, float]] = {
    "ethereum": ("ETH", 200.0, 20000.0),
    "base": ("ETH", 200.0, 20000.0),
    "arbitrum": ("ETH", 200.0, 20000.0),
    "optimism": ("ETH", 200.0, 20000.0),
    "blast": ("ETH", 200.0, 20000.0),
    "linea": ("ETH", 200.0, 20000.0),
    "scroll": ("ETH", 200.0, 20000.0),
    "polygon": ("POL", 0.05, 20.0),
    "bsc": ("BNB", 20.0, 5000.0),
    "avalanche": ("AVAX", 1.0, 500.0),
}


def native_price_is_plausible(chain: str, price: Any) -> Tuple[bool, str]:
    """Could ``price`` be the USD price of ``chain``'s native token?

    Returns ``(True, "")`` when it is inside the band, or ``(False, reason)``.
    An unknown chain returns True with an empty reason -- this refuses nonsense,
    it does not refuse chains it has not been taught.
    """
    try:
        value = float(price)
    except (TypeError, ValueError):
        return False, "not a number"
    if not (value > 0.0):
        return False, "not positive"
    band = NATIVE_PRICE_BANDS.get(str(chain or "").lower())
    if band is None:
        return True, ""
    symbol, low, high = band
    if low <= value <= high:
        return True, ""
    return False, f"${value:.6f} is outside {symbol}'s ${low:g}-${high:g} band on {chain}"


def realised_slippage_bps(
    *, expected_price: Any, executed_price: Any, leg: str
) -> Optional[float]:
    """How far the fill sat from the quote, in bps, ADVERSE POSITIVE on both legs.

    A buy that filled above its quote paid more; a sell that filled below its
    quote received less. Both are adverse, so both read positive, and the two
    legs of a round trip can be added together without a sign convention per
    leg. Returns None when either price is missing or non-positive, because a
    zero quote makes the ratio meaningless rather than zero.
    """
    try:
        expected = float(expected_price)
        executed = float(executed_price)
    except (TypeError, ValueError):
        return None
    if not (expected > 0.0) or not (executed > 0.0):
        return None
    sign = 1.0 if str(leg).lower() == "buy" else -1.0
    return sign * (executed - expected) / expected * 10_000.0


def leg_cost_usd(details: Mapping[str, Any]) -> Dict[str, Any]:
    """Split one fill row into the costs it actually measured.

    Returns gas, slippage and their total in USD, plus ``measured`` -- False
    when the row predates the instrumentation or its native price fails the
    plausibility band, so a caller can report the measured subset honestly
    instead of averaging a fiction into it.
    """
    leg = str(details.get("leg") or "").lower()
    if leg not in {"buy", "sell"}:
        mode = str(details.get("mode") or "")
        leg = "buy" if mode.endswith("_entry") else "sell" if mode.endswith("_exit") else ""

    gas_native = details.get("gas_spent_native")
    native_price = details.get("native_token_price_usd", details.get("gas_price_usd"))
    chain = details.get("chain") or details.get("_chain") or ""
    plausible, why = native_price_is_plausible(chain, native_price)

    gas_usd: Optional[float] = None
    if gas_native is not None and native_price is not None and plausible:
        try:
            gas_usd = float(gas_native) * float(native_price)
        except (TypeError, ValueError):
            gas_usd = None

    slip_bps = details.get("realised_slippage_bps")
    if slip_bps is None:
        slip_bps = realised_slippage_bps(
            expected_price=details.get("expected_price"),
            executed_price=details.get("executed_price"),
            leg=leg or "buy",
        )

    notional = details.get("notional_usd")
    if notional is None:
        notional = details.get("quote_spent") if leg == "buy" else details.get("quote_received")

    slip_usd: Optional[float] = None
    if slip_bps is not None and notional is not None:
        try:
            slip_usd = float(notional) * float(slip_bps) / 10_000.0
        except (TypeError, ValueError):
            slip_usd = None

    parts = [p for p in (gas_usd, slip_usd) if p is not None]
    return {
        "leg": leg,
        "notional_usd": notional,
        "gas_cost_usd": gas_usd,
        "realised_slippage_bps": slip_bps,
        "slippage_cost_usd": slip_usd,
        "total_cost_usd": sum(parts) if parts else None,
        "measured": gas_usd is not None and slip_usd is not None,
        "unmeasured_reason": (
            "" if gas_usd is not None and slip_usd is not None
            else why or ("no gas receipt" if gas_native is None else "no notional")
        ),
    }
