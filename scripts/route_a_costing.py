#!/usr/bin/env python
"""Cost route (a): hand-approve ONE strategy onto the live lane at a minimum clip.

WHAT THIS ANSWERS, and it is not what the item expected.

The item [efa1a134] was amended with a prediction: the toll converges to
``pool 4.08bp + slippage 13.5bp = 17.58bp`` as clip -> infinity, against a best
measured move of 8.46bp, so "there is no clip -- not $6, not $20, not $1000 --
at which the cheapest pair's best horizon clears its own toll. DO NOT EXPECT
ROUTE (a) TO SHOW A PROFIT."

That prediction is wrong on this repo's own cost model, and it is wrong in a
specific and fixable way. There are THREE asymptotic tolls in use here and they
span 15x:

    2.08 bp   services.roundtrip_cost.roundtrip_cost_rate(n, "WETH-USDC")
              -- the census entry for WETH, symbol-AWARE
   17.58 bp   the item's amendment (pool 4.08 + a 13.5bp slippage term)
   31.87 bp   roundtrip_cost_rate(n) with NO symbol -- source
              "unpriced_default"

A CORRECTION THIS SCRIPT MADE TO ITSELF, LEFT IN BECAUSE IT CHANGES THE ANSWER.
An earlier version of this file claimed the 31.87bp blind default is what the
live entry gate charges WETH, citing blocked item [4a3b3902]. That is FALSE and
the call site says so:

    trading/bot.py:2276   return roundtrip_cost_rate(notional, symbol)

[4a3b3902] was fixed in p695; the docstring above that line says "SYMBOL MATTERS
AT LEAST AS MUCH AS SIZE, and until p695 this lane threw it away." Sweeping every
caller, the only remaining symbol-blind ones are scripts (directive_path_census,
weth_moneybutton_direction), not the production lane. WETH is in
``census_symbols()``, so the gate charges WETH its REAL rate. The blind column
below is kept because it is what an UNPRICED symbol still gets -- the census
holds 9 symbols and the book trades many more -- but it is NOT WETH's price.

The shipped cost function has NO slippage term at all. Its breakdown is
``gas_rate + pool_rate`` and nothing else, so the amendment's 13.5bp is not in
the model any gate consults. Whether it is real is UNMEASURED on WETH: WETH has
zero live fills, and the census backing every number here is flagged
``quotes_not_fills: true``.

So this script does not report "route (a) is open" or "route (a) is closed".
It reports the gap against the toll WETH actually pays, and finds that the
binding constraint is neither cost accuracy nor signal but CLIP SIZE against a
$23.18 wallet.

Run:  python scripts/route_a_costing.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.roundtrip_cost import (  # noqa: E402
    ghost_clip_usd,
    min_viable_notional_usd,
    roundtrip_cost_breakdown,
    roundtrip_cost_rate,
)

# The best measured forward move on the cheapest pair, from the item's own
# amendment: WETH at a two-hour horizon, the best of 18 of 18 horizon cells on
# the three cheapest pairs. It is NOT re-measured here -- it is the item's
# number, carried through so the comparison is against what the filer asserted
# rather than against a number this script chose to suit itself.
#
# It is also the WEAKEST input in this file. p696 measured the forward-return
# estimator carrying a ~48bp positive sampling bias on 9 of 12 symbols, because
# its coverage samples FEED UPTIME. Treat 8.46bp as an upper bound on the move,
# not a point estimate.
BEST_MEASURED_MOVE_BP = 8.46
BEST_MOVE_LABEL = "WETH @ 2h (item amendment; p696 says the estimator is ~48bp biased POSITIVE)"

CLIPS = (0.75, 2.0, 6.0, 20.0, 100.0, 1000.0, 1_000_000.0)
PAIRS = ("WETH-USDC", "AERO-USDC", "CBBTC-USDC")


def _bp(rate: float) -> float:
    return rate * 1e4


def _rule(char: str = "-", n: int = 78) -> None:
    print(char * n)


def section(title: str) -> None:
    print()
    _rule("=")
    print(title)
    _rule("=")


def the_clip_the_lane_would_actually_use() -> float:
    """Resolve the clip, from the same places the lane resolves it.

    pipeline.py:6117 reads LIVE_MIN_CLIP_USD, then raises it to
    ``min_viable_notional_usd()`` if that is larger. bot.py:2112 carries a
    separate 0.75 floor in the capital plan. Print all of them rather than
    picking one silently -- an operator deciding route (a) needs to know which
    number binds.
    """
    section("1. WHAT CLIP WOULD ROUTE (a) ACTUALLY TRADE AT?")
    env_min = float(os.getenv("LIVE_MIN_CLIP_USD", "10"))
    micro = float(os.getenv("LIVE_MICRO_MIN_CLIP_USD", "0.05"))
    viable = float(min_viable_notional_usd())
    ghost = float(ghost_clip_usd())
    bot_plan_floor = 0.75  # trading/bot.py:2112, capital_plan.min_clip_usd

    print(f"  LIVE_MIN_CLIP_USD (.env)          ${env_min:>10.4f}")
    print(f"  LIVE_MICRO_MIN_CLIP_USD           ${micro:>10.4f}   (micro-mode branch only)")
    print(f"  min_viable_notional_usd()         ${viable:>10.4f}   (raises the floor if larger)")
    print(f"  ghost_clip_usd()                  ${ghost:>10.4f}   (what the ghost book is priced at)")
    print(f"  bot.py:2112 capital_plan floor    ${bot_plan_floor:>10.4f}")

    clip = max(env_min, viable)
    print()
    print(f"  ==> BINDING CLIP: ${clip:.4f}")
    print(f"      (LIVE_MIN_CLIP_USD, not raised -- viable floor ${viable:.4f} is below it)"
          if env_min >= viable else
          f"      (raised from ${env_min:.4f} by the viable-notional floor)")
    return clip


def the_three_tolls(clip: float) -> dict:
    """Print the toll-vs-clip curve, symbol-blind and symbol-aware.

    This is the measurement the item's amendment did not have. The amendment
    asserted ONE asymptote. There are three, and the gate uses the worst.
    """
    section("2. THE TOLL, MEASURED ACROSS CLIPS -- SYMBOL-BLIND vs SYMBOL-AWARE")
    header = "  clip".ljust(16) + "BLIND".rjust(10) + "".join(p.split("-")[0].rjust(11) for p in PAIRS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for n in CLIPS:
        row = f"  ${n:>12,.2f}" + f"{_bp(roundtrip_cost_rate(n)):>9.2f}bp"
        for p in PAIRS:
            row += f"{_bp(roundtrip_cost_rate(n, p)):>9.2f}bp"
        print(row)

    blind = roundtrip_cost_breakdown(clip)
    print()
    print(f"  WHY THEY DIFFER -- the breakdown at the binding clip (${clip:.2f}):")
    print()
    print(f"    SYMBOL-BLIND   gas {_bp(blind['gas_rate']):>7.2f}bp   "
          f"pool {_bp(blind['pool_rate']):>7.2f}bp   = {_bp(blind['rate']):>7.2f}bp"
          f"   source={blind['source']}")
    out = {"blind": blind, "pairs": {}}
    for p in PAIRS:
        d = roundtrip_cost_breakdown(clip, p)
        out["pairs"][p] = d
        flag = "  QUOTES-NOT-FILLS" if d.get("quotes_not_fills") else ""
        print(f"    {p:<14} gas {_bp(d['gas_rate']):>7.2f}bp   "
              f"pool {_bp(d['pool_rate']):>7.2f}bp   = {_bp(d['rate']):>7.2f}bp"
              f"   source={d['source']}{flag}")
    print()
    print("    NEITHER BREAKDOWN HAS A SLIPPAGE TERM. gas + pool is the whole model.")
    print("    The amendment's 13.5bp slippage is not in the function any gate calls,")
    print("    and on WETH it is unmeasurable: ZERO live WETH fills exist.")
    return out


def the_gap(clip: float) -> None:
    """The item's actual question, answered twice because the answer flips sign."""
    section("3. DOES THE BEST MEASURED MOVE CLEAR ITS OWN TOLL?")
    print(f"  best measured move: {BEST_MEASURED_MOVE_BP:.2f} bp   [{BEST_MOVE_LABEL}]")
    print()
    print("  a) AGAINST THE SYMBOL-BLIND DEFAULT -- this is what an UNPRICED symbol")
    print("     gets. It is NOT WETH's price: bot.py:2276 passes the symbol (fixed p695).")
    print()
    print("     clip".ljust(20) + "charged".rjust(11) + "move".rjust(10) + "gap".rjust(11) + "   verdict")
    any_clears_blind = False
    for n in CLIPS:
        t = _bp(roundtrip_cost_rate(n))
        gap = BEST_MEASURED_MOVE_BP - t
        any_clears_blind |= gap > 0
        print(f"     ${n:>12,.2f}{t:>9.2f}bp{BEST_MEASURED_MOVE_BP:>8.2f}bp{gap:>9.2f}bp"
              f"   {'CLEARS' if gap > 0 else 'refused'}")
    print()
    print(f"     ==> {'some clip clears' if any_clears_blind else 'NO CLIP CLEARS. The amendment is RIGHT here.'}")

    print()
    print("  b) AGAINST THE TOLL WETH-USDC ACTUALLY PAYS -- AND IS ACTUALLY CHARGED:")
    print()
    print("     clip".ljust(20) + "real".rjust(11) + "move".rjust(10) + "gap".rjust(11) + "   verdict")
    first_clearing = None
    for n in CLIPS:
        t = _bp(roundtrip_cost_rate(n, "WETH-USDC"))
        gap = BEST_MEASURED_MOVE_BP - t
        if gap > 0 and first_clearing is None:
            first_clearing = n
        print(f"     ${n:>12,.2f}{t:>9.2f}bp{BEST_MEASURED_MOVE_BP:>8.2f}bp{gap:>9.2f}bp"
              f"   {'CLEARS' if gap > 0 else 'refused'}")
    print()
    if first_clearing is None:
        print("     ==> NO CLIP CLEARS even on the real toll. Route (a) is closed on cost.")
    else:
        print(f"     ==> CLEARS FROM ${first_clearing:,.2f} UPWARD. The amendment's "
              f"'there is no clip -- not $6,\n"
              f"         not $20, not $1000 -- at which the cheapest pair's best horizon\n"
              f"         clears its own toll' is FALSE on this repo's own cost model.")
    print()
    print("  THE WHOLE OF ROW (b)'s SLOPE IS FIXED GAS DILUTING, AND NOTHING ELSE.")
    print(f"    gas at $6.00  {_bp(roundtrip_cost_breakdown(6.0, 'WETH-USDC')['gas_rate']):>6.2f} bp"
          f"   pool {_bp(roundtrip_cost_breakdown(6.0, 'WETH-USDC')['pool_rate']):>5.2f} bp (flat)")
    print(f"    gas at $20.00 {_bp(roundtrip_cost_breakdown(20.0, 'WETH-USDC')['gas_rate']):>6.2f} bp"
          f"   pool {_bp(roundtrip_cost_breakdown(20.0, 'WETH-USDC')['pool_rate']):>5.2f} bp (flat)")
    print("  So the clip is the only lever, and it has a floor at the pool fee.")


def the_dollars(clip: float) -> None:
    """What an operator is actually being asked to risk, and to learn what."""
    section("4. THE DOLLARS AN OPERATOR WOULD BE DECIDING ON")
    wallet = 23.18  # deployable stable, standing instructions
    charged = roundtrip_cost_rate(clip)
    real = roundtrip_cost_rate(clip, "WETH-USDC")

    print(f"  deployable stable wallet          ${wallet:>10.2f}")
    print(f"  binding clip                      ${clip:>10.2f}   "
          f"({clip / wallet * 100:.1f}% of the wallet per round trip)")
    print()
    print(f"  cost per WETH round trip          ${clip * real:>10.4f}   ({_bp(real):.2f} bp)")
    print(f"  the same trade at $20             ${20.0 * roundtrip_cost_rate(20.0, 'WETH-USDC'):>10.4f}   "
          f"({_bp(roundtrip_cost_rate(20.0, 'WETH-USDC')):.2f} bp)   "
          f"-- but {20.0 / wallet * 100:.1f}% of the wallet")
    print(f"  an UNPRICED symbol at this clip    ${clip * charged:>10.4f}   ({_bp(charged):.2f} bp)"
          f"  <- not WETH; the census holds 9 symbols")
    print()
    print("  WHAT SETTLED WETH ROUND TRIPS COST TO BUY, at the binding clip:")
    for n_trades in (10, 20, 40):
        c_real = n_trades * clip * real
        print(f"    {n_trades:>3} round trips   toll ${c_real:>7.4f}   "
              f"({c_real / wallet * 100:>5.2f}% of wallet in toll)")
    print()
    print("  THE PRICE OF THE ANSWER IS CENTS, AND THAT IS THE WHOLE CASE FOR ROUTE (a).")
    print(f"  20 settled WETH round trips cost ${20 * clip * real:.4f} in toll and would be the")
    print("  FIRST live measurement of WETH slippage this repo has ever had -- the")
    print("  dominant unmeasured term in every cost claim on the board.")
    print()
    print("  WHAT IS NOT CENTS: the clip that actually clears. Each open position at")
    print(f"  $20 is {20.0 / wallet * 100:.1f}% of capital, so this wallet supports ONE at a time.")


def the_verdict(clip: float) -> None:
    section("5. VERDICT -- STATED THE WAY THE ITEM DEMANDED IT BE STATED")
    real20 = roundtrip_cost_rate(20.0, "WETH-USDC")
    wallet = 23.18
    print("""
  THE ITEM SAID: if the answer is 'no route is open', say it in those words and
  do not soften it. I cannot say them honestly, and the reason is a measurement.

  ROUTE (a) IS NOT CLOSED ON THE COST AXIS, AND IT IS NOT BLOCKED BY A BUG
  EITHER. IT IS BOUNDED BY THE SIZE OF THE WALLET.

    * WETH is charged its real census rate -- bot.py:2276 passes the symbol.
    * At the ${clip:.2f} binding clip WETH costs {rl:.2f} bp against a {mv:.2f} bp best
      measured move. IT MISSES BY {miss:.2f} bp. A third of a basis point.
    * At $20 it costs {r20:.2f} bp and CLEARS BY {gain:.2f} bp.
    * The entire difference is the fixed gas term diluting. Nothing else in the
      model varies with size.
    * But $20 is {pct:.1f}% OF THE ${wal:.2f} DEPLOYABLE WALLET in a single round
      trip. The clip that clears the toll is a clip this wallet cannot
      diversify across even three open positions.

  THAT IS A CAPITAL-SIZE FINDING, NOT A CODE FINDING. No gate fix, no signal
  work and no cost rewrite moves it. The amendment was right that the clip lever
  has a floor; it was wrong about where the floor is (the pool fee, {pool:.2f} bp on
  WETH, not 17.58 bp) and wrong that no clip clears.

  WHAT I CANNOT CLAIM, AND WILL NOT:

    * That route (a) would be PROFITABLE. The move side ({mv:.2f} bp) is the item's
      number and p696 measured that estimator ~48bp biased POSITIVE. The cost
      side is a QUOTE census, not fills. A gap between a biased move and a
      quoted cost is two error bars overlapping, not an edge.
    * That the amendment's 13.5bp slippage is wrong. It is ABSENT from the model,
      which is not the same as refuted. WETH has zero live fills. If it is real
      and applies to WETH, the toll at $20 is {slip:.2f} bp and NOTHING clears at
      any size.

  SO THE HONEST STATE IS: route (a)'s cost case turns entirely on ONE unmeasured
  number -- WETH slippage at a real fill -- and route (a) is the only instrument
  in this repo that can measure it. 20 settled WETH round trips at the binding
  clip cost ${cost20:.4f} in toll.

  THE DECISION IS NOT MINE. This is a costing, not an approval, and I armed
  nothing. What a human is being asked to decide is TWO things, not one:
    (1) spend ~${cost20:.2f} to replace a modelled cost floor with a measured one; and
    (2) whether a ${wal:.2f} wallet can carry the ~$20 clip the arithmetic wants,
        or whether this loop needs more capital before any route can clear.
""".format(
        clip=clip,
        rl=_bp(roundtrip_cost_rate(clip, "WETH-USDC")),
        mv=BEST_MEASURED_MOVE_BP,
        miss=_bp(roundtrip_cost_rate(clip, "WETH-USDC")) - BEST_MEASURED_MOVE_BP,
        r20=_bp(real20),
        gain=BEST_MEASURED_MOVE_BP - _bp(real20),
        pct=20.0 / wallet * 100.0,
        wal=wallet,
        pool=_bp(roundtrip_cost_breakdown(20.0, "WETH-USDC")["pool_rate"]),
        slip=_bp(real20) + 13.5,
        cost20=20 * clip * roundtrip_cost_rate(clip, "WETH-USDC"),
    ))


def main() -> int:
    print()
    _rule("=")
    print("ROUTE (a) COSTING -- hand-approve one strategy at a minimum clip")
    print("item [efa1a134] / Sage p699 / COSTING ONLY, NOTHING ARMED")
    _rule("=")
    clip = the_clip_the_lane_would_actually_use()
    the_three_tolls(clip)
    the_gap(clip)
    the_dollars(clip)
    the_verdict(clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
