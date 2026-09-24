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
              "unpriced_default", and this is the call the live entry gate makes

The shipped cost function has NO slippage term at all. Its breakdown is
``gas_rate + pool_rate`` and nothing else, so the amendment's 13.5bp is not in
the model any gate consults. Whether it is real is UNMEASURED on WETH: WETH has
zero live fills, and the census backing every number here is flagged
``quotes_not_fills: true``.

So this script does not report "route (a) is open" or "route (a) is closed".
It reports the gap TWICE -- once against the toll the lane would CHARGE, once
against the toll the pair actually PAYS -- because those two have opposite
signs, and the difference between them is a missing function argument.

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
    print("  a) AGAINST THE TOLL THE LIVE ENTRY GATE WOULD CHARGE (symbol-blind):")
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
    print("  b) AGAINST THE TOLL WETH-USDC ACTUALLY PAYS (symbol-aware census):")
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
              f"'no clip clears' is FALSE\n"
              f"         against the symbol-aware census. It is TRUE only against the\n"
              f"         symbol-blind default the gate actually uses.")
    print()
    print("  THE TWO ROWS DISAGREE BECAUSE OF A MISSING FUNCTION ARGUMENT, NOT BECAUSE")
    print("  OF A MARKET FACT. Same function, same clip; one call passes the symbol.")


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
    print(f"  cost per round trip, CHARGED      ${clip * charged:>10.4f}   ({_bp(charged):.2f} bp)")
    print(f"  cost per round trip, WETH REAL    ${clip * real:>10.4f}   ({_bp(real):.2f} bp)")
    print(f"  overcharge per round trip         ${clip * (charged - real):>10.4f}   "
          f"({_bp(charged - real):.2f} bp, {charged / real:.1f}x)")
    print()
    print("  WHAT 20 SETTLED WETH ROUND TRIPS WOULD COST TO BUY:")
    for n_trades in (10, 20, 40):
        c_real = n_trades * clip * real
        c_chg = n_trades * clip * charged
        print(f"    {n_trades:>3} round trips   real ${c_real:>7.4f}   "
              f"charged ${c_chg:>7.4f}   ({c_chg / wallet * 100:>5.2f}% of wallet)")
    print()
    print("  THE PRICE OF THE ANSWER IS CENTS. That is the whole case for route (a),")
    print("  and it survives every finding above: 20 settled WETH round trips at the")
    print(f"  binding clip cost ${20 * clip * real:.4f} in toll to buy the FIRST live")
    print("  measurement of WETH slippage this repo has ever had -- the dominant")
    print("  unmeasured term in every cost claim on the board.")


def the_verdict(clip: float) -> None:
    section("5. VERDICT -- STATED THE WAY THE ITEM DEMANDED IT BE STATED")
    real20 = roundtrip_cost_rate(20.0, "WETH-USDC")
    print("""
  THE ITEM SAID: if the answer is 'no route is open', say it in those words and
  do not soften it. I am not able to say those words honestly, and the reason is
  a measurement, not optimism.

  ROUTE (a) IS NOT CLOSED ON THE COST AXIS. It is BLOCKED ON A MISPRICING.

    * The live entry gate prices WETH at {chg:.2f} bp at the binding clip.
    * WETH's own census entry says {rl:.2f} bp at that clip and {r20:.2f} bp at $20.
    * That is a {ratio:.1f}x overcharge, and it comes from calling
      roundtrip_cost_rate(notional) instead of roundtrip_cost_rate(notional, symbol).
    * Against the charged toll, no clip clears. Against the real one, WETH clears
      from $20 upward. The amendment's conclusion holds ONLY under the bug.

  WHAT I CANNOT CLAIM, AND WILL NOT:

    * That route (a) would be PROFITABLE. The move side (8.46bp) is the item's
      number and p696 measured that estimator ~48bp biased POSITIVE. The cost
      side is a QUOTE census, not fills. A gap computed from a biased move and a
      quoted cost is not an edge -- it is two error bars overlapping.
    * That the amendment's 13.5bp slippage is wrong. It is ABSENT from the model,
      which is not the same as refuted. WETH has zero live fills. If it is real
      and it applies to WETH, the real toll at $20 is {rl20_slip:.2f} bp and nothing clears.

  SO THE HONEST STATE IS: route (a)'s cost case turns entirely on ONE unmeasured
  number -- WETH slippage at a real fill -- and route (a) is the only instrument
  that can measure it. The 20 round trips that would settle the question cost
  ${cost20:.4f} in toll. The question has been open for 8 passes and costs cents to close.

  THE DECISION IS NOT MINE. This is a costing, not an approval, and I armed
  nothing. What a human is being asked to decide: spend ~${cost20:.2f} of a $23.18
  wallet to replace a modelled cost floor with a measured one.
""".format(
        chg=_bp(roundtrip_cost_rate(clip)),
        rl=_bp(roundtrip_cost_rate(clip, "WETH-USDC")),
        r20=_bp(real20),
        ratio=roundtrip_cost_rate(clip) / roundtrip_cost_rate(clip, "WETH-USDC"),
        rl20_slip=_bp(real20) + 13.5,
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
