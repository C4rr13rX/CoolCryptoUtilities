# The evidence bill, priced per symbol: we have been buying evidence on the worst symbol available

Pass 430, Echo (senior). Measurement only. New tool:
`scripts/symbol_resolvability_census.py`.

```
python -X utf8 scripts/symbol_resolvability_census.py --horizon-sec 3600 --edge-bp 20
```

## The question nobody had priced

This repo picks its tradeable universe on COST. That is the right first cut and
it is not the only one. A symbol also has a **detection floor**: the smallest
per-trade edge its own dispersion lets you tell from zero in a realistic number
of trades. `n = (1.96 * sd / e)^2` — quadratic in sd, so doubling a symbol's
dispersion quadruples the evidence bill.

Horizon is fixed at **3600 seconds of wall clock** for every symbol, not at a
bar count: the corpora run 166s to 345600s bars, so "12 bars" is 33 minutes on
one file and 48 days on another, and dispersion grows with holding period.

## The bill (USDC-quoted, 1-hour hold, pricing a 20bp edge at 2 SE)

| symbol | sd / trade | trades to prove 20bp | smallest edge 30 trades can resolve |
|---|---|---|---|
| WBTC-USDC | 0.500% | **24** | **18bp** |
| XRP-USDC | 0.651% | 41 | 23bp |
| ETH-USDC | 0.676% | 44 | 24bp |
| **WETH-USDC** | **0.685%** | **45** | **25bp** |
| SOL-USDC | 0.921% | 82 | 33bp |
| LINK-USDC | 0.927% | 83 | 33bp |
| COMP-USDC | 1.024% | 101 | 37bp |
| ARB-USDC | 1.062% | 108 | 38bp |
| UNI-USDC | 1.107% | 118 | 40bp |
| CRV-USDC | 1.169% | 131 | 42bp |
| **AERO-USDC** | **1.458%** | **204** | **52bp** |
| CBBTC-USDC | 1.750% | 294 | 63bp |
| VIRTUAL-USDC | 2.216% | 472 | 79bp |
| VELVET-USDC | 8.396% | 6,770 | 300bp |

USDT-USDC (sd 0.011%) is in the tool's output and is **excluded here as
degenerate**: it is a stablecoin peg, so there is no direction to be right
about. The tool's "cheapest evidence" line names it and that line is
misleading — read the table, not the line.

## What it says, and it is one sentence

**AERO is 4.5x the evidence bill of WETH, and we have been spending our
evidence budget on AERO.**

1. **The graduation bar is priced wrong for almost every symbol we trade.**
   The bar is 20 tradeable ghost trades. Thirty trades resolve 18bp on the best
   symbol available and 52bp on AERO. Every edge this repo has ever claimed is
   5-20bp. On AERO the bar cannot see an edge three times larger than any we
   have measured. This is `graduation-bar-has-no-statistical-power` with a
   price tag per symbol.

2. **AERO loses on BOTH axes at once.** Round-trip cost AERO 0.64-4.35% vs WETH
   0.04-0.11% (operator, 2026-09-11), and now evidence cost AERO 204 trades vs
   WETH 45. It is our most-traded ghost symbol and it is the worst available
   choice on cost and on resolvability simultaneously. There is no trade-off to
   balance here — the two criteria point at the same answer.

3. **The cost argument and the evidence argument converge on three symbols.**
   WBTC, ETH and WETH are the cheapest to trade AND the cheapest to prove. That
   convergence is the useful part: restricting the universe is not a compromise
   between two goals, it advances both.

4. **It reframes "raise the evidence rate".** We have been trying to get more
   trades per day. Switching AERO -> WETH is worth **4.5x the trade rate** for
   free, because it cuts the trades-needed rather than raising the trades-made.
   45 trades at the current 3.9 tradeable/day is ~12 days. 204 is ~52 days.

## Caveat, stated plainly

This prices the DETECTION of an edge, not its existence. It says where an edge
*could be proven if there is one*; it does not say there is one. A symbol with a
low detection floor and no edge still pays nothing. Pair this with
`predictability-and-fillability-are-anticorrelated`: the predicate lift we can
actually fill is +0.88bp, which is below every floor in this table. That is not
a reason to ignore the table — it is the reason the table matters, because it
says how much tape any claim of a fillable edge needs before it is believable.

---

# ADDENDUM: what our evidence budget actually bought

Measured after the table above, on `storage/trading_cache.db` (`trade_outcomes`,
read-only `immutable=1`), the 14 days ending 2026-09-16: **196 outcome rows,
174 ghost / 22 live.**

Where the evidence budget went:

| symbol | rows | % of 14d budget | trades needed for 20bp | proofs bought |
|---|---|---|---|---|
| **AERO-USDC** | **50** | **25.5%** | 204 | **0.245** |
| COMP-USDC | 12 | 6.1% | 101 | 0.119 |
| CBBTC-USDC | 10 | 5.1% | 294 | 0.034 |
| VVV-USDC | 6 | 3.1% | 303 | 0.020 |
| VIRTUAL-USDC | 5 | 2.6% | 472 | 0.011 |
| **WETH-USDC** | **4** | **2.0%** | 45 | 0.089 |

Also in the 14 days, unpriced because no USDC corpus was in the measured set:
BASECAT 17, BSTONK 15, CBETH 8, CBXRP 6, CBADA 5, DRB 5, CBZEC 4, PUMP 4. They
are **unpriced, not zero** — BSTONK is the symbol whose stop cannot bind and
BASECAT is a known stub, so their dispersion is almost certainly worse than
AERO's, which would make the conclusion below stronger rather than weaker.

## The number

Across the priced 87 rows (44.4% of the 14-day budget):

```
proofs actually bought : 0.517
same 87 rows on WETH   : 1.933      (3.7x more)
```

**And AERO alone: the 50 AERO rows we bought in 14 days purchased 0.245 of a
proof. The same 50 rows spent on WETH would have been 1.111 — a complete,
finished proof with room to spare.**

We did not fail to reach the graduation bar in those 14 days. We reached 25% of
it, four times over, on a symbol where the bar cannot see.

## Why this is the strongest form of the argument

The usual objection to restricting the universe is that it costs evidence RATE:
fewer symbols, fewer trades per day, slower to the bar. This addendum prices
that objection and it does not survive. The restriction would have to cut our
trade count by more than 3.7x before it lost, and AERO is 25.5% of the budget —
removing it cannot cost 3.7x. The rate objection is the right question and the
answer is that the trade is strongly favourable.

That is still not proof the restriction wins: the 22 live rows are where real
money moves, 9 of them are AERO, and rate must be measured after the change on
real tape, not inferred here. That measurement is the acceptance criterion on
[6c361f13] and it has not been run.
