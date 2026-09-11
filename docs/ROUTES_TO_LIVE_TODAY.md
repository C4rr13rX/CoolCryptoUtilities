# Plausible routes to a live trade today

Written 2026-09-11 after reading every gate at once. The operator's standing
constraint is **correctness first, no bugs** — so nothing here proposes moving
a bar to make a number pass. Each route is judged on whether it could
*honestly* clear the gates that exist.

## The situation, stated once

Four gates block, and they are four readings of one fact:

    profit_factor         0.644      needs >= 1.5
    net_expectancy       -0.01503    needs > 0
    loss_rate             0.791      needs <= 0.6
    net profit ex-BSTONK -1.6600     needs > 0

The live book is **gross -0.0849 before a single fee** over 18 real trades, so
this is not a cost problem masking a good signal. Per strategy, nothing is
profitable; the only positive *gross* is `obv_accumulation@5d` at +0.0294,
which fees turn to -0.0342.

**No route below is likely to produce a profitable live trade today.** They are
ranked by how plausibly each could produce an HONEST pass, and the honest
answer for most is "not today". That is the finding, not a failure of the list.

---

## Route 1 — The fast-hold bucket (the only real candidate)

**The lead.** Over 14 days, split by how long a position was actually held:

    under 5 min      9 trips  +0.2766 gross  66.7% win  0.71 ticks/min
    5-15 min         5 trips  -0.0698
    15-60 min       29 trips  +0.1137
    1-4 hours       30 trips  -0.2107        <- the loss lives here
    over 4 hours     5 trips  +0.0855

    HELD <= 15 min  10 trips  +0.4588% of notional  win 50.0%
    HELD  > 15 min  64 trips  -0.0065% of notional  win 45.3%

**Net of cost, it survives** — which almost nothing else in this repo does:

    clip $6   cost 0.3862%  ->  +0.0726% of notional  = +0.0044/trade
    clip $10  cost 0.3592%  ->  +0.0996%              = +0.0100/trade
    clip $25  cost 0.3349%  ->  +0.1239%              = +0.0310/trade

**Why it is not a green light.** n=10. That is below every threshold in this
system and below the n>=30 floor the loop itself adopted. It is also a split of
*realised* trips, so it carries a selection effect: a trip may have closed fast
*because* it hit its target. The module's own docstring says so.

**What would make it honest.** `stale_exit_secs` is 900s and 86% of trips
outlive it, because exits are evaluated only when a tick arrives — a wall-clock
promise on a tick-driven schedule. Enforcing the exit properly is a *correctness
fix*, not a bar change, and it would move the population toward the bucket that
pays. Then re-measure. If the fast bucket holds at n>=30 on both an up and a
down window, it is a real candidate.

**Verdict: the best route, and it is days away, not hours.**

---

## Route 2 — Fix what is measurably broken, and let the gates answer honestly

These are bugs, so they are worth doing whatever the trading outcome. Two are
already visible in the gate output itself:

1. **The live gate judges the POOLED book** — the state line reads
   `judging: (pooled book) [43 trades]`. This is the identical defect already
   corrected for graduation: a single good strategy is invisible inside a
   pooled loss. Per-strategy judging may not change today's verdict (nothing is
   profitable per strategy either) but the gate is currently answering a
   question nobody asked.
2. **Live readiness is judging on a confusion report 1789121569s old** against
   its own 900s limit, and it logged that it had already decided a verdict on
   it. A stale metric is a claim about the present made from the past.
3. **`net profit ex-BSTONK` credits a symbol whose own book is -0.1826** once
   its implausible fill is excluded. The gate is correcting for concentration
   using a number that is itself an artifact.

**Verdict: do these regardless. They will not open the gate today.**

---

## Route 3 — The horizon question, which is the operator's to decide

Measured, and the two facts are in direct conflict:

    share of ticks whose move outruns the fee:
      5m 17.8%   15m 27.5%   30m 38.3%   60m 53.3%   120m 68.0%   240m 80.5%

    clip does NOT fix it: $10 -> $250 moves the share 27.5% -> 30.0%, because
    the 0.3187% rate is charged on notional and only the fixed leg amortises.

The standing mandate says trade in single-digit-to-tens of MINUTES. The
measurement says a majority of moves only outrun the fee from **60 minutes**
out. Those cannot both be satisfied on this feed.

Note this cuts AGAINST Route 1, and the tension is real rather than a
contradiction: the fast bucket wins on *selection* (the trips that closed fast
were the good ones), while the horizon table is about the *unconditional* tape.
Route 1 only works if the selection is causal, which n=10 cannot establish.

**Verdict: a decision, not an experiment. Either accept hour-scale round trips,
or accept that the cost floor must come down first.**

---

## Route 4 — A cheaper venue

The 0.3187% proportional rate is the binding constraint in every analysis in
this repo, and it is the one input no amount of modelling changes. Nothing has
been measured about alternative routing, pools, or chains. It is the largest
unexamined lever.

**Verdict: unexamined, and it is the thing most likely to change the answer.**

---

## What is explicitly NOT a route

- **Lowering a gate.** The four blockers are four views of a real loss. Opening
  them funds ~1300 losing trades from a $19.66 wallet.
- **Lowering `enter_threshold`.** The direction head maxes at 0.535 and is 46.7%
  accurate against a 53.4% baseline. Admitting more of it admits more losses.
- **Trading a bigger clip to beat the fee.** Arithmetically dead: 25x the
  capital buys 2.5 points of clearing share.

## The honest summary

The most plausible route to a live trade *today* is none of these. The most
plausible route to a live trade that does not lose money is Route 1, gated
behind the exit-enforcement fix in Route 2, measured to n>=30 on both windows.
That is a small number of passes away, not hours — and it is the answer the
measurements support.
