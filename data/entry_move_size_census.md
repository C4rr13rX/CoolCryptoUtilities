# Entry tests DIRECTION but never MOVE SIZE — [4d3310e7], pass 112, Iris

Corpus: `organism_snapshots`, the newest 39,820 decision cycles, **484.3 hours**
to 2026-09-10 19:57, **198 symbols**. Forward returns from `market_stream`
(339 symbols), first tick at least H later, discarded if no tick lands inside
H + 10 minutes — a gap in the feed is not a zero return.

Command:

    python -X utf8 scripts/entry_move_size_census.py --limit 40000 --mult 1.0 --mult 1.5

## What was wrong

`trading/bot.py`'s model-long entry conjunction read four quantities.
`direction_prob` and `exit_conf` are confidences, `net_margin` is the margin
head's own arithmetic, and `delta` — which **is** the model's forward expected
return (`price_mu`, a dimensionless fraction) — was read for its SIGN alone:

    and delta >= 0.0

Nothing in the conjunction asked **how far**. A correctly predicted move that
cannot pay its own round trip was admitted.

## The threshold, derived

`services/roundtrip_cost.py` measures this account's settled receipts as
`cost_usd = 0.004047 + 0.003187 * notional`, so as a fraction of notional

    c(N) = 0.003187 + 0.004047 / N

`entry_fees` in that branch is already exactly `c(N)` for the notional the
entry is about to spend, so the conjunct compares two fractions of the same
notional and introduces **no new constant**. Size dependence is the point:

| notional | c(N) |
|---|---|
| $6.00 (live clip) | 0.3862% |
| $1.22 (median implied over the corpus) | 0.6500% |
| $0.75 (ghost floor) | 0.8583% |

A flat percentage is wrong at both ends; this repo has already shipped a flat
0.65%.

## How far the tape actually moves

| horizon | median abs move | clears the 0.6500% round trip |
|---|---|---|
| 15 min | 0.1227% (n=31,302) | 24.0% |
| 30 min | 0.2132% (n=29,013) | 30.8% |

So on 76% of ticks at 15 minutes a perfectly correct direction call still loses
money. That is an arithmetic property of the cost floor, not a model defect.

## Is the condition stricter?

By construction, yes and unconditionally: `entry_fees > 0` always, so
`delta >= entry_fees` implies the `delta >= 0.0` it replaces. It can only ever
refuse; it can never admit anything the old test refused.

**The conjunct in isolation**, over the same 39,820 cycles:

| test | cycles admitted |
|---|---|
| `delta >= 0.0` (before) | 19,518 (49.0%) |
| `delta >= 1.0 x c(N)` (after) | **5,451 (13.7%)** — 72.1% fewer |
| `delta >= 1.5 x c(N)` | 5,018 (12.6%) |

**The full conjunction**, same window: 9 cycles admitted before, 9 after — no
fall. That is not the conjunct failing. The model-long path admits **9 cycles
in 484 hours** because `enter_threshold` is
`max(decision_threshold 0.89, MIN_CONFIDENCE 0.65)` and `direction_prob` has
collapsed; all 9 survivors happened to carry a `delta` far above cost. The
item's "entries admitted must FALL" criterion is not measurable on a path that
admits 9 cycles in 20 days, and it is reported here as not met rather than
redefined.

## Score on the admitted subset

| set | 15 min | 30 min |
|---|---|---|
| admitted (before == after) | n=8, mean NET **-0.3774%**, median -0.3861%, win 0% | n=6, mean NET -0.3398%, win 0% |
| every cycle (baseline) | n=31,302, **median** NET -0.6197%, win 12.7% | n=29,013, median NET -0.6182%, win 15.8% |

The admitted set's mean beats the baseline's median by ~0.24pp. **n=8 is not an
edge** and is not claimed as one — it is 8 trades over 20 days and it is still
negative in absolute terms. The baseline's *mean* gross reads +26,784%, which is
feed contamination (the known two-price-regime rows), so the median is the only
honest baseline statistic here and is the one quoted.

## The finding that matters more than the fix

`delta` is not on the tape's scale, so the cost-derived floor cannot bite on
recent data. Over the newest 5,000 cycles (19.0h):

    median |delta| 93.1649%   vs   median realised |15m move| 0.1240%
    -> 751.4x too large; |delta| overstates the move on 99.1% of 4,698 cycles
    (at 30 min: 93.7103% vs 0.1921%, 487.7x, overstates on 99.0%)

A head whose median forecast is a 93% forward return makes any cost-derived
floor vacuous — which is why the same conjunct removes 72.1% of cycles over
484h and 1.1% over the newest 19h. The conjunct is correct arithmetic on an
uncalibrated input. **`price_mu`'s scale is the binding problem, and it is
filed separately rather than patched here.**

## Not touched

No confidence threshold, margin floor, or plausibility guard was changed.
`ENTRY_MIN_MOVE_COST_MULT` defaults to 1.0, which makes the bar the measured
cost floor itself and nothing more.

## Known gap, named rather than widened

The directive-driven entry path (`directive.action == "enter"`) is an `elif`
**above** this branch, so a strategy-emitted entry never reaches the model
conjunction and this conjunct does not bind on it. That is the path
`atf_static` actually trades. Filed separately.

## Addendum — price_mu's distribution, and a wrong reading of it I then corrected ([1b0fd55f])

The heading of this section originally read "price_mu is a z-score, not a
return". That is WRONG and the correction is two subsections below. The
distribution itself stands.

Measured on the same newest 5,000 cycles, the full distribution of `delta`:

    p1   -2.1002    p25  -1.5034    p50  -0.3423    p75  +0.1295
    p95  +1.1784    p99  +1.3466    max  +1.5385    min  -2.7811

    |delta| < 1%        0.68% of cycles
    delta >= 0.99      10.84%
    delta <= -0.90     36.48%
    delta in [0.9,1.0]  2.22%

It is **not** clipped at a cap — it runs from -2.78 to +1.54 with a median of
-0.342, which is the shape of a standardised variable, not of a 15-minute
return on a feed whose median absolute move is 0.1227%. `net_margin` sits on
the same scale (median -0.3488, max +1.5320), which also explains the
separately filed "net_margin is a saturated negative on every symbol": a
z-scored target is negative more often than not.

### Correction, same pass: it is not a units bug, and the cause is already on file

I wrote "the target was normalised and never inverse transformed" above and
then read `trading/data_loader.py:68-110`, which already diagnoses this range
from direct probes of the deployed model. The z-score reading is **wrong** and
is kept here only so nobody re-derives it:

> ONE FOREIGN ROW IN SIXTY SATURATES THE MODEL, AND THAT IS THE -1.2.

`PriceVolScaleNorm` makes the price channel scale-free relative to the window's
anchor, and it works — the same model returned `price_mu -0.000620` on clean
windows at price levels 1e-4 and 1.2e4, identical to six decimals across eight
orders of magnitude. What it cannot absorb is a window whose 60 rows are not
all the same asset. Probed against the deployed model on one live ETH-USDT
window:

| served window | price_mu | net_margin |
|---|---|---|
| clean | -0.2065 | -0.2130 |
| one row 100x | -1.5111 | -1.5176 |
| one foreign row at t=30 | -1.8395 | -1.8460 |
| two assets interleaved | +1.3012 | +1.2947 |

That reproduces exactly the distribution measured above, including `net_margin`
tracking `price_mu` to within ~0.007 — which is the 0.0065 fee input, not a
second independent defect. So the two heads are not two bugs, and they are not
a missing transform: they are **one contaminated served window**.

So the real question for [1b0fd55f] is narrower and sharper than the one I
filed: `sanitize_model_price_window` already exists as the repair, and my
measurement says the contaminated signature is **still present in the newest
5,000 cycles** (median `delta` -0.342, range -2.78..+1.54, only 0.68% inside
1%). Either the sanitiser is not wired into the path that builds the served
window, or it is not catching these rows. That is a seam to verify, not a model
to retrain.

### What price_mu's target actually is — proven by reading

`trading/data_loader.py` builds the label as

    next_idx      = end                      # line 1008
    current_price = price_slice[-1]          # = closes[end - 1]
    future_price  = closes[next_idx]         # line 1110
    ret = log(future_price) - log(current_price);  mu = ret

So `price_mu`'s target is the **one-bar forward log return of the training
corpus**, and `net_margin = mu_arr - (gas_arr + tax_arr)` (line 1179) is that
same label minus 0.0065 — which is exactly why the two heads track each other
to 0.0065 in every served row, and is not a second defect.

The units are therefore right and the horizon is the open question: a clean
window's `price_mu` of -0.2065 is an ordinary one-bar log return over a long
bar and an absurd one over a 30-second tick. The entry conjunct compares
`price_mu` to a **per-round-trip** cost, so if one bar is not the holding
period, that comparison is a units error at a boundary.

**Dead end recorded so it is not repeated:** `ohlcv_datasets` has 0 rows and
`ohlcv_bars` has 0 rows in `storage/trading_cache.db`, so
`granularity_seconds` cannot be read there — the corpus moved to the
Parquet/S3 market store.
