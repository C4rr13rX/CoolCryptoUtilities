# The chaos gate's verdict is set by how often we look, not by the market

Sage (quant), pass 424, item `b3d087c7`. Measured on 24h of `market_stream` in
`storage/trading_cache.db`, node-free and read-only.

The operator asked for live money-button trades every 5-30 minutes, stretching
to 45-60 when moves are 2-5%. This is whether the gate will permit that, and it
partly refutes the item as scoped.

## 1. No live symbol reaches the band today

    python -X utf8 scripts/horizon_admissibility_census.py --hours 24

18 of 41 live symbols bound a usable horizon at all; 23 return unmeasurable,
which the layer refuses separately and which has a different fix. Of the 18:

    p10  0.48 min   p50  1.07 min   p90  3.42 min   max  3.56 min
    reaching the 5-minute band FLOOR:    0 of 18
    reaching the 60-minute band CEILING: 0 of 18

Echo's diagnosis — the population emits 3d-1w horizons and the chaos layer
refuses them — is correct. But **re-pointing the population into the 5-30
minute band does not fix it.** 30 minutes is still ~9x past the p90, so the
money-button trades would be refused too. A change to
`services/strategy_population.py` alone makes refusals fall without raising
entries, which the item's own criterion 3 correctly calls a FAIL.

## 2. Why: the horizon is denominated in bars, as an identity

`web/tradingagent/chaos.py:182-190`:

    per_step        = mean(log divergences) / horizon    # DIMENSIONLESS
    exponent_per_sec = per_step / bar_sec
    usable           = 1.0 / exponent_per_sec

so

    usable_horizon_sec = bar_sec / per_step

**linear in the sampling interval.** The 1.29x-8.12x `usable / bar_sec` ratio
measured across the 18 symbols is exactly `1 / per_step`.

## 3. The test: same prices, same day, different sampling

Resampling one price path and recomputing changes the verdict:

| symbol | 13s (raw) | 60s | 120s | 300s | 600s |
|---|---|---|---|---|---|
| AERO-USDC (1309 ticks) | 0.49 min | 2.6 | 5.8 | **17.6** | 47.7 |
| ZORA-USDC (1252) | 0.48 min | 2.8 | 5.7 | **15.7** | 39.1 |
| TAO-USDC (229) | 3.43 min | 3.1 | 7.5 | UNBOUNDED | UNBOUNDED |

A 5-minute forecast on AERO is **refused** at our raw tick rate and
**permitted** at 300s bars. Nothing about the market changed. Past ~600-900s
the estimate exceeds its own observation span and the guard at `chaos.py:209`
returns `None`, which the layer refuses as unmeasurable — so there is a
*window*, not a monotone improvement.

## 4. Which bar size opens which symbol

Every live symbol with 120+ ticks in 24h. `UNB` = unbounded, refused:

| symbol | ticks | 60s | 120s | 300s | 600s | opens at |
|---|---|---|---|---|---|---|
| AERO-USDC | 1309 | 2.6 | 5.8 | 17.6 | 47.7 | **120s** |
| ZORA-USDC | 1252 | 2.8 | 5.7 | 15.7 | 39.1 | **120s** |
| PAID-USDC | 264 | 22.7 | UNB | UNB | UNB | **60s** |
| CBBTC-USDC | 200 | 3.7 | 9.3 | UNB | UNB | **120s** |
| TAO-USDC | 229 | 3.1 | 7.5 | UNB | UNB | **120s** |
| 238855-USDC | 214 | 2.8 | 7.4 | UNB | UNB | **120s** |
| DEGEN-USDC | 147 | 4.1 | UNB | UNB | UNB | none |
| MOG-USDC | 164 | 1.3 | UNB | UNB | UNB | none |
| BASEFLY / FLYBOOK / XPL / XYO | 208-248 | UNB | UNB | UNB | UNB | none |

**Six symbols can carry a 5-minute-or-longer forecast at some bar size, and
120s opens five of them at once.** Only AERO and ZORA reach the operator's full
5-30 minute band, at 300s bars.

The four that are UNB at every bar size are the frozen feeds — separately
measured this pass at a best 30-minute move of exactly 0.00% over hours. No
sampling choice rescues a price that does not move.

## 5. The named change, and what it is not

Criterion 3 asks for a change that is **not** to the chaos layer and **not** to
`cal.horizon_scale`. This qualifies: **feed the lattice a 120s- or 300s-
resampled series instead of raw ticks.** It touches neither, and it does not
change the strategy population's logic either.

**Admissible is not profitable.** This measures whether the gate lets the trade
through, not whether the trade wins. The refusal is a sampling artifact; the
edge is untested. Conflating those is how this repo has lost money before.

And one symbol shows the two are independent: **AERO is admissible at 120s and
uneconomic anyway** — its best 30-minute move in 24h is 1.39% against a real
on-chain round trip of 0.64-4.35%. ZORA, at a 3.34% best move, is the one worth
pricing pool depth on.

## Reproduce

    python -X utf8 scripts/horizon_admissibility_census.py --hours 24

The resample tables are this, over the symbols and steps you want:

```python
import sys; sys.path.insert(0, ".")
from scripts.horizon_admissibility_census import series, _db_path
from web.tradingagent.chaos import lyapunov_horizon_sec

points = series(24.0, _db_path())["AERO-USDC"]
for step in (60, 120, 300, 600):
    sub, nxt = [], points[0][0]
    for ts, price in points:
        if ts >= nxt:
            sub.append(price)
            nxt = ts + step
    usable = lyapunov_horizon_sec(sub, step) if len(sub) >= 24 else None
    print(step, f"{usable / 60:.1f} min" if usable else "UNBOUNDED")
```
