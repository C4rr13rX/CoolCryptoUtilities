# The horizon table never charged the fixed cost leg — audit of [15cc71d4]

Pass 111, Jet (auditor), 2026-09-10. Subject: the pass-107 finding "a perfect
direction call loses money at 5 and 10 minutes", published in commit `187f0d9`
and filed as item [15cc71d4].

Instrument: `scripts/head_vs_realised_census.py --horizon-table`.
Corpus: `storage/trading_cache.db`, `organism_snapshots` predictions matched
against `market_stream` prices, `--tolerance-sec 120`, `--max-abs-return 0.20`.

## 1. The headline reproduces, and gets stronger

Re-ran the published command on a fresh rolling 24h window:

    python -X utf8 scripts/head_vs_realised_census.py --hours 24 --horizon-table

      horizon      n   median|ret|%   %ticks>cost   mean|ret|%   perfect-oracle net%
          5min   5476       0.0451         14.6%      0.1840            -0.1347
         10min   5756       0.0743         20.3%      0.2515            -0.0672
         15min   5410       0.1116         25.5%      0.3121            -0.0066
         30min   5397       0.1806         36.2%      0.4693            +0.1506
         60min   4963       0.3370         51.4%      0.7691            +0.4504
        120min   4658       0.5892         63.1%      1.3681            +1.0494

Against pass 107's published row set (`5min -0.1192`, `10min -0.0364`,
`15min +0.0198`). The 5- and 10-minute ceilings are negative in both readings.
**The finding survives the audit.** It is the only number in it that does.

## 2. The bug: the fixed leg was quoted and never charged

`_print_horizon_table` printed a header reading

    cost basis 0.3187% of notional + 0.004047 fixed, measured from receipts

and then computed both of its cost-bearing columns as

    over = count(v > ROUND_TRIP_PCT)          # 0.3187 only
    net  = mean - ROUND_TRIP_PCT              # 0.3187 only

`ROUND_TRIP_FIXED` is referenced nowhere but the header string. The fixed leg
is DOLLARS and only becomes a percent when divided by the clip, and the clip
this system sends is $5:

    0.004047 / 5.00 = 0.0809% of notional

So the all-in cost is **0.3996%**, not 0.3187%, and every published row was
0.0809 points too generous. The direction of the error flatters the tape, which
is the dangerous direction.

It also accepted `--clip`, `--pct-cost` and `--fixed-cost` and ignored all three
in the table path. An operator asking "what if the cost halved?" — which is
exactly the decision this item asks them to make — would have been shown the
unchanged default table and had no way to tell.

### What the correction moves

Measured back-to-back on one tape, not computed by hand — the all-in column is
a second run of the same command after the fix:

| horizon | ceiling billing 0.3187 only | ceiling billing 0.3996 all-in | verdict |
|---|---|---|---|
| 5min  | -0.1347 | -0.2152 | negative either way |
| 10min | -0.0672 | -0.1479 | negative either way |
| 15min | -0.0066 (pass 107: **+0.0198**) | **-0.0883** | **flips** |
| 30min | +0.1506 | +0.0682 | still positive |
| 60min | +0.4504 | +0.3667 | still positive |

`%ticks>cost` falls with it: 5min 14.6% -> 12.0%, 60min 51.4% -> 45.4%. The
"a majority of moves outrun the fee from 60m out" line from `e5a9527` is no
longer true at the $5 clip: 45.4%, not 51.4%.

**The single quotable consequence: 15 minutes does not clear the cost floor
either.** Pass 107 published `+0.0198` at 15min and that number is wrong. The
first horizon whose unreachable ceiling is positive is 30 minutes, and the
first at which the MEDIAN tick clears is still ~45–60 minutes.

Fixed in this pass. `total_cost_pct(pct, fixed, clip)` now does the arithmetic
in one place, the table honours the three flags, and the header prints the
all-in number.

## 3. The census could only ever re-read the same tape

Acceptance criterion 2 asks for the table over "a second, different regime
window". There was no way to ask for one: `--hours` is a lookback from `now`,
so every invocation reads the same rolling window and a re-run is a re-read.
Shipped `--end-hours-ago`, which moves the window end back; `--split-hours` now
splits relative to the window end rather than to `now`, or with an offset the
split would fall outside the window and put every row in one era while still
printing two.

Also shipped `window_regime()`, printed in the table header, so a run states
which regime it measured instead of leaving the reader to assume. It measures
per-symbol first-to-last drift **over the symbols the sample actually used**.
Classifying over every symbol in `market_stream` reads FLAT on every window in
this database — most symbols there hold a seed price and never tick — and the
MEAN over them reaches 8.0e7 percent on one contaminated row. Median over the
sampled symbols, with the same `--max-abs-return` guard the table uses.

## 4. Criterion 3 cannot be met from the books, and that is a finding

The criterion asks where the 0.3187% goes per leg, "measured from receipts
rather than assumed". Measured, not assumed:

* `trade_outcomes.fee_cost` is a **scalar**. Its `details` column carries
  `reason / mode / strategy_id / remaining_size / retained_profit /
  accounting_version` and no cost breakdown at all, over all 216 rows.
* `services/round_trip_cost.py` derives 0.3187% as the p75 of `fee_cost /
  notional`. It is one aggregate ratio with no legs in it.
* `trade_fills` carries a `fee_rate` of **0.0038615 (0.38615%) per leg** on
  1350 of 1394 rows — a configured constant, identical on every row, not a
  measurement.
* Only **38 of 1394 fills carry a real gas receipt**, all `base`, all 122–180
  hours old.
* `slippage_bps` reads exactly **75 on every live entry**. That is the
  configured tolerance, not a measured fill. **The adverse-fill leg is not
  measured anywhere**, so the largest plausibly-reducible component is the one
  the books cannot price.
* `gas_price_usd` is misnamed and inconsistent: on `AERO-USDC` exits it reads
  `2477.16`, `2498.78`, and `0.5018` in the same book. Two of those are an
  ETH/BTC price and one is the AERO price. Nothing should be divided by that
  field until it is straightened out.

**Consequence for the operator's decision.** "Cut the cost" cannot currently be
costed, because the only leg with real receipts is gas (38 rows, 5–7 days old)
and the two candidate levers — the 0.38615% per-leg fee rate and the adverse
fill — are both constants written into the rows rather than anything measured.
Choosing cost reduction means first instrumenting the legs.

## 5. The second window: the ceiling is negative there too, and it is WORSE

    python -X utf8 scripts/head_vs_realised_census.py --hours 24 --end-hours-ago 72 --horizon-table

      window: 24.0h ending 72.0h ago   REGIME FLAT (median drift -0.199% over 21 symbols)

      horizon      n   median|ret|%   %ticks>cost   mean|ret|%   perfect-oracle net%
          5min    776       0.0068          6.2%      0.1089            -0.2907
         10min    701       0.0229         10.1%      0.1513            -0.2484
         15min    623       0.0311         11.9%      0.1797            -0.2199
         30min    579       0.0864         22.5%      0.4159            +0.0163
         60min    510       0.1969         33.9%      0.6279            +0.2283
        120min    406       0.4221         51.5%      0.6023            +0.2026

A disjoint 24h window, three days back, no overlap with the window in §1.
**5, 10 and 15 minutes are negative there too — by more than double.** 30
minutes is barely positive (+0.0163 against +0.0682). The 5/10-minute finding
is not an artefact of one tape.

Two things to say plainly rather than let them read as confirmation:

* **This is not an UP window, and this database does not contain one.** The
  criterion asked "ideally an UP window". Both windows read FLAT on median
  per-symbol drift (-0.184% and -0.199%). A scan of 24h windows every 12h
  across the whole 372h of `market_stream` did not return inside this pass
  (offset windows load ~8x the rows against a database under production write
  load), so I cannot state that no UP window exists — only that neither window
  I measured is one. **Criterion 2 is met on "a second, different window" and
  NOT met on "an UP window".** The item stays open on that half.
* **n falls 7x** (5476 -> 776). The older window has far fewer predictions, so
  its rows are thinner. It agrees in sign and direction, not in weight.

That the second window is *more* negative is the expected shape, not a
surprise: it is a quieter tape, and a quieter tape loses to a fixed toll by
more. It is evidence about the cost floor, not about the regime.

## Reproduce

    python -X utf8 -m pytest tests/test_the_horizon_ceiling_pays_the_fixed_cost_leg.py -q
    python -X utf8 scripts/head_vs_realised_census.py --hours 24 --horizon-table
