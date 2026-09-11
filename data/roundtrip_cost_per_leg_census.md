# The round trip's cost, split into its buy leg and its sell leg

Pass 117, Gale, 2026-09-11. Item [3af205c4], split out of [15cc71d4] by Jet in
pass 113 because it is the part that does not need the operator.

**Corpus.** All 1394 rows of `trade_fills` in `storage/trading_cache.db`, paired
by `trade_id` into (buy, sell). 183 ghost round trips, 12 live round trips, and
2 round trips excluded because their legs are in different lanes.

**Reproduce.** `python -X utf8 scripts/roundtrip_cost_census.py`

---

## What the four named fields actually held

Measured before any change, over all 1394 rows:

| field | rows | distinct values | what it really was |
|---|---|---|---|
| `fee_rate` | 1350 | 7 | CONFIGURED default. 0.0065 on 1253 rows. Ghost legs only. |
| `slippage_bps` | 26 | **1** | CONFIGURED tolerance: 75.0 on 26 of 26. |
| `gas_price_usd` | 18 | 7 | the NATIVE TOKEN's USD price — and on 5 of 18, the TRADED PAIR's. |
| `fee_cost` | 198 | 121 | ghost exit only, one scalar for the whole round trip. |

The item's own description said `fee_rate` was 0.0038615 on 1350 of 1394 rows.
It is not, any more: the dominant value is now **0.0065 on 1253 rows**, with
0.0038615 on 80. Both are configured defaults, so the conclusion stands and the
number quoted in the item was stale.

### Why the round trip could not be split, mechanically

The two live legs recorded different things:

| field | live_entry | live_exit |
|---|---|---|
| `gas_spent_native` | 20/20 | 18/18 |
| the native token's USD price | **0/20** | 18/18 |
| any fee field at all | **0/20** | 18/18 |

The buy leg had a receipt and nothing to value it with. Every per-leg number had
to be invented, so the cost was booked as one scalar against the exit.

---

## What is measurable per leg, and what is not

A swap receipt gives two things: the gas burned, and the amount received. So:

* **Gas is measurable per leg** — native units from the receipt, USD once the
  native token's own price is recorded beside it.
* **The DEX fee is NOT separately measurable, and is now documented as such with
  the reason.** An AMM deducts its fee from the output amount, so it arrives
  already inside `executed_price` and cannot be separated from spread or price
  impact without the pool's fee tier and its reserves at that block. Recording a
  configured 0.65% and calling it the fee is what this closes.
* **Realised slippage is measurable per leg, and carries the fee bundled** —
  `(executed_price - expected_price) / expected_price`, signed so that adverse is
  positive on both legs. Both prices have been on every row since the table
  existed, so this is measurable retroactively over the whole book.

A leg's cost is therefore `gas_usd + slippage_usd`, both measured.
`services/fill_cost.py` carries that schema with a
measured / configured / derived / unmeasurable status on every field, and
`trading/bot.py` cites it at each of the four writers.

---

## The leg split

### The 183 ghost round trips — 0% buy, 100% sell, BY CONSTRUCTION

Over 183 paired ghost round trips, **100% of the recorded cost sits on the sell
leg and 0% on the buy leg**, because the ghost writer books one round-trip
scalar (`fee_cost = notional * fee_rate`) against the exit and charges the entry
nothing. Ghost fills at the feed price by construction, so measured slippage is
`+0.00` bps on the buy and `-0.00` on the sell across all 183, and no gas is
recorded on either leg. That 0/100 is not a measurement of where cost falls; it
is the shape of the writer. It is the honest answer for the only lane with more
than 100 round trips, and it is why the live lane below is the evidence.

### The 12 live round trips — measured, and the buy leg dominates

10 of the 12 have a priced receipt on both legs.

| | buy leg | sell leg | buy share |
|---|---|---|---|
| gas, USD (median, n=10) | 0.002536 | 0.002157 | **54.0%** |
| realised slippage, bps (median, n=12) | **+21.73** | +6.04 | **78%** |
| slippage, USD at the median $0.75 clip | +0.001630 | +0.000453 | |

**The buy leg is where the cost is.** It carries 54% of the gas and 3.6x the
sell leg's slippage. No scalar booked against the exit could ever have shown
that, and it is the number the operator's "reduce the cost" option needs: the
lever is the entry fill, not the exit.

Ten of the twelve buy legs are **back-filled** — their gas was recorded but not
priced, so they are valued at the sell leg's native price from the same round
trip, minutes later on the same chain. That is stated in the output rather than
hidden, and rows written from this pass forward carry their own price.

---

## `gas_price_usd` is renamed, and it was wrong twice

It never held a gas price. It held the **native token's USD price**, and on 5 of
the first 18 live exits it held the **traded pair's** price:

    AERO-USDC   $0.4877      AERO's price
    CBETH-USDC  $2836.06     cbETH's price
    CBETH-USDC  $2840.46     cbETH's price
    AERO-USDC   $0.5018      AERO's price
    CBBTC-USDC  $80884.98    cbBTC's price   <- $0.4136 of gas on a $3.00 trade

against a real ETH price of $2498.78. `scripts/reprice_gas_in_native_token.py`
repaired `trade_outcomes` in pass ~92; **`trade_fills` was never repaired**, so
the instrumentation table the cost constant is derived from still carries all
five. This is the "corrections reach only one book" failure again, and it is why
the guard had to be a reader-side one as well as a writer-side one.

The field is now `native_token_price_usd`, with one writer and zero readers of
the old name outside the documented historical fallback. It is joined by
`native_token_price_source` — `price_book`, `route_native`, `fallback_env` or
`fallback_constant` — because **the value alone cannot show the defect**:
$2836.06 is a perfectly plausible ETH price and is in fact cbETH's. Two guards
are needed and neither is sufficient:

* the plausibility band refuses AERO's $0.49 and cbBTC's $80,884 — 3 of 5;
* the provenance field catches the two cbETH rows, which no band can refuse
  without also refusing real ETH prices.

The census **quarantines** rows that fail either, and never averages them into a
constant. Two are still live in the book today and are named in the output.

---

## The published constant, re-derived — and it MOVED

`services/roundtrip_cost.py` publishes `cost_usd = 0.004047 + 0.003187 * notional`.
Re-derived from the receipts, with the mispriced rows quarantined and the native
price taken as the median of the 15 believable rows ($2498.78):

| | published | re-derived from the rows | moved |
|---|---|---|---|
| fixed, USD per round trip | 0.004047 | **0.004570** | **+12.9%** |
| proportional rate | 0.003187 | **0.002777** | **−12.9%** |

Gas in native units: buy median 9.8874e-07 (n=20), sell median 8.4025e-07
(n=18), so the fixed part is `(9.8874e-07 + 8.4025e-07) * 2498.78 = $0.004570`.
The rate is the two legs' median realised slippage added: `(21.73 + 6.04) / 10000`.

**Neither number is changed in this pass, deliberately.** The fixed part is
understated by 12.9%, which is the direction that loses money, and raising it to
$0.004570 would also raise `min_viable_notional_usd` from $2.23 to $2.52. The
rate is overstated by 12.9%, and lowering it would LOOSEN the entry bar on the
evidence of 12 round trips at a $0.75 median clip. Changing the entry gate's
cost constant is a money-path change on a 10-row sample and deserves its own
item with its own QA, not a side effect of an instrumentation pass. Filed as a
follow-up.

---

## A second defect, found while pairing the legs

**Two round trips entered LIVE and exited in GHOST** — real money spent, a
simulated exit booked against it:

    CBBTC-USDC    live_entry -> ghost_exit   ts=1788453645  2:CBBTC-USDC:490d9726…
    BSTONK-USDC   live_entry -> ghost_exit   ts=1788455198  2:BSTONK-USDC:32fff2d5…

Both are 2026-09-03, and `tests/test_a_live_position_never_exits_in_simulation.py`
passes today, so the guard that refuses this is in place and these are pre-fix
artifacts. They are excluded from both lanes by the census rather than averaged
into either, because a round trip whose legs are in different lanes is not a cost
sample. Named here so nobody re-derives them as a live cost.

---

## What this does NOT establish

It does not establish that a cost reduction is worth taking. It establishes
where the cost is — the buy leg, 54% of gas and 78% of slippage — over 12 live
round trips, which is the whole live history this account has. 183 ghost round
trips cannot help, because the ghost lane charges a configured scalar to one leg
and measures nothing. The next real improvement to this number is more live
round trips, not more analysis of these twelve.
