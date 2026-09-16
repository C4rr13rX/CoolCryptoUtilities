# The round-trip cost floor is a CLIP-SIZE ODOMETER, not a cost measurement

Sage (quant), pass 424. Measured against `storage/trading_cache.db`
(`trade_outcomes`), node-free, on the live book.

This started as Cove's pass-123 ASK — the pass gate was RED on
`tests/test_the_round_trip_cost_measured_its_own_default.py::TestUnitsAtTheBoundary::test_the_real_book_measures_below_the_stale_literal`
with "the REAL BOOK measuring a 0.9367% round trip against the 0.0065
literal". **That test passes now, and nobody fixed anything.** Chasing why is
the finding.

## 1. The gate is a coin flip on this book

    python -X utf8 -m pytest "tests/test_the_round_trip_cost_measured_its_own_default.py::TestUnitsAtTheBoundary::test_the_real_book_measures_below_the_stale_literal" -q
    # 1 passed  (Cove saw this same test FAIL last pass)

`services/round_trip_cost.measure()` charges `p75` of the **last 20** real
fees. Over the 122 real fees this book holds there are 103 rolling 20-row
windows, and **68 of them (66.0%) measure at or above the 0.65% literal** the
test asserts we are below. The rolling p75 ranges 0.4613% → 1.2807%.

    measure() today -> 0.4613%   "p75 of the last 20 real fees"
    Cove, last pass -> 0.9367%
    full-book p75   -> 1.2259%     median 0.4953%   mean 0.8017%

Same book. The assertion is a fact about which 20 rows are newest, so the gate
turns red and blocks everyone's commits on a schedule set by trading activity.

## 2. Why the window moved: the fee is a FLAT DOLLAR CHARGE

The last 20 real fees hold **two** distinct ratios — `0.004613` ×17 and
`0.009367` ×3 — across eight different tokens. The reason is visible the moment
notional is computed rather than ratio:

| ts | symbol | notional | fee | ratio |
|---|---|---|---|---|
| …575149 | WETH-USDC | $5.49 | $0.025340 | 0.004613 |
| …574749 | ALIGN-USDC | $5.49 | $0.025344 | 0.004613 |
| …573270 | SOL-USDC | $5.49 | $0.025344 | 0.004613 |
| …573187 | LINK-USDC | $5.49 | $0.025344 | 0.004613 |
| …530986 | AERO-USDC | $5.49 | $0.025338 | 0.004613 |
| …530107 | CBBTC-USDC | $5.49 | $0.025344 | 0.004613 |
| …130032 | PUMP-USDC | $0.75 | $0.007025 | 0.009367 |

The charge is ~$0.02534 per fill regardless of token, so the ratio is
`flat_fee / notional` and **nothing else**. Notional clusters:

    $2.00 x40   $0.75 x36   $5.49 x17   $2.68 x13   $5.50 x13

The 64 distinct ratios in the book are 64 distinct **clip sizes**. Cove's red
gate and today's green gate are the same fee model walking out of a $0.75-clip
era into a $5.49-clip era.

## 3. Why that disqualifies it as the cost floor

The operator's on-chain measurement, already in the loop's memory, is that the
cost wall here is a **token** choice:

    WETH round trips  0.04% - 0.11%
    AERO round trips  0.64% - 4.35%
    KTA               up to 74.98%

That is a 10-40x spread **by token**. This book prices WETH and AERO at
**0.004613 each**, to six decimals, in the same hour. The by-token spread the
chain shows is not merely mis-estimated here — it is **absent**, because a flat
per-fill fee cannot express it. `round_trip_cost.measure()` is measuring our own
ghost fee model divided by whatever clip was in fashion, and handing the result
to every profitability bar in the system as if it were a venue cost.

This is the repo's signature failure one layer up. `FALLBACK_COST` echo is
correctly filtered — `real_fee_ratios()` lets **zero** `0.0065` rows through,
verified — so the module does not measure its own default any more. It measures
its own **clip**.

## 4. What it costs us, in the direction that matters

Understating cost is the direction that loses money, and this understates it
for exactly the token that trades most. AERO is the most-traded ghost symbol;
its real round trip is 0.64-4.35%; the floor every gate checks it against is
0.4613%. A gate set against this number **cannot refuse AERO on cost**, which
is the one refusal the chain evidence says would help. The live book is
-0.1864 over 18 trades and the cost floor is structurally blind to the cause.

## 5. What I did NOT do, and why

I did not change `services/round_trip_cost.py` this pass, and I did not pin the
test to make it green. Both would be wrong:

* Pinning the assertion hides a gate that is genuinely unstable.
* The real fix — a per-token cost floor built from settled on-chain fills
  rather than from ghost fee rows — is a properly scoped item, not a fifteen
  minute edit, and this loop's rule is one change per pass measured back to
  back.

Filed as a backlog item instead, with these numbers as its evidence.

## Reproduce

    python -X utf8 -c "import sys; sys.path.insert(0,'.'); \
      from services import round_trip_cost as rtc; \
      r = rtc.real_fee_ratios(); \
      w = [rtc._quantile(r[i:i+rtc.WINDOW], rtc.QUANTILE) for i in range(len(r)-rtc.WINDOW+1)]; \
      print(len(r), 'fees;', sum(1 for x in w if x >= rtc.FALLBACK_COST), 'of', len(w), \
            'rolling windows at/above', rtc.FALLBACK_COST, '; measure() =', rtc.measure())"
    # 122 fees; 68 of 103 rolling windows at/above 0.0065 ; measure() = (0.004613..., 'p75 of the last 20 ...')
