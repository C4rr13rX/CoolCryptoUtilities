# 22 of 198 streamed symbols carry a second price regime, and three CLANKER ticks are the 1.25-billion-percent forecast

Iris, pass 116, item [2ac8532c]. Criterion 1 only; criteria 2, 3 and 4 are not
met and are named at the end. Corpus: `storage/trading_cache.db`, table
`market_stream`, the 7 days to 2026-09-11 04:50 UTC, 43,920 ticks over 198
symbols.

    command: python -X utf8 scripts/feed_regime_contamination_census.py

## The bar that produced the forecast

`obv_accumulation` priced a CLANKER-USDC entry at `expected_return =
12551318.65` — 1.25 billion percent — because it computes
`(recent_high - last_price) / last_price` over a window whose `recent_high`
came from a different asset's price scale. Naming the ticks:

    python -X utf8 scripts/feed_regime_contamination_census.py --symbol CLANKER-USDC

    CLANKER-USDC: 495 ticks over 7d
      3 minority-regime ticks, 7.5 decades apart
        2026-09-07 05:40:04  price=13.01897021
        2026-09-07 05:40:19  price=13.01897021
        2026-09-07 05:57:11  price=13.01897021

CLANKER trades around 1e-06 on this feed. **Three ticks out of 495 (0.6%), all
carrying the identical price 13.01897021 inside a 17-minute span on one day, sit
seven and a half decades above the other 492.** A repeated identical price is
the signature of a single upstream row being re-published, not of a move.

## Every symbol with the same defect

A symbol is flagged when its 7-day max/min ratio exceeds 100x — two decades,
far beyond any real move on this feed. Its ticks are split at the geometric
midpoint of the two extremes and the smaller side is the foreign regime.

    symbol           ticks  minor  minor%  decades  example
    AERO-WETH          318      3    0.9%     3.4  p=0.496367
    BOB-USDC           135     67   49.6%     3.1  p=0.00563539
    CBETH-CBBTC        400     11    2.8%     6.9  p=2868.31
    CBETH-WETH         373      4    1.1%     3.4  p=2687.06
    CHIP-USDC           34      6   17.6%     4.5  p=0.0525066
    CLANKER-USDC       495      3    0.6%     7.5  p=13.019
    DOGE-USDC          138     44   31.9%     2.1  p=0.000782053
    EURC-WETH          289     16    5.5%     3.4  p=1.10469
    JITOSOL-CBBTC      499     40    8.0%     4.9  p=137.728
    LAPTOP-USDC        122      5    4.1%     6.9  p=128.78
    MOONBASE-USDC      581     27    4.6%     3.2  p=1.22158
    MORPHO-WETH        296      1    0.3%     3.4  p=1.25204
    OPENAI-USDC         12      3   25.0%     3.5  p=2.582e-08
    PEPE-USDC          169     14    8.3%     2.3  p=1.79506e-08
    PEPKING-USDC        90     25   27.8%     2.5  p=1.18401e-09
    SHIB-USDC           18      3   16.7%     3.8  p=5.18381e-06
    SOL-CBBTC          276     32   11.6%     4.9  p=102.006
    SPCX-USDC          124     11    8.9%     3.9  p=2.19263e-07
    TRUMP-USDC          31      1    3.2%     4.6  p=5.26236e-05
    VIRTUAL-WETH       236      1    0.4%     3.4  p=0.682724
    VVV-WETH           357      5    1.4%     3.4  p=16.4867
    🍎-USDC             31      1    3.2%     2.7  p=3.254e-07

    contaminated symbols: 22 of 198
    foreign-regime ticks: 323 of 43920 (0.74%)

Two shapes are mixed in that list and they are not the same bug:

* **A handful of foreign ticks in an otherwise clean series** — CLANKER 0.6%,
  MORPHO-WETH 0.3%, VIRTUAL-WETH 0.4%, TRUMP 3.2%. This is the shape that
  produces an unbounded ratio, because the foreign bar becomes `recent_high`
  while `last_price` stays real.
* **A series that is genuinely two series** — BOB-USDC 49.6%, DOGE-USDC 31.9%,
  PEPKING-USDC 27.8%. A near-even split is not contamination of one regime by
  another; it is two assets, or two denominations, under one ticker, and it
  matches the existing two-regime census on COMP and BASECAT.

The `-WETH` and `-CBBTC` pairs are a third thing again: those carry a token
RATIO rather than a USD price by construction, and `STRATEGY_STABLE_QUOTE_ONLY`
already refuses entries on them. They are listed because the statistic is the
same, not because an entry can reach them.

## What this does NOT settle

* **Criterion 2, how many non-entry statistics read those bars, is not
  measured.** RSI, VWAP, stop width and `target_price` are all computed over the
  same window and none of them is bounded the way `expected_return` now is.
* **Criterion 3, where the sanitisation belongs, is not decided.** Bounding each
  consumer's output is what shipped in `make_candidate` and it is a backstop,
  not a fix: the right place is almost certainly the window, once criterion 2
  says how many consumers there are.
* **Criterion 4 needs criterion 3 first.**

The bound that IS shipped, `STRATEGY_MAX_EXPECTED_RETURN` in
`trading/strategies/base.py` (commit 0fce105), refuses 1 of 123 directive-path
entries and passes the other 122; it stops the forecast from sizing a trade and
leaves everything above untouched.
