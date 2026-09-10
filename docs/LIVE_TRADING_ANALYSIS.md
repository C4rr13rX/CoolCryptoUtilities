# What actually stands between here and live trading

Measured 2026-09-10 by a second reader, independently of the loop. Every number
below was reproduced from the databases rather than taken from a report.

## The headline

**The direction head has real skill and a broken level.** Those are different
defects, they need opposite fixes, and confusing them has already cost passes.

    AUC (ranking skill)      5m 0.5605   15m 0.5936   30m 0.5640
    direction_prob p50       0.0495
    direction_prob max       0.5346
    entry needs              direction_prob >= 0.5

The head *orders* candidates better than a coin at every horizon. It also
assigns them all a probability near 0.05, so the entry test — which needs
`>= 0.5` — sees essentially nothing pass.

## The binding constraint, isolated

Entry requires six conjuncts (`trading/bot.py:7313-7320`). Over 862 cycles in
3h, each was measured separately:

    direction_prob >= 0.5        1  (0.1%)   <- THE WALL
    exit_conf      >= 0.5      308  (35.7%)
    net_margin     >= 0        327  (37.9%)
    delta          >= 0        332  (38.5%)
    ALL FOUR                     1  (0.1%)

Every other conjunct passes about 36-39% of the time. One passes 0.1%. That is
the whole reason 867 of 867 cycles are holds.

Note this **changed** during the day: `net_margin` was the binding constraint
12h earlier (its maximum was negative across every symbol) and has since
recovered to +1.082. It is no longer the wall. Any plan written against the
earlier reading is stale.

## Model or market? MODEL. Not arguable.

    head says P(up)                  ~0.05
    market delivered (tick-to-tick)   0.61 up over 3h, 28 symbols
    directional accuracy              46.7%
    always-up baseline                53.4%
    head inverted                     53.3%

The head is *worse than a coin* while claiming ~95% confidence in the down
direction. Inverting it merely recovers the baseline, which means the sign is
not reversed — **the level carries no information at all**. It is a constant
bearish bias sitting on top of a weak but genuine ranking signal.

A flat tape would produce ~0.50. Median |15-min return| was 0.2233%, so the
tape moved. The head is the defect.

## The trap underneath, which no level fix will solve

    median |15-min return|            0.2233%
    round-trip cost                   0.3187% + $0.004047 fixed
    ticks that move further than cost 37.5%

**On 62.5% of ticks a perfect direction call still loses money.** Fixing the
head to fire correctly is necessary and not sufficient; entry must also select
for *expected move size*, not just direction. This is the same cost-floor law
recorded elsewhere in this repo, arriving from a third direction.

## Regime honesty

Top-decile entry at a 15-min horizon, net of cost:

    -2h   up-share 70.1%   net +0.0257%   UP window
    -4h   up-share 58.1%   net +0.0339%   UP window
    -6h   up-share 24.3%   net -1.0038%   DOWN window

Net-positive in 2 of 2 up windows, 0 of 1 down. **That is not an edge yet.** A
long-only rule flatters itself in an up window, and this repo has twice shipped
a fake edge exactly that way. Anything claiming an edge must show both.

## The order of work

1. **Diagnose the level, not the ordering.** AUC is fine and rising with
   horizon; the level is pinned near 0.05. Find what saturates it. Prior work
   names `direction_prob_raw` p50 0.727 -> 0.084 as upstream of the calibrator,
   so this is in what feeds the head, not in calibration.
2. **Do not lower `enter_threshold` to make trades happen.** The floor is 0.5
   and the head maxes at 0.535; dropping the bar to 0.05 would admit every
   cycle at a 46.7% hit rate and pay the round-trip cost on all of them. The
   live book is already -0.186371 over 18 trades. Lowering the bar converts a
   broken head into realised losses.
3. **A calibrator is a legitimate fix only because the AUC says so.**
   Calibration is a monotone remap: it moves the level and preserves the order.
   With AUC 0.54-0.59 there is real order to preserve. Had AUC been 0.50 there
   would be nothing to rescue and a calibrator would be a way of manufacturing
   confidence from noise. State the AUC whenever proposing one.
4. **Add a move-size condition alongside direction.** 37.5% is the fraction of
   ticks worth trading at all. Entry should require expected |move| to clear
   cost, which is what `net_margin >= 0` is trying to express — verify it is
   actually charging the size-aware rate at the notional it will spend.
5. **Then re-measure the funnel.** Ghost closes per hour is the number that
   feeds graduation; nothing downstream can be judged until entries resume.

## What is NOT the problem, so nobody re-litigates it

- **The AERO ban.** ~13% of cycles, and `surviving_enter_candidates=1` on 72 of
  96 ban rows — the ban never emptied the candidate set.
- **`trading_ops` row counts.** ~3.3 rows per tick, and holds are never logged.
  Row counts are not cycle counts.
- **`price_mu` emitting a dollar price.** Real bug, now fixed, serving path
  only, on model-unavailable ticks — it reached no training target and is not
  the collapse.
- **The graduation stamp.** It works. Both leading strategies carry
  `graduated_ts`.
- **The feed.** 371 ticks/10min across 22 fresh symbols, up from 39.

## Where the live path stands

    1 FEED       PASS
    2 SIGNALS    PASS
    3 GHOST      FAIL  <- entries refused by the direction head
    4 LEDGER     PASS
    5 GRADUATION FAIL  (downstream of 3)
    6 RISK       FAIL  (downstream of 3)
    7 WALLET     PASS  $23.18
    8 EXECUTOR   FAIL  (downstream of 5)
    9 LIVE       PASS
   10 PROFIT     FAIL  -0.1864 over 18 trades

Steps 5, 6 and 8 cannot be judged while 3 is closed. **There is exactly one
thing to fix**, and it is the direction head's level.
