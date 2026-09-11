# The directive path carries 98.4% of entries and one cost test, and that test was blind to most horizons

Gale, pass 116, item [7231f8ac]. Corpus: `storage/trading_cache.db`, table
`trading_ops`, the 7 days to 2026-09-11 03:10 UTC. Every number below is
reproducible with the commands named beside it.

## 1. Which path entries actually take

`trading/bot.py`'s decision chain sets `should_enter = True` from a
strategy-emitted directive in an `elif` at line 7151, ABOVE the `elif pos is
None:` branch at 7382 that holds the model conjunction (direction_prob,
exit_conf, margin, net_margin_after_fees, expected_profit_units, delta and
expected_abs_move). Python takes the first matching `elif`, so a directive
`enter` short-circuits all seven model conjuncts.

The two paths are distinguishable in the log without instrumentation: the model
branch writes `reason = "model-long"`, the directive branch writes
`reason = directive.reason`, which always carries the strategy's own text.

Rows deduped to entries on `(symbol, ts, reason)` — `trading_ops` logs ~3.3
rows per tick, so a row count is not an entry count. Rows whose `details.source`
starts `c0d3rv2` are the ATF scout's own lane, not bot.py's chain, and are
excluded from the 128 and counted separately.

    raw buy/enter rows, 7d                                 4,382
      ghost_candidate / ghost_candidate_quote_ok           4,131   (not entries)
      ghost-entry / live-entry / live-entry-failed           238

    entries on bot.py's decision chain (deduped)             128
      via the DIRECTIVE path                                 126   98.4%
      via the model conjunction ("model-long")                 2    1.6%
      via the ghost-explore brain branch                       0

    live entries in the window                                10
      via the DIRECTIVE path                                  10   100%

    (separately, the c0d3rv2 ATF scout lane placed 109 ghost entries; that lane
    does not run this chain at all.)

Directive entries by strategy: atf_static 35, rsi_reversal 12,
obv_accumulation@1w 8, rsi_reversal@5h 6, obv_accumulation@5d 6, bus_schedule 6,
stochastic_reversal 5, rsi_reversal@12h 5, stochastic_reversal@12h 4,
money_button 4, and 35 more across 15 further variants.

**So the model conjunction binds on 1.6% of entries and 0% of live entries.**
Every conjunct added to it — including the move-size conjunct shipped for
[4d3310e7] — is measured against a path that placed two trades in a week.

## 2. What the move-size conjunct would have done to those entries

The conjunct is `delta >= min_expected_move` where `min_expected_move =
entry_fees * ENTRY_MIN_MOVE_COST_MULT`, and `entry_fees` is
`roundtrip_cost_rate(notional)` from `services/roundtrip_cost.py` — the one
place the arithmetic lives. Applied to each directive's OWN
`details.bus_plan.expected_return` against `roundtrip_cost_rate(size ×
entry_price)`:

    directive entries with a priceable bus_plan                126   (126 of 126)
      expected_return  <  roundtrip cost -> WOULD BE REFUSED    77   61%
      expected_return >=  roundtrip cost -> would pass          49   39%

    median expected_return                                 5.0000%
    min                                                    0.5107%
    max                                            1,255,131.8649%   <- see below

The second conjunct, `expected_abs_move` (volatility_rel), cannot be evaluated
for 105 of the 126: `volatility_rel` is computed inside the model branch, so a
directive entry never produces one and the logged decision has no field. Of the
21 that do carry one, all 21 fall below their round-trip cost. That 21/21 is a
real number on a small sample and is NOT evidence about the other 105 — it is
reported as measured and nothing is inferred from it.

**One directive carried `expected_return = 12,551,318.65`** — CLANKER-USDC,
obv_accumulation@3d, size 154,799.5 at an entry price of 1.023e-06. That is
1.25 billion percent and it entered. It is filed separately; this item does not
fix it.

## 3. Which exit the lattice took — the finding

`_lattice_refusal` is the only cost test on the directive path. It fails open
six ways: gate disabled, action != enter, `expected_return <= 0`,
`horizon_sec <= 0`, fewer than 64 priced ticks, or any exception. Nothing
recorded which exit was taken, so "the lattice allowed it" and "the lattice
never looked" were the same absence in the log.

Reconstructed from the logged `bus_plan.horizon` and `expected_return`:

    lattice ANSWERED and refused (entry-refused-lattice rows, 7d)   918
      probability layer (expected against round trip)                15
      chaos layer (forecast aimed past the decay horizon)           903

    of the 126 entries that GOT THROUGH:
      open exit: horizon_sec <= 0, label unreadable                  53   42%
          'atf'                                                      35
          '45m'                                                      13
          '20m'                                                       5
      reached the window test (passed, or failed open on <64 ticks
        or an exception — not separable without instrumentation)     73   58%
      open exit: expected_return <= 0                                 0
      open exit: action != enter                                      0
      gate disabled (LATTICE_GATE_ENABLED defaults "1")               0

`_HORIZON_SECONDS` at bot.py:312 listed 5m, 10m, 15m, 30m and then jumped to
1h. **"45m" and "20m" are the `default_horizon` class attribute of seven
strategies**: rsi_reversal, obv_accumulation, bollinger_squeeze,
supertrend_follow and mean_reversion at 45m; stochastic_reversal and
volume_spike at 20m. Seven strategies emitted a horizon their own cost gate
could not read, and every entry they placed at their default went unpriced.

## 4. The decision: the lattice is the right place, and it now reads the label

The size test does NOT get a second copy on the directive path. There is one
cost formula, `services/roundtrip_cost.py`, and the lattice already calls it
(`round_trip_cost=self._roundtrip_fee_rate(notional_hint=notional)` at
bot.py:1998 → `roundtrip_cost_rate`). A second comparison in the `elif` would be
a second place for the units to drift, which is the exact failure that file
exists to prevent.

The lattice was not the wrong place. It was disarmed by a fixed lookup table.
Shipped this pass:

* `horizon_seconds(label)` parses `<number><unit>` (m/min/h/hr/d/w) after
  trying the table, so 45m → 2700s and 20m → 1200s reach the cost arithmetic.
  A label naming no duration — 'atf' — still returns 0.0 and still fails open:
  this layer must never become the reason nothing trades, and inventing a
  window for a label that states none would make the chaos layer judge a
  forecast against a number nobody wrote.
* `self._lattice_last_exit` names the exit taken and the caller stamps it into
  the decision as `lattice_exit`, so the residual fail-opens are a count next
  pass instead of this reconstruction.

Measured effect on the 7-day population, by re-running the census with the
shipped parser:

    entries that got through            BEFORE      AFTER
      unreadable horizon                    53         35   (all 'atf')
        of which 45m                        13          0
        of which 20m                         5          0
      reached the window test               73         91

18 of the 126 directive entries (14%) move from "no cost test at all" to
"priced against `roundtrip_cost_rate` at their own notional". The remaining 35
are 'atf' and stay open — now visibly so.

(The `entry-refused-lattice` count is a live table and grows while you read it:
918 at 03:10 UTC, 920 four minutes later. The census prints the current value.)

Not fixed here, and named rather than widened into someone else's item:

1. The `elif` ordering itself. 98.4% of entries bypass seven model conjuncts.
   Moving the model conjunction above the directive branch is a behaviour
   change on the live path and needs its own before/after.
2. The 1.25-billion-percent `expected_return` from obv_accumulation@3d.
3. `volatility_rel` is unavailable on the directive path, so the conjunct that
   the [4d3310e7] work established as the one that actually bites cannot bind
   there even if the ordering changed.

## Commands that prove each number

    # sections 1, 2 and 3 — the census
    python -X utf8 scripts/directive_path_census.py

    # section 4 — the fix and its regression test
    python -X utf8 -m pytest tests/test_the_cost_gate_reads_a_strategys_default_horizon.py \
                             tests/test_lattice_gate.py -q

---

## Verification re-run — Iris, pass 116, 2026-09-11 04:30 UTC

The work above is Gale's, measured at 03:10 UTC and left uncommitted when the
session limit hit at 03:18. I own [7231f8ac] this pass, so I re-derived every
number rather than inheriting it, and I am committing Gale's fix under Gale's
name in the body.

**1. The census reproduces, with the window slid 80 minutes.** `trading_ops` is
a live table and the 7-day window moves, so the counts are not identical to
03:10 and should not be:

    measured                       03:10 (Gale)   04:30 (Iris)
      deduped entries on the chain        128          125
        via the DIRECTIVE path            126          123   98.4%
        via the model conjunction           2            2    1.6%
      live entries, all directive          10           10    100%
      would be refused by the
        move-size conjunct                 77           74    60%
      lattice answered and refused        918          957

    command: python -X utf8 scripts/directive_path_census.py

**2. The fix's effect on the current window is the same 18 entries.** Horizons
emitted by the 123 directive entries: atf=34, 1w=13, 12h=13, 45m=13, 5h=12,
1d=12, 5d=6, 20m=5, 3d=5, 30m=4, 15m=4, 5m=1, 1h=1. Under the old fixed table
the unreadable set was atf+45m+20m = 52 of 123 (42%); with the shipped parser
the census reports 34 (all 'atf') and 89 reaching the window test. **18 entries
(14.6%) moved from no cost test at all to priced against `roundtrip_cost_rate`
at their own notional**, and the residual 34 are now labelled
`unreadable_horizon:atf` rather than silent.

**3. The regression test is red against the old behaviour, not green both
ways.** Restoring the pre-fix lookup exactly — `_HORIZON_SECONDS.get(label,
0.0)` bound over `trading.bot.horizon_seconds` — and re-driving the test's own
fixture:

    old table 45m -> 0.0   20m -> 0.0
    OLD: sub-cost 45m directive refusal = None   exit = unreadable_horizon:45m
    => the assertion `refusal is not None` FAILS under the old table

    tests/test_the_cost_gate_reads_a_strategys_default_horizon.py
    + tests/test_lattice_gate.py : 19 passed

**4. Criterion 4 stands as Gale decided it, and I checked the claim that
matters.** There is no second copy of the cost formula: `_lattice_refusal`
reaches it through `self._roundtrip_fee_rate(notional_hint=notional)` at
bot.py:1998, which is `services/roundtrip_cost.py:roundtrip_cost_rate`, and
nothing was added to the `elif`. The lattice is the right place; it was simply
disarmed by a lookup table.

**What is still true after this pass, and is the bigger number:** the model
conjunction binds on 1.6% of entries and 0% of live entries. This item priced
14.6% more of the directive path; it did not change the ordering. Item 1 of
Gale's "not fixed here" list is the real residual and needs its own before/after
on the live path.
