# The self-pool arm cannot be powered on one corpus, and the reason is the DOWN window, not the sample size

Pass 113, Cove. Item **[5ec44914]**.

## The finding, in one sentence

Raising `--test` does not fix the 19-trade problem: measured over 40 corpora,
the longest held-out window that still reads as a DOWN window is **100 bars on
the median corpus** and holds **10 true troughs**, so no single-corpus DOWN arm
can reach the item's 60-trade floor at any window length — while the UP side
hits the sweep's 5000-bar cap on 29 of 39 corpora and is not constrained at all.

## Why this measurement, and why before any node time

[5ec44914] asks for a **powered** arm: at this feed's dispersion, detecting a
1.0pp effect needs ~63 trades per arm. Pass 111's cells held 7, 1, 2 and 9. The
obvious reply is "use a longer held-out window", and that reply is wrong in a
way that had not been written down:

A held-out window has to be two things at once and they pull apart.

* **Powered** — long enough to place enough trough trades.
* **Regime-pure** — an UP window *and* a DOWN window separately, because a
  long-only rule flatters itself in an up window and that error has already
  produced a fake 78% and a fake +0.9067% in this repo.

Gale's pass-111 windows separated 53.3% up-rate from 11.7% **only because they
were 60 bars long**. Stretch a window for power and its up-rate walks back to
the corpus mean, at which point there is no UP window and no DOWN window, only
the market. So the question is not "how long a window" but "how long a window
can still be called DOWN, and how many trades fit inside it".

This costs **zero node time** — it is arithmetic over the corpora, run before a
fabric is trained, which is the standing rule about sizing an experiment
against the budget before launching it.

## Setup

| | |
|---|---|
| instrument | `scripts/omen_self_pool_power.py` (new this pass) |
| corpora | first 40 files under `data/historical_ohlcv`, cadence ≤ 14400s |
| cadences present | 73, 94, 300, 307, 600, 3600, 7200 seconds |
| horizon | **720 minutes**, converted per corpus → 6 to 592 bars |
| label | `trading.omen_brain.label_omen`, the same function the experiment trains on |
| round-trip cost | 0.0065, `OMEN_COST_MULTIPLE` 1.5 |
| purity bar | UP forward up-rate ≥ 0.50, DOWN ≤ 0.15 |
| minimum window | 60 bars — Gale's length, below which purity is an accident |
| node | **none contacted**; production `:8090` untouched |

The horizon is given in **minutes and converted per corpus**, not in bars: the
same `--horizon 12` is 12 bars of 73s tape on one file and 12 bars of 7200s
tape on another. That is Iris's [4d0b539b] and this census would have been
meaningless without taking it.

## 1. The DOWN window is the binding constraint, and it is short

40 corpora, every one of which has *some* pure DOWN window of 60+ bars.

| longest pure DOWN window | corpora |
|---|---|
| ≥ 400 bars | **4 of 40** |
| 100–200 bars | 36 of 40 |
| **median** | **100 bars** |
| **median TRUE troughs inside it** | **10** |

The four that are long enough to be interesting:

| corpus | longest pure DOWN window | up-rate | TRUE troughs inside |
|---|---|---|---|
| `0009_ARB-USDC.json` | 1600 bars | 0.150 | 52 |
| `0002_RAIN-WETH.json` | 1600 bars | 0.150 | 39 |
| `0001_WETH-USDT.json` | 1200 bars | 0.150 | 39 |
| `0004_WBTC-WETH.json` | 800 bars | 0.150 | 26 |

Three corpora (`0001_ARB-WBTC`, `0000_ARB-WETH`, `0000_ARB-WBTC`) hold **zero**
true troughs in their longest pure DOWN window. Every up-rate prints at the
bound because the longest clearing length clears only marginally — that is
selection, not a coincidence.

## 2. The UP side is not constrained

39 of 40 corpora have a pure UP window, and **29 of 39 reach the sweep's
5000-bar cap**, median 5000 bars, median 687 true troughs.

**Stated against my own result:** that is partly because my UP bar of 0.50 is
*more permissive* than the 53.3% Gale actually measured. At 0.50 an "UP window"
is most of the tape. So the honest reading is: the UP arm has abundant power
and **weak** regime purity, and a report using a 5000-bar UP window must say
its up-rate, not call it an UP window and move on.

## 3. What n=60 per arm actually requires

Trades placed = held-out bars × the rate at which the arm *calls* trough. Pass
111's four cells fired on 9/60, 7/60, 2/60 and 1/60 of held-out bars.

| fire rate | bars needed for n=60 | best single DOWN window (1600 bars) places | median DOWN window (100 bars) places |
|---|---|---|---|
| 9/60 = 15.0% | 400 | 240 | 15.0 |
| 7/60 = 11.7% | 514 | 187 | 11.7 |
| 2/60 = 3.3% | 1800 | **53** | 3.3 |
| 1/60 = 1.7% | 3600 | **27** | 1.7 |

**At the DOWN arm's own measured fire rate the largest DOWN window in 40
corpora still falls short — 53 trades against a floor of 60.** The median
corpus places 3.3. This is not a sample that more patience fixes.

## 4. The powered design this licenses

* **UP arm:** one corpus, held-out window 2400–5000 bars. Places 280–580 trades
  at the pass-111 UP fire rate. Powered, with its up-rate quoted.
* **DOWN arm:** **pool** DOWN windows across corpora — roughly **18 median
  windows**, or 2–3 of the four long ones above, each scored against a fabric
  trained on its own corpus. There is no single-corpus route.
* **Pools queried:** two, `self_outcome` and `self_error_run`. `self_agreement`
  stays out, and the arm is reported as testing **two pools, not three**, per
  the item's second criterion — it measures 0.0025 distinct per sample because
  agreement is `agreed/asked` across query sets and the sample builder runs
  before any node exists.

## What is NOT established here

This pass did **not** re-run the arm, so the dilution question is still
**unmeasured** rather than answered either way. What changed is that the next
attempt now has a size it must hit and a route to hitting it, instead of a
60-bar window that was never going to produce a rankable number.

## The commands

    python -X utf8 scripts/omen_self_pool_power.py \
        --corpus data/historical_ohlcv/base/0004_AERO-USDC.json --horizon-minutes 720
    python -X utf8 scripts/omen_self_pool_power.py --horizon-minutes 720 --limit 40
    python -X utf8 -m pytest tests/test_a_down_window_is_too_short_to_power_an_arm.py -q

---

# 2026-09-10, pass 113, Gale — THE ARM WAS RUN, AND THE DILUTION SIGN FLIPPED

Item **[5ec44914]**, continuing Cove's sizing census above. Cove established the
size the arm must hit; this section is the arm itself, run on a window **3.3x
longer** than pass 111's.

## The headline, stated against my own preference

On a **200-bar pure DOWN window**, the self pools did **not** dilute — the
with-self arm beat the without-self arm by **2.09 percentage points per trade**
and was the only arm to clear the buy-every-bar baseline by a wide margin.
**That is the opposite sign to pass 111's verdict.**

**And I am not claiming an edge from it.** The with-self cell holds **11
trades**, under the item's n=30 floor, so by this item's own first criterion it
is reported as **unrankable** and no per-trade verdict is given. What the run
*does* license is narrower and, for this item, more useful:

> **The sign of the self-pool effect reverses when the held-out window goes
> from 60 bars to 200 bars on the same question.** A conclusion that flips when
> you change only the window length, holding the method fixed, is a conclusion
> about the window, not about the pools. That is direct evidence for the item's
> premise — the pass-111 verdict was noise — and it does not need n>=30,
> because the instability IS the observation.

## Setup

| | |
|---|---|
| node | fresh `127.0.0.1:8091`, brain dir `brain-data-p113-gale-arm`, **`pool_count` 20**, tick 0, `neurons=0 concepts=0` at start |
| identity | `brains\market_predictor_v4_meta.identity.toml` — **not v2**, see the trap below |
| production | `:8090` PID 16108 **untouched** |
| corpus | `data/historical_ohlcv/arbitrum/0002_LINK-WETH.json`, 21801 bars, 3600s cadence |
| horizon | 12 bars = **720 minutes**, stated in minutes per Iris's [4d0b539b] |
| train | bars `[3382, 3982)` -> 600 samples, balanced to 517, **ONE epoch**, 0 failed, 1.4 min |
| held-out | bars `[3994, 4194)` — **200 bars**, up-rate **15.0%**, mean forward **-1.5588%** -> a real DOWN window |
| fabric | **ONE**; arm B ran `--skip-train`, so both arms are back-to-back on the same fabric |
| pools queried | **TWO**: `self_outcome` and `self_error_run` |

## `self_agreement` is EXCLUDED, and this arm tests TWO pools not three

The item's second criterion offered two routes. I took the exclusion route and
say so plainly: **this arm tests two self pools, not three.** The distinctness
the run measured on its own samples is the reason, and it sits in the dilution
law's empty band:

    self_outcome=0.041   self_error_run=0.035   self_agreement=0.004

`self_agreement` at 0.004 is the floor. The cause is the one pass 111 already
established and nothing this pass changes it: agreement is `agreed/asked`
*across query sets*, and the sample builder runs before any node exists, so
there is nothing to disagree with. **The pool carrying the only measured signal
in this area (99.4% unanimous vs 73.3% split) is therefore still untested**, and
no verdict here is a verdict on it.

## The four cells, with n on BOTH sides

DOWN window `[3994, 4194)`, 200 bars, one fabric, back-to-back.

| | with self pools | without self pools | buy every bar |
|---|---|---|---|
| **per-trade net on buy omens** | **+0.3734%** | **-1.7172%** | **-2.2088%** |
| **n (trades placed)** | **11** | **54** | 200 |
| trough precision (paid the round trip) | 63.6% of 11 | 18.5% of 54 | — |
| total net over the window | **+0.0411** | **-0.9273** | — |
| held-out exact | 6.0% of 200 | 25.0% of 200 | majority class **57.5%** |

**Read the last row before the first.** *Both* arms are far below the 57.5%
majority class on exact accuracy. Neither arm predicts the label well. The
per-trade rows are a statement about which bars each arm chose to buy, not
about the fabric having learned the tape.

**And the two arms are not equally selective** — 11 trades against 54. A
per-trade mean compared across arms with a 5x difference in fire rate is partly
a comparison of selectivity, not of quality. The total-net row is given so both
readings are available: with-self wins on that too, +0.0411 against -0.9273.

## What this does and does not settle

**Settles:** the pass-111 dilution verdict does not survive a longer window.
Pass 111 measured DOWN with self -3.2611% (2 trades) against without -1.9445%
(9 trades); this pass measures **+0.3734% (11) against -1.7172% (54)** on the
same question with a 3.3x longer window. The sign reversed on both cells.

**Does not settle:** whether the self pools help. n=11 is unrankable, only the
DOWN window was run, and the both-windows rule is not satisfied. **The honest
status of the self pools is NOT MEASURED — neither "they dilute" nor "they
help."** Anyone quoting the +0.3734% as an edge is repeating the exact error
this item was filed to stop.

## Why n=60 per arm was not reached, in one number

The 200-bar window contains **17 true troughs** (`test label mix ... 'trough':
17`). Trough trades cannot exceed the troughs available by much, so **no
200-bar DOWN window can place 60 trough trades** — the ceiling is the tape's,
not the node's. Cove's census above shows the same wall at corpus scale: the
median corpus's longest pure DOWN window is 100 bars holding 10 troughs.
Reaching n=60 on the DOWN side needs **pooling across corpora, one fabric
each**, which is a multi-fabric run and did not fit this pass's budget.

## A NODE TRAP THAT COST THIS PASS A RUN, written down so it costs nobody else one

Starting `:8091` on `brains\market_predictor_v2.identity.toml` gives
**`pool_count` 12**, while the self pools are ids **15, 16 and 19**. The probe
reported `taught 0/600 — the node consolidated NOTHING`, and the misleading
part is that `/brain/stats` **simultaneously read `tick` 600, `total_binding`
600, `total_concepts` 10470** — the node looked like it had learned. RAM was
6.7 GB free against a 3000 MB floor, so **the RAM floor was not the cause** and
I would have chased it. Use `market_predictor_v4_meta.identity.toml`
(`pool_count` 20) for anything touching the self pools.

Separately, `omen_experiment.py`'s warm-fabric guard **worked**: a second run
against a node already holding 1200 bindings was refused rather than reported,
which is what stopped a two-epoch fabric being read as a one-epoch result.

## Two defects fixed in the planner this pass, both of which would have inflated n

1. **The plan dropped the chain.** `census_corpus` stored `path.name`, and
   `0002_LINK-WETH.json` exists under `arbitrum` *and* `polygon`. The printed
   command died on `FileNotFoundError`. It now stores the full path.
2. **The same pair on five chains counted as five independent windows.**
   LINK against WETH is the same two assets whichever chain quotes it, so
   pooling all five multiplies n by five and adds no independent tape — an n
   inflated exactly the way this item exists to stop. `print_plan` now dedupes
   by traded pair and says how many it dropped.

Both are pinned by tests in
`tests/test_a_down_window_is_too_short_to_power_an_arm.py`
(`..._names_the_chain_so_the_command_it_prints_can_actually_run`,
`..._the_same_pair_on_two_chains_is_not_two_independent_windows`), which fail
against the old behaviour.

## The commands

    # windows, chain-qualified, with the leakage constraint checked
    python -X utf8 scripts/omen_self_pool_power.py --horizon-minutes 720 --limit 40 --plan

    # node (NOTE the v4_meta identity)
    D:\Projects\W1z4rDV1510n\start_node.ps1 -Addr 127.0.0.1:8091 -BrainDir D:\Projects\W1z4rDV1510n\brain-data-p113-gale-arm -Identity brains\market_predictor_v4_meta.identity.toml -Deployment brains\market_predictor_v2.deployment.toml -MinSysAvailMb 3000

    # arm A, with self pools (trains the fabric)
    OMEN_META_COLLECTIONS=1 OMEN_BRAIN_ENDPOINT=http://127.0.0.1:8091 python -X utf8 scripts/omen_experiment.py --corpus data/historical_ohlcv/arbitrum/0002_LINK-WETH.json --horizon 12 --train 600 --test 200 --train-end 3982 --test-end 4194 --query-collections temporal,geometry,cross,self_outcome,self_error_run

    # arm B, without, SAME fabric
    OMEN_META_COLLECTIONS=1 OMEN_BRAIN_ENDPOINT=http://127.0.0.1:8091 python -X utf8 scripts/omen_experiment.py --corpus data/historical_ohlcv/arbitrum/0002_LINK-WETH.json --horizon 12 --train 600 --test 200 --train-end 3982 --test-end 4194 --skip-train --query-collections temporal,geometry,cross

    python -X utf8 -m pytest tests/test_a_down_window_is_too_short_to_power_an_arm.py -q

Node reports: `omen-LINK-WETH-h720m12b-DOWN-20260910-233414.json` (with self),
`omen-LINK-WETH-h720m12b-DOWN-20260910-233437.json` (without).

## What the next pass should run, and it is NOT this arm again

The UP half. `scripts/omen_self_pool_power.py --plan` emits a runnable UP
window on `0003_CRV-WETH.json` (`--test 5000 --test-end 5823 --train-end 811`,
h=12). At the with-self fire rate that window places several hundred trades and
would clear n=60 on its own — the UP side was never the constrained one. Run it
against a FRESH node on the v4_meta identity, and only then is there a
both-windows verdict to state.
