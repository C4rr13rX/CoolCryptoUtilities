# The GA's edge is selection luck, and the proof is the sample size

2026-09-16. First real runs of `services/ga_service.py` (which had never
completed one -- `_update()` raised TypeError on start_run's first statement).
Corpus: `data/historical_ohlcv/base`, 12,000 bars/symbol, population 8 over 4
generations (~32 genomes screened), no sentiment, no brain, fitness
out-of-sample only and baseline-penalised.

## Every seed was positive

| run | held-out acc | baseline | edge | trades | naive p |
|---|---|---|---|---|---|
| seed 909, 2 symbols | 71.97% | 51.56% | +20.41pp | 132 | <0.001 |
| seed 77,  2 symbols | 58.75% | 50.08% |  +8.67pp | 320 | 0.002 |
| seed 11,  2 symbols | 56.88% | 51.42% |  +5.46pp | 276 | 0.067 |
| seed 404, 2 symbols | 51.78% | 50.52% |  +1.27pp | 645 | 0.522 |

A naive binomial z-test calls two of these significant. They are not an edge.

## The confirming test: same seed, bigger sample

Seed 909 -- the largest edge on the smallest sample -- re-run identically over
6 symbols instead of 2. The ONLY change is how many held-out trades the
champion must survive:

    seed 909, 2 symbols:  +20.41pp on   132 trades
    seed 909, 6 symbols:   +1.88pp on 1,423 trades

corr(trades, edge) = -0.657 across all five completed runs, monotone in sample
size. A real effect holds as n grows; a selection artifact decays, because the
genomes that trade rarely have the most room to get lucky.

## The null was wrong, not just the p-value

A GA keeps the BEST of everything it evaluated, so the null is "best of ~32
coin-flippers", not "one coin-flipper". Simulated at each sample size:

| run | n | observed | null median | null 95th | percentile |
|---|---|---|---|---|---|
| 909 (6sym) | 1423 | +1.88pp | +2.66pp | +3.92pp | **5.8th** |
| 404        |  645 | +1.27pp | +3.94pp | +5.80pp | below median |
| 77         |  320 | +8.67pp | +5.63pp | +8.13pp | 95th-97th |
| 909 (2sym) |  132 | +20.41pp | +9.05pp | +12.83pp | 100th |

The widened 909 run sits at the **5.8th percentile**: 94% of pure
chance-with-selection runs do better. Seed 77's "p=0.002" is really
one-in-twenty, found by a search drawing far more than twenty times.

## What this does and does not say

It does NOT say the GA is broken. Its out-of-sample, baseline-penalised
fitness is correct and is what made this measurable at all. It says that
out-of-sample fitness for one GENOME is not out-of-sample validation of the
SEARCH, and only the first was implemented. `trading/genome/selection_null.py`
now judges every champion on completion and writes the verdict to the run
record.

## What a real result would need

Edge holding across seeds AND across symbols AND as n grows, clearing the
best-of-N null on the LARGEST sample available, with a per-trade net above the
cost floor (WETH 0.04-0.11%, AERO 0.64-4.35% per round trip -- a sub-1%
accuracy edge on AERO cannot pay for itself whatever its p-value). Nothing
here is armed for live; champion.py feeds the ghost lane and the graduation
ladder decides the rest.
