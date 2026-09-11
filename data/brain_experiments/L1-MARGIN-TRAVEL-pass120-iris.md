# L1 stickiness does not travel at a fixed band fraction — solve the margin for a CHANGE RATE

Item [746c2ece], pass 120, Iris. **Node-free**: this is encoder arithmetic over
stored bars, so it is not blocked by the RAM floor [beec23cc]. Production's
fabric on :8090 was never contacted.

Corpus: `data/historical_ohlcv/{base,arbitrum}`, cadence-filtered to a median bar
spacing of **3600s** so `--horizon 12` means twelve hours on every file rather
than 33 minutes on one and 48 days on another. Six pairs, two chains.
Train 400 bars / purge 12 / test 300, `LOOKBACK_BARS=168` ahead of the train
window. Horizon 12 bars, L2 window 12 bars, `L2_TRANSITION_STEPS=2`,
`min_support=20`, identifier ceiling 0.30.

Command:

    python -X utf8 scripts/omen_margin_travel.py --pairs 6 --per-chain 3 \
        --json-out data/brain_experiments/L1-MARGIN-TRAVEL-pass120-iris.json

## The defect

`L1_HYSTERESIS_MARGIN` is a fraction of each band's own WIDTH, and the widths are
terciles fitted per stream per corpus. So the same constant buys a different
amount of stickiness on every pair, and what the layer above L1 actually cares
about — alphabet turnover — is left to whatever the terciles happen to give.
Measured at the shipped fixed 0.50, on the TRAIN windows of these six pairs:

    33.3% .. 62.2%     spread 28.8 percentage points

Solved per corpus to a target change rate of 0.375 (the rate the fixed 0.50
achieves on AERO-USDC, the corpus every passing L2 number in `omen_layers` was
measured on):

    25.6% .. 37.3%     spread 11.8 percentage points

The residual 11.8pp is not solver error. It is the step function: the change rate
can only take values k/(n−1) and two corpora hit different steps, plus ARB-WBTC
DOWN, whose rate falls from 50.9% straight past the target to 25.6% at the first
margin that crosses it.

## Held-out L2, per pair, UP and DOWN — this is the criterion-3 table

Every number below is read on a test window the fit never saw: `relative_bands`
and the solved margin are both computed on the train frames alone. Only the held
BAND STATE crosses the boundary, which is exactly what crosses in production.

| pair | chain | window | mean fwd | trough base | fixed 0.50 L2 | solved L2 | solved margin |
|---|---|---|---|---|---|---|---|
| AERO-USDC | base | UP | +4.0310% | 17.0% | 0.2500 passes | 0.2500 passes | 0.3685 |
| AERO-USDC | base | DOWN | −2.1231% | 6.7% | 0.3300 IDENTIFIER | 0.3300 IDENTIFIER | 0.5000 |
| EURC-USDC | base | UP | +0.0871% | 1.0% | 0.4633 IDENTIFIER | **0.2033 passes** | 1.0746 |
| JITOSOL-CBBTC | base | UP | +0.8244% | 6.0% | 0.3333 IDENTIFIER | **0.3000 passes** | 0.5319 |
| JITOSOL-CBBTC | base | DOWN | −1.4129% | 15.7% | 0.3067 IDENTIFIER | **0.1867 passes** | 0.7500 |
| ARB-WBTC | arbitrum | UP | +1.8925% | 19.0% | 0.3667 IDENTIFIER | 0.3833 IDENTIFIER | 0.4082 |
| ARB-WBTC | arbitrum | DOWN | −1.8700% | 10.3% | 0.3500 IDENTIFIER | **0.1867 passes** | 1.0000 |
| ARB-WETH | arbitrum | UP | +1.5182% | 17.7% | 0.2300 passes | 0.1967 passes | 0.6876 |
| ARB-WETH | arbitrum | DOWN | −1.5619% | 15.7% | 0.3233 IDENTIFIER | **0.2833 passes** | 0.8422 |
| LINK-WETH | arbitrum | UP | +1.6433% | 17.3% | 0.4433 IDENTIFIER | 0.3067 IDENTIFIER | 0.7637 |
| LINK-WETH | arbitrum | DOWN | −1.0814% | 7.7% | 0.1733 passes | 0.0733 passes | 0.8334 |

**Verdict counts, held out, both window classes pooled only for the count:**

    fixed 0.50   3 windows pass,  8 IDENTIFIER
    solved       8 windows pass,  3 IDENTIFIER

    supported L2 groups (n>=20), summed over windows
    fixed 0.50   9
    solved      13

## What this does and does not establish

**It does establish** that the margin is solvable and that solving it travels
better than a constant: the turnover spread halves and held-out L2 goes from
failing the identifier guard on 8 of 11 windows to failing on 3. Crucially the
gain is not confined to one window class — of the five windows the solve flips
from IDENTIFIER to passing, **three are DOWN windows and two are UP**. A
long-only flattering effect cannot produce that shape.

**It does not establish that L2 predicts anything.** The identifier guard is a
necessary condition, not a sufficient one, and the support column says so
plainly: even after the solve, five of the eleven windows still have **zero**
L2 groups reaching n=20, and the best lift is 0.00x on six of them — meaning the
largest group by lift has a trough rate of zero. LINK-WETH DOWN is the one window
where the solve buys real supported mass (3 groups covering 30.0% → 4 covering
72.7%), and the pair it buys it on is also the one whose L1 collapses hardest
(vocabulary 18 → 9). That is the cost this module has documented from the start:
a layer that abstracts perfectly and predicts nothing is still worthless, and a
vocabulary of 9 over 300 bars is close to that edge.

**Three windows still fail the guard after the solve** and they are named rather
than averaged away: AERO-USDC DOWN (0.3300, and its solve returned 0.5000 —
the corpus was already at 34.3% turnover so the target changed nothing),
ARB-WBTC UP (0.3833, which is WORSE than the fixed arm's 0.3667), and LINK-WETH
UP (0.3067). The ARB-WBTC UP row is the honest counter-example: equalising
turnover can push a corpus the wrong way when its fixed rate already sat below
the target.

**Train turnover does not fully determine test turnover.** Solved to 37.3% on
LINK-WETH DOWN train, the test window came in at 22.1%; ARB-WBTC UP solved to
36.3% and tested at 40.1%. The margin is fitted on train because fitting it on
test would leak the test distribution — so the turnover the layer above actually
sees held out is only controlled indirectly, and that is a real limit of this
approach rather than a detail.

## Criterion 4 — windows excluded on label incidence, and named

`EURC-USDC` on base: **4 candidate windows excluded**, trough base rate 0.0% —
the omen label never fires, so every group's trough rate is zero and every lift
is 0/0. Its DOWN window is excluded entirely for that reason and EURC-USDC
appears above with an UP window only. This reproduces on a second pair the defect
Cove found on ARB-WETH DOWN in pass 116, so it is a property of the selection,
not of one corpus. The surviving EURC-USDC UP window has a 1.0% base rate, which
is thin enough that its otherwise-striking 0.4633 → 0.2033 improvement should be
read as an encoder result and not as evidence about the label.

## What shipped

- `trading/omen_layers.py`: `l1_change_rate` and `solve_hysteresis_margin`, plus
  `L1_TARGET_CHANGE_RATE` and `L1_MARGIN_SEARCH_CEILING`. The solver bisects the
  real `sticky_motifs` encoder, so what is solved is what ships — no second
  definition of L1 that can drift from the first. `L1_HYSTERESIS_MARGIN` is left
  in place and unchanged as the constant fallback; nothing in the live path was
  re-pointed at the solver in this pass, because an encoder change with no
  held-out edge behind it is not a change worth making to a live path.
- `scripts/omen_margin_travel.py`: the probe above. Exits nonzero when solving
  fails to close the turnover spread.
- `tests/test_the_hysteresis_margin_is_solved_not_a_constant.py`: six tests. The
  one that matters most re-pins that `sticky_motifs(..., 0)` is still
  byte-identical to per-bar `cooccurrence_motif`, so the baseline could not move
  underneath the solver.

## What I would try next

The support column, not the distinctness column. Eight of eleven windows now pass
the identifier guard and five of them still have no group at n=20 — so the next
binding constraint is exactly the one [89693706] names: support famine. Solving
the margin removed the reason L2 could not be measured; it did not produce
anything to measure yet.
