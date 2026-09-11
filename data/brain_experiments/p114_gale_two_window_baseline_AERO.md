# TWO-WINDOW BASELINE, AERO-USDC — the first omen number ever measured in an UP window

Pass 114, Gale, item [47d70b7c]. Written 2026-09-11.

**THE HEADLINE: the premise of this item is FALSIFIED on the accuracy scoreboard.**
Every omen negative on record was measured in a DOWN window, and the standing
worry was that a long-only rule had been penalised by that. It was not. The UP
window scores **28.5% exact, the same figure pass 107 reported**, and it scores
it against a *higher* majority-class baseline — so relative to baseline the UP
window is **worse** (−3.5pp) than the DOWN window (+2.0pp). The single-window
negatives were **not pessimistic**.

On the money scoreboard the sign goes the other way — buy omens beat
every-bar-buy in **both** windows — but **neither lift clears its own noise**
(p = 0.073 and p = 0.281 against a random-subset null). There is no held-out
edge here in either window, on either scoreboard.

---

## 1. Corpus, windows, protocol

| | |
|---|---|
| corpus | `data/historical_ohlcv/base/0004_AERO-USDC.json` |
| bars | 21,926 at 3600s cadence |
| pairs | 1 (single corpus — see §5, this is the binding limitation) |
| horizon | `--horizon-minutes 720` = **12 bars** of 3600s |
| round-trip cost | 0.6500%; omen threshold 0.9750% |
| node | fresh fabric, `brain-data-p114-gale-2win`, `:8091`, **`pool_count 12`** |
| identity | `market_predictor_v2.identity.toml` — the **current** topology |
| production | `:8090` untouched |

`pool_count 12` is **correct** here and is not a broken node: v2 is eleven
sensory pools plus the action pool. The board advice to switch to
`v4_meta` applies to arms that need the SELF pools (ids 15/16/19), which this
one does not — running it on v4_meta would have produced something that is not
the baseline this item exists to supply.

**ONE fabric, two disjoint held-out windows, training window pinned:**

```
UP    --train 1200 --train-end 20501 --test-end 20913
DOWN  --train 1200 --train-end 20501 --test-end 21713 --skip-train
```

| | train | held-out | up-rate | mean forward | regime |
|---|---|---|---|---|---|
| UP arm | [19301, 20501) | [20513, 20913) | **57.25%** | **+1.1568%** | UP |
| DOWN arm | [19301, 20501) | [21313, 21713) | **40.50%** | **−0.4170%** | DOWN |

The train window is byte-identical between arms and the fabric was trained
once; the DOWN arm ran `--skip-train` against the fabric the UP arm left
(`fabric before: neurons=35347 concepts=35180 binding=2152`). A full 12-bar
horizon gap sits between `train_stop` and each `test_start`, so no training
sample's future overlaps a held-out bar. Verified by `plan_windows` arithmetic,
which is unit-tested in `tests/test_a_second_window_does_not_silently_retrain.py`.

**400 held-out bars per window, not 60.** Every previous omen number on this
repo — pass 107's 28.5%, pass 111's cells — was measured on 60 held-out bars
yielding 1–9 buy omens. This is 6.7x the held-out sample and yields 52 and 70
buy omens. That change is the reason the numbers below have a standard error
worth quoting at all.

## 2. The numbers

Both arms: train recall 99.0% (198/200), garbage 5 distinct answers of 40,
query collections `(geometry, temporal, cross)` measured on this corpus,
0 colliding frame tuples (recall ceiling 100%), 400 of 400 admitted.

**Each window was read TWICE off the identical fabric** (the second read via
`--skip-train`, same samples, ~6 minutes apart). Both reads are given, because
they differ — see §2.1, which is a result in its own right.

### Scoreboard A — held-out exact accuracy against majority class

| window | read 1 | read 2 | majority baseline | difference (r1 / r2) | verdict |
|---|---|---|---|---|---|
| **UP** | **28.50%** | **27.75%** | 32.00% | **−3.50pp / −4.25pp** | **below baseline, both reads** |
| **DOWN** | **31.75%** | **30.50%** | 29.80% | **+1.95pp / +0.70pp** | at baseline |

Binomial against the majority class on read 1: UP z = −1.50, p = 0.134;
DOWN z = +0.85, p = 0.394. Neither window is distinguishable from its
majority class, and the UP window is below it on both reads.

### Scoreboard B — buy-omen net per trade against every-bar-buy

Net is forward return minus the 0.6500% round trip.

| window | buy omens | omen net/trade | every-bar net/trade | lift | p | verdict |
|---|---|---|---|---|---|---|
| **UP** r1 | 52 of 400 | +1.4887% | +0.5068% | +0.98pp | 0.073 | not significant |
| **UP** r2 | 51 of 400 | +1.4953% | +0.5068% | +0.99pp | **0.0727** | **inside the noise** |
| **DOWN** r1 | 70 of 400 | −0.8983% | −1.0670% | +0.17pp | 0.281 | not significant |
| **DOWN** r2 | 70 of 400 | −0.9576% | −1.0670% | +0.11pp | **0.3503** | **inside the noise** |

Buy hit rate 57.7% UP, 37.1% DOWN — i.e. it tracks the window's own up-rate
(57.2% / 40.5%) almost exactly, which is what a rule with no edge looks like.

### 2.1 The same fabric, read twice, does not give the same number

This was not planned; the second read existed only to regenerate the reports
with the new selection p-value in them. It is the more useful half of the pass.

| | read 1 | read 2 | swing |
|---|---|---|---|
| UP exact | 28.50% | 27.75% | **0.75pp** |
| DOWN exact | 31.75% | 30.50% | **1.25pp** |
| UP buy omens | 52 | 51 | 1 trade |
| DOWN net/trade | −0.8983% | −0.9576% | 0.06pp |

Identical fabric (`--skip-train`, `neurons=35347 concepts=35180 binding=2152`
before both), identical samples, identical seed, minutes apart. **The node is
not deterministic**, and the size of the wobble is comparable to the effects
being hunted: the DOWN window's "+1.95pp above baseline" became "+0.70pp above
baseline" on re-read, from nothing but re-asking.

**Consequence for anyone quoting a number from this brain: an accuracy
difference under about 1.5pp on 400 held-out bars is not a result**, whatever
the binomial arithmetic says, because re-asking the same fabric moves it that
far. This is the standing "89.2% and 93.6% thirty-four minutes apart" warning
measured properly — back-to-back, one fabric, one pass — and it is tighter
than that pair suggested but far from zero. It also means the only correct
comparison is two arms read in the same session, and even then a sub-2pp gap
needs repeat reads before it is quoted.

**The significance test, stated so it can be attacked.** The null is *"the k
buy omens are a random k of the 400 held-out bars"*. I drew 20,000 random
k-subsets of the actual held-out net returns and took the mean of each; the
observed omen mean is compared against that distribution. This is the right
null because the omens are a *subset* of the same 400 bars, not an independent
sample — comparing them as two independent means would understate the error.

```
UP    every-bar +0.5068%  sd 5.2616%  | k=52  null SE 0.6783%  z +1.45  p 0.073
DOWN  every-bar −1.0670%  sd 2.7181%  | k=70  null SE 0.2947%  z +0.57  p 0.281
```

The UP window's sd is **5.26%** per bar. At that dispersion a 52-trade cell
cannot resolve a 1pp effect — this is the same power arithmetic Jet applied to
[5ec44914], and it applies to my own favourable-looking number here.

**Confidence separates nothing, again.** The confidence sweep is flat across
every floor from 0.00 to 0.50 in both windows — 52 trades at +1.489% and 70 at
−0.898% regardless. Real-query confidence (med 0.969 / 0.971) sits far above
garbage confidence (med 0.627), so the fabric knows real input from noise; it
does not know a right answer from a wrong one. That reproduces the standing
finding that confidence is worthless as a correctness gate.

## 3. The four questions this item asked

**Is there held-out edge in the UP window?** No. Exact accuracy is 3.5pp
*below* the majority class. The buy-omen net beats every-bar-buy by 0.98pp but
p = 0.073 on 52 trades.

**Is there held-out edge in the DOWN window?** No. Exact accuracy is 1.95pp
above the majority class at p = 0.394, and the buy-omen lift is 0.17pp at
p = 0.281.

**At or below baseline in each window?** **UP: below baseline. DOWN: at
baseline.** An honest negative in both.

**Were the existing single-window negatives pessimistic, optimistic, or
unchanged?** **Unchanged, and if anything they were mildly *optimistic* on
accuracy.** This is the finding that matters, because the item was filed on the
suspicion that they were pessimistic:

- The UP window returns 28.5% exact — the *same* number pass 107 reported from
  the DOWN window. Adding the UP window did not raise it.
- The UP window's majority baseline is *higher* (32.0% vs 29.8%), because an up
  window has more `climb` bars. So the same accuracy scores **worse** against
  baseline in the UP window. Measured relative to baseline, the DOWN-only
  numbers on record were the *flattering* ones.
- On the money scoreboard only, the DOWN-only reading was mildly pessimistic:
  buy omens beat every-bar-buy in both windows, and the UP window is where that
  gap is largest. But the gap is inside its own noise in both.

## 4. What this settles

The standing rule — held-out edge must hold in an UP window **and** a DOWN
window — has never once been testable on this brain. It is now, and **the
current topology fails it in both.** This is the baseline everything else gets
compared against; a future arm has to beat 28.50%/32.00% UP and
31.75%/29.80% DOWN, on this fabric, on these windows.

`OMEN_STRATEGY_ENABLED` **stays 0** and was not touched. It is absent from
`.env` and `trading/strategies/omen_reversion.py:77` defaults it to `"0"`.
Held-out edge did not hold in either window, let alone both.

## 5. The limitation, named rather than worked around

**This is one corpus.** Jet's pass-113 note on this item is right that a
single-corpus protocol cannot be powered, and the numbers above show the shape
of it: the UP cell's 52 trades against a 5.26% per-bar sd cannot resolve a 1pp
effect. What this run *does* settle is the accuracy question, where 400 bars
per window is a real sample and the answer is unambiguous — the UP window is
below its baseline.

What it does **not** settle is whether the ~1pp buy-omen lift, which appears in
both windows with the same sign, is real. Two same-sign cells at p = 0.073 and
p = 0.281 is not evidence, but it is the only thing in this measurement that
points anywhere. Resolving it needs the pooled multi-corpus protocol Jet filed
separately, converting horizons per corpus because cadences run 73s to 7200s.
I did not lower `--min-support` or any other guard to manufacture calls.

## 6. Reproduce

```
python -X utf8 scripts/omen_experiment.py --corpus data/historical_ohlcv/base/0004_AERO-USDC.json \
  --train 1200 --train-end 20501 --test-end 20913 --horizon-minutes 720 --endpoint http://127.0.0.1:8091
python -X utf8 scripts/omen_experiment.py --corpus data/historical_ohlcv/base/0004_AERO-USDC.json \
  --train 1200 --train-end 20501 --test-end 21713 --skip-train --horizon-minutes 720 --endpoint http://127.0.0.1:8091
```

Machine-readable reports, both naming their regime in the filename:

- `omen-AERO-USDC-h720m12b-UP-20260911-000959.json`   (read 1, the training run)
- `omen-AERO-USDC-h720m12b-DOWN-20260911-001033.json` (read 1)
- `omen-AERO-USDC-h720m12b-UP-20260911-001557.json`   (read 2, carries the p-value)
- `omen-AERO-USDC-h720m12b-DOWN-20260911-001615.json` (read 2, carries the p-value)

The read-2 pair carries `selection_lift_per_trade`, `selection_null_se`,
`selection_z`, `selection_p_value` and `selection_trials`, shipped this pass in
`scripts/omen_experiment.py` and tested in
`tests/test_a_buy_omen_lift_is_quoted_with_its_noise.py`. Every future omen run
now prints its lift with `-> INSIDE the noise -- not an edge` or the
alternative, so no one can quote an omen-beats-buy-and-hold number without it.

Both carry `horizon_minutes`, `horizon_bars` and `bar_seconds`, so the next
pass can compare against this honestly across cadences.
