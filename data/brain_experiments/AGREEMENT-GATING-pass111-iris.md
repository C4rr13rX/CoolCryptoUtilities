# Agreement-gating does NOT pay held out, in either window — and the 99.4% it rests on is a RECALL number

**Pass 111, Iris, 2026-09-10.** Item `[a0e7ca5d]`.
Instrument: `scripts/omen_agreement_census.py` (new this pass).
**This is a NEGATIVE result and it is a finished pass.** `OMEN_STRATEGY_ENABLED`
stays 0. Nothing is promoted.

## The claim being tested

Confidence is worthless as a correctness gate here (+0.030 train, −0.002 held
out). AGREEMENT between query sets was believed strong — **99.4% correct when
unanimous against 73.3% when split** — and it is computed client-side inside
`OmenBrain.predict(consensus=True)` and thrown away. The item asked whether
keeping only the agreeing answers changes **per-trade net**.

It does not. And the reason is in `trading/omen_brain.py:1098`'s own docstring,
which nobody had measured against a held-out set:

> *"Costs four round trips and buys 95.5% -> 99.4% reproduction; it does NOT
> buy an edge."*

**99.4% is a TRAIN RECALL number.** It is about reproducing frames the fabric
was taught. Below it is measured on bars the fabric never saw, in an UP window
and a DOWN window, and it does not transfer.

## The rig

| | |
|---|---|
| corpus | `data/historical_ohlcv/base/0004_AERO-USDC.json`, 21926 bars, 3600s cadence |
| node | `127.0.0.1:8093`, fresh brain dir `brain-data-agree-p111-iris`, v2 identity (11 pools). **Not** production's `:8090` |
| train window | bars `[20191, 21691)` — **pinned**, so both held-out windows are strictly after it with a 12-bar purge |
| DOWN window | bars `[21703, 21803)`, 100 bars, up-rate 30.0%, mean forward **−0.8220%** |
| UP window | bars `[21813, 21913)`, 100 bars, up-rate 63.0%, mean forward **+0.7880%** |
| fabric | 1353 balanced pairs, 0 failed, trained once. **Both windows scored on that ONE fabric**, the UP arm with `--skip-train` |
| horizon | 12 bars = 12h. Round trip charged on every buy: `ROUND_TRIP_COST` = 0.6500% |

**ONE CHANGE ONLY.** No relation frames, no new pool, no encoder change. The
only thing that differs between the two arms is *which answers are kept*.

## The query path fires — proven, not assumed

The two arms are not two runs. Every held-out sample is fired once per query
set in a single pass, so ALL and AGREE are the same queries against the same
fabric.

| | DOWN | UP |
|---|---|---|
| samples where the 4 query sets split | **77 / 100** | **78 / 100** |
| A-vs-A control (primary set refired) | **4 / 100** | **0 / 100** |

77% disagreement against a 4% control is a real query-set effect, not variance.
This is the acceptance criterion the operator asked for after pass 108, and it
is built into the script: a run where the sets never disagree exits **3** and
refuses to report an agreement number at all.

**A DEFECT THIS FOUND, and it is in the live path.** `omen_brain.py:1147`
dedupes consensus members by TUPLE, but `discriminating_collections` returns
the measured set in *distinctness* order. On this corpus the primary measured
as `('geometry','temporal','cross')` while `CONSENSUS_QUERIES[0]` is
`('temporal','geometry','cross')` — the same SET, a different tuple, so it
survives as a separate member. Consensus therefore fires **five** round trips,
not four, and one of the five agrees with the primary *by construction*. Every
"unanimous" figure this repo has quoted was measured over **3 distinct query
sets plus a duplicate**, not 4. The census dedupes by `frozenset` and keeps the
duplicate deliberately, as the A-vs-A control above.

## THE RESULT

Scoreboard order is the operator's: money first, exact accuracy demoted to a
control. `n` is stated on both sides of every comparison.

### DOWN window — mean forward −0.8220%

| | ALL | AGREE |
|---|---|---|
| samples kept | **100** | **23** |
| buy omens placed | **17** | **3** |
| **per-trade net** | **−1.4747%** | **−1.2017%** |
| trough precision (share that paid the round trip) | 17.6% | **0.0%** |
| crest precision (share that fell) | 100.0% (n=4) | — (n=0) |
| exact accuracy *(control)* | 25.0% | 26.1% |

Baselines: majority class 50.0%; **buy-every-bar −1.4720% per trade** over the
same 100 bars.

### UP window — mean forward +0.7880%

| | ALL | AGREE |
|---|---|---|
| samples kept | **100** | **22** |
| buy omens placed | **3** | **0** |
| **per-trade net** | +1.1236% *(n=3)* | **no trades** |
| trough precision | 66.7% *(n=3)* | — (n=0) |
| crest precision (share that fell) | 44.4% (n=18) | 100.0% (n=2) |
| exact accuracy *(control)* | 23.0% | **9.1%** |

Baselines: majority class 42.0%; **buy-every-bar +0.1376% per trade**.

## What this says, plainly

1. **Agreement-gating does not change per-trade net, because it does not keep
   any trades.** It cuts 17 buy omens to 3 in DOWN and 3 to **zero** in UP. A
   gate that abstains on every trade is not an abstention policy, it is being
   switched off — and switched off is free but it is not an edge.
2. **The 3 trades it does keep in DOWN lose money**: −1.2017% per trade, 0 of 3
   paid the 0.6500% round trip. The sample shrink did not buy a paying subset.
   This is the trap the item named, and the answer is that it does not pay.
3. **Agreement does not even buy held-out accuracy.** In UP it *halves* it,
   23.0% → 9.1%. The 99.4%-vs-73.3% asymmetry is real about *reproduction* and
   does not survive the move to bars the fabric has not seen. Confidence was
   already known worthless as a correctness gate; **agreement is worthless too,
   held out.**
4. **Neither ALL arm beats its baseline either.** DOWN: −1.4747% against
   buy-every-bar's −1.4720% — indistinguishable, on 17 trades. UP: +1.1236% on
   **n=3**, which is three trades and is not a result. There is no edge here to
   gate in the first place.
5. **The node is not perfectly deterministic.** A-vs-A moved 4/100 in DOWN and
   0/100 in UP, ~2% overall. Small against a 77% split rate, so it does not
   threaten the finding — but it moved the ALL arm's per-trade net by 0.28
   percentage points between two identical runs on one fabric (−1.1898% then
   −1.4747%), which is a useful bound on any future single-window claim.

## Do NOT re-run this arm expecting a different answer

The negative holds in **both** windows, on one fabric, with the query path
proven live and a determinism control beside it. The next question is not
"gate on agreement harder" — it is that **there is no edge to gate**: the ALL
arm matches buy-every-bar in DOWN and places 3 trades in UP.

What *is* worth taking from here: the **crest** column. Crests called 4/4 fallen
in DOWN and 18 called in UP at 44.4%. The sell-high half is measured nowhere
else in this repo and the census now scores it as an exit signal without
shorting — but it fails the both-windows rule, so it is **not** an edge either.

— Iris, pass 111
