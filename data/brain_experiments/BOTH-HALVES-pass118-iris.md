# The sell-high half, scored held-out beside the buy-low half, UP and DOWN

Pass 118, Iris, 2026-09-11. Item `[3186a96e]`. **Node-free.**

Reproduce:

```
python -X utf8 scripts/omen_both_halves.py --limit 240 --report
```

Artifact: `data/brain_experiments/omen-both-halves-pass118.json`.

## What was missing

Every omen report in this repo scores buying — `buy_omens`, `buy_hit_rate`,
`buy_net_per_trade` — and carries no crest cell at all. Pass 114's true-label
ceiling (`TROUGH-BASE-RATE-AND-CEILING-pass114.md`) made that omission
expensive to keep: in the DOWN window the crest half was the BIGGER of the
two, +1.8884% per trade against the buy half's +1.6396%. That ceiling is
100% precise **by construction** — its caller was the labels — so it says
where the money is, not whether anything can reach it.

This is the other end of that question: a predictor that sees only the past,
with both of its thresholds fitted on a **training** window and scored on a
**held-out** one, reporting both halves through the same function at one
cost and one horizon.

## The corpus, the windows and the predictor

* **Corpus**: 167 eligible corpora under `data/historical_ohlcv`, measured
  cadence exactly 3600s, from the first 240 files by path. Ineligible files
  (wrong cadence, under 896 bars, unparseable) are dropped, not padded.
* **Horizon**: 720 minutes = **12 bars** at 3600s. Stated in minutes because
  `--horizon 12` has meant 12 minutes on one file and 48 days on another.
* **Cost**: round trip 0.6500%, omen threshold 0.9750% (multiple 1.5), the
  shipped values.
* **Windows**: 208 held-out bars, 600 train bars, a full 12-bar gap between
  train_stop and test_start so no training sample's future touches a held-out
  bar. The UP and DOWN windows are **scanned**, not assumed: every candidate
  stop position is measured with `window_regime` and the most extreme UP and
  the most extreme DOWN window each corpus admits are the ones scored. Every
  corpus contributed both, so the two rows below are 167 corpora each.
* **Predictor**: range position over the last 24 closes — the same feature
  `label_omen` uses to decide "at the low" and "at the high", and the only
  half of it that is knowable at the bar. `position <= b_low` is a buy call,
  `position >= b_high` is a sell call, everything else is murk. `b_low` and
  `b_high` are fitted **independently**, each on the train window alone, by
  best per-trade net over a coarse 8-value grid, with a 20-call minimum so a
  one-trade band cannot win.

This is deliberately a weak predictor. It is the floor the brain has to
beat, measured on both halves rather than on one.

## The numbers

Every cell is pooled by `omen_scoreboard.pool_scoreboards` — totals added and
divided once, never a mean of per-corpus means. All four cells clear the
30-trade readability floor by two orders of magnitude, so all four are
quotable. The baseline is every bar in the same window, mirrored for the sell
half (selling every bar).

| window | half | n | net per trade | precision | baseline | edge |
|---|---|---|---|---|---|---|
| UP | buy (trough) | 4641 | **+1.1258%** | 62.9% | +1.2206% | **−0.0948pp** |
| UP | sell (crest) | 16662 | **−2.6211%** | 19.5% | −1.2206% | **−1.4005pp** |
| DOWN | buy (trough) | 15680 | **−2.5629%** | 17.0% | −2.4246% | **−0.1382pp** |
| DOWN | sell (crest) | 4799 | **+1.2379%** | 62.5% | +2.4246% | **−1.1867pp** |

Every-bar n = 34736 per window (167 × 208).

## What it means

**The sell half is not free headroom, and pass 114's ceiling must not be
sized as if it were.** Held out, the crest half loses to its own baseline by
1.40pp in the UP window and 1.19pp in the DOWN window — the two worst cells
in the table. The buy half is roughly flat: −0.09pp and −0.14pp, which is a
weak rule reproducing the drift it sits in.

The reason the DOWN crest number looked big in pass 114 is now visible: in a
down window *everything* falls, so selling pays +2.4246% per trade with no
skill at all. A crest label picks a subset of those bars, and the subset is
worse than the whole. The same shape runs the other way in the UP window,
where the buy half's +1.1258% is below a +1.2206% you get for holding.
**A raw per-trade percentage in a directional window is mostly the window.**
Only the edge column is a statement about the caller.

The two high-precision cells (62.9% and 62.5%) are the same effect a third
time: precision against a cost threshold in a window that moves that way is
cheap, and it is not correlated with beating the baseline.

## What this does NOT say

It does not say the sell half is unlearnable. It says a 24-bar range
position, thresholded, does not learn it — and that the ceiling the item was
filed on is a property of the labels. The fitted bands say the same thing
plainly: the buy band lands on the extreme 0.05 in 46 UP and 53 DOWN corpora
but on the degenerate 0.40–0.50 (call almost every bar) in 65 UP and 40 DOWN
corpora, and the sell band lands on 0.50 — call every bar above the midpoint
— in 38 UP and 46 DOWN corpora. Fitting picks "call everything" about as
often as it picks a real extreme, which is what a feature with no signal in
it looks like when you fit a threshold to it.

**The next question this raises**: the honest baseline for a half-book in a
directional window is not "every bar", it is "every bar, in a window of the
same regime". Both are reported above and the edge column already uses the
matched one. Any future sell-side arm must be scored against the mirrored
baseline the same way, or a down window will hand it a fake +2.4% and this
repo will pay for it a fourth time.
