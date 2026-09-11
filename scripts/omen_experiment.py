"""Measure the omen brain: does it PRODUCE perfectly, and does it PAY?

Four numbers, in the order they matter. The first three are controls; a
held-out score without them has already fooled this repo once (a "78%
directional accuracy" that was the market's own down-drift, inverted,
because the model was answering the same thing to everything).

  1. TRAIN RECALL      -- re-predict frames the brain was trained on. This
                          is the "produce perfectly" bar. Anything below
                          ~100% means the substrate is losing bindings and
                          no held-out number from it can be trusted.
  2. GARBAGE CONTROL   -- predict on frames built from noise. A healthy
                          brain returns few or no confident actionable
                          omens here, and more than one distinct token
                          across the whole set.
  3. HELD-OUT          -- exact-label accuracy on bars strictly after the
                          training window, against TWO baselines: the
                          majority class, and the market's own up-rate.
  4. NET-OF-COST P/L   -- act on every admitted ``trough`` (buy, exit
                          ``horizon`` bars later), charge ROUND_TRIP_COST on
                          every round trip, and report the total. This is
                          the only line that is about money.

Usage
-----
  python -X utf8 scripts/omen_experiment.py --corpus data/historical_ohlcv/base/0004_AERO-USDC.json \
      --train 3000 --test 600 --horizon 12

The node it talks to defaults to OMEN_BRAIN_ENDPOINT (127.0.0.1:8091) --
its own process with its own brain dir, so this never touches the fabric
the live regime signal reads on :8090.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading.omen_brain import (  # noqa: E402
    COLLECTIONS, LOOKBACK_BARS, OMEN_ACTIONS, OMEN_CREST, OMEN_LABELS,
    OMEN_MURK, OMEN_TROUGH, PREDICT_COLLECTIONS, ROUND_TRIP_COST, OmenBrain,
    build_collections, collection_distinctness, discriminating_collections,
    label_omen, label_regime, omen_threshold,
)
from trading.omen_metacognition import self_frames  # noqa: E402
from trading.omen_resolved_history import ResolvedHistory  # noqa: E402

#: [1f8c2461] THE SELF-FRAME ABSTENTION GATE. Measured pass 111: the self
#: frames carry real information about the trough label in BOTH market
#: directions (self_outcome DOWN +0.183 against a null median +0.046, p=0.001;
#: UP +0.251 against +0.061, p=0.000) AND querying those same pools COST
#: 0.81pp per trade UP and 1.32pp DOWN. Both are true. The dilution law is a
#: fact about the QUERY MECHANISM, not about whether a stream is informative,
#: so the frames are used OUTSIDE the query: the fabric answers exactly as it
#: does today, and the self frame at that bar decides whether the buy is
#: PLACED. Abstention is free; a wrong trade costs ROUND_TRIP_COST.
SELF_GATE_KEYS = ("self_outcome", "self_error_run")

#: A bucket needs this many TRAIN samples before its trough rate is allowed
#: to refuse anything. Below it, "0.000 trough" is a small-sample accident --
#: at a 13-15% base rate a 10-sample bucket reads zero by chance 20% of the
#: time.
SELF_GATE_MIN_SUPPORT = 30


def fit_self_gate(train_samples: Sequence[Mapping[str, Any]],
                  *, min_support: int = SELF_GATE_MIN_SUPPORT) -> Dict[str, Any]:
    """Buckets whose TRAIN trough rate is exactly zero over enough samples.

    Fitted on the train window ONLY and frozen before a single held-out bar is
    scored. That ordering is the whole point: the L1 motif map in pass 110 was
    a map fitted to the window it was then scored on, and it died of it. The
    fitted bucket list is printed and written into the report so a reader can
    check it was not tuned to the scoring window's regime.
    """
    refuse: Dict[str, List[str]] = {}
    support: Dict[str, Dict[str, List[int]]] = {}
    for key in SELF_GATE_KEYS:
        counts: Dict[str, List[int]] = {}
        for sample in train_samples:
            frame = sample.get("self", {}).get(key)
            if frame is None:
                continue
            cell = counts.setdefault(frame, [0, 0])
            cell[0] += 1 if sample["label"] == OMEN_TROUGH else 0
            cell[1] += 1
        refuse[key] = sorted(
            frame for frame, (troughs, seen) in counts.items()
            if seen >= min_support and troughs == 0
        )
        support[key] = counts
    troughs = sum(1 for s in train_samples if s["label"] == OMEN_TROUGH)
    return {
        "min_support": int(min_support),
        "train_samples": len(train_samples),
        "train_trough_rate": troughs / max(1, len(train_samples)),
        "refuse": refuse,
        "bucket_support": support,
    }


def self_gate_refuses(gate: Mapping[str, Any],
                      sample: Mapping[str, Any]) -> str | None:
    """The key whose frozen bucket refuses this bar, or None to allow it."""
    frames = sample.get("self") or {}
    for key in SELF_GATE_KEYS:
        if frames.get(key) in gate["refuse"].get(key, ()):  # frozen list
            return key
    return None


def _causal_majority(visible) -> str:
    """The most common SETTLED outcome this bar is allowed to have seen.

    Deliberately not the node's own call: sizing the self-pool vocabulary and
    measuring the node's skill are different jobs, and only the first belongs
    in a sample builder that runs before any node exists.
    """
    actuals = [row.actual for row in visible if row.actual]
    if not actuals:
        return OMEN_MURK
    return Counter(actuals).most_common(1)[0][0]


def _pct(values: Sequence[float]) -> str:
    """min / median / p90 / max, so a floor can be read off the gap."""
    if not values:
        return "n/a"
    ordered = sorted(values)
    def at(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
    return (f"min {ordered[0]:.3f} med {at(0.5):.3f} "
            f"p90 {at(0.9):.3f} max {ordered[-1]:.3f}")


def load_bars(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    bars = raw if isinstance(raw, list) else raw.get("bars") or raw.get("data") or []
    bars = [b for b in bars if b.get("close")]
    bars.sort(key=lambda b: int(b["timestamp"]))
    return bars


def bar_seconds(bars: Sequence[Mapping[str, Any]]) -> int:
    """The corpus's own cadence, measured -- never assumed to be hourly."""
    gaps = [int(bars[i]["timestamp"]) - int(bars[i - 1]["timestamp"])
            for i in range(1, min(len(bars), 400))]
    gaps = [g for g in gaps if g > 0]
    return int(sorted(gaps)[len(gaps) // 2]) if gaps else 3600


# --- horizon: a wall-clock promise, not a bar count ------------------------
#
# Measured pass 113 over ALL 629 files in data/historical_ohlcv: the cadence
# runs 60s to 345600s. ``--horizon 12`` therefore asked about 12 minutes on one
# file and 48 DAYS on another, and every report wrote the same string "h12" for
# both. Whether a prediction target can pay for itself is a function of the
# horizon in MINUTES -- the live cost floor is 0.3592% of notional and the
# share of ticks whose realised move outruns it is 17.8% at 5 min against 80.3%
# at 240 min -- so minutes is the unit the experiment must take.

#: A corpus file whose median gap is coarser than this is a sparse or broken
#: download, not a timeframe anybody chose: a 12-bar horizon on the 345600s
#: file is a 48-day forecast, and the two 86400s files are daily candles that
#: cannot answer an intraday question. Anything at or below 4 hours is a real
#: timeframe somebody downloaded on purpose (3600s dominates, 7200s and 14400s
#: exist). SELECTION over the corpus excludes coarser files; an explicit
#: ``--corpus`` still runs, loudly, because naming one file is a decision.
MAX_CORPUS_BAR_SECONDS = 14400

#: The second half of the same bound. A file's HEAD cadence (the first 400
#: gaps, which is what ``bar_seconds`` returns and therefore what every horizon
#: is converted with) can disagree with its full-file cadence. A real feed
#: jitters -- 3648s head against 3624s full-file is one hourly instrument with
#: gaps -- but a 2.00x disagreement means the file holds TWO timeframes and the
#: horizon is wrong by that factor somewhere inside it. Measured pass 113: 31
#: of 629 files exceed this, and every one of them sits at or near an exact 2x,
#: 3x or 4x. Selection excludes them; scripts/omen_corpus_cadence_census.py
#: names each one.
MAX_CADENCE_DRIFT_RATIO = 1.25

#: 720 minutes is exactly the old 12-bar default on the 3600s cadence that most
#: of this corpus carries, so every hourly run ever recorded reproduces
#: bit-for-bit under the new unit. It is NOT a claim that 12 hours is the right
#: question to ask -- [15cc71d4] measured that a perfect direction call loses
#: money at 5 and 10 minutes on this feed, and the horizon that pays is an open
#: decision. Changing this number changes what the brain is trained to predict.
DEFAULT_HORIZON_MINUTES = 720.0


def horizon_bars(minutes: float, cadence_seconds: int) -> int:
    """A wall-clock horizon in this corpus's own bars, never fewer than one.

    Rounds to nearest so a 10-minute horizon on 600s bars is 1 bar rather than
    0; a zero-bar horizon would compare a bar's close against itself and report
    a flawless, free, entirely fictional edge.
    """
    if cadence_seconds <= 0:
        raise ValueError("cadence must be positive")
    return max(1, int(round(float(minutes) * 60.0 / cadence_seconds)))


def resolve_horizon(cadence_seconds: int, minutes: float | None = None,
                    bars: int | None = None) -> Dict[str, Any]:
    """The one place bars and minutes are reconciled, for every caller.

    Returns both units and which one the caller ASKED in, because a report
    that records only the bars cannot be compared with another corpus and a
    report that records only the minutes cannot be reproduced. Exactly one of
    ``minutes`` and ``bars`` may be given; giving both is a contradiction the
    caller has to resolve rather than have silently picked for them.
    """
    if minutes is not None and bars is not None:
        raise ValueError(
            "--horizon-minutes and --horizon are the same quantity in two "
            "units: pass one. Bars are corpus-specific; minutes are not.")
    if bars is not None:
        if int(bars) < 1:
            raise ValueError("horizon must be at least 1 bar")
        n_bars = int(bars)
        return {"horizon_bars": n_bars,
                "horizon_minutes": round(n_bars * cadence_seconds / 60.0, 4),
                "horizon_source": "bars"}
    if minutes is None:
        raise ValueError("no horizon given")
    if float(minutes) <= 0:
        raise ValueError("horizon minutes must be positive")
    n_bars = horizon_bars(minutes, cadence_seconds)
    return {"horizon_bars": n_bars, "horizon_minutes": float(minutes),
            "horizon_source": "minutes"}


#: Every report under data/brain_experiments/ must carry all three, because
#: any two of them determine the third and a reader with only one cannot tell
#: what question was asked. Checked at WRITE time so a report that would be
#: uncomparable never reaches the directory.
REPORT_HORIZON_FIELDS = ("horizon_bars", "horizon_minutes", "bar_seconds")


def validate_report_horizon(report: Mapping[str, Any]) -> None:
    """Reject a report that does not say what horizon it measured."""
    missing = [f for f in REPORT_HORIZON_FIELDS if report.get(f) is None]
    if missing:
        raise ValueError(
            f"report is missing {missing}: a report carrying only some of "
            f"{list(REPORT_HORIZON_FIELDS)} cannot be compared with a run on "
            f"another corpus, which is the whole failure this check exists for")
    bars_ = report["horizon_bars"]
    minutes_ = report["horizon_minutes"]
    cadence = report["bar_seconds"]
    if bars_ < 1 or minutes_ <= 0 or cadence <= 0:
        raise ValueError(
            f"report horizon is not positive: bars={bars_} minutes={minutes_} "
            f"bar_seconds={cadence}")
    implied = bars_ * cadence / 60.0
    # Rounding to whole bars moves the minutes by at most half a bar.
    if abs(implied - minutes_) > (cadence / 60.0) / 2 + 1e-6:
        raise ValueError(
            f"report horizon disagrees with itself: {bars_} bars of "
            f"{cadence}s is {implied:.2f} min, not {minutes_:.2f} min")


class WindowError(ValueError):
    """The requested train/test split does not fit the corpus."""


def subset_lift_pvalue(all_net: Sequence[float], k: int, observed: float,
                       trials: int = 20000, seed: int = 1234) -> Dict[str, float]:
    """Could a random k of these same bars have paid as well as the omens?

    The buy omens are a SUBSET of the held-out bars, not an independent
    sample, so comparing ``omen mean`` against ``every-bar mean`` as two
    independent means understates the error and manufactures an edge out of
    selection noise. The honest null is "these k bars were picked at random",
    and it is computed by drawing ``trials`` random k-subsets of the actual
    held-out net returns and asking where the observed mean falls.

    Pure arithmetic and seeded, so it is testable without a node and gives
    the same answer twice. Measured pass 114 on AERO-USDC: the UP window's
    buy omens paid +1.4887% against +0.5068% for every bar -- which reads as
    a 0.98pp edge and is p=0.073, because the per-bar sd is 5.26% and 52
    trades cannot resolve 1pp.
    """
    n = len(all_net)
    if n == 0 or k <= 0 or k > n:
        return {"every_bar_mean": 0.0, "null_se": 0.0, "z": 0.0, "p_value": 1.0,
                "trials": 0}
    every_bar = sum(all_net) / n
    rng = random.Random(seed)
    pool = list(all_net)
    sims = [sum(rng.sample(pool, k)) / k for _ in range(trials)]
    mean_sim = sum(sims) / len(sims)
    var = sum((v - mean_sim) ** 2 for v in sims) / len(sims)
    se = var ** 0.5
    at_or_above = sum(1 for v in sims if v >= observed)
    return {
        "every_bar_mean": every_bar,
        "null_se": se,
        "z": ((observed - every_bar) / se) if se > 0 else 0.0,
        # One-sided: we only ever claim the omens did BETTER than random.
        "p_value": at_or_above / len(sims),
        "trials": len(sims),
    }


def plan_windows(total_bars: int, train: int, test: int, horizon: int,
                 train_end: int | None = None,
                 test_end: int | None = None) -> Dict[str, int]:
    """Bar indices for the train and held-out windows.

    Pure arithmetic, kept out of ``main`` so the one property this whole
    two-window protocol rests on can be tested without a node: with
    ``train_end`` pinned, moving ``test_end`` must NOT move the train
    window. Two held-out windows measured against DIFFERENT training data
    are not two measurements of one fabric, they are two experiments, and
    comparing them says nothing -- which is the trap the default derivation
    walks into, because it computes ``train_stop`` from ``test_start``.

    A full ``horizon`` gap sits between train_stop and test_start so no
    training sample's future overlaps a held-out bar.
    """
    test_stop = (total_bars - horizon - 1) if test_end is None else int(test_end)
    test_stop = min(test_stop, total_bars - horizon - 1)
    test_start = test_stop - test
    if train_end is None:
        train_stop = test_start - horizon
    else:
        train_stop = int(train_end)
        if train_stop + horizon > test_start:
            raise WindowError(
                f"train_end {train_stop} + horizon {horizon} overruns "
                f"test_start {test_start}: the training window's future would "
                f"overlap the held-out window, which leaks the answer")
    train_start = train_stop - train
    if train_start < LOOKBACK_BARS:
        raise WindowError(
            f"train window starts at {train_start}, before the {LOOKBACK_BARS}-bar "
            f"lookback: need {LOOKBACK_BARS + train + horizon + test + horizon} "
            f"bars before this test window, have {total_bars}")
    return {"train_start": train_start, "train_stop": train_stop,
            "test_start": test_start, "test_stop": test_stop}


def window_regime(bars: Sequence[Mapping[str, Any]], start: int, stop: int,
                  horizon: int) -> Dict[str, Any]:
    """Up-rate and drift of a bar range, so a window's DIRECTION is stated.

    The standing rule here is that a single window is never evidence: a
    long-only rule flatters itself in an up window, and that error has
    already produced a fake 78% and a fake +0.9067% in this repo. So every
    window a run touches reports whether it was UP or DOWN, measured, and
    the report carries it.
    """
    forwards = []
    for index in range(max(start, 0), min(stop, len(bars) - horizon)):
        close = float(bars[index]["close"])
        if close <= 0:
            continue
        forwards.append((float(bars[index + horizon]["close"]) - close) / close)
    if not forwards:
        return {"bars": 0, "up_rate": 0.0, "mean_forward": 0.0,
                "zero_share": 0.0, "regime": "EMPTY"}
    # Bars that did not MOVE are neither up nor down, and they must not be
    # counted as down. A frozen feed republishing one price gives every bar
    # a forward of exactly 0.0; scoring those as "not up" reads up_rate 0.0
    # and would name a dead feed the DOWN window. This repo has shipped
    # frozen feeds before -- 82 of 94 symbols once held a seed price -- so
    # the zero share is measured, reported, and gates the verdict.
    moved = [f for f in forwards if f != 0.0]
    zero_share = 1.0 - len(moved) / len(forwards)
    up_rate = (sum(1 for f in moved if f > 0) / len(moved)) if moved else 0.0
    if zero_share > 0.5:
        # More than half the window did not move: that is not a direction,
        # and it is a reason to distrust the corpus slice.
        regime = "FLAT"
    else:
        # 50% +/- 2 points is not a direction either, it is a coin. Naming
        # that band FLAT stops a 50.4% window being called "the UP window".
        regime = "UP" if up_rate > 0.52 else "DOWN" if up_rate < 0.48 else "FLAT"
    return {"bars": len(forwards), "up_rate": up_rate, "zero_share": zero_share,
            "mean_forward": sum(forwards) / len(forwards), "regime": regime,
            "ts_start": int(bars[max(start, 0)]["timestamp"]),
            "ts_stop": int(bars[min(stop, len(bars)) - 1]["timestamp"])}


def backpressure_probe(endpoint: str | None) -> Dict[str, Any]:
    """Ask the node whether it will actually LEARN, before we spend an hour.

    Measured 2026-09-10 pass 108: a node with a 4096 MB consolidation floor
    on a box with 2903 MB free answers /health OK, returns a full plausible
    /brain/stats, and serves /brain/observe in 0.16s -- while REFUSING every
    supervised binding. total_binding froze at 415 while the training loop
    still looked alive.

    The client retries a refused sample WIZARD_BACKPRESSURE_RETRIES times
    (default 30) at 2s across two stages, so a run under backpressure does
    not fail fast: it takes up to 120 SECONDS PER SAMPLE and reports
    failed_pairs at the end of a window that has already closed. Nothing was
    dishonest about that accounting -- it was just far too late to act on.

    A result measured under backpressure is not a weak result, it is not a
    result: the fabric never learned the samples the report says it taught.
    So this is a hard stop, not a warning.
    """
    from http.client import HTTPConnection  # local: only the preflight needs it
    from urllib.parse import urlparse

    target = endpoint or os.getenv("OMEN_BRAIN_ENDPOINT") or "http://127.0.0.1:8091"
    parsed = urlparse(target if "//" in target else f"http://{target}")
    try:
        conn = HTTPConnection(parsed.hostname or "127.0.0.1",
                              parsed.port or 80, timeout=15)
        # An empty stream list is a no-op binding: it asks the node's ingest
        # gate the question without teaching it anything.
        conn.request("POST", "/brain/consolidate/multi",
                     json.dumps({"streams": [], "outcome_pool": 0,
                                 "outcome_frame": ""}),
                     {"Content-Type": "application/json"})
        reply = json.loads(conn.getresponse().read() or b"{}")
    except Exception as exc:  # noqa: BLE001 -- any failure here is unknown, not OK
        return {"reachable": False, "error": str(exc)}
    return {"reachable": True, "backpressure": bool(reply.get("backpressure")),
            "available_mb": reply.get("available_mb"),
            "floor_mb": reply.get("floor_mb"),
            "retry_after_ms": reply.get("retry_after_ms")}


def fabric_census(endpoint: str | None) -> Dict[str, Any]:
    """What the node's fabric holds RIGHT NOW, so a report can prove it.

    THE HOLE THIS CLOSES, and it cost item dcd6d654 three passes. Every report
    in data/brain_experiments/ names its corpus, its windows and its numbers,
    and NOT ONE of them names the fabric that produced them. The pass-107
    baseline (omen-AERO-USDC-h12-20260910-133426.json, held-out 0.2850 against
    a 0.3025 majority) cannot be shown to have come from a clean fabric,
    because nothing recorded one: no brain dir, no node, no atom count. The
    disk cannot answer it either -- no brain-data* directory under
    W1z4rDV1510n was written at all between 13:15 and 13:36 that day, so file
    mtimes do not identify the run.

    node_id does NOT discriminate: :8090 (production) and :8091 (experiment)
    both answer ``node-cd4c5a9a7225``, because the id is the host's, not the
    fabric's. total_neurons IS the discriminator, and it is the RIGHT one --
    it is a property of the fabric rather than of the directory name, so it
    catches a fresh-looking dir that loaded a warm checkpoint just as well as
    it catches a re-used one.

    An unreachable node is recorded as an error rather than raising: the
    census is evidence about the run, and losing the run to a failed census
    would be worse than a report that says the census failed.
    """
    from http.client import HTTPConnection  # local: only the census needs it
    from urllib.parse import urlparse

    target = endpoint or os.getenv("OMEN_BRAIN_ENDPOINT") or "http://127.0.0.1:8091"
    parsed = urlparse(target if "//" in target else f"http://{target}")
    out: Dict[str, Any] = {"endpoint": target}
    for label, route in (("health", "/health"), ("stats", "/brain/stats")):
        try:
            conn = HTTPConnection(parsed.hostname or "127.0.0.1",
                                  parsed.port or 80, timeout=30)
            conn.request("GET", route)
            out[label] = json.loads(conn.getresponse().read() or b"{}")
        except Exception as exc:  # noqa: BLE001 -- a failed census is evidence
            out[label] = {"error": str(exc)}
    stats = out.get("stats") or {}
    out["total_neurons"] = stats.get("total_neurons")
    out["total_concepts"] = stats.get("total_concepts")
    out["total_binding"] = stats.get("total_binding")
    out["pool_count"] = stats.get("pool_count")
    out["tick"] = stats.get("tick")
    return out


def fabric_is_empty(census: Dict[str, Any]) -> bool:
    """True only when the node ANSWERED and answered zero on every counter.

    A census that failed is not an empty fabric, and must not read as one --
    that is the exact shape of a guard that passes when its evidence is
    missing.
    """
    counters = (census.get("total_neurons"), census.get("total_concepts"),
                census.get("total_binding"), census.get("tick"))
    if any(c is None for c in counters):
        return False
    return all(int(c) == 0 for c in counters)


def build_samples(bars, symbol, chain, horizon, start, stop):
    """Frames + true label for every bar in [start, stop) that has both.

    THE SEAM THIS CLOSES, measured pass 110 on a real 19-pool node. The query
    path probe reported ``QUERY PATH DEAD`` for a query set differing only by
    pools 15/16/19 -- control 0/60, treatment 0/60 -- while the B arm fired
    SIX streams per prediction. The pools were sent and they were read. They
    moved nothing because this loop called ``build_collections`` WITHOUT
    ``history=``, so every self frame in every training set was the ``na``
    sentinel and the three pools trained as CONSTANTS. A constant stream
    cannot move a query however good the pool is.

    So a ``ResolvedHistory`` is now walked alongside the bars and handed in.
    Causality is inherited from it rather than re-implemented: the frame for
    bar ``i`` reads ``as_of(i)``, which cannot return a row whose horizon has
    not landed, and ``self_frames`` drops any unresolved row on top of that --
    the guard against the prediction_error loop that took recall 100% -> 30%.

    Passing ``history=`` is a NO-OP when ``OMEN_META_COLLECTIONS`` is off:
    ``build_collections`` reads it only behind that flag, so a non-meta run
    produces byte-identical frames to before this change.
    """
    history = ResolvedHistory(horizon)
    samples = []
    for index in range(max(start, LOOKBACK_BARS), stop):
        label = label_omen(bars, index, horizon_bars=horizon)
        # Settle FIRST: a prediction whose horizon lands exactly here is a
        # fact by the time this bar is decided. The guard is against reading
        # the OPEN call, not a settled one.
        if label is not None:
            try:
                history.settle(index, label)
            except ValueError:
                pass
        visible = history.as_of(index)
        if label is None:
            continue
        try:
            frames = build_collections(bars, index, horizon_bars=horizon,
                                       symbol=symbol, chain=chain,
                                       history=visible)
        except (ValueError, IndexError):
            continue
        # The recorded prediction is the causal majority of what this bar is
        # allowed to see -- non-oracle, and the same rule the scoreboard
        # baselines against. It is what makes pools 15/16/19 vary at all.
        history.record(index, _causal_majority(visible))
        samples.append({
            "index": index,
            "frames": frames,
            # The self frame is attached to EVERY sample regardless of
            # OMEN_META_COLLECTIONS, because the gate reads it outside the
            # query. It is not sent to the node from here, so a non-meta run
            # still produces byte-identical query frames.
            "self": self_frames(visible),
            "label": label,
            "regime": label_regime(bars, index),
            "ts": int(bars[index]["timestamp"]),
            "price": float(bars[index]["close"]),
            "forward": (float(bars[index + horizon]["close"])
                        - float(bars[index]["close"])) / float(bars[index]["close"]),
        })
    return samples


def balance(samples, rng):
    """Undersample the majority so decode cannot collapse to one attractor.

    Measured 2026-07: unbalanced training at scale made every prediction
    "steady". ``murk`` is the majority here for the same structural reason,
    so it is capped at the mean count of the actionable labels.
    """
    by_label: Dict[str, List[dict]] = {}
    for sample in samples:
        by_label.setdefault(sample["label"], []).append(sample)
    actionable = [len(v) for k, v in by_label.items() if k != OMEN_MURK]
    if not actionable:
        return samples
    cap = max(1, int(sum(actionable) / len(actionable)))
    out: List[dict] = []
    for label, group in by_label.items():
        if label == OMEN_MURK and len(group) > cap:
            # Prefer recent murk bars -- the recent regime is the one we
            # will be trading in.
            out.extend(group[-cap:])
        else:
            out.extend(group)
    rng.shuffle(out)
    return out


def garbage_frames(rng, count):
    """Frames with the right SHAPE and meaningless content.

    Same prefixes and key names as a real frame, random bucket values. A
    brain that answers these as confidently as real ones is pattern-free.
    """
    out = []
    for _ in range(count):
        frames = {}
        for collection in COLLECTIONS:
            tokens = " ".join(
                f"k{i}={rng.choice('udz')}{rng.randrange(25)}" for i in range(6))
            frames[collection.name] = f"{collection.prefix} {tokens}"
        out.append(frames)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--test", type=int, default=400)
    parser.add_argument("--horizon-minutes", type=float, default=None,
                        help=f"MINUTES of wall clock the omen is about, "
                             f"converted to bars using this corpus's own "
                             f"measured cadence (default "
                             f"{DEFAULT_HORIZON_MINUTES:.0f}, which is exactly "
                             f"the old 12-bar default on the 3600s cadence "
                             f"most of this corpus carries). Minutes is the "
                             f"default unit because bars are not comparable "
                             f"across a corpus spanning 60s to 345600s: the "
                             f"same '--horizon 12' asked about 12 minutes on "
                             f"one file and 48 DAYS on another, and both "
                             f"reports wrote 'h12'")
    parser.add_argument("--horizon", type=int, default=None,
                        help="bars ahead the omen is about -- an EXPLICIT "
                             "override of --horizon-minutes, kept so an old "
                             "run can be reproduced bar-for-bar. It asks a "
                             "different question of every cadence, so the "
                             "report records the minutes it worked out to")
    parser.add_argument("--recall-sample", type=int, default=200)
    parser.add_argument("--garbage", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--allow-warm-fabric", action="store_true",
                        help="train onto a fabric that already holds atoms. "
                             "The result is not comparable to a clean-fabric "
                             "run and the report records that it was warm.")
    parser.add_argument("--chain", default="base")
    parser.add_argument("--report-dir", default="data/brain_experiments")
    parser.add_argument("--skip-train", action="store_true",
                        help="re-measure an already-trained fabric; the "
                             "sample split is deterministic in --seed so the "
                             "recall set is the same one it was taught")
    parser.add_argument("--query-collections", default=None,
                        help="comma-separated override of the collections a "
                             "PREDICTION fires (default: measured)")
    parser.add_argument("--consensus", action="store_true",
                        help="require CONSENSUS_QUERIES to agree; abstains "
                             "with verdict='split' when they do not")
    parser.add_argument("--train-end", type=int, default=None,
                        help="PIN the last bar of the training window. Use with "
                             "--test-end to measure a second held-out window on "
                             "the SAME fabric: without it the train window is "
                             "derived from the test window, so moving the test "
                             "window silently retrains on different data")
    parser.add_argument("--test-end", type=int, default=None,
                        help="last bar of the held-out window (default: end of "
                             "corpus). This is how an UP window and a DOWN "
                             "window are each measured")
    parser.add_argument("--list-windows", action="store_true",
                        help="census the corpus for candidate held-out windows "
                             "with their up-rate and drift, then exit -- so an "
                             "UP and a DOWN window are PICKED from measurement "
                             "rather than hoped for")
    parser.add_argument("--ignore-backpressure", action="store_true",
                        help="train even when the node says it will not "
                             "consolidate. The result is NOT a measurement -- "
                             "the fabric never learns the samples the report "
                             "says it was taught")
    parser.add_argument("--self-gate", action="store_true",
                        help="[1f8c2461] Fit the self-frame abstention gate on "
                             "the TRAIN window, freeze it, and report the buy "
                             "book both ungated and gated. Never changes what "
                             "is sent to the node -- the frames are used "
                             "OUTSIDE the query.")
    parser.add_argument("--self-gate-min-support", type=int,
                        default=SELF_GATE_MIN_SUPPORT,
                        help="train samples a bucket needs before its zero "
                             "trough rate may refuse a buy")
    parser.add_argument("--guess-regime", action="store_true",
                        help="let stage 1 guess the regime instead of "
                             "computing it -- the pre-2026-09-07 behaviour, "
                             "kept so the fix stays falsifiable")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    path = Path(args.corpus)
    symbol = path.stem.split("_", 1)[-1]
    bars = load_bars(path)
    cadence = bar_seconds(bars)
    threshold = omen_threshold()

    # The horizon is settled HERE, once, in both units -- every later use of
    # args.horizon is bars, and horizon_minutes is what the report and the
    # filename say, because that is the only unit two corpora share.
    try:
        resolved = resolve_horizon(
            cadence,
            minutes=(DEFAULT_HORIZON_MINUTES
                     if args.horizon is None and args.horizon_minutes is None
                     else args.horizon_minutes),
            bars=args.horizon)
    except ValueError as exc:
        print(f"cannot resolve horizon: {exc}")
        return 2
    args.horizon = resolved["horizon_bars"]
    horizon_minutes = resolved["horizon_minutes"]

    print(f"corpus {path.name}: {len(bars)} bars, {cadence}s cadence, "
          f"symbol {symbol}")
    print(f"horizon {horizon_minutes:.0f} min = {args.horizon} bars of "
          f"{cadence}s (asked in {resolved['horizon_source']}); "
          f"omen threshold {threshold:.4%} "
          f"(round trip {ROUND_TRIP_COST:.4%})")
    if cadence > MAX_CORPUS_BAR_SECONDS:
        # Not fatal: naming one file is a decision. But a run on a 4-day gap
        # is a multi-week forecast and has to say so before it starts.
        print(f"WARNING: {cadence}s cadence is coarser than the "
              f"{MAX_CORPUS_BAR_SECONDS}s selection bound -- this file is "
              f"excluded from any census-driven selection, and this run is "
              f"forecasting {horizon_minutes / 1440:.1f} DAYS ahead")

    # A census of where the UP and DOWN windows actually ARE, so the two
    # windows the standing rule demands are picked from measurement.
    if args.list_windows:
        earliest = LOOKBACK_BARS + args.train + args.horizon + args.test
        latest = len(bars) - args.horizon - 1
        if latest < earliest:
            print(f"corpus too short for any window: need {earliest} bars "
                  f"before the first candidate, have {len(bars)}")
            return 2
        print(f"\ncandidate held-out windows of {args.test} bars "
              f"(train {args.train} + {args.horizon}-bar purge before each):")
        print(f"{'test_end':>9} {'regime':>7} {'up_rate':>8} {'mean_fwd':>10}  window")
        step = max(1, args.test // 2)
        for end in range(latest, earliest - 1, -step):
            info = window_regime(bars, end - args.test, end, args.horizon)
            if not info["bars"]:
                continue
            print(f"{end:>9} {info['regime']:>7} {info['up_rate']:>7.1%} "
                  f"{info['mean_forward']:>+9.4%}  "
                  f"[{end - args.test}, {end})")
        print("\nPick one UP and one DOWN end, then measure BOTH on ONE fabric:")
        print(f"  run 1: --train-end <T> --test-end <UP>")
        print(f"  run 2: --train-end <T> --test-end <DOWN> --skip-train")
        print("  <T> must satisfy T + horizon <= min(UP, DOWN) - test, so the "
              "fabric never saw either window.")
        return 0

    # Chronological split: train strictly before test, and leave a horizon
    # gap so no training sample's FUTURE overlaps a test bar.
    try:
        plan = plan_windows(len(bars), args.train, args.test, args.horizon,
                            train_end=args.train_end, test_end=args.test_end)
    except WindowError as exc:
        print(f"cannot plan windows: {exc}")
        return 2
    train_start = plan["train_start"]
    train_stop = plan["train_stop"]
    test_start = plan["test_start"]
    test_stop = plan["test_stop"]

    train_samples = build_samples(bars, symbol, args.chain, args.horizon,
                                  train_start, train_stop)
    test_samples = build_samples(bars, symbol, args.chain, args.horizon,
                                 test_start, test_stop)
    heldout_regime = window_regime(bars, test_start, test_stop, args.horizon)
    print(f"train bars [{train_start}, {train_stop}) -> {len(train_samples)} samples"
          + ("  (train_end PINNED)" if args.train_end is not None else ""))
    print(f"test  bars [{test_start}, {test_stop}) -> {len(test_samples)} samples")
    print(f"HELD-OUT WINDOW IS {heldout_regime['regime']}: up-rate "
          f"{heldout_regime['up_rate']:.1%}, mean forward "
          f"{heldout_regime['mean_forward']:+.4%} over "
          f"{heldout_regime['bars']} bars. ONE WINDOW IS NOT EVIDENCE -- "
          f"a long-only rule flatters itself in an UP window.")
    print("train label mix :", dict(Counter(s['label'] for s in train_samples)))
    print("test  label mix :", dict(Counter(s['label'] for s in test_samples)))

    # THE GATE IS FITTED AND FROZEN HERE, before the node is even asked for a
    # prediction, so it cannot have seen a held-out bar. The fitted bucket
    # list is printed in full for the same reason.
    self_gate = None
    if args.self_gate:
        self_gate = fit_self_gate(
            train_samples, min_support=args.self_gate_min_support)
        print(f"SELF GATE (fitted on TRAIN ONLY, frozen): "
              f"{self_gate['train_samples']} train samples, train trough rate "
              f"{self_gate['train_trough_rate']:.1%}, min support "
              f"{self_gate['min_support']}")
        for key in SELF_GATE_KEYS:
            buckets = self_gate["refuse"][key]
            print(f"   refuse {key:<15} {len(buckets)} bucket(s)")
            for frame in buckets:
                seen = self_gate["bucket_support"][key][frame][1]
                print(f"      0/{seen:<4} trough on train   {frame!r}")
            if not buckets:
                print("      (none -- no bucket reaches zero trough at this "
                      "support, so the gate cannot refuse on this key)")

    brain = OmenBrain(endpoint=args.endpoint)
    if not brain.supports_multi():
        print("FAIL: node has no /brain/predict/multi -- stale binary or wrong port")
        return 3

    # PROVENANCE. Census the fabric BEFORE a byte is taught, and refuse to
    # train onto one that already holds atoms. A fabric that inherited a
    # warm checkpoint -- production's, or the previous arm's -- looks GOOD on
    # accuracy and cannot be caught from the accuracy number afterwards.
    fabric_before = fabric_census(args.endpoint)
    print(f"   fabric before : neurons={fabric_before.get('total_neurons')} "
          f"concepts={fabric_before.get('total_concepts')} "
          f"binding={fabric_before.get('total_binding')} "
          f"pools={fabric_before.get('pool_count')} "
          f"tick={fabric_before.get('tick')} "
          f"@ {fabric_before.get('endpoint')}")
    if not args.skip_train and not fabric_is_empty(fabric_before):
        print("\nREFUSING TO TRAIN: this fabric is not clean.")
        print(f"  {fabric_before.get('total_neurons')} neurons and "
              f"{fabric_before.get('total_concepts')} concepts are already "
              f"bound at {fabric_before.get('endpoint')}.")
        print("  Whatever it learned before is inside every number this run "
              "would report, and no accuracy number can reveal that after "
              "the fact. Start a node on a FRESH brain dir:")
        print("    D:\\Projects\\W1z4rDV1510n\\start_node.ps1 -Addr "
              "127.0.0.1:8091 -BrainDir <NEW dir> "
              "-Identity brains\\market_predictor_v2.identity.toml")
        print("  --allow-warm-fabric overrides this, and the report will say "
              "so; a warm-fabric result is not comparable to a clean one.")
        if not args.allow_warm_fabric:
            return 5

    # PREFLIGHT. A node that will not consolidate makes a training run a very
    # slow no-op, and the failure is invisible until the run ends.
    if not args.skip_train:
        gate = backpressure_probe(args.endpoint)
        if gate.get("backpressure"):
            print(f"\nREFUSING TO TRAIN: the node is applying ingest backpressure.")
            print(f"  available {gate.get('available_mb')} MB against a "
                  f"{gate.get('floor_mb')} MB consolidation floor")
            print("  It answers /health, /brain/stats and /brain/observe "
                  "normally and still binds NOTHING.")
            print("  Free memory first -- kill omen nodes you are not using -- "
                  "then re-run. A run started now would take up to 120s per "
                  "sample and teach the fabric nothing.")
            print("  --ignore-backpressure overrides this, and the result is "
                  "not a measurement.")
            if not args.ignore_backpressure:
                return 4
        elif not gate.get("reachable"):
            print(f"\nWARNING: could not probe the node's ingest gate "
                  f"({gate.get('error')}). Proceeding, but if training stalls "
                  f"this is the first thing to check.")

    balanced = balance(train_samples, rng)
    print(f"balanced train   : {len(balanced)} samples, "
          f"{dict(Counter(s['label'] for s in balanced))}")

    # 0 -- THE DILUTION LAW. Which collections can discriminate at all on
    # THIS corpus, measured rather than inherited. A query fires the sharp
    # ones only; see the table above PREDICT_COLLECTIONS in trading/omen_brain.
    frame_sets = [s["frames"] for s in balanced]
    distinctness = collection_distinctness(frame_sets)
    measured_query = discriminating_collections(frame_sets)
    print("0. DISTINCTNESS  : " + " ".join(
        f"{name}={distinctness[name]:.3f}"
        for name in sorted(distinctness, key=lambda k: -distinctness[k])))
    print(f"   measured query: {measured_query}   default: {PREDICT_COLLECTIONS}")
    # THE MEASURED SET MUST BE THE SET THAT FIRES. Until 2026-09-10 this line
    # computed measured_query, PRINTED it, and then passed None -- so every run
    # reported the measured set and fired the hard-coded PREDICT_COLLECTIONS.
    # That silently defeats the dilution law this script exists to apply, and
    # it makes any experiment that CHANGES which collections discriminate
    # unfalsifiable: the query set never moves, so the arms cannot differ.
    # It is why the pass-108 relation arm came back byte-for-byte identical to
    # the flat arm over 180 held-out predictions.
    if args.query_collections:
        query = tuple(n.strip() for n in args.query_collections.split(","))
        print(f"   OVERRIDE      : {query}")
    else:
        query = tuple(measured_query)
        print(f"   FIRING        : {query} (measured on this corpus)")

    # Frame collisions cap recall no matter how good the substrate is: two
    # identical frame tuples carrying different labels cannot both be
    # reproduced. Reported so a recall number is never blamed on the brain
    # when it belongs to the corpus.
    keyed: Dict[tuple, set] = {}
    for sample in balanced:
        keyed.setdefault(tuple(sorted(sample["frames"].items())), set()).add(
            sample["label"])
    conflicts = sum(1 for labels in keyed.values() if len(labels) > 1)
    print(f"   collisions    : {len(balanced) - len(keyed)} duplicate tuples, "
          f"{conflicts} with conflicting labels "
          f"(recall ceiling {1.0 - conflicts / max(1, len(balanced)):.1%})")

    started = time.time()
    if args.skip_train:
        print("skipping training -- re-measuring the fabric already on the node")
        train_secs = 0.0
    else:
        for count, sample in enumerate(balanced, 1):
            brain.train(sample["frames"], sample["label"], sample["regime"])
            if count % 250 == 0:
                rate = count / max(1e-9, time.time() - started)
                print(f"  trained {count}/{len(balanced)} ({rate:.1f}/s, "
                      f"{brain.failed_pairs} failed)")
        train_secs = time.time() - started
        print(f"trained {brain.trained_pairs} pairs, {brain.failed_pairs} failed, "
              f"in {train_secs / 60:.1f} min")

    def predict(sample):
        return brain.predict(
            sample["frames"], symbol=symbol, chain=args.chain,
            as_of_ts=sample["ts"], price=sample["price"],
            horizon_bars=args.horizon, bar_seconds=cadence,
            # The regime is arithmetic here: we hold the bars. --guess-regime
            # restores the old chain so the two can be compared in one run.
            regime=None if args.guess_regime else sample["regime"],
            query_collections=query, consensus=args.consensus)

    # 1 -- TRAIN RECALL. Does it reproduce what it was taught?
    recall_set = rng.sample(balanced, min(args.recall_sample, len(balanced)))
    recall_hits = 0
    recall_split = 0
    recall_misses: List[Dict[str, str]] = []
    for sample in recall_set:
        omen = predict(sample)
        if omen.verdict == "split":
            recall_split += 1
            continue
        if omen.omen == sample["label"] and omen.verdict == "admitted":
            recall_hits += 1
        elif len(recall_misses) < 12:
            recall_misses.append({"expected": sample["label"],
                                  "got": omen.omen, "verdict": omen.verdict})
    # Recall is scored over the asks the brain ANSWERED. An abstention is
    # neither a hit nor a miss, and the abstention rate is printed beside it
    # so a high recall bought by refusing everything cannot hide.
    answered = len(recall_set) - recall_split
    recall = recall_hits / max(1, answered)
    print(f"\n1. TRAIN RECALL  : {recall:.1%} ({recall_hits}/{answered} answered"
          + (f", {recall_split} split/abstained of {len(recall_set)}"
             if recall_split else "") + ")")
    if recall_misses:
        print("   misses       :", recall_misses[:6])

    # 2 -- GARBAGE CONTROL.
    garbage_labels: List[str] = []
    garbage_conf: List[float] = []
    garbage_actionable = 0
    for frames in garbage_frames(rng, args.garbage):
        # No regime is supplied: noise has no bars to compute one from, and
        # this is the one caller that genuinely cannot.
        omen = brain.predict(frames, symbol="garbage", chain=args.chain,
                             as_of_ts=0, price=1.0, horizon_bars=args.horizon,
                             bar_seconds=cadence, query_collections=query)
        garbage_labels.append(f"{omen.omen}/{omen.verdict}")
        garbage_conf.append(omen.confidence)
        if omen.is_actionable:
            garbage_actionable += 1
    print(f"2. GARBAGE       : {len(set(garbage_labels))} distinct answers "
          f"of {len(garbage_labels)}, {garbage_actionable} actionable, "
          f"conf {_pct(garbage_conf)}")

    # 3 -- HELD-OUT, against two baselines.
    predicted: List[str] = []
    exact = 0
    trades: List[float] = []
    admitted = 0
    real_conf: List[float] = []
    #: Every buy omen, paired with its confidence, so the floor that would
    #: have been best is READ OFF the run instead of guessed at.
    buys_by_conf: List[tuple] = []
    #: The gate's book, and the precision of both books. Trough PRECISION is
    #: reported on each side because "the gate helped" and "the gate stopped
    #: trading" look identical in a per-trade mean: refusing every buy is not
    #: an edge.
    gated_trades: List[float] = []
    gated_is_trough: List[bool] = []
    ungated_is_trough: List[bool] = []
    gate_refusals: Counter = Counter()
    for sample in test_samples:
        omen = predict(sample)
        predicted.append(omen.omen if omen.verdict == "admitted" else "__hold__")
        real_conf.append(omen.confidence)
        if omen.verdict == "admitted":
            admitted += 1
            if omen.omen == sample["label"]:
                exact += 1
        # 4 -- money. Only a BUY-LOW omen opens a position; the live lane is
        # long-only, so a crest is an abstention, not a short.
        if omen.is_actionable and omen.action == "buy":
            trades.append(sample["forward"] - ROUND_TRIP_COST)
            buys_by_conf.append((omen.confidence, sample["forward"] - ROUND_TRIP_COST))
            # THE GATE, scored on the SAME predictions rather than a second
            # run: the ungated and gated books differ only by the frozen
            # bucket list, so no run-to-run fabric variance can leak into the
            # comparison. (89.2% and 93.6% thirty-four minutes apart is why.)
            if self_gate is not None:
                refused_by = self_gate_refuses(self_gate, sample)
                if refused_by is None:
                    gated_trades.append(sample["forward"] - ROUND_TRIP_COST)
                    gated_is_trough.append(sample["label"] == OMEN_TROUGH)
                else:
                    gate_refusals[refused_by] += 1
            ungated_is_trough.append(sample["label"] == OMEN_TROUGH)

    truth = Counter(s["label"] for s in test_samples)
    majority = max(truth.values()) / max(1, len(test_samples))
    up_rate = sum(1 for s in test_samples if s["forward"] > 0) / max(1, len(test_samples))
    accuracy = exact / max(1, admitted)
    print(f"3. HELD-OUT      : {accuracy:.1%} exact on {admitted} admitted "
          f"of {len(test_samples)}")
    print(f"   baselines     : majority class {majority:.1%}, market up-rate {up_rate:.1%}")
    print(f"   predicted mix : {dict(Counter(predicted))}")

    # 4 -- money, net of cost.
    total = sum(trades)
    wins = sum(1 for t in trades if t > 0)
    hit = wins / max(1, len(trades))
    buy_and_hold = [s["forward"] - ROUND_TRIP_COST for s in test_samples]
    print(f"4. NET OF COST   : {len(trades)} buy omens, {hit:.1%} paid, "
          f"total {total:+.4f} ({total / max(1, len(trades)):+.4%} per trade)")
    print(f"   every-bar buy : {len(buy_and_hold)} trades, "
          f"{sum(buy_and_hold) / max(1, len(buy_and_hold)):+.4%} per trade")
    # The lift over every-bar-buy is the number a reader will quote as an
    # edge, so it does not get printed without its own noise beside it.
    lift = subset_lift_pvalue(buy_and_hold, len(trades),
                              total / max(1, len(trades)))
    verdict = ("beats a random subset of the same bars"
               if lift["p_value"] < 0.05
               else "INSIDE the noise -- not an edge")
    print(f"   selection test: lift "
          f"{total / max(1, len(trades)) - lift['every_bar_mean']:+.4%} per trade, "
          f"null SE {lift['null_se']:.4%}, z {lift['z']:+.2f}, "
          f"p {lift['p_value']:.4f} over {lift['trials']} random {len(trades)}-subsets "
          f"-> {verdict}")
    print(f"   real conf     : {_pct(real_conf)}")

    # Confidence sweep -- what a floor would have done. Reported, never
    # auto-applied: a floor picked to maximise the number it is scored on is
    # not a floor, it is a fit.
    sweep = []
    for floor in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        kept = [pnl for conf, pnl in buys_by_conf if conf >= floor]
        sweep.append({
            "floor": floor, "trades": len(kept),
            "per_trade": (sum(kept) / len(kept)) if kept else 0.0,
            "hit_rate": (sum(1 for p in kept if p > 0) / len(kept)) if kept else 0.0,
        })
    print("   conf sweep    : " + " | ".join(
        f"{s['floor']:.2f}->{s['trades']}t {s['per_trade']:+.3%}" for s in sweep))

    # 5 -- THE SELF-FRAME ABSTENTION GATE. Same fabric, same predictions, one
    # frozen bucket list between the two books.
    gate_result: Dict[str, Any] | None = None
    if self_gate is not None:
        ungated_n = len(trades)
        gated_n = len(gated_trades)
        ungated_per = total / max(1, ungated_n)
        gated_per = sum(gated_trades) / max(1, gated_n)
        ungated_prec = (sum(ungated_is_trough) / max(1, len(ungated_is_trough)))
        gated_prec = (sum(gated_is_trough) / max(1, len(gated_is_trough)))
        refused = ungated_n - gated_n
        print(f"5. SELF GATE     : ungated {ungated_n} buys "
              f"{ungated_per:+.4%}/trade, trough precision {ungated_prec:.1%}")
        print(f"   gated         : {gated_n} buys {gated_per:+.4%}/trade, "
              f"trough precision {gated_prec:.1%}  "
              f"({refused} refused: {dict(gate_refusals)})")
        # An abstention gate can "improve" a per-trade mean by refusing almost
        # everything, and it can only be judged against how much it refused.
        if gated_n == 0:
            note = ("REFUSES EVERY BUY -- that is being switched off, not an "
                    "edge")
        elif gated_n < 30:
            note = (f"UNRANKABLE -- {gated_n} trades after gating is below the "
                    "n=30 floor, a per-trade mean here has no standard error")
        elif refused == 0:
            note = "NO-OP -- the frozen buckets refused nothing in this window"
        else:
            note = (f"{gated_per - ungated_per:+.4%}/trade against the same "
                    f"fabric ungated, keeping {gated_n}/{ungated_n} buys")
        print(f"   verdict       : {note}")
        gate_result = {
            "min_support": self_gate["min_support"],
            "train_samples": self_gate["train_samples"],
            "train_trough_rate": self_gate["train_trough_rate"],
            "refuse_buckets": {k: list(v) for k, v in self_gate["refuse"].items()},
            "refuse_bucket_train_support": {
                key: {frame: self_gate["bucket_support"][key][frame][1]
                      for frame in self_gate["refuse"][key]}
                for key in SELF_GATE_KEYS
            },
            "ungated_trades": ungated_n,
            "gated_trades": gated_n,
            "refused_trades": refused,
            "refused_by_key": dict(gate_refusals),
            "ungated_net_per_trade": ungated_per,
            "gated_net_per_trade": gated_per,
            "delta_net_per_trade": gated_per - ungated_per,
            "ungated_trough_precision": ungated_prec,
            "gated_trough_precision": gated_prec,
            "verdict": note,
        }

    fabric_after = fabric_census(args.endpoint)
    report = {
        # PROVENANCE FIRST. A number whose fabric is unknown is not a
        # measurement, and every reader of this file must see that before the
        # accuracy.
        "node_endpoint": fabric_before.get("endpoint"),
        "node_health": fabric_before.get("health"),
        "fabric_before": fabric_before,
        "fabric_after": fabric_after,
        "fabric_was_clean": fabric_is_empty(fabric_before),
        "allow_warm_fabric": bool(args.allow_warm_fabric),
        "corpus": str(path), "symbol": symbol, "bars": len(bars),
        # All three, always: any two determine the third, and a reader with
        # only the bars cannot compare this run with one on another cadence.
        "bar_seconds": cadence, "horizon_bars": args.horizon,
        "horizon_minutes": horizon_minutes,
        "horizon_source": resolved["horizon_source"],
        "round_trip_cost": ROUND_TRIP_COST, "omen_threshold": threshold,
        "train_window": [train_start, train_stop],
        "test_window": [test_start, test_stop],
        "train_end_pinned": args.train_end is not None,
        "heldout_regime": heldout_regime["regime"],
        "heldout_window_up_rate": heldout_regime["up_rate"],
        "heldout_window_mean_forward": heldout_regime["mean_forward"],
        "heldout_ts_range": [heldout_regime.get("ts_start"),
                             heldout_regime.get("ts_stop")],
        "trained_pairs": brain.trained_pairs, "failed_pairs": brain.failed_pairs,
        "train_seconds": round(train_secs, 1),
        "skipped_training": bool(args.skip_train),
        "collection_distinctness": distinctness,
        "measured_query_collections": list(measured_query),
        # What ACTUALLY fired, not what the default would have been.
        "query_collections": list(query),
        "query_source": "override" if args.query_collections else "measured",
        "regime_source": "stage1_guess" if args.guess_regime else "computed",
        "conflicting_frame_tuples": conflicts,
        "recall_ceiling": 1.0 - conflicts / max(1, len(balanced)),
        "consensus": bool(args.consensus),
        "train_recall": recall, "recall_sample": len(recall_set),
        "recall_answered": answered, "recall_abstained": recall_split,
        "recall_misses": recall_misses,
        "garbage_distinct": len(set(garbage_labels)),
        "garbage_total": len(garbage_labels),
        "garbage_actionable": garbage_actionable,
        "garbage_confidence": sorted(garbage_conf),
        "real_confidence_percentiles": _pct(real_conf),
        "confidence_sweep": sweep,
        "heldout_admitted": admitted, "heldout_total": len(test_samples),
        "heldout_exact_accuracy": accuracy,
        "baseline_majority": majority, "baseline_up_rate": up_rate,
        "predicted_mix": dict(Counter(predicted)),
        "true_mix": dict(truth),
        "buy_omens": len(trades), "buy_hit_rate": hit,
        "buy_net_total": total,
        "buy_net_per_trade": total / max(1, len(trades)),
        "every_bar_net_per_trade": (sum(buy_and_hold) / max(1, len(buy_and_hold))),
        "selection_lift_per_trade": (total / max(1, len(trades))
                                     - lift["every_bar_mean"]),
        "selection_null_se": lift["null_se"],
        "selection_z": lift["z"],
        "selection_p_value": lift["p_value"],
        "selection_trials": lift["trials"],
        "self_gate": gate_result,
    }
    # Refuse to WRITE an uncomparable report rather than discover six passes
    # later that a number cannot be placed against another corpus.
    validate_report_horizon(report)
    report_dir = ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # The regime goes in the FILENAME: two windows measured on one fabric
    # produce two reports, and a reader must not have to open them to see
    # which is the UP one. The MINUTES go in it too -- "h12" was the same
    # string for a 12-minute and a 48-day forecast.
    out = (report_dir /
           f"omen-{symbol}-h{horizon_minutes:.0f}m{args.horizon}b"
           f"-{heldout_regime['regime']}-{stamp}.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
