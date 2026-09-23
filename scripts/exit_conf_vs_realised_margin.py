"""Does ``exit_conf`` predict the margin a trade ACTUALLY realised?

WHY THIS EXISTS. ``exit_conf`` is the ``confidence`` conjunct of the entry test
(``trading/scheduler.py:778`` reads ``pred_summary["exit_conf"]``, gated by
``SCHEDULER_MIN_CONFIDENCE``, default 0.6). Pass 467 measured it over 1818
snapshots in 6h and it cleared that floor **0 times**: min 0.3359, p50 0.4991,
max 0.5436. The entry test is an AND, so every other conjunct was decorative.

Its training label is deterministic --  ``trading/data_loader.py:1222``::

    exit_conf = 1.0 / (1.0 + np.exp(-np.abs(net_margin) * 10.0))

which is ``sigmoid(|net_margin| * 10)``. Because ``|net_margin| >= 0`` that label
is **bounded below by 0.5** and is ``+1.0`` correlated with ``|net_margin|`` BY
CONSTRUCTION. The served head put 1050 of 1779 values *below* 0.5 -- a region
the label cannot occupy -- and correlated ``+0.7653`` with the signed margin
against ``-0.8296`` with the absolute one.

WHAT PASS 467 COULD NOT CONCLUDE, AND WHY THIS SCRIPT IS THE NEXT STEP. Those
correlations are HEAD-AGAINST-HEAD: ``net_margin`` is itself a model output
(``model_definition.py:399``), not the training target. So they establish that
the heads CONTRADICT EACH OTHER -- not that ``exit_conf`` is inverted against
ground truth. Only a realised outcome can settle that, and that is what this
reads.

HOW TO READ THE RESULT. Three outcomes, all informative, none of them a licence
to touch the floor:

  * corr POSITIVE and strong -> ``exit_conf`` is doing its job and the
    NET_MARGIN head is the broken one. The gate is being refused by a good head
    sitting next to a bad one.
  * corr NEAR ZERO -> ``exit_conf`` carries no information, and the floor is
    refusing noise. Cheapest outcome to act on.
  * corr NEGATIVE -> genuinely inverted, and the entry funnel selects the
    smallest margins on purpose.

DO NOT LOWER ``SCHEDULER_MIN_CONFIDENCE`` UNDER ANY OF THEM. 0.6 is correct
against a label whose median is 0.7630 on live margins. A floor lowered to admit
this input deletes the only check that noticed the input was broken, and because
the head is anti-correlated with |margin| the admitted set would be the
SMALLEST-margin ticks. Every branch above is fixed upstream of the floor.

THE GHOST/LIVE SPLIT IS NOT OPTIONAL. 87% of ``trade_fills`` rows are ghost and
read as PERFECT fills, which drags any cost or margin statistic toward zero.
This script reports every ``status`` bucket separately and pools them only as a
clearly-labelled ALL row, so a pooled number can never be quoted on its own by
accident.

MATCHING, and its honest limitation. Each outcome is matched to the most recent
snapshot for the SAME symbol at or before the outcome's ``ts``, within
``--max-stale-sec``. ``trade_outcomes.ts`` is the row's timestamp, not
necessarily the ENTRY time, so a matched snapshot may sit closer to the exit
than the entry. That biases toward WEAKER correlation, so a strong result
survives the caveat and a null one does not settle anything on its own. Tighten
``--max-stale-sec`` to trade sample size for match quality.

USAGE::

    python -X utf8 scripts/exit_conf_vs_realised_margin.py
    python -X utf8 scripts/exit_conf_vs_realised_margin.py --max-stale-sec 900

Tracked on backlog item 946b2c53.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
import os
import sqlite3
import statistics
import time
from typing import Dict, List, Sequence, Tuple

DEFAULT_DB = os.getenv("TRADING_DB_PATH", "storage/trading_cache.db")


_BANNER_PRINTED = False


def _plausibility_banner(rows: Sequence[Sequence]) -> List[str]:
    """Name how many of ``rows`` the ledger would refuse, without dropping them.

    ``trade_outcomes`` is a PRE-GUARD table: ``ledger._is_implausible`` refuses
    a row at ``record()`` time and the row is written anyway, so every census
    reading the table directly reads a dirtier book than graduation does.
    Import failure is reported, never swallowed -- a silently absent banner is
    indistinguishable from a clean book.
    """
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from services import outcome_plausibility as op
    except Exception as exc:  # pragma: no cover - import guard
        return ["!! UNCENSORED and UNMEASURED: services.outcome_plausibility "
                "would not import (%s), so the contamination below is not "
                "even sized." % exc]
    judged = []
    for r in rows:
        blob = {}
        try:
            blob = json.loads(r[7]) if r[7] else {}
        except Exception:
            blob = {}
        judged.append({"entry_price": r[2], "exit_price": r[3],
                       "gross_profit": r[5], "net_profit": r[6],
                       "strategy_id": blob.get("strategy_id", "")})
    return op.banner(judged, censored=False)

#: Below this a correlation is an anecdote. Buckets under it are reported as
#: UNDERPOWERED rather than given a number somebody will quote.
MIN_N = 8


def _corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson r, or nan when either series is constant."""
    if len(x) < 2:
        return float("nan")
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    )
    return num / den if den else float("nan")


def build_snapshot_index(
    conn: sqlite3.Connection, *, since: float = 0.0
) -> Dict[str, List[Tuple[float, float]]]:
    """``symbol -> [(ts, exit_conf)]``, ascending, for bisect matching.

    Snapshots are scanned once. The symbol lives inside the payload and has been
    written under three different keys over this store's life, so all three are
    tried rather than assuming the current one.
    """
    idx: Dict[str, List[Tuple[float, float]]] = collections.defaultdict(list)
    cur = conn.execute(
        "SELECT ts, payload FROM organism_snapshots WHERE ts > ? ORDER BY ts ASC",
        (float(since),),
    )
    for ts, payload in cur:
        try:
            blob = json.loads(payload) or {}
        except (TypeError, ValueError):
            continue
        pred = blob.get("prediction") or {}
        conf = pred.get("exit_conf")
        if conf is None:
            continue
        symbol = (
            blob.get("symbol")
            or (blob.get("state") or {}).get("symbol")
            or pred.get("symbol")
        )
        if not symbol:
            continue
        try:
            idx[str(symbol)].append((float(ts), float(conf)))
        except (TypeError, ValueError):
            continue
    return idx


def match_outcomes(
    conn: sqlite3.Connection,
    idx: Dict[str, List[Tuple[float, float]]],
    *,
    max_stale_sec: float,
) -> Tuple[Dict[str, List[Tuple[float, float]]], int, int]:
    """``status -> [(exit_conf, realised_return)]`` plus miss counters.

    ``realised_return`` is ``(exit_price - entry_price) / entry_price``: the
    GROSS move the trade actually got. Gross is deliberate -- the question here
    is whether the head saw the MOVE, and netting it would fold this repo's
    unresolved cost arithmetic into a signal measurement.
    """
    rows = conn.execute(
        "SELECT symbol, ts, entry_price, exit_price, status, gross_profit, "
        "net_profit, details "
        "FROM trade_outcomes WHERE entry_price > 0 AND exit_price > 0"
    ).fetchall()

    # UNCENSORED BY DESIGN, and the banner says by how much. This report asks
    # whether the exit_conf head SAW THE MOVE, so a row the ledger refuses for
    # an implausible booked return is still a row the head scored -- dropping
    # it would delete evidence about the head. Printing the count is what keeps
    # that a choice rather than an oversight.
    # main() prints this banner BEFORE the snapshot index so a killed run keeps
    # it; it is re-emitted here only when match_outcomes is called directly.
    if not _BANNER_PRINTED:
        for line in _plausibility_banner(rows):
            print(line)

    paired: Dict[str, List[Tuple[float, float]]] = collections.defaultdict(list)
    no_symbol = 0
    too_stale = 0
    for symbol, ts, entry_price, exit_price, status, _g, _n, _d in rows:
        series = idx.get(str(symbol))
        if not series:
            no_symbol += 1
            continue
        stamps = [t for t, _ in series]
        i = bisect.bisect_right(stamps, float(ts)) - 1
        if i < 0 or (float(ts) - stamps[i]) > float(max_stale_sec):
            too_stale += 1
            continue
        realised = (float(exit_price) - float(entry_price)) / float(entry_price)
        paired[str(status or "(none)")].append((series[i][1], realised))
    return paired, no_symbol, too_stale


def render(paired: Dict[str, List[Tuple[float, float]]]) -> None:
    pooled = [p for bucket in paired.values() for p in bucket]
    buckets = sorted(paired.items(), key=lambda kv: -len(kv[1]))
    buckets.append(("ALL (pooled -- do not quote alone)", pooled))

    print()
    print("  %-38s %6s  %22s  %22s"
          % ("status bucket", "n", "corr(conf, realised)", "corr(conf,|realised|)"))
    for name, bucket in buckets:
        if len(bucket) < MIN_N:
            print("  %-38s %6d  UNDERPOWERED (n < %d)" % (name, len(bucket), MIN_N))
            continue
        conf = [c for c, _ in bucket]
        real = [r for _, r in bucket]
        print("  %-38s %6d  %+22.4f  %+22.4f"
              % (name, len(bucket), _corr(conf, real),
                 _corr(conf, [abs(r) for r in real])))

    print()
    if len(pooled) < MIN_N:
        print("  VERDICT: not enough matched pairs to say anything. Raise "
              "--max-stale-sec or widen the outcome window, and do NOT report "
              "a number from this run.")
        return
    conf = [c for c, _ in pooled]
    r = _corr(conf, [abs(x) for _, x in pooled])
    print("  The LABEL is +1.0 against |realised| by construction "
          "(sigmoid(|net_margin|*10), data_loader.py:1222).")
    if r > 0.2:
        print("  VERDICT: POSITIVE (%+.4f). exit_conf tracks the realised move. "
              "Suspect the NET_MARGIN head, not this one." % r)
    elif r < -0.2:
        print("  VERDICT: NEGATIVE (%+.4f). exit_conf is inverted against ground "
              "truth, so the floor selects the SMALLEST moves." % r)
    else:
        print("  VERDICT: NEAR ZERO (%+.4f). exit_conf carries no information "
              "about the realised move; the floor is refusing noise." % r)
    print("  In every case the fix is UPSTREAM of SCHEDULER_MIN_CONFIDENCE. "
          "0.6 is correct against the label. Do not lower it.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--max-stale-sec", type=float, default=3600.0,
                    help="how close a snapshot must sit to the outcome to count")
    ap.add_argument("--since-days", type=float, default=0.0,
                    help="only index snapshots newer than this; 0 means all")
    args = ap.parse_args()

    since = (time.time() - args.since_days * 86400.0) if args.since_days else 0.0
    conn = sqlite3.connect(args.db)
    try:
        # THE BANNER RUNS FIRST, BEFORE THE EXPENSIVE INDEX, and that ordering
        # is the whole point. build_snapshot_index() below scans a ~22GB store
        # and can take many minutes; stdout is block-buffered when redirected,
        # so a run killed during the index loses EVERY buffered line and reads
        # as "silent, exit 0". Printing the censoring statement before the
        # index means the one thing a reader must not miss survives a kill.
        # Its own query is 265 rows and costs nothing.
        for line in _plausibility_banner(conn.execute(
                "SELECT symbol, ts, entry_price, exit_price, status, "
                "gross_profit, net_profit, details FROM trade_outcomes "
                "WHERE entry_price > 0 AND exit_price > 0").fetchall()):
            print(line, flush=True)
        globals()['_BANNER_PRINTED'] = True

        idx = build_snapshot_index(conn, since=since)
        total = sum(len(v) for v in idx.values())
        print("symbols indexed: %d   snapshots carrying exit_conf AND a symbol: %d"
              % (len(idx), total))
        paired, no_symbol, too_stale = match_outcomes(
            conn, idx, max_stale_sec=args.max_stale_sec)
        matched = sum(len(v) for v in paired.values())
        print("outcomes matched: %d   unmatched (symbol never snapshotted): %d   "
              "dropped (no snapshot within %.0fs): %d"
              % (matched, no_symbol, too_stale, args.max_stale_sec))
        render(paired)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
