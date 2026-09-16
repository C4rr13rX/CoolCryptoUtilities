"""Per-strategy census of the stale_ghost_book refusal in _ghost_validation.

WHY
---
``scripts/live_gate_map.py`` reported (2026-09-11) that ``stale_ghost_book`` is
the reason 13 of 38 strategies are refused by ``_ghost_validation`` -- more than
``negative_margin`` or any other. Read literally that says "this strategy has
not CLOSED anything recently", which is a very different statement from "this
strategy loses money", and it points the fix at the exit schedule rather than at
the edge.

Before acting on that reading it has to be checked, because the predicate is:

    trading/pipeline.py:4869
    stale_samples = stale_guard > 0 and (not trades or last_trade_age > stale_guard)

That ``not trades`` disjunct means a strategy with an EMPTY book inside the gate
window reports the same reason as one whose book is merely old. And
``stale_ghost_book`` is ranked FIRST in the reason chain (pipeline.py:5288),
above ``insufficient_samples``, so the empty case can never surface under its
own name. This script separates the two and counts each.

It changes nothing. Every guardrail is read from the environment exactly as the
gate reads it; the counterfactual pass is computed by calling the gate's own
``_ghost_validation`` a second time with GHOST_MAX_STALE_SEC=0 IN PROCESS and
restoring the previous value in a finally, which is the gate's own documented
"guard off" setting (``stale_guard > 0`` is the enable test) and is never
written to .env or to any config.

    python scripts/ghost_stale_census.py

Written by Sage, pass 425, for backlog item dbbfea96.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from services.env_loader import EnvLoader

    EnvLoader.load()
except Exception:  # noqa: BLE001 - a diagnostic must still print without a vault
    pass


def _flt(v, d):
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return d


def main() -> int:
    # Same construction live_gate_map.print_live uses, including the news stub:
    # HistoricalDataLoader._load_news blocks for minutes and this asks nothing
    # about news.
    import trading.data_loader as dl

    dl.HistoricalDataLoader._load_news = lambda self: []
    from trading.pipeline import TrainingPipeline
    from trading.metrics import MetricsCollector
    from db import get_db

    p = TrainingPipeline(db=get_db())
    # The gate builds its own collector the first time it is asked; do the same
    # so the snapshot below reads the identical store.
    try:
        p._ghost_validation()
    except Exception as exc:  # noqa: BLE001
        print(f"pooled _ghost_validation raised: {type(exc).__name__}: {exc}")
    metrics = getattr(p, "metrics", None)
    if metrics is None:
        metrics = MetricsCollector(getattr(p, "db", None))

    lookback = int(os.getenv("GHOST_VALIDATION_LOOKBACK_SEC",
                             str(getattr(p, "focus_lookback_sec", 172800))))
    stale_guard = _flt(os.getenv("GHOST_MAX_STALE_SEC", str(min(lookback, 86400))),
                       86400.0)

    print("=" * 96)
    print("stale_ghost_book census -- per strategy, against the gate's own guardrails")
    print("=" * 96)
    print(f"  GHOST_VALIDATION_LOOKBACK_SEC = {lookback} ({lookback/3600.0:.1f}h)"
          "   <- trades outside this are not loaded at all")
    print(f"  GHOST_MAX_STALE_SEC           = {stale_guard:.0f} "
          f"({stale_guard/3600.0:.1f}h)   <- newest exit older than this is 'stale'")
    print()

    # The population: every strategy the ghost book has ever attributed an exit
    # to. Taken from tradeable_evidence so it is the same roster live_gate_map
    # prints, not a roster invented here.
    try:
        from services.tradeable_evidence import reconstruct

        evidence = reconstruct()
        sids = sorted({str(s).strip() for s in evidence if str(s or "").strip()})
    except Exception as exc:  # noqa: BLE001
        print(f"reconstruct() failed: {type(exc).__name__}: {exc}")
        return 2

    now = time.time()
    rows = []
    for sid in sids:
        try:
            gate = p._ghost_validation(strategy_id=sid)
        except Exception as exc:  # noqa: BLE001
            rows.append({"sid": sid, "err": f"{type(exc).__name__}: {exc}"})
            continue
        # The trades the gate itself judged, re-read the same way it reads them.
        try:
            trades = metrics.ghost_trade_snapshot(
                limit=500, lookback_sec=lookback, strategy_id=sid)
        except Exception:  # noqa: BLE001
            trades = []
        last_ts = max((float(getattr(t, "exit_ts", 0.0)) for t in trades), default=0.0)
        age = (now - last_ts) if last_ts > 0 else None

        # The SAME question with no lookback at all: when did this strategy last
        # close anything, ever? This is what separates "closed nothing recently"
        # from "closed nothing inside the window we look at".
        try:
            allt = metrics.ghost_trade_snapshot(
                limit=500, lookback_sec=10 ** 9, strategy_id=sid)
        except Exception:  # noqa: BLE001
            allt = []
        last_ever = max((float(getattr(t, "exit_ts", 0.0)) for t in allt), default=0.0)
        age_ever = (now - last_ever) if last_ever > 0 else None

        # Counterfactual: the same gate with the staleness guard OFF, which is
        # what GHOST_MAX_STALE_SEC=0 means to the gate's own enable test. This
        # is how we learn whether staleness is the ONLY refusal or the first of
        # several. Restored unconditionally.
        prev = os.environ.get("GHOST_MAX_STALE_SEC")
        os.environ["GHOST_MAX_STALE_SEC"] = "0"
        try:
            gate_nostale = p._ghost_validation(strategy_id=sid)
        except Exception as exc:  # noqa: BLE001
            gate_nostale = {"ready": False, "reason": f"raised:{type(exc).__name__}"}
        finally:
            if prev is None:
                os.environ.pop("GHOST_MAX_STALE_SEC", None)
            else:
                os.environ["GHOST_MAX_STALE_SEC"] = prev

        rows.append({
            "sid": sid,
            "n": len(trades),
            "n_ever": len(allt),
            "age": age,
            "age_ever": age_ever,
            "reason": str(gate.get("reason") or ""),
            "ready": bool(gate.get("ready")),
            "stale_flag": bool(gate.get("stale")),
            "reason_nostale": str(gate_nostale.get("reason") or ""),
            "ready_nostale": bool(gate_nostale.get("ready")),
        })

    def _age(v):
        if v is None:
            return "never"
        return f"{v/3600.0:.1f}h"

    print(f"{'strategy':<30} {'n_win':>5} {'n_all':>5} {'newest_in_win':>13} "
          f"{'newest_ever':>11}  {'reason':<22} {'reason_if_stale_off':<22}")
    print("-" * 122)
    for r in sorted(rows, key=lambda x: (x.get("reason") or "", x["sid"])):
        if r.get("err"):
            print(f"{r['sid'][:30]:<30} ERROR {r['err']}")
            continue
        print(f"{r['sid'][:30]:<30} {r['n']:>5} {r['n_ever']:>5} "
              f"{_age(r['age']):>13} {_age(r['age_ever']):>11}  "
              f"{r['reason'][:22]:<22} {r['reason_nostale'][:22]:<22}")

    good = [r for r in rows if not r.get("err")]
    stale = [r for r in good if r["reason"] == "stale_ghost_book"]
    empty_in_window = [r for r in stale if r["n"] == 0]
    old_book = [r for r in stale if r["n"] > 0]
    never_closed = [r for r in stale if r["n_ever"] == 0]
    only_stale = [r for r in stale if r["ready_nostale"]]

    print()
    print("=" * 96)
    print("WHAT THE NUMBER MEANS")
    print("=" * 96)
    print(f"  strategies measured                         : {len(good)}")
    print(f"  refused with reason=stale_ghost_book        : {len(stale)}")
    print(f"    ... of which have ZERO trades in the      : {len(empty_in_window)}"
          "   <- 'not trades' disjunct, pipeline.py:4869")
    print(f"        {lookback/3600.0:.0f}h gate window (an EMPTY book, not a stale one)")
    print(f"    ... of which have a book but the newest    : {len(old_book)}"
          "   <- a genuinely STALE book")
    print(f"        exit is older than the stale guard")
    print(f"    ... of which have NEVER closed a ghost     : {len(never_closed)}"
          "   <- no exit at any time, any window")
    print(f"        trade at any point in the store")
    print()
    print(f"  refused ONLY by staleness (pass with the    : {len(only_stale)}")
    print(f"    guard off, everything else unchanged)")
    if only_stale:
        for r in only_stale:
            print(f"      {r['sid']}  n={r['n']} newest={_age(r['age'])}")
    else:
        print("      none -- every stale strategy is refused by something else too,")
        print("      so removing or widening the stale guard opens NOTHING.")
    print()
    print("  next reasons once staleness is set aside:")
    from collections import Counter

    for reason, c in Counter(r["reason_nostale"] for r in stale).most_common():
        print(f"      {reason or '(ready)':<26} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
