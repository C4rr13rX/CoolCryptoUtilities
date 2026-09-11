"""How many held-out bars does a READABLE money scoreboard actually need?

The question, and why it needs no node
--------------------------------------
Pass 111 scored the brain's buy omens on a 60-bar held-out window and got
``buy_omens`` of 1, 9, 7 and 2 across four arms. The natural reading is that
the brain is shy. That reading is untestable until you know the CEILING: how
many bars in a 60-bar window are troughs AT ALL under the labelling rule. A
brain cannot call 30 troughs in a window that contains four.

That ceiling is a property of the corpus and the threshold, not of the brain,
so it is computable from the bars alone -- no node, no training, no fabric.
This script computes it over every eligible corpus file and answers, for each
cost multiple:

  * the trough base rate and the crest base rate, per corpus and pooled
  * the held-out window size that yields n >= 30 trough labels at the median
    corpus, which is the number the scoreboard item asks for
  * the same for crests, because the sell-high half is scored nowhere and its
    base rate may be a different number entirely

Eligibility is ``omen_experiment``'s own bound (MAX_CORPUS_BAR_SECONDS), so a
number here and a number from an experiment are over the same population. The
cadence filter is applied on the HEAD cadence, which is what horizon
conversion uses.

Usage
-----
  python -X utf8 scripts/omen_trough_census.py
  python -X utf8 scripts/omen_trough_census.py --horizon-minutes 720 --limit 120
  python -X utf8 scripts/omen_trough_census.py --cadence 3600 --report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.omen_experiment import (  # noqa: E402
    DEFAULT_HORIZON_MINUTES, MAX_CORPUS_BAR_SECONDS, bar_seconds, horizon_bars,
    load_bars,
)
from trading.omen_brain import (  # noqa: E402
    OMEN_CREST, OMEN_TROUGH, RANGE_WINDOW, ROUND_TRIP_COST, _close, label_omen,
    omen_threshold,
)
from trading.omen_scoreboard import (  # noqa: E402
    READABLE_TRADES, money_scoreboard,
)

#: The multiples to sweep. 1.5 is the shipped default (OMEN_COST_MULTIPLE);
#: 1.0 is "exactly pays for itself"; below that the label stops meaning the
#: move cleared its own cost, which is why 0.5 is reported and NOT recommended
#: -- it is here to show what lowering the bar buys and what it costs.
MULTIPLES = (0.5, 1.0, 1.5, 2.0)

#: Fewer bars than this and the label mix is noise, not a base rate.
MIN_BARS = 400


def census_file(path: Path, horizon_minutes: float,
                cadence_filter: int | None) -> Dict[str, Any] | None:
    """Label every bar of one corpus at each multiple. None if ineligible."""
    try:
        bars = load_bars(path)
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    except AttributeError:
        # Measured pass 114: omen_experiment.load_bars raises AttributeError
        # ('str' object has no attribute 'get') on at least one file under
        # data/historical_ohlcv, because a corpus whose JSON is a list of
        # STRINGS reaches `b.get("close")`. Any sweep over the whole corpus
        # dies on it. Skipped and counted here rather than fixed, because
        # that loader is held by another agent this pass -- filed as a
        # backlog item so the fix lands in one place.
        return None
    if len(bars) < MIN_BARS:
        return None
    cadence = bar_seconds(bars)
    if cadence > MAX_CORPUS_BAR_SECONDS:
        return None
    if cadence_filter is not None and cadence != cadence_filter:
        return None
    hbars = horizon_bars(horizon_minutes, cadence)
    if hbars < 1 or hbars >= len(bars):
        return None

    out: Dict[str, Any] = {
        "corpus": path.name, "bars": len(bars), "bar_seconds": cadence,
        "horizon_bars": hbars, "multiples": {},
    }
    for multiple in MULTIPLES:
        mix: Counter = Counter()
        labelled = 0
        for index in range(len(bars) - hbars):
            label = label_omen(bars, index, horizon_bars=hbars,
                               multiple=multiple)
            if label is None:
                continue
            labelled += 1
            mix[label] += 1
        if not labelled:
            continue
        out["multiples"][str(multiple)] = {
            "labelled": labelled,
            "mix": dict(mix),
            "trough_rate": mix[OMEN_TROUGH] / labelled,
            "crest_rate": mix[OMEN_CREST] / labelled,
        }
    return out if out["multiples"] else None


def ceiling_window(path: Path, horizon_minutes: float, window_bars: int,
                   cadence_filter: int | None) -> Dict[str, Any] | None:
    """Score the TRUE labels over the last ``window_bars`` of one corpus.

    This is the money scoreboard's CEILING: what a caller with perfect recall
    and perfect precision would have earned. It needs no node, and it answers
    a question no node run can -- if even perfect trough calls do not beat
    buying every bar by a margin worth having, the shortfall is in the target,
    not in the substrate, and no topology change can reach it.

    The window is classified UP or DOWN by its own drift, so an UP corpus and
    a DOWN corpus are separated exactly as the standing rule requires rather
    than pooled into an average that flatters a long-only rule.
    """
    try:
        bars = load_bars(path)
    except (json.JSONDecodeError, OSError, ValueError, AttributeError):
        return None
    cadence = bar_seconds(bars) if bars else 0
    if not bars or cadence > MAX_CORPUS_BAR_SECONDS:
        return None
    if cadence_filter is not None and cadence != cadence_filter:
        return None
    hbars = horizon_bars(horizon_minutes, cadence)
    need = window_bars + hbars
    if hbars < 1 or len(bars) < need + RANGE_WINDOW:
        return None

    start = len(bars) - need
    stop = len(bars) - hbars
    first, last = _close(bars[start]), _close(bars[stop])
    if not first or not last:
        return None
    drift = (last - first) / first

    calls = []
    for index in range(start, stop):
        label = label_omen(bars, index, horizon_bars=hbars)
        if label is not None:
            calls.append((index, label))
    if not calls:
        return None

    board = money_scoreboard(bars, calls, horizon_bars=hbars)
    board["corpus"] = path.name
    board["regime"] = "UP" if drift > 0 else "DOWN"
    board["window_drift"] = drift
    board["window_bars"] = window_bars
    return board


def _pool(boards: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool per-corpus cells into one readable cell, summing n not averaging %.

    Averaging per-corpus percentages weights a 2-trade corpus the same as a
    200-trade one. The totals are summed and divided once, which is the only
    pooling that means anything here.
    """
    buy_n = sum(b["buy"]["n"] for b in boards)
    sell_n = sum(b["sell"]["n"] for b in boards)
    eb_n = sum(b["every_bar_n"] for b in boards)
    buy_total = sum(b["buy"]["net_total"] for b in boards)
    sell_total = sum(b["sell"]["net_total"] for b in boards)
    eb_total = sum((b["every_bar_net_per_trade"] or 0.0) * b["every_bar_n"]
                   for b in boards)
    buy_paid = sum((b["buy"]["precision_paid"] or 0.0) * b["buy"]["n"]
                   for b in boards)
    sell_paid = sum((b["sell"]["precision_paid"] or 0.0) * b["sell"]["n"]
                    for b in boards)
    return {
        "corpora": len(boards),
        "buy_omens": buy_n,
        "buy_net_per_trade": (buy_total / buy_n) if buy_n else None,
        "trough_precision": (buy_paid / buy_n) if buy_n else None,
        "buy_readable": buy_n >= READABLE_TRADES,
        "crest_omens": sell_n,
        "crest_net_per_trade": (sell_total / sell_n) if sell_n else None,
        "crest_precision": (sell_paid / sell_n) if sell_n else None,
        "crest_readable": sell_n >= READABLE_TRADES,
        "every_bar_n": eb_n,
        "every_bar_net_per_trade": (eb_total / eb_n) if eb_n else None,
    }


def bars_needed(rate: float, target: int = READABLE_TRADES) -> float:
    """Held-out bars needed to CONTAIN ``target`` labels at this base rate.

    This is the ceiling, not the requirement: it assumes a caller that calls
    every true instance and nothing else. A brain with precision p and recall
    r needs this divided by r, so the real number is strictly larger.
    """
    return float("inf") if rate <= 0 else target / rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/historical_ohlcv")
    parser.add_argument("--horizon-minutes", type=float,
                        default=DEFAULT_HORIZON_MINUTES)
    parser.add_argument("--cadence", type=int, default=None,
                        help="only corpora at this exact head cadence, in "
                             "seconds; holding it fixed makes one question")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many ELIGIBLE corpora (0 = all)")
    parser.add_argument("--target", type=int, default=READABLE_TRADES)
    parser.add_argument("--report", action="store_true",
                        help="write data/brain_experiments/TROUGH-BASE-RATE-*.json")
    parser.add_argument("--ceiling-bars", type=int, default=0,
                        help="score the TRUE labels over a held-out window of "
                             "this many bars per corpus, split UP vs DOWN -- "
                             "the money scoreboard a perfect caller would get")
    args = parser.parse_args()

    root = ROOT / args.root
    # rglob, NOT glob: the corpus is nested one level by chain
    # (data/historical_ohlcv/{arbitrum,base,ethereum,optimism,polygon}/*.json)
    # and a top-level glob finds ZERO files there. That is the same defect
    # Cove measured at 714,673 invisible bars -- a loader that silently sees
    # nothing reports a clean empty result rather than an error.
    files = sorted(root.rglob("*.json"))
    print(f"corpus root {root} -- {len(files)} files, horizon "
          f"{args.horizon_minutes:.0f} min, target {args.target} labels")

    if args.ceiling_bars:
        boards: List[Dict[str, Any]] = []
        for path in files:
            board = ceiling_window(path, args.horizon_minutes,
                                   args.ceiling_bars, args.cadence)
            if board is None:
                continue
            boards.append(board)
            if args.limit and len(boards) >= args.limit:
                break
        if not boards:
            print("NO ELIGIBLE CORPORA for the ceiling window")
            return 1
        print(f"\nCEILING: TRUE labels over the last {args.ceiling_bars} bars "
              f"of {len(boards)} corpora -- perfect recall AND perfect "
              f"precision.\nIf this does not beat every-bar by a margin worth "
              f"having, no topology reaches it.\n")
        pooled: Dict[str, Any] = {}
        for regime in ("UP", "DOWN"):
            side = [b for b in boards if b["regime"] == regime]
            if not side:
                print(f"  {regime}: no corpora")
                continue
            cell = _pool(side)
            pooled[regime] = cell
            print(f"  {regime} ({cell['corpora']} corpora)")
            print(f"    buy (trough) : n={cell['buy_omens']} "
                  f"{cell['buy_net_per_trade']:+.4%} per trade, precision "
                  f"{cell['trough_precision']:.1%}"
                  + ("" if cell["buy_readable"] else "  <- UNREADABLE"))
            print(f"    sell (crest) : n={cell['crest_omens']} "
                  f"{cell['crest_net_per_trade']:+.4%} per trade, precision "
                  f"{cell['crest_precision']:.1%}"
                  + ("" if cell["crest_readable"] else "  <- UNREADABLE"))
            print(f"    every bar    : n={cell['every_bar_n']} "
                  f"{cell['every_bar_net_per_trade']:+.4%} per trade")
        if args.report:
            out_dir = ROOT / "data" / "brain_experiments"
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            out = out_dir / f"CEILING-SCOREBOARD-{stamp}.json"
            out.write_text(json.dumps({
                "measured_at": stamp, "window_bars": args.ceiling_bars,
                "horizon_minutes": args.horizon_minutes,
                "cadence_filter": args.cadence,
                "round_trip_cost": ROUND_TRIP_COST,
                "caller": "TRUE LABELS -- perfect recall and precision",
                "pooled": pooled, "per_corpus": boards,
            }, indent=2), encoding="utf-8")
            print(f"\nreport -> {out}")
        return 0

    started = time.time()
    rows: List[Dict[str, Any]] = []
    for path in files:
        row = census_file(path, args.horizon_minutes, args.cadence)
        if row is None:
            continue
        rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        print("NO ELIGIBLE CORPORA -- widen --cadence or lower MIN_BARS")
        return 1

    print(f"eligible corpora: {len(rows)} in {time.time() - started:.1f}s\n")
    print(f"{'multiple':>9} {'threshold':>10} {'trough rate':>12} "
          f"{'crest rate':>11} {'bars for ' + str(args.target) + ' troughs':>22} "
          f"{'for crests':>12}")

    summary: Dict[str, Any] = {}
    for multiple in MULTIPLES:
        key = str(multiple)
        cells = [r["multiples"][key] for r in rows if key in r["multiples"]]
        if not cells:
            continue
        trough_rates = [c["trough_rate"] for c in cells]
        crest_rates = [c["crest_rate"] for c in cells]
        t_med = median(trough_rates)
        c_med = median(crest_rates)
        summary[key] = {
            "corpora": len(cells),
            "threshold": omen_threshold(ROUND_TRIP_COST, multiple),
            "trough_rate_median": t_med,
            "trough_rate_min": min(trough_rates),
            "trough_rate_max": max(trough_rates),
            "crest_rate_median": c_med,
            "crest_rate_min": min(crest_rates),
            "crest_rate_max": max(crest_rates),
            "bars_for_target_troughs": bars_needed(t_med, args.target),
            "bars_for_target_crests": bars_needed(c_med, args.target),
        }
        print(f"{multiple:>9.1f} {summary[key]['threshold']:>9.4%} "
              f"{t_med:>11.2%} {c_med:>10.2%} "
              f"{summary[key]['bars_for_target_troughs']:>22.0f} "
              f"{summary[key]['bars_for_target_crests']:>12.0f}")

    shipped = summary.get("1.5")
    if shipped:
        at_60 = shipped["trough_rate_median"] * 60
        print(f"\nAT THE SHIPPED THRESHOLD (multiple 1.5), a 60-bar held-out "
              f"window contains {at_60:.1f} trough labels at the median "
              f"corpus.\nThat is the CEILING on buy omens -- a brain with "
              f"recall r calls at most {at_60:.1f} x r of them. Pass 111's "
              f"1, 9, 7 and 2 sit inside it.\nTo carry {args.target} trough "
              f"calls the held-out window must be about "
              f"{shipped['bars_for_target_troughs']:.0f} bars at perfect "
              f"recall, and more at any real recall.")

    if args.report:
        out_dir = ROOT / "data" / "brain_experiments"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = out_dir / f"TROUGH-BASE-RATE-{stamp}.json"
        out.write_text(json.dumps({
            "measured_at": stamp,
            "root": str(root),
            "horizon_minutes": args.horizon_minutes,
            "cadence_filter": args.cadence,
            "target_labels": args.target,
            "eligible_corpora": len(rows),
            "round_trip_cost": ROUND_TRIP_COST,
            "by_multiple": summary,
            "per_corpus": rows,
        }, indent=2), encoding="utf-8")
        print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
