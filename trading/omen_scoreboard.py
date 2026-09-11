"""The money scoreboard: per-trade net that CANNOT be quoted without its n.

Why this file exists
--------------------
Measured 2026-09-10 from the four pass-111 artifacts in data/brain_experiments
(omen-AERO-USDC-h12-{UP,DOWN}-...json): ``buy_omens`` across the four arms was
1, 9, 7 and 2 out of 60 admitted held-out bars. The UP-base cell read
``buy_net_per_trade`` +0.0031 at ``buy_hit_rate`` 1.0 -- against an every-bar
+0.00034 -- and it was the only positive cell in the table, so it is the one
that gets quoted. It is ONE TRADE. A per-trade mean on one trade has no
standard error; it is one number wearing a percentage sign.

The failure is not arithmetic, it is presentation: ``buy_net_per_trade`` and
``buy_omens`` are two separate keys in a 50-key report, and a reader quoting
the first never sees the second. So this module makes them ONE object that
renders as one string, and refuses to render a per-trade net without the n
that produced it. A cell below the readability floor renders as UNREADABLE
with its n, never as a percentage.

It also scores the half that is measured nowhere. Every omen report in this
repo scores BUYING (trough calls). The operator's 2026-09-10 18:35 note is
that the sell-high half has never been scored at all: a crest call that
correctly predicts a fall is worth exactly as much as a trough call that
predicts a rise, and a brain that is good at one and useless at the other is
a different instrument from one that is mediocre at both. ``crest_precision``
here is scored against forward returns on the same bars, same cost, same
horizon, so the two halves are comparable numbers rather than one number and
an absence.

The readability floor
---------------------
``READABLE_TRADES`` is 30 and it is a floor on being QUOTED, not a claim that
30 is enough to detect a 1pp edge -- at this feed's dispersion (per-trade
absolute returns run 2-3%) detecting 1.0pp needs ~63 trades per arm at sd=2%
and ~141 at sd=3%. 30 is the point below which the mean is visibly driven by
individual trades. Anything under it is reported with its n and the word
UNREADABLE; anything over it is still reported with its n.

Nothing here queries a node. It takes bars and the calls somebody else made,
so the same function scores a brain's predictions, a baseline's, or the true
labels' ceiling.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from trading.omen_brain import (
    COST_MULTIPLE, OMEN_CREST, OMEN_TROUGH, ROUND_TRIP_COST, omen_threshold,
)

#: Below this many trades a per-trade mean is driven by individual trades and
#: must not be quoted as an edge. See the module docstring for why it is a
#: quoting floor rather than a power calculation.
READABLE_TRADES = 30


def _close(bar: Mapping[str, Any]) -> Optional[float]:
    try:
        return float(bar["close"])
    except (KeyError, TypeError, ValueError):
        return None


def forward_return(bars: Sequence[Mapping[str, Any]], index: int,
                   horizon_bars: int) -> Optional[float]:
    """Close-to-close forward return, or None when the future is off the end.

    None is NOT zero. A bar whose future is not in the corpus has no outcome
    and must be dropped by the caller, never scored as a flat trade -- padding
    the tail with zeros is how a losing tail becomes a break-even one.
    """
    future = index + horizon_bars
    if index < 0 or future >= len(bars):
        return None
    entry = _close(bars[index])
    exit_ = _close(bars[future])
    if entry is None or exit_ is None or entry == 0:
        return None
    return (exit_ - entry) / entry


def _cell_from_totals(n: int, net_total: float, paid: int) -> Dict[str, Any]:
    """One side of the book from its TOTALS: n, total, per-trade, precision.

    Split out from ``_cell`` so that pooling many corpora into one cell goes
    through exactly the same readability rule as scoring one corpus. A pooled
    cell that computed its own ``readable`` would be free to disagree with a
    per-corpus one, and the disagreement would be invisible in the report.
    """
    return {
        "n": n,
        "paid": paid,
        "net_total": net_total,
        # max(1, n) would report 0.0000% on an empty cell, which reads as a
        # measured break-even rather than as nothing measured. None reads as
        # what it is.
        "net_per_trade": (net_total / n) if n else None,
        "precision_paid": (paid / n) if n else None,
        "readable": n >= READABLE_TRADES,
    }


def _cell(nets: Sequence[float], paid: int) -> Dict[str, Any]:
    """One side of the book: n, total, per-trade, precision, readability."""
    return _cell_from_totals(len(nets), sum(nets), paid)


def sell_every_bar(baseline: Optional[float],
                   cost: float) -> Optional[float]:
    """What SELLING every bar pays, from what BUYING every bar pays.

    This is not ``-baseline``, and the difference is two round trips.
    ``baseline`` is ``mean(forward) - cost``; selling every bar is
    ``-mean(forward) - cost``, so the mirror is ``-baseline - 2*cost``.
    Negating the buy baseline CREDITS the round trip to the seller instead of
    charging it, which flatters every sell cell by 2x the cost -- 1.30pp at
    the shipped 0.6500% -- and this file shipped exactly that mirror on its
    sell line until pass 118 measured an edge of exactly -2*cost on a
    synthetic window where the true edge is zero.
    """
    if baseline is None:
        return None
    return -baseline - 2.0 * abs(cost)


def _edge(cell: Mapping[str, Any],
          baseline: Optional[float]) -> Optional[float]:
    """The cell's per-trade net MINUS the baseline it has to beat.

    Measured pass 118 over 167 held-out corpora: selling every bar in a DOWN
    window pays +1.1246% per trade with no skill whatsoever, so a crest cell
    reading +1.2379% is not a +1.24% edge -- it is +0.1133pp, an ELEVENTH of
    what the level says. Buying every bar in an UP window pays +1.2206% for
    the same reason, so a buy cell of +1.1258% is a NEGATIVE edge. A raw
    per-trade percentage in a directional window is mostly the window, and
    every "headroom" claim this repo has made about the sell half came from
    reading the level as if it were the edge.

    The caller passes the baseline this half actually competes with: buying
    every bar for the buy half, ``sell_every_bar(...)`` for the sell half.
    ``None`` when there is no cell or no baseline to compare it against --
    an absent comparison must not read as a zero edge.
    """
    net = cell.get("net_per_trade")
    if net is None or baseline is None:
        return None
    return net - baseline


def money_scoreboard(
    bars: Sequence[Mapping[str, Any]],
    calls: Iterable[Tuple[int, str]],
    *,
    horizon_bars: int,
    cost: float = ROUND_TRIP_COST,
    multiple: float = COST_MULTIPLE,
    horizon_minutes: Optional[float] = None,
    bar_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Score BOTH halves of the book against forward returns.

    ``calls`` is (bar_index, predicted_omen) pairs -- whatever the caller
    decided, whether that came from a node, a rule or the true labels. Bars
    whose future is off the end of the corpus are dropped and counted, because
    a dropped bar and a flat bar are different things.

    The buy half charges the full round trip to the trade, which is what a
    long entry actually pays. The sell half is scored as the AVOIDED move:
    a crest call is right when the forward return is below minus the cost, so
    it is the mirror of the buy test at the same threshold and the two
    precisions are directly comparable.
    """
    threshold = omen_threshold(cost, multiple)
    buy_nets: List[float] = []
    buy_paid = 0
    sell_nets: List[float] = []
    sell_paid = 0
    every_bar: List[float] = []
    dropped = 0
    scored_indices: List[int] = []

    for index, label in calls:
        forward = forward_return(bars, index, horizon_bars)
        if forward is None:
            dropped += 1
            continue
        scored_indices.append(index)
        every_bar.append(forward - cost)
        if label == OMEN_TROUGH:
            buy_nets.append(forward - cost)
            if forward > cost:
                buy_paid += 1
        elif label == OMEN_CREST:
            # Selling high pays when the price then falls by more than the
            # round trip: the holder who exits keeps a move the holder who
            # stayed gives back. Signed so that positive is money kept.
            sell_nets.append(-forward - cost)
            if -forward > cost:
                sell_paid += 1

    buy = _cell(buy_nets, buy_paid)
    sell = _cell(sell_nets, sell_paid)
    baseline = (sum(every_bar) / len(every_bar)) if every_bar else None
    return {
        "horizon_bars": horizon_bars,
        # The wall-clock horizon, taken from the caller and never guessed --
        # 12 bars is 12 minutes on one corpus and 12 hours on another, so
        # bars alone cannot say whether two boards asked the same question.
        "horizon_minutes": horizon_minutes,
        "bar_seconds": bar_seconds,
        "round_trip_cost": cost,
        "omen_threshold": threshold,
        "readable_trades_floor": READABLE_TRADES,
        "scored_bars": len(scored_indices),
        "dropped_no_future": dropped,
        "buy": buy,
        "sell": sell,
        # The honest baseline, on exactly the bars that were scored.
        "every_bar_n": len(every_bar),
        "every_bar_net_per_trade": baseline,
        # The EDGE, which is the only number in here that is a statement
        # about the caller rather than about the window it was run in.
        "sell_every_bar_net_per_trade": sell_every_bar(baseline, cost),
        "buy_edge_vs_baseline": _edge(buy, baseline),
        "crest_edge_vs_baseline": _edge(sell, sell_every_bar(baseline, cost)),
        # Kept under the old key names so a reader of an existing report finds
        # the same numbers, but a reader of THIS dict cannot find the net
        # without walking past the n that sits beside it.
        "buy_omens": buy["n"],
        "buy_net_per_trade": buy["net_per_trade"],
        "trough_precision": buy["precision_paid"],
        "crest_omens": sell["n"],
        "crest_net_per_trade": sell["net_per_trade"],
        "crest_precision": sell["precision_paid"],
        "readable": buy["readable"] and sell["readable"],
    }


def pool_scoreboards(boards: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Add up many per-corpus scoreboards into ONE board of the same shape.

    Why this is here rather than in the experiment script
    -----------------------------------------------------
    A single 208-bar held-out window cannot carry 30 trades on BOTH halves,
    so the only way to read the sell half at all is to pool many corpora.
    Pooling is where the arithmetic quietly goes wrong: averaging per-corpus
    ``net_per_trade`` values weights a 2-trade corpus the same as a 90-trade
    one, and that is how a two-trade outlier becomes a headline. This pools
    the TOTALS and divides once, and it runs the pooled n through the same
    ``READABLE_TRADES`` floor as a single corpus, so a pooled cell and a
    per-corpus cell cannot disagree about whether they may be quoted.

    A board with a different ``horizon_bars`` or ``round_trip_cost`` is not
    poolable with the others -- those are two different games priced at two
    different costs -- so this raises rather than silently mixing them. The
    horizon may legitimately differ in BARS across corpora of different
    cadence; a caller pooling across cadences must pool in minutes, which is
    why the mismatch is an error the caller has to answer rather than a
    warning it can skip.
    """
    boards = list(boards)
    if not boards:
        raise ValueError("nothing to pool: pool_scoreboards needs >= 1 board")
    horizons = {b["horizon_bars"] for b in boards}
    costs = {round(float(b["round_trip_cost"]), 12) for b in boards}
    minutes = {b.get("horizon_minutes") for b in boards}
    if len(costs) != 1:
        raise ValueError(
            f"refusing to pool boards priced at different costs: "
            f"{sorted(costs)}")
    # Bars are not the question; MINUTES are. A 3600s corpus asks 720 minutes
    # in 12 bars and a 300s corpus asks the same 720 minutes in 144, and those
    # two boards ARE poolable. Two boards that disagree in minutes are not,
    # whatever their bar counts say. Boards with no minutes recorded fall back
    # to the strict bar test, because an unknown horizon cannot be checked.
    if None in minutes:
        if len(horizons) != 1:
            raise ValueError(
                f"refusing to pool boards scored at different bar-horizons "
                f"{sorted(horizons)} with no horizon_minutes to compare them "
                f"by: pass horizon_minutes to money_scoreboard")
    elif len(minutes) != 1:
        raise ValueError(
            f"refusing to pool boards scored at different wall-clock "
            f"horizons: horizon_minutes {sorted(minutes)}")

    def _sum(side: str, key: str) -> float:
        return sum(b[side][key] for b in boards)

    buy = _cell_from_totals(int(_sum("buy", "n")), _sum("buy", "net_total"),
                            int(_sum("buy", "paid")))
    sell = _cell_from_totals(int(_sum("sell", "n")), _sum("sell", "net_total"),
                             int(_sum("sell", "paid")))
    every_n = sum(int(b["every_bar_n"]) for b in boards)
    every_total = sum((b["every_bar_net_per_trade"] or 0.0) * int(b["every_bar_n"])
                      for b in boards)
    pooled_baseline = (every_total / every_n) if every_n else None
    spanned: Dict[int, int] = {}
    for b in boards:
        spanned[int(b["horizon_bars"])] = spanned.get(int(b["horizon_bars"]), 0) + 1
    return {
        "horizon_bars": boards[0]["horizon_bars"],
        # Every bar-horizon this pool spans, with a count, so a pool across
        # cadences says so rather than wearing the first board's number.
        "horizon_bars_spanned": dict(sorted(spanned.items())),
        "horizon_minutes": boards[0].get("horizon_minutes"),
        "round_trip_cost": boards[0]["round_trip_cost"],
        "omen_threshold": boards[0]["omen_threshold"],
        "readable_trades_floor": READABLE_TRADES,
        "pooled_boards": len(boards),
        "scored_bars": sum(int(b["scored_bars"]) for b in boards),
        "dropped_no_future": sum(int(b["dropped_no_future"]) for b in boards),
        "buy": buy,
        "sell": sell,
        "every_bar_n": every_n,
        "every_bar_net_per_trade": pooled_baseline,
        "sell_every_bar_net_per_trade": sell_every_bar(pooled_baseline,
                                                       boards[0]["round_trip_cost"]),
        "buy_edge_vs_baseline": _edge(buy, pooled_baseline),
        "crest_edge_vs_baseline": _edge(
            sell, sell_every_bar(pooled_baseline, boards[0]["round_trip_cost"])),
        "buy_omens": buy["n"],
        "buy_net_per_trade": buy["net_per_trade"],
        "trough_precision": buy["precision_paid"],
        "crest_omens": sell["n"],
        "crest_net_per_trade": sell["net_per_trade"],
        "crest_precision": sell["precision_paid"],
        "readable": buy["readable"] and sell["readable"],
    }


def _side_line(name: str, cell: Mapping[str, Any],
               baseline: Optional[float]) -> str:
    n = cell["n"]
    if n == 0:
        return f"   {name:<14}: n=0 -- NO CALLS, nothing measured"
    net = cell["net_per_trade"]
    prec = cell["precision_paid"]
    body = (f"   {name:<14}: n={n} {net:+.4%} per trade, "
            f"precision {prec:.1%}")
    if not cell["readable"]:
        # No edge is printed here on purpose. An unreadable cell's edge is
        # the most quotable-looking number in the line and the least
        # supported one.
        return (body + f"  <- UNREADABLE, n={n} is below the "
                f"{READABLE_TRADES}-trade floor; do not quote this as an edge")
    if baseline is not None:
        # The baseline this half actually competes with, and the difference,
        # because the difference is the only part that is about the caller.
        # Measured pass 118: a DOWN-window crest cell of +1.2379% sits 1.19
        # points BELOW a sell-every-bar +2.4246%, and printing the level
        # without the edge is how that got quoted as headroom.
        body += (f"  (every-bar {baseline:+.4%}, "
                 f"EDGE {net - baseline:+.4%})")
    return body


def render_scoreboard(board: Mapping[str, Any], *, title: str = "money") -> str:
    """One string carrying every per-trade net WITH the n that produced it.

    There is deliberately no code path here that emits a per-trade percentage
    without an ``n=`` in the same line. The test named for this
    (tests/test_a_per_trade_net_is_never_printed_without_its_n.py) asserts it
    line by line, so a future edit that splits the two fails the gate.
    """
    baseline = board.get("every_bar_net_per_trade")
    lines = [
        f"  {title}: {board['scored_bars']} bars scored, "
        f"{board['dropped_no_future']} dropped for no future, "
        f"threshold {board['omen_threshold']:.4%} on a "
        f"{board['round_trip_cost']:.4%} round trip",
        _side_line("buy (trough)", board["buy"], baseline),
        _side_line("sell (crest)", board["sell"],
                   sell_every_bar(baseline, board["round_trip_cost"])),
    ]
    if baseline is not None:
        lines.append(f"   every bar     : n={board['every_bar_n']} "
                     f"{baseline:+.4%} per trade")
    return "\n".join(lines)
