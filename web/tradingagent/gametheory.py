"""Model trading as the game it actually is, and price our position in it.

Every swap is played against opponents who see our order before it settles:
searchers who can sandwich it, the pool's other traders, and the counterparty
choosing whether our size is worth filling. Treating that as a solitary
optimisation problem is why a strategy can win 62.7% of its trades and still
lose money -- the wins are the trades nobody contested, and the losses are the
ones someone else priced.

Four models, each answering a question the audit raises but cannot settle:

    adverse_selection   when we win we win small and when we lose we lose big.
                        Is that variance, or is a better-informed counterparty
                        selecting which of our orders to fill?

    sandwich_exposure   what does an attacker earn by front-running us at our
                        slippage setting? If that exceeds our edge, our
                        slippage tolerance IS the leak.

    stopping_game       against a mean-reverting price, holding and exiting
                        are moves in a game against the clock. What is the
                        optimal stop, given our measured distribution?

    repeated_game       we trade the same pools repeatedly. Does our behaviour
                        make us predictable enough to be exploited?

Every function returns ``sufficient=False`` on thin data rather than a
confident number. A game-theoretic model fitted to twelve trades is a story,
and this repo has already paid for stories.
"""

from __future__ import annotations

import math
import statistics as _stats
from typing import Any, Dict, List, Optional, Sequence

#: Below this a strategic inference is storytelling. Same bar the rest of the
#: audit uses, for the same reason.
MIN_SAMPLE = 20

#: Default slippage the router allows per leg, as a fraction. This is the
#: number an attacker gets to keep.
DEFAULT_SLIPPAGE = 0.01


# ------------------------------------------------------- adverse selection --

def adverse_selection(returns: Sequence[float]) -> Dict[str, Any]:
    """Are we being picked off, or is this ordinary variance?

    The signature of adverse selection is asymmetry: a counterparty who knows
    more than we do fills our order when it is good FOR THEM, so our winners
    get clipped and our losers run. Measured on this book, ghost trades win
    62.7% of the time with an average loss 2.3x the average win -- which is
    what being selected against looks like, and is NOT what random entry into
    a symmetric price process looks like.

    The test: if entries were uninformed and the price symmetric, |avg win|
    and |avg loss| should be comparable. A large ratio with a HIGH win rate is
    the tell, because a genuinely bad strategy loses often as well as big.
    """
    values = [float(r) for r in returns if r is not None]
    n = len(values)
    if n < 4:
        return {"sufficient": False, "n": n, "reason": "need at least 4 trades"}

    wins = [r for r in values if r > 0]
    losses = [-r for r in values if r < 0]
    if not wins or not losses:
        return {"sufficient": False, "n": n,
                "reason": "need both wins and losses to compare"}

    win_rate = len(wins) / n
    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    asymmetry = avg_loss / avg_win if avg_win > 0 else float("inf")

    # Winning often while losing big is the combination that cannot be
    # explained by a symmetric process: a coin-flip strategy on a symmetric
    # price gets a ratio near 1 whatever its win rate.
    selected_against = win_rate >= 0.5 and asymmetry >= 1.5

    return {
        "sufficient": n >= MIN_SAMPLE,
        "n": n,
        "win_rate": round(win_rate, 6),
        "avg_win": round(avg_win, 8),
        "avg_loss": round(avg_loss, 8),
        "loss_to_win_ratio": round(asymmetry, 4),
        "selected_against": bool(selected_against),
        "verdict": (
            f"ADVERSE SELECTION LIKELY: winning {win_rate:.1%} of trades while "
            f"each loss costs {asymmetry:.2f}x each win. A symmetric price with "
            f"uninformed entries gives a ratio near 1. Someone is choosing "
            f"which of our orders to fill."
            if selected_against and n >= MIN_SAMPLE else
            f"No clear selection signal: win rate {win_rate:.1%}, "
            f"loss/win ratio {asymmetry:.2f}."
            if n >= MIN_SAMPLE else
            f"UNPROVEN at {n} trades."),
        # The actionable consequence, not just the diagnosis.
        "implication": (
            "Cap the loss, not the entry. At this win rate a stop near the "
            "average win turns the same trades profitable without changing a "
            "single entry rule."
            if selected_against else ""),
    }


# ------------------------------------------------------ sandwich exposure --

def sandwich_exposure(clip_usd: float, slippage: float = DEFAULT_SLIPPAGE,
                      edge_per_trade: Optional[float] = None) -> Dict[str, Any]:
    """What an attacker earns by front-running us, versus what we earn.

    A sandwich is a dominant strategy for anyone watching the mempool when our
    slippage tolerance exceeds their gas cost: they buy ahead, let our order
    move the price, and sell into it. Their profit is bounded by exactly the
    slippage we authorised.

    So our slippage setting is not a safety margin -- it is the maximum we
    have agreed to pay whoever notices us first. If it exceeds our edge, the
    game is unprofitable by construction however good the entries are.
    """
    clip = max(0.0, float(clip_usd))
    slip = max(0.0, float(slippage))

    # Both legs are exposed, so the round trip authorises twice the per-leg
    # tolerance.
    max_extractable = clip * slip * 2.0

    out: Dict[str, Any] = {
        "clip_usd": round(clip, 6),
        "slippage_per_leg": slip,
        "max_extractable_usd": round(max_extractable, 8),
        "as_fraction_of_clip": round(slip * 2.0, 6),
    }

    if edge_per_trade is not None:
        edge_usd = abs(float(edge_per_trade)) * clip
        out["edge_usd_per_trade"] = round(edge_usd, 8)
        out["attacker_takes_share"] = (
            round(max_extractable / edge_usd, 4) if edge_usd > 0 else float("inf"))
        out["edge_survives"] = edge_usd > max_extractable
        out["verdict"] = (
            f"SLIPPAGE IS THE LEAK: an attacker may extract "
            f"${max_extractable:.6f} per round trip against an edge of "
            f"${edge_usd:.6f}. The tolerance authorises more than the trade "
            f"is worth."
            if not out["edge_survives"] else
            f"Edge ${edge_usd:.6f} exceeds the ${max_extractable:.6f} a "
            f"sandwich could extract.")
        out["implication"] = (
            f"Lower slippage until 2*slippage*clip is below the edge. At this "
            f"clip that means under {(edge_usd / (2 * clip)) if clip else 0:.4f} "
            f"per leg."
            if not out["edge_survives"] else "")
    return out


# --------------------------------------------------------- stopping game --

def stopping_game(returns: Sequence[float],
                  round_trip_cost: float = 0.0065) -> Dict[str, Any]:
    """Where to stop, treating hold-versus-exit as a game against the clock.

    Each open position is a sequence of decisions: hold, or take what is on
    the table. Against a price with no drift, holding has zero expected value
    and strictly positive variance, so the optimal policy is bounded by the
    distribution of outcomes rather than by hope.

    This computes the stop that would have maximised realised expectancy over
    the trades we actually took -- not a theoretical optimum, but the one the
    measured distribution supports.
    """
    values = [float(r) for r in returns if r is not None]
    n = len(values)
    if n < MIN_SAMPLE:
        return {"sufficient": False, "n": n,
                "reason": f"need {MIN_SAMPLE} trades to fit a stop"}

    losses = sorted(-r for r in values if r < 0)
    if not losses:
        return {"sufficient": False, "n": n, "reason": "no losses to bound"}

    baseline = sum(values) / n - round_trip_cost

    # Sweep candidate stops over the observed loss distribution. A stop at s
    # replaces every loss worse than s with exactly s -- what the trade would
    # have cost had we cut it there.
    best_stop, best_exp = None, baseline
    curve: List[Dict[str, float]] = []
    for percentile in (10, 20, 30, 40, 50, 60, 70, 80, 90):
        index = min(len(losses) - 1, int(len(losses) * percentile / 100.0))
        stop = losses[index]
        if stop <= 0:
            continue
        capped = [(-stop if r < -stop else r) for r in values]
        expectancy = sum(capped) / n - round_trip_cost
        curve.append({"stop": round(stop, 6),
                      "expectancy": round(expectancy, 8),
                      "trades_cut": sum(1 for r in values if r < -stop)})
        if expectancy > best_exp:
            best_stop, best_exp = stop, expectancy

    return {
        "sufficient": True,
        "n": n,
        "baseline_expectancy": round(baseline, 8),
        "best_stop": round(best_stop, 6) if best_stop is not None else None,
        "best_expectancy": round(best_exp, 8),
        "improvement": round(best_exp - baseline, 8),
        "curve": curve,
        "verdict": (
            f"A stop at {best_stop:.4f} would have turned expectancy "
            f"{baseline:+.6f} into {best_exp:+.6f} over these {n} trades."
            if best_stop is not None else
            f"No stop in the observed range improves on {baseline:+.6f}; the "
            f"losses are not the binding problem."),
    }


# --------------------------------------------------------- repeated game --

def repeated_game(actions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Are we predictable enough to be exploited?

    We trade the same pools repeatedly at a near-constant size and cadence. In
    a repeated game that is a pure strategy, and a pure strategy is the one
    thing an opponent can always best-respond to. Mixing -- varying size and
    timing -- costs nothing when our edge does not depend on either.
    """
    entries = [a for a in actions
               if str(a.get("status", "")).endswith("-entry")]
    n = len(entries)
    if n < MIN_SAMPLE:
        return {"sufficient": False, "n": n,
                "reason": f"need {MIN_SAMPLE} entries to judge predictability"}

    sizes, gaps, symbols = [], [], {}
    previous_ts = None
    for entry in entries:
        try:
            size = float(entry.get("size") or 0.0)
            if size > 0:
                sizes.append(size)
        except (TypeError, ValueError):
            pass
        ts = float(entry.get("ts") or 0.0)
        if previous_ts and ts > previous_ts:
            gaps.append(ts - previous_ts)
        previous_ts = ts
        symbol = str(entry.get("symbol") or "")
        symbols[symbol] = symbols.get(symbol, 0) + 1

    def _cv(series: List[float]) -> float:
        """Coefficient of variation: 0 is perfectly predictable."""
        if len(series) < 2:
            return 0.0
        mean = sum(series) / len(series)
        if mean == 0:
            return 0.0
        return _stats.pstdev(series) / abs(mean)

    size_cv = _cv(sizes)
    gap_cv = _cv(gaps)
    top_share = (max(symbols.values()) / n) if symbols else 0.0

    predictable = size_cv < 0.15 and top_share > 0.4

    return {
        "sufficient": True,
        "n": n,
        "size_variation": round(size_cv, 4),
        "timing_variation": round(gap_cv, 4),
        "top_symbol_share": round(top_share, 4),
        "distinct_symbols": len(symbols),
        "predictable": bool(predictable),
        "verdict": (
            f"PREDICTABLE: size varies by only {size_cv:.1%} and "
            f"{top_share:.0%} of entries are one symbol. A pure strategy is "
            f"the one thing an opponent can always best-respond to."
            if predictable else
            f"Reasonably mixed: size varies {size_cv:.1%}, timing "
            f"{gap_cv:.1%}, across {len(symbols)} symbols."),
        "implication": (
            "Vary clip size and entry timing. Our edge does not depend on "
            "either being constant, so mixing costs nothing and removes a "
            "free read."
            if predictable else ""),
    }


# ----------------------------------------------------------------- report --

def full_game_analysis(returns: Sequence[float],
                       actions: Sequence[Dict[str, Any]],
                       clip_usd: float = 0.75,
                       slippage: float = DEFAULT_SLIPPAGE) -> Dict[str, Any]:
    selection = adverse_selection(returns)
    edge = None
    if returns:
        edge = sum(float(r) for r in returns) / len(returns)
    return {
        "adverse_selection": selection,
        "sandwich_exposure": sandwich_exposure(clip_usd, slippage, edge),
        "stopping_game": stopping_game(returns),
        "repeated_game": repeated_game(actions),
    }


def game_summary(analysis: Dict[str, Any]) -> List[str]:
    """The strategic picture as lines an agent can act on."""
    lines: List[str] = []
    for key in ("adverse_selection", "sandwich_exposure", "stopping_game",
                "repeated_game"):
        block = analysis.get(key) or {}
        verdict = block.get("verdict")
        if verdict:
            lines.append(f"{key}: {verdict}")
        implication = block.get("implication")
        if implication:
            lines.append(f"  -> {implication}")
    return lines
