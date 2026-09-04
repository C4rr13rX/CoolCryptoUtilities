"""Audit every action the pipeline took, with exact numbers and a significance test.

The system already computes distributions, Kelly and confusion matrices in
``trading/metrics.py``. What it never had was the question those feed: given
what we ACTUALLY did, is the result distinguishable from luck?

That gap is why this repo has repeatedly acted on noise. A strategy was
promoted on 3 trades, demoted on 2 consecutive losses, and once graduated on
records that were pure fabrication. Every one of those decisions would have
failed a sample-size test that nobody was running.

So this module answers four questions, in exact arithmetic over the recorded
actions rather than over a model of them:

    algebra       what did each round trip actually cost and return, gross,
                  net of fees, and in the units the wallet moved in?
    probability   given the win rate and the payoff, what is the expected
                  value of the next trade, and what fraction of the bankroll
                  does Kelly justify?
    statistics    is this edge distinguishable from zero at this sample size,
                  or is it noise dressed as a result?
    calculus      how is performance CHANGING -- the first derivative of
                  cumulative P/L, and whether the trend is accelerating or
                  decaying?

Every function refuses to answer rather than guess. ``n=0`` returns zeros with
``sufficient=False``, because a confident number computed from nothing is more
dangerous than an admitted gap -- that is the failure that produced four
strategies with invented records.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB = ROOT / "storage" / "trading_cache.db"

#: Below this, a win rate is not a measurement. Chosen because the normal
#: approximation to the binomial needs roughly this much to be meaningful, and
#: because this repo has already promoted a strategy on three trades.
MIN_SAMPLE = 20

#: Below this magnitude an acceleration is numerical noise, not a trend.
#: Subtracting equal per-trade gains leaves values around 1e-18, and treating
#: those as a real second derivative reports a straight line as decaying.
_ACCEL_EPS = 1e-12

#: Round-trip cost in fraction. Two legs plus slippage: a move smaller than
#: this is not profit, it is the fee arriving late.
DEFAULT_ROUND_TRIP_COST = 0.0065


# ------------------------------------------------------------- gathering --

def _connect() -> Optional[sqlite3.Connection]:
    try:
        return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    except Exception:
        return None


def recorded_actions(since_sec: float = 86400 * 7) -> List[Dict[str, Any]]:
    """Every trading action on record, as exact rows rather than a summary.

    This is the ledger the audit is computed over. It reads what HAPPENED --
    including refusals and failures -- because a picture built only from
    completed trades hides the reason most of them never completed.
    """
    conn = _connect()
    if conn is None:
        return []

    cutoff = time.time() - since_sec
    out: List[Dict[str, Any]] = []
    try:
        rows = conn.execute(
            "SELECT ts, symbol, action, status, details FROM trading_ops "
            "WHERE ts > ? ORDER BY ts", (cutoff,))
        for ts, symbol, action, status, details in rows:
            record: Dict[str, Any] = {
                "ts": float(ts or 0.0),
                "symbol": str(symbol or ""),
                "action": str(action or ""),
                "status": str(status or ""),
            }
            try:
                payload = json.loads(details) if details else {}
            except Exception:
                payload = {}
            for key in ("profit", "net_pnl", "size", "entry_price", "exit_price",
                        "reason", "exit_reason", "strategy_id", "tx_hash"):
                if key in payload:
                    record[key] = payload[key]
            out.append(record)
    except Exception:
        pass
    return out


def _outcomes(actions: Sequence[Dict[str, Any]], *, live_only: bool = False
              ) -> List[float]:
    """Realised returns, as fractions. One number per closed round trip."""
    values: List[float] = []
    for row in actions:
        status = row.get("status") or ""
        if live_only and not status.startswith("live"):
            continue
        if not live_only and not (status.startswith("ghost-exit")
                                  or status.startswith("live-exit")):
            continue
        if live_only and "exit" not in status:
            continue
        raw = row.get("profit", row.get("net_pnl"))
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(value) or math.isinf(value):
            continue
        values.append(value)
    return values


# --------------------------------------------------------------- algebra --

def algebra(actions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Exact arithmetic over what was done. No estimates, no models.

    Counts every status, so refusals are as visible as fills. A pipeline that
    refuses 26 entries and fills 2 is not a pipeline with 2 trades -- it is one
    with 28 decisions, and the 26 are the story.
    """
    counts: Dict[str, int] = {}
    for row in actions:
        status = row.get("status") or "?"
        counts[status] = counts.get(status, 0) + 1

    settled = sum(v for k, v in counts.items() if k.endswith("swap-settled"))
    refused = sum(v for k, v in counts.items()
                  if "refused" in k or "blocked" in k or "failed" in k)
    entries = sum(v for k, v in counts.items() if k.endswith("-entry"))
    exits = sum(v for k, v in counts.items() if k.endswith("-exit"))

    returns = _outcomes(actions)
    gross_win = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r < 0))

    return {
        "actions_total": len(actions),
        "by_status": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "settled_swaps": settled,
        "refused_or_failed": refused,
        "entries": entries,
        "exits": exits,
        # Entries without matching exits are capital that went out and did not
        # come back. This number being large is the shape of stranded funds.
        "unclosed": max(0, entries - exits),
        "gross_win": round(gross_win, 8),
        "gross_loss": round(gross_loss, 8),
        "net": round(gross_win - gross_loss, 8),
        "profit_factor": round(gross_win / gross_loss, 6) if gross_loss > 0
                         else (999.0 if gross_win > 0 else 0.0),
    }


# ----------------------------------------------------------- probability --

def probability(returns: Sequence[float],
                round_trip_cost: float = DEFAULT_ROUND_TRIP_COST) -> Dict[str, Any]:
    """Expected value of the next trade, and what Kelly justifies risking.

    Expectancy is computed NET OF THE ROUND TRIP. A strategy with a positive
    gross edge and a negative net one is a strategy that pays the exchange to
    take its risk, and reporting the gross number is how that gets missed.
    """
    n = len(returns)
    if n == 0:
        return {"sufficient": False, "n": 0, "reason": "no closed trades"}

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    win_rate = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else (999.0 if avg_win else 0.0)

    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    expectancy_net = expectancy - round_trip_cost

    try:
        from trading.metrics import kelly_fraction

        kelly = kelly_fraction(win_rate, payoff)
    except Exception:
        kelly = 0.0

    return {
        "sufficient": n >= MIN_SAMPLE,
        "n": n,
        "win_rate": round(win_rate, 6),
        "avg_win": round(avg_win, 8),
        "avg_loss": round(avg_loss, 8),
        "payoff_ratio": round(payoff, 6),
        "expectancy_gross": round(expectancy, 8),
        "round_trip_cost": round_trip_cost,
        "expectancy_net": round(expectancy_net, 8),
        # The honest answer when the edge does not clear its own costs.
        "edge_survives_fees": expectancy_net > 0,
        "kelly_fraction": round(kelly, 6),
        # Half-Kelly is the size to actually use: full Kelly assumes the
        # measured edge is the true edge, and at n<100 it is mostly variance.
        "kelly_half": round(kelly / 2.0, 6),
    }


# ------------------------------------------------------------ statistics --

def statistics(returns: Sequence[float]) -> Dict[str, Any]:
    """Is this edge distinguishable from zero, or is it noise?

    A one-sample t-test on the returns. The null hypothesis is that the true
    mean is zero -- that we have no edge and the result is variance. Failing to
    reject it is not a failure of the strategy; it is the honest statement that
    the sample cannot tell us either way, and acting as though it could is what
    put fabricated records into production here.
    """
    n = len(returns)
    if n < 2:
        return {"sufficient": False, "n": n, "reason": "need at least 2 trades"}

    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(variance)
    stderr = std / math.sqrt(n) if n > 0 else 0.0
    t_stat = (mean / stderr) if stderr > 0 else 0.0

    # Two-sided p-value via the normal approximation. Adequate at n>=20 and
    # conservative below it, which is the direction we want to err.
    p_value = math.erfc(abs(t_stat) / math.sqrt(2.0))

    try:
        from trading.metrics import distribution_report

        dist = distribution_report(returns)
    except Exception:
        dist = {}

    # 95% CI on the mean. If it straddles zero, we cannot say the edge exists.
    margin = 1.96 * stderr
    ci_low, ci_high = mean - margin, mean + margin

    return {
        "sufficient": n >= MIN_SAMPLE,
        "n": n,
        "mean": round(mean, 8),
        "std": round(std, 8),
        "stderr": round(stderr, 8),
        "t_statistic": round(t_stat, 6),
        "p_value": round(p_value, 6),
        "significant_at_05": bool(p_value < 0.05 and n >= MIN_SAMPLE),
        "ci95_low": round(ci_low, 8),
        "ci95_high": round(ci_high, 8),
        # The plain-language answer, so nobody has to interpret a p-value
        # under time pressure.
        "verdict": _verdict(n, mean, p_value, ci_low, ci_high),
        "distribution": dist,
    }


def _verdict(n: int, mean: float, p_value: float,
             ci_low: float, ci_high: float) -> str:
    if n < MIN_SAMPLE:
        return (f"UNPROVEN: {n} trades is below the {MIN_SAMPLE} needed to "
                f"tell an edge from variance. Do not size on this.")
    if ci_low <= 0.0 <= ci_high:
        return (f"NOT SIGNIFICANT: the 95% interval [{ci_low:+.5f}, "
                f"{ci_high:+.5f}] contains zero, so no edge is demonstrated.")
    if mean > 0:
        return (f"POSITIVE EDGE: mean {mean:+.5f} per trade, p={p_value:.4f}, "
                f"interval excludes zero.")
    return (f"NEGATIVE EDGE: mean {mean:+.5f} per trade, p={p_value:.4f}. "
            f"This is losing money measurably, not by chance.")


# -------------------------------------------------------------- calculus --

def calculus(returns: Sequence[float]) -> Dict[str, Any]:
    """How performance is CHANGING, not just where it stands.

    A cumulative total says where we are. Its derivative says whether we are
    still getting there -- a strategy that made money and is now handing it
    back looks identical to a healthy one until you differentiate.
    """
    n = len(returns)
    if n < 3:
        return {"sufficient": False, "n": n, "reason": "need at least 3 trades"}

    cumulative: List[float] = []
    running = 0.0
    for r in returns:
        running += r
        cumulative.append(running)

    # First derivative: per-trade rate of change of the cumulative curve.
    first = [cumulative[i] - cumulative[i - 1] for i in range(1, n)]
    # Second derivative: is that rate itself rising or falling?
    second = [first[i] - first[i - 1] for i in range(1, len(first))]

    # Least-squares slope of the cumulative curve -- the trend line an eye
    # would draw, computed rather than guessed.
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(cumulative) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    slope = (sum((xs[i] - mean_x) * (cumulative[i] - mean_y)
                 for i in range(n)) / denom) if denom else 0.0

    # Drawdown from the running peak: what has already been handed back.
    peak = cumulative[0]
    max_dd = 0.0
    for value in cumulative:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)

    recent = first[-5:] if len(first) >= 5 else first
    recent_rate = sum(recent) / len(recent) if recent else 0.0
    accel = sum(second) / len(second) if second else 0.0

    return {
        "sufficient": n >= MIN_SAMPLE,
        "n": n,
        "cumulative_final": round(cumulative[-1], 8),
        "trend_slope": round(slope, 8),
        "recent_rate": round(recent_rate, 8),
        "acceleration": round(accel, 8),
        "peak": round(max(cumulative), 8),
        "max_drawdown": round(max_dd, 8),
        # Giving back more than a third of the peak is the shape of a strategy
        # whose edge has gone, whatever the total still says.
        "giving_back": bool(max(cumulative) > 0
                            and max_dd > max(cumulative) * 0.33),
        # Acceleration is compared against a tolerance, not against zero.
        # Floating-point subtraction of equal gains leaves values like -1e-18,
        # which read as "decaying" when the curve is perfectly straight --
        # a steadily profitable strategy would have been reported as fading.
        "direction": ("improving" if slope > 0 and accel >= -_ACCEL_EPS else
                      "decaying" if slope > 0 else
                      "worsening" if slope < 0 else "flat"),
    }


# ----------------------------------------------------------------- audit --

def full_audit(since_sec: float = 86400 * 7) -> Dict[str, Any]:
    """The complete picture, over everything actually recorded."""
    actions = recorded_actions(since_sec)
    ghost_returns = _outcomes(actions)
    live_returns = _outcomes(actions, live_only=True)

    return {
        "window_hours": round(since_sec / 3600, 1),
        "as_of": time.time(),
        "algebra": algebra(actions),
        "ghost": {
            "probability": probability(ghost_returns),
            "statistics": statistics(ghost_returns),
            "calculus": calculus(ghost_returns),
        },
        "live": {
            "probability": probability(live_returns),
            "statistics": statistics(live_returns),
            "calculus": calculus(live_returns),
        },
    }


def audit_summary(audit: Optional[Dict[str, Any]] = None) -> List[str]:
    """The audit as sentences an agent can act on."""
    audit = audit or full_audit()
    lines: List[str] = []

    alg = audit.get("algebra") or {}
    lines.append(
        f"{alg.get('actions_total', 0)} recorded actions: "
        f"{alg.get('settled_swaps', 0)} settled, "
        f"{alg.get('refused_or_failed', 0)} refused or failed, "
        f"{alg.get('unclosed', 0)} positions opened and not closed.")
    lines.append(
        f"net {alg.get('net', 0):+.6f} "
        f"(gross win {alg.get('gross_win', 0):.6f}, "
        f"gross loss {alg.get('gross_loss', 0):.6f}, "
        f"profit factor {alg.get('profit_factor', 0)}).")

    for scope in ("ghost", "live"):
        block = audit.get(scope) or {}
        stats = block.get("statistics") or {}
        prob = block.get("probability") or {}
        calc = block.get("calculus") or {}
        if not stats.get("n"):
            lines.append(f"{scope}: no closed trades.")
            continue
        lines.append(f"{scope}: {stats.get('verdict')}")
        if prob.get("n"):
            lines.append(
                f"{scope}: win rate {prob.get('win_rate', 0):.3f}, payoff "
                f"{prob.get('payoff_ratio', 0):.3f}, expectancy net of fees "
                f"{prob.get('expectancy_net', 0):+.6f} -- edge survives fees: "
                f"{prob.get('edge_survives_fees')}. Half-Kelly size "
                f"{prob.get('kelly_half', 0):.4f} of bankroll.")
        if calc.get("n"):
            lines.append(
                f"{scope}: trend {calc.get('direction')}, slope "
                f"{calc.get('trend_slope', 0):+.6f}/trade, max drawdown "
                f"{calc.get('max_drawdown', 0):.6f}"
                + (" -- GIVING BACK its peak." if calc.get("giving_back") else "."))
    return lines
