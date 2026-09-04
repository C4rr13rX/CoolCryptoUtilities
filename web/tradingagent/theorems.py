"""Propose, test and retire mathematical claims about when we make money.

The audit says what happened. Game theory says who we are playing. Neither
proposes anything NEW -- and the standing models were built for markets that
are not this one: liquid, continuous, low-fee, with counterparties who cannot
see our order before it settles. None of those hold here.

So this is the discovery loop. A theorem is a falsifiable claim of the form
"when CONDITION holds, expected return differs from baseline by EFFECT", and
it lives on the same ladder everything else does:

    PROPOSED   stated, never tested. Carries no weight.
    TESTING    being measured against held-out trades.
    SUPPORTED  effect held on data it was not fitted to, at p < 0.05.
    REFUTED    it did not. Kept, because knowing what fails is worth as much
               as knowing what works, and deleting it invites re-proposing it.

The rule that makes this honest is the split: a theorem is FITTED on the first
portion of history and JUDGED on the rest. Without that, any claim can be made
true by looking at the data it came from -- which is how this repo produced
strategies whose entire records were fabrication. A theorem that only works on
its own training window is not a discovery, it is a memory.

Nothing here reaches live trading. A SUPPORTED theorem becomes a ghost
experiment, and the existing graduation ladder decides the rest.
"""

from __future__ import annotations

import math
import statistics as _stats
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: A theorem judged on fewer than this many held-out trades is not judged.
MIN_HOLDOUT = 15

#: Fraction of history used to FIT. The rest is never seen while forming the
#: claim, and is the only evidence allowed to support it.
FIT_FRACTION = 0.6


# ------------------------------------------------------------- primitives --

def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _welch_t(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Welch's t and a two-sided p, for samples with unequal variance.

    Welch rather than Student because the two groups a theorem splits are
    almost never the same size or spread -- assuming they are inflates
    significance in exactly the direction that flatters a false claim.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0, 1.0
    var_a = _stats.variance(a)
    var_b = _stats.variance(b)
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, 1.0
    t = (_mean(a) - _mean(b)) / se
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return t, p


def _cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Effect size. Significance says an effect is real; this says if it matters."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = math.sqrt((_stats.variance(a) + _stats.variance(b)) / 2.0)
    return (_mean(a) - _mean(b)) / pooled if pooled else 0.0


# ------------------------------------------------------------- the object --

class Theorem:
    """A falsifiable claim about when returns differ from baseline.

    The predicate takes one trade record and answers whether the condition
    holds. Everything else -- the fit, the hold-out, the verdict -- follows
    from that, so a claim cannot be stated in a way that resists testing.
    """

    def __init__(self, name: str, statement: str,
                 predicate: Callable[[Dict[str, Any]], bool],
                 rationale: str = "") -> None:
        self.name = name
        self.statement = statement
        self.predicate = predicate
        self.rationale = rationale

    def evaluate(self, trades: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Fit on the first slice, judge on the rest. Never both on one."""
        usable = [t for t in trades if isinstance(t.get("return"), (int, float))]
        n = len(usable)
        if n < MIN_HOLDOUT * 2:
            return {"name": self.name, "statement": self.statement,
                    "status": "PROPOSED", "n": n,
                    "verdict": f"UNTESTABLE: {n} trades, need "
                               f"{MIN_HOLDOUT * 2} to fit and hold out."}

        split = int(n * FIT_FRACTION)
        fit, holdout = usable[:split], usable[split:]

        def _split_by(rows):
            yes = [float(r["return"]) for r in rows if self._safe(r)]
            no = [float(r["return"]) for r in rows if not self._safe(r)]
            return yes, no

        fit_yes, fit_no = _split_by(fit)
        out_yes, out_no = _split_by(holdout)

        if len(fit_yes) < 3 or len(fit_no) < 3:
            return {"name": self.name, "statement": self.statement,
                    "status": "PROPOSED", "n": n,
                    "verdict": "UNTESTABLE: the condition does not split the "
                               "fitting window into two usable groups."}

        fit_effect = _mean(fit_yes) - _mean(fit_no)

        if len(out_yes) < 3 or len(out_no) < 3:
            return {"name": self.name, "statement": self.statement,
                    "status": "TESTING", "n": n,
                    "fit_effect": round(fit_effect, 8),
                    "verdict": f"FITTED but not yet judged: the hold-out has "
                               f"{len(out_yes)} matching and {len(out_no)} "
                               f"non-matching trades, too few to decide."}

        t_stat, p_value = _welch_t(out_yes, out_no)
        out_effect = _mean(out_yes) - _mean(out_no)
        effect_size = _cohens_d(out_yes, out_no)

        # Held-out significance AND an effect pointing the same way as the fit.
        # A claim that reverses sign out of sample was fitting noise, however
        # significant the reversal looks.
        same_direction = (fit_effect > 0) == (out_effect > 0)
        supported = bool(p_value < 0.05 and same_direction
                         and abs(effect_size) >= 0.2)

        return {
            "name": self.name,
            "statement": self.statement,
            "rationale": self.rationale,
            "status": "SUPPORTED" if supported else "REFUTED",
            "n": n,
            "fit_n": len(fit), "holdout_n": len(holdout),
            "fit_effect": round(fit_effect, 8),
            "holdout_effect": round(out_effect, 8),
            "holdout_matching": len(out_yes),
            "holdout_other": len(out_no),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "effect_size": round(effect_size, 4),
            "same_direction": same_direction,
            "verdict": (
                f"SUPPORTED: on held-out trades the condition changes return "
                f"by {out_effect:+.6f} (p={p_value:.4f}, d={effect_size:.2f}), "
                f"in the same direction as the fit."
                if supported else
                f"REFUTED: held-out effect {out_effect:+.6f} "
                f"(p={p_value:.4f}, d={effect_size:.2f})"
                + ("" if same_direction else
                   f", and it REVERSES the fitted effect of {fit_effect:+.6f} "
                   f"-- the claim was fitting noise.")),
        }

    def _safe(self, row: Dict[str, Any]) -> bool:
        try:
            return bool(self.predicate(row))
        except Exception:
            return False


# ------------------------------------------------------ standing theorems --

def standing_theorems() -> List[Theorem]:
    """The claims this system currently operates on, stated so they can fail.

    These are not new. They are the assumptions already baked into thresholds
    and gates, written down as falsifiable statements so the agent can see
    what it is inheriting and test whether any of it is true here.
    """
    return [
        Theorem(
            "fee_floor",
            "A move smaller than the round trip cannot be profit.",
            lambda t: abs(float(t.get("return") or 0.0)) > 0.0065,
            "Two legs at ~0.3% plus slippage. Assumed everywhere; never tested."),
        Theorem(
            "dense_feed",
            "Trades on symbols with more ticks resolve better, because a stop "
            "can only fire when a tick arrives.",
            lambda t: int(t.get("ticks_1h") or 0) >= 20,
            "The density gate assumes this. It has never been measured against "
            "realised returns."),
        Theorem(
            "short_hold",
            "Shorter holds do better on this feed, because a rising price "
            "continues rising only 44-50% of the time.",
            lambda t: float(t.get("hold_sec") or 0.0) <= 1800,
            "The money_button premise, stated as something that can be wrong."),
        Theorem(
            "small_clip",
            "Smaller clips do better, because price impact and sandwich "
            "extraction both scale with size.",
            lambda t: float(t.get("size_usd") or 0.0) <= 0.75,
            "Implied by every clip ceiling in the system."),
    ]


# ---------------------------------------------------- discovery scaffolds --

def candidate_theorems(trades: Sequence[Dict[str, Any]]) -> List[Theorem]:
    """New claims to test, generated from what the data actually contains.

    Deliberately mechanical rather than clever. These are the axes along which
    this book plausibly differs, turned into statements; the agent proposes
    the interesting ones, and this ensures the obvious ones are never missed
    because nobody thought to look.
    """
    out: List[Theorem] = []

    holds = [float(t.get("hold_sec") or 0.0) for t in trades
             if t.get("hold_sec")]
    if len(holds) >= MIN_HOLDOUT * 2:
        median_hold = _stats.median(holds)
        out.append(Theorem(
            "hold_below_median",
            f"Holding under {median_hold:.0f}s beats holding longer.",
            lambda t, m=median_hold: float(t.get("hold_sec") or 0.0) <= m,
            "Split at the median of what we actually held, not at a "
            "threshold someone guessed."))

    vols = [float(t.get("volatility_pct") or 0.0) for t in trades
            if t.get("volatility_pct")]
    if len(vols) >= MIN_HOLDOUT * 2:
        median_vol = _stats.median(vols)
        out.append(Theorem(
            "calm_beats_volatile",
            f"Entering when hourly range is under {median_vol:.2f}% beats "
            f"entering when it is higher.",
            lambda t, m=median_vol: float(t.get("volatility_pct") or 0.0) <= m,
            "Volatility is usually assumed good for a short-horizon lane. On a "
            "book being selected against it may be the opposite."))

    out.append(Theorem(
        "loss_capped",
        "Trades that never exceeded a 1.3% drawdown outperform the rest.",
        lambda t: float(t.get("return") or 0.0) > -0.013,
        "The stopping game found 0.0013 optimal on this distribution. Stated "
        "as a theorem so it is tested out of sample rather than trusted."))

    return out


def evaluate_all(trades: Sequence[Dict[str, Any]],
                 extra: Optional[Sequence[Theorem]] = None) -> Dict[str, Any]:
    """Test everything we claim to believe, plus everything worth trying."""
    theorems = list(standing_theorems()) + list(candidate_theorems(trades))
    if extra:
        theorems.extend(extra)

    results = [t.evaluate(trades) for t in theorems]
    supported = [r for r in results if r.get("status") == "SUPPORTED"]
    refuted = [r for r in results if r.get("status") == "REFUTED"]

    return {
        "tested": len(results),
        "supported": len(supported),
        "refuted": len(refuted),
        "results": results,
        # Ordered by effect size: the biggest true difference is the one worth
        # acting on, not merely the most significant.
        "best": sorted(supported,
                       key=lambda r: -abs(r.get("effect_size") or 0.0))[:5],
    }


def theorem_summary(report: Dict[str, Any]) -> List[str]:
    lines = [f"{report['tested']} theorems tested: {report['supported']} "
             f"supported on held-out data, {report['refuted']} refuted."]
    for result in report.get("results", []):
        lines.append(f"  [{result.get('status')}] {result.get('name')}: "
                     f"{result.get('statement')}")
        lines.append(f"      {result.get('verdict')}")
    return lines


# ------------------------------------------------------------ trade rows --

def trades_with_features(since_sec: float = 86400 * 7) -> List[Dict[str, Any]]:
    """Closed trades, each carrying the features a theorem can test.

    Built by pairing entries with their exits, because the features that
    matter -- how long it was held, how volatile the symbol was -- only exist
    across the pair. An exit alone knows its return and nothing about how it
    got there.
    """
    import json as _json
    import sqlite3 as _sqlite3
    import time as _time
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[2]
    out: List[Dict[str, Any]] = []
    try:
        conn = _sqlite3.connect(
            f"file:{root / 'storage' / 'trading_cache.db'}?mode=ro", uri=True)
        cutoff = _time.time() - since_sec

        # Tick density per symbol, so the dense-feed claim can be tested.
        density = {}
        for symbol, count in conn.execute(
                "SELECT symbol, COUNT(*) FROM market_stream WHERE ts > ? "
                "GROUP BY symbol", (cutoff,)):
            density[str(symbol)] = int(count or 0)

        open_at: Dict[str, Dict[str, Any]] = {}
        for ts, symbol, status, details in conn.execute(
                "SELECT ts, symbol, status, details FROM trading_ops "
                "WHERE ts > ? AND (status LIKE '%-entry' OR status LIKE '%-exit') "
                "ORDER BY ts", (cutoff,)):
            symbol = str(symbol or "")
            try:
                payload = _json.loads(details) if details else {}
            except Exception:
                payload = {}

            if str(status).endswith("-entry"):
                open_at[symbol] = {"ts": float(ts or 0.0), "payload": payload}
                continue

            opened = open_at.pop(symbol, None)
            raw = payload.get("profit", payload.get("net_pnl"))
            try:
                ret = float(raw)
            except (TypeError, ValueError):
                continue
            if ret != ret or ret in (float("inf"), float("-inf")):
                continue

            entry_payload = (opened or {}).get("payload") or {}
            try:
                size_usd = float(entry_payload.get("size") or 0.0)
            except (TypeError, ValueError):
                size_usd = 0.0

            out.append({
                "symbol": symbol,
                "return": ret,
                "hold_sec": (float(ts) - opened["ts"]) if opened else None,
                "size_usd": size_usd,
                "ticks_1h": density.get(symbol, 0) / max(1.0, since_sec / 3600.0),
                "live": str(status).startswith("live"),
                "ts": float(ts or 0.0),
            })
    except Exception:
        pass
    return out
