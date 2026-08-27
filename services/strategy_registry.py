"""Commission, decommission, and re-experiment on strategies.

The lifecycle surface behind the strategies screen:

  * see every strategy, discovered or hand-written, with its live record
  * commission one (allow it to trade) or decommission it (stop it)
  * run a GA search for NEW strategies under a chosen objective
  * know which AI model/brain every strategy was derived from
  * re-run an existing strategy's experiment against a DIFFERENT model and
    compare, without losing the original

OBJECTIVES
----------
"Best" is not one number, so the objective is a parameter. Net profit,
inference accuracy, expectancy, consistency, and drawdown-adjusted return
optimise for genuinely different strategies, and a caller may also scope a
search to a metric window (a symbol set, a date range, a horizon).

Every objective is still computed OUT-OF-SAMPLE. Choosing "prioritise net
profit" changes what the search rewards; it does not license rewarding a
number the strategy never earned on held-out data.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Anchored to the repo root: web workers run from web/ while the trading
# process runs from the repo root, so a relative default silently split
# these files in two and the dashboard read an empty one.
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_REGISTRY_PATH = _ROOT / "data" / "strategy_registry.json"
REGISTRY_PATH = Path(os.getenv("STRATEGY_REGISTRY_PATH", str(_DEFAULT_REGISTRY_PATH)))

_lock = threading.RLock()


# --------------------------------------------------------------------------
# Objectives
# --------------------------------------------------------------------------

def _obj_net_profit(g: Any) -> float:
    return float(getattr(g, "oos_expectancy", 0.0)) * float(getattr(g, "oos_trades", 0))


def _obj_accuracy(g: Any) -> float:
    """Inference accuracy, credited only above the majority-class baseline."""
    return max(0.0, float(getattr(g, "oos_edge", 0.0)))


def _obj_expectancy(g: Any) -> float:
    return float(getattr(g, "oos_expectancy", 0.0))


def _obj_consistency(g: Any) -> float:
    """Edge scaled by sample size: rewards an edge that keeps showing up."""
    import math
    n = float(getattr(g, "oos_trades", 0))
    return max(0.0, float(getattr(g, "oos_edge", 0.0))) * math.sqrt(min(1.0, n / 300.0))


def _obj_balanced(g: Any) -> float:
    return float(getattr(g, "fitness", 0.0))


OBJECTIVES: Dict[str, Callable[[Any], float]] = {
    "net_profit": _obj_net_profit,
    "accuracy": _obj_accuracy,
    "expectancy": _obj_expectancy,
    "consistency": _obj_consistency,
    "balanced": _obj_balanced,
}

OBJECTIVE_LABELS = {
    "net_profit": "Prioritise net profit (expectancy x trade count)",
    "accuracy": "Prioritise inference accuracy (edge over baseline)",
    "expectancy": "Prioritise per-trade expectancy",
    "consistency": "Prioritise a repeatable edge (sample-weighted)",
    "balanced": "Balanced (edge x expectancy x confidence)",
}


def score(genome: Any, objective: str = "balanced") -> float:
    """Objective score. A genome with no out-of-sample edge scores zero under
    EVERY objective -- the choice reweights real performance, it never
    substitutes for it."""
    if float(getattr(genome, "oos_edge", 0.0)) <= 0.0:
        return 0.0
    return OBJECTIVES.get(objective, _obj_balanced)(genome)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def _load() -> Dict[str, Any]:
    try:
        if REGISTRY_PATH.exists():
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"strategies": {}}


def _under_test() -> bool:
    """Is this a test run writing to the PRODUCTION registry?

    Tests that forget to patch REGISTRY_PATH silently write fabricated
    strategies into the real lifetime record. Observed 2026-08-27: a strategy
    literally named "s" with 360 ghost and 108 live trades, and an
    rsi_reversal live record of 9 losses of exactly -0.5, both from test
    fixtures -- while the database held ZERO live rows. The live-path check
    then reported "live P/L -3.84 over 117 trades" for trades that never
    happened.

    Fabricated performance data is the one thing this file must never hold, so
    a write from a test run to the default path is refused rather than trusted.
    """
    if "PYTEST_CURRENT_TEST" not in os.environ:
        return False
    return REGISTRY_PATH == _DEFAULT_REGISTRY_PATH


def _save(state: Dict[str, Any]) -> None:
    if _under_test():
        raise RuntimeError(
            "refusing to write the production strategy registry from a test; "
            "patch services.strategy_registry.REGISTRY_PATH to a temp file"
        )
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, REGISTRY_PATH)


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------

def register_strategy(
    *,
    name: str,
    kind: str = "genome",
    genes: Optional[Dict[str, Any]] = None,
    run_id: str = "",
    objective: str = "balanced",
    model_id: str = "",
    model_name: str = "",
    metrics: Optional[Dict[str, Any]] = None,
    commissioned: bool = False,
) -> Dict[str, Any]:
    """Add a strategy, stamped with the model it was derived from.

    ``model_id``/``model_name`` are the provenance Adam asked for: a strategy
    found using one brain must be labelled as such, so a later run on a
    different brain is a comparison rather than a silent overwrite.
    """
    strategy_id = "%s_%s" % (kind, uuid.uuid4().hex[:10])
    entry = {
        "strategy_id": strategy_id,
        "name": name or strategy_id,
        "kind": kind,
        "genes": genes or {},
        "run_id": run_id,
        "objective": objective,
        "objective_label": OBJECTIVE_LABELS.get(objective, objective),
        "model_id": model_id,
        "model_name": model_name or model_id or "(no brain)",
        "metrics": metrics or {},
        "commissioned": bool(commissioned),
        "created_at": time.time(),
        "experiments": [],
    }
    with _lock:
        state = _load()
        state.setdefault("strategies", {})[strategy_id] = entry
        _save(state)
    return entry


def list_strategies(include_decommissioned: bool = True) -> List[Dict[str, Any]]:
    out = list(_load().get("strategies", {}).values())
    if not include_decommissioned:
        out = [s for s in out if s.get("commissioned")]
    out.sort(key=lambda s: (not s.get("commissioned"), -float(s.get("created_at", 0))))
    return out


def get_strategy(strategy_id: str) -> Optional[Dict[str, Any]]:
    return _load().get("strategies", {}).get(strategy_id)


def set_commissioned(strategy_id: str, commissioned: bool) -> Optional[Dict[str, Any]]:
    """Commission or decommission. Decommissioning is always allowed;
    commissioning requires a real out-of-sample edge."""
    with _lock:
        state = _load()
        entry = state.get("strategies", {}).get(strategy_id)
        if not entry:
            return None
        if commissioned:
            edge = float((entry.get("metrics") or {}).get("oos_edge", 0.0) or 0.0)
            if edge <= 0.0:
                entry["commission_error"] = (
                    "refused: no out-of-sample edge over the baseline"
                )
                state["strategies"][strategy_id] = entry
                _save(state)
                return entry
        entry["commissioned"] = bool(commissioned)
        entry.pop("commission_error", None)
        entry["commission_changed_at"] = time.time()
        state["strategies"][strategy_id] = entry
        _save(state)
    return entry


def add_experiment(
    strategy_id: str,
    *,
    run_id: str,
    model_id: str,
    model_name: str,
    objective: str,
    metrics: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Record a re-run of this strategy against a (possibly different) model.

    Appends rather than replaces, so the same strategy evaluated on two brains
    keeps both results side by side for comparison.
    """
    with _lock:
        state = _load()
        entry = state.get("strategies", {}).get(strategy_id)
        if not entry:
            return None
        entry.setdefault("experiments", []).append({
            "experiment_id": uuid.uuid4().hex[:10],
            "run_id": run_id,
            "model_id": model_id,
            "model_name": model_name or model_id or "(no brain)",
            "objective": objective,
            "metrics": metrics,
            "ts": time.time(),
        })
        state["strategies"][strategy_id] = entry
        _save(state)
    return entry


def compare_experiments(strategy_id: str) -> List[Dict[str, Any]]:
    """This strategy's results across every model it was tested on."""
    entry = get_strategy(strategy_id)
    if not entry:
        return []
    rows = [{
        "model_id": entry.get("model_id", ""),
        "model_name": entry.get("model_name", "(no brain)"),
        "objective": entry.get("objective", ""),
        "metrics": entry.get("metrics", {}),
        "origin": True,
    }]
    for exp in entry.get("experiments", []):
        rows.append({
            "model_id": exp.get("model_id", ""),
            "model_name": exp.get("model_name", "(no brain)"),
            "objective": exp.get("objective", ""),
            "metrics": exp.get("metrics", {}),
            "experiment_id": exp.get("experiment_id"),
            "origin": False,
        })
    rows.sort(key=lambda r: float((r.get("metrics") or {}).get("oos_edge", 0.0) or 0.0), reverse=True)
    return rows


def record_outcome(
    strategy_id: str,
    *,
    profit: float,
    mode: str = "ghost",
    symbol: str = "",
    ts: Optional[float] = None,
) -> Dict[str, Any]:
    """Fold one closed trade into a strategy's LIFETIME record.

    Deliberately separate from ``StrategyLedger``. That ledger is a rolling
    promotion window and gets reset -- observed 2026-08-26, a reset left 18
    trades on record against 245 actual exits, which made the strategy's real
    history unreadable. Lifetime counters are append-only and survive every
    ledger reset, so "how has this strategy ever actually done" always has an
    answer.

    Ghost and live are tracked separately: simulated profit and realised
    profit are not the same claim and must never be summed into one number.
    """
    now = float(ts if ts is not None else time.time())
    key = "live" if str(mode).lower().startswith("live") else "ghost"
    with _lock:
        state = _load()
        entry = state.get("strategies", {}).get(strategy_id)
        if not entry:
            # Auto-register on first sight. Strategies that predate this
            # registry (or are hand-written rather than GA-discovered) still
            # need a lifetime record -- dropping their outcomes would leave
            # the screen blank for exactly the strategies that have been
            # trading longest.
            entry = {
                "strategy_id": strategy_id,
                "name": strategy_id,
                "kind": "builtin",
                "genes": {},
                "objective": "",
                "model_id": "",
                "model_name": "(no brain)",
                "metrics": {},
                "commissioned": True,
                "created_at": now,
                "experiments": [],
                "auto_registered": True,
            }
            state.setdefault("strategies", {})[strategy_id] = entry
        lifetime = entry.setdefault("lifetime", {})
        stats = lifetime.setdefault(key, {
            "trades": 0, "wins": 0, "losses": 0,
            "gross_win": 0.0, "gross_loss": 0.0, "total_profit": 0.0,
            "best": 0.0, "worst": 0.0,
            "peak_profit": 0.0, "max_drawdown": 0.0,
            "consecutive_losses": 0, "max_consecutive_losses": 0,
            "first_ts": now, "last_ts": now, "symbols": {},
        })
        p = float(profit)
        stats["trades"] += 1
        stats["total_profit"] = float(stats["total_profit"]) + p
        if p > 0:
            stats["wins"] += 1
            stats["gross_win"] = float(stats["gross_win"]) + p
            stats["consecutive_losses"] = 0
            stats["best"] = max(float(stats["best"]), p)
        else:
            stats["losses"] += 1
            stats["gross_loss"] = float(stats["gross_loss"]) + abs(p)
            stats["consecutive_losses"] = int(stats["consecutive_losses"]) + 1
            stats["max_consecutive_losses"] = max(
                int(stats["max_consecutive_losses"]), int(stats["consecutive_losses"])
            )
            stats["worst"] = min(float(stats["worst"]), p)
        stats["peak_profit"] = max(float(stats["peak_profit"]), float(stats["total_profit"]))
        stats["max_drawdown"] = max(
            float(stats["max_drawdown"]),
            float(stats["peak_profit"]) - float(stats["total_profit"]),
        )
        stats["last_ts"] = now
        if symbol:
            stats["symbols"][symbol] = int(stats["symbols"].get(symbol, 0)) + 1
        state["strategies"][strategy_id] = entry
        _save(state)
    return entry


def lifetime_metrics(strategy_id: str) -> Dict[str, Any]:
    """Derived lifetime performance, ghost and live reported separately."""
    entry = get_strategy(strategy_id) or {}
    lifetime = entry.get("lifetime", {}) or {}
    out: Dict[str, Any] = {}
    for key in ("ghost", "live"):
        s = lifetime.get(key)
        if not s:
            out[key] = {"trades": 0}
            continue
        trades = int(s.get("trades", 0))
        wins = int(s.get("wins", 0))
        gross_win = float(s.get("gross_win", 0.0))
        gross_loss = float(s.get("gross_loss", 0.0))
        total = float(s.get("total_profit", 0.0))
        avg_win = gross_win / wins if wins else 0.0
        losses = int(s.get("losses", 0))
        avg_loss = gross_loss / losses if losses else 0.0
        out[key] = {
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / trades) if trades else 0.0,
            "total_profit": total,
            "avg_profit": (total / trades) if trades else 0.0,
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0),
            "payoff_ratio": (avg_win / avg_loss) if avg_loss > 0 else 0.0,
            "best": float(s.get("best", 0.0)),
            "worst": float(s.get("worst", 0.0)),
            "max_drawdown": float(s.get("max_drawdown", 0.0)),
            "max_consecutive_losses": int(s.get("max_consecutive_losses", 0)),
            "first_ts": float(s.get("first_ts", 0.0)),
            "last_ts": float(s.get("last_ts", 0.0)),
            "active_days": max(0.0, (float(s.get("last_ts", 0.0)) - float(s.get("first_ts", 0.0))) / 86400.0),
            "distinct_symbols": len(s.get("symbols", {}) or {}),
            "top_symbols": sorted(
                (s.get("symbols", {}) or {}).items(), key=lambda kv: -kv[1]
            )[:5],
        }
    return out


def delete_strategy(strategy_id: str) -> bool:
    with _lock:
        state = _load()
        if strategy_id not in state.get("strategies", {}):
            return False
        del state["strategies"][strategy_id]
        _save(state)
    return True
