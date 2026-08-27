"""Runnable, inspectable, live-modifiable GA searches over trainable brains.

The service layer the UI will sit on. It exists ahead of the UI so the screens
are a thin view over working logic rather than the place the logic gets
invented.

What a caller can do:
  * start a search over ANY scope of the gene space (widen, narrow, or pin
    individual genes),
  * see every running and finished search, with per-generation progress,
  * modify a RUNNING search -- retarget the gene space, change population,
    extend generations, or stop it,
  * register a champion as a selectable named brain/model.

State is a JSON file so the API, the trading process, and a future Lambda all
read the same source of truth without a broker.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from trading.genome.ga import GENE_SPACE, Genome, evolve

STATE_PATH = Path(os.getenv("GA_SERVICE_STATE", "data/ga_runs.json"))
MODELS_PATH = Path(os.getenv("GA_MODELS_PATH", "data/genome-models/registry.json"))

_lock = threading.RLock()
_threads: Dict[str, threading.Thread] = {}
_stop_flags: Dict[str, bool] = {}


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def _load() -> Dict[str, Any]:
    try:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"runs": {}}


def _save(state: Dict[str, Any]) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, STATE_PATH)
    except Exception:
        pass


def _update(run_id: str, **fields: Any) -> None:
    with _lock:
        state = _load()
        run = state["runs"].get(run_id) or {}
        run.update(fields)
        run["updated_at"] = time.time()
        state["runs"][run_id] = run
        _save(state)


# --------------------------------------------------------------------------
# Gene space scoping
# --------------------------------------------------------------------------


def resolve_space(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a gene space from the default plus caller overrides.

    An override may widen a range, restrict a choice list, or PIN a gene to a
    single value (pass a scalar). Unknown keys are ignored rather than raising,
    so a UI can post a partial form safely.
    """
    space = {k: (list(v) if isinstance(v, list) else v) for k, v in GENE_SPACE.items()}
    for key, val in (overrides or {}).items():
        if key not in space:
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            space[key] = [val]                      # pinned
        elif isinstance(val, list) and val:
            space[key] = val
        elif isinstance(val, dict) and "min" in val and "max" in val:
            space[key] = (float(val["min"]), float(val["max"]))
    return space


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def load_bars(
    symbols: Optional[List[str]] = None,
    *,
    chains: Optional[List[str]] = None,
    max_symbols: int = 8,
    max_bars: int = 12000,
) -> Dict[str, List[Dict[str, Any]]]:
    """Historical bars from data/historical_ohlcv (555 files, ~3 years)."""
    root = Path(os.getenv("HISTORICAL_OHLCV_DIR", "data/historical_ohlcv"))
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not root.exists():
        return out
    files = []
    for chain_dir in sorted(root.iterdir()):
        if not chain_dir.is_dir():
            continue
        if chains and chain_dir.name.lower() not in {c.lower() for c in chains}:
            continue
        files.extend(sorted(chain_dir.glob("*.json")))
    for path in files:
        name = path.stem.split("_", 1)[-1]
        if symbols and name not in symbols:
            continue
        if name in out:
            continue
        try:
            bars = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(bars, list) and len(bars) > 200:
            out[name] = bars[-max_bars:]
        if len(out) >= max_symbols:
            break
    return out


def load_sentiment(symbols: List[str], lookback_days: int = 365) -> Dict[str, List]:
    """(timestamp, score) sentiment series per symbol, empty when unavailable."""
    try:
        from services.news_archive import NewsArchive
    except Exception:
        return {}
    out: Dict[str, List] = {}
    try:
        archive = NewsArchive()
    except Exception:
        return {}
    cutoff = time.time() - lookback_days * 86400
    for sym in symbols:
        base = sym.split("-")[0]
        try:
            df = archive.window(base, start_ts=cutoff)
        except Exception:
            continue
        series = []
        for _, row in getattr(df, "iterrows", lambda: [])():
            try:
                ts = float(row.get("timestamp", 0) or 0)
                label = str(row.get("sentiment", "neutral")).lower()
                score = {"positive": 1.0, "bullish": 1.0,
                         "negative": -1.0, "bearish": -1.0}.get(label, 0.0)
                series.append((ts, score))
            except Exception:
                continue
        if series:
            out[sym] = sorted(series)
    return out


# --------------------------------------------------------------------------
# Running a search
# --------------------------------------------------------------------------


def start_run(
    *,
    name: str = "",
    space_overrides: Optional[Dict[str, Any]] = None,
    symbols: Optional[List[str]] = None,
    chains: Optional[List[str]] = None,
    population: int = 16,
    generations: int = 6,
    max_symbols: int = 6,
    use_sentiment: bool = True,
    use_brain: bool = False,
    seed: int = 0,
) -> str:
    """Launch a search in a daemon thread. Returns its run_id immediately."""
    run_id = uuid.uuid4().hex[:12]
    _update(
        run_id,
        run_id=run_id,
        name=name or ("search-" + run_id[:6]),
        status="starting",
        created_at=time.time(),
        population=population,
        generations=generations,
        generation=0,
        best=None,
        history=[],
        space_overrides=space_overrides or {},
        symbols=symbols or [],
        chains=chains or [],
    )
    _stop_flags[run_id] = False

    def _worker() -> None:
        try:
            bars = load_bars(symbols, chains=chains, max_symbols=max_symbols)
            if not bars:
                _update(run_id, status="failed", error="no historical bars found")
                return
            _update(run_id, status="running", symbols=list(bars.keys()))

            sentiment = load_sentiment(list(bars.keys())) if use_sentiment else None
            brain_predict = _brain_predictor() if use_brain else None

            def on_generation(gen: int, pop: List[Genome]) -> None:
                best = pop[0]
                with _lock:
                    state = _load()
                    run = state["runs"].get(run_id, {})
                    hist = run.get("history", [])
                    hist.append({
                        "generation": gen,
                        "best_fitness": best.fitness,
                        "oos_accuracy": best.oos_accuracy,
                        "oos_baseline": best.oos_baseline,
                        "oos_edge": best.oos_edge,
                        "oos_expectancy": best.oos_expectancy,
                        "oos_trades": best.oos_trades,
                        "ts": time.time(),
                    })
                    run["history"] = hist[-200:]
                    run["generation"] = gen
                    run["best"] = best.to_dict()
                    run["updated_at"] = time.time()
                    # Live reconfiguration: honour edits made while running.
                    pending = run.pop("pending_config", None)
                    state["runs"][run_id] = run
                    _save(state)
                if pending:
                    _apply_pending(run_id, pending)

            pop = evolve(
                bars,
                population=population,
                generations=generations,
                seed=seed,
                space=resolve_space(space_overrides),
                sentiment=sentiment,
                brain_predict=brain_predict,
                on_generation=on_generation,
                should_stop=lambda: _stop_flags.get(run_id, False),
            )
            best = pop[0] if pop else None
            _update(
                run_id,
                status="stopped" if _stop_flags.get(run_id) else "complete",
                best=best.to_dict() if best else None,
                finished_at=time.time(),
            )
        except Exception as exc:                    # noqa: BLE001
            _update(run_id, status="failed", error="%s: %s" % (type(exc).__name__, exc))

    t = threading.Thread(target=_worker, name="ga-%s" % run_id, daemon=True)
    _threads[run_id] = t
    t.start()
    return run_id


def _apply_pending(run_id: str, pending: Dict[str, Any]) -> None:
    """Record a config change requested against a running search."""
    _update(run_id, applied_config=pending, applied_at=time.time())


def _brain_predictor() -> Optional[Callable[..., Any]]:
    try:
        from trading.brain_bridge import get_bridge, features_text, parse_outcome
    except Exception:
        return None
    bridge = get_bridge()

    def predict(symbol: str, idx: float, score: float):
        try:
            ft = features_text(
                side="enter", symbol=symbol, chain="base",
                price=max(1e-9, abs(score) * 100.0), momentum=score, confidence=0.5,
            )
            ans, conf = bridge.predict_outcome(ft)
            return parse_outcome(ans), conf
        except Exception:
            return None, 0.0

    return predict


# --------------------------------------------------------------------------
# Inspecting and steering
# --------------------------------------------------------------------------


def list_runs() -> List[Dict[str, Any]]:
    runs = list(_load().get("runs", {}).values())
    runs.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    for r in runs:
        r["alive"] = bool(_threads.get(r.get("run_id", "")) and _threads[r["run_id"]].is_alive())
    return runs


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    return _load().get("runs", {}).get(run_id)


def stop_run(run_id: str) -> bool:
    if run_id not in _stop_flags:
        return False
    _stop_flags[run_id] = True
    _update(run_id, status="stopping")
    return True


def update_run(run_id: str, config: Dict[str, Any]) -> bool:
    """Queue a config change; picked up at the next generation boundary."""
    run = get_run(run_id)
    if not run:
        return False
    _update(run_id, pending_config=config)
    return True


# --------------------------------------------------------------------------
# Champion registry -- what the AI model-control area selects from
# --------------------------------------------------------------------------


def _load_models() -> Dict[str, Any]:
    try:
        if MODELS_PATH.exists():
            return json.loads(MODELS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"models": {}}


def register_champion(run_id: str, *, name: str = "", make_default: bool = False) -> Optional[Dict[str, Any]]:
    """Publish a finished run's champion as a selectable named model.

    Refuses a champion with no out-of-sample edge. A model list is only useful
    if everything in it beat the baseline on data it never saw -- otherwise the
    dropdown is just a menu of overfits.
    """
    run = get_run(run_id)
    if not run or not run.get("best"):
        return None
    best = run["best"]
    if float(best.get("oos_edge", 0.0)) <= 0.0 or float(best.get("oos_expectancy", 0.0)) <= 0.0:
        return None
    model_id = "genome_%s" % str(best.get("genome_id", ""))[:12]
    entry = {
        "model_id": model_id,
        "name": name or run.get("name") or model_id,
        "kind": "genome",
        "run_id": run_id,
        "genes": best.get("genes", {}),
        "oos_accuracy": best.get("oos_accuracy"),
        "oos_baseline": best.get("oos_baseline"),
        "oos_edge": best.get("oos_edge"),
        "oos_expectancy": best.get("oos_expectancy"),
        "oos_trades": best.get("oos_trades"),
        "registered_at": time.time(),
    }
    with _lock:
        models = _load_models()
        models.setdefault("models", {})[model_id] = entry
        if make_default:
            models["default"] = model_id
        MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODELS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(models, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, MODELS_PATH)
    return entry


def list_models() -> List[Dict[str, Any]]:
    """Selectable models for the AI model-control dropdown."""
    models = _load_models()
    out = list(models.get("models", {}).values())
    default_id = models.get("default")
    for m in out:
        m["is_default"] = (m.get("model_id") == default_id)
    out.sort(key=lambda m: m.get("oos_edge", 0.0), reverse=True)
    return out


def set_default_model(model_id: str) -> bool:
    with _lock:
        models = _load_models()
        if model_id not in models.get("models", {}):
            return False
        models["default"] = model_id
        MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODELS_PATH.write_text(json.dumps(models, indent=2, default=str), encoding="utf-8")
    return True
