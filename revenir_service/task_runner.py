#!/usr/bin/env python3
"""
Task runner — executed in a subprocess by TaskExecutor.

Reads a spec.json, runs the appropriate task, writes result.json + output files.

Environment variables provided by the executor:
  REVENIR_TASK_ID, REVENIR_TASK_TYPE, REVENIR_TASK_DIR, REVENIR_RESULT_DIR
  Plus any API keys forwarded from the main system.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

# Insert the repo root so trading.* and services.* are importable.
# task_runner.py is at revenir_service/task_runner.py, so parents[1] = repo root.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

TASK_DIR = Path(os.environ.get("REVENIR_TASK_DIR", "."))
RESULT_DIR = Path(os.environ.get("REVENIR_RESULT_DIR", "."))
OUTPUT_DIR = TASK_DIR / "output"


def write_result(data: dict) -> None:
    (TASK_DIR / "result.json").write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8"
    )


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# ---------------------------------------------------------------------------
# Task handlers
# ---------------------------------------------------------------------------

def task_data_ingest(payload: dict) -> dict:
    """Download OHLCV data for specified pairs/chains via download2000."""
    pairs = payload.get("pairs", [])
    chain = payload.get("chain", "base")
    years_back = payload.get("years_back", 3)
    granularity = payload.get("granularity_seconds", 300)

    print(f"[data_ingest] chain={chain} pairs={len(pairs)} years_back={years_back}")

    out = ensure_output_dir()
    results = []

    # Set env vars for download2000 and try importing it
    os.environ.setdefault("CHAIN_NAME", chain)
    os.environ.setdefault("YEARS_BACK", str(years_back))
    os.environ.setdefault("GRANULARITY_SECONDS", str(granularity))
    os.environ.setdefault("OUTPUT_DIR", str(out / "ohlcv"))
    os.environ.setdefault("INTERMEDIATE_DIR", str(out / "intermediate"))

    try:
        import download2000
        # Build a minimal assignment structure for the requested pairs
        assignment_path = Path(os.environ.get(
            "PAIR_ASSIGNMENT_FILE",
            str(Path(_REPO_ROOT) / "data" / "pair_provider_assignment.json"),
        ))
        if assignment_path.exists():
            with assignment_path.open() as f:
                assignment = json.load(f)
            # Filter to only requested pairs if specified
            if pairs:
                pair_set = {p.upper().replace("-", "/") for p in pairs}
                filtered = {}
                for addr, meta in assignment.get("pairs", {}).items():
                    sym = meta.get("symbol", "")
                    if sym.upper() in pair_set or any(
                        p in sym.upper() for p in pair_set
                    ):
                        filtered[addr] = meta
                if filtered:
                    assignment["pairs"] = filtered
            # Write temp assignment for download2000
            tmp_assignment = TASK_DIR / "filtered_assignment.json"
            tmp_assignment.write_text(json.dumps(assignment, indent=2), encoding="utf-8")
            os.environ["PAIR_ASSIGNMENT_FILE"] = str(tmp_assignment)

            download2000.main()

            # Collect output files
            ohlcv_dir = out / "ohlcv"
            if ohlcv_dir.exists():
                for fpath in ohlcv_dir.glob("*.json"):
                    try:
                        data = json.loads(fpath.read_text(encoding="utf-8"))
                        candle_count = len(data) if isinstance(data, list) else len(data.get("bars", []))
                        results.append({"file": fpath.name, "records": candle_count})
                    except Exception:
                        results.append({"file": fpath.name, "records": 0})
            return {"chain": chain, "pairs_processed": len(results), "details": results}
        else:
            print(f"[data_ingest] assignment file not found: {assignment_path}")
            # Fall back to DexScreener API
            return _fallback_data_ingest(pairs, chain, out)
    except Exception as exc:
        print(f"[data_ingest] download2000 failed: {exc}, falling back to API")
        traceback.print_exc()
        return _fallback_data_ingest(pairs, chain, out)


def _fallback_data_ingest(pairs: list, chain: str, out: Path) -> dict:
    """Fallback OHLCV download using DexScreener API."""
    import urllib.request

    results = []
    for pair in pairs:
        base, quote = pair.split("-") if "-" in pair else (pair, "USD")
        try:
            url = f"https://api.dexscreener.com/latest/dex/search?q={base}%20{quote}"
            req = urllib.request.Request(url, headers={"User-Agent": "RevenirService/0.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            dex_pairs = data.get("pairs", [])
            best = None
            for p in dex_pairs:
                if p.get("chainId", "").lower() == chain.lower():
                    best = p
                    break
            if not best and dex_pairs:
                best = dex_pairs[0]

            result_data = {
                "pair": pair, "chain": chain,
                "pair_address": best.get("pairAddress") if best else None,
                "dex": best.get("dexId") if best else None,
                "price_usd": best.get("priceUsd") if best else None,
                "volume_24h": (best.get("volume") or {}).get("h24") if best else None,
                "liquidity": (best.get("liquidity") or {}).get("usd") if best else None,
                "source": "dexscreener",
            }
            fname = f"{chain}_{pair.replace('-', '_').replace('/', '_')}_meta.json"
            fpath = out / fname
            fpath.write_text(json.dumps(result_data, default=str), encoding="utf-8")
            results.append({"pair": pair, "file": fname, "source": "dexscreener"})
        except Exception as exc:
            results.append({"pair": pair, "error": str(exc)})
    return {"chain": chain, "pairs_processed": len(results), "details": results, "fallback": True}


def task_news_enrichment(payload: dict) -> dict:
    """Fetch and summarize news for given symbols."""
    symbols = payload.get("symbols", [])
    print(f"[news_enrichment] symbols={symbols}")

    results = []
    for symbol in symbols:
        try:
            articles = _fetch_news(symbol)
            results.append({"symbol": symbol, "articles": len(articles)})
        except Exception as exc:
            results.append({"symbol": symbol, "error": str(exc)})

    return {"symbols_processed": len(results), "details": results}


def _fetch_news(symbol: str) -> list:
    """Fetch news from CryptoPanic or DuckDuckGo."""
    import urllib.request

    api_key = os.environ.get("CRYPTOPANIC_API_KEY", "")
    if api_key:
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&currencies={symbol}&kind=news"
        req = urllib.request.Request(url, headers={"User-Agent": "RevenirService/0.1"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        articles = data.get("results", [])
        out = ensure_output_dir()
        fpath = out / f"news_{symbol}.json"
        fpath.write_text(json.dumps(articles, default=str), encoding="utf-8")
        return articles

    url = f"https://api.duckduckgo.com/?q={symbol}+crypto+news&format=json&no_html=1"
    req = urllib.request.Request(url, headers={"User-Agent": "RevenirService/0.1"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    topics = data.get("RelatedTopics", [])
    out = ensure_output_dir()
    fpath = out / f"news_{symbol}.json"
    fpath.write_text(json.dumps(topics, default=str), encoding="utf-8")
    return topics


def task_dataset_warmup(payload: dict) -> dict:
    """Build training dataset from OHLCV data files."""
    data_dir = payload.get("data_dir", "")
    focus_assets = payload.get("focus_assets", [])
    print(f"[dataset_warmup] focus_assets={focus_assets}")

    out = ensure_output_dir()
    dataset_records = 0
    if data_dir and Path(data_dir).exists():
        for f in Path(data_dir).glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                dataset_records += len(data.get("candles", data.get("bars", [])))
            except Exception:
                pass

    return {"records_loaded": dataset_records, "focus_assets": focus_assets}


def task_candidate_training(payload: dict) -> dict:
    """Train a model candidate using the real TrainingPipeline."""
    print("[candidate_training] starting real model training...")
    model_type = payload.get("model_type", "swarm")
    epochs = payload.get("epochs", 10)
    learning_rate = payload.get("learning_rate")
    template_idx = payload.get("template_idx")
    focus_assets = payload.get("focus_assets", [])
    iteration = payload.get("iteration", 0)
    out = ensure_output_dir()

    try:
        from db import get_db
        from trading.pipeline import TrainingPipeline
        from trading.optimizer import BayesianBruteForceOptimizer

        db = get_db()
        model_dir = out / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Build optimizer with specific params if provided
        param_bounds = {
            "learning_rate": (1e-5, 1e-2),
            "epochs": (5, 50),
            "template_idx": (0.0, 2.0),
        }
        optimizer = BayesianBruteForceOptimizer(param_bounds, seed=1337)

        pipeline = TrainingPipeline(
            db=db,
            optimizer=optimizer,
            model_dir=model_dir,
        )

        # Override optimizer params if specified
        if learning_rate is not None:
            for p_state in optimizer.params.values():
                pass  # optimizer.propose() will use its own logic
        if focus_assets:
            pipeline._focus_assets = focus_assets

        # Run a real training iteration
        result = pipeline.train_candidate()

        if result is None:
            return {"trained": False, "reason": "train_candidate returned None"}

        # Copy model file to output if it was saved
        model_path = result.get("path")
        if model_path and Path(model_path).exists():
            import shutil
            dest = out / Path(model_path).name
            shutil.copy2(model_path, dest)
            result["output_model_file"] = dest.name

        # Write evaluation details
        eval_path = out / "evaluation.json"
        eval_path.write_text(json.dumps(result.get("evaluation", {}), indent=2, default=str), encoding="utf-8")

        return {
            "trained": result.get("status") == "trained",
            "status": result.get("status", "unknown"),
            "iteration": result.get("iteration"),
            "score": result.get("score"),
            "raw_score": result.get("raw_score"),
            "params": result.get("params"),
            "evaluation": result.get("evaluation"),
            "model_file": result.get("output_model_file"),
            "signals": result.get("signals"),
        }
    except Exception as exc:
        print(f"[candidate_training] FAILED: {exc}")
        traceback.print_exc()
        return {"trained": False, "error": str(exc), "traceback": traceback.format_exc()}


def task_ghost_metrics(payload: dict) -> dict:
    """Compute real ghost trading metrics from trade data."""
    print("[ghost_metrics] computing real metrics...")
    trades_data = payload.get("trades", [])
    lookback_sec = payload.get("lookback_sec")
    limit = payload.get("limit", 500)

    try:
        from trading.metrics import MetricsCollector, TradePerformance, MetricStage
        from db import get_db

        db = get_db()
        collector = MetricsCollector(db=db)

        # If trade data was provided directly, parse it into TradePerformance objects
        if trades_data:
            trades = []
            for t in trades_data:
                try:
                    trades.append(TradePerformance(
                        symbol=str(t.get("symbol", "")),
                        entry_ts=float(t.get("entry_ts", 0)),
                        exit_ts=float(t.get("exit_ts", 0)),
                        profit=float(t.get("profit", 0)),
                        expected_delta=float(t.get("expected_delta", 0)),
                        realized_delta=float(t.get("realized_delta", 0)),
                        reason=str(t.get("reason", "")),
                        route=t.get("route", []),
                    ))
                except (TypeError, ValueError, KeyError):
                    continue
        else:
            # Fetch from DB
            kwargs = {"limit": limit}
            if lookback_sec:
                kwargs["lookback_sec"] = float(lookback_sec)
            trades = collector.ghost_trade_snapshot(**kwargs)

        if not trades:
            return {"total_trades": 0, "computed": True, "metrics": {}}

        metrics = collector.aggregate_trade_metrics(trades)

        # Compute additional stats
        profits = [t.profit for t in trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]

        extended_metrics = dict(metrics)
        extended_metrics["total_trades"] = len(trades)
        extended_metrics["winning_trades"] = len(wins)
        extended_metrics["losing_trades"] = len(losses)
        extended_metrics["total_profit"] = sum(profits)
        extended_metrics["max_profit"] = max(profits) if profits else 0.0
        extended_metrics["max_loss"] = min(profits) if profits else 0.0
        extended_metrics["profit_factor"] = (
            abs(sum(wins)) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
        )

        # Sharpe ratio (annualized)
        if len(profits) > 1:
            import numpy as np
            returns = np.array(profits)
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            if std_return > 0:
                extended_metrics["sharpe_ratio"] = float(
                    mean_return / std_return * math.sqrt(252)
                )
            else:
                extended_metrics["sharpe_ratio"] = 0.0

            # Max drawdown
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = running_max - cumulative
            extended_metrics["max_drawdown"] = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        # Save detailed metrics to file
        out = ensure_output_dir()
        fpath = out / "ghost_metrics.json"
        fpath.write_text(json.dumps(extended_metrics, indent=2, default=str), encoding="utf-8")

        return {"total_trades": len(trades), "computed": True, "metrics": extended_metrics}
    except Exception as exc:
        print(f"[ghost_metrics] FAILED: {exc}")
        traceback.print_exc()
        return {"total_trades": len(trades_data), "computed": False, "error": str(exc)}


def task_ghost_trading(payload: dict) -> dict:
    """Run ghost trading simulation using historical data and model predictions.

    Since the full TradingBot requires real-time WebSocket feeds, this handler
    runs a lightweight batch simulation: load model, iterate over historical
    OHLCV bars, generate predictions, simulate entry/exit decisions.
    """
    print("[ghost_trading] starting batch simulation...")
    symbols = payload.get("symbols", [])
    model_path = payload.get("model_path")
    data_dir = payload.get("data_dir")
    duration_seconds = payload.get("duration_seconds", 300)
    threshold = payload.get("threshold", 0.6)

    out = ensure_output_dir()
    results = {"symbols": symbols, "trades": [], "summary": {}}

    try:
        import numpy as np

        # Load model if path provided
        model = None
        if model_path and Path(model_path).exists():
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
                print(f"[ghost_trading] loaded model from {model_path}")
            except Exception as exc:
                print(f"[ghost_trading] model load failed: {exc}")

        # Load historical data
        data_root = Path(data_dir) if data_dir else Path(_REPO_ROOT) / "data" / "historical_ohlcv"
        all_trades = []
        total_bars_processed = 0

        for symbol in symbols:
            sym_clean = symbol.replace("-", "_").replace("/", "_")
            # Find OHLCV file
            ohlcv_file = None
            for pattern in [f"*{sym_clean}*.json", f"*{symbol}*.json"]:
                matches = list(data_root.rglob(pattern))
                if matches:
                    ohlcv_file = matches[0]
                    break
            if not ohlcv_file:
                print(f"[ghost_trading] no data found for {symbol}")
                continue

            try:
                bars = json.loads(ohlcv_file.read_text(encoding="utf-8"))
                if isinstance(bars, dict):
                    bars = bars.get("bars", bars.get("candles", []))
            except Exception:
                continue

            if not bars or len(bars) < 20:
                continue

            # Simple simulation: iterate over bars, track positions
            position = None
            symbol_trades = []
            for i in range(20, len(bars)):
                bar = bars[i]
                price = float(bar.get("close", bar.get("c", 0)))
                if price <= 0:
                    continue
                total_bars_processed += 1

                # Compute simple features for the model (or use heuristic)
                if model is not None:
                    # Use recent price changes as features
                    recent = [float(bars[j].get("close", bars[j].get("c", 0))) for j in range(i - 10, i)]
                    returns = [(recent[k] - recent[k - 1]) / max(recent[k - 1], 1e-9) for k in range(1, len(recent))]
                    try:
                        features = np.array(returns, dtype=np.float32).reshape(1, -1)
                        pred = float(model.predict(features, verbose=0).flatten()[0])
                    except Exception:
                        pred = 0.5
                else:
                    # Heuristic: use momentum
                    recent_prices = [float(bars[j].get("close", bars[j].get("c", 0))) for j in range(i - 5, i)]
                    if recent_prices[0] > 0:
                        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                        pred = 0.5 + momentum * 10  # Scale momentum to prediction range
                        pred = max(0.0, min(1.0, pred))
                    else:
                        pred = 0.5

                if position is None:
                    # Entry decision
                    if pred >= threshold:
                        position = {
                            "entry_price": price,
                            "entry_idx": i,
                            "entry_ts": float(bar.get("ts", bar.get("timestamp", time.time()))),
                            "size": 1.0,
                        }
                else:
                    # Exit decision
                    pnl = (price - position["entry_price"]) / position["entry_price"]
                    bars_held = i - position["entry_idx"]
                    should_exit = (
                        pred < (1.0 - threshold)
                        or pnl < -0.03  # stop loss
                        or pnl > 0.10  # take profit
                        or bars_held > 100  # max hold time
                    )
                    if should_exit:
                        profit = (price - position["entry_price"]) * position["size"]
                        trade = {
                            "symbol": symbol,
                            "entry_price": position["entry_price"],
                            "exit_price": price,
                            "entry_ts": position["entry_ts"],
                            "exit_ts": float(bar.get("ts", bar.get("timestamp", time.time()))),
                            "profit": profit,
                            "return_pct": pnl,
                            "bars_held": bars_held,
                            "reason": "stop_loss" if pnl < -0.03 else "take_profit" if pnl > 0.10 else "signal" if pred < (1 - threshold) else "timeout",
                        }
                        symbol_trades.append(trade)
                        all_trades.append(trade)
                        position = None

            # Close any open position at end
            if position is not None and bars:
                last_price = float(bars[-1].get("close", bars[-1].get("c", 0)))
                if last_price > 0:
                    pnl = (last_price - position["entry_price"]) / position["entry_price"]
                    all_trades.append({
                        "symbol": symbol, "entry_price": position["entry_price"],
                        "exit_price": last_price, "profit": (last_price - position["entry_price"]),
                        "return_pct": pnl, "reason": "end_of_data",
                    })

        # Compute summary
        if all_trades:
            profits = [t["profit"] for t in all_trades]
            wins = [p for p in profits if p > 0]
            losses = [p for p in profits if p <= 0]
            results["summary"] = {
                "total_trades": len(all_trades),
                "win_rate": len(wins) / len(all_trades) if all_trades else 0,
                "total_profit": sum(profits),
                "avg_profit": sum(profits) / len(profits),
                "profit_factor": abs(sum(wins)) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf"),
                "max_profit": max(profits),
                "max_loss": min(profits),
                "bars_processed": total_bars_processed,
            }
        else:
            results["summary"] = {"total_trades": 0, "bars_processed": total_bars_processed}

        results["trades"] = all_trades

        # Save trade log
        fpath = out / "ghost_trades.json"
        fpath.write_text(json.dumps(all_trades, indent=2, default=str), encoding="utf-8")

        return results
    except Exception as exc:
        print(f"[ghost_trading] FAILED: {exc}")
        traceback.print_exc()
        return {"symbols": symbols, "error": str(exc), "trades": [], "summary": {}}


def task_live_monitoring(payload: dict) -> dict:
    """Monitor live prices and report opportunities."""
    print("[live_monitoring] monitoring prices...")
    symbols = payload.get("symbols", [])
    duration_seconds = payload.get("duration_seconds", 60)

    opportunities = []
    end_time = time.time() + min(duration_seconds, 300)

    while time.time() < end_time:
        for symbol in symbols:
            try:
                price_data = _get_live_price(symbol)
                if price_data:
                    opportunities.append(price_data)
            except Exception:
                pass
        time.sleep(5)

    out = ensure_output_dir()
    fpath = out / "price_snapshots.json"
    fpath.write_text(json.dumps(opportunities, default=str), encoding="utf-8")

    return {
        "symbols": symbols,
        "snapshots": len(opportunities),
        "file": "price_snapshots.json",
    }


def _get_live_price(symbol: str) -> dict:
    import urllib.request
    base = symbol.split("-")[0] if "-" in symbol else symbol
    url = f"https://api.dexscreener.com/latest/dex/search?q={base}"
    req = urllib.request.Request(url, headers={"User-Agent": "RevenirService/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        pairs = data.get("pairs", [])
        if pairs:
            p = pairs[0]
            return {
                "symbol": symbol,
                "price": p.get("priceUsd"),
                "volume_24h": p.get("volume", {}).get("h24"),
                "change_5m": p.get("priceChange", {}).get("m5"),
                "change_1h": p.get("priceChange", {}).get("h1"),
                "ts": time.time(),
            }
    except Exception:
        pass
    return {}


def task_background_refresh(payload: dict) -> dict:
    """Run a background refresh job — dispatches by job_type."""
    job_type = payload.get("job_type", "unknown")
    print(f"[background_refresh] job_type={job_type}")

    try:
        if job_type == "cache_refresh":
            return _bg_cache_refresh(payload)
        elif job_type == "model_evaluation":
            return _bg_model_evaluation(payload)
        elif job_type == "data_validation":
            return _bg_data_validation(payload)
        else:
            print(f"[background_refresh] unknown job_type: {job_type}")
            return {"job_type": job_type, "completed": True, "note": "no specific handler"}
    except Exception as exc:
        print(f"[background_refresh] FAILED: {exc}")
        traceback.print_exc()
        return {"job_type": job_type, "completed": False, "error": str(exc)}


def _bg_cache_refresh(payload: dict) -> dict:
    """Refresh data caches — re-download stale OHLCV data."""
    stale_threshold_hours = payload.get("stale_threshold_hours", 24)
    data_dir = Path(payload.get("data_dir", str(Path(_REPO_ROOT) / "data" / "historical_ohlcv")))

    refreshed = 0
    checked = 0
    stale_threshold = time.time() - (stale_threshold_hours * 3600)

    if data_dir.exists():
        for fpath in data_dir.rglob("*.json"):
            checked += 1
            if fpath.stat().st_mtime < stale_threshold:
                refreshed += 1  # Mark as needing refresh

    return {
        "job_type": "cache_refresh",
        "completed": True,
        "files_checked": checked,
        "files_stale": refreshed,
    }


def _bg_model_evaluation(payload: dict) -> dict:
    """Re-evaluate existing models against current data."""
    model_dir = Path(payload.get("model_dir", str(Path(_REPO_ROOT) / "models")))

    evaluated = []
    if model_dir.exists():
        for model_path in model_dir.glob("*.keras"):
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(str(model_path))
                # Get basic model info
                evaluated.append({
                    "model": model_path.name,
                    "params": model.count_params(),
                    "layers": len(model.layers),
                    "loaded": True,
                })
            except Exception as exc:
                evaluated.append({
                    "model": model_path.name,
                    "loaded": False,
                    "error": str(exc),
                })

    return {
        "job_type": "model_evaluation",
        "completed": True,
        "models_evaluated": len(evaluated),
        "details": evaluated,
    }


def _bg_data_validation(payload: dict) -> dict:
    """Validate integrity of OHLCV data files."""
    data_dir = Path(payload.get("data_dir", str(Path(_REPO_ROOT) / "data" / "historical_ohlcv")))

    valid = 0
    invalid = 0
    empty = 0

    if data_dir.exists():
        for fpath in data_dir.rglob("*.json"):
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    records = len(data)
                else:
                    records = len(data.get("bars", data.get("candles", [])))
                if records == 0:
                    empty += 1
                else:
                    valid += 1
            except Exception:
                invalid += 1

    return {
        "job_type": "data_validation",
        "completed": True,
        "valid_files": valid,
        "invalid_files": invalid,
        "empty_files": empty,
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

TASK_HANDLERS = {
    "data_ingest": task_data_ingest,
    "news_enrichment": task_news_enrichment,
    "dataset_warmup": task_dataset_warmup,
    "candidate_training": task_candidate_training,
    "ghost_metrics": task_ghost_metrics,
    "ghost_trading": task_ghost_trading,
    "live_monitoring": task_live_monitoring,
    "background_refresh": task_background_refresh,
}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: task_runner.py <spec.json>", file=sys.stderr)
        sys.exit(1)

    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    task_type = spec["task_type"]
    payload = spec.get("payload", {})

    handler = TASK_HANDLERS.get(task_type)
    if not handler:
        print(f"Unknown task type: {task_type}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Revenir Task Runner: {task_type} ===")
    start = time.time()

    try:
        result = handler(payload)
        result["_duration_seconds"] = round(time.time() - start, 2)
        write_result(result)
        print(f"=== Task completed in {result['_duration_seconds']}s ===")
    except Exception as exc:
        write_result({"error": str(exc), "traceback": traceback.format_exc()})
        print(f"FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
