from __future__ import annotations

import math
import time
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


def build_snapshot(
    *,
    bot: Any,
    sample: Optional[Dict[str, Any]],
    pred_summary: Optional[Dict[str, Any]],
    brain_summary: Optional[Dict[str, Any]],
    directive: Optional[Any],
    decision: Optional[Dict[str, Any]],
    latency_s: Optional[float],
    latency_window: Optional[Iterable[float]],
    pending_depth: int,
    discovery_snapshot: Optional[Dict[str, Any]],
    last_windows: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Create a JSON-serialisable organism snapshot for dashboards and analytics.
    """
    now = time.time()
    sample_payload = _prepare_sample(sample)
    prediction_payload = _sanitize(pred_summary or {})
    brain_payload = _sanitize(brain_summary or {})
    decision_payload = _sanitize(decision or {})
    directive_payload = directive.to_dict() if directive else None

    positions_payload = _prepare_positions(getattr(bot, "positions", {}))
    exposure_payload = _prepare_exposure(getattr(bot, "active_exposure", {}))
    queue_preview = _prepare_queue_preview(getattr(bot, "queue", []))
    portfolio_payload = _prepare_portfolio(bot)
    pipeline_payload = _prepare_pipeline(bot)
    scheduler_payload = _prepare_scheduler(bot)
    latency_stats = _prepare_latency_stats(latency_window)
    windows_payload = _prepare_windows(last_windows)
    discovery_payload = discovery_snapshot or {}

    graph_metadata, graph_edges = _prepare_graph(bot)
    totals_payload = {
        "stable_bank": float(getattr(bot, "stable_bank", 0.0)),
        "total_profit": float(getattr(bot, "total_profit", 0.0)),
        "realized_profit": float(getattr(bot, "realized_profit", 0.0)),
        "equity": _safe_float(_call_optional(bot, "current_equity")),
        "wins": int(getattr(bot, "wins", 0)),
        "total_trades": int(getattr(bot, "total_trades", 0)),
    }
    mode = "live" if getattr(bot, "live_trading_enabled", False) else "ghost"
    ghost_session = int(getattr(bot, "ghost_session_id", 0) or 0)
    activity_payload = _prepare_activity(bot)
    gas_strategy = _sanitize(getattr(bot, "_last_gas_strategy", None) or {})

    organism_graph = _build_organism_graph(
        brain_payload,
        exposure_payload,
        positions_payload,
        scheduler_payload,
        graph_metadata,
        graph_edges,
        decision_payload,
        queue_preview,
        pending_depth,
        portfolio=portfolio_payload,
        totals=totals_payload,
        discovery=discovery_payload,
        mode=mode,
        ghost_session=ghost_session,
        activity=activity_payload,
        windows=windows_payload,
    )

    snapshot = {
        "timestamp": now,
        "latency_ms": float(latency_s * 1000.0) if latency_s is not None else None,
        "mode": mode,
        "ghost_session": ghost_session,
        "sample": sample_payload,
        "prediction": prediction_payload,
        "brain": brain_payload,
        "directive": directive_payload,
        "decision": decision_payload,
        "positions": positions_payload,
        "exposure": exposure_payload,
        "queue_preview": queue_preview,
        "queue_depth": len(getattr(bot, "queue", [])),
        "pending_samples": int(pending_depth),
        "scheduler": scheduler_payload,
        "portfolio": portfolio_payload,
        "pipeline": pipeline_payload,
        "latency_stats": latency_stats,
        "brain_windows": windows_payload,
        "discovery": discovery_payload,
        "organism_graph": organism_graph,
        "totals": totals_payload,
        "activity": activity_payload,
        "process_clusters": _build_process_clusters(
            brain_payload,
            scheduler_payload,
            exposure_payload,
            activity_payload,
            totals_payload,
            pipeline=pipeline_payload,
        ),
        "gas_strategy": gas_strategy,
    }
    transition_plan = brain_payload.get("transition_plan")
    if isinstance(transition_plan, dict) and transition_plan:
        snapshot["transition_plan"] = transition_plan
    return snapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_sample(sample: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not sample:
        return {}
    keys = ("symbol", "chain", "price", "volume", "ts", "source")
    payload = {}
    for key in keys:
        if key not in sample:
            continue
        if key in {"price", "volume", "ts"}:
            payload[key] = _safe_float(sample.get(key))
        else:
            payload[key] = sample.get(key)
    raw_sentiment = sample.get("sentiment")
    if raw_sentiment is not None:
        payload["sentiment"] = _safe_float(raw_sentiment)
    return payload


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(v) for v in value]
    if hasattr(value, "tolist"):
        return _sanitize(value.tolist())
    if isinstance(value, (int, float)):
        return _safe_float(value)
    if isinstance(value, bool) or value is None:
        return value
    return value


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _prepare_positions(raw_positions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    prepared: Dict[str, Dict[str, Any]] = {}
    for symbol, payload in raw_positions.items():
        prepared[symbol] = {
            "entry_price": _safe_float(payload.get("entry_price")),
            "size": _safe_float(payload.get("size")),
            "entry_ts": _safe_float(payload.get("entry_ts")),
            "route": payload.get("route"),
            "target_price": _safe_float(payload.get("target_price")),
            "trade_id": payload.get("trade_id"),
            "brain_snapshot": _sanitize(payload.get("brain_snapshot")),
        }
        if "fingerprint" in payload:
            prepared[symbol]["fingerprint"] = _sanitize(payload.get("fingerprint"))
    return prepared


def _prepare_exposure(exposure_map: Dict[str, Any]) -> Dict[str, float]:
    return {symbol: _safe_float(val) for symbol, val in exposure_map.items()}


def _prepare_queue_preview(queue: List[Any], limit: int = 6) -> List[Dict[str, Any]]:
    if not queue:
        return []
    tail = queue[-limit:]
    return [_sanitize(item) for item in tail]


def _prepare_portfolio(bot: Any) -> Dict[str, Any]:
    portfolio = getattr(bot, "portfolio", None)
    if portfolio is None:
        return {}
    try:
        summary = portfolio.summary()
    except Exception:
        summary = {}
    holdings: Dict[str, Dict[str, Any]] = {}
    try:
        for (chain, symbol), holding in getattr(portfolio, "holdings", {}).items():
            holdings[f"{chain}:{symbol}"] = {
                "quantity": _safe_float(getattr(holding, "quantity", 0.0)),
                "usd": _safe_float(getattr(holding, "usd", 0.0)),
            }
    except Exception:
        holdings = {}
    return {
        "summary": _sanitize(summary),
        "holdings": holdings,
        "native_balances": {
            chain: _safe_float(balance)
            for chain, balance in getattr(portfolio, "native_balances", {}).items()
        },
    }


def _prepare_pipeline(bot: Any) -> Dict[str, Any]:
    pipeline = getattr(bot, "pipeline", None)
    if pipeline is None:
        return {}
    payload: Dict[str, Any] = {}
    payload["iteration"] = int(getattr(pipeline, "iteration", 0) or 0)
    payload["active_accuracy"] = _safe_float(getattr(pipeline, "active_accuracy", 0.0))
    payload["decision_threshold"] = _safe_float(getattr(pipeline, "decision_threshold", 0.0))
    try:
        payload["horizon_bias"] = _sanitize(pipeline.horizon_bias())
    except Exception:
        payload["horizon_bias"] = {}
    try:
        payload["confusion_summary"] = _sanitize(pipeline.confusion_summary())
    except Exception:
        payload["confusion_summary"] = _sanitize(getattr(pipeline, "_last_confusion_summary", {}))
    payload["transition_plan"] = _sanitize(getattr(bot, "_transition_plan", {}))
    dataset_meta = getattr(pipeline, "_last_dataset_meta", {})
    if isinstance(dataset_meta, dict) and dataset_meta:
        payload["dataset"] = {
            "samples": _safe_float(dataset_meta.get("samples", 0.0)),
            "positive_ratio": _safe_float(dataset_meta.get("positive_ratio", 0.0)),
            "news_coverage_ratio": _safe_float(dataset_meta.get("news_coverage_ratio", 0.0)),
            "horizon_weight_mean": _safe_float(dataset_meta.get("horizon_weight_mean", 0.0)),
            "horizon_weight_min": _safe_float(dataset_meta.get("horizon_weight_min", 0.0)),
            "horizon_weight_max": _safe_float(dataset_meta.get("horizon_weight_max", 0.0)),
            "horizon_deficit": _sanitize(dataset_meta.get("horizon_deficit", {})),
            "horizon_profile": _sanitize(dataset_meta.get("horizon_profile", {})),
        }
    news_items = getattr(getattr(pipeline, "data_loader", None), "news_items", None)
    if isinstance(news_items, list) and news_items:
        source_counts: Dict[str, int] = {}
        for item in news_items:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip()
            if source:
                source_counts[source] = source_counts.get(source, 0) + 1
        payload["news"] = {
            "items": len(news_items),
            "sources": len(source_counts),
            "top_sources": [
                name
                for name, _ in sorted(source_counts.items(), key=lambda entry: entry[1], reverse=True)[:8]
            ],
        }
    candidate_feedback = getattr(pipeline, "_last_candidate_feedback", {})
    if isinstance(candidate_feedback, dict) and candidate_feedback:
        payload["candidate"] = {
            "ghost_trades": _safe_float(candidate_feedback.get("ghost_trades", 0.0)),
            "ghost_win_rate": _safe_float(candidate_feedback.get("ghost_win_rate", 0.0)),
            "ghost_realized_margin": _safe_float(candidate_feedback.get("ghost_realized_margin", 0.0)),
            "ghost_pred_margin": _safe_float(candidate_feedback.get("ghost_pred_margin", 0.0)),
            "false_positive_rate": _safe_float(candidate_feedback.get("false_positive_rate", 0.0)),
            "best_threshold": _safe_float(candidate_feedback.get("best_threshold", 0.0)),
        }
    return payload


def _prepare_scheduler(bot: Any) -> List[Dict[str, Any]]:
    scheduler = getattr(bot, "scheduler", None)
    if scheduler is None:
        return []
    try:
        return _sanitize(scheduler.snapshot())
    except Exception:
        return []


def _prepare_latency_stats(latency_window: Optional[Iterable[float]]) -> Dict[str, float]:
    if not latency_window:
        return {}
    values = [float(val) for val in latency_window if val is not None]
    if not values:
        return {}
    values.sort()
    avg_ms = mean(values) * 1000.0
    index = int(math.ceil(0.95 * len(values))) - 1
    index = max(0, min(index, len(values) - 1))
    p95_ms = values[index] * 1000.0
    return {
        "avg_ms": float(avg_ms),
        "p95_ms": float(p95_ms),
        "count": len(values),
    }


def _prepare_windows(last_windows: Optional[Dict[str, Dict[str, Any]]], limit: int = 120) -> Dict[str, Any]:
    if not last_windows:
        return {}
    output: Dict[str, Any] = {}
    for symbol, payload in last_windows.items():
        entry: Dict[str, Any] = {}
        prices = payload.get("prices") or {}
        sentiments = payload.get("sentiment") or {}
        if prices:
            entry["prices"] = {label: _series_to_list(values, limit=limit) for label, values in prices.items()}
        if sentiments:
            entry["sentiment"] = {label: _series_to_list(values, limit=limit) for label, values in sentiments.items()}
        output[symbol] = entry
    return output


def _prepare_activity(bot: Any) -> Dict[str, Any]:
    db = getattr(bot, "db", None)
    if db is None:
        return {}

    activity: Dict[str, Any] = {
        "ghost_trades": [],
        "live_trades": [],
        "feedback_events": [],
        "metrics": [],
    }

    try:
        ghost_trades = db.fetch_trades(wallets=["ghost"], limit=24)
    except Exception:
        ghost_trades = []
    activity["ghost_trades"] = [
        {
            "ts": _safe_float(trade.get("ts")),
            "wallet": trade.get("wallet"),
            "chain": trade.get("chain"),
            "symbol": trade.get("symbol"),
            "action": trade.get("action"),
            "status": trade.get("status"),
            "details": _sanitize(trade.get("details")),
        }
        for trade in ghost_trades[:24]
        if isinstance(trade, dict)
    ]

    try:
        live_trades = db.fetch_trades(wallets=["live"], limit=24)
    except Exception:
        live_trades = []
    activity["live_trades"] = [
        {
            "ts": _safe_float(trade.get("ts")),
            "wallet": trade.get("wallet"),
            "chain": trade.get("chain"),
            "symbol": trade.get("symbol"),
            "action": trade.get("action"),
            "status": trade.get("status"),
            "details": _sanitize(trade.get("details")),
        }
        for trade in live_trades[:24]
        if isinstance(trade, dict)
    ]

    try:
        feedback_events = db.fetch_feedback_events(limit=24)
    except Exception:
        feedback_events = []
    activity["feedback_events"] = [
        {
            "ts": _safe_float(event.get("ts")),
            "source": event.get("source"),
            "severity": (event.get("severity") or "").lower(),
            "label": event.get("label"),
            "details": _sanitize(event.get("details")),
        }
        for event in feedback_events[:24]
        if isinstance(event, dict)
    ]

    try:
        metrics = db.fetch_metrics(limit=24)
    except Exception:
        metrics = []
    activity["metrics"] = [
        {
            "ts": _safe_float(entry.get("ts")),
            "stage": entry.get("stage"),
            "category": entry.get("category"),
            "name": entry.get("name"),
            "value": _safe_float(entry.get("value")),
            "meta": _sanitize(entry.get("meta")),
        }
        for entry in metrics[:24]
        if isinstance(entry, dict)
    ]

    return activity


def _build_process_clusters(
    brain: Dict[str, Any],
    scheduler: List[Dict[str, Any]],
    exposure: Dict[str, float],
    activity: Dict[str, Any],
    totals: Dict[str, Any],
    pipeline: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    diagnostics = brain.get("swarm_diagnostics") or []
    if diagnostics:
        avg_energy = sum(entry.get("energy", 0.0) for entry in diagnostics) / max(len(diagnostics), 1)
        clusters.append(
            {
                "label": "Swarm",
                "energy": float(avg_energy),
                "nodes": len(diagnostics),
                "detail": diagnostics[:6],
            }
        )
    if scheduler:
        pressure = min(1.0, len(scheduler) / 10.0)
        clusters.append(
            {
                "label": "Scheduler",
                "energy": float(pressure),
                "nodes": len(scheduler),
                "detail": scheduler[:6],
            }
        )
    if exposure:
        gross = sum(abs(val) for val in exposure.values())
        clusters.append(
            {
                "label": "Exposure",
                "energy": float(min(1.0, gross)),
                "nodes": len(exposure),
                "detail": [{"symbol": sym, "value": val} for sym, val in list(exposure.items())[:8]],
            }
        )
    if totals:
        equity = float(totals.get("equity") or 0.0)
        stable = float(totals.get("stable_bank") or 0.0)
        denom = max(1.0, equity + stable)
        clusters.append(
            {
                "label": "Equity",
                "energy": float(min(1.0, (equity + stable) / denom)),
                "nodes": 1,
                "detail": totals,
            }
        )
    if pipeline:
        confusion = pipeline.get("confusion_summary") or {}
        horizon_metrics = confusion.get("horizons") or {}
        sample_total = _safe_float(confusion.get("total_samples", 0.0))
        accuracy = _safe_float(pipeline.get("active_accuracy", 0.0))
        energy = min(1.0, max(accuracy, sample_total / 250.0 if sample_total else 0.0))
        clusters.append(
            {
                "label": "Pipeline",
                "energy": float(max(0.0, energy)),
                "nodes": max(1, len(horizon_metrics) if isinstance(horizon_metrics, dict) else 1),
                "detail": {
                    "iteration": pipeline.get("iteration"),
                    "accuracy": accuracy,
                    "dominant": confusion.get("dominant"),
                },
            }
        )
    if activity:
        clusters.append(
            {
                "label": "Activity",
                "energy": float(min(1.0, len(activity.get("metrics", [])) / 16.0)),
                "nodes": sum(len(activity.get(key, [])) for key in ("ghost_trades", "live_trades", "feedback_events")),
                "detail": {
                    "ghost_trades": activity.get("ghost_trades", [])[:3],
                    "live_trades": activity.get("live_trades", [])[:3],
                },
            }
        )
    transition_plan = brain.get("transition_plan") or {}
    plan_horizons = transition_plan.get("horizons") if isinstance(transition_plan, dict) else {}
    if isinstance(plan_horizons, dict) and plan_horizons:
        coverage = float(transition_plan.get("coverage") or 0.0)
        clusters.append(
            {
                "label": "Horizon Readiness",
                "energy": float(max(0.0, min(1.0, coverage))),
                "nodes": len(plan_horizons),
                "detail": [
                    {
                        "horizon": label,
                        "precision": _safe_float(metrics.get("precision")),
                        "allowed": bool(metrics.get("allowed")),
                    }
                    for label, metrics in list(plan_horizons.items())[:6]
                ],
            }
        )
    return clusters


def _series_to_list(values: Any, *, limit: int) -> List[float]:
    if hasattr(values, "tolist"):
        seq = values.tolist()
    else:
        seq = list(values)
    if len(seq) > limit:
        seq = seq[-limit:]
    return [_safe_float(val) for val in seq]


def _prepare_graph(bot: Any) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    graph = getattr(bot, "graph", None)
    if graph is None:
        return {}, []
    nodes_meta: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    try:
        raw_nodes = graph.nodes_snapshot()
        sorted_nodes = sorted(
            raw_nodes.items(),
            key=lambda item: item[1].get("last_update", 0.0),
            reverse=True,
        )
        # keep the most recent nodes for visual clarity
        for key, meta in sorted_nodes[:48]:
            nodes_meta[key] = {
                "kind": meta.get("kind", "asset"),
                "value": _safe_float(meta.get("value")),
                "last_update": _safe_float(meta.get("last_update")),
                "metadata": _sanitize(meta.get("metadata")),
            }
        adjacency = graph.snapshot()
        for src, dests in adjacency.items():
            if src not in nodes_meta:
                continue
            for dst, weight in dests.items():
                if dst not in nodes_meta:
                    continue
                weight_f = _safe_float(weight)
                if abs(weight_f) < 0.02:
                    continue
                edges.append(
                    {
                        "source": src,
                        "target": dst,
                        "weight": weight_f,
                    }
                )
    except Exception:
        return {}, []
    return nodes_meta, edges


def _build_organism_graph(
    brain: Dict[str, Any],
    exposure: Dict[str, float],
    positions: Dict[str, Dict[str, Any]],
    scheduler: List[Dict[str, Any]],
    graph_nodes: Dict[str, Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
    decision: Dict[str, Any],
    queue_preview: List[Dict[str, Any]],
    pending_depth: int,
    *,
    portfolio: Optional[Dict[str, Any]] = None,
    totals: Optional[Dict[str, Any]] = None,
    discovery: Optional[Dict[str, Any]] = None,
    mode: str = "ghost",
    ghost_session: int = 0,
    activity: Optional[Dict[str, Any]] = None,
    windows: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    portfolio = portfolio or {}
    totals = totals or {}
    discovery = discovery or {}
    activity = activity or {}
    windows = windows or {}
    mode = (mode or "ghost").lower()
    ghost_session = max(0, int(ghost_session or 0))

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    def add_node(node_id: str, **attrs: Any) -> None:
        node = nodes.setdefault(node_id, {"id": node_id})
        node.update(attrs)

    def add_edge(source: str, target: str, *, weight: float, kind: str) -> None:
        edges.append(
            {
                "source": source,
                "target": target,
                "weight": float(weight),
                "kind": kind,
            }
        )

    def ensure_asset(symbol: Any) -> str:
        symbol_str = str(symbol or "?").upper()
        node_id = f"asset:{symbol_str}"
        if node_id not in nodes:
            add_node(node_id, label=symbol_str, group="asset", status="watch", exposure=0.0)
        return node_id

    def pnl_status(value: Optional[float]) -> str:
        if value is None:
            return "idle"
        if value > 0.0001:
            return "strong"
        if value < -0.0001:
            return "cautious" if value > -1.0 else "halted"
        return "soft"

    def trade_status(raw_status: Optional[str]) -> str:
        status = str(raw_status or "").lower()
        if any(flag in status for flag in ("fail", "error", "reject", "cancel")):
            return "halted"
        if any(flag in status for flag in ("pending", "waiting", "queued")):
            return "cautious"
        if any(flag in status for flag in ("fill", "done", "success", "complete", "executed", "filled")):
            return "strong"
        return "soft"

    def severity_status(severity: Optional[str]) -> str:
        level = str(severity or "").lower()
        if level in {"critical", "fatal", "error"}:
            return "halted"
        if level in {"warning", "warn"}:
            return "cautious"
        if level in {"info", "notice"}:
            return "soft"
        return "idle"

    def metric_status(value: Optional[float]) -> str:
        if value is None:
            return "idle"
        if value >= 0.75:
            return "strong"
        if value >= 0.5:
            return "engaged"
        if value >= 0.25:
            return "soft"
        if value < 0:
            return "cautious"
        return "idle"

    def scaled_value(amount: float, *, base: float = 0.35) -> float:
        return max(0.2, min(1.2, math.log1p(abs(amount)) * 0.22 + base))

    brain_status = "ok"
    if brain.get("reflex_block_active"):
        brain_status = "halted"
    elif brain.get("scenario_defer"):
        brain_status = "cautious"
    elif abs(_safe_float(brain.get("swarm_bias"))) < 1e-3:
        brain_status = "idle"

    add_node(
        "brain",
        label="Neural Core",
        group="system",
        status=brain_status,
        value=_safe_float(brain.get("graph_confidence", 1.0)),
    )
    add_node(
        "module:swarm",
        label="Swarm",
        group="module",
        status=_bias_status(_safe_float(brain.get("swarm_bias"))),
        value=_safe_float(brain.get("swarm_bias")),
    )
    add_node(
        "module:memory",
        label="Memory",
        group="module",
        status=_bias_status(_safe_float(brain.get("memory_bias"))),
        value=_safe_float(brain.get("memory_bias")),
    )
    add_node(
        "module:scenario",
        label="Scenario",
        group="module",
        status="halted" if brain.get("scenario_defer") else "ok",
        value=_safe_float(brain.get("scenario_spread")),
    )
    arb_signal = brain.get("arb_signal") or {}
    add_node(
        "module:arb",
        label="Volatility Arb",
        group="module",
        status=_arb_status(arb_signal),
        value=_safe_float(arb_signal.get("implied_edge")),
    )
    add_node(
        "module:scheduler",
        label="Scheduler",
        group="system",
        status="busy" if scheduler else "idle",
        queue=pending_depth,
    )
    add_node(
        "module:queue",
        label="Execution Queue",
        group="system",
        status="busy" if queue_preview else "idle",
        depth=len(queue_preview),
    )

    feedback_events = activity.get("feedback_events") or []
    metrics_events = activity.get("metrics") or []

    add_node(
        "module:feedback",
        label="Feedback Loop",
        group="module",
        status="busy" if feedback_events else "idle",
        value=scaled_value(len(feedback_events), base=0.25),
    )
    add_node(
        "module:metrics",
        label="Metrics Hub",
        group="module",
        status="busy" if metrics_events else "idle",
        value=scaled_value(len(metrics_events), base=0.25),
    )

    holdings_map = portfolio.get("holdings") or {}
    native_balances = portfolio.get("native_balances") or {}
    holdings_total_usd = sum(max(0.0, _safe_float(payload.get("usd"))) for payload in holdings_map.values())
    native_total = sum(max(0.0, _safe_float(amount)) for amount in native_balances.values())
    portfolio_total = holdings_total_usd + native_total
    add_node(
        "module:portfolio",
        label="Wallet Stack",
        group="system",
        status="busy" if portfolio_total > 0 else "idle",
        value=scaled_value(portfolio_total, base=0.4),
    )

    add_edge("brain", "module:swarm", weight=max(0.2, abs(_safe_float(brain.get("swarm_bias")))), kind="signal")
    add_edge("brain", "module:memory", weight=max(0.2, abs(_safe_float(brain.get("memory_bias")))), kind="signal")
    add_edge("brain", "module:scenario", weight=max(0.2, abs(_safe_float(brain.get("scenario_spread")))), kind="signal")
    if arb_signal:
        add_edge("brain", "module:arb", weight=max(0.2, abs(_safe_float(arb_signal.get("spread")))), kind="signal")
    add_edge("brain", "module:feedback", weight=0.6 + len(feedback_events) * 0.05, kind="feedback")
    add_edge("brain", "module:metrics", weight=0.5 + len(metrics_events) * 0.04, kind="metric")
    add_edge("brain", "module:portfolio", weight=0.8, kind="operational")
    add_edge("module:scheduler", "module:queue", weight=float(len(queue_preview) + 1), kind="operational")

    transition_plan = brain.get("transition_plan") or {}
    if isinstance(transition_plan, dict) and transition_plan:
        plan_node = "module:transition"
        add_node(
            plan_node,
            label="Ghost→Live Plan",
            group="module",
            status="strong" if transition_plan.get("live_ready") else "watch",
            value=_safe_float(transition_plan.get("coverage", 0.0)),
        )
        add_edge("brain", plan_node, weight=0.6 + _safe_float(transition_plan.get("coverage", 0.0)), kind="transition")
        plan_horizons = transition_plan.get("horizons")
        if isinstance(plan_horizons, dict):
            for label, metrics in plan_horizons.items():
                node_id = f"transition:{label}"
                add_node(
                    node_id,
                    label=f"{label} • {_safe_float(metrics.get('precision')):.2f}",
                    group="transition",
                    status="engaged" if metrics.get("allowed") else "idle",
                    value=_safe_float(metrics.get("lift", metrics.get("precision", 0.0))),
                )
                add_edge(plan_node, node_id, weight=0.4 + _safe_float(metrics.get("precision", 0.0)), kind="transition")

    add_edge("module:portfolio", "module:scheduler", weight=0.6, kind="operational")

    add_node(
        "session:mode",
        label=f"Mode • {mode.upper()}",
        group="session",
        status="engaged" if mode == "live" else "soft",
        value=1.0 if mode == "live" else 0.7,
    )
    add_node(
        "session:ghost",
        label=f"Ghost Session #{ghost_session or 0}",
        group="session",
        status="soft" if mode == "live" else "engaged",
        value=scaled_value(float(max(1, ghost_session or 1)), base=0.3),
    )
    add_node(
        "session:live",
        label="Live Wallet",
        group="session",
        status="engaged" if mode == "live" else "idle",
        value=0.8 if mode == "live" else 0.4,
    )

    add_edge("session:mode", "brain", weight=1.0, kind="state")
    add_edge("session:mode", "module:scheduler", weight=0.6, kind="session")
    add_edge("session:mode", "session:ghost", weight=0.7, kind="session")
    add_edge("session:mode", "session:live", weight=0.7, kind="session")
    add_edge("session:ghost", "module:queue", weight=0.8, kind="session")
    add_edge("session:live", "module:portfolio", weight=0.8, kind="session")

    swarm_votes = brain.get("swarm_votes") or []
    for vote in swarm_votes[:10]:
        horizon = str(vote.get("horizon") or "?")
        expected = _safe_float(vote.get("expected"))
        confidence = _safe_float(vote.get("confidence"))
        vote_id = f"swarm_vote:{horizon}"
        add_node(
            vote_id,
            label=f"{horizon} • {expected * 100:.1f}% / {confidence * 100:.0f}%",
            group="vote",
            status="strong" if expected > 0 else ("halted" if expected < 0 else "soft"),
            value=scaled_value(confidence, base=0.25),
        )
        add_edge("module:swarm", vote_id, weight=max(0.12, abs(expected)) + confidence * 0.2, kind="vote")

    total_exposure = sum(abs(val) for val in exposure.values()) or 1.0
    for symbol, value in exposure.items():
        node_id = ensure_asset(symbol)
        in_position = symbol in positions
        add_node(
            node_id,
            label=str(symbol),
            group="asset",
            status="position" if in_position else "watch",
            exposure=_safe_float(value),
        )
        weight = max(0.05, abs(value) / total_exposure)
        add_edge("module:scheduler", node_id, weight=weight, kind="exposure")
        if in_position:
            add_edge("module:queue", node_id, weight=weight, kind="position")

    for symbol, payload in positions.items():
        asset_id = ensure_asset(symbol)
        node_id = f"ghost:{symbol}"
        add_node(
            node_id,
            label=f"Ghost {symbol}",
            group="ghost",
            status="position",
            value=scaled_value(_safe_float(payload.get("size")), base=0.28),
        )
        add_edge(asset_id, node_id, weight=0.9, kind="ghost")
        add_edge("session:ghost", node_id, weight=0.7, kind="ghost")

    for idx, queued in enumerate(queue_preview):
        node_id = f"queue_item:{idx}"
        symbol = queued.get("symbol") or queued.get("decision", {}).get("symbol") or "?"
        status = str(queued.get("status") or "pending").lower()
        add_node(
            node_id,
            label=f"Q[{idx}] {symbol}",
            group="queue",
            status=trade_status(status),
            value=scaled_value(_safe_float(queued.get("size") or 1.0), base=0.25),
        )
        add_edge("module:queue", node_id, weight=0.6, kind="queue")
        add_edge(node_id, ensure_asset(symbol), weight=0.4, kind="queue")

    decision_action = decision.get("action")
    if isinstance(decision_action, str) and decision_action.lower() in {"enter", "exit"}:
        symbol = decision.get("symbol")
        if symbol:
            add_node(
                "decision",
                label=f"Decision:{decision_action}",
                group="event",
                status=decision_action.lower(),
            )
            add_edge("brain", "decision", weight=0.9, kind="decision")
            add_edge("decision", ensure_asset(symbol), weight=0.9, kind="decision")

    # Portfolio holdings
    if holdings_map:
        sorted_holdings = sorted(
            holdings_map.items(),
            key=lambda item: _safe_float(item[1].get("usd")),
            reverse=True,
        )
        for key, payload in sorted_holdings[:36]:
            usd = _safe_float(payload.get("usd"))
            quantity = _safe_float(payload.get("quantity"))
            node_id = f"holding:{key}"
            add_node(
                node_id,
                label=f"{key} • {quantity:.4g}",
                group="holding",
                status="position" if usd > 0 else "watch",
                value=scaled_value(usd, base=0.22),
            )
            weight = max(0.05, usd / max(holdings_total_usd, 1.0))
            add_edge("module:portfolio", node_id, weight=weight, kind="holding")
            asset_symbol = key.split(":")[-1]
            add_edge(node_id, ensure_asset(asset_symbol), weight=0.4 + weight, kind="holding")

    if native_balances:
        for chain, amount in list(native_balances.items())[:12]:
            amt_float = _safe_float(amount)
            node_id = f"native:{chain}"
            add_node(
                node_id,
                label=f"{chain.upper()} Native • {amt_float:.4g}",
                group="native",
                status="position" if amt_float > 0 else "idle",
                value=scaled_value(amt_float, base=0.2),
            )
            add_edge("module:portfolio", node_id, weight=0.4 + min(0.6, amt_float / max(native_total, 1.0)), kind="holding")

    # Totals / ledger nodes
    stable_bank = _safe_float(totals.get("stable_bank"))
    total_profit = _safe_float(totals.get("total_profit"))
    realized_profit = _safe_float(totals.get("realized_profit"))
    equity = totals.get("equity")
    wins = int(totals.get("wins") or 0)
    total_trades = int(totals.get("total_trades") or 0)

    ledger_nodes = [
        ("stable_bank", "Stable Bank", stable_bank),
        ("total_profit", "Total Profit", total_profit),
        ("realized", "Realised Profit", realized_profit),
    ]
    if equity is not None:
        ledger_nodes.append(("equity", "Equity", _safe_float(equity)))
    for key, label, value in ledger_nodes:
        node_id = f"finance:{key}"
        add_node(
            node_id,
            label=f"{label} • ${value:,.2f}",
            group="finance",
            status=pnl_status(value),
            value=scaled_value(value, base=0.3),
        )
        add_edge("module:portfolio", node_id, weight=0.9, kind="finance")

    if total_trades > 0:
        win_rate = wins / max(total_trades, 1)
        add_node(
            "finance:winrate",
            label=f"Win Rate • {win_rate * 100:.1f}%",
            group="finance",
            status="strong" if win_rate >= 0.55 else ("cautious" if win_rate < 0.35 else "soft"),
            value=scaled_value(win_rate, base=0.25),
        )
        add_edge("module:portfolio", "finance:winrate", weight=0.6 + win_rate * 0.6, kind="finance")

    # Activity: ghost trades
    ghost_trades = activity.get("ghost_trades") or []
    for idx, trade in enumerate(ghost_trades[:18]):
        symbol = trade.get("symbol") or "?"
        node_id = f"ghost_trade:{idx}"
        size = _safe_float((trade.get("details") or {}).get("size"))
        add_node(
            node_id,
            label=f"Ghost {symbol} • {trade.get('action', '').upper()}",
            group="ghost_trade",
            status=trade_status(trade.get("status")),
            value=scaled_value(size, base=0.22),
        )
        add_edge("session:ghost", node_id, weight=0.6, kind="trade")
        add_edge(node_id, ensure_asset(symbol), weight=0.5, kind="trade")

    live_trades = activity.get("live_trades") or []
    for idx, trade in enumerate(live_trades[:18]):
        symbol = trade.get("symbol") or "?"
        node_id = f"live_trade:{idx}"
        size = _safe_float((trade.get("details") or {}).get("size"))
        add_node(
            node_id,
            label=f"Live {symbol} • {trade.get('action', '').upper()}",
            group="live_trade",
            status=trade_status(trade.get("status")),
            value=scaled_value(size, base=0.22),
        )
        add_edge("session:live", node_id, weight=0.6, kind="trade")
        add_edge(node_id, ensure_asset(symbol), weight=0.5, kind="trade")

    for idx, event in enumerate(feedback_events[:12]):
        node_id = f"feedback:{idx}"
        add_node(
            node_id,
            label=f"{event.get('label') or event.get('source')}",
            group="feedback",
            status=severity_status(event.get("severity")),
            value=scaled_value(1 + idx * 0.1, base=0.21),
        )
        add_edge("module:feedback", node_id, weight=0.6, kind="feedback")

    for idx, metric in enumerate(metrics_events[:12]):
        node_id = f"metric:{idx}"
        label = f"{metric.get('stage') or metric.get('category')}: {metric.get('name')}"
        add_node(
            node_id,
            label=label,
            group="metric",
            status=metric_status(metric.get("value")),
            value=scaled_value(_safe_float(metric.get("value")), base=0.2),
        )
        add_edge("module:metrics", node_id, weight=0.5, kind="metric")

    discovery_events = discovery.get("recent_events") or []
    total_liquidity = sum(max(0.0, _safe_float(evt.get("liquidity_usd"))) for evt in discovery_events)
    for idx, evt in enumerate(discovery_events[:16]):
        symbol = evt.get("symbol") or "?"
        change = _safe_float(evt.get("price_change_24h"))
        liquidity = _safe_float(evt.get("liquidity_usd"))
        node_id = f"discovery:event:{idx}"
        add_node(
            node_id,
            label=f"{symbol} • {change:+.1f}%",
            group="discovery",
            status="strong" if change > 0 else ("cautious" if change < 0 else "soft"),
            value=scaled_value(abs(change), base=0.22),
        )
        weight = max(0.08, liquidity / max(total_liquidity, 1.0))
        add_edge("module:arb", node_id, weight=weight, kind="discovery")
        add_edge(node_id, ensure_asset(symbol), weight=0.5, kind="discovery")

    honeypots = discovery.get("recent_honeypots") or []
    for idx, evt in enumerate(honeypots[:8]):
        symbol = evt.get("symbol") or "?"
        node_id = f"discovery:honeypot:{idx}"
        add_node(
            node_id,
            label=f"Honeypot • {symbol}",
            group="discovery",
            status="halted",
            value=0.6,
        )
        add_edge(node_id, ensure_asset(symbol), weight=0.4, kind="discovery")
        add_edge("module:arb", node_id, weight=0.4, kind="discovery")

    status_counts = discovery.get("status_counts") or {}
    for status_key, total in list(status_counts.items())[:6]:
        node_id = f"discovery:status:{status_key}"
        add_node(
            node_id,
            label=f"{status_key} • {total}",
            group="discovery",
            status="soft",
            value=scaled_value(float(total), base=0.2),
        )
        add_edge(node_id, "module:arb", weight=0.4 + min(0.6, float(total) / 10.0), kind="discovery")

    if windows:
        for idx, (symbol, payload) in enumerate(windows.items()):
            if idx >= 24:
                break
            asset_id = ensure_asset(symbol)
            price_series = payload.get("prices") or {}
            sentiment_series = payload.get("sentiment") or {}
            volatility = 0.0
            for series in list(price_series.values())[:4]:
                if series:
                    volatility = max(volatility, abs(max(series) - min(series)))
            window_id = f"window:{symbol}"
            add_node(
                window_id,
                label=f"{symbol} signals",
                group="window",
                status="soft",
                value=scaled_value(volatility or 0.2, base=0.24),
            )
            add_edge("module:memory", window_id, weight=0.45 + min(0.55, volatility), kind="memory")
            add_edge(window_id, asset_id, weight=0.35 + min(0.35, volatility), kind="memory")
            for series_name, series in list(price_series.items())[:4]:
                fluct = 0.0
                if series:
                    fluct = abs(max(series) - min(series))
                series_id = f"window:{symbol}:price:{series_name}"
                add_node(
                    series_id,
                    label=f"{series_name.upper()} Δ{fluct:.4f}",
                    group="window_series",
                    status="soft",
                    value=scaled_value(fluct or 0.1, base=0.18),
                )
                add_edge(window_id, series_id, weight=0.4 + min(0.6, fluct), kind="memory")
            for series_name, series in list(sentiment_series.items())[:3]:
                sentiment_mean = float(sum(series) / len(series)) if series else 0.0
                series_id = f"window:{symbol}:sent:{series_name}"
                add_node(
                    series_id,
                    label=f"Sent {series_name} • {sentiment_mean:+.3f}",
                    group="window_series",
                    status="soft",
                    value=scaled_value(abs(sentiment_mean) or 0.05, base=0.16),
                )
                add_edge(window_id, series_id, weight=0.3 + min(0.5, abs(sentiment_mean)), kind="memory")

    # include condensed neuro graph nodes
    for node_key, meta in graph_nodes.items():
        graph_id = f"graph:{node_key}"
        add_node(
            graph_id,
            label=node_key,
            group=f"graph:{meta.get('kind', 'asset')}",
            status="graph",
            value=_safe_float(meta.get("value")),
        )

    for edge in graph_edges:
        src = f"graph:{edge['source']}"
        dst = f"graph:{edge['target']}"
        if src in nodes and dst in nodes:
            add_edge(src, dst, weight=max(0.12, abs(_safe_float(edge.get("weight")))), kind="neuro")

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
    }


def _bias_status(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 0.02:
        return "idle"
    if magnitude < 0.08:
        return "soft"
    if magnitude < 0.15:
        return "engaged"
    return "strong"


def _arb_status(arb_signal: Dict[str, Any]) -> str:
    action = (arb_signal or {}).get("action")
    if not action:
        return "idle"
    action = str(action).lower()
    if action in {"enter", "accumulate", "long"}:
        return "long"
    if action in {"exit", "distribute", "short"}:
        return "short"
    return "watch"


def _call_optional(obj: Any, method: str) -> Optional[float]:
    func = getattr(obj, method, None)
    if callable(func):
        try:
            return func()
        except Exception:
            return None
    return None
