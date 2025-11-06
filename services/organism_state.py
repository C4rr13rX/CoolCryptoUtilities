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
    scheduler_payload = _prepare_scheduler(bot)
    latency_stats = _prepare_latency_stats(latency_window)
    windows_payload = _prepare_windows(last_windows)
    discovery_payload = discovery_snapshot or {}

    graph_metadata, graph_edges = _prepare_graph(bot)
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
    )

    snapshot = {
        "timestamp": now,
        "latency_ms": float(latency_s * 1000.0) if latency_s is not None else None,
        "mode": "live" if getattr(bot, "live_trading_enabled", False) else "ghost",
        "ghost_session": getattr(bot, "ghost_session_id", 0),
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
        "latency_stats": latency_stats,
        "brain_windows": windows_payload,
        "discovery": discovery_payload,
        "organism_graph": organism_graph,
        "totals": {
            "stable_bank": float(getattr(bot, "stable_bank", 0.0)),
            "total_profit": float(getattr(bot, "total_profit", 0.0)),
            "realized_profit": float(getattr(bot, "realized_profit", 0.0)),
            "equity": _safe_float(_call_optional(bot, "current_equity")),
            "wins": int(getattr(bot, "wins", 0)),
            "total_trades": int(getattr(bot, "total_trades", 0)),
        },
    }
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
) -> Dict[str, Any]:
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

    add_edge("brain", "module:swarm", weight=abs(_safe_float(brain.get("swarm_bias"))), kind="signal")
    add_edge("brain", "module:memory", weight=abs(_safe_float(brain.get("memory_bias"))), kind="signal")
    add_edge("brain", "module:scenario", weight=abs(_safe_float(brain.get("scenario_spread"))), kind="signal")
    if arb_signal:
        add_edge("brain", "module:arb", weight=abs(_safe_float(arb_signal.get("spread"))), kind="signal")
    add_edge("module:scheduler", "module:queue", weight=float(len(queue_preview) + 1), kind="operational")

    total_exposure = sum(abs(val) for val in exposure.values()) or 1.0
    for symbol, value in exposure.items():
        node_id = f"asset:{symbol}"
        in_position = symbol in positions
        add_node(
            node_id,
            label=symbol,
            group="asset",
            status="position" if in_position else "watch",
            exposure=_safe_float(value),
        )
        weight = abs(value) / total_exposure
        add_edge("module:scheduler", node_id, weight=weight, kind="exposure")
        if in_position:
            add_edge("module:queue", node_id, weight=weight, kind="position")

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
            add_edge("brain", "decision", weight=1.0, kind="decision")
            add_edge("decision", f"asset:{symbol}", weight=1.0, kind="decision")

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
            add_edge(src, dst, weight=abs(_safe_float(edge.get("weight"))), kind="neuro")

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
