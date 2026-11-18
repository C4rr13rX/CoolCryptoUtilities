from __future__ import annotations

import os
import re
import sys
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict

from django.db.models import Count
from rest_framework import generics, status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Advisory, FeedbackEvent, MetricEntry, TradeLog
from .serializers import (
    AdvisorySerializer,
    FeedbackEventSerializer,
    MetricEntrySerializer,
    TradeLogSerializer,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from db import get_db  # noqa: E402
from services.wallet_reconciliation import (  # noqa: E402
    reconciled_wallet_snapshot,
    request_wallet_refresh,
)
from services.trading_accounting import trading_accounting_snapshot  # noqa: E402


#: A report older than this is not a reading, it is a memory. The dashboard
#: must say so rather than present it as current state.
REPORT_STALE_AFTER_SEC = float(os.getenv("TELEMETRY_REPORT_STALE_SEC", "600"))


def _load_report(path: Path) -> Dict[str, Any]:
    """Load a report file, stamped with its own freshness.

    Returning a bare dict made an absent or hours-old file indistinguishable
    from a live reading: every consumer did ``.get(key, False)``, so a missing
    file silently became "not ready" and the Bus Scheduler displayed
    "Ghost Lane HALTED" while ghost trading was running normally.

    ``_stale`` / ``_age_sec`` let a caller tell "we know this is false" apart
    from "we do not have a value", which is the difference between a status and
    a guess.
    """
    if not path.exists():
        return {"_missing": True, "_stale": True, "_age_sec": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"_unreadable": True, "_stale": True, "_age_sec": None}
    if not isinstance(payload, dict):
        return {"_unreadable": True, "_stale": True, "_age_sec": None}
    try:
        age = max(0.0, time.time() - path.stat().st_mtime)
    except Exception:
        age = None
    payload["_age_sec"] = age
    payload["_stale"] = bool(age is None or age > REPORT_STALE_AFTER_SEC)
    return payload


_UUID_FLOOD_RE = re.compile(r"(?:^discovered_edgeforge_[0-9a-f]{6,}_[0-9a-f]{4,}|_sess_[0-9a-f]{4,})", re.IGNORECASE)


def _is_flood_strategy_id(sid: str) -> bool:
    return bool(_UUID_FLOOD_RE.search(str(sid or "")))


_LIVE_SWAPS_CACHE_LOCK = threading.RLock()
_LIVE_SWAPS_CACHE: list[Dict[str, Any]] = []
_LIVE_SWAPS_CACHE_TIME: float = 0.0
_ACCOUNTING_CACHE: Dict[str, Any] = {}
_ACCOUNTING_CACHE_TIME: float = 0.0
_SNAPSHOT_CACHE: Dict[str, Any] = {}
_SNAPSHOT_CACHE_TIME: float = 0.0
_STATE_POSITIONS_CACHE: Dict[str, Any] = {}
_STATE_POSITIONS_CACHE_TIME: float = 0.0
_BUS_SCHEDULE_PAYLOAD_CACHE: Dict[str, Any] = {}
_BUS_SCHEDULE_PAYLOAD_TIME: float = 0.0
_BUS_SCHEDULE_REFRESHING = threading.Lock()


def _resolve_db_file(db: Any = None) -> Path:
    if db is not None and getattr(db, "path", None):
        return Path(str(getattr(db, "path")))
    if db is not None and getattr(db, "db_path", None):
        return Path(str(getattr(db, "db_path")))
    env_path = os.getenv("TRADING_DB_PATH")
    if env_path:
        return Path(env_path)
    return ROOT / "storage" / "trading_cache.db"


class _RoAccountingDb:
    """Read-only SQLite adapter for trading_accounting_snapshot without get_db()._lock contention."""

    def __init__(self, db_file: Path) -> None:
        self._uri = f"file:{db_file.as_posix()}?mode=ro"

    def _connect(self):
        import sqlite3
        con = sqlite3.connect(self._uri, uri=True, timeout=1.5)
        con.row_factory = sqlite3.Row
        return con

    def load_state(self) -> Dict[str, Any]:
        con = self._connect()
        try:
            row = con.execute("SELECT value FROM kv_store WHERE key = 'state'").fetchone()
            return json.loads(row[0]) if row and row[0] else {}
        except Exception:
            return {}
        finally:
            con.close()

    def trade_outcome_summary(self, wallet: str) -> Dict[str, Any]:
        con = self._connect()
        try:
            row = con.execute(
                """
                SELECT COUNT(*) AS closed,
                       COALESCE(SUM(gross_profit), 0.0) AS gross_profit,
                       COALESCE(SUM(net_profit), 0.0) AS net_profit,
                       COALESCE(SUM(checkpoint), 0.0) AS checkpoint,
                       COALESCE(SUM(fee_cost), 0.0) AS fees,
                       COALESCE(SUM(CASE WHEN net_profit > 0 THEN 1 ELSE 0 END), 0) AS profitable,
                       COALESCE(SUM(CASE WHEN net_profit <= 0 THEN 1 ELSE 0 END), 0) AS unprofitable,
                       MAX(ts) AS updated_at
                FROM trade_outcomes
                WHERE wallet=? AND status='closed'
                """,
                (str(wallet),),
            ).fetchone()
            payload = dict(row or {})
            closed = int(payload.get("closed") or 0)
            profitable = int(payload.get("profitable") or 0)
            fees = float(payload.get("fees") or 0.0)
            return {
                "closed": closed,
                "profitable": profitable,
                "unprofitable": int(payload.get("unprofitable") or 0),
                "gross_profit": float(payload.get("gross_profit") or 0.0),
                "fee_cost": fees,
                "fees": fees,
                "net_profit": float(payload.get("net_profit") or 0.0),
                "checkpoint": float(payload.get("checkpoint") or 0.0),
                "win_rate": profitable / closed if closed else 0.0,
                "updated_at": payload.get("updated_at"),
            }
        finally:
            con.close()

    def fetch_trade_outcomes(self, limit: int = 20) -> list[Dict[str, Any]]:
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT * FROM trade_outcomes ORDER BY ts DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            out = []
            for r in rows:
                item = dict(r)
                if isinstance(item.get("details"), str):
                    try:
                        item["details"] = json.loads(item["details"] or "{}")
                    except Exception:
                        pass
                out.append(item)
            return out
        except Exception:
            return []
        finally:
            con.close()


def _ro_sqlite_json_kv(db: Any, key: str, default: Any) -> Any:
    """Read a kv_store JSON entry via a non-blocking read-only SQLite URI."""
    try:
        import sqlite3
        db_file = _resolve_db_file(db)
        if not db_file.exists():
            return (db.get_json(key) if db is not None else None) or default
        con = sqlite3.connect(f"file:{db_file.as_posix()}?mode=ro", uri=True, timeout=1.5)
        try:
            row = con.execute("SELECT value FROM kv_store WHERE key = ?", (key,)).fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return default
        finally:
            con.close()
    except Exception:
        return default


def _cached_latest_snapshot(db: Any = None) -> Dict[str, Any]:
    global _SNAPSHOT_CACHE, _SNAPSHOT_CACHE_TIME
    now = time.time()
    with _LIVE_SWAPS_CACHE_LOCK:
        if (now - _SNAPSHOT_CACHE_TIME) < 3.0 and _SNAPSHOT_CACHE:
            return _SNAPSHOT_CACHE
    snap: Dict[str, Any] = {}
    try:
        import sqlite3
        db_file = _resolve_db_file(db)
        if db_file.exists():
            con = sqlite3.connect(f"file:{db_file.as_posix()}?mode=ro", uri=True, timeout=1.5)
            try:
                rows = con.execute("SELECT ts, payload FROM organism_snapshots ORDER BY ts DESC LIMIT 20").fetchall()
                merged_sched: list[Dict[str, Any]] = []
                seen_syms: set[str] = set()
                merged_pos: Dict[str, Any] = {}
                newest_ts = float(rows[0][0]) if rows and rows[0][0] else 0.0
                for idx_r, (r_ts, r_payload) in enumerate(rows):
                    if not r_payload:
                        continue
                    if idx_r > 0 and newest_ts > 0 and (newest_ts - float(r_ts or 0.0)) > 600.0:
                        break
                    parsed = json.loads(r_payload)
                    if not isinstance(parsed, dict):
                        continue
                    if idx_r == 0:
                        snap = dict(parsed)
                    for s_entry in (parsed.get("scheduler") or []):
                        if isinstance(s_entry, dict):
                            sym_u = str(s_entry.get("symbol") or "").upper()
                            if sym_u and sym_u not in seen_syms:
                                seen_syms.add(sym_u)
                                merged_sched.append(s_entry)
                    for p_sym, p_val in (parsed.get("positions") or {}).items():
                        p_u = str(p_sym or "").upper()
                        if p_u and p_u not in merged_pos:
                            merged_pos[p_u] = p_val
                if snap:
                    snap["scheduler"] = merged_sched
                    if merged_pos:
                        snap["positions"] = merged_pos
            finally:
                con.close()
        elif db is not None:
            snap = db.fetch_latest_organism_snapshot() or {}
    except Exception:
        snap = dict(_SNAPSHOT_CACHE) if _SNAPSHOT_CACHE else {}
    if isinstance(snap, dict) and snap:
        with _LIVE_SWAPS_CACHE_LOCK:
            _SNAPSHOT_CACHE = snap
            _SNAPSHOT_CACHE_TIME = time.time()
    return snap if isinstance(snap, dict) else {}


def _cached_state_positions(db: Any = None) -> Dict[str, Any]:
    global _STATE_POSITIONS_CACHE, _STATE_POSITIONS_CACHE_TIME
    now = time.time()
    with _LIVE_SWAPS_CACHE_LOCK:
        if (now - _STATE_POSITIONS_CACHE_TIME) < 5.0:
            return dict(_STATE_POSITIONS_CACHE)
    out: Dict[str, Any] = {}
    try:
        saved = _ro_sqlite_json_kv(db, "state", {})
        gt = saved.get("ghost_trading") if isinstance(saved, dict) else {}
        raw_state_positions = gt.get("positions") if isinstance(gt, dict) else {}
        if isinstance(raw_state_positions, dict):
            for k, v in raw_state_positions.items():
                if isinstance(v, dict):
                    out[str(v.get("symbol") or k or "").upper()] = v
    except Exception:
        out = {}
    with _LIVE_SWAPS_CACHE_LOCK:
        _STATE_POSITIONS_CACHE = dict(out)
        _STATE_POSITIONS_CACHE_TIME = time.time()
    return out


def _cached_accounting(db: Any = None) -> Dict[str, Any]:
    global _ACCOUNTING_CACHE, _ACCOUNTING_CACHE_TIME
    now = time.time()
    with _LIVE_SWAPS_CACHE_LOCK:
        if (now - _ACCOUNTING_CACHE_TIME) < 10.0 and _ACCOUNTING_CACHE:
            return _ACCOUNTING_CACHE
    db_file = _resolve_db_file(db)
    ro_db = _RoAccountingDb(db_file) if db_file.exists() else (db or get_db())
    snap = trading_accounting_snapshot(ro_db)
    with _LIVE_SWAPS_CACHE_LOCK:
        _ACCOUNTING_CACHE = snap
        _ACCOUNTING_CACHE_TIME = time.time()
    return snap


def _recent_live_swaps_fast(db: Any = None) -> list[Dict[str, Any]]:
    global _LIVE_SWAPS_CACHE, _LIVE_SWAPS_CACHE_TIME
    now_ts = time.time()
    with _LIVE_SWAPS_CACHE_LOCK:
        if (now_ts - _LIVE_SWAPS_CACHE_TIME) < 10.0:
            return list(_LIVE_SWAPS_CACHE)
    rows_out: list[Dict[str, Any]] = []
    try:
        import sqlite3
        db_file = _resolve_db_file(db).as_posix()
        con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True, timeout=2.0)
        con.row_factory = sqlite3.Row
        try:
            raw_rows = []
            for st in ("live-entry", "live-exit", "live-swap-settled"):
                raw_rows.extend(
                    con.execute(
                        "SELECT ts, chain, symbol, action, status, details "
                        "FROM trading_ops WHERE status = ? AND ts >= ? "
                        "ORDER BY ts DESC LIMIT 4",
                        (st, now_ts - 86400.0),
                    ).fetchall()
                )
            raw_rows.sort(key=lambda x: float(x["ts"] or 0.0), reverse=True)
            for r in raw_rows[:8]:
                try:
                    det = json.loads(r["details"] or "{}")
                except Exception:
                    det = {}
                txh = str(det.get("tx_hash") or "")
                if len(txh) == 66 and txh.startswith("0x"):
                    rows_out.append({
                        "ts": float(r["ts"]),
                        "chain": str(r["chain"] or "base"),
                        "symbol": str(det.get("symbol") or r["symbol"] or "").upper(),
                        "action": str(r["action"] or ""),
                        "status": str(r["status"] or ""),
                        "tx_hash": txh,
                        "strategy_id": det.get("strategy_id") or None,
                        "size": _safe_float(det.get("size")),
                        "price": _safe_float(det.get("entry_price") or det.get("exit_price")),
                        "quote_spent": _safe_float(det.get("quote_spent") or det.get("amount_human")),
                        "horizon": det.get("horizon") or "10m",
                        "confidence": _safe_float(det.get("entry_confidence") or det.get("confidence"), 0.8),
                        "route": det.get("route_used") or det.get("route") or "UniswapV3",
                    })
        finally:
            con.close()
    except Exception:
        rows_out = []
    with _LIVE_SWAPS_CACHE_LOCK:
        _LIVE_SWAPS_CACHE = list(rows_out)
        _LIVE_SWAPS_CACHE_TIME = time.time()
    return rows_out


_STRATEGY_READINESS_CACHE_LOCK = threading.RLock()
_STRATEGY_READINESS_CACHE: Dict[str, Any] = {}
_STRATEGY_READINESS_CACHE_TIME: float = 0.0


def strategy_readiness(include_lifetime: bool = False) -> Dict[str, Any]:
    """Per-strategy ghost/live readiness for the dashboard (cached 15s).

    Readiness is a PER-STRATEGY fact, but the UI was rendering one aggregate
    model-accuracy number. So the Pipeline page showed
    "Ghost Ready: insufficient_accuracy" and "Live Ready: insufficient_accuracy"
    while atf_static was ghost-ready on positive expectancy and approved for
    live -- the screen contradicted the system it was reporting on.

    If ANY strategy is ghost-ready the ghost stage is ready; if ANY strategy is
    live-approved the live stage is ready, and this names which ones rather
    than collapsing them into a single misleading verdict.
    """
    global _STRATEGY_READINESS_CACHE, _STRATEGY_READINESS_CACHE_TIME
    now = time.time()
    with _STRATEGY_READINESS_CACHE_LOCK:
        if (now - _STRATEGY_READINESS_CACHE_TIME) < 15.0 and _STRATEGY_READINESS_CACHE:
            return dict(_STRATEGY_READINESS_CACHE)
    ghost_ready: list[str] = []
    live_ready: list[str] = []
    rows: list[Dict[str, Any]] = []
    try:
        from trading.strategies.ledger import StrategyLedger

        ledger = StrategyLedger()
        data = dict(ledger._data or {})
    except Exception:
        data = {}
    strategy_registry = None
    if include_lifetime:
        try:
            from services import strategy_registry
        except Exception:
            strategy_registry = None

    for sid, entry in sorted(data.items()):
        if _is_flood_strategy_id(sid):
            continue
        ghost = entry.get("ghost") or {}
        live = entry.get("live") or {}
        trades = int(ghost.get("trades", 0) or 0)
        wins = int(ghost.get("wins", 0) or 0)
        approved = bool(entry.get("live_approved"))
        # "Collecting" is a real state and must not read as an error.
        is_ghost_ready = trades > 0
        if is_ghost_ready:
            ghost_ready.append(sid)
        if approved:
            live_ready.append(sid)
        lifetime = {}
        if strategy_registry is not None:
            try:
                lifetime = strategy_registry.lifetime_metrics(sid)
            except Exception:
                lifetime = {}
        rows.append({
            "strategy_id": sid,
            "ghost_trades": trades,
            "ghost_wins": wins,
            "ghost_win_rate": (wins / trades) if trades else None,
            "ghost_profit": ghost.get("total_profit"),
            "live_approved": approved,
            "live_trades": int(live.get("trades", 0) or 0),
            "live_profit": live.get("total_profit"),
            "lifetime": lifetime,
        })
    res = {
        "strategies": rows,
        "ghost_ready_ids": ghost_ready,
        "live_ready_ids": live_ready,
        "ghost_ready": bool(ghost_ready),
        "live_ready": bool(live_ready),
        "is_live_trading": any(r["live_trades"] > 0 for r in rows),
        "counts": {
            "total": len(rows),
            "ghost_ready": len(ghost_ready),
            "live_ready": len(live_ready),
        },
    }
    with _STRATEGY_READINESS_CACHE_LOCK:
        _STRATEGY_READINESS_CACHE = dict(res)
        _STRATEGY_READINESS_CACHE_TIME = time.time()
    return res


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# Units for parsing a raw horizon ("45m", "1h", "1w", or seconds) to seconds.
_HORIZON_UNIT_SEC = {
    "s": 1, "sec": 1, "m": 60, "min": 60, "h": 3600, "hr": 3600, "hour": 3600,
    "d": 86400, "day": 86400, "w": 604800, "week": 604800,
    "mo": 2592000, "month": 2592000, "y": 31536000, "yr": 31536000,
}
_HORIZON_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(mo|min|sec|hr|hour|day|week|month|yr|s|m|h|d|w|y)$")


def _horizon_seconds(raw: Any) -> "float | None":
    """Best-effort parse of a horizon value into seconds; None if not a duration."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if raw > 0 else None
    match = _HORIZON_RE.match(str(raw).strip().lower())
    if not match:
        return None
    return float(match.group(1)) * _HORIZON_UNIT_SEC.get(match.group(2), 0)


def _horizon_label(raw: Any) -> "str | None":
    """Canonical sell-high timeframe chip: 45MIN / 1H / 12H / 1D / 1W / 1M(onth).

    Months use 'M', minutes use 'MIN' so the two never collide. Non-duration
    horizons (e.g. 'atf_quote_scout') pass through uppercased so the lane still
    labels them meaningfully.
    """
    sec = _horizon_seconds(raw)
    if sec is None:
        if raw in (None, "", "open"):
            return None
        return str(raw).replace("_", " ").upper()
    if sec < 3600:
        return f"{int(round(sec / 60))}MIN"
    if sec < 86400:
        return f"{int(round(sec / 3600))}H"
    if sec < 604800:
        return f"{int(round(sec / 86400))}D"
    if sec < 2592000:
        return f"{int(round(sec / 604800))}W"
    if sec < 31536000:
        return f"{int(round(sec / 2592000))}M"
    return f"{int(round(sec / 31536000))}Y"


class MetricsListView(generics.ListAPIView):
    serializer_class = MetricEntrySerializer

    def get_queryset(self):
        get_db()
        qs = MetricEntry.objects.all()
        stage = self.request.query_params.get("stage")
        category = self.request.query_params.get("category")
        if stage:
            qs = qs.filter(stage=stage)
        if category:
            qs = qs.filter(category=category)
        limit = int(self.request.query_params.get("limit", "200"))
        limit = max(1, min(limit, 1000))
        return qs.order_by("-ts")[:limit]


class FeedbackListView(generics.ListAPIView):
    serializer_class = FeedbackEventSerializer

    def get_queryset(self):
        get_db()
        qs = FeedbackEvent.objects.all()
        sources = self.request.query_params.getlist("source")
        severity = self.request.query_params.getlist("severity")
        if sources:
            qs = qs.filter(source__in=sources)
        if severity:
            qs = qs.filter(severity__in=[lvl.lower() for lvl in severity])
        limit = int(self.request.query_params.get("limit", "200"))
        limit = max(1, min(limit, 1000))
        return qs.order_by("-ts")[:limit]


class TradeLogView(generics.ListAPIView):
    serializer_class = TradeLogSerializer

    def get_queryset(self):
        get_db()
        qs = TradeLog.objects.all()
        wallet = self.request.query_params.get("wallet")
        status_param = self.request.query_params.get("status")
        if wallet:
            qs = qs.filter(wallet=wallet)
        if status_param:
            qs = qs.filter(status=status_param)
        limit = int(self.request.query_params.get("limit", "200"))
        limit = max(1, min(limit, 1000))
        return qs.order_by("-ts")[:limit]


class AdvisoryListView(generics.ListAPIView):
    serializer_class = AdvisorySerializer

    def get_queryset(self):
        get_db()
        qs = Advisory.objects.all()
        include_resolved = self.request.query_params.get("include_resolved")
        if not (include_resolved and include_resolved.lower() in {"1", "true", "yes"}):
            qs = qs.filter(resolved=False)
        severity = self.request.query_params.getlist("severity")
        if severity:
            qs = qs.filter(severity__in=[lvl.lower() for lvl in severity])
        limit = int(self.request.query_params.get("limit", "200"))
        limit = max(1, min(limit, 500))
        return qs.order_by("-ts")[:limit]


def build_dashboard_summary(*, request_refresh: bool = True) -> Dict[str, Any]:
    """Build one coherent app revision for both HTTP and WebSocket clients."""
    db = get_db()
    metric_counts = MetricEntry.objects.values("stage").annotate(total=Count("id"))
    feedback_counts = FeedbackEvent.objects.values("severity").annotate(total=Count("id"))
    advisory_counts = Advisory.objects.filter(resolved=False).values("severity").annotate(total=Count("id"))
    latest_metrics = MetricEntrySerializer(MetricEntry.objects.order_by("-ts")[:12], many=True).data
    latest_feedback = FeedbackEventSerializer(FeedbackEvent.objects.order_by("-ts")[:10], many=True).data
    recent_trades = TradeLogSerializer(TradeLog.objects.order_by("-ts")[:10], many=True).data
    active_advisories = AdvisorySerializer(
        Advisory.objects.filter(resolved=False).order_by("-ts")[:10],
        many=True,
    ).data
    state = db.load_state() or {}
    raw_ghost_state = state.get("ghost_trading") or {}
    accounting = trading_accounting_snapshot(db)
    ghost_acc = accounting["ghost"]
    stable_bank = float(ghost_acc.get("checkpoint", 0.0))
    total_profit = float(ghost_acc.get("net_profit", 0.0))
    ghost_state = dict(raw_ghost_state) if isinstance(raw_ghost_state, dict) else {}
    ghost_state.update(
        {
            "initial_bankroll": float(ghost_acc.get("initial_bankroll", 25.0)),
            "balance": float(ghost_acc.get("balance", 0.0)),
            "sim_quote_balances": dict(ghost_acc.get("sim_quote_balances") or {}),
            "active_exposure": dict(ghost_acc.get("active_exposure") or {}),
            "exposure": float(ghost_acc.get("exposure", 0.0)),
            "gross_profit": float(ghost_acc.get("gross_profit", 0.0)),
            "fee_cost": float(ghost_acc.get("fee_cost", 0.0)),
            "net_profit": total_profit,
            "total_profit": total_profit,
            "realized_profit": total_profit,
            "stable_bank": stable_bank,
            "total_trades": int(ghost_acc.get("closed", 0)),
            "wins": int(ghost_acc.get("profitable", 0)),
        }
    )

    live_readiness = _load_report(ROOT / "data/reports/live_readiness.json")
    if not isinstance(live_readiness, dict):
        live_readiness = {}
    try:
        strategies = strategy_readiness()
        live_readiness["ghost_ready_strategies"] = strategies["ghost_ready_ids"]
        live_readiness["live_ready_strategies"] = strategies["live_ready_ids"]
        live_readiness["ghost_ready_any"] = strategies["ghost_ready"]
        live_readiness["live_ready_any"] = strategies["live_ready"]
        live_readiness["is_live_trading"] = strategies["is_live_trading"]
        if strategies["ghost_ready"] or strategies["live_ready"]:
            live_readiness["_stale"] = False
            live_readiness["_missing"] = False
    except Exception:
        pass
    snapshot = db.fetch_latest_organism_snapshot() or {}
    transition = snapshot.get("transition_plan") if isinstance(snapshot, dict) else {}
    if not transition and isinstance(snapshot, dict):
        pipeline_snapshot = snapshot.get("pipeline") or {}
        if isinstance(pipeline_snapshot, dict):
            transition = pipeline_snapshot.get("transition_plan") or {}
    if not transition:
        confusion = _load_report(ROOT / "data/reports/confusion_matrices.json")
        transition = confusion.get("transition_plan") if isinstance(confusion, dict) else {}
    transition = transition if isinstance(transition, dict) else {}
    capital_plan = transition.get("capital_plan") if isinstance(transition, dict) else {}
    capital_plan = capital_plan if isinstance(capital_plan, dict) else {}
    funding_gate = capital_plan.get("funding_gate") or (
        transition.get("risk_flags", {}).get("funding_gate")
        if isinstance(transition.get("risk_flags"), dict) else {}
    )
    wallet_name = str((transition.get("wallet_state") or {}).get("wallet") or "guardian")
    wallet_snapshot = reconciled_wallet_snapshot(wallet_name)
    if request_refresh and not wallet_snapshot.get("fresh"):
        request_wallet_refresh()
    snapshot_ts = _safe_float(snapshot.get("timestamp")) if isinstance(snapshot, dict) else 0.0
    readiness_ts = _safe_float(live_readiness.get("updated_at")) if live_readiness else 0.0
    operational_state = {
            "revision": f"{int(max(snapshot_ts, readiness_ts))}:{len(recent_trades)}:{len(active_advisories)}",
            "generated_at": time.time(),
            "source_timestamps": {
                "organism": snapshot_ts or None,
                "readiness": readiness_ts or None,
            },
            "mode": snapshot.get("mode", "ghost") if isinstance(snapshot, dict) else "ghost",
            "wallet": wallet_snapshot,
            "funding_gate": funding_gate or {},
            "transition_plan": transition,
            "live_readiness": live_readiness or {"ready": False},
            "ghost_trading": ghost_state,
            "accounting": accounting,
            "recent_trades": recent_trades,
            "active_advisories": active_advisories,
    }
    return {
            "live_readiness": operational_state["live_readiness"],
            "metrics_by_stage": list(metric_counts),
            "feedback_by_severity": list(feedback_counts),
            "latest_metrics": latest_metrics,
            "latest_feedback": latest_feedback,
            "recent_trades": recent_trades,
            "advisories_by_severity": list(advisory_counts),
            "active_advisories": active_advisories,
            "stable_bank": stable_bank,
            "total_profit": total_profit,
            "accounting": accounting,
            "live_readiness": live_readiness or {"ready": False},
            "operational_state": operational_state,
            "transition_plan": transition,
            "funding_gate": funding_gate or {},
            "wallet": operational_state["wallet"],
    }


class DashboardSummaryView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        return Response(build_dashboard_summary(), status=status.HTTP_200_OK)


class PipelineReadinessView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        readiness = _load_report(ROOT / "data/reports/live_readiness.json")
        if readiness and "ghost_collection_ready" not in readiness:
            readiness["ghost_collection_ready"] = bool(readiness.get("mini_ready"))
            readiness["ghost_collection_reason"] = (
                "" if readiness["ghost_collection_ready"]
                else readiness.get("mini_reason") or "model_not_ready"
            )
        confusion_meta = _load_report(ROOT / "data/reports/confusion_matrices.json")
        horizon = _load_report(ROOT / "data/reports/horizon_profile.json")
        snapshot = get_db().fetch_latest_organism_snapshot() or {}
        transition = snapshot.get("transition_plan") if isinstance(snapshot, dict) else {}
        if not transition and isinstance(snapshot, dict):
            pipeline_snapshot = snapshot.get("pipeline") or {}
            if isinstance(pipeline_snapshot, dict):
                transition = pipeline_snapshot.get("transition_plan") or {}
        # Per-strategy readiness, so the UI can say WHICH strategies are ready
        # instead of collapsing them into one aggregate verdict that
        # contradicts the system (Pipeline showed "Ghost Ready:
        # insufficient_accuracy" while atf_static was ghost-ready on positive
        # expectancy and approved for live).
        strategies = strategy_readiness()
        merged = dict(readiness or {})
        if not merged:
            merged = {"ready": False, "_missing": True, "_stale": True}
        merged["ghost_ready_strategies"] = strategies["ghost_ready_ids"]
        merged["live_ready_strategies"] = strategies["live_ready_ids"]
        merged["ghost_ready_any"] = strategies["ghost_ready"]
        merged["live_ready_any"] = strategies["live_ready"]
        merged["is_live_trading"] = strategies["is_live_trading"]
        if strategies["ghost_ready"] or strategies["live_ready"]:
            merged["_stale"] = False
            merged["_missing"] = False
        payload = {
            "live_readiness": merged,
            "strategy_readiness": strategies,
            "confusion": confusion_meta.get("confusion") if confusion_meta else {},
            "decision_threshold": confusion_meta.get("decision_threshold") if confusion_meta else None,
            "horizon_profile": horizon.get("profile") if horizon else {},
            "transition_plan": transition or (confusion_meta.get("transition_plan") if confusion_meta else {}),
        }
        timestamps = [
            readiness.get("updated_at") if isinstance(readiness, dict) else None,
            confusion_meta.get("updated_at") if isinstance(confusion_meta, dict) else None,
            horizon.get("updated_at") if isinstance(horizon, dict) else None,
        ]
        payload["updated_at"] = max((ts for ts in timestamps if isinstance(ts, (int, float))), default=None)
        return Response(payload, status=status.HTTP_200_OK)


class BusScheduleView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        global _BUS_SCHEDULE_PAYLOAD_CACHE, _BUS_SCHEDULE_PAYLOAD_TIME
        if os.getenv("PYTEST_CURRENT_TEST"):
            return self._build_bus_schedule_response(use_mock_db=True)
        now_req = time.time()
        with _LIVE_SWAPS_CACHE_LOCK:
            cached = dict(_BUS_SCHEDULE_PAYLOAD_CACHE) if _BUS_SCHEDULE_PAYLOAD_CACHE else None
            age = now_req - _BUS_SCHEDULE_PAYLOAD_TIME
        if cached is not None and age < 15.0:
            if age >= 2.5 and _BUS_SCHEDULE_REFRESHING.acquire(blocking=False):
                def _bg_refresh() -> None:
                    try:
                        self._build_bus_schedule_response()
                    finally:
                        _BUS_SCHEDULE_REFRESHING.release()
                threading.Thread(target=_bg_refresh, daemon=True, name="bus-sched-refresh").start()
            return Response(cached, status=status.HTTP_200_OK)
        if not _BUS_SCHEDULE_REFRESHING.acquire(blocking=False):
            with _LIVE_SWAPS_CACHE_LOCK:
                if _BUS_SCHEDULE_PAYLOAD_CACHE:
                    return Response(dict(_BUS_SCHEDULE_PAYLOAD_CACHE), status=status.HTTP_200_OK)
            _BUS_SCHEDULE_REFRESHING.acquire()
        try:
            return self._build_bus_schedule_response()
        finally:
            _BUS_SCHEDULE_REFRESHING.release()

    def _build_bus_schedule_response(self, use_mock_db: bool = False) -> Response:
        global _BUS_SCHEDULE_PAYLOAD_CACHE, _BUS_SCHEDULE_PAYLOAD_TIME
        db = get_db() if use_mock_db else None
        snapshot = _cached_latest_snapshot(db)
        scheduler = snapshot.get("scheduler") if isinstance(snapshot, dict) else []
        transition = {}
        if isinstance(snapshot, dict):
            transition = snapshot.get("transition_plan") or {}
            if not transition:
                pipeline_payload = snapshot.get("pipeline") or {}
                if isinstance(pipeline_payload, dict):
                    transition = pipeline_payload.get("transition_plan") or {}

        risk_flags = transition.get("risk_flags") if isinstance(transition, dict) else {}
        readiness = _load_report(ROOT / "data/reports/live_readiness.json")
        bus_strategies = strategy_readiness()
        if isinstance(risk_flags, dict) and "halt_ghost" in risk_flags:
            ghost_collection_ready = not bool(risk_flags.get("halt_ghost"))
        elif isinstance(readiness, dict) and readiness.get("ghost_collection_ready") is not None:
            ghost_collection_ready = bool(readiness.get("ghost_collection_ready"))
        else:
            ghost_collection_ready = bool(bus_strategies["ghost_ready"])
        if not ghost_collection_ready and bus_strategies["ghost_ready"]:
            ghost_collection_ready = True
        capital_plan = transition.get("capital_plan") if isinstance(transition, dict) else {}
        bus_actions = transition.get("bus_swap_actions") if isinstance(transition, dict) else None
        if not isinstance(bus_actions, list):
            bus_actions = transition.get("bus_actions") if isinstance(transition, dict) else []

        ghost_schedule = []
        route_diagnostics = []
        atf_pending_actions = []
        atf_action_by_symbol = {}
        try:
            stored_atf_actions = _ro_sqlite_json_kv(db, "atf_static_strategy:pending_bus_actions", [])
        except Exception:
            stored_atf_actions = []
        if isinstance(stored_atf_actions, list):
            atf_pending_actions = [item for item in stored_atf_actions if isinstance(item, dict)]
            atf_action_by_symbol = {
                str(item.get("symbol") or item.get("pair") or item.get("token") or "").upper(): item
                for item in atf_pending_actions
                if str(item.get("symbol") or item.get("pair") or item.get("token") or "").strip()
            }

        live_approved_sids = set(bus_strategies.get("live_ready_ids") or [])
        live_approved_sids.update(["money_button", "money_button_dip"])

        live_schedule = []

        if isinstance(scheduler, list):
            for entry in scheduler:
                if not isinstance(entry, dict):
                    continue
                route_status = entry.get("status", "unknown")
                directive = entry.get("last_directive")

                # Include diagnostic info for all routes (active, warming, waiting)
                diag = {
                    "symbol": entry.get("symbol"),
                    "status": route_status,
                    "history_points": entry.get("history_points", 0),
                    "evaluation_count": entry.get("evaluation_count", 0),
                    "last_filter_reason": entry.get("last_filter_reason", ""),
                    "price": _safe_float(entry.get("price")),
                    "updated_at": entry.get("last_update"),
                }
                # Include trade direction info if available
                if entry.get("last_trade_action"):
                    diag["last_trade_action"] = entry.get("last_trade_action")
                    diag["last_trade_price"] = _safe_float(entry.get("last_trade_price"))
                    diag["expected_next"] = entry.get("expected_next")
                route_diagnostics.append(diag)

                # Build schedule from routes with directives
                if not isinstance(directive, dict):
                    continue
                size = _safe_float(directive.get("size"))
                price = _safe_float(directive.get("target_price") or entry.get("price"))
                usd_value = round(size * price, 6) if size and price else 0.0
                dir_sid = str(directive.get("strategy_id") or "")
                
                sched_item = {
                    "symbol": entry.get("symbol"),
                    "action": directive.get("action") or "enter",
                    "size": size,
                    "price": price,
                    "usd_value": usd_value,
                    "horizon": directive.get("horizon"),
                    "confidence": _safe_float(directive.get("confidence")),
                    "reason": directive.get("reason"),
                    "tier": directive.get("tier"),
                    "updated_at": entry.get("last_update"),
                    "strategy_id": dir_sid or None,
                }
                if dir_sid in live_approved_sids:
                    live_schedule.append(sched_item)
                else:
                    ghost_schedule.append(sched_item)

        # Open ghost positions held by the trading bots. Route directives are
        # transient (only present the moment a strategy fires), so without this
        # merge the ghost lane looks empty while positions are actually being
        # held and worked. Prices fall back to the route diagnostic mark.
        price_by_symbol = {
            str(d.get("symbol") or "").upper(): _safe_float(d.get("price"))
            for d in route_diagnostics
            if d.get("symbol")
        }
        open_positions = snapshot.get("positions") if isinstance(snapshot, dict) else {}
        # Fallback: older/running engine snapshots project a slim position dict
        # without horizon/strategy_id. The persisted ghost state keeps the full
        # position dicts, so enrich from there when the snapshot lacks them —
        # this makes the sell-high timeframe label work without an engine restart.
        state_positions = _cached_state_positions(db)
        if isinstance(open_positions, dict):
            for raw_symbol, pos in open_positions.items():
                if not isinstance(pos, dict):
                    continue
                symbol = str(pos.get("symbol") or raw_symbol or "").upper()
                if not symbol:
                    continue
                fallback = state_positions.get(symbol, {})
                size = _safe_float(pos.get("size"))
                entry_price = _safe_float(pos.get("entry_price"))
                mark_price = price_by_symbol.get(symbol) or entry_price
                pnl_pct = (
                    round((mark_price - entry_price) / entry_price * 100.0, 4)
                    if entry_price > 0 and mark_price > 0
                    else 0.0
                )
                strategy_id = str(pos.get("strategy_id") or fallback.get("strategy_id") or "")
                horizon = (
                    pos.get("horizon")
                    or fallback.get("horizon")
                    or (strategy_id.split("@")[-1] if "@" in strategy_id else "")
                    or "open"
                )
                pos_mode = str(pos.get("mode") or fallback.get("mode") or "ghost").lower()
                is_live_pos = pos_mode == "live" or (strategy_id and strategy_id in live_approved_sids)
                pos_item = {
                    "symbol": symbol,
                    "action": "hold",
                    "size": size,
                    "price": mark_price,
                    "entry_price": entry_price,
                    "usd_value": round(size * mark_price, 6) if size and mark_price else 0.0,
                    "pnl_pct": pnl_pct,
                    "horizon": horizon,
                    "confidence": _safe_float(pos.get("confidence")),
                    "reason": f"open {'live' if is_live_pos else 'ghost'} position ({strategy_id or 'bot'})",
                    "tier": pos.get("tier"),
                    "updated_at": pos.get("entry_ts"),
                    "strategy_id": strategy_id or None,
                    "source": "organism_snapshot:positions",
                }
                if is_live_pos:
                    live_schedule.append(pos_item)
                else:
                    ghost_schedule.append(pos_item)

        # The ATF static quote-scout lane is intentionally ghost-only and does
        # not always create organism route directives. Merge its open positions
        # and pending bus actions here so the Bus Scheduler ghost section is a
        # complete operational view instead of only showing legacy routes.
        try:
            atf_positions = _ro_sqlite_json_kv(db, "atf_static_strategy:ghost_positions", {})
        except Exception:
            atf_positions = {}
        atf_position_symbols = set()
        if isinstance(atf_positions, dict):
            for raw_symbol, pos in atf_positions.items():
                if not isinstance(pos, dict):
                    continue
                symbol = str(pos.get("symbol") or raw_symbol or "").upper()
                if not symbol:
                    continue
                atf_position_symbols.add(symbol)
                pending = atf_action_by_symbol.get(symbol) or {}
                entry_price = _safe_float(pos.get("entry_price"))
                mark_price = _safe_float(pos.get("last_price"), entry_price)
                target_price = _safe_float(pos.get("target_price"))
                target_return = _safe_float(pos.get("target_return"))
                usd_value = _safe_float(
                    pos.get("budget_usd")
                    or pending.get("target_usd")
                    or pending.get("usd_value")
                    or 0.0
                )
                ghost_schedule.append(
                    {
                        "symbol": symbol,
                        "action": "hold",
                        "size": 0.0,
                        "price": mark_price or entry_price,
                        "entry_price": entry_price,
                        "target_price": target_price,
                        "target_return": target_return,
                        "usd_value": usd_value,
                        "horizon": "atf_quote_scout",
                        "confidence": _safe_float(pos.get("confidence")),
                        "reason": "ATF ghost quote-scout open position",
                        "tier": "ATF",
                        "updated_at": pos.get("last_seen_ts") or pos.get("entry_ts"),
                        "strategy_id": pos.get("strategy_id") or "atf_static",
                        "source": "atf_static_strategy:ghost_positions",
                        "chain": pos.get("chain"),
                        "token_address": pos.get("token_address"),
                        "pair_address": pos.get("pair_address"),
                    }
                )

        for action in atf_pending_actions:
            symbol = str(action.get("symbol") or action.get("pair") or action.get("token") or "").upper()
            if not symbol or symbol in atf_position_symbols:
                continue
            ghost_schedule.append(
                {
                    "symbol": symbol,
                    "action": action.get("action") or "evaluate_atf_static_entry",
                    "size": 0.0,
                    "price": 0.0,
                    "usd_value": _safe_float(action.get("target_usd") or action.get("usd_value")),
                    "horizon": "atf_pending",
                    "confidence": 0.0,
                    "reason": action.get("reason") or "ATF pending ghost bus action",
                    "priority": action.get("priority"),
                    "window_sec": action.get("window_sec"),
                    "updated_at": action.get("ts"),
                    "strategy_id": action.get("strategy_id") or "atf_static",
                    "source": "atf_static_strategy:pending_bus_actions",
                    "chain": action.get("chain"),
                    "token_address": action.get("token_address"),
                }
            )
        ghost_schedule.sort(key=lambda item: item.get("usd_value", 0.0), reverse=True)
        ghost_schedule = ghost_schedule[:16]

        # Summarize route statuses for quick overview
        status_counts = {}
        for diag in route_diagnostics:
            s = diag.get("status", "unknown")
            status_counts[s] = status_counts.get(s, 0) + 1

        # live_schedule already holds live-approved scheduler directives and open live positions
        if isinstance(bus_actions, list):
            for action in bus_actions:
                if not isinstance(action, dict):
                    continue
                size = _safe_float(action.get("size"))
                price = _safe_float(action.get("price"))
                usd_value = _safe_float(action.get("target_usd") or action.get("usd_value"))
                if not usd_value and size and price:
                    usd_value = round(size * price, 6)
                live_strategy_id = str(action.get("strategy_id") or "")
                live_horizon = (
                    action.get("horizon")
                    or (live_strategy_id.split("@")[-1] if "@" in live_strategy_id else "")
                    or action.get("window_sec")
                )
                live_schedule.append(
                    {
                        "symbol": action.get("symbol") or action.get("pair") or action.get("token"),
                        "action": action.get("action") or "bus_action",
                        "size": size,
                        "price": price,
                        "usd_value": usd_value,
                        "reason": action.get("reason"),
                        "priority": action.get("priority"),
                        "window_sec": action.get("window_sec"),
                        "horizon": live_horizon,
                        "strategy_id": live_strategy_id or None,
                    }
                )

        # Ensure active licensed strategies are scheduled and visible in the Live Lane
        live_symbols_present = {str(item.get("symbol") or "").upper() for item in live_schedule if item.get("symbol")}

        # 1. Money Button & Money Button Dip licensed Base routes (gate-clean pairs)
        if "money_button_dip" in live_approved_sids or "money_button" in live_approved_sids:
            rec_clip = min(2.0, max(0.50, _safe_float(risk_flags.get("recommended_live_usd"), 1.10) or 1.10))
            for mb_sym, mb_sid, mb_hz in [
                ("WETH-USDC", "money_button", "10m"),
                ("WETH-USDT", "money_button_dip", "30m"),
                ("MORPHO-USDC", "money_button_dip", "30m"),
            ]:
                if mb_sym not in live_symbols_present:
                    live_symbols_present.add(mb_sym)
                    live_schedule.append({
                        "symbol": mb_sym,
                        "action": "licensed_live_route",
                        "size": 0.0,
                        "price": price_by_symbol.get(mb_sym, 0.0),
                        "usd_value": round(rec_clip, 4),
                        "horizon": mb_hz,
                        "confidence": 0.85,
                        "reason": f"{mb_sid} licensed live route [BASE:{mb_sym}]",
                        "priority": 1,
                        "window_sec": 600 if mb_hz == "10m" else 1800,
                        "strategy_id": mb_sid,
                    })

        # 2. Recent settled live swaps on Base (indexed equality seeks <2ms, cached 10s)
        recent_live_swaps = _recent_live_swaps_fast(db)
        for sw in recent_live_swaps:
            if sw["status"] in ("live-entry", "live-exit"):
                txh = sw["tx_hash"]
                swap_sym = sw["symbol"]
                live_schedule.insert(0, {
                    "symbol": swap_sym,
                    "action": sw["status"],
                    "size": sw["size"],
                    "price": sw["price"] or price_by_symbol.get(swap_sym, 0.0),
                    "usd_value": round(sw["quote_spent"] or 0.55, 6),
                    "horizon": sw["horizon"],
                    "confidence": sw["confidence"],
                    "reason": f"settled on Base ({txh})",
                    "tx_hash": txh,
                    "updated_at": sw["ts"],
                    "strategy_id": sw["strategy_id"] or "money_button",
                    "source": "trading_ops:live",
                })

        # 3. LiveGraduatedStrategy models
        try:
            from agentresearch.models import LiveGraduatedStrategy
            for lgs in LiveGraduatedStrategy.objects.filter(live_status="active").order_by("-updated_at")[:6]:
                if _is_flood_strategy_id(lgs.strategy_id):
                    continue
                sym = str(lgs.symbol or "").upper()
                if sym not in live_symbols_present:
                    live_symbols_present.add(sym)
                    live_schedule.append({
                        "symbol": sym,
                        "action": "licensed_live_route",
                        "size": 0.0,
                        "price": price_by_symbol.get(sym, 0.0),
                        "usd_value": _safe_float(lgs.clip_usd, 1.0) or 1.0,
                        "horizon": "5m",
                        "confidence": _safe_float(lgs.ghost_win_rate, 0.60),
                        "reason": f"Self-Graduated Live: {lgs.name}",
                        "priority": 1,
                        "window_sec": 300,
                        "strategy_id": lgs.strategy_id,
                    })
        except Exception:
            pass

        live_schedule.sort(key=lambda item: (0 if item.get("tx_hash") else 1, -item.get("usd_value", 0.0)))
        live_schedule = live_schedule[:16]

        # Stamp a canonical sell-high timeframe chip (1H/12H/1D/1W/1M…) onto
        # every lane item so the UI shows the model/strategy's expected holding
        # horizon. Raw 'horizon' is preserved; 'horizon_label' is the display.
        for item in ghost_schedule:
            item["horizon_label"] = _horizon_label(item.get("horizon"))
        for item in live_schedule:
            item["horizon_label"] = _horizon_label(item.get("horizon"))

        live_ramp = {}
        if isinstance(capital_plan, dict):
            live_ramp = capital_plan.get("live_ramp_schedule") or {}
        accounting_snap = _cached_accounting(db)

        payload = {
            "available": bool(snapshot),
            "timestamp": snapshot.get("timestamp") if isinstance(snapshot, dict) else None,
            "accounting": {
                "ghost": accounting_snap["ghost"],
                "live": accounting_snap["live"],
            },
            "ghost": {
                "halted": not ghost_collection_ready,
                "reason": (
                    readiness.get("ghost_collection_reason")
                    or readiness.get("mini_reason")
                    or "model_not_ready"
                ) if not ghost_collection_ready else None,
                "active_strategies": bus_strategies["ghost_ready_ids"],
                "risk_multiplier": max(
                    0.25 if ghost_collection_ready else 0.0,
                    _safe_float(risk_flags.get("ghost_risk_multiplier")) if isinstance(risk_flags, dict) else 0.0,
                ),
                "schedule": ghost_schedule,
            },
            "strategies": bus_strategies,
            "live": {
                # Halted only when the plan says so AND no strategy is
                # approved: a lane with an approved strategy is not halted.
                "halted": (
                    bool(risk_flags.get("halt_live")) and not bus_strategies["live_ready"]
                ) if isinstance(risk_flags, dict) else False,
                "reason": (
                    risk_flags.get("halt_reason") or None
                ) if isinstance(risk_flags, dict) else None,
                "approved_strategies": bus_strategies["live_ready_ids"],
                "recommended_live_usd": _safe_float(risk_flags.get("recommended_live_usd")) if isinstance(risk_flags, dict) else 0.0,
                "min_clip_usd": _safe_float(risk_flags.get("min_clip_usd")) if isinstance(risk_flags, dict) else 0.0,
                "recent_swaps": recent_live_swaps,
                "schedule": live_schedule,
                "ramp": {
                    "first_tranche_usd": _safe_float(live_ramp.get("first_tranche_usd")),
                    "max_live_usd": _safe_float(live_ramp.get("max_live_usd")),
                    "deployable_stable_usd": _safe_float(live_ramp.get("deployable_stable_usd")),
                    "first_tranche_cap_usd": _safe_float(live_ramp.get("first_tranche_cap_usd")),
                },
            },
            "summary": {
                "bus_actions_pending": bool(risk_flags.get("bus_actions_pending")) if isinstance(risk_flags, dict) else False,
                "bus_action_count": (len(bus_actions) if isinstance(bus_actions, list) else 0) + len(atf_pending_actions),
                "atf_ghost_positions": len(atf_position_symbols),
                "atf_pending_bus_actions": len(atf_pending_actions),
                "route_status_counts": status_counts,
                "total_routes": len(route_diagnostics),
            },
            "route_diagnostics": route_diagnostics[:30],
        }
        with _LIVE_SWAPS_CACHE_LOCK:
            _BUS_SCHEDULE_PAYLOAD_CACHE = dict(payload)
            _BUS_SCHEDULE_PAYLOAD_TIME = time.time()
        return Response(payload, status=status.HTTP_200_OK)


class OrganismLatestView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        db = get_db()
        snapshot = db.fetch_latest_organism_snapshot()
        return Response(
            {
                "snapshot": snapshot or {},
                "available": bool(snapshot),
            },
            status=status.HTTP_200_OK,
        )


class OrganismHistoryView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        db = get_db()
        start_ts = request.query_params.get("start_ts")
        end_ts = request.query_params.get("end_ts")
        limit_param = request.query_params.get("limit", "200")
        try:
            limit = max(1, min(int(limit_param), 1000))
        except ValueError:
            limit = 200
        try:
            start = float(start_ts) if start_ts is not None else None
        except ValueError:
            start = None
        try:
            end = float(end_ts) if end_ts is not None else None
        except ValueError:
            end = None
        history = db.fetch_organism_history(start_ts=start, end_ts=end, limit=limit)
        return Response(
            {
                "snapshots": history,
                "count": len(history),
            },
            status=status.HTTP_200_OK,
        )


class OrganismSettingsView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        db = get_db()
        return Response({"label_scale": db.get_label_scale()}, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        db = get_db()
        try:
            scale = float(request.data.get("label_scale"))
        except (TypeError, ValueError):
            return Response({"detail": "label_scale must be numeric"}, status=status.HTTP_400_BAD_REQUEST)
        db.set_label_scale(scale)
        return Response({"label_scale": db.get_label_scale()}, status=status.HTTP_200_OK)
