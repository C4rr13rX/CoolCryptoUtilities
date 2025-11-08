from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

try:
    from filter_scams import FilterScamTokens  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FilterScamTokens = None

from db import TradingDatabase, get_db
from services.logging_utils import log_message
from services.public_api_clients import aggregate_market_data
from trading.constants import PRIMARY_CHAIN

WINDOW_OPTIONS: Dict[str, int] = {
    "24h": 24 * 3600,
    "3d": 3 * 24 * 3600,
    "1w": 7 * 24 * 3600,
    "2w": 14 * 24 * 3600,
    "1m": 30 * 24 * 3600,
}

HIST_ROOT = Path(os.getenv("HISTORICAL_DATA_ROOT", "data/historical_ohlcv"))
PAIR_INDEX_TEMPLATE = "data/pair_index_{chain}.json"
DEFAULT_SCAN_CHAIN = os.getenv("SIGNAL_SCAN_CHAIN", PRIMARY_CHAIN or "base").strip().lower() or "base"
SCAM_REGISTRY_KEY = "scam_registry"


@dataclass
class SignalRecord:
    symbol: str
    chain: str
    window: str
    start_price: float
    latest_price: float
    change_pct: float
    start_ts: float
    latest_ts: float
    avg_volume: Optional[float]
    address: Optional[str] = None
    source: str = "stream"

    @property
    def direction(self) -> str:
        if self.change_pct > 0:
            return "bullish"
        if self.change_pct < 0:
            return "bearish"
        return "neutral"


def scan_price_signals(
    window_key: str,
    *,
    direction: Literal["bullish", "bearish", "all"] = "bullish",
    limit: int = 50,
    min_volume: float = 0.0,
) -> Tuple[List[Dict[str, object]], Dict[str, Any]]:
    db = get_db()
    window_seconds = WINDOW_OPTIONS.get(window_key, WINDOW_OPTIONS["24h"])
    diagnostics: Dict[str, Any] = {
        "window": window_key,
        "direction": direction,
        "limit": limit,
        "min_volume": min_volume,
    }
    records: List[SignalRecord] = []

    stream_records, stream_meta = _scan_market_stream(window_seconds, window_key, min_volume=min_volume)
    diagnostics.update(stream_meta)
    records.extend(stream_records)

    historical_records, hist_meta = _scan_historical_signals(
        window_seconds,
        window_key,
        min_volume=min_volume,
        direction=direction,
        desired=limit * 4,
    )
    diagnostics.update(hist_meta)
    records.extend(historical_records)

    if len(records) < limit:
        public_records, public_meta = _scan_public_market(
            limit - len(records),
            direction=direction,
            min_volume=min_volume,
        )
        diagnostics.update(public_meta)
        records.extend(public_records)

    if not records:
        diagnostics["result_count"] = 0
        diagnostics["message"] = "No pairs met the selection criteria."
        return [], diagnostics

    records = _dedupe_records(records)
    filtered = _filter_by_direction(records, direction)
    ranked = _rank_records(filtered, direction, limit)
    survivors, scam_meta = _apply_scam_filter(ranked, db)
    diagnostics.update(scam_meta)
    survivors = _rank_records(survivors, direction, limit)
    diagnostics["result_count"] = len(survivors)
    diagnostics["message"] = "ok" if survivors else "No pairs passed the filters."
    return [
        {
            "symbol": rec.symbol,
            "chain": rec.chain,
            "window": rec.window,
            "change_pct": rec.change_pct,
            "start_price": rec.start_price,
            "latest_price": rec.latest_price,
            "start_ts": rec.start_ts,
            "latest_ts": rec.latest_ts,
            "avg_volume": rec.avg_volume,
            "direction": rec.direction,
            "source": rec.source,
            "pair_address": rec.address,
        }
        for rec in survivors
    ], diagnostics


def _scan_market_stream(
    window_seconds: int,
    window_label: str,
    *,
    min_volume: float,
) -> Tuple[List[SignalRecord], Dict[str, Any]]:
    end_ts = time.time()
    start_ts = end_ts - window_seconds
    db = get_db()
    pairs = db.list_market_pairs_since(start_ts)
    records: List[SignalRecord] = []
    for symbol, chain in pairs:
        start_info = db.get_market_price(symbol, chain, ts=start_ts, after=True)
        latest_info = db.get_market_price(symbol, chain, ts=None, after=False)
        if not start_info or not latest_info:
            continue
        start_price, start_price_ts, _ = start_info
        latest_price, latest_price_ts, _ = latest_info
        if start_price is None or start_price <= 0 or latest_price is None:
            continue
        change_pct = ((latest_price - start_price) / start_price) * 100.0
        avg_vol = db.average_volume(symbol, chain, start_ts)
        if avg_vol is not None and avg_vol < min_volume:
            continue
        records.append(
            SignalRecord(
                symbol=symbol,
                chain=chain,
                window=window_label,
                start_price=float(start_price),
                latest_price=float(latest_price),
                change_pct=float(change_pct),
                start_ts=float(start_price_ts),
                latest_ts=float(latest_price_ts),
                avg_volume=float(avg_vol) if avg_vol is not None else None,
                source="stream",
            )
        )
    return records, {
        "stream_pairs": len(pairs),
        "stream_hits": len(records),
    }


def _scan_historical_signals(
    window_seconds: int,
    window_label: str,
    *,
    min_volume: float,
    direction: Literal["bullish", "bearish", "all"],
    desired: int,
) -> Tuple[List[SignalRecord], Dict[str, Any]]:
    pair_index_path = Path(PAIR_INDEX_TEMPLATE.format(chain=DEFAULT_SCAN_CHAIN))
    if not pair_index_path.exists():
        return [], {"historical_considered": 0, "historical_hits": 0}
    try:
        pair_index = json.loads(pair_index_path.read_text(encoding="utf-8"))
    except Exception:
        return [], {"historical_considered": 0, "historical_hits": 0}

    chain_dir = HIST_ROOT / DEFAULT_SCAN_CHAIN
    if not chain_dir.exists():
        chain_dir = HIST_ROOT

    ordered_pairs = sorted(
        pair_index.items(),
        key=lambda item: int((item[1] or {}).get("index", 0)),
    )
    total_pairs = len(ordered_pairs)
    records: List[SignalRecord] = []
    for address, meta in ordered_pairs:
        symbol = str((meta or {}).get("symbol") or "").upper()
        if not symbol:
            continue
        index = int((meta or {}).get("index", len(records)))
        path = _candidate_file(chain_dir, index, symbol)
        if not path or not path.exists():
            continue
        try:
            candles = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        record = _build_historical_record(
            candles,
            symbol=symbol,
            chain=DEFAULT_SCAN_CHAIN,
            window_label=window_label,
            window_seconds=window_seconds,
            address=address,
        )
        if not record:
            continue
        if record.avg_volume is not None and record.avg_volume < min_volume:
            continue
        if direction == "bullish" and record.change_pct <= 0:
            continue
        if direction == "bearish" and record.change_pct >= 0:
            continue
        record.source = "historical"
        records.append(record)
        if len(records) >= desired:
            break
    return records, {
        "historical_considered": total_pairs,
        "historical_hits": len(records),
    }


def _candidate_file(root: Path, index: int, symbol: str) -> Optional[Path]:
    symbol_safe = symbol.replace("/", "-").replace(":", "-")
    filename = f"{index:04d}_{symbol_safe}.json"
    candidates = [
        root / filename,
        HIST_ROOT / filename,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            try:
                if candidate.stat().st_size > 2:
                    return candidate
            except Exception:
                continue
    fallback = list(root.glob(f"*_{symbol_safe}.json"))
    if fallback:
        return fallback[0]
    fallback = list(HIST_ROOT.glob(f"*_{symbol_safe}.json"))
    return fallback[0] if fallback else None


def _scan_public_market(
    limit: int,
    *,
    direction: Literal["bullish", "bearish", "all"],
    min_volume: float,
) -> Tuple[List[SignalRecord], Dict[str, Any]]:
    try:
        snapshots = aggregate_market_data(top_n=max(10, limit * 3))
    except Exception as exc:
        log_message("signals", f"public market fetch failed: {exc}", severity="warning")
        return [], {"public_hits": 0}
    records: List[SignalRecord] = []
    now = time.time()
    for snap in snapshots:
        change = float(snap.percent_change_24h or 0.0)
        volume = float(snap.volume_24h or 0.0)
        if min_volume and volume < min_volume:
            continue
        if direction == "bullish" and change <= 0:
            continue
        if direction == "bearish" and change >= 0:
            continue
        try:
            latest_price = float(snap.price_usd or 0.0)
        except Exception:
            latest_price = 0.0
        if latest_price <= 0:
            continue
        start_price = latest_price / (1 + change / 100.0) if change else latest_price
        records.append(
            SignalRecord(
                symbol=str(snap.symbol or "").upper(),
                chain="public",
                window="24h",
                start_price=start_price,
                latest_price=latest_price,
                change_pct=change,
                start_ts=now - WINDOW_OPTIONS["24h"],
                latest_ts=now,
                avg_volume=volume,
                source="public",
            )
        )
        if len(records) >= limit:
            break
    return records, {"public_hits": len(records)}


def _build_historical_record(
    candles: Sequence[Dict[str, object]],
    *,
    symbol: str,
    chain: str,
    window_label: str,
    window_seconds: int,
    address: str,
) -> Optional[SignalRecord]:
    if not candles:
        return None
    try:
        latest = candles[-1]
        latest_ts = int(latest.get("timestamp") or 0)
    except Exception:
        return None
    cutoff = latest_ts - window_seconds
    start_row = None
    volumes: List[float] = []
    for row in candles:
        try:
            ts = int(row.get("timestamp") or 0)
        except Exception:
            continue
        if ts < cutoff:
            continue
        if start_row is None:
            start_row = row
        try:
            net = abs(float(row.get("net_volume") or 0.0))
        except Exception:
            net = 0.0
        try:
            buy = abs(float(row.get("buy_volume") or 0.0))
        except Exception:
            buy = 0.0
        try:
            sell = abs(float(row.get("sell_volume") or 0.0))
        except Exception:
            sell = 0.0
        volume_val = max(net, buy + sell)
        if volume_val > 0:
            volumes.append(volume_val)
    if start_row is None:
        return None
    try:
        start_price = float(start_row.get("close") or start_row.get("open") or 0.0)
        latest_price = float(latest.get("close") or latest.get("open") or 0.0)
    except Exception:
        return None
    if not math.isfinite(start_price) or start_price <= 0:
        return None
    if not math.isfinite(latest_price) or latest_price <= 0:
        return None
    change_pct = ((latest_price - start_price) / start_price) * 100.0
    avg_volume = (sum(volumes) / len(volumes)) if volumes else None
    return SignalRecord(
        symbol=symbol,
        chain=chain,
        window=window_label,
        start_price=start_price,
        latest_price=latest_price,
        change_pct=change_pct,
        start_ts=float(start_row.get("timestamp") or 0.0),
        latest_ts=float(latest_ts),
        avg_volume=avg_volume,
        address=address,
        source="historical",
    )


def _dedupe_records(records: Sequence[SignalRecord]) -> List[SignalRecord]:
    dedup: Dict[Tuple[str, str], SignalRecord] = {}
    for rec in records:
        key = (rec.chain, rec.symbol)
        existing = dedup.get(key)
        if not existing or abs(rec.change_pct) > abs(existing.change_pct):
            dedup[key] = rec
    return list(dedup.values())


def _filter_by_direction(
    records: Sequence[SignalRecord],
    direction: Literal["bullish", "bearish", "all"],
) -> List[SignalRecord]:
    if direction == "all":
        return list(records)
    return [rec for rec in records if rec.direction == direction]


def _rank_records(
    records: Sequence[SignalRecord],
    direction: Literal["bullish", "bearish", "all"],
    limit: int,
) -> List[SignalRecord]:
    if not records:
        return []
    reverse = direction != "bearish"
    sorted_records = sorted(records, key=lambda rec: rec.change_pct, reverse=reverse)
    if limit > 0:
        sorted_records = sorted_records[:limit]
    return sorted_records


def _apply_scam_filter(
    records: Sequence[SignalRecord],
    db: TradingDatabase,
) -> Tuple[List[SignalRecord], Dict[str, Any]]:
    if not records:
        return [], {"scam_filtered": 0}
    registry = db.get_json(SCAM_REGISTRY_KEY) or {}
    known_specs = set(registry.get("specs", []))
    known_symbols = set(registry.get("symbols", []))

    survivors: List[SignalRecord] = []
    candidates: Dict[str, SignalRecord] = {}
    for rec in records:
        spec = _spec(rec.chain, rec.address) if rec.address else None
        if rec.symbol in known_symbols:
            continue
        if spec and spec in known_specs:
            continue
        if spec:
            candidates[spec] = rec
        survivors.append(rec)

    if not candidates or not FilterScamTokens:
        return survivors, {"scam_filtered": 0}

    try:
        filt = FilterScamTokens()
    except Exception as exc:  # pragma: no cover - missing credentials
        log_message("signals", f"scam filter unavailable: {exc}", severity="warning")
        return survivors, {"scam_filtered": 0}

    specs = list(candidates.keys())
    try:
        result = filt.filter(specs)
    except Exception as exc:
        log_message("signals", f"scam filter failed: {exc}", severity="warning")
        return survivors, {"scam_filtered": 0}

    allowed_specs = {_normalize_spec(token) for token in result.tokens}
    flagged_specs = [spec for spec in specs if spec not in allowed_specs]
    if not flagged_specs:
        return survivors, {"scam_filtered": 0}

    flagged_set = set(flagged_specs)
    survivors = [rec for rec in survivors if _spec(rec.chain, rec.address) not in flagged_set]
    reasons = getattr(result, "reasons", {}) or {}
    flagged_symbols = []
    for spec in flagged_specs:
        rec = candidates.get(spec)
        if not rec:
            continue
        known_specs.add(spec)
        known_symbols.add(rec.symbol)
        flagged_symbols.append(rec.symbol)
        addr_key = (rec.address or "").lower()
        detail = ", ".join(reasons.get(addr_key, [])) or "flagged_by_filter"
        log_message(
            "signals",
            f"filtered scam candidate {rec.symbol}",
            severity="warning",
            details={"chain": rec.chain, "address": rec.address, "reason": detail},
        )
    db.set_json(
        SCAM_REGISTRY_KEY,
        {
            "specs": sorted(known_specs),
            "symbols": sorted(known_symbols),
        },
    )
    return survivors, {"scam_filtered": len(flagged_symbols)}


def _spec(chain: str, address: Optional[str]) -> Optional[str]:
    if not chain or not address:
        return None
    return f"{chain.lower()}:{address.lower()}"


def _normalize_spec(token: object) -> str:
    value = str(token or "")
    if ":" in value:
        return value.lower()
    return value.lower()
