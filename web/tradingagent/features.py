"""Everything measurable about a closed round trip.

WHY THIS EXISTS SEPARATELY
--------------------------
A search can only find patterns in features it can see. The theorem layer was
reading seven fields -- symbol, return, hold_sec, size_usd, ticks_1h, live,
ts -- from ``trading_ops``, and that is both the wrong SOURCE and too narrow
a vocabulary.

Wrong source, because ``trading_ops`` is an append-only event log that keeps
pre-fix artifacts forever: one 2026-09-04 row carries a profit of -0.4177
where the receipt puts it at -0.0169, and summing that table reported a
losing system that was in fact up. ``trade_outcomes`` is the book of record
and carries the decomposition -- gross_profit and fee_cost separately, entry
and exit price, quantity -- which is what makes cost-aware features possible
at all.

Too narrow, because the questions worth asking are about things that were not
in the seven: was this trade expensive relative to its size? did it exit for
a reason or time out? was the market persistent or mean-reverting when we
entered? A genetic search over features that cannot express those cannot
discover them, however long it runs.

WHAT MAKES A FEATURE ADMISSIBLE
-------------------------------
It must be knowable BEFORE the trade closes, or it is not a predictor -- it
is the outcome wearing a different name. ``fee_ratio`` uses fee_cost, which
is only known after the fill, so it is marked post-hoc and excluded from
anything predictive. Keeping both kinds in one place, explicitly labelled, is
safer than keeping the post-hoc ones out: a search that cannot see them
cannot be caught using them.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "storage" / "trading_cache.db"

#: Features a predictor may use: all knowable at entry time.
PREDICTIVE_FEATURES = (
    "hold_sec", "size_usd", "notional_usd", "entry_price",
    "ticks_1h", "hour_of_day", "day_of_week",
    "is_live", "is_stable_quote",
    "hurst", "return_autocorrelation", "usable_horizon_sec",
    "horizon_fits", "chaos_character",
)

#: Known only after the fact. Useful for explaining what happened; never for
#: predicting what will. Labelled so a search cannot quietly rely on them.
POST_HOC_FEATURES = (
    "return", "gross_return", "fee_ratio", "exit_reason", "hit_target",
)


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _tick_density(conn, since_sec: float) -> Dict[str, float]:
    """Ticks per hour per symbol -- a liquidity proxy we can read cheaply."""
    out: Dict[str, float] = {}
    hours = max(1.0, since_sec / 3600.0)
    try:
        for symbol, count in conn.execute(
                "SELECT symbol, COUNT(*) FROM market_stream WHERE ts > ? "
                "GROUP BY symbol", (time.time() - since_sec,)):
            out[str(symbol or "")] = float(count) / hours
    except Exception:  # noqa: BLE001
        pass
    return out


def _price_series(conn, symbol: str, limit: int = 400) -> List[float]:
    try:
        rows = conn.execute(
            "SELECT price FROM market_stream WHERE symbol = ? AND price > 0 "
            "ORDER BY ts DESC LIMIT ?", (symbol, int(limit))).fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [float(r[0]) for r in reversed(rows) if r and r[0]]


def closed_round_trips(since_sec: float = 86400 * 14,
                       with_chaos: bool = True) -> List[Dict[str, Any]]:
    """Every closed round trip, fully featured.

    Read from ``trade_outcomes`` -- the book of record -- rather than from the
    event log, so the money figures are the ones the receipts support.
    """
    rows: List[Dict[str, Any]] = []
    try:
        conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    except Exception:  # noqa: BLE001
        return rows

    try:
        density = _tick_density(conn, since_sec)
        chaos_cache: Dict[str, Dict[str, Any]] = {}
        cutoff = time.time() - since_sec

        conn.row_factory = sqlite3.Row
        for row in conn.execute(
                "SELECT * FROM trade_outcomes WHERE status = 'closed' "
                "AND ts > ? ORDER BY ts", (cutoff,)):
            net = _safe_float(row["net_profit"])
            if net is None:
                continue

            symbol = str(row["symbol"] or "")
            entry = _safe_float(row["entry_price"]) or 0.0
            exit_price = _safe_float(row["exit_price"]) or 0.0
            qty = _safe_float(row["quantity"]) or 0.0
            gross = _safe_float(row["gross_profit"])
            fee = _safe_float(row["fee_cost"]) or 0.0
            notional = entry * qty

            try:
                payload = json.loads(row["details"]) if row["details"] else {}
            except Exception:  # noqa: BLE001
                payload = {}
            if not isinstance(payload, dict):
                payload = {}

            ts = _safe_float(row["ts"]) or 0.0
            entry_ts = _safe_float(payload.get("entry_ts"))
            hold = (ts - entry_ts) if entry_ts else _safe_float(payload.get("age_sec"))

            # RETURN AS A FRACTION, not as dollars. A dollar figure measures
            # position size as much as it measures edge, and the ghost book's
            # notional spans 144x within a single symbol.
            ret = (net / notional) if notional > 0.01 else None
            if ret is None:
                continue

            record: Dict[str, Any] = {
                "symbol": symbol,
                "strategy_id": str(payload.get("strategy_id") or ""),
                "chain": str(row["chain"] or ""),
                "ts": ts,

                # --- outcome (post-hoc) ---
                "return": ret,
                "gross_return": (gross / notional) if (gross is not None and notional > 0.01) else None,
                "fee_ratio": (fee / notional) if notional > 0.01 else None,
                "exit_reason": str(payload.get("exit_reason") or payload.get("reason") or ""),
                "hit_target": bool(str(payload.get("exit_reason") or "").startswith("target")),

                # --- knowable at entry ---
                "hold_sec": hold,
                "size_usd": qty,
                "notional_usd": notional,
                "entry_price": entry,
                "exit_price": exit_price,
                "ticks_1h": density.get(symbol, 0.0),
                "hour_of_day": int(time.gmtime(ts).tm_hour) if ts else None,
                "day_of_week": int(time.gmtime(ts).tm_wday) if ts else None,
                "is_live": str(row["wallet"] or "") == "live",
                "is_stable_quote": str(row["quote_token"] or "").upper() in
                                   {"USDC", "USDT", "DAI", "USDBC"},
            }

            # --- what the series looked like (knowable at entry) ---
            if with_chaos and symbol:
                if symbol not in chaos_cache:
                    try:
                        from .chaos import chaos_profile

                        series = _price_series(conn, symbol)
                        chaos_cache[symbol] = chaos_profile(symbol, series, 300.0)
                    except Exception:  # noqa: BLE001
                        chaos_cache[symbol] = {}
                profile = chaos_cache.get(symbol) or {}
                record["hurst"] = profile.get("hurst")
                record["return_autocorrelation"] = profile.get("return_autocorrelation")
                record["usable_horizon_sec"] = profile.get("usable_horizon_sec")
                record["chaos_character"] = profile.get("character")
                usable = profile.get("usable_horizon_sec")
                record["horizon_fits"] = (
                    bool(hold is not None and usable and hold <= usable)
                    if (hold is not None and usable) else None)

            rows.append(record)
    except Exception:  # noqa: BLE001
        return rows
    finally:
        conn.close()

    return rows


def feature_vocabulary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """What each predictive feature actually ranges over in this data.

    A search needs the real distribution to pick thresholds worth testing --
    proposing "hold_sec > 86400" against a book whose longest hold is 40
    minutes wastes a generation on a predicate that can never be true.
    """
    vocab: Dict[str, Dict[str, Any]] = {}
    for name in PREDICTIVE_FEATURES:
        values = [r.get(name) for r in rows if r.get(name) is not None]
        numeric = [float(v) for v in values if isinstance(v, (int, float))
                   and not isinstance(v, bool)]
        if numeric:
            numeric.sort()
            vocab[name] = {
                "kind": "numeric",
                "n": len(numeric),
                "min": numeric[0],
                "max": numeric[-1],
                "quartiles": [
                    numeric[len(numeric) // 4],
                    numeric[len(numeric) // 2],
                    numeric[(3 * len(numeric)) // 4],
                ],
            }
            continue
        categorical = {str(v) for v in values}
        if categorical:
            vocab[name] = {"kind": "categorical", "n": len(values),
                           "values": sorted(categorical)[:12]}
    return vocab
