"""Trade ↔ brain wiring.

Two-way bridge between the trading bot and the W1z4rD brain substrate:

  observe_outcome(features_text, outcome_text)
      Post-trade. Pushes features → POOL_TEXT and outcome → POOL_ACTION
      back-to-back so the substrate forms a cross-pool binding
      (features ↔ outcome) at the next tick. After N trades, the brain
      has a learned classifier on real PnL outcomes.

  query_confidence(features_text) -> (answer, confidence)
      Pre-trade. Observes the candidate features into POOL_TEXT, calls
      /brain/integrate to decode what the brain would expect in
      POOL_ACTION, returns the decoded text + the integrated confidence.
      Bot uses confidence to size the trade (Kelly-fractional) — no
      binary 70% gate.

Keep-alive HTTP connection (urllib creates a fresh TCP per request on
Windows; that turns a 60 ms brain call into a 4.5 s one). Same pattern
as wizard_trainer._push_texts_brain.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
from http.client import HTTPConnection, BadStatusLine, RemoteDisconnected
from typing import Optional, Tuple
from urllib.parse import urlparse

# Pool ids — match wizard_trainer._POOL_TEXT / _POOL_ACTION.
POOL_TEXT = 1
POOL_ACTION = 2


def _b64url(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii").rstrip("=")


def _b64url_decode(s: str) -> str:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad).decode("utf-8", errors="replace")


class BrainBridge:
    def __init__(self, endpoint: Optional[str] = None, timeout: float = 30.0) -> None:
        ep = endpoint or os.getenv("BRAIN_ENDPOINT", "http://127.0.0.1:8090")
        u = urlparse(ep)
        self._host = u.hostname or "127.0.0.1"
        self._port = u.port or 8090
        # A loaded brain (3M+ concepts) routinely takes 1-5s per
        # /brain/observe — emergence-check scan over recent_atoms. The
        # default 5s was tripping on every other call; 30s is the safe
        # ceiling and individual calls are still bounded.
        self._timeout = timeout
        self._lock = threading.Lock()
        self._conn: Optional[HTTPConnection] = None
        self._failed_at: float = 0.0
        # Retry one failed-call backoff fast — the brain is usually
        # transient-slow, not permanently down.
        self._backoff = 5.0

    def _reset(self) -> None:
        try:
            if self._conn:
                self._conn.close()
        except Exception:
            pass
        self._conn = HTTPConnection(self._host, self._port, timeout=self._timeout)

    def _ensure(self) -> bool:
        if self._failed_at and (time.time() - self._failed_at) < self._backoff:
            return False
        if self._conn is None:
            try:
                self._reset()
            except Exception:
                self._failed_at = time.time()
                return False
        return True

    def _post(self, path: str, payload: bytes) -> Optional[bytes]:
        if not self._ensure():
            return None
        try:
            self._conn.request("POST", path, payload,
                               {"Content-Type": "application/json"})
            r = self._conn.getresponse()
            return r.read()
        except (BadStatusLine, RemoteDisconnected, ConnectionError, OSError, TimeoutError):
            # One reset + retry.
            try:
                self._reset()
                self._conn.request("POST", path, payload,
                                   {"Content-Type": "application/json"})
                r = self._conn.getresponse()
                return r.read()
            except Exception:
                self._failed_at = time.time()
                self._conn = None
                return None
        except Exception:
            self._failed_at = time.time()
            return None

    def _observe(self, pool_id: int, text: str) -> bool:
        payload = json.dumps({"pool_id": pool_id, "frame": _b64url(text)}).encode("utf-8")
        return self._post("/brain/observe", payload) is not None

    def _tick(self) -> bool:
        return self._post("/brain/tick", b"") is not None

    def observe_outcome(self, features_text: str, outcome_text: str) -> bool:
        """Push features → POOL_TEXT and outcome → POOL_ACTION, then tick.

        Both observes happen inside the same brain tick window, so the
        fabric grows a cross-pool terminal between any neuron firing in
        the features pool and any neuron firing in the action pool.
        Recurrence of (similar features → similar outcome) lifts that
        terminal toward a binding.
        """
        with self._lock:
            if not (self._observe(POOL_TEXT, features_text)
                    and self._observe(POOL_ACTION, outcome_text)
                    and self._tick()):
                return False
        return True

    def query_confidence(self, features_text: str) -> Tuple[Optional[str], float]:
        """Return (decoded_action_text_or_None, integrated_confidence).

        Observes features into POOL_TEXT then calls /brain/integrate
        which decodes the strongest trained binding into POOL_ACTION
        bytes. The integrate response carries `integrated_confidence`
        (combined fabric + EEM + annealer signal). Callers use this to
        size trades.
        """
        with self._lock:
            if not self._observe(POOL_TEXT, features_text):
                return None, 0.0
            payload = json.dumps({
                "query_pool":  POOL_TEXT,
                "target_pool": POOL_ACTION,
            }).encode("utf-8")
            body = self._post("/brain/integrate", payload)
            if body is None:
                return None, 0.0
        try:
            data = json.loads(body.decode("utf-8", errors="replace"))
        except Exception:
            return None, 0.0
        ans_b64 = data.get("answer")
        answer = _b64url_decode(ans_b64) if ans_b64 else None
        conf = float(data.get("integrated_confidence") or 0.0)
        return answer, conf


# --- features + outcome formatting ----------------------------------------
# The substrate doesn't care about the literal format — it only cares that
# the same situation maps to the same byte string so atoms recur. Stable
# canonical formatting matters more than information density.

def features_text(
    *,
    side: str,
    symbol: str,
    chain: str,
    price: float,
    spread_bps: Optional[float] = None,
    momentum: Optional[float] = None,
    confidence: Optional[float] = None,
) -> str:
    """Canonical pre-trade features for brain observation.

    Buckets continuous values into stable atoms (price by log-decade,
    spread by bps band, momentum by sign+magnitude band). Identical
    market situations produce identical strings — that's what gives the
    substrate something to compress.
    """
    import math
    def bucket(x: Optional[float], grid: list[float]) -> str:
        if x is None or not math.isfinite(x):
            return "na"
        for lo, hi, label in zip(grid[:-1], grid[1:], [f"b{i}" for i in range(len(grid)-1)]):
            if lo <= x < hi:
                return label
        return "bX"
    price_bucket = bucket(price, [0, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1e9])
    spread_bucket = bucket(spread_bps, [0, 5, 10, 25, 50, 100, 250, 1000])
    mom_bucket = bucket(momentum, [-1e9, -0.05, -0.01, -0.001, 0.001, 0.01, 0.05, 1e9])
    conf_bucket = bucket(confidence, [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01])
    return (
        f"trade {side.lower()} {symbol.lower()} {chain.lower()} "
        f"p={price_bucket} s={spread_bucket} m={mom_bucket} c={conf_bucket}"
    )


def outcome_text(pnl_pct: float) -> str:
    """Canonical post-trade outcome bucket.

    Five buckets covering the realised PnL distribution. The brain
    learns features → outcome by repeated co-firing, so the bucket count
    is the resolution of the classifier.
    """
    if pnl_pct >= 0.02:
        return "outcome win_big"
    if pnl_pct >= 0.002:
        return "outcome win"
    if pnl_pct >= -0.002:
        return "outcome flat"
    if pnl_pct >= -0.02:
        return "outcome loss"
    return "outcome loss_big"


# Process-singleton — the bot is one process; one keep-alive connection is
# all we need. Created on first use.
_BRIDGE: Optional[BrainBridge] = None


def get_bridge() -> BrainBridge:
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = BrainBridge()
    return _BRIDGE
