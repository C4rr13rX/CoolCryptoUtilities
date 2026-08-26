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


def _event_loop_is_running() -> bool:
    """True when called from inside a running asyncio event loop."""
    try:
        import asyncio

        asyncio.get_running_loop()
        return True
    except Exception:
        return False


def _brain_blocking_allowed() -> bool:
    """Escape hatch: allow blocking brain calls on the loop anyway."""
    import os

    return os.getenv("BRAIN_BRIDGE_ALLOW_BLOCKING", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }

# Pool ids — must match the brain the node is running and
# wizard_trainer's env-driven pools (same env names used here).
#
# The node now runs the market identity spec
# (W1z4rDV1510n/brains/market_small.identity.toml):
#   ohlcv   = 1  (sensory input — features go here)
#   news    = 2  (sensory input — wizard_trainer pushes news here)
#   outcome = 3  (action — the decode target for predictions)
# The default multimodal topology used POOL_ACTION=4; bindings trained
# against that brain live in a different pool space, so switching
# topologies means retraining (the supervisor scripts repopulate).
POOL_TEXT = int(os.getenv("WIZARD_MARKET_INPUT_POOL", "1"))
POOL_ACTION = int(os.getenv("WIZARD_MARKET_OUTCOME_POOL", "3"))


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
        #: Queries skipped because an event loop was running.
        self._skipped_in_loop = 0
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
        # Honour backoff only when the failure is fresh AND we've had a
        # *streak* of failures — single slow call shouldn't black-hole
        # the next minute of training pushes. The supervisor's success
        # rate is what matters; bots in the live trade path also benefit
        # from observe attempts even when one fails.
        if self._failed_at and (time.time() - self._failed_at) < self._backoff:
            # Allow a probe through every backoff interval so the bridge
            # can recover quickly when the brain transient-slowed.
            self._failed_at = 0.0
        if self._conn is None:
            try:
                self._reset()
            except Exception:
                self._failed_at = time.time()
                return False
        return True

    def _post(self, path: str, payload: bytes) -> Optional[bytes]:
        # Never block a running event loop.
        #
        # This is a synchronous http.client call with a 30s timeout, and it is
        # reached from `bot.py::_handle_sample`, which the market streams await.
        # A py-spy dump of production caught the loop parked here:
        #
        #     readinto (socket.py:719)          <- blocking socket read
        #     _post (brain_bridge.py)
        #     query_confidence -> _brain_record_entry -> _handle_sample
        #     run_forever (asyncio/base_events.py:683)
        #
        # Every market stream shares that loop, so one slow brain query froze
        # price collection for ALL symbols. Measured effect: writes arriving in
        # bursts ~12 minutes apart -- 7 symbols at an identical timestamp, then
        # nothing -- against an isolated stream's 1/second.
        #
        # The brain is an advisory signal; the price feed is the product. When
        # a loop is running, skip the query rather than stall ingestion. The
        # caller already treats a None answer as "no opinion".
        if _event_loop_is_running() and not _brain_blocking_allowed():
            self._skipped_in_loop += 1
            return None
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

    def train_binding(self, features_text: str, outcome_text: str) -> bool:
        """Supervised feature→outcome binding via /brain/consolidate.

        Unlike observe_outcome (Hebbian co-firing inside one tick), this
        is the explicit paired-training surface: the node wires the
        features frame in POOL_TEXT to the outcome frame in POOL_ACTION
        and reports whether the binding consolidated.

        Honors the node's ingest backpressure: when the machine is below
        its politeness RAM floor the node replies backpressure=true, and
        this method sleeps and retries (bounded) so training slows down
        instead of losing pairs or suffocating the host.
        """
        payload = json.dumps({
            "input_pool":    POOL_TEXT,
            "input_frame":   _b64url(features_text),
            "outcome_pool":  POOL_ACTION,
            "outcome_frame": _b64url(outcome_text),
        }).encode("utf-8")
        max_backoff_attempts = int(os.getenv("WIZARD_BACKPRESSURE_RETRIES", "30"))
        for _attempt in range(max(1, max_backoff_attempts)):
            with self._lock:
                body = self._post("/brain/consolidate", payload)
            if body is None:
                return False
            try:
                data = json.loads(body.decode("utf-8", errors="replace"))
            except Exception:
                return False
            if data.get("backpressure"):
                delay = min(30.0, float(data.get("retry_after_ms") or 2000) / 1000.0)
                time.sleep(delay)
                continue
            return bool(data.get("consolidated") is True)
        return False

    def predict_outcome(self, features_text: str) -> Tuple[Optional[str], float]:
        """Read-only prediction via /brain/predict.

        Query activation is never admitted to the learning moment — the
        node activates the features frame, reads the strongest POOL_ACTION
        binding, then clears the prediction activation. Safe for held-out
        evaluation (no leakage of test features into training state).
        """
        payload = json.dumps({
            "query_pool":  POOL_TEXT,
            "target_pool": POOL_ACTION,
            "frame":       _b64url(features_text),
        }).encode("utf-8")
        with self._lock:
            body = self._post("/brain/predict", payload)
        if body is None:
            return None, 0.0
        try:
            data = json.loads(body.decode("utf-8", errors="replace"))
        except Exception:
            return None, 0.0
        ans_b64 = data.get("answer")
        answer = _b64url_decode(ans_b64) if ans_b64 else None
        conf = float(data.get("integrated_confidence") or data.get("confidence") or 0.0)
        return answer, conf

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


# Canonical five-bucket labels and their byte-disjoint brain tokens.
# The substrate has no tokenizer — atoms are bytes — so "outcome loss_big"
# CONTAINS "outcome loss" and the frequent class's binding mass swallows the
# rare class on decode (measured 2026-07-09: every recall miss on the OHLCV
# corpus was loss_big→loss / win_big→win; disjoint tokens took train recall
# from 96% to 100%). Each token shares no substring with any other.
OUTCOME_TOKENS = {
    "win_big": "surge",
    "win": "gain",
    "flat": "steady",
    "loss": "drop",
    "loss_big": "plunge",
}
_TOKEN_OUTCOMES = {v: k for k, v in OUTCOME_TOKENS.items()}


def outcome_text(pnl_pct: float) -> str:
    """Canonical post-trade outcome bucket.

    Five buckets covering the realised PnL distribution. The brain
    learns features → outcome by repeated co-firing, so the bucket count
    is the resolution of the classifier. Emits byte-disjoint tokens —
    see OUTCOME_TOKENS for why.
    """
    if pnl_pct >= 0.02:
        return f"outcome {OUTCOME_TOKENS['win_big']}"
    if pnl_pct >= 0.002:
        return f"outcome {OUTCOME_TOKENS['win']}"
    if pnl_pct >= -0.002:
        return f"outcome {OUTCOME_TOKENS['flat']}"
    if pnl_pct >= -0.02:
        return f"outcome {OUTCOME_TOKENS['loss']}"
    return f"outcome {OUTCOME_TOKENS['loss_big']}"


def parse_outcome(answer: Optional[str]) -> Optional[str]:
    """Map a brain answer back to the canonical label.

    Understands both the disjoint tokens and the legacy win/loss words
    (checked longest-first so 'loss_big' isn't shadowed by 'loss') for
    brains trained before the 2026-07 token change.
    """
    value = (answer or "").lower()
    for token, canonical in _TOKEN_OUTCOMES.items():
        if token in value:
            return canonical
    for legacy in ("win_big", "loss_big", "win", "flat", "loss"):
        if legacy in value:
            return legacy
    return None


# Process-singleton — the bot is one process; one keep-alive connection is
# all we need. Created on first use.
_BRIDGE: Optional[BrainBridge] = None


def get_bridge() -> BrainBridge:
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = BrainBridge()
    return _BRIDGE
