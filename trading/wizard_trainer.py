"""trading/wizard_trainer.py — W1z4rD Vision Node integration for OHLCV training.

Formats OHLCV price + volume data as natural-language text and pushes it to
the W1z4rD merged main node so the Hebbian brain substrate can learn
market-domain associations (price levels, trend names, volatility vocabulary)
under the same cross-pool pipeline used for every other corpus.

Two modes (per `WIZARD_USE_BRAIN_PREFIX`):
  * BRAIN MODE (default, "1") — push texts via the canonical Phase A-E
    surface:
        POST /brain/observe {pool_id: 1, frame: <b64url(text)>}     # text pool
        POST /brain/tick                                             # close moment
    Regime queries hit POST /brain/integrate with the formatted query
    text observed into the text pool.  This routes the training through
    the same substrate the wizard chat / C0d3rV2 agent uses, so
    market vocabulary integrates with the rest of the brain's
    knowledge graph (EEM facts, hypothesis queue, etc.).
  * LEGACY MODE ("0") — push via POST /neuro/train and query via
    POST /neuro/query (the legacy crates/core NeuroRuntime path).
    Kept so existing trading-pipeline state on legacy fabric
    snapshots stays trainable while the brain catches up.

The trading server doesn't have to be running for this module to be
importable; it just sits idle until the trading pipeline begins
processing OHLCV data, at which point push_ohlcv_batch / query_regime
fire over HTTP.

Usage:
    from trading.wizard_trainer import WizardTrainer

    trainer = WizardTrainer()
    # After loading OHLCV samples:
    trainer.push_ohlcv_batch(symbol="ETH/USDC", samples=[(ts, price, vol), ...])
    # Before inference:
    regime = trainer.query_regime(symbol="ETH/USDC", current_price=1850.0)
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_WIZARD_ENDPOINT = os.getenv("WIZARD_NODE_URL", "http://localhost:8090")
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Throttle: don't push more than N texts per batch to the neuro endpoint
_TRAIN_BATCH_MAX = int(os.getenv("WIZARD_TRAIN_BATCH_MAX", "64"))
# TTL for probing the node; skip training if offline
_PROBE_CACHE_TTL = 60.0

# Market brain pool ids. They are configurable so another deployment can use
# the same Django integration without changing source.
_POOL_TEXT = int(os.getenv("WIZARD_MARKET_INPUT_POOL", "1"))
_POOL_NEWS = int(os.getenv("WIZARD_MARKET_NEWS_POOL", "2"))
_POOL_ACTION = int(os.getenv("WIZARD_MARKET_OUTCOME_POOL", "3"))


def _brain_mode() -> bool:
    """True (default) -> route training through /brain/observe + /brain/tick.
    False -> legacy /neuro/train.  Same env-var convention as wizard_session.
    """
    raw = os.getenv("WIZARD_USE_BRAIN_PREFIX", "1").strip().lower()
    return raw not in {"0", "false", "no"}


class WizardTrainer:
    """
    Pushes OHLCV summary text to the W1z4rD neuro/train endpoint and queries
    it for market-regime context before inference.
    """

    def __init__(self, endpoint: str = _WIZARD_ENDPOINT) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._online: Optional[bool] = None
        self._probe_ts: float = 0.0
        # Non-blocking regime cache.  Strategies in the scheduler call
        # cached_regime() instead of query_regime() so the brain HTTP
        # round-trip never blocks trade evaluation.  Background daemons
        # refresh entries on a per-symbol cooldown.
        self._regime_lock = threading.Lock()
        self._regime_cache: Dict[str, "BrainSignal"] = {}
        self._regime_inflight: Dict[str, float] = {}  # symbol -> launch ts
        self._latest_candles: Dict[str, Dict[str, float]] = {}
        try:
            self._regime_ttl = float(os.getenv("WIZARD_REGIME_TTL_SEC", "20"))
        except Exception:
            self._regime_ttl = 20.0
        try:
            self._regime_timeout = float(os.getenv("WIZARD_REGIME_TIMEOUT_SEC", "1.5"))
        except Exception:
            self._regime_timeout = 1.5
        # Capability + observability: set when the node 404s the learning
        # endpoint; last_push_stats feeds health displays.
        self._consolidate_unsupported_ts: float = 0.0
        self.last_push_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Node health
    # ------------------------------------------------------------------

    def is_online(self) -> bool:
        now = time.time()
        if self._online is not None and now - self._probe_ts < _PROBE_CACHE_TTL:
            return self._online
        try:
            url = f"{self._endpoint}/health"
            with urllib.request.urlopen(url, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            status_raw = str(data.get("status") or "").strip().lower()
            self._online = bool(status_raw in ("ok", "online") or data.get("version") or data.get("node_id"))
        except Exception:
            self._online = False
        self._probe_ts = now
        return bool(self._online)

    def status(self) -> Dict[str, Any]:
        """Feeder health for readiness reports / dashboards."""
        return {
            "endpoint": self._endpoint,
            "online": bool(self._online),
            "consolidate_supported": not bool(self._consolidate_unsupported_ts),
            "last_push": dict(self.last_push_stats),
        }

    # ------------------------------------------------------------------
    # OHLCV → text corpus
    # ------------------------------------------------------------------

    # Bucket edges for quantizing continuous values into RECURRING tokens.
    # This is the fix for unbounded brain growth: the old frame embedded a
    # unique timestamp and full-precision OHLCV, so every frame was a new
    # byte string -> new atoms -> new concepts, forever (the brain never saw
    # a pattern recur, so it could never consolidate/plateau). Quantized
    # tokens mean the same market SITUATION emits the same bytes, atoms
    # recur, concept formation saturates, and RAM bounds itself by design.
    _RET_EDGES = (-.02, -.01, -.005, -.002, -.0008, .0008, .002, .005, .01, .02)
    _RANGE_EDGES = (.001, .002, .004, .007, .012, .02, .04)
    _POS_EDGES = (.12, .25, .38, .5, .62, .75, .88)

    @staticmethod
    def _outcome_token(ret: float) -> str:
        """Byte-disjoint outcome label (shares no substring across classes,
        so the substrate can't confuse loss for loss_big on decode)."""
        if ret >= 0.02:
            return "surge"
        if ret >= 0.002:
            return "gain"
        if ret >= -0.002:
            return "steady"
        if ret >= -0.02:
            return "drop"
        return "plunge"

    @staticmethod
    def _bucket(value: float, edges: Tuple[float, ...]) -> int:
        lo = 0
        for e in edges:
            if value < e:
                return lo
            lo += 1
        return lo

    def _format_ohlcv_sample(self, symbol: str, candle: Dict[str, float],
                             trend: str, vol_regime: str) -> str:
        """Quantized, RECURRING feature frame for one candle.

        No timestamp and no raw prices — only bucketed intra-bar shape plus
        the caller's multi-bar trend/volatility labels. The identical format
        is used for training (push) and inference (query) so the brain
        matches situations. See _RET_EDGES for why quantization matters.
        """
        base = symbol.split("/")[0].split("_")[0].upper()
        quote = symbol.split("/")[-1].split("_")[0].upper() if "/" in symbol else "USD"
        o = float(candle.get("open") or 0.0)
        h = float(candle.get("high") or 0.0)
        low = float(candle.get("low") or 0.0)
        c = float(candle.get("close") or 0.0)
        denom = o if o > 0 else (c or 1e-12)
        bar_ret = (c - o) / denom
        bar_range = (h - low) / denom if denom else 0.0
        pos = ((c - low) / (h - low)) if h > low else 0.5
        return (
            f"market={base}/{quote} "
            f"barret=b{self._bucket(bar_ret, self._RET_EDGES)} "
            f"range=b{self._bucket(bar_range, self._RANGE_EDGES)} "
            f"pos=b{self._bucket(pos, self._POS_EDGES)} "
            f"trend={trend.replace(' ', '_')} "
            f"volatility={vol_regime}"
        )

    @staticmethod
    def _normalise_candle(sample: Any) -> Optional[Dict[str, float]]:
        try:
            if isinstance(sample, dict):
                close = float(sample.get("close") or sample.get("price") or 0)
                return {
                    "timestamp": float(sample.get("timestamp") or sample.get("ts") or 0),
                    "open": float(sample.get("open") or close),
                    "high": float(sample.get("high") or close),
                    "low": float(sample.get("low") or close),
                    "close": close,
                    "volume": float(sample.get("net_volume") or sample.get("volume") or 0),
                } if close > 0 else None
            values = list(sample)
            if len(values) >= 6:
                ts, opn, high, low, close, volume = values[:6]
            elif len(values) >= 3:
                ts, close, volume = values[:3]
                opn = high = low = close
            else:
                return None
            result = {"timestamp": float(ts), "open": float(opn), "high": float(high),
                      "low": float(low), "close": float(close), "volume": float(volume)}
            return result if result["close"] > 0 else None
        except (TypeError, ValueError, OverflowError):
            return None

    def _classify_trend(self, prices: Sequence[float]) -> str:
        if len(prices) < 3:
            return "unknown"
        slope = (prices[-1] - prices[0]) / (abs(prices[0]) + 1e-9)
        if slope > 0.03:
            return "strong uptrend"
        if slope > 0.01:
            return "mild uptrend"
        if slope < -0.03:
            return "strong downtrend"
        if slope < -0.01:
            return "mild downtrend"
        return "sideways"

    def _classify_vol(self, prices: Sequence[float]) -> str:
        if len(prices) < 3:
            return "unknown"
        rets = [(prices[i] - prices[i-1]) / (abs(prices[i-1]) + 1e-9) for i in range(1, len(prices))]
        std = math.sqrt(sum(r*r for r in rets) / max(len(rets), 1))
        if std > 0.04:
            return "extreme"
        if std > 0.02:
            return "high"
        if std > 0.005:
            return "moderate"
        return "low"

    def push_ohlcv_batch(
        self,
        symbol: str,
        samples: Sequence[Any],
        *,
        max_items: int = _TRAIN_BATCH_MAX,
    ) -> int:
        """
        Format up to `max_items` OHLCV samples as text and push to /neuro/train.
        Returns the number of items actually pushed, or 0 if the node is offline.
        """
        if not self.is_online():
            return 0
        if not samples:
            return 0

        # Downsample to at most max_items evenly spaced
        if len(samples) > max_items:
            step = len(samples) / max_items
            samples = [samples[int(i * step)] for i in range(max_items)]

        candles = [c for c in (self._normalise_candle(s) for s in samples) if c]
        if len(candles) < 2:
            return 0
        pairs: List[Tuple[str, str]] = []
        for index, (current, future) in enumerate(zip(candles, candles[1:])):
            history = [c["close"] for c in candles[max(0, index - 7):index + 1]]
            trend = self._classify_trend(history)
            vol_regime = self._classify_vol(history)
            ret = (future["close"] - current["close"]) / max(abs(current["close"]), 1e-12)
            feature = self._format_ohlcv_sample(symbol, current, trend, vol_regime)
            # Quantized, byte-disjoint outcome (surge/gain/steady/drop/plunge)
            # — the same bounded-token scheme the trading path uses. The old
            # `future_return={ret:.9g} future_close={...:.9g}` embedded unique
            # floats, so every outcome minted new atoms too (double growth).
            outcome = f"outcome {self._outcome_token(ret)}"
            pairs.append((feature, outcome))
        latest_history = [c["close"] for c in candles[-8:]]
        self._latest_candles[symbol] = {
            **candles[-1], "trend": self._classify_trend(latest_history),
            "volatility": self._classify_vol(latest_history),
        }
        return self._consolidate_pairs(pairs)

    def push_news_batch(
        self,
        items: List[Dict[str, Any]],
        *,
        max_items: int = 200,
    ) -> int:
        """Push news articles into the brain as text observations.

        Each item is reduced to `[NEWS][SYMBOL] title — sentiment` so
        the brain's text pool sees the headline next to symbol context.
        Returns the count actually pushed.  When `items` is large we
        downsample to `max_items` evenly so a 3-year-spanning corpus
        load still completes in seconds.
        """
        if not self.is_online():
            return 0
        if not items:
            return 0
        if len(items) > max_items:
            step = len(items) / max_items
            items = [items[int(i * step)] for i in range(max_items)]
        texts: List[str] = []
        for it in items:
            try:
                title = str(it.get("title") or it.get("headline") or "").strip()
                if not title:
                    continue
                tokens = it.get("tokens") or it.get("symbols") or []
                if isinstance(tokens, (list, tuple)):
                    sym_str = ",".join(str(t).upper() for t in tokens if t)[:80]
                else:
                    sym_str = str(tokens).upper()[:80]
                sentiment = it.get("sentiment") or ""
                source = it.get("source") or ""
                head = f"[NEWS] [{sym_str}] " if sym_str else "[NEWS] "
                tail = f" :: {sentiment}" if sentiment else ""
                src = f" ({source})" if source else ""
                texts.append((head + title + src + tail)[:480])
            except Exception:
                continue
        if not texts:
            return 0
        return self._push_texts(texts)

    def push_market_summary(
        self,
        symbol: str,
        summary: Dict[str, Any],
    ) -> bool:
        """
        Push a richer market summary dict as a text blob to /neuro/train.
        `summary` should contain keys like price, expected_return, direction, confidence.
        """
        if not self.is_online():
            return False
        base = symbol.split("/")[0].upper()
        price = float(summary.get("price", 0.0))
        ret = float(summary.get("expected_return", 0.0))
        direction = "bullish" if ret > 0 else "bearish"
        confidence = float(summary.get("confidence", 0.5))
        conf_word = "high" if confidence > 0.7 else ("moderate" if confidence > 0.5 else "low")
        horizon = str(summary.get("horizon", "unknown"))
        text = (
            f"{base} {direction} forecast for {horizon} horizon. "
            f"Expected return {ret:.2%}. "
            f"Model confidence {conf_word} ({confidence:.0%}). "
            f"Current price {price:.6g}."
        )
        return self._push_texts([text]) > 0

    # ------------------------------------------------------------------
    # Regime query
    # ------------------------------------------------------------------

    def query_regime(
        self,
        symbol: str,
        current_price: float,
        *,
        timeout: float = 2.0,
    ) -> Optional[str]:
        """
        Ask the W1z4rD node for a market-regime context string.
        Returns a short description like "strong uptrend, high volatility"
        or None if the node is offline or doesn't respond.
        """
        if not self.is_online():
            return None
        candle = dict(self._latest_candles.get(symbol) or {
            "timestamp": time.time(), "open": current_price, "high": current_price,
            "low": current_price, "close": current_price, "volume": 0.0,
        })
        candle["close"] = float(current_price)
        query = self._format_ohlcv_sample(
            symbol, candle, str(candle.get("trend", "unknown")),
            str(candle.get("volatility", "unknown")))

        if _brain_mode():
            return self._query_regime_brain(query, timeout)
        return self._query_regime_legacy(query, timeout)

    def predict_horizon(
        self,
        symbol: str,
        current_price: float,
        *,
        target_margin: float = 0.02,
        horizon_hours: float = 4.0,
        timeout: float = 2.0,
    ) -> Optional[Dict[str, Any]]:
        """Ask the brain whether `symbol` is likely to rise by at least
        `target_margin` within `horizon_hours` from `current_price`.

        Returns a dict with keys:
          - direction_prob:  0..1 probability the move happens
          - target_price:    current_price * (1 + target_margin)
          - horizon_hours:   echoed input
          - regime_text:     raw brain answer for transparency
          - confidence:      [0,1] proxy from keyword density
        Or None if the brain is offline or returned nothing.

        This is the "buy low, expect to sell high at +M% within T"
        question wrapped as a brain query.  Consumers can iterate over
        a grid of (margin, horizon) tuples to find the brain's
        preferred trade.
        """
        if not self.is_online():
            return None
        if not (current_price and current_price > 0):
            return None
        base = symbol.split("/")[0].split("-")[0].upper()
        target_price = float(current_price) * (1.0 + float(target_margin))
        h = float(horizon_hours)
        query = (
            f"{base} price {current_price:.6g}. "
            f"Will {base} reach {target_price:.6g} within {h:.1f} hours? "
            f"Trend? Volatility?"
        )
        text = (self._query_regime_brain(query, timeout)
                if _brain_mode() else self._query_regime_legacy(query, timeout))
        if not text:
            return None
        try:
            direction, confidence = _parse_regime_text(text)
        except Exception:
            direction, confidence = 0.5, 0.0
        return {
            "symbol":         symbol,
            "current_price":  float(current_price),
            "target_price":   target_price,
            "target_margin":  float(target_margin),
            "horizon_hours":  h,
            "direction_prob": float(direction),
            "confidence":     float(confidence),
            "regime_text":    text,
        }

    def _query_regime_brain(self, query: str, timeout: float) -> Optional[str]:
        """Read-only market prediction; the query is never learned."""
        try:
            frame = base64.urlsafe_b64encode(
                query.encode("utf-8")
            ).decode("ascii").rstrip("=")
            predict_url = f"{self._endpoint}/brain/predict"
            predict_payload = json.dumps({
                "query_pool":  _POOL_TEXT,
                "target_pool": _POOL_ACTION,
                "frame": frame,
            }).encode("utf-8")
            req = urllib.request.Request(
                predict_url, data=predict_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            answer_b64 = data.get("answer")
            if not answer_b64:
                return None
            pad = "=" * (-len(answer_b64) % 4)
            try:
                return base64.urlsafe_b64decode(answer_b64 + pad).decode(
                    "utf-8", errors="replace").strip() or None
            except Exception:
                return None
        except Exception:
            return None

    def _consolidate_pairs(self, pairs: Sequence[Tuple[str, str]]) -> int:
        """Persist only feature frames paired with observed future outcomes.

        Hardened after the 2026-07 incident where 4952/5000 pushes silently
        failed: the running node was an old build without /brain/consolidate
        (404) and a single transport error aborted the whole batch. Now:
          - a 404 marks the capability missing, logs once, and stops the
            batch (retried after the probe TTL, in case the node upgrades);
          - transport errors reset the keep-alive connection and continue;
          - failure stats are kept on self.last_push_stats for dashboards.
        """
        from urllib.parse import urlparse
        from http.client import HTTPConnection
        u = urlparse(self._endpoint)
        host, port = u.hostname or "127.0.0.1", u.port or 8090
        now = time.time()
        if self._consolidate_unsupported_ts and now - self._consolidate_unsupported_ts < _PROBE_CACHE_TTL * 5:
            self.last_push_stats = {"pushed": 0, "failed": len(pairs), "reason": "endpoint_unsupported"}
            return 0
        conn = HTTPConnection(host, port, timeout=8)
        pushed = 0
        failed = 0
        reason = ""
        try:
            for feature, outcome in pairs:
                enc = lambda text: base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")
                payload = json.dumps({
                    "input_pool": _POOL_TEXT, "input_frame": enc(feature),
                    "outcome_pool": _POOL_ACTION, "outcome_frame": enc(outcome),
                }).encode()
                try:
                    conn.request("POST", "/brain/consolidate", payload,
                                 {"Content-Type": "application/json"})
                    response = conn.getresponse()
                    raw = response.read().decode("utf-8", errors="replace") or "{}"
                except Exception:
                    # transport hiccup: reset the keep-alive socket, keep going
                    failed += 1
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = HTTPConnection(host, port, timeout=8)
                    continue
                if response.status == 404:
                    # Node build without the learning surface — pushing more
                    # is pointless until the node is upgraded/restarted.
                    self._consolidate_unsupported_ts = now
                    failed += len(pairs) - pushed - failed
                    reason = "endpoint_unsupported"
                    print(
                        "[wizard-trainer] node at %s:%s lacks /brain/consolidate "
                        "(404) — brain learning disabled until the node is rebuilt/restarted"
                        % (host, port)
                    )
                    break
                try:
                    body = json.loads(raw)
                except Exception:
                    body = {}
                if body.get("backpressure"):
                    # The node's machine is below its politeness RAM floor —
                    # stop the batch; the remaining pairs simply wait for the
                    # next feeder cycle. Not a failure: the brain is asking
                    # the data stream to slow down.
                    reason = "backpressure"
                    break
                if response.status == 200 and body.get("consolidated") is True:
                    pushed += 1
                else:
                    failed += 1
        finally:
            try:
                conn.close()
            except Exception:
                pass
        self.last_push_stats = {"pushed": pushed, "failed": failed, "reason": reason or ("ok" if pushed else "unknown")}
        return pushed

    def _query_regime_legacy(self, query: str, timeout: float) -> Optional[str]:
        """Legacy /neuro/query path on the crates/core NeuroRuntime
        fabric."""
        try:
            url = f"{self._endpoint}/neuro/query"
            payload = json.dumps({"query": query, "top_k": 3}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            results = data.get("results") or []
            if results:
                # Return the top match's text as context
                return str(results[0].get("text", "")).strip() or None
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Internal HTTP push
    # ------------------------------------------------------------------

    def _push_texts(self, texts: List[str]) -> int:
        if not texts:
            return 0
        if _brain_mode():
            return self._push_texts_brain(texts)
        return self._push_texts_legacy(texts)

    def _push_texts_brain(self, texts: List[str]) -> int:
        """Unlabelled text is intentionally not learned in brain mode.

        Legacy callers may still submit text, but the prediction/consolidation
        architecture requires an external outcome before a Hebbian tick.
        """
        return 0

    def _push_texts_brain_unsafe_legacy(self, texts: List[str]) -> int:
        """Deprecated observe/tick implementation retained for migration.
        cycle: observe → tick.  Uses a single keep-alive http.client
        connection — urllib was creating a fresh TCP connection per
        call on Windows, which made each push ~4.5 s even though the
        brain endpoint itself responds in 60 ms.  With keep-alive the
        per-text cost drops to ~60 ms (~75x speedup)."""
        from urllib.parse import urlparse
        from http.client import HTTPConnection, BadStatusLine, RemoteDisconnected
        u = urlparse(self._endpoint)
        host = u.hostname or "127.0.0.1"
        port = u.port or 8090
        pushed = 0
        conn = HTTPConnection(host, port, timeout=5)
        try:
            for text in texts:
                try:
                    frame = base64.urlsafe_b64encode(
                        text.encode("utf-8")
                    ).decode("ascii").rstrip("=")
                    payload = json.dumps({
                        "pool_id": _POOL_TEXT,
                        "frame":   frame,
                    }).encode("utf-8")
                    # observe
                    conn.request("POST", "/brain/observe", payload,
                                 {"Content-Type": "application/json"})
                    r = conn.getresponse(); r.read()
                    # tick
                    conn.request("POST", "/brain/tick", b"",
                                 {"Content-Type": "application/json"})
                    r = conn.getresponse(); r.read()
                    pushed += 1
                except (BadStatusLine, RemoteDisconnected, ConnectionError) as e:
                    # Reset connection on transport error and retry once
                    try: conn.close()
                    except Exception: pass
                    conn = HTTPConnection(host, port, timeout=5)
                    continue
                except urllib.error.URLError:
                    self._online = False
                    self._probe_ts = 0.0
                    break
                except Exception:
                    continue
        finally:
            try: conn.close()
            except Exception: pass
        return pushed

    def _push_texts_brain_OLD_PER_REQUEST(self, texts: List[str]) -> int:
        """Old urllib-based path retained for reference; do not call."""
        observe_url = f"{self._endpoint}/brain/observe"
        tick_url    = f"{self._endpoint}/brain/tick"
        pushed = 0
        for text in texts:
            try:
                frame = base64.urlsafe_b64encode(
                    text.encode("utf-8")
                ).decode("ascii").rstrip("=")
                payload = json.dumps({
                    "pool_id": _POOL_TEXT,
                    "frame":   frame,
                }).encode("utf-8")
                req = urllib.request.Request(
                    observe_url, data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    resp.read()
                tick_req = urllib.request.Request(
                    tick_url, data=b"",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(tick_req, timeout=5) as resp:
                    resp.read()
                pushed += 1
            except urllib.error.URLError:
                self._online = False
                self._probe_ts = 0.0
                break
            except Exception:
                continue
        return pushed

    def _push_texts_legacy(self, texts: List[str]) -> int:
        """Legacy /neuro/train path (crates/core NeuroRuntime).  Kept
        for back-compat with corpora already trained on the legacy
        fabric snapshot."""
        url = f"{self._endpoint}/neuro/train"
        payload = json.dumps({"texts": texts}).encode("utf-8")
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()  # consume response
            return len(texts)
        except urllib.error.URLError:
            self._online = False
            self._probe_ts = 0.0  # force re-probe next time
            return 0
        except Exception:
            return 0


# ---------------------------------------------------------------------
# Standalone OHLCV→brain feeder
# ---------------------------------------------------------------------
#
# The TF training cycle is the only existing path that pushed OHLCV
# samples to the brain.  When TF fails to load on Windows (DLL OOM,
# missing dep, VC redist drift) the model build crashes BEFORE the
# brain push fires — so the brain silently stops learning even though
# the OHLCV corpus is up to date.
#
# This feeder is a thin background thread that doesn't import TF at
# all: it walks `data/historical_ohlcv/{chain}/*.json`, picks the
# tail N candles per file, and pushes them via push_ohlcv_batch on
# a configurable cadence.  Always runs alongside the TF path; either
# can fail independently without starving the other.

_BRAIN_FEEDER_STATE: Dict[str, Any] = {
    "thread": None,
    "stop": False,
    "last_run": 0.0,
    "last_pushed": 0,
}


# Per-file cursor: how far back in history we've already streamed.
# A cycle advances the cursor by `window_candles` per pass so the brain
# eventually sees the full corpus, not just the tail forever.  Stored in
# _BRAIN_FEEDER_STATE["cursors"] = { (chain, sym): rows_streamed }.
_BRAIN_FEEDER_STATE["cursors"] = {}
_BRAIN_FEEDER_STATE["file_offset"] = 0


def _build_training_package(
    *,
    chain: str,
    sym: str,
    rows: List[Dict[str, Any]],
    cursor: int,
    window_candles: int,
    news_lookback_hours: float,
) -> Tuple[List[Dict[str, float]], List[Dict[str, Any]], int]:
    """Construct a (ohlcv_samples, news_items, next_cursor) bundle.

    The user's "training package" concept: the brain should see a
    coherent window of price action AND any news from the same time
    period in the same cycle, so it can learn news↔price correlations.

    cursor is rows-from-newest (0 = most recent).  Each cycle we
    advance cursor by window_candles so the brain eventually walks the
    full corpus backwards in time.
    """
    n = len(rows)
    if n == 0 or cursor >= n:
        return [], [], 0  # wrap to start of newest
    end_idx = n - cursor
    start_idx = max(0, end_idx - window_candles)
    window = rows[start_idx:end_idx]
    samples: List[Dict[str, float]] = []
    for r in window:
        try:
            candle = WizardTrainer._normalise_candle(r)
            if candle:
                samples.append(candle)
        except Exception:
            continue
    # News for the same time window — READ from the cache only.  News
    # fetching is too slow (30+s per symbol against RSS+crawler
    # network paths) to do inline; it has its own background worker
    # below.  The OHLCV cycle just consumes whatever the news worker
    # has produced so far.  Set WIZARD_BRAIN_FEEDER_NEWS_INLINE=1 to
    # restore the old synchronous behavior.
    news_items: List[Dict[str, Any]] = []
    cache = _BRAIN_FEEDER_STATE.setdefault("news_cache", {})
    key = sym.split("-", 1)[0].upper()
    cached = cache.get(key)
    if cached:
        news_items = cached[1]
    elif os.getenv("WIZARD_BRAIN_FEEDER_NEWS_INLINE", "0").lower() in {"1","true","yes"}:
        try:
            from datetime import datetime, timezone, timedelta
            from services.news_lab import collect_news_for_terms
            if samples:
                window_end = datetime.fromtimestamp(samples[-1]["timestamp"], tz=timezone.utc)
                window_start = window_end - timedelta(hours=news_lookback_hours)
                result = collect_news_for_terms(
                    tokens=[key], start=window_start, end=window_end)
                news_items = result.get("items") or []
                cache[key] = (time.time(), news_items)
        except Exception:
            pass
    next_cursor = cursor + window_candles
    if next_cursor >= n:
        next_cursor = 0  # wrap so the brain re-trains on newest data eventually
    return samples, news_items, next_cursor


def _brain_feeder_loop(
    chains: Sequence[str],
    interval_sec: float,
    tail_candles: int,
    data_root: Optional[str],
) -> None:
    """Indefinite training-package feeder.

    Each cycle, for each (chain, symbol):
      1. Build a price+news bundle for the next historical window.
      2. Push the OHLCV samples to the brain's text pool.
      3. Push the news headlines to the same text pool — same cycle,
         so Hebbian co-firing wires the news↔price association.
      4. Advance the cursor; wrap to newest when we've covered the
         corpus so the brain stays current with the latest tail.
    """
    import json as _j
    from pathlib import Path as _P
    root = _P(data_root) if data_root else _REPO_ROOT / "data" / "historical_ohlcv"
    try:
        news_per_window = int(os.getenv("WIZARD_BRAIN_FEEDER_NEWS_PER_WINDOW", "16"))
        news_lookback_h = float(os.getenv("WIZARD_BRAIN_FEEDER_NEWS_LOOKBACK_H", "72"))
        max_packages = max(1, int(os.getenv("WIZARD_BRAIN_FEEDER_MAX_PACKAGES", "2")))
    except Exception:
        news_per_window, news_lookback_h, max_packages = 16, 72.0, 2
    while not _BRAIN_FEEDER_STATE.get("stop"):
        try:
            trainer = get_trainer()
            if not trainer.is_online():
                time.sleep(min(60.0, interval_sec))
                continue
            total_pushed = 0
            cursors = _BRAIN_FEEDER_STATE["cursors"]
            files = [(chain, jf) for chain in chains
                     for jf in sorted((root / chain).glob("*.json"))
                     if (root / chain).exists()]
            if files:
                offset = int(_BRAIN_FEEDER_STATE.get("file_offset", 0)) % len(files)
                selected = [files[(offset + i) % len(files)]
                            for i in range(min(max_packages, len(files)))]
                _BRAIN_FEEDER_STATE["file_offset"] = (offset + len(selected)) % len(files)
                for chain, jf in selected:
                    try:
                        sym = jf.stem.split("_", 1)[-1]
                        with jf.open("r", encoding="utf-8") as fh:
                            rows = _j.load(fh)
                        if not isinstance(rows, list) or not rows:
                            continue
                        key = (chain, sym)
                        cursor = int(cursors.get(key, 0))
                        samples, news_items, next_cursor = _build_training_package(
                            chain=chain, sym=sym, rows=rows, cursor=cursor,
                            window_candles=tail_candles,
                            news_lookback_hours=news_lookback_h)
                        cursors[key] = next_cursor
                        if samples:
                            total_pushed += trainer.push_ohlcv_batch(
                                sym, samples, max_items=max(tail_candles, 32))
                        if news_items:
                            total_pushed += trainer.push_news_batch(
                                news_items, max_items=news_per_window)
                    except Exception:
                        continue
            _BRAIN_FEEDER_STATE["last_pushed"] = total_pushed
            _BRAIN_FEEDER_STATE["last_run"] = time.time()
        except Exception:
            pass
        # Sleep in 5s slices so stop is responsive
        slept = 0.0
        while slept < interval_sec and not _BRAIN_FEEDER_STATE.get("stop"):
            time.sleep(5.0)
            slept += 5.0


def start_brain_feeder(
    *,
    chains: Sequence[str] = ("base",),
    interval_sec: Optional[float] = None,
    tail_candles: Optional[int] = None,
    data_root: Optional[str] = None,
) -> bool:
    """Start the background OHLCV→brain feeder.

    Idempotent — calling more than once is a no-op (returns False).
    Returns True on first start, False if already running or disabled.
    Tune via env:
      WIZARD_BRAIN_FEEDER_ENABLED  (default 1)
      WIZARD_BRAIN_FEEDER_INTERVAL (default 120 sec)
      WIZARD_BRAIN_FEEDER_TAIL     (default 16 candles per file per cycle)
      WIZARD_BRAIN_FEEDER_MAX_PACKAGES (default 2 files per cycle)
    """
    if os.getenv("WIZARD_BRAIN_FEEDER_ENABLED", "1").lower() in {"0", "false", "no"}:
        return False
    if _BRAIN_FEEDER_STATE.get("thread") is not None:
        t = _BRAIN_FEEDER_STATE["thread"]
        if t.is_alive():
            return False
    if interval_sec is None:
        try:
            interval_sec = float(os.getenv("WIZARD_BRAIN_FEEDER_INTERVAL", "120"))
        except Exception:
            interval_sec = 120.0
    if tail_candles is None:
        try:
            tail_candles = int(os.getenv("WIZARD_BRAIN_FEEDER_TAIL", "16"))
        except Exception:
            tail_candles = 16
    _BRAIN_FEEDER_STATE["stop"] = False
    t = threading.Thread(
        target=_brain_feeder_loop,
        args=(list(chains), float(interval_sec), int(tail_candles), data_root),
        daemon=True,
        name="wizard-brain-feeder",
    )
    t.start()
    _BRAIN_FEEDER_STATE["thread"] = t
    return True


def stop_brain_feeder() -> None:
    _BRAIN_FEEDER_STATE["stop"] = True


def _news_worker_loop(symbols_provider, interval_sec: float, lookback_h: float) -> None:
    """Background worker that pre-fetches news for active symbols and
    populates _BRAIN_FEEDER_STATE['news_cache'].  Runs at its own
    slow cadence (default 30 min) independent of the OHLCV feeder so
    the per-symbol RSS/crawler fetches don't block ingest.
    """
    from datetime import datetime, timezone, timedelta
    while not _BRAIN_FEEDER_STATE.get("stop"):
        try:
            symbols = symbols_provider() or []
            if symbols:
                try:
                    from services.news_lab import collect_news_for_terms
                    end = datetime.now(timezone.utc)
                    start = end - timedelta(hours=lookback_h)
                    cache = _BRAIN_FEEDER_STATE.setdefault("news_cache", {})
                    for sym in symbols:
                        if _BRAIN_FEEDER_STATE.get("stop"):
                            break
                        try:
                            res = collect_news_for_terms(
                                tokens=[sym], start=start, end=end)
                            cache[sym] = (time.time(), res.get("items") or [])
                        except Exception:
                            continue
                except Exception:
                    pass
        except Exception:
            pass
        slept = 0.0
        while slept < interval_sec and not _BRAIN_FEEDER_STATE.get("stop"):
            time.sleep(5.0)
            slept += 5.0


def start_news_worker(
    *,
    chains: Sequence[str] = ("base",),
    interval_sec: Optional[float] = None,
    lookback_hours: Optional[float] = None,
    data_root: Optional[str] = None,
) -> bool:
    """Spawn the background news pre-fetcher.  Idempotent."""
    if os.getenv("WIZARD_BRAIN_NEWS_WORKER_ENABLED", "1").lower() in {"0","false","no"}:
        return False
    if _BRAIN_FEEDER_STATE.get("news_thread") is not None:
        t = _BRAIN_FEEDER_STATE["news_thread"]
        if t.is_alive():
            return False
    if interval_sec is None:
        try:
            interval_sec = float(os.getenv("WIZARD_BRAIN_NEWS_INTERVAL", "1800"))
        except Exception:
            interval_sec = 1800.0
    if lookback_hours is None:
        try:
            lookback_hours = float(os.getenv("WIZARD_BRAIN_NEWS_LOOKBACK_H", "168"))
        except Exception:
            lookback_hours = 168.0
    from pathlib import Path as _P
    root = _P(data_root) if data_root else _REPO_ROOT / "data" / "historical_ohlcv"

    def _symbols():
        seen = set()
        for ch in chains:
            cdir = root / ch
            if not cdir.exists():
                continue
            for jf in cdir.glob("*.json"):
                sym = jf.stem.split("_", 1)[-1]
                base = sym.split("-", 1)[0].upper()
                if base:
                    seen.add(base)
        return sorted(seen)

    t = threading.Thread(
        target=_news_worker_loop,
        args=(_symbols, float(interval_sec), float(lookback_hours)),
        daemon=True,
        name="wizard-news-worker",
    )
    t.start()
    _BRAIN_FEEDER_STATE["news_thread"] = t
    return True


# ---------------------------------------------------------------------
# Live-tick path
# ---------------------------------------------------------------------
#
# The historical feeder above walks the OHLCV files on disk.  For live
# market data — the price ticks coming in over websocket / REST — we
# need a separate, lightweight path that pushes each tick to the brain
# as it arrives.  This used to ride on the TF prediction callback, but
# that path is gated on TF being loadable; when TF is broken we lost
# all live ingestion.  The live-tick path here has zero TF dependency.

_LIVE_TICK_STATE: Dict[str, Any] = {
    "buffers": {},          # symbol -> list of recent ticks
    "last_pushed_ts": {},   # symbol -> last brain-push timestamp
}


def push_live_tick(
    symbol: str,
    price: float,
    volume: float = 0.0,
    ts: Optional[float] = None,
    *,
    min_interval_sec: float = 1.0,
) -> bool:
    """Push a single live market tick to the brain.

    Throttled per-symbol so a 100ms tick stream doesn't slam /brain/observe.
    Default cooldown 1s per symbol (was 5s) — captures every meaningful
    price move per symbol per second across the universe.  Tune via
    WIZARD_LIVE_TICK_MIN_SEC.  Returns True if a sample was pushed.
    """
    if not symbol or not (price and price > 0):
        return False
    try:
        cooldown = float(os.getenv("WIZARD_LIVE_TICK_MIN_SEC", str(min_interval_sec)))
    except Exception:
        cooldown = min_interval_sec
    now_ts = float(ts) if ts is not None else time.time()
    last = _LIVE_TICK_STATE["last_pushed_ts"].get(symbol, 0.0)
    if (now_ts - last) < cooldown:
        return False
    try:
        trainer = get_trainer()
        if not trainer.is_online():
            return False
        buffer = _LIVE_TICK_STATE["buffers"].setdefault(symbol, [])
        buffer.append((now_ts, float(price), float(volume)))
        if len(buffer) > 2:
            del buffer[:-2]
        if len(buffer) < 2:
            return False
        if trainer.push_ohlcv_batch(symbol, list(buffer), max_items=2) < 1:
            return False
        _LIVE_TICK_STATE["last_pushed_ts"][symbol] = now_ts
        return True
    except Exception:
        return False


def brain_feeder_status() -> Dict[str, Any]:
    t = _BRAIN_FEEDER_STATE.get("thread")
    return {
        "running":     bool(t and t.is_alive()),
        "last_run":    _BRAIN_FEEDER_STATE.get("last_run", 0.0),
        "last_pushed": _BRAIN_FEEDER_STATE.get("last_pushed", 0),
    }


# ---------------------------------------------------------------------
# Brain signal cache (non-blocking parallel-strategy plumbing)
# ---------------------------------------------------------------------

class BrainSignal:
    """Parsed regime answer that the scheduler can consume directly.

    `direction_prob` is bull-leaning probability in [0, 1].
    `confidence` is the brain's own [0, 1] confidence (when surfaced
    by /brain/integrate) or a regime-keyword-derived proxy.
    `regime_text` is the raw decoded answer for logs/UI.
    `ts` is the unix time the signal was produced.
    """

    __slots__ = ("symbol", "direction_prob", "confidence", "regime_text", "ts")

    def __init__(self, symbol: str, direction_prob: float, confidence: float,
                 regime_text: str, ts: float) -> None:
        self.symbol = symbol
        self.direction_prob = float(max(0.0, min(1.0, direction_prob)))
        self.confidence = float(max(0.0, min(1.0, confidence)))
        self.regime_text = regime_text
        self.ts = float(ts)

    def is_fresh(self, ttl: float) -> bool:
        return (time.time() - self.ts) <= ttl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol":          self.symbol,
            "direction_prob":  self.direction_prob,
            "confidence":      self.confidence,
            "regime_text":     self.regime_text,
            "ts":              self.ts,
        }


_BULL_TOKENS = {"bull", "bullish", "uptrend", "future_direction=up", "rally", "long", "buy", "breakout",
                "rising", "strong", "accumulat", "support"}
_BEAR_TOKENS = {"bear", "bearish", "downtrend", "future_direction=down", "sell", "short", "breakdown",
                "falling", "weak", "distribut", "resistance", "reject"}


def _parse_regime_text(text: str) -> Tuple[float, float]:
    """Cheap regex tally of bull vs bear keywords.

    Returns (direction_prob, confidence) in [0,1].  Direction is 0.5
    when no keywords match (neutral); confidence is the share of
    matched tokens (a proxy for how on-topic the brain's reply was).
    """
    if not text:
        return 0.5, 0.0
    blob = text.lower()
    bull = sum(1 for tok in _BULL_TOKENS if tok in blob)
    bear = sum(1 for tok in _BEAR_TOKENS if tok in blob)
    total = bull + bear
    if total == 0:
        return 0.5, 0.0
    direction = bull / total
    # Total keyword density (capped at 1.0) is the confidence proxy.
    # A two-word reply with one bull token => conf 0.5; longer/denser
    # replies trend higher.
    word_count = max(1, len(re.findall(r"\w+", blob)))
    density = min(1.0, total / max(1.0, word_count / 4.0))
    return direction, density


def _trainer_cached_regime(self, symbol: str, current_price: float
                           ) -> Optional[BrainSignal]:
    """Non-blocking cache read.  Returns the latest fresh BrainSignal
    for `symbol`, or None.  Spawns a background refresher when the
    cached entry is stale and no fetch is already in-flight."""
    now = time.time()
    with self._regime_lock:
        cached = self._regime_cache.get(symbol)
        inflight_ts = self._regime_inflight.get(symbol, 0.0)
        fresh = cached is not None and cached.is_fresh(self._regime_ttl)
        # Refresh if stale AND no fetch in the last 2*TTL window.
        should_refresh = (not fresh) and (now - inflight_ts) > (2.0 * self._regime_ttl)
        if should_refresh:
            self._regime_inflight[symbol] = now
    if should_refresh:
        # Spawn a daemon thread; never blocks the caller.
        t = threading.Thread(
            target=self._refresh_regime,
            args=(symbol, float(current_price)),
            daemon=True,
            name=f"brain-regime-{symbol}",
        )
        t.start()
    return cached if cached and cached.is_fresh(self._regime_ttl) else None


def _trainer_refresh_regime(self, symbol: str, current_price: float) -> None:
    """Background refresher.  Runs in a daemon thread."""
    try:
        text = self.query_regime(symbol, current_price, timeout=self._regime_timeout)
        if not text:
            return
        direction, confidence = _parse_regime_text(text)
        signal = BrainSignal(symbol, direction, confidence, text, time.time())
        with self._regime_lock:
            self._regime_cache[symbol] = signal
    except Exception:
        # best-effort: failures are silent so they don't spam logs
        pass
    finally:
        with self._regime_lock:
            self._regime_inflight.pop(symbol, None)


# Bind the cache helpers to the class without polluting the dataclass-style
# block above (they reference instance fields added in __init__).
WizardTrainer.cached_regime = _trainer_cached_regime   # type: ignore[attr-defined]
WizardTrainer._refresh_regime = _trainer_refresh_regime  # type: ignore[attr-defined]


# Module-level singleton so callers don't need to instantiate
_default_trainer: Optional[WizardTrainer] = None


def get_trainer() -> WizardTrainer:
    global _default_trainer
    if _default_trainer is None:
        _default_trainer = WizardTrainer()
    return _default_trainer
