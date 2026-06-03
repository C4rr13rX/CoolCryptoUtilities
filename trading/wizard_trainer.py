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
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

_WIZARD_ENDPOINT = os.getenv("WIZARD_NODE_URL", "http://localhost:8090")

# Throttle: don't push more than N texts per batch to the neuro endpoint
_TRAIN_BATCH_MAX = int(os.getenv("WIZARD_TRAIN_BATCH_MAX", "64"))
# TTL for probing the node; skip training if offline
_PROBE_CACHE_TTL = 60.0

# Standard pool ids (must match brain_api.rs constants).
_POOL_TEXT   = 1
_POOL_ACTION = 4


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

    # ------------------------------------------------------------------
    # OHLCV → text corpus
    # ------------------------------------------------------------------

    def _format_ohlcv_sample(
        self,
        symbol: str,
        ts: float,
        price: float,
        volume: float,
        trend: str,
        vol_regime: str,
    ) -> str:
        """Convert a single OHLCV point into a Hebbian-trainable sentence."""
        base = symbol.split("/")[0].split("_")[0].upper()
        quote = symbol.split("/")[-1].split("_")[0].upper() if "/" in symbol else "USD"
        price_fmt = f"{price:.6g}"
        vol_fmt = f"{volume:.4g}"
        return (
            f"{base} price {price_fmt} {quote}. "
            f"Volume {vol_fmt}. "
            f"Trend {trend}. "
            f"Volatility {vol_regime}."
        )

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
        samples: List[Tuple[float, float, float]],  # (ts, price, volume)
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

        prices = [float(s[1]) for s in samples if float(s[1]) > 0]
        trend = self._classify_trend(prices) if prices else "unknown"
        vol_regime = self._classify_vol(prices) if prices else "unknown"

        texts = []
        for ts, price, volume in samples:
            if price <= 0:
                continue
            texts.append(self._format_ohlcv_sample(symbol, ts, float(price), float(volume), trend, vol_regime))

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
        base = symbol.split("/")[0].upper()
        query = f"{base} market regime price {current_price:.6g}"

        if _brain_mode():
            return self._query_regime_brain(query, timeout)
        return self._query_regime_legacy(query, timeout)

    def _query_regime_brain(self, query: str, timeout: float) -> Optional[str]:
        """Brain substrate route: observe the query into POOL_TEXT,
        then POST /brain/integrate to get the trained regime
        text decoded from POOL_ACTION.  The answer comes back
        base64url-encoded in the canonical /brain/integrate
        response shape (see brain_api.rs::h_integrate)."""
        try:
            frame = base64.urlsafe_b64encode(
                query.encode("utf-8")
            ).decode("ascii").rstrip("=")
            observe_url = f"{self._endpoint}/brain/observe"
            observe_payload = json.dumps({
                "pool_id": _POOL_TEXT,
                "frame":   frame,
            }).encode("utf-8")
            req = urllib.request.Request(
                observe_url, data=observe_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp.read()

            integrate_url = f"{self._endpoint}/brain/integrate"
            integrate_payload = json.dumps({
                "query_pool":  _POOL_TEXT,
                "target_pool": _POOL_ACTION,
            }).encode("utf-8")
            req = urllib.request.Request(
                integrate_url, data=integrate_payload,
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
        """Route each text into the brain substrate as one observe
        cycle: observe → tick.  No advance_tick between observes in
        a single text, so each text is one moment fingerprint.
        Counts texts successfully pushed; on transport error marks
        the node offline so the next probe re-checks."""
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


# Module-level singleton so callers don't need to instantiate
_default_trainer: Optional[WizardTrainer] = None


def get_trainer() -> WizardTrainer:
    global _default_trainer
    if _default_trainer is None:
        _default_trainer = WizardTrainer()
    return _default_trainer
