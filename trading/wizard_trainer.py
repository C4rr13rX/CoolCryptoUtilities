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
        # Non-blocking regime cache.  Strategies in the scheduler call
        # cached_regime() instead of query_regime() so the brain HTTP
        # round-trip never blocks trade evaluation.  Background daemons
        # refresh entries on a per-symbol cooldown.
        self._regime_lock = threading.Lock()
        self._regime_cache: Dict[str, "BrainSignal"] = {}
        self._regime_inflight: Dict[str, float] = {}  # symbol -> launch ts
        try:
            self._regime_ttl = float(os.getenv("WIZARD_REGIME_TTL_SEC", "20"))
        except Exception:
            self._regime_ttl = 20.0
        try:
            self._regime_timeout = float(os.getenv("WIZARD_REGIME_TIMEOUT_SEC", "1.5"))
        except Exception:
            self._regime_timeout = 1.5

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


def _brain_feeder_loop(
    chains: Sequence[str],
    interval_sec: float,
    tail_candles: int,
    data_root: Optional[str],
) -> None:
    import json as _j
    from pathlib import Path as _P
    root = _P(data_root) if data_root else _P("data") / "historical_ohlcv"
    while not _BRAIN_FEEDER_STATE.get("stop"):
        try:
            trainer = get_trainer()
            if not trainer.is_online():
                time.sleep(min(60.0, interval_sec))
                continue
            total_pushed = 0
            for chain in chains:
                cdir = root / chain
                if not cdir.exists():
                    continue
                for jf in cdir.glob("*.json"):
                    try:
                        sym = jf.stem.split("_", 1)[-1]
                        with jf.open("r", encoding="utf-8") as fh:
                            rows = _j.load(fh)
                        if not isinstance(rows, list) or not rows:
                            continue
                        tail = rows[-tail_candles:]
                        samples = []
                        for r in tail:
                            try:
                                ts = float(r.get("timestamp", 0))
                                p = float(r.get("close", 0) or r.get("price", 0))
                                v = float(r.get("net_volume", 0) or r.get("volume", 0))
                                if p > 0:
                                    samples.append((ts, p, v))
                            except Exception:
                                continue
                        if samples:
                            total_pushed += trainer.push_ohlcv_batch(sym, samples, max_items=32)
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


_BULL_TOKENS = {"bull", "bullish", "uptrend", "rally", "long", "buy", "breakout",
                "rising", "strong", "accumulat", "support"}
_BEAR_TOKENS = {"bear", "bearish", "downtrend", "sell", "short", "breakdown",
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
        # Auto-start the OHLCV→brain feeder so the brain keeps learning
        # even when the TF training cycle is broken (DLL OOM / model
        # build failure).  Idempotent.  Disable with
        # WIZARD_BRAIN_FEEDER_ENABLED=0 if it ever conflicts.
        try:
            start_brain_feeder(chains=("base", "arbitrum", "optimism", "polygon"))
        except Exception:
            pass
    return _default_trainer
