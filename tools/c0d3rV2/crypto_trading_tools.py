"""Specialised C0d3rV2 tools for crypto trading.

C0d3rV2 is the AGENT; the crypto wizard brain is the MODEL. These are the tools
that let the agent do trading work rather than chat or coding work: rank what
to trade, ask the brain which way it goes, and check whether either has earned
the right to touch real money.

They replace ``crypto_paper_trade.score_pair``, which was measured on 131,200
real bars 2026-08-27 and found to make selection WORSE than random:

    top decile     mean -0.00086/trade   win 46.3%
    all bars       mean -0.00023/trade   win 48.3%
    lift           -0.00063/trade

It weights momentum and buy pressure, which mean-revert at this horizon, so it
reliably buys the top of a move. Every tool here reports the same
lift-over-baseline number that exposed it, so a replacement cannot quietly be
worse either.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.c0d3rV2.tool_registry import Tool

_ROOT = Path(__file__).resolve().parents[2]


def _load_bars(symbols: Optional[List[str]] = None, max_symbols: int = 5,
               max_bars: int = 3000) -> Dict[str, List[Dict[str, Any]]]:
    root = Path(os.getenv("HISTORICAL_OHLCV_DIR", str(_ROOT / "data" / "historical_ohlcv")))
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not root.exists():
        return out
    for path in sorted(root.glob("*/*.json")):
        name = path.stem.split("_", 1)[-1]
        if symbols and name not in symbols:
            continue
        if name in out:
            continue
        try:
            bars = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(bars, list) and len(bars) > 200:
            out[name] = bars[-max_bars:]
        if len(out) >= max_symbols:
            break
    return out


class CryptoSelectorScoreTool(Tool):
    """Rank cryptos by a GA-searchable selector, not by hand-tuned weights."""

    name = "crypto_selector_score"
    description = (
        "Score crypto pairs for how attractive they are to BUY right now, using "
        "mean-reversion, drawdown, volume-surge, volatility and pressure "
        "features. Returns a ranked list with the feature vector behind each "
        "score. Replaces score_pair, whose picks measured WORSE than random."
    )
    use_when = (
        "Use when choosing WHICH crypto to trade, before asking the brain which "
        "direction it will go. Not a direction signal on its own."
    )
    params_schema = {
        "symbols": "list[str] — optional pairs to score; defaults to the stored corpus",
        "weights": "dict — optional selector weights (w_reversion, w_momentum, ...)",
        "top": "int — how many to return, default 10",
    }

    def execute(self, params: dict) -> dict:
        from trading.genome.selectors import (
            FEATURES, SELECTOR_GENE_SPACE, feature_vector, selector_score,
        )
        symbols = params.get("symbols") or None
        weights = params.get("weights") or {}
        if not weights:
            # Default: buy weakness. This is the direction the measurement
            # pointed, and the opposite of what score_pair did.
            weights = {"w_%s" % k: 0.0 for k in FEATURES}
            weights.update({"w_reversion": 1.0, "w_drawdown": 0.6,
                            "w_range_position": 0.5, "w_momentum": -0.3})
        top = int(params.get("top") or 10)
        bars = _load_bars(symbols, max_symbols=max(top, 8))
        if not bars:
            return {"error": "no historical bars available", "ranked": []}
        ranked = []
        for symbol, series in bars.items():
            window = series[-25:]
            ranked.append({
                "symbol": symbol,
                "score": round(selector_score(window, weights), 6),
                "features": {k: round(v, 6) for k, v in feature_vector(window).items()},
            })
        ranked.sort(key=lambda row: -row["score"])
        return {
            "ranked": ranked[:top],
            "weights": weights,
            "gene_space": {k: list(v) for k, v in SELECTOR_GENE_SPACE.items()},
            "note": "score ranks WHAT to trade; direction comes from the brain",
        }


class CryptoSelectorBacktestTool(Tool):
    """Prove a selector beats random picking before it is used."""

    name = "crypto_selector_backtest"
    description = (
        "Score a selector weight vector on withheld historical data and report "
        "its lift over picking at random, plus win rates and post-fee means. "
        "This is the measurement that showed score_pair was worse than random."
    )
    use_when = (
        "Use before trusting ANY selection rule, including a GA champion, and "
        "whenever selection quality is in question."
    )
    params_schema = {
        "weights": "dict — selector weights to test",
        "symbols": "list[str] — optional pairs; defaults to the stored corpus",
        "top_fraction": "number — fraction treated as picks, default 0.10",
        "horizon": "int — bars ahead to measure, default 6",
    }

    def execute(self, params: dict) -> dict:
        from trading.genome.selectors import evaluate_selector
        weights = params.get("weights") or {}
        if not weights:
            return {"error": "weights are required"}
        bars = _load_bars(params.get("symbols") or None, max_symbols=5, max_bars=2000)
        if not bars:
            return {"error": "no historical bars available"}
        result = evaluate_selector(
            weights, bars,
            horizon=int(params.get("horizon") or 6),
            top_fraction=float(params.get("top_fraction") or 0.10),
        )
        payload = result.to_dict()
        payload["symbols"] = list(bars.keys())
        payload["verdict"] = (
            "beats random selection" if result.passed else "NO better than random"
        )
        return payload


class CryptoBrainDirectionTool(Tool):
    """Ask the wizard brain which way a pair goes, with a refusal path."""

    name = "crypto_brain_direction"
    description = (
        "Ask the crypto wizard brain for a direction call on a pair, via the "
        "trading agent. Returns the decision, its confidence, and the reason -- "
        "including an explicit refusal when the price cannot be corroborated or "
        "confidence is below the floor."
    )
    use_when = (
        "Use after selecting a candidate, to decide whether and how to trade "
        "it. A refusal is a valid answer and must not be retried around."
    )
    params_schema = {
        "symbol": "str — pair, e.g. AERO-USDC",
        "prices": "list[number] — recent price history, oldest first",
        "quoted_price": "number — the price being considered",
        "size_usd": "number — intended position size",
    }

    def execute(self, params: dict) -> dict:
        symbol = str(params.get("symbol") or "").upper()
        prices = params.get("prices") or []
        if not symbol or not prices:
            return {"error": "symbol and prices are required"}
        try:
            from trading.brain.trading_agent import WizardTradingAgent
            from trading.brain_bridge import get_bridge, features_text, parse_outcome
        except Exception as exc:  # noqa: BLE001
            return {"error": "brain unavailable: %s" % exc}

        bridge = get_bridge()

        class _BrainAdapter:
            def predict(self, sym, price, momentum):
                text = features_text(side="enter", symbol=sym, chain="base",
                                     price=price, momentum=momentum, confidence=0.5)
                answer, confidence = bridge.predict_outcome(text)
                return parse_outcome(answer), confidence

        agent = WizardTradingAgent(_BrainAdapter())
        decision = agent.decide(
            symbol=symbol,
            prices=[float(p) for p in prices],
            quoted_price=float(params.get("quoted_price") or prices[-1]),
            size_usd=float(params.get("size_usd") or 0.0),
        )
        return decision.to_dict()


class CryptoBrainValidationTool(Tool):
    """Gate: has the brain earned the right to trade real money?"""

    name = "crypto_brain_validation"
    description = (
        "Validate the crypto wizard brain on historical data before production: "
        "a control on fabricated symbols, a chronological split, and the "
        "majority-class baseline. Reports pass/fail with the numbers."
    )
    use_when = (
        "Use before enabling the brain for live trading, and after any training "
        "run, to check whether it has actually learned anything."
    )
    params_schema = {
        "symbols": "list[str] — optional pairs; defaults to the stored corpus",
        "min_edge": "number — required edge over baseline, default 0.02",
    }

    def execute(self, params: dict) -> dict:
        try:
            from trading.brain.trading_agent import validate_on_history
            from trading.brain_bridge import get_bridge, features_text, parse_outcome
        except Exception as exc:  # noqa: BLE001
            return {"error": "brain unavailable: %s" % exc}
        bars = _load_bars(params.get("symbols") or None, max_symbols=3, max_bars=1500)
        if not bars:
            return {"error": "no historical bars available"}
        bridge = get_bridge()

        class _BrainAdapter:
            def predict(self, sym, price, momentum):
                text = features_text(side="enter", symbol=sym, chain="base",
                                     price=price, momentum=momentum, confidence=0.5)
                answer, confidence = bridge.predict_outcome(text)
                return parse_outcome(answer), confidence

        result = validate_on_history(
            _BrainAdapter(), bars,
            min_edge=float(params.get("min_edge") or 0.02),
        )
        payload = result.to_dict()
        payload["symbols"] = list(bars.keys())
        payload["may_trade_live"] = bool(result.passed)
        return payload


TRADING_TOOLS = [
    CryptoSelectorScoreTool,
    CryptoSelectorBacktestTool,
    CryptoBrainDirectionTool,
    CryptoBrainValidationTool,
]
