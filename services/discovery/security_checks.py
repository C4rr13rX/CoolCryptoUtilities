from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import requests


@dataclass
class SecurityReport:
    verdict: str
    confidence: float
    details: Dict[str, object]


def goplus_token_security(address: str, chain_id: str) -> Optional[SecurityReport]:
    api_key = os.getenv("GOPLUS_APP_KEY")
    if not api_key:
        return None
    params = {
        "chain_id": chain_id,
        "contract_addresses": address,
        "api_key": api_key,
    }
    resp = requests.get("https://api.gopluslabs.io/api/v1/token_security/10", params=params, timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("code") != 1:
        return None
    result = payload.get("result", {}).get(address.lower())
    if not result:
        return None
    is_honeypot = result.get("is_honeypot", "0") == "1"
    tax_buy = float(result.get("buy_tax", 0)) if result.get("buy_tax") else 0.0
    tax_sell = float(result.get("sell_tax", 0)) if result.get("sell_tax") else 0.0
    details = {
        "buy_tax": tax_buy,
        "sell_tax": tax_sell,
        "owner_percent": result.get("owner_percent"),
        "is_honeypot": result.get("is_honeypot"),
    }
    verdict = "honeypot" if is_honeypot else "safe"
    confidence = 0.85 if verdict == "honeypot" else 0.65
    return SecurityReport(verdict=verdict, confidence=confidence, details=details)


def heuristic_screen(
    *,
    tax_buy: float,
    tax_sell: float,
    liquidity_usd: Optional[float],
    price_change_24h: Optional[float],
) -> SecurityReport:
    flags = []
    if tax_buy > 0.25 or tax_sell > 0.25:
        flags.append("high_tax")
    if liquidity_usd is not None and liquidity_usd < 15000:
        flags.append("low_liquidity")
    if price_change_24h is not None and price_change_24h < -75:
        flags.append("sharp_drop")
    verdict = "honeypot" if flags else "safe"
    confidence = 0.5 if verdict == "safe" else 0.6
    return SecurityReport(verdict=verdict, confidence=confidence, details={"flags": flags})
