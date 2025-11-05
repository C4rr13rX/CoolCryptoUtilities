from __future__ import annotations

import os


PRIMARY_CHAIN = os.getenv("PRIMARY_CHAIN", "base").strip().lower() or "base"
PRIMARY_SYMBOL = os.getenv("PRIMARY_SYMBOL", "ETH-USDC").strip().upper() or "ETH-USDC"
PRIMARY_BASE, PRIMARY_QUOTE = (
    PRIMARY_SYMBOL.split("-", 1) if "-" in PRIMARY_SYMBOL else (PRIMARY_SYMBOL, "USDC")
)

MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE_REQUIRED", "0.9"))
SMALL_PROFIT_FLOOR = float(os.getenv("SMALL_PROFIT_FLOOR", "0.25"))  # USD-equivalent target
MAX_QUOTE_SHARE = float(os.getenv("MAX_QUOTE_SHARE", "0.25"))

GAS_PROFIT_BUFFER = float(os.getenv("GAS_PROFIT_BUFFER", "1.25"))
FALLBACK_NATIVE_PRICE = float(os.getenv("FALLBACK_NATIVE_PRICE", "1800.0"))
