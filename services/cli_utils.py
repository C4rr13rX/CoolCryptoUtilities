from __future__ import annotations
from decimal import Decimal, getcontext
from web3 import Web3
getcontext().prec = 80

ZEROX_NATIVE = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"

def is_native(tok: str) -> bool:
    t = (tok or "").strip().lower()
    return t in ("eth", "native", ZEROX_NATIVE.lower())

def normalize_for_0x(token: str) -> str:
    return ZEROX_NATIVE if is_native(token) else Web3.to_checksum_address(token)

def to_base_units(amount: str, decimals: int) -> int:
    q = Decimal(str(amount).strip())
    factor = Decimal(10) ** int(decimals)
    return int((q * factor).to_integral_value(rounding="ROUND_FLOOR"))

def wei_to_eth(n: int) -> str:
    try:
        return str(Web3.from_wei(int(n), "ether"))
    except Exception:
        x = int(n); s = f"{x/10**18:.18f}".rstrip("0").rstrip(".")
        return s or "0"

EXPLORER_TX = {
    "arbitrum": "https://arbiscan.io/tx/",
    "ethereum": "https://etherscan.io/tx/",
    "base": "https://basescan.org/tx/",
    "optimism": "https://optimistic.etherscan.io/tx/",
    "polygon": "https://polygonscan.com/tx/",
}
def explorer_for(chain: str) -> str | None:
    return EXPLORER_TX.get((chain or "").lower())
