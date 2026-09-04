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

def from_base_units(raw: int, decimals: int) -> str:
    """Exact human string for `raw` base units -- the inverse of to_base_units.

    THE POINT IS THAT NOTHING IS LOST. Callers used to hand swap() a
    ``f"{amount:.6f}"`` string, which silently floors the sell amount at six
    decimal places. Measured 2026-09-03 on base, exit
    0x9ffdd1cfe3f17fbeddb2b3091b49f8f6524b5a0538ae9a2f3aadb56610264e6c held
    0.000111373 cbETH and sold 0.000111000 -- the format string, not the
    router, kept 0.000000373 cbETH back. That residue is still in the wallet
    (balances row CBETH quantity 3.73253011411E-7), it is too small to be
    worth a swap of its own, and the retries it caused are what ended the
    only burst of rapid trading this system has produced. cbETH has 18
    decimals; six of them is not a rounding choice, it is a leak.

    Fixed-point, never exponent notation: ``str(Decimal('3.7e-7'))`` is
    "3.7E-7" and ``to_base_units`` would have to parse that back. ``format(q,
    "f")`` keeps it as 0.00000037, which round-trips exactly at the module's
    80-digit precision for every ERC-20 decimals value (0-255 fits easily).
    """
    return format(Decimal(int(raw)).scaleb(-int(decimals)), "f")

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
