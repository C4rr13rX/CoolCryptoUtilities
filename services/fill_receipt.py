"""Read what a swap actually filled from its transaction receipt.

A fill used to be measured as a wallet-balance delta: read the portfolio
before the swap, read it after, subtract. That is wrong twice over, and both
failures were observed on 2026-09-03 when R3V3N!R placed its first two real
on-chain trades and recorded BOTH of them as ``no_fill_detected``:

  0xfd133cfe29d0018e778275bb8a9c2ba89b8d63cf4fabe87d773dfe8c9688b880
      0.750000 USDC out, 19.488243 BASECAT in, block 50821325, status 0x1
  0xd4c2d4df7886a80f4113314d518772e80113a68dd64c3321f2e873aec7c9c196
      0.750000 USDC out, 1.546281 AERO in, status 0x1

1. ``base_received`` read 0.0 because the portfolio only knows tokens the
   transfer indexer has already discovered. BASECAT had no row in ``balances``
   at all. Buying a token for the FIRST TIME therefore always reads zero
   received -- which is every opening trade this bot will ever make.

2. ``quote_spent`` read 1.5 for a 0.75 swap because the other bot's swap
   settled inside the same measurement window. GhostSupervisor runs one bot
   per symbol against ONE wallet, so no bot can measure its own fill by
   wallet delta, ever.

The receipt has neither problem. Its ERC-20 ``Transfer`` logs name the exact
amounts that moved to and from our address in THAT transaction: authoritative,
per-trade, immune to both the discovery lag and the concurrency.

This module is deliberately pure -- it parses a receipt dict and nothing else,
so it can be tested against the two real receipts above without a network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

# keccak256("Transfer(address,address,uint256)")
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

WEI_PER_ETHER = 10**18


def _hex_str(value: Any) -> str:
    """Normalise a receipt field to a lowercase 0x-string.

    Receipts arrive either as raw JSON-RPC (every field a hex string) or as a
    web3 AttributeDict (HexBytes for topics/data, ints for gasUsed). Both are
    accepted; anything else returns "" rather than raising, because a parse
    failure must degrade to "unknown fill", never to an exception on the money
    path.
    """
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return "0x" + bytes(value).hex()
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return hex(value)
    text = str(value).strip().lower()
    if not text:
        return ""
    return text if text.startswith("0x") else "0x" + text


def _hex_int(value: Any) -> Optional[int]:
    """Parse a receipt integer field. None when it cannot be read."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = _hex_str(value)
    if not text or text == "0x":
        return None
    try:
        return int(text, 16)
    except (TypeError, ValueError):
        return None


def _addr(value: Any) -> str:
    """Lowercase 20-byte address from an address field or a padded topic."""
    text = _hex_str(value)
    if not text.startswith("0x"):
        return ""
    body = text[2:]
    if len(body) < 40:
        return ""
    # A topic is the address left-padded to 32 bytes; the address is the last
    # 20. Comparing the padded form against a bare address is how a match is
    # silently missed, so always reduce both sides to these 40 characters.
    return "0x" + body[-40:]


def receipt_status(receipt: Mapping[str, Any]) -> Optional[bool]:
    """True/False from the receipt status; None when the field is absent.

    None is "unknown", never "reverted" -- a receipt we could not fully read
    must not be reported as a failed trade.
    """
    if "status" not in receipt:
        return None
    parsed = _hex_int(receipt.get("status"))
    if parsed is None:
        return None
    return parsed == 1


def gas_native_from_receipt(receipt: Mapping[str, Any]) -> float:
    """Gas actually paid, in native units (ether), from gasUsed x price.

    Returns 0.0 when either field is missing. That is a floor, not a guess:
    understating gas overstates profit, so callers that price a trade off this
    should treat 0.0 as "gas unknown" and fall back to their own estimate.
    """
    used = _hex_int(receipt.get("gasUsed"))
    price = _hex_int(receipt.get("effectiveGasPrice"))
    if price is None:
        price = _hex_int(receipt.get("gasPrice"))
    if not used or not price:
        return 0.0
    return (used * price) / WEI_PER_ETHER


def transfer_deltas(logs: Iterable[Any], wallet: str) -> dict[str, int]:
    """Net raw ERC-20 movement per token for `wallet`, in base units.

    Positive means the wallet received. Every Transfer touching the wallet is
    summed, so a router that refunds dust of the sell token, or a
    fee-on-transfer token that pays out in two legs, nets correctly instead of
    being read from whichever log happened to come first.

    Logs whose topics are not exactly ``[Transfer, from, to]`` are skipped:
    the pool's own ``Swap`` event sits in the same receipt and carries the same
    amounts in a different layout, so matching loosely double-counts.
    """
    me = _addr(wallet)
    deltas: dict[str, int] = {}
    if not me:
        return deltas
    for entry in logs or ():
        if not isinstance(entry, Mapping):
            continue
        topics = entry.get("topics") or []
        if len(topics) != 3:
            continue
        if _hex_str(topics[0]) != TRANSFER_TOPIC:
            continue
        sender = _addr(topics[1])
        recipient = _addr(topics[2])
        if me not in (sender, recipient):
            continue
        amount = _hex_int(entry.get("data"))
        if amount is None:
            continue
        token = _addr(entry.get("address"))
        if not token:
            continue
        # A self-transfer nets to zero, which is the correct answer.
        if recipient == me:
            deltas[token] = deltas.get(token, 0) + amount
        if sender == me:
            deltas[token] = deltas.get(token, 0) - amount
    return deltas


def _human(raw: int, decimals: int) -> float:
    if decimals < 0:
        decimals = 0
    return raw / float(10**decimals)


@dataclass(frozen=True)
class ReceiptFill:
    """What one swap transaction moved, read from its own receipt.

    ``ok`` means both legs were found with positive amounts -- the only state
    in which a caller may open or close a position on these numbers. When it is
    False, ``reason`` says which leg is missing and the caller must fall back
    rather than record a zero fill.
    """

    ok: bool = False
    reason: str = "not_read"
    status: Optional[bool] = None
    sold_raw: int = 0
    bought_raw: int = 0
    sold: float = 0.0
    bought: float = 0.0
    gas_native: float = 0.0
    deltas: dict[str, int] = field(default_factory=dict)

    @property
    def price(self) -> float:
        """Executed price in sold-per-bought units (quote per base on entry)."""
        if self.bought <= 0.0:
            return 0.0
        return self.sold / self.bought


def read_fill(
    receipt: Optional[Mapping[str, Any]],
    *,
    wallet: str,
    sell_token: str,
    buy_token: str,
    sell_decimals: int,
    buy_decimals: int,
) -> ReceiptFill:
    """Extract the sold/bought amounts of one swap from its receipt.

    ``sell_token``/``buy_token`` are the ERC-20 contract addresses of the two
    legs. Native legs have no Transfer log and are reported unresolved
    (``reason="native_leg"``) so the caller can fall back; the live trading
    path is ERC-20 on both sides (USDC in, token out and back), so this is not
    the path that matters here.
    """
    if not receipt:
        return ReceiptFill(ok=False, reason="no_receipt")

    status = receipt_status(receipt)
    gas_native = gas_native_from_receipt(receipt)
    if status is False:
        return ReceiptFill(ok=False, reason="reverted", status=status, gas_native=gas_native)

    sell_addr = _addr(sell_token)
    buy_addr = _addr(buy_token)
    if not sell_addr or not buy_addr:
        return ReceiptFill(
            ok=False, reason="native_leg", status=status, gas_native=gas_native
        )

    deltas = transfer_deltas(receipt.get("logs") or [], wallet)
    sold_raw = -deltas.get(sell_addr, 0)
    bought_raw = deltas.get(buy_addr, 0)

    # "Nothing touched us" and "both legs point the wrong way" are different
    # failures: the first means we read the wrong wallet or the wrong receipt,
    # the second means the caller named the legs backwards. Collapsing them
    # sends whoever debugs this looking in the wrong place.
    if not deltas:
        reason = "no_transfer_to_wallet"
    elif sold_raw <= 0 and bought_raw <= 0:
        reason = "both_legs_missing"
    elif sold_raw <= 0:
        reason = "sell_leg_missing"
    elif bought_raw <= 0:
        reason = "buy_leg_missing"
    else:
        reason = ""

    return ReceiptFill(
        ok=not reason,
        reason=reason or "",
        status=status,
        sold_raw=max(0, sold_raw),
        bought_raw=max(0, bought_raw),
        sold=_human(max(0, sold_raw), sell_decimals),
        bought=_human(max(0, bought_raw), buy_decimals),
        gas_native=gas_native,
        deltas=deltas,
    )
