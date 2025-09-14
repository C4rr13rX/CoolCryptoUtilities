# services/bridge_service.py
from __future__ import annotations
from decimal import Decimal, InvalidOperation
from typing import Optional, Dict, Any

from router_wallet import UltraSwapBridge, NATIVE

def _to_decimal_str(x: str) -> str:
    s = (x or "").strip()
    if s.startswith("."):
        s = "0" + s
    return s

class BridgeService:
    """
    Thin wrapper over UltraSwapBridge's LI.FI-powered quote/execute.

    - Accepts amount_human (e.g. "0.25") or amount_raw (base units).
    - If dst_token/to_token omitted, uses the same token id on the destination.
    - slippage_bps: 100 => 1.00%
    """
    def __init__(self, core: UltraSwapBridge):
        # IMPORTANT: do NOT name this 'bridge' or you'll shadow the method below.
        self.core = core

    def _amount_to_float(
        self, *, chain: str, token: str, amount_human: Optional[str], amount_raw: Optional[int]
    ) -> float:
        if amount_human is not None:
            try:
                return float(Decimal(_to_decimal_str(amount_human)))
            except (InvalidOperation, ValueError):
                raise ValueError(f"Invalid amount_human: {amount_human!r}")
        if amount_raw is not None:
            dec = 18 if (str(token).lower() in ("eth", "native", NATIVE.lower())) \
                else int(self.core.erc20_decimals(chain, token))
            return float(Decimal(int(amount_raw)) / (Decimal(10) ** Decimal(dec)))
        raise ValueError("Provide amount_human or amount_raw")

    def quote(
        self,
        *,
        src_chain: str,
        dst_chain: str,
        token: str,
        amount_human: Optional[str] = None,
        amount_raw: Optional[int] = None,
        dst_token: Optional[str] = None,
        slippage_bps: int = 100,
    ) -> Dict[str, Any]:
        amt = self._amount_to_float(chain=src_chain, token=token, amount_human=amount_human, amount_raw=amount_raw)
        if amt <= 0:
            raise ValueError("Amount must be > 0")
        dst_tok = dst_token or token
        slip = float(slippage_bps) / 10_000.0
        pv = self.core.quote(src_chain, dst_chain, token, dst_tok, amt, slippage=slip)
        print(
            f"[bridge/quote] {src_chain}:{token} -> {dst_chain}:{dst_tok} "
            f"amt={amt} (minOut≈{pv.to_amount_min:.6g}) gas≈${(pv.gas_usd or 0):.2f}"
        )
        return {
            "from_chain": pv.from_chain,
            "to_chain": pv.to_chain,
            "from_symbol": pv.from_symbol,
            "to_symbol": pv.to_symbol,
            "from_amount": pv.from_amount,
            "to_amount": pv.to_amount,
            "to_amount_min": pv.to_amount_min,
            "gas_usd": pv.gas_usd,
            "tx_request": pv.tx_request,
        }

    def bridge(
        self,
        *,
        src_chain: str,
        dst_chain: str,
        token: str,
        amount_human: Optional[str] = None,
        amount_raw: Optional[int] = None,
        dst_token: Optional[str] = None,
        to_token: Optional[str] = None,   # backward-compat alias
        slippage_bps: int = 100,
        wait: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a LI.FI bridge (and swap if the route includes it).
        Returns UltraSwapBridge.execute(...) result.
        """
        amt = self._amount_to_float(chain=src_chain, token=token, amount_human=amount_human, amount_raw=amount_raw)
        if amt <= 0:
            raise ValueError("Amount must be > 0")

        # Honor either name; prefer dst_token if both set
        dst_tok = dst_token if dst_token is not None else (to_token if to_token is not None else token)
        slip = float(slippage_bps) / 10_000.0

        # Optional preview (nice log; non-blocking)
        try:
            _ = self.quote(
                src_chain=src_chain, dst_chain=dst_chain, token=token,
                amount_human=str(amt), dst_token=dst_tok, slippage_bps=slippage_bps
            )
        except Exception as e:
            print(f"[bridge] quote error: {e!r}")
            raise

        # Execute (auto-approve handled inside UltraSwapBridge.execute)
        try:
            res = self.core.execute(
                src_chain, dst_chain, token, dst_tok, amt, slippage=slip, wait=wait
            )
            print(f"[bridge] tx={res.get('txHash')}")
            if wait and res.get("bridgeStatus"):
                print(f"[bridge] final status={res['bridgeStatus'].get('status')}")
            return res
        except Exception as e:
            print(f"[bridge] execute error: {e!r}")
            raise
