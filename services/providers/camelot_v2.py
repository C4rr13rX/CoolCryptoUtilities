from __future__ import annotations
import os, time
from typing import Any, Dict, Optional, List, Tuple
from web3 import Web3

# ---- V2 routers per chain (override via ENV: CAMELOT_V2_ROUTER_<CHAIN>) ----
CAMELOT_V2_ROUTER: Dict[str, str] = {
    "arbitrum": "0xC873FECbd354f5A56E00E710B90EF4201db2448d",  # Camelot V2
    "ethereum": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",  # Uniswap V2 Router02
    "polygon":  "0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff",  # QuickSwap V2
    "optimism": "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506",  # Sushi V2
    # Base is optional; set CAMELOT_V2_ROUTER_BASE in .env if you have a V2 router you want to use
    "base":     os.getenv("CAMELOT_V2_ROUTER_BASE", "").strip() or "",
}

# ---- Wrapped-native per chain (override via ENV: WNATIVE_<CHAIN>) ----
WNATIVE: Dict[str, str] = {
    "arbitrum": "0x82aF49447D8a07e3bd95BDdB56f35241523fBab1",
    "ethereum": "0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2",
    "polygon":  "0x0d500B1d8E8Ef31E21C99d1Db9A6444d3ADf1270",
    "optimism": "0x4200000000000000000000000000000000000006",
    "base":     "0x4200000000000000000000000000000000000006",
}

# Camelot’s router has a referrer parameter; standard V2 routers do not.
HAS_REFERRER: Dict[str, bool] = {"arbitrum": True}

# --- ABIs ---
_ABI_V2_COMMON = [
  {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},
             {"internalType":"address[]","name":"path","type":"address[]"}],
   "name":"getAmountsOut","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],
   "stateMutability":"view","type":"function"}
]

_ABI_V2_SWAP_STD = [
  {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},
             {"internalType":"uint256","name":"amountOutMin","type":"uint256"},
             {"internalType":"address[]","name":"path","type":"address[]"},
             {"internalType":"address","name":"to","type":"address"},
             {"internalType":"uint256","name":"deadline","type":"uint256"}],
   "name":"swapExactTokensForTokensSupportingFeeOnTransferTokens",
   "outputs":[], "stateMutability":"nonpayable","type":"function"}
]

_ABI_V2_SWAP_CAMELOT = [
  {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},
             {"internalType":"uint256","name":"amountOutMin","type":"uint256"},
             {"internalType":"address[]","name":"path","type":"address[]"},
             {"internalType":"address","name":"to","type":"address"},
             {"internalType":"address","name":"referrer","type":"address"},
             {"internalType":"uint256","name":"deadline","type":"uint256"}],
   "name":"swapExactTokensForTokensSupportingFeeOnTransferTokens",
   "outputs":[], "stateMutability":"nonpayable","type":"function"}
]

class CamelotV2Local:
    def __init__(self, w3_provider_callable):
        self._w3 = w3_provider_callable

    def _cfg(self, chain: str):
        ch = chain.lower().strip()
        # env overrides
        router = os.getenv(f"CAMELOT_V2_ROUTER_{ch.upper()}",
                  CAMELOT_V2_ROUTER.get(ch, "")).strip()
        wnat   = os.getenv(f"WNATIVE_{ch.upper()}",
                  WNATIVE.get(ch, "0x0000000000000000000000000000000000000000")).strip()
        hasref = os.getenv(f"CAMELOT_V2_HAS_REFERRER_{ch.upper()}",
                  "true" if HAS_REFERRER.get(ch, False) else "false").strip().lower() in ("1","true","yes","y")
        return router, wnat, hasref

    def quote_and_build(self, chain: str, token_in: str, token_out: str, amount_in: int, *,
                        slippage_bps: int=100, recipient: Optional[str]=None) -> Dict[str,Any]:
        ch = chain.lower().strip()
        router_addr, wnative_addr, has_ref = self._cfg(ch)
        if not router_addr:
            return {"__error__": f"CamelotV2: no router configured for {chain}. "
                                 f"Set CAMELOT_V2_ROUTER_{ch.upper()} in your .env."}

        w3 = self._w3(chain)
        router = Web3.to_checksum_address(router_addr)
        t_in  = Web3.to_checksum_address(token_in)
        t_out = Web3.to_checksum_address(token_out)
        wnat  = Web3.to_checksum_address(wnative_addr)

        # Two handles: one for quote, one for the right swap signature
        router_quote = w3.eth.contract(address=router, abi=_ABI_V2_COMMON)
        router_swap  = w3.eth.contract(address=router, abi=_ABI_V2_SWAP_CAMELOT if has_ref else _ABI_V2_SWAP_STD)

        paths: List[Tuple[List[str], int]] = []
        # direct
        try:
            out = router_quote.functions.getAmountsOut(int(amount_in), [t_in, t_out]).call()[-1]
            if int(out) > 0: paths.append(([t_in, t_out], int(out)))
        except Exception: pass
        # via WNATIVE
        try:
            out1 = router_quote.functions.getAmountsOut(int(amount_in), [t_in, wnat]).call()[-1]
            out2 = router_quote.functions.getAmountsOut(int(out1), [wnat, t_out]).call()[-1]
            if int(out1) > 0 and int(out2) > 0: paths.append(([t_in, wnat, t_out], int(out2)))
        except Exception: pass

        if not paths:
            return {"__error__": "CamelotV2: no path (direct or via WNATIVE)"}

        path, best_out = max(paths, key=lambda x: x[1])
        out_min = int(best_out * (10_000 - int(slippage_bps)) // 10_000)
        deadline = int(time.time()) + 600
        to_addr = Web3.to_checksum_address(recipient or (w3.eth.default_account or "0x0000000000000000000000000000000000000000"))

        if has_ref:
            referrer = Web3.to_checksum_address(os.getenv("CAMELOT_REFERRER", "0x0000000000000000000000000000000000000000"))
            fn = router_swap.functions.swapExactTokensForTokensSupportingFeeOnTransferTokens(
                int(amount_in), out_min, path, to_addr, referrer, deadline
            )
        else:
            fn = router_swap.functions.swapExactTokensForTokensSupportingFeeOnTransferTokens(
                int(amount_in), out_min, path, to_addr, deadline
            )

        data = fn._encode_transaction_data()
        try:
            gas_est = int(w3.eth.estimate_gas({"from": to_addr, "to": router, "data": data, "value": 0}) * 1.2)
        except Exception:
            gas_est = 250_000

        return {
            "aggregator": "CamelotV2",
            "allowanceTarget": router,
            "tx": {"to": router, "data": data, "value": 0, "gas": gas_est},
            "buyAmount": str(best_out),
            "sellAmount": str(int(amount_in)),
            "sellToken": t_in, "buyToken": t_out,
            "route": path,
        }
