from __future__ import annotations
import json, os, urllib.parse, urllib.request, time
from typing import Any, Dict, Optional, List
from web3 import Web3

# --- 0x v2 Allowance-Holder ---
# Docs: unified host, headers 0x-api-key + 0x-version, chainId/taker/sellAmount params, tx in 'transaction'
# https://0x.org/docs/upgrading/upgrading_to_swap_v2  https://0x.org/docs/0x-swap-api/guides/swap-tokens-with-0x-swap-api
class ZeroXV2AllowanceHolder:
    BASE = "https://api.0x.org"
    def __init__(self, api_key: Optional[str]=None, *, timeout_s: float=25.0):
        self.api_key = api_key or os.getenv("ZEROX_API_KEY")
        self.timeout_s = float(timeout_s)
    def _headers(self) -> Dict[str,str]:
        if not self.api_key:
            raise RuntimeError("0x v2: missing ZEROX_API_KEY")
        return {"0x-api-key": self.api_key, "0x-version": "v2"}
    def quote(self, *, chain_id: int, sell_token: str, buy_token: str, sell_amount: int, taker: str, slippage_bps: int=100) -> Dict[str,Any]:
        if int(sell_amount) <= 0: raise ValueError("0x v2: sellAmount must be > 0")
        q = {
            "chainId": str(int(chain_id)),
            "sellToken": sell_token,
            "buyToken":  buy_token,
            "sellAmount": str(int(sell_amount)),  # must be STRING int
            "taker": Web3.to_checksum_address(taker),
            "slippageBps": str(int(slippage_bps)),
            "intentOnFilling": "true",
        }
        url = f"{self.BASE}/swap/allowance-holder/quote?{urllib.parse.urlencode(q)}"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"0x v2 {e.code}: {body}") from None
        tx = data.get("transaction") or {}
        issues = data.get("issues") or {}
        allow  = issues.get("allowance") or {}
        # spender to approve (AllowanceHolder/Permit2) — per docs
        data["allowanceTarget"] = allow.get("spender") or data.get("allowanceTarget")
        data["tx"] = tx
        return data

# --- Camelot V2 (Arbitrum) ---
# Function includes 'referrer' parameter (ABI differs from UniswapV2) — must encode correctly.
# Verified in explorers/audits.
CAMELOT_V2_ROUTER = {"arbitrum": "0xC873FECbd354f5A56E00E710B90EF4201db2448d"}
WETH_ARBITRUM = "0x82aF49447D8a07e3bd95BDdB56f35241523fBab1"
_ABI_V2_ROUTER = [
  {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},
             {"internalType":"uint256","name":"amountOutMin","type":"uint256"},
             {"internalType":"address[]","name":"path","type":"address[]"},
             {"internalType":"address","name":"to","type":"address"},
             {"internalType":"address","name":"referrer","type":"address"},
             {"internalType":"uint256","name":"deadline","type":"uint256"}],
   "name":"swapExactTokensForTokensSupportingFeeOnTransferTokens",
   "outputs":[], "stateMutability":"nonpayable","type":"function"},
  {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},
             {"internalType":"address[]","name":"path","type":"address[]"}],
   "name":"getAmountsOut",
   "outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],
   "stateMutability":"view","type":"function"}
]

class CamelotV2Local:
    def __init__(self, w3_provider_callable):
        # accept a callable to get Web3 per chain (from UltraSwapBridge)
        self._w3 = w3_provider_callable
    def quote_and_build(self, chain: str, token_in: str, token_out: str, amount_in: int, *, slippage_bps: int=100) -> Dict[str,Any]:
        ch = chain.lower()
        if ch not in CAMELOT_V2_ROUTER:
            return {"__error__": f"CamelotV2 unsupported on {chain}"}
        w3 = self._w3(chain)
        router_addr = Web3.to_checksum_address(CAMELOT_V2_ROUTER[ch])
        t_in  = Web3.to_checksum_address(token_in)
        t_out = Web3.to_checksum_address(token_out)
        weth  = Web3.to_checksum_address(WETH_ARBITRUM)
        router = w3.eth.contract(router_addr, abi=_ABI_V2_ROUTER)

        paths = []
        try:
            out = router.functions.getAmountsOut(int(amount_in), [t_in, t_out]).call()[-1]
            if int(out) > 0: paths.append(([t_in, t_out], int(out)))
        except Exception: pass
        try:
            out1 = router.functions.getAmountsOut(int(amount_in), [t_in, weth]).call()[-1]
            out2 = router.functions.getAmountsOut(int(out1), [weth, t_out]).call()[-1]
            if int(out1) > 0 and int(out2) > 0: paths.append(([t_in, weth, t_out], int(out2)))
        except Exception: pass
        if not paths:
            return {"__error__": "CamelotV2: no path (direct or via WETH)"}

        path, best_out = max(paths, key=lambda x: x[1])
        out_min = int(best_out * (10_000 - slippage_bps) / 10_000)
        deadline = int(time.time()) + 600
        fn = router.functions.swapExactTokensForTokensSupportingFeeOnTransferTokens(
            int(amount_in), out_min, path, w3.to_checksum_address(w3.eth.default_account or "0x0000000000000000000000000000000000000000"),
            "0x0000000000000000000000000000000000000000", deadline
        )
        data = fn._encode_transaction_data()
        # try estimate gas
        try:
            gas_est = int(w3.eth.estimate_gas({"from": w3.eth.default_account, "to": router_addr, "data": data, "value": 0}) * 1.2)
        except Exception:
            gas_est = 250000
        return {"aggregator":"CamelotV2","allowanceTarget":router_addr,"tx":{"to":router_addr,"data":data,"value":0,"gas":gas_est},"buyAmount":str(best_out)}
