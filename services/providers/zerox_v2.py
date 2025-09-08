from __future__ import annotations
import json, os, urllib.parse, urllib.request
from typing import Any, Dict, Optional
from web3 import Web3

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
            "sellAmount": str(int(sell_amount)),  # stringified int
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
        data["allowanceTarget"] = allow.get("spender") or data.get("allowanceTarget")
        data["tx"] = tx
        return data

# ===== Camelot V2 (Arbitrum; local on-chain; keyless) =====
# Router has a 'referrer' parameter in swapExactTokensForTokensSupportingFeeOnTransferTokens.
# Paladin audit confirms the extra parameter. (Arbitrum router: 0xC873…2448d)
# https://paladinsec.co/assets/audits/20221030_Paladin_Camelot_Final_Report.pdf
CAMELOT_V2_ROUTER = {"arbitrum": "0xC873FECbd354f5A56E00E710B90EF4201db2448d"}
WETH_ARBITRUM = "0x82aF49447D8a07e3bd95BDdB56f35241523fBab1"
_ABI_V2_CAMELOT = [
  {"inputs":[
      {"internalType":"uint256","name":"amountIn","type":"uint256"},
      {"internalType":"uint256","name":"amountOutMin","type":"uint256"},
      {"internalType":"address[]","name":"path","type":"address[]"},
      {"internalType":"address","name":"to","type":"address"},
      {"internalType":"address","name":"referrer","type":"address"},
      {"internalType":"uint256","name":"deadline","type":"uint256"}],
   "name":"swapExactTokensForTokensSupportingFeeOnTransferTokens",
   "outputs":[], "stateMutability":"nonpayable","type":"function"},
  {"inputs":[
      {"internalType":"uint256","name":"amountIn","type":"uint256"},
      {"internalType":"address[]","name":"path","type":"address[]"}],
   "name":"getAmountsOut",
   "outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],
   "stateMutability":"view","type":"function"}
]

