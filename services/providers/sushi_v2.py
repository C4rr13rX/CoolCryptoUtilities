from __future__ import annotations
import time
from typing import Any, Dict, Optional, List, Tuple
from web3 import Web3

SUSHI_V2_ROUTER = {"arbitrum": "0x1b02da8cB0d097eB8D57A175B88c7D8b47997506"}
_ABI_V2_GENERIC = [
  {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},
             {"internalType":"uint256","name":"amountOutMin","type":"uint256"},
             {"internalType":"address[]","name":"path","type":"address[]"},
             {"internalType":"address","name":"to","type":"address"},
             {"internalType":"uint256","name":"deadline","type":"uint256"}],
   "name":"swapExactTokensForTokensSupportingFeeOnTransferTokens",
   "outputs":[], "stateMutability":"nonpayable","type":"function"},
  {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},
             {"internalType":"address[]","name":"path","type":"address[]"}],
   "name":"getAmountsOut",
   "outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],
   "stateMutability":"view","type":"function"}
]

class SushiV2Local:
    def __init__(self, w3_provider_callable, weth_addr_by_chain: Optional[Dict[str,str]]=None):
        self._w3 = w3_provider_callable
        self._weth = {"arbitrum": WETH_ARBITRUM}
        if weth_addr_by_chain: self._weth.update({k.lower():v for k,v in weth_addr_by_chain.items()})
    def quote_and_build(self, chain: str, token_in: str, token_out: str, amount_in: int, *, slippage_bps: int=100, recipient: Optional[str]=None) -> Dict[str,Any]:
        ch = chain.lower()
        if ch not in SUSHI_V2_ROUTER:
            return {"__error__": f"Sushi V2 unsupported on {chain}"}
        w3 = self._w3(chain)
        router_addr = Web3.to_checksum_address(SUSHI_V2_ROUTER[ch])
        t_in  = Web3.to_checksum_address(token_in)
        t_out = Web3.to_checksum_address(token_out)
        weth  = Web3.to_checksum_address(self._weth[ch])
        router = w3.eth.contract(router_addr, abi=_ABI_V2_GENERIC)
        paths: List[Tuple[List[str],int]] = []
        # direct
        try:
            out = router.functions.getAmountsOut(int(amount_in), [t_in, t_out]).call()[-1]
            if int(out) > 0: paths.append(([t_in, t_out], int(out)))
        except Exception: pass
        # via WETH
        try:
            out1 = router.functions.getAmountsOut(int(amount_in), [t_in, weth]).call()[-1]
            out2 = router.functions.getAmountsOut(int(out1), [weth, t_out]).call()[-1]
            if int(out1) > 0 and int(out2) > 0: paths.append(([t_in, weth, t_out], int(out2)))
        except Exception: pass
        if not paths:
            return {"__error__": "SushiV2: no path (direct or via WETH)"}
        path, best_out = max(paths, key=lambda x: x[1])
        out_min = int(best_out * (10_000 - slippage_bps) / 10_000)
        deadline = int(time.time()) + 600
        to_addr = Web3.to_checksum_address(recipient or (w3.eth.default_account or "0x0000000000000000000000000000000000000000"))
        fn = router.functions.swapExactTokensForTokensSupportingFeeOnTransferTokens(
            int(amount_in), out_min, path, to_addr, deadline
        )
        data = fn._encode_transaction_data()
        try:
            gas_est = int(w3.eth.estimate_gas({"from": to_addr, "to": router_addr, "data": data, "value": 0}) * 1.2)
        except Exception:
            gas_est = 250000
        return {
            "aggregator": "SushiV2",
            "allowanceTarget": router_addr,
            "tx": {"to": router_addr, "data": data, "value": 0, "gas": gas_est},
            "buyAmount": str(best_out),
            "sellAmount": str(int(amount_in)),
            "sellToken": t_in, "buyToken": t_out,
        }

