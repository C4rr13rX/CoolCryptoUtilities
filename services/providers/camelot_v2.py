from __future__ import annotations
import time
from typing import Any, Dict, Optional, List, Tuple
from web3 import Web3

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

class CamelotV2Local:
    def __init__(self, w3_provider_callable):
        self._w3 = w3_provider_callable
    def quote_and_build(self, chain: str, token_in: str, token_out: str, amount_in: int, *, slippage_bps: int=100, recipient: Optional[str]=None) -> Dict[str,Any]:
        ch = chain.lower()
        if ch not in CAMELOT_V2_ROUTER:
            return {"__error__": f"CamelotV2 unsupported on {chain}"}
        w3 = self._w3(chain)
        router_addr = Web3.to_checksum_address(CAMELOT_V2_ROUTER[ch])
        t_in  = Web3.to_checksum_address(token_in)
        t_out = Web3.to_checksum_address(token_out)
        weth  = Web3.to_checksum_address(WETH_ARBITRUM)
        router = w3.eth.contract(router_addr, abi=_ABI_V2_CAMELOT)

        paths: List[Tuple[List[str],int]] = []
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
        to_addr = Web3.to_checksum_address(recipient or (w3.eth.default_account or "0x0000000000000000000000000000000000000000"))
        fn = router.functions.swapExactTokensForTokensSupportingFeeOnTransferTokens(
            int(amount_in), out_min, path, to_addr, "0x0000000000000000000000000000000000000000", deadline
        )
        data = fn._encode_transaction_data()
        try:
            gas_est = int(w3.eth.estimate_gas({"from": to_addr, "to": router_addr, "data": data, "value": 0}) * 1.2)
        except Exception:
            gas_est = 250000
        return {
            "aggregator": "CamelotV2",
            "allowanceTarget": router_addr,
            "tx": {"to": router_addr, "data": data, "value": 0, "gas": gas_est},
            "buyAmount": str(best_out),
            "sellAmount": str(int(amount_in)),
            "sellToken": t_in, "buyToken": t_out,
        }

# ===== Uniswap V3 (Arbitrum; local on-chain; keyless) =====
# Official Arbitrum deployments: QuoterV2 0x61fFE0…B21e, SwapRouter02 0x68b346…5Fc45, WETH 0x82aF49…3fBab1
# https://docs.uniswap.org/contracts/v3/reference/deployments/arbitrum-deployments
UNI_V3 = {
    "arbitrum": {
        "QUOTER_V2":   "0x61fFE014bA17989E743c5F6cB21bF9697530B21e",
        "SWAP_ROUTER": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",  # SwapRouter02
        "WETH":        "0x82aF49447D8a07e3bd95BDdB56f35241523fBab1",
    }
}
# Minimal ABIs (QuoterV2 + SwapRouter02)
_ABI_UNI_QUOTER_V2 = [
  {"inputs":[{"components":[
      {"internalType":"address","name":"tokenIn","type":"address"},
      {"internalType":"address","name":"tokenOut","type":"address"},
      {"internalType":"uint24","name":"fee","type":"uint24"},
      {"internalType":"uint256","name":"amountIn","type":"uint256"},
      {"internalType":"uint160","name":"sqrtPriceLimitX96","type":"uint160"}],
    "internalType":"struct IQuoterV2.QuoteExactInputSingleParams","name":"params","type":"tuple"}],
   "name":"quoteExactInputSingle",
   "outputs":[
      {"internalType":"uint256","name":"amountOut","type":"uint256"},
      {"internalType":"uint160","name":"sqrtPriceX96After","type":"uint160"},
      {"internalType":"uint32","name":"initializedTicksCrossed","type":"uint32"},
      {"internalType":"uint256","name":"gasEstimate","type":"uint256"}],
   "stateMutability":"nonpayable","type":"function"},
  {"inputs":[
      {"internalType":"bytes","name":"path","type":"bytes"},
      {"internalType":"uint256","name":"amountIn","type":"uint256"}],
   "name":"quoteExactInput",
   "outputs":[
      {"internalType":"uint256","name":"amountOut","type":"uint256"},
      {"internalType":"uint160","name":"sqrtPriceX96After","type":"uint160"},
      {"internalType":"uint32","name":"initializedTicksCrossed","type":"uint32"},
      {"internalType":"uint256","name":"gasEstimate","type":"uint256"}],
   "stateMutability":"nonpayable","type":"function"}
]
_ABI_UNI_ROUTER02 = [
  {"inputs":[{"components":[
      {"internalType":"address","name":"tokenIn","type":"address"},
      {"internalType":"address","name":"tokenOut","type":"address"},
      {"internalType":"uint24","name":"fee","type":"uint24"},
      {"internalType":"address","name":"recipient","type":"address"},
      {"internalType":"uint256","name":"deadline","type":"uint256"},
      {"internalType":"uint256","name":"amountIn","type":"uint256"},
      {"internalType":"uint256","name":"amountOutMinimum","type":"uint256"},
      {"internalType":"uint160","name":"sqrtPriceLimitX96","type":"uint160"}],
    "internalType":"struct ISwapRouter.ExactInputSingleParams","name":"params","type":"tuple"}],
   "name":"exactInputSingle","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],
   "stateMutability":"payable","type":"function"},
  {"inputs":[{"components":[
      {"internalType":"bytes","name":"path","type":"bytes"},
      {"internalType":"address","name":"recipient","type":"address"},
      {"internalType":"uint256","name":"deadline","type":"uint256"},
      {"internalType":"uint256","name":"amountIn","type":"uint256"},
      {"internalType":"uint256","name":"amountOutMinimum","type":"uint256"}],
    "internalType":"struct ISwapRouter.ExactInputParams","name":"params","type":"tuple"}],
   "name":"exactInput","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],
   "stateMutability":"payable","type":"function"}
]
def _enc_addr(a: str) -> bytes:
    return bytes.fromhex(Web3.to_checksum_address(a)[2:])
def _enc_fee(fee: int) -> bytes:
    return int(fee).to_bytes(3, byteorder="big")
def _encode_path(tokens: List[str], fees: List[int]) -> bytes:
    # token0 | fee0 | token1 | fee1 | token2 | ...
    if len(tokens) != len(fees) + 1: raise ValueError("path: tokens must be n+1 of fees")
    b = b""
    for i in range(len(fees)):
        b += _enc_addr(tokens[i]) + _enc_fee(fees[i])
    b += _enc_addr(tokens[-1])
    return b

