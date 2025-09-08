
from __future__ import annotations
import time
from typing import Any, Dict, Optional, List
from web3 import Web3

# Supported chains + canonical contracts
# (Router is SwapRouter02; Quoter is QuoterV2)
UNI_V3 = {
    "ethereum": {
        "SWAP_ROUTER": "0x68b3465833FB72A70ecDF485E0e4C7bD8665Fc45",  # SwapRouter02
        "QUOTER_V2":   "0x61fFE014bA17989E743c5F6cB21bF9697530B21e",
        "WETH":        "0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2",
    },
    "arbitrum": {
        "SWAP_ROUTER": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",  # SwapRouter02
        "QUOTER_V2":   "0x61fFE014bA17989E743c5F6cB21bF9697530B21e",
        "WETH":        "0x82aF49447D8a07e3bd95BDdB56f35241523fBab1",
    },
}

# --- Minimal ABIs (QuoterV2 + SwapRouter02) ---
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
    # token | fee | token | fee | token ...
    if len(tokens) != len(fees) + 1:
        raise ValueError("path: tokens must be n+1 of fees")
    b = b""
    for i in range(len(fees)):
        b += _enc_addr(tokens[i]) + _enc_fee(int(fees[i]))
    b += _enc_addr(tokens[-1])
    return b


class UniswapV3Local:
    FEES = (500, 3000, 10000)  # 0.05%, 0.3%, 1.0%

    def __init__(self, w3_provider_callable):
        self._w3 = w3_provider_callable

    def _dbg(self, *a):
        import os
        if os.getenv("DEBUG_SWAP","0") in ("1","true","TRUE"):
            print("[UniV3]", *a)

    def _cfg_norm(self, chain: str) -> Dict[str,str]:
        ch = chain.lower().strip()
        if ch not in UNI_V3:
            raise ValueError(f"UniswapV3 unsupported on {chain}")
        raw = UNI_V3[ch]
        # accept any casing of keys that might already exist
        router = raw.get("SWAP_ROUTER") or raw.get("ROUTER") or raw.get("router")
        quoter = raw.get("QUOTER_V2")   or raw.get("quoter2") or raw.get("QUOTER2")
        weth   = raw.get("WETH")        or raw.get("weth")
        if not (router and quoter and weth):
            raise KeyError(f"UniswapV3 config incomplete for {chain}")
        return {"SWAP_ROUTER": router, "QUOTER_V2": quoter, "WETH": weth}

    def _contracts(self, w3, conf):
        quoter = w3.eth.contract(Web3.to_checksum_address(conf["QUOTER_V2"]), abi=_ABI_UNI_QUOTER_V2)
        router = w3.eth.contract(Web3.to_checksum_address(conf["SWAP_ROUTER"]), abi=_ABI_UNI_ROUTER02)
        return quoter, router

    def quote_and_build(self, chain: str, token_in: str, token_out: str, amount_in: int, *, slippage_bps: int=100, recipient: Optional[str]=None) -> Dict[str,Any]:
        conf = self._cfg_norm(chain)
        w3 = self._w3(chain)
        t_in  = Web3.to_checksum_address(token_in)
        t_out = Web3.to_checksum_address(token_out)
        weth  = Web3.to_checksum_address(conf["WETH"])
        quoter, router = self._contracts(w3, conf)
        recip = Web3.to_checksum_address(recipient or (w3.eth.default_account or "0x0000000000000000000000000000000000000001"))

        if int(amount_in) <= 0:
            return {"__error__":"UniswapV3: amount_in must be > 0"}
        if t_in == t_out:
            return {"__error__":"UniswapV3: tokenIn == tokenOut"}

        best = None  # (out, kind, meta)

        # 1) single-hop over common fee tiers
        for fee in self.FEES:
            try:
                out, *_ = quoter.functions.quoteExactInputSingle(
                    (t_in, t_out, int(fee), int(amount_in), 0)
                ).call()
                if int(out) > 0 and (best is None or int(out) > best[0]):
                    best = (int(out), "single", {"fee": int(fee)})
                    self._dbg(f"1-hop found: fee={fee} out={int(out)}")
            except Exception as e:
                self._dbg(f"1-hop probe fee={fee} error: {e!r}")

        # 2) two-hop via WETH if needed
        if (best is None) and (t_in != weth) and (t_out != weth):
            for f1 in self.FEES:
                for f2 in self.FEES:
                    try:
                        path = _encode_path([t_in, weth, t_out], [int(f1), int(f2)])
                        out, *_ = quoter.functions.quoteExactInput(path, int(amount_in)).call()
                        if int(out) > 0 and (best is None or int(out) > best[0]):
                            best = (int(out), "path", {"fees": (int(f1), int(f2))})
                            self._dbg(f"2-hop via WETH found: fees={f1}/{f2} out={int(out)}")
                    except Exception as e:
                        self._dbg(f"2-hop probe fees={f1}/{f2} error: {e!r}")

        if best is None:
            return {"__error__": "UniswapV3: no viable pool (direct or via WETH)"}

        out_min = int(best[0] * (10_000 - int(slippage_bps)) // 10_000)
        deadline = int(time.time()) + 600
        router_to = Web3.to_checksum_address(conf["SWAP_ROUTER"])

        if best[1] == "single":
            fee = int(best[2]["fee"])
            fn = router.functions.exactInputSingle((
                t_in, t_out, fee, recip, deadline,
                int(amount_in), int(out_min), 0
            ))
        else:
            f1, f2 = best[2]["fees"]
            path = _encode_path([t_in, weth, t_out], [int(f1), int(f2)])
            fn = router.functions.exactInput((path, recip, deadline, int(amount_in), int(out_min)))

        data = fn._encode_transaction_data()
        try:
            gas_est = int(w3.eth.estimate_gas({"from": recip, "to": router_to, "data": data, "value": 0}) * 1.2)
        except Exception:
            gas_est = 300_000

        return {
            "aggregator": "UniswapV3",
            "allowanceTarget": router_to,
            "tx": {"to": router_to, "data": data, "value": 0, "gas": gas_est},
            "buyAmount": str(best[0]),
            "route": best[1],
        }
