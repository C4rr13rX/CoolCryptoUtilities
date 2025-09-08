from __future__ import annotations
import time
from typing import Any, Dict, Optional, List
from web3 import Web3

UNI_V3 = {
    'ethereum': {'router': '0x68b3465833FB72A70ecDF485E0e4C7bD8665Fc45', 'quoter2': '0x61fFE014bA17989E743c5F6cB21bF9697530B21e', 'WETH': '0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2'},
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


class UniswapV3Local:
    def _cfg_norm(self, chain: str) -> dict:
        ch = chain.lower().strip()
        if ch not in UNI_V3:
            raise ValueError(f"UniswapV3 unsupported on {chain}")
        raw = UNI_V3[ch]
        router = raw.get('ROUTER') or raw.get('router')
        quoter = raw.get('QUOTER_V2') or raw.get('quoter2') or raw.get('quoter')
        weth   = raw.get('WETH') or raw.get('weth')
        if not router or not quoter or not weth:
            raise KeyError(f"UniswapV3 config incomplete for {chain}; need router, QUOTER_V2/quoter2, WETH")
        return {'ROUTER': router, 'SWAP_ROUTER': router, 'QUOTER_V2': quoter, 'WETH': weth}

    FEES = (500, 3000, 10000)
    def __init__(self, w3_provider_callable):
        self._w3 = w3_provider_callable
    def _cfg(self, chain: str) -> Dict[str,str]:
        ch = chain.lower()
        if ch not in UNI_V3: raise ValueError(f"UniswapV3 unsupported on {chain}")
        return UNI_V3[ch]
    def _quoter(self, w3, conf):  # build contract
        return w3.eth.contract(Web3.to_checksum_address(conf["QUOTER_V2"]), abi=_ABI_UNI_QUOTER_V2)
    def _router(self, w3, conf):
        addr = conf.get('SWAP_ROUTER') or conf.get('ROUTER') or conf.get('router')
        if not addr:
            raise KeyError('UniswapV3: router address missing')
        try:
            abi = _ABI_UNI_ROUTER02  # SwapRouter02
        except NameError:
            abi = _ABI_UNI_ROUTER    # fallback name
        return w3.eth.contract(Web3.to_checksum_address(addr), abi=abi)

    def quote_and_build(self, chain: str, token_in: str, token_out: str, amount_in: int, *, slippage_bps: int=100, recipient: Optional[str]=None) -> Dict[str,Any]:
        conf = self._cfg_norm(chain)
        w3 = self._w3(chain)
        t_in  = Web3.to_checksum_address(token_in)
        t_out = Web3.to_checksum_address(token_out)
        weth  = Web3.to_checksum_address(conf["WETH"])
        quoter = self._quoter(w3, conf)
        router = self._router(w3, conf)
        best = None  # (kind, meta, out)

        # --- direct (single pool) over common fee tiers
        for fee in self.FEES:
            try:
                out, *_ = quoter.functions.quoteExactInputSingle(
                    (t_in, t_out, int(fee), int(amount_in), 0)
                ).call()
                if int(out) > 0 and (best is None or int(out) > best[2]):
                    best = ("single", {"fee": fee}, int(out))
            except Exception:
                pass

        # --- 2-hop via WETH (path)
        for f1 in self.FEES:
            for f2 in self.FEES:
                try:
                    path = _encode_path([t_in, weth, t_out], [int(f1), int(f2)])
                    out, *_ = quoter.functions.quoteExactInput(path, int(amount_in)).call()
                    if int(out) > 0 and (best is None or int(out) > best[2]):
                        best = ("path", {"fees": (f1, f2)}, int(out))
                except Exception:
                    pass

        if not best:
            return {"__error__": "UniswapV3: no viable pool (direct or via WETH)"}

        out_min = int(best[2] * (10_000 - int(slippage_bps)) / 10_000)
        deadline = int(time.time()) + 600
        to_addr = Web3.to_checksum_address(recipient or (w3.eth.default_account or "0x0000000000000000000000000000000000000000"))
        router_to = Web3.to_checksum_address(conf["SWAP_ROUTER"])

        if best[0] == "single":
            fee = int(best[1]["fee"])
            fn = router.functions.exactInputSingle((
                t_in, t_out, fee, to_addr, deadline, int(amount_in), out_min, 0
            ))
        else:
            f1, f2 = best[1]["fees"]
            path = _encode_path([t_in, weth, t_out], [int(f1), int(f2)])
            fn = router.functions.exactInput((path, to_addr, deadline, int(amount_in), out_min))

        data = fn._encode_transaction_data()
        # estimate gas (payable = yes, but value=0 for ERC-20 -> ERC-20)
        try:
            gas_est = int(w3.eth.estimate_gas({"from": to_addr, "to": router_to, "data": data, "value": 0}) * 1.2)
        except Exception:
            gas_est = 300000
        return {
            "aggregator": "UniswapV3",
            "allowanceTarget": router_to,
            "tx": {"to": router_to, "data": data, "value": 0, "gas": gas_est},
            "buyAmount": str(best[2]),
            "sellAmount": str(int(amount_in)),
            "sellToken": t_in, "buyToken": t_out,
        }

# ===== Sushi (UniswapV2-style) on Arbitrum; local on-chain; keyless =====
# Arbitrum router (classic V2): 0x1b02da8cB0d097eB8D57A175B88c7D8b47997506
# https://arbiscan.io/address/0x1b02da8cb0d097eb8d57a175b88c7d8b47997506
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


def _to_checksum(addr, Web3):
    return Web3.to_checksum_address(addr)

def _encode_v3_path(tokens, fees):
    # Encode a Uniswap V3 path: token(20) + fee(3) + token(20) [+ fee + token]...
    if len(tokens) != len(fees) + 1:
        raise ValueError("V3 path: tokens must be n+1 for n fees")
    parts = []
    for i in range(len(fees)):
        t = tokens[i].lower().replace("0x","")
        n = tokens[i+1].lower().replace("0x","")
        f = int(fees[i])
        if not (0 <= f < (1<<24)):
            raise ValueError("fee must fit uint24")
        parts.append(t.rjust(40,"0"))
        parts.append(f"{f:06x}")
        parts.append(n.rjust(40,"0"))
    return "0x" + "".join(parts)


# ## UNI_V3_ROBUST_IMPL
def quote_and_build(self, chain: str, token_in: str, token_out: str, amount_in: int, *, slippage_bps: int=100):
    import time as _t
    ch = chain.lower().strip()
    conf = self._cfg_norm(ch)
    w3 = self._w3(chain)
    # Choose ABI name present in file
    try:
        _abi_router = _ABI_UNI_ROUTER02
    except NameError:
        _abi_router = _ABI_UNI_ROUTER
    router = w3.eth.contract(Web3.to_checksum_address(conf["SWAP_ROUTER"]), abi=_abi_router)
    quoter = w3.eth.contract(Web3.to_checksum_address(conf["QUOTER_V2"]), abi=_ABI_UNI_QUOTER_V2)
    recip  = w3.eth.default_account or "0x0000000000000000000000000000000000000000"

    t_in  = _to_checksum(token_in, Web3)
    t_out = _to_checksum(token_out, Web3)
    weth  = _to_checksum(conf["WETH"], Web3)
    if t_in == t_out:
        return {"__error__":"UniswapV3: tokenIn == tokenOut"}
    if int(amount_in) <= 0:
        return {"__error__":"UniswapV3: amount_in must be > 0"}

    # Probe single-hop: 0.05%, 0.3%, 1%
    fees_direct = [500, 3000, 10000]
    best = None
    for fee in fees_direct:
        try:
            out, _, _, _ = quoter.functions.quoteExactInputSingle(
                (t_in, t_out, int(amount_in), int(fee), 0)
            ).call()
            if int(out) > 0:
                best = ("single", (int(fee),), int(out))
                break
        except Exception:
            pass

    # Multi-hop via WETH if needed and neither side is WETH
    if best is None and t_in != weth and t_out != weth:
        for f1, f2 in [(500,500),(500,3000),(3000,500),(3000,3000),(3000,10000)]:
            try:
                path = _encode_v3_path([t_in, weth, t_out], [int(f1), int(f2)])
                out, _, _, _ = quoter.functions.quoteExactInput(path, int(amount_in)).call()
                if int(out) > 0:
                    best = ("multi", (int(f1), int(f2)), int(out))
                    break
            except Exception:
                pass

    if best is None:
        return {"__error__":"UniswapV3: no viable pool (direct or via WETH)"}

    kind, fees, out_amt = best
    out_min = int(out_amt * (10_000 - int(slippage_bps)) // 10_000)
    deadline = int(_t.time()) + 600

    if kind == "single":
        fee = int(fees[0])
        fn = router.functions.exactInputSingle((
            t_in, t_out, fee, recip, deadline, int(amount_in), int(out_min), 0
        ))
    else:
        path = _encode_v3_path([t_in, weth, t_out], [int(fees[0]), int(fees[1])])
        fn = router.functions.exactInput((path, recip, deadline, int(amount_in), int(out_min)))

    data = fn._encode_transaction_data()
    to_addr = conf["SWAP_ROUTER"]
    try:
        gas = int(w3.eth.estimate_gas({"from": recip, "to": to_addr, "data": data, "value": 0}) * 1.2)
    except Exception:
        gas = 400000

    return {
        "aggregator":"UniswapV3",
        "allowanceTarget": to_addr,
        "tx": {"to": to_addr, "data": data, "value": 0, "gas": gas},
        "buyAmount": str(out_amt),
        "route": {"kind": kind, "fees": fees}
    }
