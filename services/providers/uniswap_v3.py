
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


# --- Uniswap V3 Factory (for getPool) ---
_ABI_UNI_FACTORY = [
  {"inputs":[
      {"internalType":"address","name":"tokenA","type":"address"},
      {"internalType":"address","name":"tokenB","type":"address"},
      {"internalType":"uint24","name":"fee","type":"uint24"}],
   "name":"getPool",
   "outputs":[{"internalType":"address","name":"pool","type":"address"}],
   "stateMutability":"view","type":"function"}
]

UNI_V3_FACTORY = {
  # Uniswap V3 Core Factory address is the same on many networks (verify per chain)
  "ethereum": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
  "arbitrum": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
}



def _uni_v3_encode_path(tokens: list[str], fees: list[int]) -> bytes:
    """Encode V3 path: token(20) + fee(3) + token(20) [+ ...]."""
    from eth_abi import encode
    path = b""
    if len(tokens) < 2 or len(fees) != len(tokens) - 1:
        raise ValueError("invalid path")
    def _to20(a: str) -> bytes:
        from web3 import Web3
        return bytes.fromhex(Web3.to_checksum_address(a)[2:])
    def _to3fee(f: int) -> bytes:
        return int(f).to_bytes(3, "big")
    path += _to20(tokens[0])
    for i, f in enumerate(fees):
        path += _to3fee(int(f))
        path += _to20(tokens[i+1])
    return path


def uniswap_v3_quote_and_build_v2(self, chain: str, token_in: str, token_out: str, amount_in: int, *, slippage_bps: int=100) -> dict:
    """
    Factory-backed Uniswap V3 quoting:
      - normalize native→WETH
      - probe fee tiers [500, 3000, 10000] with Factory.getPool before quoter call
      - try 1-hop first; if neither side is WETH, try WETH multihop
      - build Router02 tx (exactInputSingle or exactInput)
    """
    import time
    from web3 import Web3
    ch = chain.lower().strip()
    conf = self._cfg_norm(chain)  # already normalizes keys
    router_addr = Web3.to_checksum_address(conf["SWAP_ROUTER"])
    quoter_addr = Web3.to_checksum_address(conf["QUOTER_V2"])
    weth_addr   = Web3.to_checksum_address(conf["WETH"])
    w3 = self._w3(chain)
    acct = w3.eth.default_account or "0x0000000000000000000000000000000000000000"

    def _dbg(*a):
        import os
        if os.getenv("DEBUG_SWAP","0") in ("1","true","TRUE"):
            print(*a)

    # normalize native sentinel to WETH
    def _norm(a: str) -> str:
        low = str(a).lower()
        if low in ("eth","native","0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"):
            return weth_addr
        return Web3.to_checksum_address(a)

    t_in  = _norm(token_in)
    t_out = _norm(token_out)
    if int(amount_in) <= 0:
        return {"__error__":"amount_in must be > 0"}

    # contracts
    quoter = w3.eth.contract(quoter_addr, abi=_ABI_UNI_QUOTER_V2)
    quoter_v1 = None
    qv1_addr = UNI_V3_QUOTER_V1.get(ch)
    if qv1_addr:
        try:
            quoter_v1 = w3.eth.contract(Web3.to_checksum_address(qv1_addr), abi=_ABI_UNI_QUOTER_V1)
        except Exception:
            quoter_v1 = None
    router = w3.eth.contract(router_addr, abi=_ABI_UNI_ROUTER02)
    # factory (for existence check)
    factory_addr = UNI_V3_FACTORY.get(ch)
    if not factory_addr:
        return {"__error__": f"UniswapV3 Factory not configured for {chain}"}
    factory = w3.eth.contract(Web3.to_checksum_address(factory_addr), abi=_ABI_UNI_FACTORY)

    fees = [500, 3000, 10000]

    best = None  # (route_kind, fee_or_tuple, amountOut, extra)
    # route_kind: "1hop" or "2hop"
    # extra: for 1hop -> None; for 2hop -> (fee_in, fee_out)

    # 1) probe direct (1-hop)
    for f in fees:
        try:
            pool = factory.functions.getPool(t_in, t_out, int(f)).call()
            if str(pool).lower() == '0x0000000000000000000000000000000000000000':
                _dbg(f"[UniV3] no pool for 1-hop fee={f}")
                continue
            try:
                out, *_ = quoter.functions.quoteExactInputSingle(
                (t_in, t_out, int(f), int(amount_in), 0)
            ).call()
            except Exception as e_v2:
                _dbg(f"[UniV3] 1-hop v2 failed fee={f} err={e_v2!r}; trying v1…")
                if quoter_v1 is not None:
                    out = quoter_v1.functions.quoteExactInputSingle(t_in, t_out, int(f), int(amount_in), 0).call()
                else:
                    raise
            _dbg(f"[UniV3] 1-hop fee={f} amountOut={int(out)}")
            if int(out) > 0 and (best is None or int(out) > best[2]):
                best = ("1hop", f, int(out), None)
        except Exception as e:
            _dbg(f"[UniV3] 1-hop probe fee={f} error: {e!r}")

    # 2) probe via WETH (2-hop) if neither side is WETH
    if best is None and t_in != weth_addr and t_out != weth_addr:
        for f1 in fees:
            try:
                pool1 = factory.functions.getPool(t_in, weth_addr, int(f1)).call()
                if int(pool1, 16) == 0:
                    continue
                out1, *_ = quoter.functions.quoteExactInputSingle(
                    (t_in, weth_addr, int(f1), int(amount_in), 0)
                ).call()
            except Exception as e_v2:
                _dbg(f"[UniV3] 1-hop v2 failed fee={f} err={e_v2!r}; trying v1…")
                if quoter_v1 is not None:
                    out = quoter_v1.functions.quoteExactInputSingle(t_in, t_out, int(f), int(amount_in), 0).call()
                else:
                    raise
            except Exception:
                continue
            if int(out1) <= 0:
                continue
            for f2 in fees:
                try:
                    pool2 = factory.functions.getPool(weth_addr, t_out, int(f2)).call()
                    if int(pool2, 16) == 0:
                        continue
                    out2, *_ = quoter.functions.quoteExactInputSingle(
                        (weth_addr, t_out, int(f2), int(out1), 0)
                    ).call()
            except Exception as e_v2:
                _dbg(f"[UniV3] 1-hop v2 failed fee={f} err={e_v2!r}; trying v1…")
                if quoter_v1 is not None:
                    out = quoter_v1.functions.quoteExactInputSingle(t_in, t_out, int(f), int(amount_in), 0).call()
                else:
                    raise
                    _dbg(f"[UniV3] 2-hop fees=({f1},{f2}) amountOut={int(out2)}")
                    if int(out2) > 0 and (best is None or int(out2) > best[2]):
                        best = ("2hop", (f1, f2), int(out2), None)
                except Exception as e:
                    _dbg(f"[UniV3] 2-hop probe fees=({f1},{f2}) error: {e!r}")

    if best is None:
        return {"__error__": "UniswapV3: no viable pool (direct or via WETH)"}

    # build router tx
    deadline = int(time.time()) + 900
    out_min  = int(best[2] * (10_000 - int(slippage_bps)) // 10_000)

    if best[0] == "1hop":
        fee = int(best[1])
        params = (t_in, t_out, fee, acct, deadline, int(amount_in), out_min, 0)
        data = router.functions.exactInputSingle(params)._encode_transaction_data()
        gas_est = 0
        try:
            gas_est = int(w3.eth.estimate_gas({"from": acct, "to": router_addr, "data": data, "value": 0}) * 1.2)
        except Exception:
            gas_est = 350000
        return {
            "aggregator": "UniswapV3",
            "allowanceTarget": router_addr,
            "tx": {"to": router_addr, "data": data, "value": 0, "gas": gas_est},
            "buyAmount": str(best[2]),
            "route": {"kind":"1hop","fee":fee}
        }

    # 2-hop via WETH
    (f1, f2) = best[1]
    path = _encode_path([t_in, weth_addr, t_out], [int(f1), int(f2)])
    params = (path, acct, deadline, int(amount_in), out_min)
    data = router.functions.exactInput(params)._encode_transaction_data()
    gas_est = 0
    try:
        gas_est = int(w3.eth.estimate_gas({"from": acct, "to": router_addr, "data": data, "value": 0}) * 1.2)
    except Exception:
        gas_est = 400000
    return {
        "aggregator": "UniswapV3",
        "allowanceTarget": router_addr,
        "tx": {"to": router_addr, "data": data, "value": 0, "gas": gas_est},
        "buyAmount": str(best[2]),
        "route": {"kind":"2hop","fees":[int(f1), int(f2)]}
    }

UniswapV3Local.quote_and_build = uniswap_v3_quote_and_build_v2


# Optional Quoter V1 fallback (mainnet)
UNI_V3_QUOTER_V1 = {
    "ethereum": "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
}


_ABI_UNI_QUOTER_V1 = [
  {"inputs":[
      {"internalType":"address","name":"tokenIn","type":"address"},
      {"internalType":"address","name":"tokenOut","type":"address"},
      {"internalType":"uint24","name":"fee","type":"uint24"},
      {"internalType":"uint256","name":"amountIn","type":"uint256"},
      {"internalType":"uint160","name":"sqrtPriceLimitX96","type":"uint160"}],
   "name":"quoteExactInputSingle",
   "outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],
   "stateMutability":"nonpayable","type":"function"},
  {"inputs":[{"internalType":"bytes","name":"path","type":"bytes"}],
   "name":"quoteExactInput",
   "outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],
   "stateMutability":"nonpayable","type":"function"}
]

def uniswap_v3_quote_and_build_v2(self, chain: str, token_in: str, token_out: str, amount_in: int, *, slippage_bps: int=100) -> dict:
    """Robust UniV3 quote: Factory check → QuoterV2 with V1 fallback; 1-hop + 2-hop via WETH; build Router02 tx."""
    import time, os
    from web3 import Web3
    def _dbg(msg: str):
        if os.getenv("DEBUG_SWAP","0") in ("1","true","TRUE"):
            print(msg)

    ch = chain.lower().strip()
    w3 = self._w3(chain)
    conf = self._cfg_norm(chain) if hasattr(self, "_cfg_norm") else {}
    router_addr = Web3.to_checksum_address(conf.get("SWAP_ROUTER") or conf.get("ROUTER"))
    quoter_v2_addr = Web3.to_checksum_address(conf.get("QUOTER_V2"))
    factory_addr   = Web3.to_checksum_address( (UNI_V3_FACTORY.get(ch) or UNI_V3_FACTORY.get("ethereum")) )
    weth_addr      = Web3.to_checksum_address(conf.get("WETH"))
    if not router_addr or not quoter_v2_addr or not weth_addr:
        return {"__error__": "UniswapV3: missing config (router/quoter/weth)"}

    t_in  = Web3.to_checksum_address(token_in if token_in.lower() not in ("eth","native","0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee") else weth_addr)
    t_out = Web3.to_checksum_address(token_out if token_out.lower() not in ("eth","native","0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee") else weth_addr)
    if t_in == t_out:
        return {"__error__": "UniswapV3: token_in == token_out"}

    factory = w3.eth.contract(factory_addr, abi=_ABI_UNI_FACTORY)
    quoter  = w3.eth.contract(quoter_v2_addr, abi=_ABI_UNI_QUOTER_V2)
    quoter_v1 = None
    if ch in UNI_V3_QUOTER_V1:
        try:
            quoter_v1 = w3.eth.contract(Web3.to_checksum_address(UNI_V3_QUOTER_V1[ch]), abi=_ABI_UNI_QUOTER_V1)
        except Exception:
            quoter_v1 = None

    fees = [500, 3000, 10000]
    best = None  # (out, kind, meta)
    # 1) Direct 1-hop
    for f in fees:
        try:
            pool = factory.functions.getPool(t_in, t_out, int(f)).call()
            if str(pool).lower() == "0x0000000000000000000000000000000000000000":
                _dbg(f"[UniV3] no pool 1-hop fee={f}")
                continue
            _dbg(f"[UniV3] pool 1-hop fee={f}: {pool}")
            try:
                out, *_ = quoter.functions.quoteExactInputSingle(t_in, t_out, int(f), int(amount_in), 0).call()
            except Exception as e2:
                _dbg(f"[UniV3] v2 quoteExactInputSingle failed fee={f}: {e2!r}")
                if quoter_v1 is not None:
                    out = quoter_v1.functions.quoteExactInputSingle(t_in, t_out, int(f), int(amount_in), 0).call()
                else:
                    continue
            out = int(out)
            if out > 0 and (best is None or out > best[0]):
                best = (out, "1hop", {"fee": int(f)})
                _dbg(f"[UniV3] 1-hop candidate fee={f} out={out}")
        except Exception as e:
            _dbg(f"[UniV3] 1-hop probe fee={f} error: {e!r}")

    # 2) 2-hop via WETH
    if t_in != weth_addr and t_out != weth_addr:
        for f1 in fees:
            for f2 in fees:
                try:
                    p1 = factory.functions.getPool(t_in, weth_addr, int(f1)).call()
                    p2 = factory.functions.getPool(weth_addr, t_out, int(f2)).call()
                    if str(p1).lower() == "0x0000000000000000000000000000000000000000" or str(p2).lower() == "0x0000000000000000000000000000000000000000":
                        _dbg(f"[UniV3] no pool 2-hop fees=({f1},{f2})")
                        continue
                    _dbg(f"[UniV3] pools 2-hop fees=({f1},{f2}): {p1} , {p2}")
                    path = _encode_path([t_in, weth_addr, t_out], [int(f1), int(f2)])
                    try:
                        out2, *_ = quoter.functions.quoteExactInput(path, int(amount_in)).call()
                    except Exception as e2b:
                        _dbg(f"[UniV3] v2 quoteExactInput failed fees=({f1},{f2}): {e2b!r}")
                        if quoter_v1 is not None:
                            out2 = quoter_v1.functions.quoteExactInput(path, int(amount_in)).call()
                        else:
                            continue
                    out2 = int(out2)
                    if out2 > 0 and (best is None or out2 > best[0]):
                        best = (out2, "2hop", {"fees": (int(f1), int(f2)), "path": path})
                        _dbg(f"[UniV3] 2-hop candidate fees=({f1},{f2}) out={out2}")
                except Exception as e:
                    _dbg(f"[UniV3] 2-hop probe fees=({f1},{f2}) error: {e!r}")

    if best is None:
        return {"__error__": "UniswapV3: no viable pool (direct or via WETH)"}

    recipient = w3.eth.default_account
    if not recipient:
        return {"__error__": "UniswapV3: missing default_account for tx build"}

    amount_out = best[0]
    out_min = int(amount_out * (10_000 - int(slippage_bps)) // 10_000)
    deadline = int(time.time()) + 600
    router = w3.eth.contract(router_addr, abi=_ABI_UNI_ROUTER02)

    if best[1] == "1hop":
        fee = int(best[2]["fee"])
        fn = router.functions.exactInputSingle((
            t_in, t_out, fee, recipient, deadline, int(amount_in), int(out_min), 0
        ))
    else:
        path = best[2]["path"]
        fn = router.functions.exactInput((
            path, recipient, deadline, int(amount_in), int(out_min)
        ))

    data = fn._encode_transaction_data()
    gas_est = 0
    try:
        gas_est = int(w3.eth.estimate_gas({"from": recipient, "to": router_addr, "data": data, "value": 0}) * 1.2)
    except Exception:
        gas_est = 350000

    return {
        "aggregator": "UniswapV3",
        "allowanceTarget": router_addr,
        "tx": {"to": router_addr, "data": data, "value": 0, "gas": gas_est},
        "buyAmount": str(int(amount_out)),
        "meta": best[1:]
    }

# Bind override
UniswapV3Local.quote_and_build = uniswap_v3_quote_and_build_v2
