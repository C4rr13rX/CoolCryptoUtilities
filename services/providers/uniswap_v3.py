from __future__ import annotations
import os, time
from typing import Any, Dict, List, Optional, Tuple
from web3 import Web3

# --------------------------------------------------------------------------
# Canonical addresses (Uniswap v3)
# - Router02 + QuoterV2 are the same on mainnet + most L2s (we map per chain)
# - We also include Factory and WETH for mainnet + arbitrum
# --------------------------------------------------------------------------
UNI_V3 = {
    "ethereum": {
        "ROUTER":   "0x68b3465833FB72A70ecDF485E0e4C7bD8665Fc45",  # SwapRouter02
        "QUOTER_V2":"0x61fFE014bA17989E743c5F6cB21bF9697530B21e",  # QuoterV2
        "WETH":     "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    },
    "arbitrum": {
        "ROUTER":   "0x68b3465833FB72A70ecDF485E0e4C7bD8665Fc45",
        "QUOTER_V2":"0x61fFE014bA17989E743c5F6cB21bF9697530B21e",
        "WETH":     "0x82aF49447D8a07e3bd95BDdB56f35241523fBab1",
    },
}

UNI_V3_FACTORY = {
    "ethereum": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
    "arbitrum": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
}

# Optional Quoter V1 fallback (mainnet)
UNI_V3_QUOTER_V1 = {
    "ethereum": "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
}

# --------------------------------------------------------------------------
# Minimal ABIs (Factory, Quoters, Router02)
# --------------------------------------------------------------------------
_ABI_UNI_FACTORY = [
  {"inputs":[{"internalType":"address","name":"tokenA","type":"address"},
             {"internalType":"address","name":"tokenB","type":"address"},
             {"internalType":"uint24","name":"fee","type":"uint24"}],
   "name":"getPool","outputs":[{"internalType":"address","name":"pool","type":"address"}],
   "stateMutability":"view","type":"function"}
]

# QuoterV2: quoteExactInputSingle((address,address,uint24,uint256,uint160)) and quoteExactInput(bytes,uint256)
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
      {"internalType":"uint160[]","name":"sqrtPriceX96AfterList","type":"uint160[]"},
      {"internalType":"uint32[]","name":"initializedTicksCrossedList","type":"uint32[]"},
      {"internalType":"uint256","name":"gasEstimate","type":"uint256"}],
   "stateMutability":"nonpayable","type":"function"},
  {"inputs":[{"internalType":"bytes","name":"path","type":"bytes"},
             {"internalType":"uint256","name":"amountIn","type":"uint256"}],
   "name":"quoteExactInput",
   "outputs":[
      {"internalType":"uint256","name":"amountOut","type":"uint256"},
      {"internalType":"uint160[]","name":"sqrtPriceX96AfterList","type":"uint160[]"},
      {"internalType":"uint32[]","name":"initializedTicksCrossedList","type":"uint32[]"},
      {"internalType":"uint256","name":"gasEstimate","type":"uint256"}],
   "stateMutability":"nonpayable","type":"function"}
]

# Quoter V1 (mainnet-only fallback): quoteExactInputSingle(address,address,uint24,uint256,uint160)
_ABI_UNI_QUOTER_V1 = [
  {"inputs":[
      {"internalType":"address","name":"tokenIn","type":"address"},
      {"internalType":"address","name":"tokenOut","type":"address"},
      {"internalType":"uint24","name":"fee","type":"uint24"},
      {"internalType":"uint256","name":"amountIn","type":"uint256"},
      {"internalType":"uint160","name":"sqrtPriceLimitX96","type":"uint160"}],
   "name":"quoteExactInputSingle","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],
   "stateMutability":"nonpayable","type":"function"}
]

# SwapRouter02: exactInputSingle((address,address,uint24,address,uint256,uint256,uint256,uint160))
#               exactInput((bytes,address,uint256,uint256))
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

# --------------------------------------------------------------------------
def _dbg(*a):
    if os.getenv("DEBUG_SWAP","0") in ("1","true","TRUE"):
        try: print(*a)
        except Exception: pass

def _encode_path(legs: List[Tuple[str,int]]) -> bytes:
    """
    Encode a Uniswap V3 path: token(20) + fee(3) + token(20) [+ fee + token]...
    legs: [(tokenIn, fee), (tokenOut, fee?) ...] — for N tokens, there are N-1 fees.
    For 1-hop, pass [(tokenIn, fee), (tokenOut, 0)] — the last fee is ignored.
    """
    b = b""
    for i, (tok, fee) in enumerate(legs):
        b += bytes.fromhex(Web3.to_checksum_address(tok)[2:])
        if i < len(legs)-1:  # fee between tokens
            b += int(fee).to_bytes(3, "big")
    return b

class UniswapV3Local:
    def __init__(self, w3_provider_callable):
        self._w3 = w3_provider_callable

    def _cfg_norm(self, chain: str) -> dict:
        ch = chain.lower().strip()
        if ch not in UNI_V3:
            raise ValueError(f"UniswapV3 unsupported on {chain}")
        c = UNI_V3[ch]
        router = c.get("SWAP_ROUTER") or c.get("ROUTER")
        quoter = c.get("QUOTER_V2") or c.get("quoter2")
        weth = c.get("WETH")
        if not (router and quoter and weth):
            raise ValueError(f"UniswapV3 config incomplete for {chain}")
        return {"ROUTER": router, "SWAP_ROUTER": router, "QUOTER_V2": quoter, "WETH": weth}

    def _factory(self, w3, chain: str):
        adr = UNI_V3_FACTORY.get(chain.lower())
        return w3.eth.contract(Web3.to_checksum_address(adr), abi=_ABI_UNI_FACTORY)

    def _router(self, w3, conf: dict):
        return w3.eth.contract(Web3.to_checksum_address(conf["SWAP_ROUTER"]), abi=_ABI_UNI_ROUTER02)

    def _quoter_v2(self, w3, conf: dict):
        return w3.eth.contract(Web3.to_checksum_address(conf["QUOTER_V2"]), abi=_ABI_UNI_QUOTER_V2)

    def _quoter_v1_opt(self, w3, chain: str):
        adr = UNI_V3_QUOTER_V1.get(chain.lower())
        return w3.eth.contract(Web3.to_checksum_address(adr), abi=_ABI_UNI_QUOTER_V1) if adr else None

    # ----------------------------------------------------------------------
    def quote_and_build(self, chain: str, token_in: str, token_out: str, amount_in: int, *, slippage_bps: int=100, recipient: Optional[str]=None) -> Dict[str,Any]:
        """
        Probe fee tiers (500/3000/10000) 1-hop first; if none, probe 2-hop via WETH.
        Build exactInputSingle or exactInput tx on SwapRouter02 for best quote.
        Returns: {"aggregator":"UniswapV3","allowanceTarget":router,"buyAmount":str(out),
                  "tx":{"to":router,"data":..., "value":0,"gas":estimate}}
        or {"__error__": "..."} on failure.
        """
        ch = chain.lower().strip()
        w3 = self._w3(chain)
        conf = self._cfg_norm(chain)
        router = self._router(w3, conf)
        quoter2 = self._quoter_v2(w3, conf)
        quoter1 = self._quoter_v1_opt(w3, chain)
        factory = self._factory(w3, chain)
        to_addr = Web3.to_checksum_address(recipient or (w3.eth.default_account or "0x0000000000000000000000000000000000000000"))

        # normalize "native" to WETH if someone passed it through
        def _norm(x:str)->str:
            x = (x or "").strip().lower()
            if x in ("eth","native","0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"):
                return conf["WETH"]
            return x
        t_in  = Web3.to_checksum_address(_norm(token_in))
        t_out = Web3.to_checksum_address(_norm(token_out))
        if t_in == t_out: return {"__error__":"UniswapV3: token_in == token_out"}

        fees = [500, 3000, 10000]
        amount_in = int(amount_in)
        if amount_in <= 0:
            return {"__error__": "UniswapV3: amount_in must be > 0"}

        call_from = {"from": to_addr}  # helps some RPCs with eth_call

        best: Tuple[str, List[int], int, int] | None = None
        # (mode, [fees...], out_amount, gas_est)

        # ---------- 1) Try 1-hop ----------
        for f in fees:
            try:
                pool = factory.functions.getPool(t_in, t_out, f).call()
                _dbg(f"[UniV3] 1hop fee={f} pool={pool}")
                if int(pool, 16) == 0:
                    continue
                # Try V2
                try:
                    out, _, _, gas_est = quoter2.functions.quoteExactInputSingle(
                        (t_in, t_out, f, amount_in, 0)
                    ).call(call_from)
                    _dbg(f"[UniV3] 1hop V2 fee={f} out={int(out)} gasEst={int(gas_est)}")
                except Exception as e_v2:
                    _dbg(f"[UniV3] 1hop V2 fee={f} error: {e_v2!r}")
                    if not quoter1:
                        continue
                    try:
                        out = quoter1.functions.quoteExactInputSingle(t_in, t_out, f, amount_in, 0).call(call_from)
                        gas_est = 0
                        _dbg(f"[UniV3] 1hop V1 fee={f} out={int(out)}")
                    except Exception as e_v1:
                        _dbg(f"[UniV3] 1hop V1 fee={f} error: {e_v1!r}")
                        continue

                out = int(out)
                if out <= 0:
                    continue
                if (best is None) or (out > best[2]):
                    best = ("1hop", [f], out, int(gas_est))
            except Exception as e:
                _dbg(f"[UniV3] 1hop fee={f} probe error: {e!r}")

        # ---------- 2) Try 2-hop via WETH ----------
        if best is None:
            WETH = Web3.to_checksum_address(conf["WETH"])
            # in->WETH and WETH->out must both exist (any fee combo)
            hop_cands: List[Tuple[int,int]] = []
            for f1 in fees:
                try:
                    p1 = factory.functions.getPool(t_in, WETH, f1).call()
                    if int(p1,16) == 0: 
                        continue
                    for f2 in fees:
                        try:
                            p2 = factory.functions.getPool(WETH, t_out, f2).call()
                            if int(p2,16) == 0:
                                continue
                            hop_cands.append((f1,f2))
                        except Exception:
                            pass
                except Exception:
                    pass

            for (f1,f2) in hop_cands:
                path = _encode_path([(t_in, f1), (WETH, f2), (t_out, 0)])
                try:
                    out, _, _, gas_est = quoter2.functions.quoteExactInput(path, amount_in).call(call_from)
                    _dbg(f"[UniV3] 2hop fees=({f1},{f2}) out={int(out)} gasEst={int(gas_est)}")
                except Exception as e_v2:
                    _dbg(f"[UniV3] 2hop V2 fees=({f1},{f2}) error: {e_v2!r}")
                    # There is no multi-hop in V1; skip
                    continue
                out = int(out)
                if out <= 0:
                    continue
                if (best is None) or (out > best[2]):
                    best = ("2hop", [f1, f2], out, int(gas_est))

        if best is None:
            return {"__error__": "UniswapV3: no viable pool (direct or via WETH)"}

        # ---------- Build tx on Router02 ----------
        mode, fees_sel, best_out, _ = best
        out_min = best_out * (10_000 - int(slippage_bps)) // 10_000
        deadline = int(time.time()) + 10 * 60

        router_addr = Web3.to_checksum_address(conf["SWAP_ROUTER"])
        if mode == "1hop":
            f = fees_sel[0]
            fn = router.functions.exactInputSingle((
                t_in, t_out, int(f), to_addr, deadline, int(amount_in), int(out_min), 0
            ))
        else:
            f1, f2 = fees_sel
            path = _encode_path([(t_in, f1), (Web3.to_checksum_address(conf["WETH"]), f2), (t_out, 0)])
            fn = router.functions.exactInput((path, to_addr, deadline, int(amount_in), int(out_min)))

        data = fn._encode_transaction_data()
        # Estimate gas best-effort
        try:
            gas_est = int(w3.eth.estimate_gas({"from": to_addr, "to": router_addr, "data": data, "value": 0}) * 1.2)
        except Exception:
            gas_est = 400_000

        return {
            "aggregator": "UniswapV3",
            "allowanceTarget": router_addr,
            "buyAmount": str(int(best_out)),
            "tx": {"to": router_addr, "data": data, "value": 0, "gas": gas_est},
        }
