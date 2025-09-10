from __future__ import annotations
import os
from typing import Any, Dict, Optional
from web3 import Web3

# ---------------------------------------------------------------------
# Canonical per-chain config (addresses are mainnet-accurate)
# - SWAP_ROUTER: Uniswap V3 SwapRouter02
# - QUOTER_V1 : Legacy quoter (simple, view, 5 positional args)
# - QUOTER_V2 : Newer quoter (tuple param)
# - WETH      : Wrapped native
# ---------------------------------------------------------------------
UNI_V3: Dict[str, Dict[str, str]] = {
    "ethereum": {
        "SWAP_ROUTER": "0x68b3465833FB72A70ecDF485E0e4C7bD8665Fc45",  # SwapRouter02
        "QUOTER_V1":   "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
        "QUOTER_V2":   "0x61fFE014bA17989E743c5F6cB21bF9697530B21e",
        "WETH":        "0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2",
    },
    "arbitrum": {
        "SWAP_ROUTER": "0x68b3465833FB72A70ecDF485E0e4C7bD8665Fc45",  # same bytecode on many L2s
        "QUOTER_V1":   "0x0000000000000000000000000000000000000000",  # set real address if you want V1 on Arbitrum
        # "QUOTER_V2": "0x...",  # optional; add when needed
        "WETH":        "0x82aF49447D8a07e3bd95BDdB56f35241523fBab1",
    },
}

# ---------------------------------------------------------------------
# Minimal ABIs
# ---------------------------------------------------------------------

# QuoterV1:
# quoteExactInputSingle(address tokenIn, address tokenOut, uint24 fee, uint256 amountIn, uint160 sqrtPriceLimitX96)
#   -> (uint256 amountOut)
_ABI_QUOTER_V1 = [{
    "inputs": [
        {"internalType":"address","name":"tokenIn","type":"address"},
        {"internalType":"address","name":"tokenOut","type":"address"},
        {"internalType":"uint24","name":"fee","type":"uint24"},
        {"internalType":"uint256","name":"amountIn","type":"uint256"},
        {"internalType":"uint160","name":"sqrtPriceLimitX96","type":"uint160"}
    ],
    "name":"quoteExactInputSingle",
    "outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],
    "stateMutability":"view",
    "type":"function"
}]

# QuoterV2:
# quoteExactInputSingle((address tokenIn,address tokenOut,uint24 fee,uint256 amountIn,uint160 sqrtPriceLimitX96) params)
#   -> (uint256 amountOut, uint160[] sqrtPriceX96AfterList, uint32[] initializedTicksCrossedList, uint256 gasEstimate)
_ABI_QUOTER_V2 = [{
    "inputs":[
        {"components":[
            {"internalType":"address","name":"tokenIn","type":"address"},
            {"internalType":"address","name":"tokenOut","type":"address"},
            {"internalType":"uint24","name":"fee","type":"uint24"},
            {"internalType":"uint256","name":"amountIn","type":"uint256"},
            {"internalType":"uint160","name":"sqrtPriceLimitX96","type":"uint160"}
        ],"internalType":"struct IQuoterV2.QuoteExactInputSingleParams","name":"params","type":"tuple"}
    ],
    "name":"quoteExactInputSingle",
    "outputs":[
        {"internalType":"uint256","name":"amountOut","type":"uint256"},
        {"internalType":"uint160[]","name":"sqrtPriceX96AfterList","type":"uint160[]"},
        {"internalType":"uint32[]","name":"initializedTicksCrossedList","type":"uint32[]"},
        {"internalType":"uint256","name":"gasEstimate","type":"uint256"}
    ],
    "stateMutability":"nonpayable",
    "type":"function"
}]

# SwapRouter02:
# exactInputSingle((address tokenIn,address tokenOut,uint24 fee,address recipient,uint256 amountIn,uint256 amountOutMinimum,uint160 sqrtPriceLimitX96))
#   -> (uint256 amountOut)
_ABI_ROUTER02 = [{
    "inputs":[{"components":[
        {"internalType":"address","name":"tokenIn","type":"address"},
        {"internalType":"address","name":"tokenOut","type":"address"},
        {"internalType":"uint24","name":"fee","type":"uint24"},
        {"internalType":"address","name":"recipient","type":"address"},
        {"internalType":"uint256","name":"amountIn","type":"uint256"},
        {"internalType":"uint256","name":"amountOutMinimum","type":"uint256"},
        {"internalType":"uint160","name":"sqrtPriceLimitX96","type":"uint160"}
    ],"internalType":"struct ISwapRouter.ExactInputSingleParams","name":"params","type":"tuple"}],
    "name":"exactInputSingle",
    "outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],
    "stateMutability":"payable",
    "type":"function"
}]

# ---------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------
class UniswapV3Local:
    """
    Local (keyless) Uniswap V3 provider that:
      - quotes using QuoterV1 (preferred) and optionally QuoterV2,
      - builds a SwapRouter02 exactInputSingle transaction,
      - does NOT require a default account if 'recipient' is provided.
    """

    def __init__(self, w3_provider_callable):
        # callable like: lambda ch: bridge._rb__w3(ch)
        self._w3 = w3_provider_callable

    # -------------------------- utils --------------------------
    def _dbg(self, *a):
        if os.getenv("DEBUG_SWAP","0").lower() in ("1","true","yes"):
            print(*a)

    def _cfg(self, chain: str) -> Dict[str, str]:
        ch = chain.lower().strip()
        if ch not in UNI_V3:
            raise ValueError(f"UniswapV3 unsupported on {chain}")
        return UNI_V3[ch]

    @staticmethod
    def _is_native_sentinel(t: str) -> bool:
        s = (t or "").strip().lower()
        return s in ("eth","native","0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")

    @staticmethod
    def _norm_addr(x: str) -> str:
        return Web3.to_checksum_address(x)

    def _router(self, w3: Web3, conf: Dict[str, str]):
        return w3.eth.contract(self._norm_addr(conf["SWAP_ROUTER"]), abi=_ABI_ROUTER02)

    def _qv1(self, w3: Web3, conf: Dict[str, str]):
        addr = conf.get("QUOTER_V1")
        if not addr or int(addr, 16) == 0:
            return None
        return w3.eth.contract(self._norm_addr(addr), abi=_ABI_QUOTER_V1)

    def _qv2(self, w3: Web3, conf: Dict[str, str]):
        addr = conf.get("QUOTER_V2")
        if not addr or int(addr, 16) == 0:
            return None
        return w3.eth.contract(self._norm_addr(addr), abi=_ABI_QUOTER_V2)

    # -------------------- main entrypoint ----------------------
    def quote_and_build(
        self,
        chain: str,
        token_in: str,
        token_out: str,
        amount_in: int,
        *,
        slippage_bps: int = 100,
        recipient: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
          {
            'aggregator': 'UniswapV3',
            'allowanceTarget': <router>,
            'tx': { 'to', 'data', 'value', 'gas'? },
            'buyAmount': str(best_out),
            'fee': <best_fee_bps>
          }
        """
        w3: Web3 = self._w3(chain)
        conf = self._cfg(chain)

        weth = self._norm_addr(conf["WETH"])
        t_in  = weth if self._is_native_sentinel(token_in)  else self._norm_addr(token_in)
        t_out = weth if self._is_native_sentinel(token_out) else self._norm_addr(token_out)

        if int(amount_in) <= 0:
            return {"__error__": "UniswapV3: amount_in must be > 0"}
        if t_in == t_out:
            return {"__error__": "UniswapV3: token_in == token_out"}

        # -------------------- quote best direct fee tier --------------------
        fees = (500, 3000, 10000)
        best_out, best_fee = 0, 0

        q1 = self._qv1(w3, conf)
        if q1 is not None:
            for f in fees:
                try:
                    # V1 expects **five** positional args
                    out = q1.functions.quoteExactInputSingle(t_in, t_out, int(f), int(amount_in), 0).call()
                    if int(out) > best_out:
                        best_out, best_fee = int(out), int(f)
                    self._dbg(f"[UniV3] V1 fee={f} amountOut={int(out)}")
                except Exception as e:
                    self._dbg(f"[UniV3] V1 fee={f} error: {e!r}")

        if best_out <= 0:
            q2 = self._qv2(w3, conf)
            if q2 is not None:
                for f in fees:
                    try:
                        # V2 expects a **single tuple** param (NOT multiple args)
                        out, *_ = q2.functions.quoteExactInputSingle((t_in, t_out, int(f), int(amount_in), 0)).call()
                        if int(out) > best_out:
                            best_out, best_fee = int(out), int(f)
                        self._dbg(f"[UniV3] V2 fee={f} amountOut={int(out)}")
                    except Exception as e:
                        self._dbg(f"[UniV3] V2 fee={f} error: {e!r}")

        if best_out <= 0:
            return {"__error__": "UniswapV3: no viable pool (direct)"}

        # -------------------- build router tx --------------------
        min_out = max(1, best_out * (10_000 - int(slippage_bps)) // 10_000)

        # Decide recipient:
        # - prefer explicit kwarg
        # - else use whatever default_account may be set on this provider
        recp = recipient or w3.eth.default_account
        if not recp:
            return {"__error__": "UniswapV3: no recipient (set w3.eth.default_account or pass recipient=)"}
        recp = self._norm_addr(recp)

        router = self._router(w3, conf)
        params = (
            t_in,
            t_out,
            int(best_fee),
            recp,
            int(amount_in),
            int(min_out),
            0,  # sqrtPriceLimitX96
        )

        # exactInputSingle takes **one** tuple param
        fn = router.functions.exactInputSingle(params)
        data = fn._encode_transaction_data()
        to = self._norm_addr(conf["SWAP_ROUTER"])

        # Gas estimate is nice-to-have; never fatal
        gas = None
        try:
            gas = int(w3.eth.estimate_gas({"from": recp, "to": to, "data": data, "value": 0}) * 1.2)
        except Exception as e:
            self._dbg(f"[UniV3] gas estimate error: {e!r}")
            gas = None

        return {
            "aggregator": "UniswapV3",
            "allowanceTarget": to,
            "tx": {"to": to, "data": data, "value": 0, **({"gas": int(gas)} if gas else {})},
            "buyAmount": str(best_out),
            "fee": best_fee,
        }
