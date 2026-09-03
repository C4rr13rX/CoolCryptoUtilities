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
    "base": {
        "SWAP_ROUTER": "0x2626664c2603336E57B271c5C0b26F421741e481",  # SwapRouter02 on Base
        "QUOTER_V2":   "0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a",  # QuoterV2 on Base
        "WETH":        "0x4200000000000000000000000000000000000006",
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
#   -> (uint256 amountOut, uint160 sqrtPriceX96After, uint32 initializedTicksCrossed, uint256 gasEstimate)
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
        {"internalType":"uint160","name":"sqrtPriceX96After","type":"uint160"},
        {"internalType":"uint32","name":"initializedTicksCrossed","type":"uint32"},
        {"internalType":"uint256","name":"gasEstimate","type":"uint256"}
    ],
    "stateMutability":"nonpayable",
    "type":"function"
}]

_ABI_QUOTER_V2_EXACT_INPUT = [{
    "inputs":[
        {"internalType":"bytes","name":"path","type":"bytes"},
        {"internalType":"uint256","name":"amountIn","type":"uint256"}
    ],
    "name":"quoteExactInput",
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

# SwapRouter02:
# exactInput((bytes path,address recipient,uint256 amountIn,uint256 amountOutMinimum))
#   -> (uint256 amountOut)
# SwapRouter02 dropped the `deadline` field that SwapRouter01 carried; the
# four-field tuple below is what is deployed at the SWAP_ROUTER addresses above.
_ABI_ROUTER02_EXACT_INPUT = [{
    "inputs":[{"components":[
        {"internalType":"bytes","name":"path","type":"bytes"},
        {"internalType":"address","name":"recipient","type":"address"},
        {"internalType":"uint256","name":"amountIn","type":"uint256"},
        {"internalType":"uint256","name":"amountOutMinimum","type":"uint256"}
    ],"internalType":"struct IV3SwapRouter.ExactInputParams","name":"params","type":"tuple"}],
    "name":"exactInput",
    "outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],
    "stateMutability":"payable",
    "type":"function"
}]

# Tokens a two-hop route may pass THROUGH, in the order they are tried.
#
# Measured 2026-09-03 against the Base QuoterV2: of 21 ATF candidates whose
# swap-quote probe failed, every single failure was reported as "no viable
# pool (direct)" -- this provider only ever asked for a USDC->token pool. Base
# tokens are overwhelmingly paired against WETH, not USDC, so a direct-only
# quoter cannot price them and the live lane could not spend a cent on any of
# them. BSTONK quotes through WETH (187.58 tokens for $0.25, matching the
# 0.001281 feed price) and CBADA through cbBTC; both return 0 direct.
MID_TOKENS: Dict[str, tuple] = {
    "ethereum": (
        "0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2",  # WETH
        "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
    ),
    "base": (
        "0x4200000000000000000000000000000000000006",  # WETH
        "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf",  # cbBTC
    ),
    "arbitrum": (
        "0x82aF49447D8a07e3bd95BDdB56f35241523fBab1",  # WETH
        "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9",  # USDT
    ),
}

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
        return w3.eth.contract(
            self._norm_addr(conf["SWAP_ROUTER"]),
            abi=_ABI_ROUTER02 + _ABI_ROUTER02_EXACT_INPUT,
        )

    def _mid_tokens(self, chain: str, t_in: str, t_out: str) -> list:
        """Intermediates to try for a two-hop route, ends excluded."""
        ch = chain.lower().strip()
        env = os.getenv(f"UNIV3_MID_TOKENS_{ch.upper()}", "").strip()
        raw = [p for p in env.replace(";", ",").split(",") if p.strip()] if env else list(MID_TOKENS.get(ch, ()))
        mids = []
        for addr in raw:
            try:
                a = self._norm_addr(addr.strip())
            except Exception:
                continue
            if a in (t_in, t_out) or a in mids:
                continue
            mids.append(a)
        return mids

    def _qv1(self, w3: Web3, conf: Dict[str, str]):
        addr = conf.get("QUOTER_V1")
        if not addr or int(addr, 16) == 0:
            return None
        return w3.eth.contract(self._norm_addr(addr), abi=_ABI_QUOTER_V1)

    def _qv2(self, w3: Web3, conf: Dict[str, str]):
        addr = conf.get("QUOTER_V2")
        if not addr or int(addr, 16) == 0:
            return None
        return w3.eth.contract(self._norm_addr(addr), abi=_ABI_QUOTER_V2 + _ABI_QUOTER_V2_EXACT_INPUT)

    @staticmethod
    def _v3_path(token_in: str, fee: int, token_out: str) -> bytes:
        return bytes.fromhex(token_in[2:]) + int(fee).to_bytes(3, "big") + bytes.fromhex(token_out[2:])

    @staticmethod
    def _v3_path_multi(tokens: list, fees: list) -> bytes:
        """Encode token0 (fee0) token1 (fee1) token2 ... for exactInput.

        ``len(fees)`` must be ``len(tokens) - 1``; the encoding is 20-byte
        addresses separated by 3-byte fees, which is what both QuoterV2's
        quoteExactInput and SwapRouter02's exactInput consume.
        """
        if len(tokens) < 2 or len(fees) != len(tokens) - 1:
            raise ValueError("v3 path needs len(fees) == len(tokens) - 1")
        blob = bytes.fromhex(tokens[0][2:])
        for fee, token in zip(fees, tokens[1:]):
            blob += int(fee).to_bytes(3, "big") + bytes.fromhex(token[2:])
        return blob

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
        fees = (100, 500, 3000, 10000)
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
                    for f in fees:
                        try:
                            # Some QuoterV2 deployments/RPCs revert on
                            # quoteExactInputSingle but succeed through the
                            # encoded-path quoteExactInput interface.
                            path = self._v3_path(t_in, int(f), t_out)
                            out, *_ = q2.functions.quoteExactInput(path, int(amount_in)).call()
                            if int(out) > best_out:
                                best_out, best_fee = int(out), int(f)
                            self._dbg(f"[UniV3] V2 path fee={f} amountOut={int(out)}")
                        except Exception as e:
                            self._dbg(f"[UniV3] V2 path fee={f} error: {e!r}")

        # -------------------- two-hop fallback --------------------
        # A direct USDC->token pool is the exception on an L2, not the rule.
        # Without this the provider reported "no viable pool" for tokens that
        # are perfectly tradeable one hop away, and since 0x is returning
        # 403 and Camelot/Sushi are unconfigured on base, that verdict was
        # the whole router: the live lane had no way to buy them at all.
        best_path: Optional[bytes] = None
        best_route: list = [t_in, t_out]
        multihop = os.getenv("UNIV3_MULTIHOP", "1").strip().lower() not in {"0", "false", "no", "off"}
        if best_out <= 0 and multihop:
            q2 = self._qv2(w3, conf)
            if q2 is not None:
                for mid in self._mid_tokens(chain, t_in, t_out):
                    for f1 in fees:
                        for f2 in fees:
                            try:
                                path = self._v3_path_multi([t_in, mid, t_out], [f1, f2])
                                out, *_ = q2.functions.quoteExactInput(path, int(amount_in)).call()
                            except Exception as e:
                                self._dbg(f"[UniV3] hop {f1}/{f2} via {mid} error: {e!r}")
                                continue
                            if int(out) > best_out:
                                best_out, best_fee = int(out), int(f1)
                                best_path, best_route = path, [t_in, mid, t_out]
                                self._dbg(f"[UniV3] hop {f1}/{f2} via {mid} amountOut={int(out)}")
                    if best_out > 0:
                        # First intermediate that prices the pair wins; trying
                        # the rest costs 16 more eth_calls for a rounding
                        # difference on a dust-sized clip.
                        break

        if best_out <= 0:
            return {"__error__": "UniswapV3: no viable pool (direct or 2-hop)"}

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
        if best_path is not None:
            # exactInput takes (path, recipient, amountIn, amountOutMinimum)
            fn = router.functions.exactInput((best_path, recp, int(amount_in), int(min_out)))
        else:
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
            # Which pools this quote actually went through. A two-hop fill
            # prices differently from the direct one the caller may assume,
            # and the reconciler has no other way to tell them apart.
            "route": list(best_route),
            "hops": len(best_route) - 1,
        }
