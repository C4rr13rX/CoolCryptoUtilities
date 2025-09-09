from __future__ import annotations
import time
from typing import Any, Dict, Optional
from web3 import Web3

# --- Canonical addresses (mainnet) ---
UNI_V3 = {
    "ethereum": {
        "SWAP_ROUTER": "0x68b3465833FB72A70ecDF485E0e4C7bD8665Fc45",  # SwapRouter02
        "QUOTER_V1":   "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
        "WETH":        "0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2",
    },
    "arbitrum": {
        "SWAP_ROUTER": "0x68b3465833FB72A70ecDF485E0e4C7bD8665Fc45",  # same on many L2s
        "QUOTER_V1":   "0x0000000000000000000000000000000000000000",  # set if you plan to use Arbitrum V1
        "WETH":        "0x82aF49447D8a07e3bd95BDdB56f35241523fBab1",
    },
}

# --- Minimal ABIs (V1 quoter + Router02 exactInputSingle) ---
# QuoterV1: quoteExactInputSingle(address tokenIn, address tokenOut, uint24 fee, uint256 amountIn, uint160 sqrtPriceLimitX96) returns (uint256 amountOut)
_ABI_QUOTER_V1 = [{
    "inputs": [
        {"internalType":"address","name":"tokenIn","type":"address"},
        {"internalType":"address","name":"tokenOut","type":"address"},
        {"internalType":"uint24","name":"fee","type":"uint24"},
        {"internalType":"uint256","name":"amountIn","type":"uint256"},
        {"internalType":"uint160","name":"sqrtPriceLimitX96","type":"uint160"}
    ],
    "name":"quoteExactInputSingle","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],
    "stateMutability":"view","type":"function"
}]

# Router02: exactInputSingle((address,address,uint24,address,uint256,uint256,uint160)) returns (uint256 amountOut)
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
    "stateMutability":"payable","type":"function"
}]

class UniswapV3Local:
    def __init__(self, w3_provider_callable):
        # callable like: lambda ch: bridge._rb__w3(ch)
        self._w3 = w3_provider_callable

    def _cfg(self, chain: str) -> Dict[str,str]:
        ch = chain.lower().strip()
        if ch not in UNI_V3:
            raise ValueError(f"UniswapV3 unsupported on {chain}")
        return UNI_V3[ch]

    def _is_native_sentinel(self, t: str) -> bool:
        s = (t or "").lower()
        return s in ("eth","native","0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")

    def _qv1(self, w3: Web3, conf: Dict[str,str]):
        return w3.eth.contract(Web3.to_checksum_address(conf["QUOTER_V1"]), abi=_ABI_QUOTER_V1)

    def _router(self, w3: Web3, conf: Dict[str,str]):
        return w3.eth.contract(Web3.to_checksum_address(conf["SWAP_ROUTER"]), abi=_ABI_ROUTER02)

    def quote_and_build(self, chain: str, token_in: str, token_out: str, amount_in: int, *, slippage_bps: int=100) -> Dict[str,Any]:
        """
        Returns: { 'aggregator':'UniswapV3', 'allowanceTarget': <router>,
                   'tx': {to,data,value,gas?}, 'buyAmount': str(best_out), 'fee': best_fee }
        """
        w3 = self._w3(chain)
        conf = self._cfg(chain)
        weth = Web3.to_checksum_address(conf["WETH"])
        t_in  = weth if self._is_native_sentinel(token_in)  else Web3.to_checksum_address(token_in)
        t_out = weth if self._is_native_sentinel(token_out) else Web3.to_checksum_address(token_out)

        if t_in == t_out:
            return {"__error__": "UniswapV3: token_in == token_out"}

        # Probe fees via QuoterV1 (500/3000/10000)
        fees = [500, 3000, 10000]
        qv1 = self._qv1(w3, conf)
        best = (0, 0)  # (out, fee)
        for f in fees:
            try:
                out = qv1.functions.quoteExactInputSingle(t_in, t_out, f, int(amount_in), 0).call()
                if int(out) > best[0]:
                    best = (int(out), int(f))
            except Exception:
                pass

        if best[0] <= 0:
            return {"__error__": "UniswapV3: no viable pool (direct)"}

        best_out, best_fee = best
        # amountOutMinimum with slippage
        min_out = max(1, best_out * (10_000 - int(slippage_bps)) // 10_000)

        # Build Router02 exactInputSingle
        router = self._router(w3, conf)
        recipient = w3.eth.default_account or w3.eth.accounts[0] if w3.eth.accounts else None
        if not recipient:
            raise RuntimeError("UniswapV3: no default account set on provider")

        params = (t_in, t_out, int(best_fee), Web3.to_checksum_address(recipient),
                  int(amount_in), int(min_out), 0)  # sqrtPriceLimitX96 = 0 (no limit)

        fn = router.functions.exactInputSingle(params)
        data = fn._encode_transaction_data()

        to = Web3.to_checksum_address(conf["SWAP_ROUTER"])
        # Try estimate gas (helpful for preflight and to surface issues)
        try:
            gas = int(w3.eth.estimate_gas({"from": recipient, "to": to, "data": data, "value": 0}) * 1.2)
        except Exception:
            gas = None

        return {
            "aggregator": "UniswapV3",
            "allowanceTarget": to,
            "tx": {"to": to, "data": data, "value": 0, **({"gas": gas} if gas else {})},
            "buyAmount": str(best_out),
            "fee": best_fee,
        }

# ======================================================================
# ## UNI_V3__FINAL_CLEAN_OVERRIDE (recipient-aware; no default-account requirement)
try:
    _ABI_QUOTER_V1
except NameError:
    _ABI_QUOTER_V1 = [{
        "inputs":[
            {"internalType":"address","name":"tokenIn","type":"address"},
            {"internalType":"address","name":"tokenOut","type":"address"},
            {"internalType":"uint24","name":"fee","type":"uint24"},
            {"internalType":"uint256","name":"amountIn","type":"uint256"},
            {"internalType":"uint160","name":"sqrtPriceLimitX96","type":"uint160"}
        ],
        "name":"quoteExactInputSingle",
        "outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],
        "stateMutability":"view","type":"function"
    }]

try:
    _ABI_UNI_QUOTER_V2
except NameError:
    _ABI_UNI_QUOTER_V2 = [{
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
        "stateMutability":"nonpayable","type":"function"
    },
    {
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
        "stateMutability":"nonpayable","type":"function"
    }]

try:
    _ABI_ROUTER02
except NameError:
    try:
        _ABI_ROUTER02 = _ABI_UNI_ROUTER02
    except NameError:
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
            "stateMutability":"payable","type":"function"
        }]

def _v3_path(tokens, fees):
    if len(tokens) != len(fees) + 1:
        raise ValueError("V3 path: tokens must be n+1 for n fees")
    out = []
    for i, fee in enumerate(fees):
        out.append(tokens[i].lower().replace("0x","").rjust(40,"0"))
        out.append(f"{int(fee):06x}")
    out.append(tokens[-1].lower().replace("0x","").rjust(40,"0"))
    return "0x" + "".join(out)

def _univ3_quote_and_build_final(self, chain: str, token_in: str, token_out: str, amount_in: int, *, slippage_bps: int=100, recipient: Optional[str]=None) -> Dict[str,Any]:
    from web3 import Web3
    import time

    ch = chain.lower().strip()
    # tolerate both old/new config keys
    try:
        conf = self._cfg_norm(ch)  # has SWAP_ROUTER, QUOTER_V2, WETH
    except Exception:
        # fallback to older _cfg()
        conf = getattr(self, "_cfg")(ch)

    w3 = self._w3(ch)
    router = w3.eth.contract(Web3.to_checksum_address(conf.get("SWAP_ROUTER") or conf.get("ROUTER")), abi=_ABI_ROUTER02)

    # recp: prefer caller-supplied, otherwise any default; never raise if missing
    recp = Web3.to_checksum_address(recipient) if recipient else (w3.eth.default_account or "0x0000000000000000000000000000000000000000")

    weth = Web3.to_checksum_address(conf["WETH"])
    def _norm(x: str) -> str:
        s = (x or "").strip().lower()
        if s in ("eth","native","0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"):
            return weth
        return Web3.to_checksum_address(x)

    t_in  = _norm(token_in)
    t_out = _norm(token_out)
    if int(amount_in) <= 0:
        return {"__error__":"UniswapV3: amount_in must be > 0"}
    if t_in == t_out:
        return {"__error__":"UniswapV3: token_in == token_out"}

    # pick fee via QuoterV1 if present; else try V2
    best_out, best_fee = 0, 0
    fees = (500, 3000, 10000)

    qv1_addr = (UNI_V3.get(ch, {}) or {}).get("QUOTER_V1")
    if qv1_addr:
        qv1 = w3.eth.contract(Web3.to_checksum_address(qv1_addr), abi=_ABI_QUOTER_V1)
        for f in fees:
            try:
                out = qv1.functions.quoteExactInputSingle(t_in, t_out, int(f), int(amount_in), 0).call()
                if int(out) > best_out:
                    best_out, best_fee = int(out), int(f)
            except Exception:
                pass

    if best_out <= 0 and conf.get("QUOTER_V2"):
        qv2 = w3.eth.contract(Web3.to_checksum_address(conf["QUOTER_V2"]), abi=_ABI_UNI_QUOTER_V2)
        for f in fees:
            try:
                out, *_ = qv2.functions.quoteExactInputSingle((t_in, t_out, int(f), int(amount_in), 0)).call()
                if int(out) > best_out:
                    best_out, best_fee = int(out), int(f)
            except Exception:
                pass

    if best_out <= 0:
        return {"__error__":"UniswapV3: no viable pool (direct)"}

    out_min = max(1, best_out * (10_000 - int(slippage_bps)) // 10_000)
    params = (t_in, t_out, int(best_fee), Web3.to_checksum_address(recp), int(amount_in), int(out_min), 0)
    fn = router.functions.exactInputSingle(params)
    data = fn._encode_transaction_data()
    to = Web3.to_checksum_address(conf.get("SWAP_ROUTER") or conf.get("ROUTER"))

    # Estimate gas only if we have a non-zero 'from'; never crash if it fails
    gas = None
    try:
        if recp != "0x0000000000000000000000000000000000000000":
            gas = int(w3.eth.estimate_gas({"from": recp, "to": to, "data": data, "value": 0}) * 1.2)
    except Exception:
        gas = None

    return {
        "aggregator": "UniswapV3",
        "allowanceTarget": to,
        "buyAmount": str(best_out),
        "fee": best_fee,
        "tx": {"to": to, "data": data, "value": 0, **({"gas": int(gas)} if gas else {})},
    }

# Make this the authoritative implementation regardless of earlier class defs
UniswapV3Local.quote_and_build = _univ3_quote_and_build_final  # type: ignore
