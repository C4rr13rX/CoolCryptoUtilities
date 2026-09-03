"""A token with no direct USDC pool is still buyable one hop away.

Measured 2026-09-03 against the Base QuoterV2: of the 21 ATF candidates whose
swap-quote probe was failing, EVERY failure came back as
``UniswapV3: no viable pool (direct)``. The provider only ever asked for a
USDC->token pool, and base tokens are overwhelmingly paired against WETH. With
0x answering ``403 You cannot consume this service`` and Camelot/Sushi
unconfigured on base, that single direct-only quoter WAS the whole router --
so the live lane could not spend a cent on any of those tokens, and the reason
it gave ("no viable pool") was not true of the chain, only of our query.

BSTONK prices at 188.13 tokens for $0.25 through
USDC -(0.01%)- WETH -(1%)- BSTONK, against a feed price of 0.001281 ($0.25 buys
195 gross, less the 1% pool fee and impact). It quotes 0 direct.

These tests pin the fallback without touching the network.
"""
from __future__ import annotations

import pytest

from services.providers.uniswap_v3 import UniswapV3Local


_USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
_WETH = "0x4200000000000000000000000000000000000006"
_BSTONK = "0x0F61Edbfe6Cd86024C0f210c0695B08df55fdfc9"
_CBBTC = "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf"
_RECIPIENT = "0x291c854811e92906a658Fb94Aa511bF919f968ad"


class _Fn:
    def __init__(self, payload: str):
        self._payload = payload

    def _encode_transaction_data(self) -> str:
        return self._payload


class _FakeRouter:
    """Records which router entrypoint the built tx used."""

    def __init__(self, sink: dict):
        self._sink = sink
        self.functions = self

    def exactInputSingle(self, params):
        self._sink["call"] = ("exactInputSingle", params)
        return _Fn("0x414bf389")

    def exactInput(self, params):
        self._sink["call"] = ("exactInput", params)
        return _Fn("0xb858183f")


class _FakeQuoterV2:
    """Quotes only the routes named in ``priced`` (raw base units)."""

    def __init__(self, priced: dict, calls: list):
        self._priced = priced
        self._calls = calls
        self.functions = self

    def quoteExactInputSingle(self, params):
        self._calls.append(("single", params[0], params[1], params[2]))
        return self._Call(self._priced.get(("direct", params[2]), 0), tuple_out=True)

    def quoteExactInput(self, path, amount_in):
        self._calls.append(("path", bytes(path), int(amount_in)))
        return self._Call(self._priced.get(bytes(path), 0), tuple_out=True)

    class _Call:
        def __init__(self, out: int, tuple_out: bool):
            self._out = out
            self._tuple = tuple_out

        def call(self):
            if self._out <= 0:
                raise RuntimeError("execution reverted")
            return (self._out, 0, 0, 0) if self._tuple else self._out


class _FakeW3:
    class _Eth:
        default_account = None

        @staticmethod
        def estimate_gas(_tx):
            raise RuntimeError("no estimate in tests")

    eth = _Eth()


def _provider(monkeypatch, priced: dict, calls: list, sink: dict) -> UniswapV3Local:
    uni = UniswapV3Local(lambda _ch: _FakeW3())
    monkeypatch.setattr(uni, "_qv1", lambda _w3, _conf: None)
    monkeypatch.setattr(uni, "_qv2", lambda _w3, _conf: _FakeQuoterV2(priced, calls))
    monkeypatch.setattr(uni, "_router", lambda _w3, _conf: _FakeRouter(sink))
    return uni


def _path(tokens, fees) -> bytes:
    return UniswapV3Local._v3_path_multi(
        [UniswapV3Local._norm_addr(t) for t in tokens], list(fees)
    )


def test_path_encoding_is_20_3_20_3_20():
    """QuoterV2 and SwapRouter02 both read addresses split by 3-byte fees."""
    blob = _path([_USDC, _WETH, _BSTONK], [100, 10000])
    assert len(blob) == 20 + 3 + 20 + 3 + 20 == 66
    assert blob[:20].hex() == _USDC[2:].lower()
    assert int.from_bytes(blob[20:23], "big") == 100
    assert blob[23:43].hex() == _WETH[2:].lower()
    assert int.from_bytes(blob[43:46], "big") == 10000
    assert blob[46:].hex() == _BSTONK[2:].lower()


def test_path_encoding_rejects_mismatched_fee_count():
    with pytest.raises(ValueError):
        UniswapV3Local._v3_path_multi([_USDC, _WETH, _BSTONK], [100])


def test_two_hop_quote_when_no_direct_pool_exists(monkeypatch):
    """The BSTONK case: nothing direct, priced through WETH."""
    hop = _path([_USDC, _WETH, _BSTONK], [100, 10000])
    calls: list = []
    sink: dict = {}
    uni = _provider(monkeypatch, {hop: 188132424470836079300}, calls, sink)

    q = uni.quote_and_build(
        "base", _USDC, _BSTONK, 250_000, slippage_bps=100, recipient=_RECIPIENT
    )

    assert "__error__" not in q, q
    assert q["hops"] == 2
    assert q["route"] == [
        UniswapV3Local._norm_addr(_USDC),
        UniswapV3Local._norm_addr(_WETH),
        UniswapV3Local._norm_addr(_BSTONK),
    ]
    # raw base units, str -- the contract every caller of buyAmount reads
    assert q["buyAmount"] == "188132424470836079300"
    assert isinstance(q["buyAmount"], str)
    # a multi-hop fill MUST go through exactInput, not exactInputSingle:
    # exactInputSingle cannot express the intermediate pool.
    assert sink["call"][0] == "exactInput"
    path, recipient, amount_in, min_out = sink["call"][1]
    assert path == hop
    assert recipient == UniswapV3Local._norm_addr(_RECIPIENT)
    assert amount_in == 250_000
    # 100 bps of slippage off the quote
    assert min_out == 188132424470836079300 * 9900 // 10_000


def test_direct_pool_still_uses_exact_input_single(monkeypatch):
    """The cheap path is unchanged: one pool, one hop, exactInputSingle."""
    calls: list = []
    sink: dict = {}
    uni = _provider(monkeypatch, {("direct", 100): 323}, calls, sink)

    q = uni.quote_and_build(
        "base", _USDC, _CBBTC, 250_000, slippage_bps=100, recipient=_RECIPIENT
    )

    assert q["hops"] == 1
    assert q["buyAmount"] == "323"
    assert sink["call"][0] == "exactInputSingle"
    # no path quoting was needed once a direct pool priced
    assert not any(kind == "path" and len(blob) > 43 for kind, blob, *_ in
                   [(c[0], c[1] if isinstance(c[1], bytes) else b"") for c in calls])


def test_unroutable_token_says_both_attempts_failed(monkeypatch):
    """OPENAI has no v3 pool at all -- the refusal must not read as direct-only."""
    uni = _provider(monkeypatch, {}, [], {})

    q = uni.quote_and_build(
        "base", _USDC, _BSTONK, 250_000, slippage_bps=100, recipient=_RECIPIENT
    )

    assert "__error__" in q
    assert "2-hop" in q["__error__"]


def test_multihop_can_be_switched_off(monkeypatch):
    """UNIV3_MULTIHOP=0 restores the previous direct-only behaviour."""
    monkeypatch.setenv("UNIV3_MULTIHOP", "0")
    hop = _path([_USDC, _WETH, _BSTONK], [100, 10000])
    calls: list = []
    uni = _provider(monkeypatch, {hop: 188132424470836079300}, calls, {})

    q = uni.quote_and_build(
        "base", _USDC, _BSTONK, 250_000, slippage_bps=100, recipient=_RECIPIENT
    )

    assert "__error__" in q
    # the 2-hop path was never even quoted
    assert not [c for c in calls if c[0] == "path" and len(c[1]) == 66]


def test_intermediates_never_include_the_endpoints(monkeypatch):
    """Buying WETH itself must not try USDC -> WETH -> WETH."""
    uni = UniswapV3Local(lambda _ch: _FakeW3())
    mids = uni._mid_tokens(
        "base", UniswapV3Local._norm_addr(_USDC), UniswapV3Local._norm_addr(_WETH)
    )
    assert UniswapV3Local._norm_addr(_WETH) not in mids
    assert UniswapV3Local._norm_addr(_USDC) not in mids
    assert UniswapV3Local._norm_addr(_CBBTC) in mids


def test_intermediates_are_overridable_per_chain(monkeypatch):
    monkeypatch.setenv("UNIV3_MID_TOKENS_BASE", f"{_CBBTC}, {_WETH}")
    uni = UniswapV3Local(lambda _ch: _FakeW3())
    mids = uni._mid_tokens(
        "base", UniswapV3Local._norm_addr(_USDC), UniswapV3Local._norm_addr(_BSTONK)
    )
    assert mids == [UniswapV3Local._norm_addr(_CBBTC), UniswapV3Local._norm_addr(_WETH)]
