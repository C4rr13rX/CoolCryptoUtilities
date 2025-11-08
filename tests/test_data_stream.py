from __future__ import annotations

from trading.data_stream import _split_symbol


def test_split_symbol_normalizes_usdbc():
    base, quote = _split_symbol("WETH-USDbC")
    assert base == "ETH"
    assert quote == "USDC"
