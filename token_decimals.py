"""Canonical ERC-20 decimals for well-known tokens.

Alchemy metadata occasionally fails or rate-limits; defaulting a miss to 18
silently deflates 6-decimal stables (USDC/USDT/USDbC) by 10^12 and poisons the
balance cache. Address entries are authoritative and override cached values;
symbol entries are a fallback for bridged/unlisted deployments.

Zero-dependency module so balances.py, cache.py, services and repair scripts
can all import it without side effects.
"""
from __future__ import annotations

from typing import Dict, Optional

# chain -> lowercase contract address -> decimals
KNOWN_TOKEN_DECIMALS: Dict[str, Dict[str, int]] = {
    "ethereum": {
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": 6,   # USDC
        "0xdac17f958d2ee523a2206206994597c13d831ec7": 6,   # USDT
        "0x6c3ea9036406852006290770bedfcaba0e23a0e8": 6,   # PYUSD
        "0x6b175474e89094c44da98b954eedeac495271d0f": 18,  # DAI
        "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": 8,   # WBTC
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": 18,  # WETH
        "0x056fd409e1d7a124bd7017459dfea2f387b6d5cd": 2,   # GUSD
    },
    "base": {
        "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913": 6,   # USDC
        "0xd9aaec86b65d86f6a7b5b1b0c42ffa531710b6ca": 6,   # USDbC
        "0xfde4c96c8593536e31f229ea8f37b2ada2699bb2": 6,   # USDT
        "0x50c5725949a6f0c72e6c4a641f24049a917db0cb": 18,  # DAI
        "0x4200000000000000000000000000000000000006": 18,  # WETH
        "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf": 8,   # cbBTC
    },
    "arbitrum": {
        "0xaf88d065e77c8cc2239327c5edb3a432268e5831": 6,   # USDC (native)
        "0xff970a61a04b1ca14834a43f5de4533ebddb5cc8": 6,   # USDC.e
        "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9": 6,   # USDT
        "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1": 18,  # DAI
        "0x2f2a2543b76a4166549f7aab2e75bef0aefc5b0f": 8,   # WBTC
        "0x82af49447d8a07e3bd95bd0d56f35241523fbab1": 18,  # WETH
    },
    "optimism": {
        "0x0b2c639c533813f4aa9d7837caf62653d097ff85": 6,   # USDC (native)
        "0x7f5c764cbc14f9669b88837ca1490cca17c31607": 6,   # USDC.e
        "0x94b008aa00579c1307b0ef2c499ad98a8ce58e58": 6,   # USDT
        "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1": 18,  # DAI
        "0x68f180fcce6836688e9084f035309e29bf0a2095": 8,   # WBTC
        "0x4200000000000000000000000000000000000006": 18,  # WETH
    },
    "polygon": {
        "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359": 6,   # USDC (native)
        "0x2791bca1f2de4661ed88a30c99a7a9449aa84174": 6,   # USDC.e
        "0xc2132d05d31c914a87c6611c10748aeb04b58e8f": 6,   # USDT
        "0x8f3cf7ad23cd3cadbd9735aff958023239c6a063": 18,  # DAI
        "0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6": 8,   # WBTC
        "0x7ceb23fd6bc0add59e62ac25578270cff1b9f619": 18,  # WETH
        "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270": 18,  # WMATIC/WPOL
    },
}

# Uppercase symbol -> decimals. Only include symbols whose decimals are
# consistent across every major deployment we touch.
KNOWN_SYMBOL_DECIMALS: Dict[str, int] = {
    "USDC": 6,
    "USDC.E": 6,
    "USDBC": 6,
    "USDT": 6,
    "PYUSD": 6,
    "GUSD": 2,
    "WBTC": 8,
    "CBBTC": 8,
    "DAI": 18,
    "WETH": 18,
    "ETH": 18,
    "WMATIC": 18,
    "MATIC": 18,
    "WPOL": 18,
    "FRAX": 18,
    "LUSD": 18,
    "TUSD": 18,
    "BUSD": 18,
    "USDP": 18,
}


def known_token_decimals(
    chain: Optional[str],
    token: Optional[str],
    symbol: Optional[str] = None,
) -> Optional[int]:
    """Return authoritative decimals for a token, or None if unknown.

    Address match (per chain) wins; symbol match is the fallback so bridged
    deployments of the same asset still resolve correctly.
    """
    chain_l = (chain or "").strip().lower()
    token_l = (token or "").strip().lower()
    if chain_l and token_l:
        dec = KNOWN_TOKEN_DECIMALS.get(chain_l, {}).get(token_l)
        if dec is not None:
            return dec
    if symbol:
        sym = str(symbol).strip().upper()
        if sym:
            return KNOWN_SYMBOL_DECIMALS.get(sym)
    return None
