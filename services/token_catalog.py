from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence


DEFAULT_CORE_TOKENS: Dict[str, Dict[str, str]] = {
    "ethereum": {
        "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
        "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
        "LINK": "0x514910771AF9Ca656af840dff83E8264EcF986CA",
        "LDO": "0x5A98FcBEA516a30Ff331B9858E181E211cD5De0",
        "UNI": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
        "MKR": "0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2",
    },
    "arbitrum": {
        "USDC": "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8",
        "USDC.E": "0xaf88d065e77c8cc2239327c5edb3a432268e5831",
        "USDT": "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9",
        "ARB": "0x912CE59144191C1204E64559FE8253a0e49E6548",
        "GMX": "0xfc5A1A6EB076aF92fFA5E8dF0aBcf320B5aC539D",
        "RDNT": "0x3082CC23568Ea640225c2467653dB90e9250AaA0",
        "LINK": "0xf97f4df75117a78c1a5a0dbb814af92458539fb4",
    },
    "optimism": {
        "USDC": "0x7F5c764cBc14f9669B88837ca1490cCa17c31607",
        "USDT": "0x94b008aa00579c1307b0ef2c499ad98a8ce58e58",
        "DAI": "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1",
        "OP": "0x4200000000000000000000000000000000000042",
        "LINK": "0x350a791bfc2c21f9ed5d10980dad2e2638ffa7f6",
        "WBTC": "0x68f180fcce6836688e9084f035309e29bf0a2095",
    },
    "base": {
        "USDC": "0x833589fCD6eDb6E08f4c7C32D4f71B54bda02913",
        "DAI": "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb",
        "WETH": "0x4200000000000000000000000000000000000006",
    },
    "polygon": {
        "USDC": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
        "USDT": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",
        "DAI": "0x8f3cf7ad23cd3cadbd9735aff958023239c6a063",
        "WBTC": "0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6",
        "LINK": "0x53E0bca35eC356BD5ddDFebbFC0cA8a3a78AbC5B",
        "AAVE": "0xD6DF932A45C0f255f85145f286eA0B292B21C90B",
    },
    "bsc": {
        "USDC": "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d",
        "USDT": "0x55d398326f99059ff775485246999027b3197955",
        "BUSD": "0xe9e7cea3dedca5984780bafc599bd69add087d56",
        "WBNB": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",
        "CAKE": "0x0e09fabb73bd3ade0a17ecc321fd13a19e81ce82",
    },
    "avalanche": {
        "USDC": "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E",
        "USDT": "0x9702230A8Ea53601f5cD2dc00fDBc13d4dF4A8c7",
        "DAI.E": "0xd586E7F844cEa2F87f50152665BCbc2C279D8d70",
        "WBTC.E": "0x50b7545627a5162F82A992c33b87aDc75187B218",
        "WAVAX": "0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7",
    },
}

# Directories that may contain tokenlists (relative to repo root by default).
# Users can drop JSON tokenlists in any of these folders and they will be picked
# up automatically without additional configuration. The env overrides still
# work for bespoke deployments.
DEFAULT_TOKENLIST_DIRS = (
    "config/tokenlists",
    "data/tokenlists",
    "storage/tokenlists",
)


CHAIN_ID_TO_NAME: Dict[int, str] = {
    1: "ethereum",
    10: "optimism",
    56: "bsc",
    137: "polygon",
    8453: "base",
    42161: "arbitrum",
    43114: "avalanche",
}


def _normalize_addr(addr: str) -> str:
    return addr.strip()


def _merge_maps(base: Dict[str, Dict[str, str]], extra: Mapping[str, Mapping[str, str]]) -> None:
    for chain, tokens in (extra or {}).items():
        chain_l = str(chain).lower().strip()
        if not chain_l:
            continue
        dest = base.setdefault(chain_l, {})
        for symbol, addr in (tokens or {}).items():
            sym = str(symbol).upper().strip()
            raw = str(addr or "").strip()
            if not sym or not raw:
                continue
            dest[sym] = _normalize_addr(raw)


def _merge_tokenlist_entries(base: Dict[str, Dict[str, str]], entries: Sequence[Mapping[str, object]]) -> None:
    for entry in entries:
        chain_id = entry.get("chainId") if isinstance(entry, Mapping) else None
        address = entry.get("address") if isinstance(entry, Mapping) else None
        symbol = entry.get("symbol") if isinstance(entry, Mapping) else None
        if chain_id is None or address is None or symbol is None:
            continue
        try:
            chain = CHAIN_ID_TO_NAME[int(chain_id)]
        except Exception:
            continue
        chain_l = chain.strip().lower()
        if not chain_l:
            continue
        dest = base.setdefault(chain_l, {})
        dest[str(symbol).upper().strip()] = _normalize_addr(str(address))


def _tokenlist_dirs(default: str) -> Iterable[Path]:
    raw = os.getenv("WALLET_CORE_TOKENLIST_DIR", default)
    if not raw:
        return []
    paths = []
    for chunk in str(raw).split(os.pathsep):
        chunk = chunk.strip()
        if not chunk:
            continue
        paths.append(Path(chunk).expanduser())
    return paths


def _merge_tokenlist_files(base: Dict[str, Dict[str, str]], paths: Iterable[Path]) -> None:
    for directory in paths:
        if not directory.exists() or not directory.is_dir():
            continue
        for file_path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(file_path.read_text())
            except Exception:
                continue
            if isinstance(payload, Mapping) and isinstance(payload.get("tokens"), Sequence):
                _merge_tokenlist_entries(base, payload["tokens"])  # type: ignore[arg-type]
            elif isinstance(payload, Mapping):
                _merge_maps(base, payload)  # type: ignore[arg-type]


def load_core_tokens(
    extra_path: str | os.PathLike[str] | None = None,
    *,
    tokenlist_dir: str | os.PathLike[str] | None = None,
) -> Dict[str, Dict[str, str]]:
    """
    Build the core-token map by starting with DEFAULT_CORE_TOKENS and merging:
      • config/wallet_core_tokens.json (if present or overridden via path)
      • WALLET_CORE_TOKENS_JSON env blob (JSON string)
    """
    merged = deepcopy(DEFAULT_CORE_TOKENS)
    path = Path(extra_path or os.getenv("WALLET_CORE_TOKENS_FILE", "config/wallet_core_tokens.json")).expanduser()
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            _merge_maps(merged, payload)
        except Exception:
            pass
    default_dirs = os.pathsep.join(DEFAULT_TOKENLIST_DIRS)
    dir_arg = tokenlist_dir or os.getenv("WALLET_CORE_TOKENLIST_DIR_DEFAULT", default_dirs)
    _merge_tokenlist_files(merged, _tokenlist_dirs(str(dir_arg)))
    env_blob = os.getenv("WALLET_CORE_TOKENS_JSON")
    if env_blob:
        try:
            payload = json.loads(env_blob)
            _merge_maps(merged, payload)
        except Exception:
            pass
    return merged


_CORE_TOKEN_MAP = load_core_tokens()


def get_core_token_map() -> Dict[str, Dict[str, str]]:
    """Return a copy of the merged core-token map so callers can mutate safely."""
    return deepcopy(_CORE_TOKEN_MAP)


def core_tokens_for_chain(chain: str) -> Dict[str, str]:
    return deepcopy(_CORE_TOKEN_MAP.get(chain.lower(), {}))
