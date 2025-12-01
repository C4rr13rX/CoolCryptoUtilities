from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

import aiohttp
from web3 import Web3
from web3.providers.rpc import HTTPProvider

SWAP_TOPIC = Web3.keccak(text="Swap(address,uint256,uint256,uint256,uint256,address)").hex()

ERC20_ABI = [
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "stateMutability": "view", "type": "function"},
    {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "stateMutability": "view", "type": "function"},
]

PAIR_ABI = [
    {"constant": True, "inputs": [], "name": "token0", "outputs": [{"name": "", "type": "address"}], "stateMutability": "view", "type": "function"},
    {"constant": True, "inputs": [], "name": "token1", "outputs": [{"name": "", "type": "address"}], "stateMutability": "view", "type": "function"},
]


def _load_pair_index(chain: str) -> Dict[str, Dict[str, Any]]:
    path = Path("data") / f"pair_index_{chain}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@dataclass
class TokenMetadata:
    address: str
    decimals: int
    symbol: str


class OnChainPairFeed:
    """
    Lightweight on-chain listener that subscribes to Swap events for a UniswapV2-style pair
    and emits price updates through a callback.
    """

    def __init__(self, *, chain: str, symbol: str) -> None:
        self.chain = chain.lower()
        self.symbol = symbol.upper()
        self._lookup_symbol = self._normalize_symbol(self.symbol)
        env_prefix = self.chain.upper()
        self.wss_url = os.getenv(f"{env_prefix}_WSS_URL") or os.getenv("GLOBAL_WSS_URL")
        self.rpc_url = os.getenv(f"{env_prefix}_RPC_URL") or os.getenv("GLOBAL_RPC_URL")
        self._session: Optional[aiohttp.ClientSession] = None
        self._http_web3: Optional[Web3] = None
        self._pair_address: Optional[str] = None
        self._token0: Optional[TokenMetadata] = None
        self._token1: Optional[TokenMetadata] = None
        self._base_is_token0: Optional[bool] = None
        self._subscription_id: Optional[str] = None
        self.available: bool = bool(self.wss_url and self.rpc_url)
        self._stop = asyncio.Event()
        if self.available:
            try:
                self._initialise_metadata()
            except Exception as exc:
                print(f"[onchain-feed] unable to initialise metadata for {self.symbol}: {exc}")
                self.available = False

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        sym = symbol.upper()
        if sym.endswith("USDBC"):
            sym = sym.replace("USDBC", "USDbC")
        return sym

    def _initialise_metadata(self) -> None:
        if not self.rpc_url:
            raise RuntimeError("RPC URL not configured.")
        web3 = Web3(HTTPProvider(self.rpc_url, request_kwargs={"timeout": 10}))
        index = _load_pair_index(self.chain)
        pair_address = None
        symbols_available = []
        target_parts = self._lookup_symbol.split("-") if "-" in self._lookup_symbol else [self._lookup_symbol]
        for addr, info in index.items():
            sym = str(info.get("symbol", "")).upper()
            symbols_available.append(sym)
            if sym == self._lookup_symbol:
                pair_address = Web3.to_checksum_address(addr)
                break
            parts = sym.split("-") if "-" in sym else [sym]
            if len(parts) == len(target_parts) == 2:
                base_match = parts[0] == target_parts[0]
                quote = parts[1]
                normalized_quote = (
                    quote.replace("USDBC", "USDbC")
                    .replace("USDB", "USDbC")
                    .replace("USDC", "USDbC")
                )
                quote_match = normalized_quote == target_parts[1]
                if base_match and quote_match:
                    pair_address = Web3.to_checksum_address(addr)
                    break
        if not pair_address:
            raise RuntimeError(
                f"Pair {self._lookup_symbol} not found in pair index for {self.chain}. "
                f"Available sample: {symbols_available[:5]}"
            )
        contract = web3.eth.contract(address=pair_address, abi=PAIR_ABI)
        token0_addr = contract.functions.token0().call()
        token1_addr = contract.functions.token1().call()
        token0 = self._load_token_metadata(web3, token0_addr)
        token1 = self._load_token_metadata(web3, token1_addr)
        base_symbol, quote_symbol = self._lookup_symbol.split("-")
        base_match_token0 = token0.symbol.upper() == base_symbol.upper()
        base_match_token1 = token1.symbol.upper() == base_symbol.upper()
        if base_match_token0:
            self._base_is_token0 = True
        elif base_match_token1:
            self._base_is_token0 = False
        else:
            # fallback to address heuristics
            self._base_is_token0 = True
        self._pair_address = pair_address
        self._token0 = token0
        self._token1 = token1
        self._http_web3 = web3

    def _load_token_metadata(self, web3: Web3, address: str) -> TokenMetadata:
        contract = web3.eth.contract(address=address, abi=ERC20_ABI)
        try:
            decimals = int(contract.functions.decimals().call())
        except Exception:
            decimals = 18
        try:
            symbol_value = contract.functions.symbol().call()
            if isinstance(symbol_value, bytes):
                symbol = symbol_value.decode(errors="ignore").strip("\x00") or address[:6]
            else:
                symbol = str(symbol_value)
        except Exception:
            symbol = address[:6]
        return TokenMetadata(address=Web3.to_checksum_address(address), decimals=decimals, symbol=symbol)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, emitter: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        if not self.available or not self._pair_address:
            return
        backoff = 5.0
        while not self._stop.is_set():
            try:
                await self._stream_once(emitter)
                backoff = 5.0
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"[onchain-feed] stream error for {self.symbol}: {exc}")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 1.5, 120.0)

    async def stop(self) -> None:
        self._stop.set()
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _stream_once(self, emitter: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        if not self._pair_address or not self.wss_url:
            return
        if self._session is None:
            self._session = aiohttp.ClientSession()
        subscribe_payload = {
            "id": 1,
            "method": "eth_subscribe",
            "params": [
                "logs",
                {
                    "address": self._pair_address,
                    "topics": [SWAP_TOPIC],
                },
            ],
        }
        async with self._session.ws_connect(self.wss_url, heartbeat=30, timeout=30) as ws:
            await ws.send_json(subscribe_payload)
            subscription_id = None
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    payload = json.loads(msg.data)
                    if "id" in payload and payload.get("id") == 1:
                        subscription_id = payload.get("result")
                        self._subscription_id = subscription_id
                        continue
                    if payload.get("method") != "eth_subscription":
                        continue
                    params = payload.get("params") or {}
                    if params.get("subscription") != subscription_id:
                        continue
                    log = params.get("result")
                    sample = self._parse_swap_log(log)
                    if sample:
                        await emitter(sample)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
                if self._stop.is_set():
                    break
            if subscription_id:
                try:
                    await ws.send_json({"id": 99, "method": "eth_unsubscribe", "params": [subscription_id]})
                except Exception:
                    pass

    def _parse_swap_log(self, log: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not log or not self._token0 or not self._token1 or self._base_is_token0 is None:
            return None
        data_hex = log.get("data")
        if not isinstance(data_hex, str) or not data_hex.startswith("0x"):
            return None
        try:
            amounts = self._decode_amounts(data_hex)
        except Exception:
            return None
        amount0_in, amount1_in, amount0_out, amount1_out = amounts
        if self._base_is_token0:
            base_raw = self._choose_nonzero(amount0_in, amount0_out)
            quote_raw = self._choose_nonzero(amount1_out, amount1_in)
            base_decimals = self._token0.decimals
            quote_decimals = self._token1.decimals
        else:
            base_raw = self._choose_nonzero(amount1_in, amount1_out)
            quote_raw = self._choose_nonzero(amount0_out, amount0_in)
            base_decimals = self._token1.decimals
            quote_decimals = self._token0.decimals
        if base_raw <= 0 or quote_raw <= 0:
            return None
        base_amount = base_raw / (10 ** base_decimals)
        quote_amount = quote_raw / (10 ** quote_decimals)
        if base_amount <= 0 or quote_amount <= 0:
            return None
        price = quote_amount / base_amount
        volume = max(quote_amount, base_amount)
        return {
            "symbol": self.symbol,
            "chain": self.chain,
            "price": float(price),
            "volume": float(volume),
            "ts": time.time(),
            "source": "onchain",
            "raw": {
                "log": log,
                "base_amount": base_amount,
                "quote_amount": quote_amount,
            },
        }

    @staticmethod
    def _decode_amounts(data_hex: str) -> tuple[int, int, int, int]:
        payload = data_hex[2:]
        if len(payload) < 256:
            raise ValueError("insufficient data length for swap log")
        chunks = [payload[i : i + 64] for i in range(0, 256, 64)]
        return tuple(int(chunk, 16) for chunk in chunks)  # type: ignore[return-value]

    @staticmethod
    def _choose_nonzero(primary: int, secondary: int) -> int:
        return primary if primary > 0 else secondary
