from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, MutableMapping, Optional

import certifi
import requests


class CovalentError(RuntimeError):
    """Raised when the Covalent API rejects a call or returns malformed data."""


@dataclass
class _ChainConfig:
    slug: str
    chain_id: int


def _default_chain_map() -> Dict[str, _ChainConfig]:
    return {
        "ethereum": _ChainConfig("1", 1),
        "base": _ChainConfig("8453", 8453),
        "arbitrum": _ChainConfig("42161", 42161),
        "optimism": _ChainConfig("10", 10),
        "polygon": _ChainConfig("137", 137),
        "bsc": _ChainConfig("56", 56),
        "avalanche": _ChainConfig("43114", 43114),
    }


class CovalentClient:
    """Thin helper around the Covalent HTTP API used for balances/transfers."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        session: Optional[requests.Session] = None,
        chain_map: Optional[Dict[str, _ChainConfig]] = None,
    ) -> None:
        if not api_key:
            raise ValueError("CovalentClient requires an API key")
        self.api_key = api_key.strip()
        self.base_url = (base_url or os.getenv("COVALENT_BASE_URL", "https://api.covalenthq.com/v1")).rstrip("/")
        self.timeout = max(3, int(timeout or int(os.getenv("COVALENT_TIMEOUT_SEC", "15") or "15")))
        self.session = session or requests.Session()
        self.session.headers.update({"accept": "application/json"})
        self.chain_map = chain_map or _default_chain_map()

    # ------------------------------------------------------------------ helpers
    @classmethod
    def from_env(cls) -> Optional["CovalentClient"]:
        key = os.getenv("COVALENT_KEY", "").strip()
        if not key:
            return None
        return cls(key)

    def _chain_slug(self, chain: str) -> _ChainConfig:
        try:
            return self.chain_map[chain.lower().strip()]
        except KeyError as exc:  # pragma: no cover - configuration error
            raise CovalentError(f"unsupported chain '{chain}' for Covalent") from exc

    def _request(self, path: str, params: Optional[MutableMapping[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        query: Dict[str, Any] = {"key": self.api_key}
        if params:
            query.update(params)
        resp = self.session.get(url, params=query, timeout=self.timeout, verify=certifi.where())
        try:
            payload = resp.json()
        except ValueError as exc:  # pragma: no cover - safety
            raise CovalentError(f"invalid json from covalent ({exc})") from exc
        if resp.status_code >= 400 or payload.get("error"):
            msg = payload.get("error_message") or payload.get("error") or resp.text
            raise CovalentError(f"Covalent request failed ({msg})")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise CovalentError("Covalent response missing data")
        return data

    # ------------------------------------------------------------------ public
    def fetch_balances(self, chain: str, wallet: str, *, include_spam: bool = False) -> List[Dict[str, Any]]:
        cfg = self._chain_slug(chain)
        params = {
            "quote-currency": os.getenv("COVALENT_QUOTE_CURRENCY", "USD"),
            "nft": "false",
            "no-nft-fetch": "true",
        }
        if include_spam:
            params["match"] = ""
        data = self._request(f"{cfg.slug}/address/{wallet}/balances_v2/", params=params)
        items = data.get("items") or []
        if not isinstance(items, list):
            return []
        return [dict(item) for item in items]

    def fetch_transfers(
        self,
        chain: str,
        wallet: str,
        *,
        page_size: int = 100,
        max_pages: int = 2,
    ) -> List[Dict[str, Any]]:
        cfg = self._chain_slug(chain)
        rows: List[Dict[str, Any]] = []
        page = 0
        while page < max_pages:
            params = {
                "page-number": page,
                "page-size": page_size,
            }
            data = self._request(f"{cfg.slug}/address/{wallet}/transfers_v2/", params=params)
            items = data.get("items") or []
            if not isinstance(items, list):
                break
            for item in items:
                token_addr = (item or {}).get("contract_address")
                transfers = (item or {}).get("transfers") or []
                for tr in transfers:
                    if not isinstance(tr, dict):
                        continue
                    rows.append(
                        {
                            "hash": tr.get("tx_hash"),
                            "logIndex": tr.get("log_index") or tr.get("log_offset"),
                            "blockNumber": tr.get("block_height"),
                            "timestamp": tr.get("block_signed_at"),
                            "from": tr.get("from_address"),
                            "to": tr.get("to_address"),
                            "rawContract": {"address": token_addr},
                            "value": tr.get("delta"),
                        }
                    )
            pagination = data.get("pagination") or {}
            has_more = bool(pagination.get("has_more"))
            if not has_more:
                break
            next_page = pagination.get("page_number")
            if isinstance(next_page, int):
                page = next_page + 1
            else:
                page += 1
        return rows


__all__ = ["CovalentClient", "CovalentError"]
