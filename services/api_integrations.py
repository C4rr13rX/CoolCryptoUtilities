from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import requests
from django.db import transaction

from securevault.models import SecureSetting
from services.secure_settings import decrypt_secret, encrypt_secret, mask_value

DEFAULT_CATEGORY = "default"
TIMEOUT = 8


@dataclass
class IntegrationSpec:
    name: str
    label: str
    description: str
    url: Optional[str] = None
    tester: Optional[Callable[[str], Dict[str, str]]] = None
    secret: bool = True


def _rpc_payload(method: str) -> Dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": int(time.time()),
        "method": method,
        "params": [],
    }


def _test_alchemy(key: str) -> Dict[str, str]:
    resp = requests.post(
        f"https://eth-mainnet.g.alchemy.com/v2/{key}",
        json=_rpc_payload("eth_blockNumber"),
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if "result" not in data:
        raise ValueError("Missing result in response")
    return {"status": "ok", "detail": "RPC reachable"}


def _test_infura(key: str) -> Dict[str, str]:
    resp = requests.post(
        f"https://mainnet.infura.io/v3/{key}",
        json=_rpc_payload("eth_blockNumber"),
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if "result" not in data:
        raise ValueError("Missing result in response")
    return {"status": "ok", "detail": "RPC reachable"}


def _test_cryptopanic(key: str) -> Dict[str, str]:
    resp = requests.get(
        "https://cryptopanic.com/api/v1/posts/",
        params={"public": "true", "auth_token": key, "kind": "news", "limit": 1},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return {"status": "ok", "detail": "API reachable"}


def _test_ankr(key: str) -> Dict[str, str]:
    resp = requests.post(
        f"https://rpc.ankr.com/multichain/{key}",
        json=_rpc_payload("eth_blockNumber"),
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if "result" not in data:
        raise ValueError("Missing result in response")
    return {"status": "ok", "detail": "RPC reachable"}


def _test_zerox(key: str) -> Dict[str, str]:
    resp = requests.get(
        "https://api.0x.org/swap/v1/prices",
        params={"buyToken": "USDC", "sellToken": "USDT", "buyAmount": "1000000"},
        headers={"0x-api-key": key},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return {"status": "ok", "detail": "Quote fetched"}


def _test_lifi(key: str) -> Dict[str, str]:
    resp = requests.get("https://li.quest/v1/status", headers={"x-api-key": key}, timeout=TIMEOUT)
    resp.raise_for_status()
    return {"status": "ok", "detail": "Status reachable"}


def _test_thegraph(key: str) -> Dict[str, str]:
    query = {"query": "{ indexingStatuses { subgraph } }", "variables": {}}
    resp = requests.post(
        f"https://gateway.thegraph.com/api/{key}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV",
        json=query,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return {"status": "ok", "detail": "Gateway reachable"}


def _test_goplus(key: str) -> Dict[str, str]:
    resp = requests.get(
        "https://api.gopluslabs.io/api/v1/token_security/1",
        params={"contract_addresses": "0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2"},
        headers={"API-KEY": key},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("result"):
        raise ValueError("Empty result")
    return {"status": "ok", "detail": "Security data returned"}


def _test_openai(key: str) -> Dict[str, str]:
    resp = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return {"status": "ok", "detail": "Models listed"}


INTEGRATIONS: Dict[str, IntegrationSpec] = {
    "ALCHEMY_API_KEY": IntegrationSpec(
        name="ALCHEMY_API_KEY",
        label="Alchemy",
        description="Used for RPC + NFT data",
        url="https://dashboard.alchemy.com/",
        tester=_test_alchemy,
    ),
    "INFURA_API_KEY": IntegrationSpec(
        name="INFURA_API_KEY",
        label="Infura",
        description="Fallback RPC provider",
        url="https://app.infura.io/",
        tester=_test_infura,
    ),
    "CRYPTOPANIC_API_KEY": IntegrationSpec(
        name="CRYPTOPANIC_API_KEY",
        label="CryptoPanic",
        description="News aggregator",
        url="https://cryptopanic.com/developers/api/",
        tester=_test_cryptopanic,
    ),
    "ANKR_API_KEY": IntegrationSpec(
        name="ANKR_API_KEY",
        label="Ankr",
        description="Historical data + multichain RPC",
        url="https://www.ankr.com/docs/",
        tester=_test_ankr,
    ),
    "ZEROX_API_KEY": IntegrationSpec(
        name="ZEROX_API_KEY",
        label="0x",
        description="Swap routing quotes",
        url="https://dashboard.0x.org/",
        tester=_test_zerox,
    ),
    "LIFI_API_KEY": IntegrationSpec(
        name="LIFI_API_KEY",
        label="LI.FI",
        description="Bridge + swap orchestration",
        url="https://developers.li.fi/",
        tester=_test_lifi,
    ),
    "THEGRAPH_API_KEY": IntegrationSpec(
        name="THEGRAPH_API_KEY",
        label="The Graph",
        description="Uniswap + other subgraphs",
        url="https://thegraph.com/studio/",
        tester=_test_thegraph,
    ),
    "GOPLUS_APP_KEY": IntegrationSpec(
        name="GOPLUS_APP_KEY",
        label="GoPlus App Key",
        description="Token security checks",
        url="https://gopluslabs.io/",
        tester=_test_goplus,
    ),
    "GOPLUS_APP_SECRET": IntegrationSpec(
        name="GOPLUS_APP_SECRET",
        label="GoPlus Secret",
        description="Partner secret (stored for reference)",
        url="https://gopluslabs.io/",
        tester=None,
    ),
    "OPENAI_API_KEY": IntegrationSpec(
        name="OPENAI_API_KEY",
        label="OpenAI",
        description="Used for Codex + Br∆nD D0z3r AI generation",
        url="https://platform.openai.com/api-keys",
        tester=_test_openai,
    ),
}


def list_integrations(user) -> List[Dict[str, Any]]:
    settings = {
        setting.name: setting
        for setting in SecureSetting.objects.filter(user=user, name__in=INTEGRATIONS.keys(), category=DEFAULT_CATEGORY)
    }
    rows: List[Dict[str, Any]] = []
    for spec in INTEGRATIONS.values():
        setting = settings.get(spec.name)
        preview = mask_value("secret") if setting else ""
        rows.append(
            {
                "name": spec.name,
                "label": spec.label,
                "description": spec.description,
                "url": spec.url,
                "has_value": bool(setting),
                "preview": preview,
                "can_test": bool(spec.tester),
            }
        )
    return rows


def update_integration(user, name: str, value: str | None) -> None:
    if name not in INTEGRATIONS:
        raise ValueError(f"Unknown integration {name}")
    if value is None or not str(value).strip():
        SecureSetting.objects.filter(user=user, name=name, category=DEFAULT_CATEGORY).delete()
        return
    with transaction.atomic():
        setting, _ = SecureSetting.objects.get_or_create(
            user=user,
            name=name,
            category=DEFAULT_CATEGORY,
            defaults={"is_secret": True},
        )
        payload = encrypt_secret(str(value).strip())
        setting.is_secret = True
        setting.value_plain = None
        setting.ciphertext = payload["ciphertext"]
        setting.encapsulated_key = payload["encapsulated_key"]
        setting.nonce = payload["nonce"]
        setting.save()


def test_integration(name: str, value: str) -> Dict[str, str]:
    spec = INTEGRATIONS.get(name)
    if not spec:
        raise ValueError(f"Unknown integration {name}")
    if not spec.tester:
        raise ValueError("Testing not defined for this key")
    return spec.tester(value)


def get_integration_value(user, name: str, *, reveal: bool = False) -> Optional[str]:
    if name not in INTEGRATIONS:
        raise ValueError(f"Unknown integration {name}")
    setting = SecureSetting.objects.filter(user=user, name=name, category=DEFAULT_CATEGORY).first()
    if not setting:
        return None
    if setting.is_secret:
        if not reveal:
            return None
        try:
            return decrypt_secret(setting.encapsulated_key, setting.ciphertext, setting.nonce)
        except Exception:
            return setting.value_plain or ""
    return setting.value_plain or ""
