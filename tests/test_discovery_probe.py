from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")

import django  # noqa: E402

django.setup()

from services.discovery.coordinator import DiscoveryCoordinator
from services.discovery.trending_fetcher import TrendingToken


class _ProbeStub:
    objects: "_ProbeManager"

    def __init__(self, *, symbol: str, chain: str, defaults: Dict[str, object]) -> None:
        self.symbol = symbol
        self.chain = chain
        self.success = defaults.get("success", False)
        self.failure_reason = defaults.get("failure_reason")
        self.metadata = dict(defaults.get("metadata", {}))
        self.created_at = datetime.now(timezone.utc)

    def save(self, update_fields: Tuple[str, ...] | None = None) -> None:
        return


class _ProbeManager:
    def __init__(self) -> None:
        self._instances: Dict[Tuple[str, str], _ProbeStub] = {}

    def get_or_create(self, *, symbol: str, chain: str, defaults: Dict[str, object]) -> Tuple[_ProbeStub, bool]:
        key = (symbol, chain)
        if key in self._instances:
            return self._instances[key], False
        probe = _ProbeStub(symbol=symbol, chain=chain, defaults=defaults)
        self._instances[key] = probe
        return probe, True


_ProbeStub.objects = _ProbeManager()


@dataclass
class _TokenStub:
    symbol: str
    chain: str


def _make_trending(*, liquidity: float, volume: float) -> TrendingToken:
    return TrendingToken(
        symbol="AAA-USDC",
        chain="base",
        pair_address="0xabc",
        dex_id="dex",
        price_usd=0.001,
        volume_24h_usd=volume,
        liquidity_usd=liquidity,
        price_change_1h=0.05,
        price_change_6h=0.12,
        price_change_24h=0.32,
        metadata={},
    )


@pytest.fixture(autouse=True)
def patch_swap_probe(monkeypatch: pytest.MonkeyPatch):
    manager = _ProbeManager()
    _ProbeStub.objects = manager
    monkeypatch.setattr("services.discovery.coordinator.SwapProbe", _ProbeStub, raising=True)
    return manager


def test_virtual_probe_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCOVERY_MIN_LIQUIDITY", "20000")
    monkeypatch.setenv("DISCOVERY_MIN_VOLUME", "50000")
    coordinator = DiscoveryCoordinator()
    token = _TokenStub(symbol="AAA-USDC", chain="base")
    trending = _make_trending(liquidity=50000, volume=90000)

    probe = coordinator._run_swap_probe(token, trending)
    assert probe.success is True
    assert probe.failure_reason is None
    assert probe.metadata["mode"] == "virtual"
    assert probe.metadata["liquidity_usd"] == pytest.approx(50000)


def test_virtual_probe_flags_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCOVERY_MIN_LIQUIDITY", "80000")
    monkeypatch.setenv("DISCOVERY_MIN_VOLUME", "120000")
    coordinator = DiscoveryCoordinator()
    token = _TokenStub(symbol="BBB-USDC", chain="base")
    trending = _make_trending(liquidity=25000, volume=60000)

    probe = coordinator._run_swap_probe(token, trending)
    assert probe.success is False
    assert probe.failure_reason.startswith("insufficient_")
    assert "liquidity_gap" in probe.metadata
    assert "volume_gap" in probe.metadata
