from __future__ import annotations

from pathlib import Path

from services.token_safety import TokenSafetyRegistry
from services.wallet_state import _filter_scams


def _registry(tmp_path: Path) -> TokenSafetyRegistry:
    return TokenSafetyRegistry(ttl_sec=60, cache_path=tmp_path / "token_safety.json")


def test_token_safety_verdict_includes_severity(tmp_path) -> None:
    reg = _registry(tmp_path)
    spec = "ethereum:0x0000000000000000000000000000000000000001"
    reg._cache_result(spec, "flagged", ["is_honeypot"])  # type: ignore[attr-defined]
    verdict = reg.verdict_for("ethereum", "0x0000000000000000000000000000000000000001")
    assert verdict["severity"] == "critical"
    assert verdict["reasons"] == ["is_honeypot"]


def test_filter_scams_returns_flagged_metadata(tmp_path) -> None:
    reg = _registry(tmp_path)
    token = "0x0000000000000000000000000000000000000002"
    reg._cache_result(f"ethereum:{token}", "flagged", ["high_tax"])  # type: ignore[attr-defined]
    rows = [{"chain": "ethereum", "token": token, "usd": 1}]
    survivors, flagged = _filter_scams(rows, registry=reg)
    assert survivors == []
    assert flagged and flagged[0]["severity"] == "medium"
