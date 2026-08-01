import json
import time

from services import wallet_reconciliation


def test_stale_wallet_cache_is_never_reported_as_current_money(monkeypatch) -> None:
    monkeypatch.setattr(wallet_reconciliation, "load_wallet_state", lambda: {
        "wallet": "0xabc",
        "updated_at": "2020-01-01T00:00:00Z",
        "balances": [{"chain": "base", "symbol": "USDC", "quantity": "50", "usd": 50}],
    })
    result = wallet_reconciliation.reconciled_wallet_snapshot("guardian")
    assert result["fresh"] is False
    assert result["total_usd"] is None
    assert result["cached_total_usd"] == 50.0


def test_fresh_complete_snapshot_resolves_alias_and_reports_total(monkeypatch) -> None:
    monkeypatch.setattr(wallet_reconciliation, "load_wallet_state", lambda: {
        "wallet": "0xabc",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "balances": [
            {"chain": "base", "symbol": "USDC", "quantity": "2.5", "usd": 2.5},
            {"chain": "base", "symbol": "OLD", "quantity": "0", "usd": 99},
        ],
    })
    result = wallet_reconciliation.reconciled_wallet_snapshot("guardian")
    assert result["wallet"] == "0xabc"
    assert result["fresh"] is True
    assert result["total_usd"] == 2.5
    assert len(result["balances"]) == 1
