from __future__ import annotations

import time
from typing import Dict, Any

from cache import CacheTransfers
from db import TradingDatabase


def _transfer_event(token: str, block: int, hash_suffix: str) -> Dict[str, Any]:
    return {
        "hash": f"0xfeed{hash_suffix}",
        "logIndex": block,
        "blockNumber": hex(block),
        "timestamp": "2024-01-01T00:00:00Z",
        "rawContract": {"address": token},
        "from": "0x0000000000000000000000000000000000000001",
        "to": "0x0000000000000000000000000000000000000002",
        "value": "1",
    }


def test_popular_tokens_prefers_recent(tmp_path) -> None:
    db = TradingDatabase(path=str(tmp_path / "cache.db"))
    ct = CacheTransfers(db=db, indexer=None)
    wallet = "0xabc"
    chain = "ethereum"
    events = [
        _transfer_event("0xaaaa", 10, "01"),
        _transfer_event("0xbbbb", 12, "02"),
        _transfer_event("0xcccc", 14, "03"),
    ]
    ct.merge_new(wallet, chain, events)

    tokens = ct.popular_tokens(wallet, chain, limit=2, within_minutes=None)
    assert tokens == ["0xcccc", "0xbbbb"]

    # Age out the oldest token and ensure the TTL filter excludes it.
    with db._conn:
        db._conn.execute(
            "UPDATE transfers SET inserted_at=? WHERE token=?",
            (time.time() - 3600, "0xaaaa"),
        )
    filtered = ct.popular_tokens(wallet, chain, limit=3, within_minutes=5)
    assert "0xaaaa" not in filtered
