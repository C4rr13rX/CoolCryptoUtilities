from __future__ import annotations

import time
from pathlib import Path

from db import TradingDatabase


def test_pair_suppression_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "suppression.db"
    db = TradingDatabase(path=str(db_path))

    symbol = "WETH-TEST"
    assert db.get_pair_suppression(symbol) is None
    assert db.is_pair_suppressed(symbol) is False

    db.record_pair_suppression(symbol, "no_live_market", ttl_seconds=5, metadata={"source": "unit-test"})

    entry = db.get_pair_suppression(symbol)
    assert entry is not None
    assert entry["reason"] == "no_live_market"
    assert entry["metadata"]["source"] == "unit-test"
    assert db.is_pair_suppressed(symbol) is True

    # Force expiration and ensure automatic clearing.
    with db._conn:  # type: ignore[attr-defined]
        db._conn.execute(
            "UPDATE pair_suppression SET release_ts=? WHERE symbol=?",
            (time.time() - 1, symbol),
        )

    assert db.is_pair_suppressed(symbol) is False
    assert db.get_pair_suppression(symbol) is None
