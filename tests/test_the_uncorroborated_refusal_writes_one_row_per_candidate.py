"""The scout's uncorroborated-price refusal must be one row per candidate.

WHY THIS TEST EXISTS. The gate at ``services/atf_static_strategy.py``
(``_corroborated_price``, called from the entry loop of
``_run_ghost_quote_scout``) refused candidates and left exactly one trace: an
aggregate line in ``system.log`` reading "refused N uncorroborated
candidate(s)", where N is a COUNT and the symbol list is truncated to eight.
Six passes of this loop measured that funnel from the OUTSIDE and guessed at
it, because there was no database row to count. This test pins the fix:

  1. one ``trading_ops`` row per REFUSED candidate, carrying the symbol and
     the branch that refused it;
  2. the rows RECONCILE to the aggregate -- the number of rows equals the N in
     the aggregate line, because both are driven by the same list;
  3. the gate is NOT loosened: a refused candidate is still refused, and no
     ghost-entry row is written for it.

SCOUT LANE ONLY. ``ghost-entry`` has two writers with different denominators
(this scout, and trading/bot.py's directive path which never reaches this
gate), so these rows count SCOUT candidates and nothing else.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest

from services import atf_static_strategy as atf


class _FakeDb:
    """Records every ``log_trade`` call; a feed with no ticks at all."""

    def __init__(self, prices: Dict[str, float] | None = None) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.prices = prices or {}
        self.json: Dict[str, Any] = {}

    # -- the feed ---------------------------------------------------------
    def recent_market_prices(self, symbol, chain, since_ts=None, limit=25):
        price = self.prices.get(str(symbol).upper())
        return [(price,)] * 3 if price else []

    def get_market_price(self, symbol, chain, ts=None, after=True):
        price = self.prices.get(str(symbol).upper())
        return (price,) if price else None

    # -- state ------------------------------------------------------------
    def get_json(self, key, default=None):
        return self.json.get(key, default)

    def set_json(self, key, value):
        self.json[key] = value

    # -- the instrument under test ---------------------------------------
    def log_trade(self, **kwargs):
        self.rows.append(kwargs)

    # -- everything else the loop may reach for --------------------------
    def fetch_trades(self, *a, **k):
        return []


def _signal(symbol: str, price: float) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "price_usd": price,
        "quote_probe": {"ok": True},
        "target_return": 0.02,
    }


@pytest.fixture()
def _env(monkeypatch):
    # The gate ON, at its production thresholds, and every downstream refusal
    # that could pre-empt it neutralised so the test isolates THIS gate.
    monkeypatch.setenv("ATF_STATIC_QUOTE_GHOST_SCOUT_ENABLED", "1")
    monkeypatch.setenv("ATF_STATIC_REQUIRE_FEED_PRICE", "1")
    monkeypatch.setenv("ATF_STATIC_FEED_MAX_AGE_SEC", "900")
    monkeypatch.setenv("ATF_STATIC_MAX_FEED_DEV", "0.35")
    monkeypatch.setattr(atf, "_strategy_edge_refusal", lambda *a, **k: None)
    monkeypatch.setattr(atf, "_symbol_edge_refusal", lambda *a, **k: None)
    yield


def _refusal_rows(db: _FakeDb) -> List[Dict[str, Any]]:
    return [r for r in db.rows
            if r.get("status") == "entry-refused-uncorroborated-price"]


def test_a_silent_feed_writes_one_row_per_refused_candidate(_env):
    """Three candidates, a feed with no ticks: three rows, three symbols."""
    db = _FakeDb(prices={})
    signals = [_signal("MAMO-USDC", 0.0105),
               _signal("UNI-USDC", 6.40),
               _signal("AERO-USDC", 0.5153)]

    atf._run_ghost_quote_scout(
        db=db, signals=signals, chain="base",
        quote_token="USDC", max_positions=10,
    )

    rows = _refusal_rows(db)
    assert len(rows) == 3, "one row per refused candidate, not one per loop"
    assert sorted(r["symbol"] for r in rows) == ["AERO-USDC", "MAMO-USDC", "UNI-USDC"]
    for row in rows:
        details = row["details"]
        # The BRANCH, named -- this is what the aggregate line could never say.
        assert details["reason"] == "no_tick_within_max_age"
        assert details["strategy_id"] == atf.SCOUT_STRATEGY_ID
        assert details["lane"] == "scout"
        assert details["feed"] is None
        assert float(details["max_age_sec"]) == 900.0


def test_the_deviation_branch_is_distinguishable_from_the_silence_branch(_env):
    """A feed that disagrees is a DIFFERENT reason, with both numbers on it."""
    # Feed says 1.00, the quote says 2.00: 100% deviation against a 35% bar.
    db = _FakeDb(prices={"MAMO-USDC": 1.0})
    atf._run_ghost_quote_scout(
        db=db, signals=[_signal("MAMO-USDC", 2.0)], chain="base",
        quote_token="USDC", max_positions=10,
    )

    rows = _refusal_rows(db)
    assert len(rows) == 1
    details = rows[0]["details"]
    assert details["reason"] == "quote_deviates_from_feed"
    assert pytest.approx(details["feed"], rel=1e-9) == 1.0
    assert pytest.approx(details["quoted"], rel=1e-9) == 2.0
    assert pytest.approx(details["deviation"], rel=1e-9) == 1.0
    assert pytest.approx(details["max_dev"], rel=1e-9) == 0.35


def test_the_rows_reconcile_to_the_aggregate_and_the_gate_is_not_loosened(_env, monkeypatch):
    """Row count == the N in 'refused N uncorroborated candidate(s)'.

    A disagreement here would mean the instrument sits on a different path
    from the one that is refusing -- the exact failure the item warned about.
    """
    logged: List[str] = []

    import services.logging_utils as logging_utils
    monkeypatch.setattr(logging_utils, "log_message",
                        lambda *a, **k: logged.append(str(a[1]) if len(a) > 1 else ""))

    db = _FakeDb(prices={})
    signals = [_signal("MAMO-USDC", 0.0105), _signal("UNI-USDC", 6.40)]
    atf._run_ghost_quote_scout(
        db=db, signals=signals, chain="base",
        quote_token="USDC", max_positions=10,
    )

    rows = _refusal_rows(db)
    aggregate = [line for line in logged
                 if "uncorroborated candidate(s)" in line]
    assert aggregate, "the aggregate line must still be emitted -- nothing removed"
    n = int(aggregate[0].split("refused ", 1)[1].split(" ", 1)[0])
    assert len(rows) == n, (
        "per-candidate rows (%d) must reconcile to the aggregate N (%d)"
        % (len(rows), n)
    )

    # AND THE GATE IS NOT LOOSENED: nothing entered.
    entries = [r for r in db.rows if r.get("status") == "ghost-entry"]
    assert entries == [], "seeing the refusal must not turn it into an entry"
