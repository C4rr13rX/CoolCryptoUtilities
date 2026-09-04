"""Reconciliation must not stop the bot that is holding an open position.

``GhostTradingSupervisor.reconcile_pairs`` makes room for a fresh ATF candidate
by REPLACING an existing bot: ``_free_bot_slot_for`` walks the pool backwards,
picks the first bot whose ``primary_symbol`` is not itself an ATF priority, and
awaits ``old_bot.stop()``.

That victim test was blind to the one property that makes a bot
irreplaceable. Every exit decision -- ``MAX_HOLD_SECONDS``, the stop loss, the
confidence-drop exit -- is made in ``TradingBot._interpret_predictions``, which
is reached only from a market sample for the symbol in hand. So the bot running
a symbol is the single place a position in that symbol can ever be CLOSED, and
stopping it converts an open position into tokens nothing will try to sell.

``_held_position_symbols`` exists for exactly that rule and is already consulted
seventy lines above, when ADDING bots. Ignoring it when REMOVING them
reintroduced the stranding it was written to end -- four of twelve open
positions with no ticking feed behind them, one of them held for 362.9 hours.

Not theoretical capacity pressure. Measured 2026-09-04 21:30-22:46 from the
ghost-supervisor log, twelve bots were evicted in 75 minutes:

    22:46:02  DEFI-USDC -> SPCX-USDC,  MOONBASE-USDC -> LFG-USDC
    22:32:26  LIQUIDBGT-USDC -> MOONBASE-USDC,  VI-USDC -> DEFI-USDC
    22:17:26  KEYCAT-USDC -> VI-USDC,  KYNDO-USDC -> LIQUIDBGT-USDC
    22:08:52  JACKET-USDC -> KYNDO-USDC,  CBETH-USDC -> KEYCAT-USDC
    ...

at a resolved ``pair_limit`` of 18 where ``full_slots`` is 0, so replacement is
the ONLY way a new candidate gets in and it ran on every reconcile. CBETH-USDC
is in that list, and CBETH-USDC is a symbol this wallet has repeatedly been left
holding unbooked tokens in.

The rule pinned here: a bot whose symbol appears in the position book is passed
over and an unheld one further down the pool is taken instead; when every
candidate is held, no replacement happens at all and the ATF signal waits. A
skipped entry costs an opportunity, a position nothing can sell costs capital.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import trading.selector as selector


class _Bot:
    """Enough of a TradingBot for reconcile's dedupe and eviction to run."""

    def __init__(self, symbol: str) -> None:
        self.primary_symbol = symbol
        self.bus_routes = {symbol: symbol.split("-")}
        self.live_trading_enabled = True
        self.stable_checkpoint_ratio = 0.0
        self.max_trade_share = 0.0
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True

    def configure_route(self, symbol, tokens):  # pragma: no cover - new bots only
        self.primary_symbol = str(symbol).upper()

    def apply_transition_plan(self, plan):  # pragma: no cover - not asserted here
        pass


def _reconcile(monkeypatch, *, pool, held, atf, pair_limit):
    """Run reconcile_pairs against a stubbed selection; return (result, pool).

    The returned list is a snapshot taken BEFORE the call: reconcile pops the
    bot it evicts out of ``sup.bots``, so reading that list afterwards cannot
    tell "was never stopped" from "was removed".
    """
    bots = [_Bot(symbol) for symbol in pool]
    created = list(bots)

    monkeypatch.setattr(selector, "_held_position_symbols", lambda db: list(held))
    monkeypatch.setattr(
        selector,
        "select_pairs",
        lambda limit=0, **kw: [
            selector.PairCandidate(
                symbol=s, tokens=s.split("-"), avg_volume=0.0,
                volatility=0.0, score=1.0, datapath=Path("."),
            )
            for s in atf
        ],
    )
    # max_limit is the real ceiling the held-position boost is allowed to
    # stretch to (measured 18-120 in production); pinning it AT pair_limit
    # would make the boost a no-op and hide the add path entirely.
    monkeypatch.setattr(
        selector, "resolve_pair_limit",
        lambda base, **kw: (pair_limit, {"max_limit": 120}),
    )
    monkeypatch.setattr(
        selector, "MarketDataStream", lambda **kw: object(), raising=False
    )

    def _new_bot(**kw):
        return _Bot("PENDING")

    monkeypatch.setattr(selector, "TradingBot", _new_bot)
    monkeypatch.setitem(
        __import__("sys").modules,
        "services.atf_static_strategy",
        type(
            "_M", (),
            {"latest_signals": staticmethod(lambda *a, **k: [{"symbol": s} for s in atf])},
        )(),
    )

    class _Pipeline:
        system_profile = None

        def live_readiness_report(self):
            return {}

        def ghost_live_transition_plan(self):
            return {}

    sup = selector.GhostTradingSupervisor.__new__(selector.GhostTradingSupervisor)
    sup.db = object()
    sup.pipeline = _Pipeline()
    sup.pair_limit = pair_limit
    sup.stream_total = 0
    sup.bots = bots
    sup.data_streams = []
    sup._tasks = []
    sup.stable_checkpoint_ratio = 0.0
    sup.rotator = None

    async def _go():
        # _run_bot_forever would start a real stream for any bot reconcile adds.
        sup._run_bot_forever = lambda bot: asyncio.sleep(0)
        return await sup.reconcile_pairs()

    return asyncio.run(_go()), created


# ---------------------------------------------------------------- the rule


def test_the_bot_holding_a_position_is_not_the_one_evicted(monkeypatch):
    """The CBETH case: pool full, a new ATF candidate, one bot holds tokens.

    HELD-USDC sits at the END of the pool, which is exactly where the backwards
    scan looks first -- so the old code stopped it and left the position with
    nothing that could sell it.
    """
    result, bots = _reconcile(
        monkeypatch,
        pool=["SPARE-USDC", "HELD-USDC"],
        held=["HELD-USDC"],
        atf=["NEW-USDC"],
        pair_limit=2,
    )

    by_symbol = {b.primary_symbol: b for b in bots}
    assert by_symbol["HELD-USDC"].stopped is False, "the held position lost its bot"
    assert by_symbol["SPARE-USDC"].stopped is True, "the spare bot should have gone"
    assert result["replaced_bots"] == [{"old": "SPARE-USDC", "new": "NEW-USDC"}]
    assert "NEW-USDC" in result["added_bots"], "the candidate still got in"


def test_no_replacement_happens_when_every_bot_is_held(monkeypatch):
    """Fail CLOSED. Waiting on a signal is cheaper than stranding capital."""
    result, bots = _reconcile(
        monkeypatch,
        pool=["A-USDC", "B-USDC"],
        held=["A-USDC", "B-USDC"],
        atf=["NEW-USDC"],
        pair_limit=2,
    )

    assert [b.stopped for b in bots] == [False, False]
    assert result["replaced_bots"] == []
    assert result["added_bots"] == []


def test_an_empty_book_leaves_replacement_exactly_as_it_was(monkeypatch):
    """The guard must not freeze rotation when nothing is held."""
    result, bots = _reconcile(
        monkeypatch,
        pool=["SPARE-USDC", "OTHER-USDC"],
        held=[],
        atf=["NEW-USDC"],
        pair_limit=2,
    )

    assert result["replaced_bots"] == [{"old": "OTHER-USDC", "new": "NEW-USDC"}]
    assert {b.primary_symbol for b in bots if b.stopped} == {"OTHER-USDC"}


def test_a_held_symbol_that_is_also_an_atf_priority_is_still_safe(monkeypatch):
    """Both exemptions at once must not cancel each other out."""
    result, bots = _reconcile(
        monkeypatch,
        pool=["SPARE-USDC", "HELD-USDC"],
        held=["HELD-USDC"],
        atf=["HELD-USDC", "NEW-USDC"],
        pair_limit=2,
    )

    by_symbol = {b.primary_symbol: b for b in bots}
    assert by_symbol["HELD-USDC"].stopped is False


def test_a_held_symbol_with_no_bot_is_still_added(monkeypatch):
    """The protect list must not have broken the add list.

    `held_symbols` (add) drops anything already covered; `held_all` (protect)
    must not. Reading one for both jobs is how this regressed.
    """
    result, _bots = _reconcile(
        monkeypatch,
        pool=["SPARE-USDC"],
        held=["STRANDED-USDC"],
        atf=[],
        pair_limit=1,
    )

    assert "STRANDED-USDC" in result["added_bots"], (
        "a held symbol with no bot was not given one"
    )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
