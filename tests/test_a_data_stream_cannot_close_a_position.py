"""A held symbol needs a BOT. A data-only stream is not coverage.

``reconcile_pairs`` already knows that "a symbol we hold must have a bot that
can close it" -- ``_held_position_symbols`` exists for that rule and reads the
book correctly. The answer was then thrown away one line later, by testing the
held book against ``existing``, which deliberately includes
``self.data_streams``.

That set membership is right for pair SELECTION (nothing should be streamed
twice) and wrong for the held book. A data-only stream runs
``_run_data_stream_forever`` -- websocket to ``market_stream`` -- and never
calls ``_handle_sample``. Every exit rule in the bot hangs off
``_handle_sample``: the stop, the profit target, the timed exit, the confidence
drop and ``MAX_HOLD_FORCE_SECONDS``. So a data stream produces a healthy-looking
price for a symbol that has nothing able to SELL it, and the position is marked
covered precisely because of the thing that cannot close it.

Measured 2026-09-05, both live positions in the book were in that state:

    CBBTC-USDC   held 19.8h   50 ticks/h in market_stream, no bot
    CBXRP-USDC   held 19.1h   49 ticks/h in market_stream, no bot

with 100 ``entry-refused-duplicate`` rows on CBBTC-USDC and ZERO live trades on
the day -- ``atf_static`` is the only live-approved strategy and both of its
slots were held by positions it could not reach. The status line read "0 open"
throughout, because it was reading ``state['positions']`` instead of
``state['ghost_trading']['positions']``, so nothing surfaced it.

The rules pinned here:

  * a held symbol covered only by a data-only stream still gets a bot;
  * that data stream is retired when the bot takes over, so one pair is not
    streamed twice and the data slot comes back;
  * a held symbol already covered by a bot does not get a second one (the
    duplicate-bot bug: 13 bots on BASECAT-USDC on 2026-08-31, each with its own
    ``self.positions``);
  * when no bot slot is free, a held symbol is left UNCOVERED rather than given
    a data-only stream. Re-adding one would mark it handled at the next
    reconcile and strand it again -- the failure repeating itself.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import trading.selector as selector


class _Bot:
    def __init__(self, symbol: str) -> None:
        self.primary_symbol = symbol
        self.bus_routes = {symbol: symbol.split("-")}
        self.live_trading_enabled = True
        self.stable_checkpoint_ratio = 0.0
        self.max_trade_share = 0.0
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True

    def configure_route(self, symbol, tokens):
        self.primary_symbol = str(symbol).upper()

    def apply_transition_plan(self, plan):
        pass


class _DataStream:
    """A data-only stream: it has a symbol and it can never close anything."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


def _reconcile(monkeypatch, *, pool, streams, held, atf,
               pair_limit, ceiling=120, stream_total=8, replace="0"):
    bots = [_Bot(symbol) for symbol in pool]
    data_streams = [_DataStream(symbol) for symbol in streams]
    # A snapshot taken BEFORE the call: reconcile pops the stream it retires
    # out of ``sup.data_streams``, so reading that list afterwards cannot tell
    # "was never stopped" from "was removed".
    created_streams = list(data_streams)

    monkeypatch.setenv("ATF_STATIC_REPLACE_BOTS", replace)
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
    monkeypatch.setattr(
        selector, "resolve_pair_limit",
        lambda base, **kw: (pair_limit, {"max_limit": ceiling}),
    )
    monkeypatch.setattr(
        selector, "MarketDataStream",
        lambda **kw: _DataStream(str(kw.get("symbol") or "")), raising=False,
    )
    monkeypatch.setattr(selector, "TradingBot", lambda **kw: _Bot("PENDING"))
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
    sup.stream_total = stream_total
    sup.bots = bots
    sup.data_streams = data_streams
    sup._tasks = []
    sup.stable_checkpoint_ratio = 0.0
    sup.rotator = None

    async def _go():
        sup._run_bot_forever = lambda bot: asyncio.sleep(0)
        sup._run_data_stream_forever = lambda stream: asyncio.sleep(0)
        return await sup.reconcile_pairs()

    return asyncio.run(_go()), sup, created_streams


def test_a_held_symbol_with_only_a_data_stream_still_gets_a_bot(monkeypatch):
    """The exact CBBTC-USDC case: it ticks, and nothing can sell it."""
    result, sup, _ = _reconcile(
        monkeypatch,
        pool=["AERO-USDC"],
        streams=["CBBTC-USDC"],
        held=["CBBTC-USDC"],
        atf=[],
        pair_limit=4,
    )
    assert "CBBTC-USDC" in result["added_bots"], (
        "a data-only stream marked the position covered; it cannot close it"
    )
    assert "CBBTC-USDC" in {
        str(getattr(b, "primary_symbol", "")).upper() for b in sup.bots
    }


def test_the_data_stream_is_retired_when_the_bot_takes_over(monkeypatch):
    """One pair, one stream. The bot replaces it rather than doubling it."""
    _result, sup, created = _reconcile(
        monkeypatch,
        pool=["AERO-USDC"],
        streams=["CBBTC-USDC"],
        held=["CBBTC-USDC"],
        atf=[],
        pair_limit=4,
    )
    assert "CBBTC-USDC" not in {
        str(getattr(s, "symbol", "")).upper() for s in sup.data_streams
    }
    assert created[0].stopped, "the superseded data stream must be stopped"


def test_both_stranded_symbols_are_given_bots(monkeypatch):
    """CBBTC and CBXRP together -- both of atf_static's blocked slots."""
    result, _sup, _ = _reconcile(
        monkeypatch,
        pool=["AERO-USDC"],
        streams=["CBBTC-USDC", "CBXRP-USDC"],
        held=["CBBTC-USDC", "CBXRP-USDC"],
        atf=[],
        pair_limit=4,
    )
    assert set(result["added_bots"]) >= {"CBBTC-USDC", "CBXRP-USDC"}


def test_a_held_symbol_already_run_by_a_bot_gets_no_second_bot(monkeypatch):
    """The 2026-08-31 duplicate-bot bug must not come back."""
    result, sup, _ = _reconcile(
        monkeypatch,
        pool=["CBBTC-USDC"],
        streams=[],
        held=["CBBTC-USDC"],
        atf=[],
        pair_limit=4,
    )
    assert "CBBTC-USDC" not in result["added_bots"]
    assert sum(
        1 for b in sup.bots
        if str(getattr(b, "primary_symbol", "")).upper() == "CBBTC-USDC"
    ) == 1


def test_a_held_symbol_is_never_given_a_data_stream_instead_of_a_bot(monkeypatch):
    """With no bot slot free, leave it uncovered -- do not re-strand it.

    A data-only stream would mark the symbol covered at the next reconcile,
    which is exactly how it stayed unclosable for 19.8h.
    """
    _result, sup, _ = _reconcile(
        monkeypatch,
        pool=["AERO-USDC"],
        streams=[],
        held=["CBBTC-USDC"],
        atf=[],
        pair_limit=1,
        ceiling=1,        # no room for the held-position boost to stretch into
        stream_total=8,   # ...but plenty of DATA slots, which must not be used
        replace="0",
    )
    assert "CBBTC-USDC" not in {
        str(getattr(s, "symbol", "")).upper() for s in sup.data_streams
    }, "a data-only stream cannot close a position and must not stand in for one"


def test_an_empty_book_leaves_data_streams_alone(monkeypatch):
    """Nothing held -- the data-only tier keeps working exactly as before."""
    _result, sup, created = _reconcile(
        monkeypatch,
        pool=["AERO-USDC"],
        streams=["CBBTC-USDC"],
        held=[],
        atf=[],
        pair_limit=4,
    )
    assert "CBBTC-USDC" in {
        str(getattr(s, "symbol", "")).upper() for s in sup.data_streams
    }
    assert not created[0].stopped
