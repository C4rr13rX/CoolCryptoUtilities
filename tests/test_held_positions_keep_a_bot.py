"""A symbol we hold a position in must always have a bot that can close it.

Every exit decision -- ``MAX_HOLD_SECONDS``, the stop loss, the confidence-drop
exit -- is made inside ``TradingBot._interpret_predictions``, which is reached
only from a market sample: it reads ``symbol`` off the sample and then looks up
``self.positions.get(symbol)``. So an exit can only ever happen for a symbol
that just ticked, on a bot that is running for it.

``build()`` composed the bot pool from ATF signals, focus assets, the genome
seed and ``select_pairs()``, and never from the position book. A symbol that
dropped out of that selection kept its row in the persisted book and lost the
only thing that could close it. Measured 2026-09-02 against the live book, 4 of
12 open positions had no ticking feed behind them:

    UNI-USDC      rsi_reversal@1w       held 362.9h   never ticked
    HIGH-USDC     (none)                held 237.1h   never ticked
    ARB-USDC      obv_accumulation@5h   held  11.5h   last tick 10.9h ago
    VIRTUAL-USDC  obv_accumulation@5d   held   0.6h   last tick  0.6h ago

against a ``MAX_HOLD_SECONDS`` of 3600. VIRTUAL-USDC is the live case rather
than historical damage -- opened 34 minutes earlier, not one tick since entry,
in a process that had been up the whole time.

That is a graduation blocker. Promotion is scored on CLOSED ghost trades, so a
stranded position is a round trip that never reaches the ledger.
"""

from __future__ import annotations

import trading.selector as selector


class _FakeDB:
    def __init__(self, state):
        self._state = state

    def load_state(self):
        return self._state


def _book(*symbols):
    return _FakeDB(
        {"ghost_trading": {"positions": {s: {"entry_price": 1.0, "size": 1.0} for s in symbols}}}
    )


# ---------------------------------------------------------------- the reader


def test_held_symbols_are_read_from_the_position_book():
    db = _book("UNI-USDC", "VIRTUAL-USDC")
    assert selector._held_position_symbols(db) == ["UNI-USDC", "VIRTUAL-USDC"]


def test_symbols_are_normalised_and_deduped():
    db = _FakeDB(
        {
            "ghost_trading": {
                "positions": {
                    "uni-usdc": {"size": 1.0},
                    "UNI-USDC": {"size": 1.0},
                    "  arb-usdc  ": {"size": 1.0},
                }
            }
        }
    )
    assert selector._held_position_symbols(db) == ["UNI-USDC", "ARB-USDC"]


def test_a_row_that_is_not_a_position_is_skipped():
    """Giving a non-dict row a bot would burn a slot and close nothing."""
    db = _FakeDB(
        {"ghost_trading": {"positions": {"UNI-USDC": "not-a-position", "ARB-USDC": {"size": 1.0}}}}
    )
    assert selector._held_position_symbols(db) == ["ARB-USDC"]


def test_a_missing_or_broken_book_changes_nothing():
    """Pair selection must be left exactly as it was, never crashed."""

    class _Boom:
        def load_state(self):
            raise RuntimeError("state unreadable")

    assert selector._held_position_symbols(_Boom()) == []
    assert selector._held_position_symbols(_FakeDB({})) == []
    assert selector._held_position_symbols(_FakeDB({"ghost_trading": None})) == []
    assert selector._held_position_symbols(_FakeDB({"ghost_trading": {"positions": []}})) == []
    assert selector._held_position_symbols(_FakeDB("not-a-dict")) == []


def test_empty_book_yields_no_symbols():
    assert selector._held_position_symbols(_book()) == []


# ------------------------------------------------------- the ordering + limit


def _build_pool(monkeypatch, *, held, selected, atf, pair_limit, condemned=()):
    """Run build() against stubbed selection and capture the bot pool.

    ``condemned`` is the FIXTURE for ``_no_strategy_may_enter``. Patching it is
    not tidiness: unpatched, it asks ``symbol_edge_gate``,
    ``stop_survivability_gate`` and ``symbol_motion_gate``, all three of which
    read ``storage/trading_cache.db`` -- so every assertion below about ORDER
    was being decided by this week's prices. Measured 2026-09-10 with
    ``scripts/live_data_predicate_census.py --prove``: against the live
    database ``test_an_empty_book_leaves_selection_untouched`` FAILED and
    against an empty one it PASSED, because the motion gate condemns
    CBBTC-USDC today ("0.7% of 2389 15-minute windows cleared the 0.65% round
    trip") and ``_sink_condemned`` correctly moved it to the back.

    The sinking is production behaviour and is deliberate (commit 8060697,
    "379 of 596 decision cycles in 6h ran on symbols no strategy may enter"),
    so it is PINNED by ``test_a_condemned_candidate_sinks_below_a_clean_one``
    below rather than suppressed -- from this fixture, never from the tape.
    """
    monkeypatch.setattr(selector, "_held_position_symbols", lambda db: list(held))
    condemned_upper = {str(s).strip().upper() for s in condemned}
    monkeypatch.setattr(
        selector, "_no_strategy_may_enter",
        lambda symbol: ("fixture: condemned"
                        if str(symbol or "").strip().upper() in condemned_upper
                        else None),
    )
    monkeypatch.setattr(
        selector,
        "select_pairs",
        lambda limit=0, **kw: [
            selector.PairCandidate(
                symbol=s, tokens=s.split("-"), avg_volume=0.0,
                volatility=0.0, score=0.0, datapath=selector.Path("."),
            )
            for s in selected
        ],
    )
    monkeypatch.setattr(selector, "_genome_universe_symbols", lambda limit: [])
    monkeypatch.setattr(
        selector, "resolve_pair_limit",
        lambda base, **kw: (pair_limit, {"max_limit": 120}),
    )

    built = []

    class _Bot:
        live_trading_enabled = True
        stable_checkpoint_ratio = 0.0
        max_trade_share = 0.0
        bus_routes: dict = {}

        def __init__(self, **kw):
            pass

        def configure_route(self, symbol, tokens):
            built.append(symbol)

    monkeypatch.setattr(selector, "TradingBot", _Bot)
    monkeypatch.setattr(selector, "MarketDataStream", lambda **kw: object())

    class _Pipeline:
        system_profile = None

        def ghost_focus_assets(self):
            return [], {}

        def live_readiness_report(self):
            return {}

        def ghost_live_transition_plan(self):
            return {}

        def horizon_bias(self):
            return {}

    sup = selector.GhostTradingSupervisor.__new__(selector.GhostTradingSupervisor)
    sup.db = object()
    sup.pipeline = _Pipeline()
    sup.pair_limit = pair_limit
    sup.stream_total = pair_limit
    sup.bots = []
    sup.data_streams = []
    sup.stable_checkpoint_ratio = 0.0
    monkeypatch.setattr(
        selector, "latest_signals", lambda *a, **k: [], raising=False
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "services.atf_static_strategy",
        type("_M", (), {"latest_signals": staticmethod(lambda *a, **k: [{"symbol": s} for s in atf])})(),
    )
    sup.build()
    return built


def test_a_held_symbol_that_selection_dropped_still_gets_a_bot(monkeypatch):
    """The UNI-USDC case: held for 15 days, nowhere in the candidate list."""
    built = _build_pool(
        monkeypatch, held=["UNI-USDC"], selected=["CBBTC-USDC", "AERO-USDC"],
        atf=[], pair_limit=4,
    )
    assert "UNI-USDC" in built


def test_held_positions_outrank_new_candidates(monkeypatch):
    built = _build_pool(
        monkeypatch, held=["UNI-USDC", "ARB-USDC"],
        selected=["CBBTC-USDC"], atf=["BASECAT-USDC"], pair_limit=4,
    )
    assert built[:2] == ["UNI-USDC", "ARB-USDC"]


def test_the_limit_stretches_so_no_held_position_is_truncated(monkeypatch):
    """The measured trap: 12 open positions against a pair_limit of 8.

    Ordering held symbols first is not enough on its own -- ``all_ordered``
    is sliced to ``pair_limit``, so four of the twelve would have been dropped
    straight back into the state this rescues them from.
    """
    held = ["H%d-USDC" % i for i in range(12)]
    built = _build_pool(
        monkeypatch, held=held, selected=["CBBTC-USDC"], atf=[], pair_limit=8,
    )
    assert set(held) <= set(built), "a held position was truncated away"


def test_a_held_symbol_is_not_given_two_bots(monkeypatch):
    """Held and also returned by select_pairs must still mean one bot."""
    built = _build_pool(
        monkeypatch, held=["AERO-USDC"],
        selected=["AERO-USDC", "CBBTC-USDC"], atf=[], pair_limit=4,
    )
    assert built.count("AERO-USDC") == 1


def test_an_empty_book_leaves_selection_untouched(monkeypatch):
    """No positions held -> exactly the pairs selection asked for.

    With nothing condemned, so that this measures what its name says -- the
    POSITION BOOK's effect on selection -- and not the market.
    """
    built = _build_pool(
        monkeypatch, held=[], selected=["CBBTC-USDC", "AERO-USDC"],
        atf=[], pair_limit=4,
    )
    assert built == ["CBBTC-USDC", "AERO-USDC"]


def test_a_condemned_candidate_sinks_below_a_clean_one(monkeypatch):
    """The behaviour the live gates were silently exercising, pinned.

    ``_sink_condemned`` moves a symbol no strategy may enter to the back of
    the pool rather than dropping it, so a slot is not spent on a symbol the
    entry gate refuses on sight. Fed from the fixture, this holds whatever the
    tape says today; before the fixture existed it was being asserted by
    accident, in the opposite direction, by a test about something else.
    """
    built = _build_pool(
        monkeypatch, held=[], selected=["CBBTC-USDC", "AERO-USDC"],
        atf=[], pair_limit=4, condemned=["CBBTC-USDC"],
    )
    assert built == ["AERO-USDC", "CBBTC-USDC"]
    assert set(built) == {"CBBTC-USDC", "AERO-USDC"}, "condemned means sunk, never dropped"


def test_a_held_position_outranks_its_own_condemnation(monkeypatch):
    """``protected`` exists so a gate cannot strand the position it holds.

    A held symbol is condemned by the same gates as anything else -- CBBTC-USDC
    is condemned today -- and sinking it below the pair limit would take away
    the only bot that can close it, which is the exact failure this file is
    named for.
    """
    built = _build_pool(
        monkeypatch, held=["CBBTC-USDC"], selected=["AERO-USDC"],
        atf=[], pair_limit=4, condemned=["CBBTC-USDC"],
    )
    assert built[0] == "CBBTC-USDC", "a held position lost its place to a gate"
