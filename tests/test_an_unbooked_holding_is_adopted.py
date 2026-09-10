"""Tokens the wallet holds and the book does not are booked as a live position.

Every exit in this bot is driven off ``self.positions[symbol]``. A settled buy
that leaves no position is therefore never offered to the take-profit, the
stop, or the timed exit -- nothing will ever try to sell it. It is not a trade,
it is a donation.

Measured 2026-09-04, on-chain ``balanceOf`` for
0x291c854811e92906a658Fb94Aa511bF919f968ad against the persisted position book:

    BSTONK  360.2642432254   1 settled buy,  0.75 USDC   no position
    CBBTC     0.0000370900   4 settled buys, 3.00 USDC   no position

3.75 USDC behind five settled swaps, against a stable leg of 13.6963 -- 21% of
the book, stranded. That is what "13 buys against 4 sells" looks like from the
wallet's side.

Two mechanisms put them there. A live-approved entry released the live position
it landed on (four ``position-released`` rows carrying ``released_mode:
"live"``, pinned in test_a_live_slot_refuses_every_entry.py), and a live
position was closed by simulation when the bot-level live flag flapped (pinned
in test_a_live_position_never_exits_in_simulation.py). Both are fixed. Neither
fix returns the tokens that are already outside the book.

The reconstruction is a measurement, not a guess, and this file pins that:

  * the SIZE is the chain's, read now, in raw base units;
  * the BASIS is the USDC those settled entries actually spent over the base
    they actually recorded receiving -- for CBBTC 3.00 / 3.709e-05 = 80884.34,
    and those four recorded sizes sum to 3.709e-05, which is the on-chain
    balance to the last raw unit;
  * ``entry_ts`` is the OLDEST unmatched buy, so the max-hold clock counts from
    when the money actually left, not from when the book noticed;
  * a NEW trade_id is minted. The annulled exits these positions were
    fictionally closed by carry ``remaining_size: 0.0``, and ``_load_state``
    drops any position whose newest outcome says that -- adopting under the old
    id would book a position that vanishes on the next restart.
"""

from __future__ import annotations

import time

import pytest

from trading.bot import TradingBot

SYMBOL = "CBBTC-USDC"
CBBTC = "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf"
DECIMALS = 8
ONCHAIN_RAW = 3709                      # 0.0000370900 cbBTC
MARK = 81099.6196791696                 # coinpaprika print, 2026-09-03

# The four settled buys, verbatim from trading_ops.
BUYS = [
    (1788452530.052378, 9.28e-06, 0.75, 80818.96551724139,
     "0x53f7303248255d6923b42c5920a58413c8b4c85b57cb54df01be0e14899b07ad",
     "2:CBBTC-USDC:c9d8cee9485047ec8fa7c10ab247e3b0"),
    (1788452667.676974, 9.27e-06, 0.75, 80906.14886731392,
     "0x3c992f2709bb54a39dc59d832aa3cc6c4dd0f006c40579f36e3e69cd6f295666",
     "2:CBBTC-USDC:c83106b5b6414ebcb1da57a16afcef27"),
    (1788453502.200222, 9.26e-06, 0.75, 80993.52051835853,
     "0xd9f7c07aba10d585a02426c23e771e73b3babc8b927fdd3cf8d230103f826d3e",
     "2:CBBTC-USDC:12cd624b75084440b00d46076ba16a2d"),
    (1788453645.202722, 9.28e-06, 0.75, 80818.96551724139,
     "0x0dfedf3a7c4f77a35706ddca6813f9ec93455c0a5d7e5d3c7b9d2760ab8f0959",
     "2:CBBTC-USDC:490d9726bea04d8f8ea5c8394c867995"),
]


class _Stub:
    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


def _entry_row(ts, size, spent, price, tx, trade_id):
    return {
        "ts": ts,
        "wallet": "live",
        "chain": "base",
        "symbol": SYMBOL,
        "action": "enter",
        "status": "live-entry",
        "details": {
            "symbol": SYMBOL,
            "status": "live-entry",
            "executed": True,
            "size": size,
            "quote_spent": spent,
            "entry_price": price,
            "entry_ts": ts,
            "timestamp": ts,
            "tx_hash": tx,
            "trade_id": trade_id,
            "strategy_id": "atf_static",
            "route": ["CBBTC", "USDC"],
            "target_price": price * 1.05,
            "gas_spent_native": 2.6e-06,
        },
    }


def _settled_row(tx, trade_id, purpose="live_entry", ts=0.0):
    """A settled swap in the shape the chain actually writes.

    ``purpose`` IS NOT OPTIONAL AND IT IS WHY THESE TESTS WERE RED.

    This fixture predates the ``purpose`` filter in
    ``bot._unmatched_live_entry_details``, which walks the settled rows and
    keeps only ``purpose == "live_entry"``, breaking on ``live_exit``. With no
    such key the fixture's rows read as neither, every one was skipped, the
    scan returned [] and ``_adopt_orphaned_live_holding`` returned None -- so
    all ten tests in this file failed on the same line for a reason that had
    nothing to do with adoption.

    Production is right and must not be loosened. Measured 2026-09-10 over the
    last 40 ``live-swap-settled`` rows in the live database: ``purpose`` is
    present on 40 of 40, reading live_entry 20, live_exit 19, quote_topup 1.
    Dropping that filter would let a settled SELL, or the stablecoin top-up
    swap, be counted as an unmatched BUY -- which is how a basis gets computed
    over money that was never spent on the base asset.
    """
    return {
        "ts": ts,
        "wallet": "live",
        "chain": "base",
        "symbol": SYMBOL,
        "action": "swap",
        "status": "live-swap-settled",
        "details": {
            "tx_hash": tx,
            "trade_id": trade_id,
            "purpose": purpose,
            "buy": CBBTC,
            "sell": "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
            "confirmed": True,
            "ok": True,
        },
    }


class _DB(_Stub):
    """Serves trading_ops the way db.fetch_trades does: NEWEST FIRST."""

    def __init__(self, entries=None, exit_ts=None):
        self.logged: list = []
        rows = [_entry_row(*b) for b in (entries if entries is not None else BUYS)]
        if exit_ts is not None:
            rows.append(
                {
                    "ts": exit_ts,
                    "wallet": "live",
                    "chain": "base",
                    "symbol": SYMBOL,
                    "action": "exit",
                    "status": "live-exit",
                    "details": {"tx_hash": "0xsold", "executed": True},
                }
            )
        self._rows = sorted(rows, key=lambda r: r["ts"], reverse=True)
        # NEWEST FIRST, and carrying a real ``ts`` -- because that is what
        # ``db.fetch_trades`` returns and what the scan is written against.
        # These rows were built in BUYS order (oldest first) with ts pinned to
        # 0.0, so the fixture handed a newest-first reader an oldest-first list
        # with no ordering information in it at all. Two consequences: the
        # adopted tx list came back reversed, and -- far worse for anything
        # this file is meant to guard -- the "a settled sell closes everything
        # older" break in `_unmatched_live_entry_details` would have broken on
        # the wrong rows, which no test here could have caught.
        self._settled = sorted(
            (_settled_row(b[4], b[5], ts=b[0]) for b in BUYS),
            key=lambda r: r["ts"],
            reverse=True,
        )

    def fetch_trades(self, *, limit=200, statuses=None, wallets=None,
                     symbol=None, since_ts=None):
        wanted = set(statuses or [])
        if "live-swap-settled" in wanted:
            return list(self._settled)[:limit]
        return [r for r in self._rows if r["status"] in wanted][:limit]

    def log_trade(self, **kwargs):
        self.logged.append(kwargs)
        return True

    def statuses(self):
        return [row.get("status") for row in self.logged]


class _Swapper:
    def __init__(self, raw=ONCHAIN_RAW, decimals=DECIMALS):
        self._reading = (raw, decimals)

    def token_balance_raw(self, chain, token):
        assert token.lower() == CBBTC, f"exit aimed at the wrong contract: {token}"
        return self._reading


def _bot(db=None, swapper=None) -> TradingBot:
    bot = TradingBot.__new__(TradingBot)
    TradingBot._ensure_runtime_state(bot)
    bot.db = db if db is not None else _DB()
    bot.metrics = _Stub()
    bot.positions = {}
    bot.bus_routes = {}
    bot.primary_chain = "base"
    bot.stable_tokens = {"USDC"}
    bot.live_trading_enabled = True
    bot.ghost_session_id = 2
    bot._bridge = _Stub()
    bot.saved = 0

    def _save():
        bot.saved += 1

    bot._save_state = _save
    bot._claim_position_symbol = lambda sym: None
    bot._init_bridge = lambda: _Stub()
    bot._resolve_live_trade_asset = lambda chain, sym, explicit=None: (sym, explicit or "")
    bot._verified_address = lambda chain, sym, addr, source: addr
    bot._new_swapper = lambda: (swapper if swapper is not None else _Swapper())
    return bot


@pytest.fixture(autouse=True)
def _live_execution(monkeypatch):
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "0")
    monkeypatch.delenv("EXIT_DUST_SWEEP_USD", raising=False)
    monkeypatch.setenv("WALLET_DUST_USD", "0.50")


def test_the_unbooked_cbbtc_holding_is_adopted_as_a_live_position():
    bot = _bot()

    pos = bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK)

    assert pos is not None, "3.00 USDC of settled buys stayed outside the book"
    assert bot.positions[SYMBOL] is pos
    assert pos["mode"] == "live"


def test_the_size_is_the_chains_and_not_the_books():
    """0.0000370900 -- raw 3709 at 8 decimals, read now, not remembered."""
    bot = _bot()
    pos = bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK)
    assert pos["size"] == pytest.approx(3.709e-05, rel=1e-12)


def test_the_basis_is_usdc_spent_over_base_received():
    """3.00 / 3.709e-05 = 80884.34, against a cbBTC print of 81099.62.

    Not the feed price, not the last entry's price, and not an average of the
    four quoted prices -- the money that actually left over the tokens that
    actually arrived.
    """
    bot = _bot()
    pos = bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK)

    assert pos["quote_spent"] == pytest.approx(3.0)
    assert pos["adopted_recorded_size"] == pytest.approx(3.709e-05, rel=1e-12)
    assert pos["entry_price"] == pytest.approx(3.0 / 3.709e-05, rel=1e-12)
    assert pos["entry_price"] == pytest.approx(80884.335, abs=0.01)
    # Chain and entries agree to the last raw unit here, so nothing is
    # extrapolated and the reconstruction is a measurement end to end.
    assert pos["adopted_unaccounted_base"] == 0.0
    assert pos["basis_estimated"] is False
    assert pos["adopted"] is True
    assert pos["adopted_from_tx_hashes"] == [b[4] for b in reversed(BUYS)]
    assert all(len(h) == 66 for h in pos["adopted_from_tx_hashes"])


def test_the_cost_covers_the_whole_holding_not_just_the_accounted_part():
    """The fabricated-win case, with this wallet's real AERO numbers.

    The exit books ``gross_profit = quote_received - quote_spent * (base_sold /
    held_size)``. Size comes from the chain, cost from the entry rows, and they
    do not have to agree -- an older buy that was only partly sold leaves base
    behind that no unmatched entry accounts for. Measured 2026-09-04:
    3.040902389960829 AERO on chain against 1.494620938 recorded by the one
    unmatched entry, which spent 0.750000 USDC. Costing 3.04 tokens at 0.75
    books ``3.0409 * 0.5008 - 0.75 = +0.7729`` gross the moment it is adopted:
    a 103% win on a position that has not moved.

    The whole holding is costed at the basis that WAS measured, and the
    extrapolation is named rather than hidden.
    """
    onchain = 2 * ONCHAIN_RAW                    # twice what the entries recorded
    bot = _bot(swapper=_Swapper(raw=onchain))

    pos = bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK)

    held = pos["size"]
    assert held == pytest.approx(2 * 3.709e-05, rel=1e-12)
    assert pos["entry_price"] == pytest.approx(3.0 / 3.709e-05, rel=1e-12)
    assert pos["quote_spent"] == pytest.approx(pos["entry_price"] * held, rel=1e-12)
    assert pos["quote_spent"] == pytest.approx(6.0, rel=1e-12)

    # What the exit would book on a full close at the adoption mark: the
    # position has not moved, so neither has the P/L.
    cost_portion = pos["quote_spent"] * (held / held)
    assert held * pos["entry_price"] - cost_portion == pytest.approx(0.0, abs=1e-9)

    # And the extrapolation is on the record, not silently folded in.
    assert pos["adopted_unaccounted_base"] == pytest.approx(3.709e-05, rel=1e-12)
    assert pos["adopted_recorded_quote_spent"] == pytest.approx(3.0)
    assert pos["basis_estimated"] is True


def test_the_clock_starts_at_the_oldest_unmatched_buy():
    """max_hold and the timed exit must count from when the money left."""
    bot = _bot()
    pos = bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK)
    assert pos["entry_ts"] == pytest.approx(BUYS[0][0])
    assert pos["ts"] == pos["entry_ts"]


def test_a_new_trade_id_is_minted_not_the_annulled_one_reused():
    """The old ids' newest outcomes carry remaining_size 0.0.

    ``_load_state`` drops any position whose newest outcome says that, so
    adopting under an old id books a position that dies on the next restart.
    """
    bot = _bot()
    pos = bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK)

    assert pos["trade_id"] not in [b[5] for b in BUYS]
    assert pos["trade_id"].startswith("2:CBBTC-USDC:")
    assert set(pos["adopted_from_trade_ids"]) == {b[5] for b in BUYS}


def test_the_exit_is_aimed_at_the_contract_the_money_went_into():
    """Resolved from the settled swap's ``buy``, never from the ticker.

    131 of 408 base symbols map to more than one contract; adopting the wrong
    one aims a sell at an asset the wallet does not hold.
    """
    bot = _bot()
    pos = bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK)
    assert pos["base_token_address"].lower() == CBBTC


def test_the_adoption_is_persisted_and_logged():
    bot = _bot()
    bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK)

    assert bot.saved == 1, "an adoption that is not saved dies with the process"
    assert "live-position-adopted" in bot.db.statuses()
    row = [r for r in bot.db.logged if r["status"] == "live-position-adopted"][-1]
    assert row["wallet"] == "live"
    assert row["details"]["onchain_raw"] == ONCHAIN_RAW
    assert row["details"]["decimals"] == DECIMALS


def test_a_settled_sell_closes_everything_older():
    """Buys before the newest settled live-exit are not re-adopted."""
    bot = _bot(db=_DB(exit_ts=BUYS[-1][0] + 1.0))
    assert bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK) is None


def test_a_symbol_already_in_the_book_is_left_alone():
    bot = _bot()
    held = {"mode": "ghost", "size": 1.0, "entry_price": 1.0,
            "entry_ts": time.time(), "ts": time.time()}
    bot.positions[SYMBOL] = held
    assert bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK) is None
    assert bot.positions[SYMBOL] is held


def test_dust_is_not_adopted():
    """Below the sweep floor the holding cannot pay for the swap that clears it.

    Booking it would only produce an exit that can never fill -- the
    no_tx_hash retry loop that ended the one burst of rapid trading this system
    has produced (10 refusals on 0.00000037 cbETH, 16:38-17:13 on 2026-09-03).
    """
    bot = _bot(swapper=_Swapper(raw=100))     # 0.000001 cbBTC = 0.081 USD
    assert bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK) is None
    assert SYMBOL not in bot.positions


def test_a_zero_balance_is_not_adopted():
    bot = _bot(swapper=_Swapper(raw=0))
    assert bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK) is None


def test_an_unreadable_balance_refuses_rather_than_guessing():
    class _Blind:
        def token_balance_raw(self, chain, token):
            return None

    bot = _bot(swapper=_Blind())
    assert bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK) is None
    assert SYMBOL not in bot.positions


def test_a_stub_contract_is_never_adopted():
    """NEVER TRADE A TOKEN WHOSE CONTRACT HAS NO CODE ON CHAIN.

    ``_resolve_live_trade_asset`` lets an explicit address win over every
    symbol lookup, which also skips the chain interrogation that
    ``_resolve_token_address`` performs -- so the guard is asked here. BASECAT
    (0xB2000000...) has ONE byte of code, ``decimals()`` still answers 18, and
    its two settled buys are unsellable to this day. Adopting a stub would book
    a position whose exit can never fill: the no_tx_hash retry loop.
    """
    bot = _bot()
    refused = []

    def _guard(chain, sym, addr, source):
        refused.append((sym, addr, source))
        return None

    bot._verified_address = _guard

    assert bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK) is None
    assert SYMBOL not in bot.positions
    assert refused and refused[0][2] == "orphan_adoption"


def test_an_unexecuted_entry_is_not_evidence_of_a_holding():
    """live-entry rows with no tx_hash are the no_fill_detected class."""
    ghosts = [(ts, size, spent, price, "", tid) for ts, size, spent, price, _tx, tid in BUYS]
    bot = _bot(db=_DB(entries=ghosts))
    assert bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK) is None


def test_the_chain_is_not_read_again_within_the_throttle():
    """One symbol may not bill an RPC per tick."""
    calls = {"n": 0}

    class _Counting(_Swapper):
        def token_balance_raw(self, chain, token):
            calls["n"] += 1
            return super().token_balance_raw(chain, token)

    bot = _bot(swapper=_Counting())
    assert bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK) is not None
    bot.positions.pop(SYMBOL)
    assert bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK) is None
    assert calls["n"] == 1


def test_nothing_is_adopted_when_live_execution_is_not_armed(monkeypatch):
    """A dry run must never book a position claiming real tokens."""
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "1")
    bot = _bot()
    assert bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK) is None

    bot = _bot()
    bot.live_trading_enabled = False
    bot._orphan_adoption_checked_at.clear()
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "0")
    assert bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK) is None


# THE FIXTURE PINS THE SYMBOL EDGE GATE, IT DOES NOT WEAKEN IT.
#
# These tests are about what happens to the POSITION BOOK once an entry
# fires. The symbol edge gate sits upstream of all of it, and it answers
# from production: services/symbol_edge_gate opens storage/trading_cache.db
# at test time and reads the live closed book. On 2026-09-10 BASECAT-USDC
# crossed the ban threshold -- 35 closed round trips at mean -0.852% against
# 0.465% cost, gross -1.4379 -- and these tests went red with no code change.
# A test whose verdict moves when the bots trade is not testing the code.
#
# The ban is CORRECT and stays: services.symbol_edge_gate.refusal_reason
# still returns it, and the gate's own coverage lives with the gate. Pinned
# here is only this file's precondition -- that the entry is reached at all.
@pytest.fixture(autouse=True)
def _entry_reaches_the_position_book(monkeypatch):
    monkeypatch.setattr(
        "trading.bot.symbol_edge_refusal", lambda _symbol, _strategy_id=None: None
    )
