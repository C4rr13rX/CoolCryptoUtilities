"""A live buy that settles on chain must reach the book even if nothing books it.

A live entry writes TWO rows, minutes apart:

    live-swap-settled   written the instant the chain confirms, by the swapper,
                        carrying the 66-character tx_hash
    live-entry          written after the wallet resync and the receipt read,
                        by the booking path, carrying ``size``/``quote_spent``

Measured 2026-09-04 on CBBTC-USDC: tx
0xabcace1328c0463fc8b0f234c10cb56b59157ac794f638d20e2069f416410174 settled at
13:06:16 and was booked at 13:10:58 -- **4m42s** in which real money is gone
and nothing durable points at it.

Every recovery path in trading/bot.py used to reconstruct the wallet's
unmatched buys from the BOOKING rows alone, so a buy that settles and is never
booked is invisible to all of them. What that cost, on this wallet, that day:

    12:53:04  0xf1c6c076d50e94640396f8c2c8f12babf980faedb95b7ed84a2b5f22fe98a813
              OUT 0.874616 USDC, IN 1095 raw cbBTC
              never booked -- production restarted 12:56:01 inside the window
    13:06:16  0xabcace1328c0463fc8b0f234c10cb56b59157ac794f638d20e2069f416410174
              OUT 0.874616 USDC, IN 1094 raw cbBTC
              booked 13:10:58

``balanceOf`` for 0x291c854811e92906a658Fb94Aa511bF919f968ad confirmed 2189 raw
= 1095 + 1094: BOTH buys held, nothing sold, 1.749232 USDC spent against a book
that recorded a single 0.874616 position. Two consequences, and which one fires
depends only on the dust floor:

  * ``_size_live_exit`` sizes the sell as ``min(position_size, onchain)`` =
    1094 raw and leaves 1095 behind. That residual is worth $0.875 against an
    EXIT_DUST_SWEEP_USD of $0.50, so it is NOT swept -- and the exit then
    writes ``remaining_size: 0.0`` and drops the position, after which the
    settled sell hides the leftover from every recovery path there is. $0.875
    stranded: 5% of the book and six times the whole live P/L of +0.1423.

  * Had the floor been higher, the sweep would have sold all 2189 raw for
    ~1.749 USDC while ``allocation_ratio = min(1.0, base_sold / held_size)``
    capped the cost at the recorded 0.874616 -- booking **+0.874 gross, a
    fabricated 100% win** on a round trip that actually broke even.

And because the slot looked empty, the bot bought the same symbol twice
13 minutes apart: buys outrunning sells, which is how this wallet converts
stable coin into stranded tokens.

The numbers below are those transactions.
"""

from __future__ import annotations

import pytest

from trading.bot import TradingBot

SYMBOL = "CBBTC-USDC"
CBBTC = "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf"
USDC = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
DECIMALS = 8
WALLET = "0x291c854811e92906a658Fb94Aa511bF919f968ad"

# The two settled buys, verbatim from trading_ops and from their receipts.
TX_UNBOOKED = "0xf1c6c076d50e94640396f8c2c8f12babf980faedb95b7ed84a2b5f22fe98a813"
TX_BOOKED = "0xabcace1328c0463fc8b0f234c10cb56b59157ac794f638d20e2069f416410174"
TS_UNBOOKED = 1788540784.0                  # 12:53:04
TS_BOOKED = 1788541516.4620965              # 13:06:16

SPENT_EACH = 0.874616                       # USDC out, both transactions
BASE_UNBOOKED = 1095 / 10 ** DECIMALS       # 1.095e-05 cbBTC
BASE_BOOKED = 1094 / 10 ** DECIMALS         # 1.094e-05 cbBTC
ONCHAIN_RAW = 2189                          # balanceOf, read 2026-09-04 13:15
MARK = 79946.61791590494                    # the booked position's entry_price


class _Stub:
    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


class _Bridge(_Stub):
    def get_address(self):
        return WALLET


class _Fill:
    """The shape services/fill_receipt.ReceiptFill presents to the caller."""

    def __init__(self, *, ok=True, sold=0.0, bought=0.0, reason="", gas_native=0.0):
        self.ok = ok
        self.reason = reason
        self.sold = sold
        self.bought = bought
        self.gas_native = gas_native


def _settled_row(ts, tx, purpose="live_entry", amount="0.874616"):
    """A live-swap-settled row exactly as the swapper writes it.

    ``purpose`` and ``confirmed``/``ok`` are present and ``executed`` is NOT --
    verified against the stored JSON in storage/trading_cache.db, where every
    live-swap-settled row carries purpose <str> and confirmed=True <bool> and
    no live-swap-settled row has ever carried ``executed``.
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
            "trade_id": f"2:{SYMBOL}:{tx[-32:]}",
            "route": "UniswapV3",
            "confirmed": True,
            "ok": True,
            "reason": "",
            "purpose": purpose,
            "amount_human": amount,
            "sell": USDC if purpose == "live_entry" else CBBTC,
            "buy": CBBTC if purpose == "live_entry" else USDC,
            "symbol": SYMBOL,
            "strategy_id": "atf_static",
        },
    }


def _booking_row(ts, tx, size, spent):
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
            "entry_price": spent / size,
            "entry_ts": ts,
            "timestamp": ts,
            "tx_hash": tx,
            "trade_id": f"2:{SYMBOL}:{tx[-32:]}",
            "strategy_id": "atf_static",
            "route": ["CBBTC", "USDC"],
            "gas_spent_native": 9.35698968166e-07,
        },
    }


class _DB(_Stub):
    """Serves trading_ops the way db.fetch_trades does: NEWEST FIRST."""

    def __init__(self, *, booking=None, settled=None):
        self.logged: list = []
        self._rows = sorted(
            booking if booking is not None else [
                _booking_row(TS_BOOKED, TX_BOOKED, BASE_BOOKED, SPENT_EACH)
            ],
            key=lambda r: r["ts"],
            reverse=True,
        )
        self._settled = sorted(
            settled if settled is not None else [
                _settled_row(TS_UNBOOKED, TX_UNBOOKED),
                _settled_row(TS_BOOKED, TX_BOOKED),
            ],
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
    def __init__(self, raw=ONCHAIN_RAW, fills=None):
        self._reading = (raw, DECIMALS)
        self._fills = fills if fills is not None else {
            TX_UNBOOKED: _Fill(sold=SPENT_EACH, bought=BASE_UNBOOKED,
                               gas_native=9.3e-07),
            TX_BOOKED: _Fill(sold=SPENT_EACH, bought=BASE_BOOKED,
                             gas_native=9.35698968166e-07),
        }
        self.receipts_read: list = []

    def token_balance_raw(self, chain, token):
        assert token.lower() == CBBTC, f"aimed at the wrong contract: {token}"
        return self._reading

    def read_fill(self, chain, txh, *, sell, buy, wallet=None, **kwargs):
        # The receipt is per-transaction; the sell/buy it is read against are
        # the settled row's own, so a buy resolves USDC out / cbBTC in.
        assert sell.lower() == USDC and buy.lower() == CBBTC
        assert wallet == WALLET, "a fill measured against somebody else's transfers"
        self.receipts_read.append(txh)
        return self._fills.get(txh, _Fill(ok=False, reason="no_receipt"))


def _booked_position():
    """The position as the book actually held it at 13:15 on 2026-09-04."""
    return {
        "mode": "live",
        "strategy_id": "atf_static",
        "entry_price": MARK,
        "size": BASE_BOOKED,
        "ts": TS_BOOKED,
        "entry_ts": TS_BOOKED,
        "trade_id": f"2:{SYMBOL}:30795ab0f5c84b72899f341acb2e9711",
        "route": ["CBBTC", "USDC"],
        "quote_spent": SPENT_EACH,
        "gas_spent_native": 9.35698968166e-07,
        "entry_tx_hash": TX_BOOKED,
        "base_token_address": "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf",
        "quote_token_address": USDC,
        "trigger_state": {"high_watermark": MARK},
        "exit_sequence": 0,
    }


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
    bot._bridge = _Bridge()
    bot.saved = 0

    def _save():
        bot.saved += 1

    bot._save_state = _save
    bot._claim_position_symbol = lambda sym: None
    bot._init_bridge = lambda: _Bridge()
    bot._resolve_live_trade_asset = lambda chain, sym, explicit=None: (
        sym, explicit or (CBBTC if sym.startswith("CBBTC") else USDC)
    )
    bot._verified_address = lambda chain, sym, addr, source: addr
    bot._swapper = swapper if swapper is not None else _Swapper()
    bot._new_swapper = lambda: bot._swapper
    return bot


@pytest.fixture(autouse=True)
def _live_execution(monkeypatch):
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "0")
    monkeypatch.delenv("EXIT_DUST_SWEEP_USD", raising=False)
    monkeypatch.setenv("WALLET_DUST_USD", "0.50")


# ---------------------------------------------------------------------------
# The settled swap is the record
# ---------------------------------------------------------------------------

def test_a_settled_buy_with_no_booking_row_is_still_found():
    """The 12:53 buy exists only as live-swap-settled. It must not be invisible."""
    bot = _bot()

    unmatched = bot._unmatched_live_entry_details(SYMBOL, chain="base")

    hashes = [d["tx_hash"] for d in unmatched]
    assert hashes == [TX_BOOKED, TX_UNBOOKED], "newest first, and BOTH buys"
    assert all(len(h) == 66 for h in hashes)


def test_the_unbooked_buys_fill_is_read_from_its_own_receipt():
    """Never guessed, never taken from the wallet delta.

    ``amount_human`` on a settled BUY is the USDC that went out, not the base
    that came in -- the only place the base still exists is the receipt, and
    the receipt is per-transaction so a sibling bot's swap cannot pollute it.
    """
    bot = _bot()

    unmatched = bot._unmatched_live_entry_details(SYMBOL, chain="base")
    recovered = [d for d in unmatched if d["tx_hash"] == TX_UNBOOKED][0]

    assert recovered["size"] == pytest.approx(BASE_UNBOOKED, rel=1e-12)
    assert recovered["quote_spent"] == pytest.approx(SPENT_EACH)
    assert recovered["recovered_from_settled_swap"] is True
    # The buy that WAS booked keeps the fill the entry path measured, so no
    # receipt is paid for twice.
    assert bot._swapper.receipts_read == [TX_UNBOOKED]


def test_a_settled_sell_still_closes_everything_older():
    """The cutoff is the newest exit of EITHER kind, applied to BOTH scans."""
    bot = _bot(db=_DB(settled=[
        _settled_row(TS_UNBOOKED, TX_UNBOOKED),
        _settled_row(TS_BOOKED, TX_BOOKED),
        _settled_row(TS_BOOKED + 60.0, "0x" + "e" * 64,
                     purpose="live_exit", amount="0.00002189"),
    ]))

    assert bot._unmatched_live_entry_details(SYMBOL, chain="base") == []


# ---------------------------------------------------------------------------
# The position is grown to match the chain
# ---------------------------------------------------------------------------

def test_the_position_grows_to_cover_the_buy_it_did_not_account_for():
    bot = _bot()
    pos = _booked_position()
    bot.positions[SYMBOL] = pos

    out = bot._reconcile_live_position_against_settled(SYMBOL, chain="base", pos=pos)

    assert out is not None
    assert out["size"] == pytest.approx(ONCHAIN_RAW / 10 ** DECIMALS, rel=1e-12)
    assert out["size"] == pytest.approx(BASE_BOOKED + BASE_UNBOOKED, rel=1e-12)
    assert out["quote_spent"] == pytest.approx(2 * SPENT_EACH, rel=1e-12)
    assert out["quote_spent"] == pytest.approx(1.749232, rel=1e-12)
    assert out["entry_price"] == pytest.approx(1.749232 / (ONCHAIN_RAW / 1e8), rel=1e-12)
    assert TX_UNBOOKED.lower() in out["reconciled_from_tx_hashes"]
    assert bot.saved == 1, "a repair that is not saved dies with the process"
    assert "live-position-reconciled" in bot.db.statuses()


def test_the_clock_counts_from_when_the_money_actually_left():
    """max_hold must run from the 12:53 buy, not from the 13:06 one."""
    bot = _bot()
    pos = _booked_position()

    out = bot._reconcile_live_position_against_settled(SYMBOL, chain="base", pos=pos)

    assert out["entry_ts"] == pytest.approx(TS_UNBOOKED)
    assert out["ts"] == out["entry_ts"]


def test_reconciling_twice_does_not_double_count():
    bot = _bot()
    pos = _booked_position()

    first = bot._reconcile_live_position_against_settled(SYMBOL, chain="base", pos=pos)
    again = bot._reconcile_live_position_against_settled(SYMBOL, chain="base", pos=first)

    assert again["size"] == pytest.approx(ONCHAIN_RAW / 10 ** DECIMALS, rel=1e-12)
    assert again["quote_spent"] == pytest.approx(2 * SPENT_EACH, rel=1e-12)
    assert bot.saved == 1, "the second pass had nothing to repair"


# ---------------------------------------------------------------------------
# What the repair is FOR: the exit
# ---------------------------------------------------------------------------

def test_the_exit_sells_the_whole_holding_once_the_book_agrees():
    """AN EXIT MUST SELL THE WHOLE POSITION.

    Before the repair the position claims 1094 raw of a 2189 raw holding, so
    ``_size_live_exit`` sells 1094 and strands 1095. That residual is worth
    $0.875 against a $0.50 sweep floor, so the sweep does not rescue it, and
    the exit then drops the position -- the 12:53 buy is gone for good.
    """
    bot = _bot()
    swapper = bot._swapper
    stale = _booked_position()

    before = bot._size_live_exit(
        swapper, chain="base", token=CBBTC, symbol=SYMBOL,
        position_size=stale["size"], price=MARK,
    )
    assert before["exit_raw"] == 1094
    assert before["onchain_raw"] == ONCHAIN_RAW
    assert before["swept"] is False
    stranded = (ONCHAIN_RAW - before["exit_raw"]) / 10 ** DECIMALS * MARK
    assert stranded == pytest.approx(0.875, abs=0.005), "$0.875 left behind"

    repaired = bot._reconcile_live_position_against_settled(
        SYMBOL, chain="base", pos=stale
    )
    after = bot._size_live_exit(
        swapper, chain="base", token=CBBTC, symbol=SYMBOL,
        position_size=repaired["size"], price=MARK,
    )
    assert after["exit_raw"] == ONCHAIN_RAW, "the whole holding, nothing stranded"


def test_the_round_trip_is_priced_honestly_and_not_as_a_hundred_percent_win():
    """``allocation_ratio = min(1.0, base_sold / held_size)`` caps at 1.0.

    Selling 2189 raw against a book that recorded 1094 costs the sale at the
    single recorded 0.874616 and books +0.874 gross -- a fabricated 100% win,
    six times the entire live P/L, on a round trip that has not moved. Once
    the book agrees with the chain the same sale prices out at zero.
    """
    proceeds = 2 * SPENT_EACH               # sold back at the price it was bought

    stale = _booked_position()
    ratio = min(1.0, (ONCHAIN_RAW / 10 ** DECIMALS) / stale["size"])
    fabricated = proceeds - stale["quote_spent"] * ratio
    assert ratio == 1.0
    assert fabricated == pytest.approx(0.874616, rel=1e-9), "the fabricated win"

    bot = _bot()
    repaired = bot._reconcile_live_position_against_settled(
        SYMBOL, chain="base", pos=_booked_position()
    )
    ratio = min(1.0, (ONCHAIN_RAW / 10 ** DECIMALS) / repaired["size"])
    honest = proceeds - repaired["quote_spent"] * ratio
    assert ratio == pytest.approx(1.0)
    assert honest == pytest.approx(0.0, abs=1e-9), "it broke even, and says so"


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------

def test_an_unreadable_receipt_never_becomes_a_guessed_fill():
    """The money is gone and unmeasurable. Refuse, and say which transaction."""
    swapper = _Swapper(fills={
        TX_BOOKED: _Fill(sold=SPENT_EACH, bought=BASE_BOOKED),
        TX_UNBOOKED: _Fill(ok=False, reason="decimals_unknown"),
    })
    bot = _bot(swapper=swapper)
    pos = _booked_position()

    out = bot._reconcile_live_position_against_settled(SYMBOL, chain="base", pos=pos)

    assert out["size"] == pytest.approx(BASE_BOOKED, rel=1e-12), "not inflated"
    assert out["quote_spent"] == pytest.approx(SPENT_EACH)
    assert bot._unreconciled_settled_buy[SYMBOL] == TX_UNBOOKED
    assert bot.saved == 0


def test_the_flag_clears_once_the_receipt_can_be_read():
    """An RPC outage must not lock the symbol out permanently."""
    bot = _bot()
    bot._unreconciled_settled_buy[SYMBOL] = TX_UNBOOKED

    bot._unmatched_live_entry_details(SYMBOL, chain="base")

    assert SYMBOL not in bot._unreconciled_settled_buy


def test_the_chain_is_the_ceiling_on_what_a_position_may_claim():
    """Receipts prove what a buy delivered, not that we still hold it.

    A sell this book never saw would make the sum an overstatement, and a
    position larger than the wallet produces an exit that cannot fill.
    """
    bot = _bot(swapper=_Swapper(raw=1094))       # the 12:53 buy is no longer held
    pos = _booked_position()

    out = bot._reconcile_live_position_against_settled(SYMBOL, chain="base", pos=pos)

    assert out["size"] == pytest.approx(BASE_BOOKED, rel=1e-12)
    assert out["quote_spent"] == pytest.approx(SPENT_EACH)
    assert "live-position-reconciled" not in bot.db.statuses()


def test_a_ghost_position_is_never_touched():
    bot = _bot()
    ghost = dict(_booked_position(), mode="ghost")

    out = bot._reconcile_live_position_against_settled(SYMBOL, chain="base", pos=ghost)

    assert out is ghost
    assert out["size"] == pytest.approx(BASE_BOOKED, rel=1e-12)
    assert bot.saved == 0
