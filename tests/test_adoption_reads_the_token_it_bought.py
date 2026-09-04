"""Orphan adoption must read the balance of the token it BOUGHT.

A round trip shares ONE trade_id across both of its settled swaps, and
``db.fetch_trades`` returns newest first. ``_adopt_orphaned_live_holding``
matched the settled row on that trade_id, so it broke on the SELL -- and a
sell's ``buy`` field is the QUOTE token.

Measured on-chain 2026-09-04, wallet 0x291c854811e92906a658Fb94Aa511bF919f968ad
on base, trade 2:CBETH-USDC:44c3665bffdf4861a56112da28ead2c3:

    23:15 BUY   sell=0x833589fcd6edb6e08f4c7c32d4f71b54bda02913 (USDC)
                buy =0x2ae3f1ec7f1f5012cfeab0185bfc7aa3cf0dec22 (CBETH)
                tx 0x5de159efe0d946b683c08f00773c43fbd7e0803f45953f5e5ab670f8af7d5077
    00:17 SELL  sell=0x2ae3f1ec7f1f5012cfeab0185bfc7aa3cf0dec22 (CBETH)
                buy =0x833589fcd6edb6e08f4c7c32d4f71b54bda02913 (USDC)
                tx 0x4ca1a606eb33ef24df951f15177532a2d9803554082ddb4c8677cc5d9bbc7e2d

    balanceOf(CBETH, wallet) = 0x0                 -> 0 CBETH
    balanceOf(USDC,  wallet) = 0x010bad5a          -> 17.542490 USDC

The scan broke on the 00:17 row, adopted CBETH at USDC's address, read the
STABLE LEG as the position size and booked 17.54249 CBETH at 2861.26 -- a
$50,193 position in a token the wallet holds none of. ``_position_is_real_on_chain``
resolves by symbol, read the right contract, and dropped it; the next pass
re-adopted it. Seven adoptions against eight drops in three hours, and while it
stood every CBETH entry was refused ``symbol_already_held_by_same_strategy``.

The exit path is why this is capital and not just noise: the adopted position
carries ``base_token_address``, and ``_execute_decision`` sizes a live sell from
it. An exit on the phantom would have read 17.542490 from the stable leg and
sold the entire book as though it were cbETH.

The entry's own tx_hash is the BUY transaction. It identifies exactly one
settled row, and the sell never carries it.
"""

from __future__ import annotations

import pytest

from trading.bot import TradingBot

SYMBOL = "CBETH-USDC"
CBETH = "0x2ae3f1ec7f1f5012cfeab0185bfc7aa3cf0dec22"
USDC = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"

BUY_TX = "0x5de159efe0d946b683c08f00773c43fbd7e0803f45953f5e5ab670f8af7d5077"
SELL_TX = "0x4ca1a606eb33ef24df951f15177532a2d9803554082ddb4c8677cc5d9bbc7e2d"
TRADE_ID = "2:CBETH-USDC:44c3665bffdf4861a56112da28ead2c3"

# The two balances, exactly as the chain reported them.
CBETH_RAW, CBETH_DECIMALS = 0, 18
USDC_RAW, USDC_DECIMALS = 17542490, 6

MARK = 2861.2608972460434


class _Stub:
    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


def _entry_row(ts=1788491703.8893654):
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
            "size": 0.000262122199594547,
            "quote_spent": 0.75,
            "entry_price": MARK,
            "entry_ts": ts,
            "timestamp": ts,
            "tx_hash": BUY_TX,
            "trade_id": TRADE_ID,
            "strategy_id": "atf_static",
            "route": ["CBETH", "USDC"],
            "gas_spent_native": 1.26257655e-06,
        },
    }


def _settled(ts, tx, buy, sell):
    return {
        "ts": ts,
        "wallet": "live",
        "chain": "base",
        "symbol": SYMBOL,
        "action": "swap",
        "status": "live-swap-settled",
        "details": {
            "tx_hash": tx,
            # BOTH swaps of the round trip carry the SAME trade_id. That is the
            # whole trap: it does not identify which side this row is.
            "trade_id": TRADE_ID,
            "buy": buy,
            "sell": sell,
            "confirmed": True,
            "ok": True,
        },
    }


class _DB(_Stub):
    """Serves trading_ops the way db.fetch_trades does: NEWEST FIRST."""

    def __init__(self):
        self.logged: list = []
        self._rows = [_entry_row()]
        self._settled = [
            _settled(1788495452.0, SELL_TX, USDC, CBETH),   # newest: the SELL
            _settled(1788491703.0, BUY_TX, CBETH, USDC),    # the BUY
        ]

    def fetch_trades(self, *, limit=200, statuses=None, wallets=None,
                     symbol=None, since_ts=None):
        wanted = set(statuses or [])
        if "live-swap-settled" in wanted:
            return list(self._settled)[:limit]
        return [r for r in self._rows if r["status"] in wanted][:limit]

    def log_trade(self, **kwargs):
        self.logged.append(kwargs)
        return True


class _Swapper:
    """Answers balanceOf per contract, with the two real readings."""

    def __init__(self):
        self.asked: list = []

    def token_balance_raw(self, chain, token):
        self.asked.append(str(token).lower())
        if str(token).lower() == USDC:
            return (USDC_RAW, USDC_DECIMALS)
        if str(token).lower() == CBETH:
            return (CBETH_RAW, CBETH_DECIMALS)
        raise AssertionError(f"balance asked about an unexpected contract: {token}")


def _bot(swapper):
    bot = TradingBot.__new__(TradingBot)
    TradingBot._ensure_runtime_state(bot)
    bot.db = _DB()
    bot.metrics = _Stub()
    bot.positions = {}
    bot.bus_routes = {}
    bot.primary_chain = "base"
    bot.stable_tokens = {"USDC"}
    bot.live_trading_enabled = True
    bot.ghost_session_id = 2
    bot._bridge = _Stub()
    bot._save_state = lambda: None
    bot._claim_position_symbol = lambda sym: None
    bot._init_bridge = lambda: _Stub()
    # The real resolver: an explicit address wins, otherwise the ticker maps to
    # its contract. This is what production does.
    def _resolve(chain, sym, explicit=None):
        if explicit:
            return sym, str(explicit)
        return sym, {"CBETH": CBETH, "USDC": USDC}.get(str(sym).upper(), "")

    bot._resolve_live_trade_asset = _resolve
    bot._verified_address = lambda chain, sym, addr, source: addr
    bot._new_swapper = lambda: swapper
    return bot


@pytest.fixture(autouse=True)
def _live_execution(monkeypatch):
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "0")
    monkeypatch.delenv("EXIT_DUST_SWEEP_USD", raising=False)
    monkeypatch.setenv("WALLET_DUST_USD", "0.50")


def test_the_stable_leg_is_never_adopted_as_a_position():
    """The wallet holds 0 CBETH. Nothing may be booked."""
    swapper = _Swapper()
    bot = _bot(swapper)

    pos = bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK)

    assert pos is None, (
        "adopted a CBETH position while balanceOf(CBETH) is 0 -- "
        f"it read {swapper.asked}"
    )
    assert bot.positions == {}


def test_the_balance_is_read_from_the_token_that_was_bought():
    """Never from the quote token, whose balance is the wallet's cash."""
    swapper = _Swapper()
    bot = _bot(swapper)

    bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK)

    assert USDC not in swapper.asked, (
        "asked the STABLE LEG how much CBETH we hold; a position sized from "
        "that answer sells the whole book on its first exit"
    )
    assert swapper.asked == [CBETH]


def test_no_position_claims_the_quote_tokens_address():
    """base_token_address drives the live sell, so it must never be USDC."""
    swapper = _Swapper()
    bot = _bot(swapper)

    bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK)

    for sym, pos in bot.positions.items():
        assert str(pos.get("base_token_address", "")).lower() != USDC, (
            f"{sym} would size its exit from the stable leg"
        )


def test_a_settled_buy_with_real_tokens_behind_it_is_still_adopted():
    """The fix must not stop adoption working -- only stop it reading USDC."""
    swapper = _Swapper()
    bot = _bot(swapper)
    # Same round trip, but the wallet really does still hold the cbETH.
    held_raw = 262122199594547          # 0.000262122199594547 at 18 decimals

    def _balance(chain, token):
        swapper.asked.append(str(token).lower())
        if str(token).lower() == CBETH:
            return (held_raw, CBETH_DECIMALS)
        if str(token).lower() == USDC:
            return (USDC_RAW, USDC_DECIMALS)
        raise AssertionError(token)

    swapper.token_balance_raw = _balance

    pos = bot._adopt_orphaned_live_holding(SYMBOL, chain="base", price=MARK)

    assert pos is not None, "a genuinely held position was left outside the book"
    assert pos["base_token_address"].lower() == CBETH
    assert pos["size"] == pytest.approx(0.000262122199594547, rel=1e-12)
