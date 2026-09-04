"""Gas is paid in ETH, so it must be valued at ETH's price -- not the pair's.

Measured on-chain 2026-09-04. A CBBTC-USDC exit
(0x340034967fb6145db692f039f7a95108295249aa2f0f2f987d17aa69e061705e) closed a
$3.00 position built from four 0.75 USDC entries. It burned 5.11e-06 ETH of
gas across the five transactions -- about 1.3 cents. The books recorded:

    gross_profit  -0.004102     (a -0.137% price move on $3.00: correct)
    fee_cost       0.4135511    (13.8% of the position)
    net_profit    -0.41765310

0.4135511 / 5.11e-06 = 80,884 -- the price of CBBTC, not of ETH. That single
fabricated loss was 38x the sum of every live win the system had produced
(+0.011014), which is precisely the condition the standing orders name as a
broken mechanism.

Two independent defects at one boundary produced it, both in
``TradingBot._estimate_native_price``:

  1. ``price_candidate`` was seeded from ``price`` -- the traded pair's price --
     before the native lookup, and nothing reset it, so a failed lookup left
     the pair's price standing in as the gas token's price.
  2. The lookup could not succeed anyway. ``db.fetch_price`` returns a
     ``sqlite3.Row``, which has no ``.get``; ``row.get("usd")`` raised
     AttributeError into a bare ``except Exception: pass``. And every row in
     ``prices`` is written under chain ``global`` (37 rows, measured
     2026-09-04), so ``fetch_price("base", "ETH")`` returns None regardless.

Defect 2 is what kept defect 1 permanently live: the lookup never once
overwrote the seed.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.bot import TradingBot  # noqa: E402

#: The global consensus ETH price at the time of the measurement.
ETH_USD = 2498.77748000803
#: The CBBTC price the gas was wrongly charged at.
CBBTC_USD = 80884.33540037746
#: Gas actually burned across the four entries and the exit, from the receipts.
CBBTC_ROUND_TRIP_GAS_ETH = (
    9.9197711e-07 + 1.083233711378e-06 + 9.1349965e-07 + 9.0961728e-07
    + 1.21454e-06
)


def _row(chain: str, token: str, usd) -> sqlite3.Row:
    """A real sqlite3.Row, usd stored as TEXT, exactly as production does."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE prices (chain TEXT, token TEXT, usd TEXT, source TEXT, ts REAL)")
    conn.execute(
        "INSERT INTO prices VALUES (?,?,?,?,?)",
        (chain, token, None if usd is None else str(usd), "consensus", time.time()),
    )
    return conn.execute("SELECT * FROM prices").fetchone()


class _PriceBook:
    """Mirrors the real store: everything under chain ``global``, usd as TEXT."""

    def __init__(self, rows=None):
        self.rows = rows if rows is not None else {("global", "eth"): ETH_USD}
        self.asked = []

    def fetch_price(self, chain, token):
        key = (str(chain).lower(), str(token).lower())
        self.asked.append(key)
        if key not in self.rows:
            return None
        return _row(key[0], key[1], self.rows[key])


class _EmptyBook(_PriceBook):
    def __init__(self):
        super().__init__(rows={})


def _bot(book) -> TradingBot:
    bot = TradingBot.__new__(TradingBot)
    bot.db = book
    bot.stable_tokens = {"USDC", "USDT", "DAI"}
    return bot


# --------------------------------------------------------------------------
# The row contract: sqlite3.Row, TEXT usd, global-only chain
# --------------------------------------------------------------------------

def test_a_sqlite_row_is_read_not_swallowed():
    """``.get`` on a sqlite3.Row raises; the helper must index it instead."""
    row = _row("global", "eth", ETH_USD)
    assert not hasattr(row, "get"), "fixture no longer reproduces the real type"
    assert isinstance(row["usd"], str), "usd is TEXT in the real schema"
    assert TradingBot._price_row_usd(row) == pytest.approx(ETH_USD)


@pytest.mark.parametrize("bad", [None, {"usd": None}, {"usd": ""}, {"usd": "nan"},
                                 {"usd": "not-a-number"}, {"usd": "-3"}, {"usd": "0"}])
def test_an_unusable_price_row_reads_as_zero_not_as_garbage(bad):
    assert TradingBot._price_row_usd(bad) == 0.0


def test_the_price_book_is_consulted_under_global():
    """Nothing is stored per-chain, so a per-chain-only lookup finds nothing."""
    book = _PriceBook()
    bot = _bot(book)
    assert bot._lookup_usd_price("base", "ETH") == pytest.approx(ETH_USD)
    assert ("base", "eth") in book.asked, "per-chain must still be tried first"
    assert ("global", "eth") in book.asked, "and global must be the fallback"


# --------------------------------------------------------------------------
# The defect itself
# --------------------------------------------------------------------------

def test_gas_on_a_cbbtc_route_is_priced_in_eth_not_in_cbbtc():
    bot = _bot(_PriceBook())
    native = bot._estimate_native_price("base", ["CBBTC", "USDC"], CBBTC_USD, "CBBTC-USDC")
    assert native == pytest.approx(ETH_USD)

    fee = CBBTC_ROUND_TRIP_GAS_ETH * native
    assert fee == pytest.approx(0.01278, abs=1e-4)
    # The number that was actually booked, and must never be booked again.
    assert fee < 0.02, f"gas on a $3.00 position must be cents, got {fee}"


def test_the_traded_price_never_leaks_in_when_the_lookup_fails():
    """The seed was the whole defect: a failed lookup left the pair's price."""
    bot = _bot(_EmptyBook())
    native = bot._estimate_native_price("base", ["CBBTC", "USDC"], CBBTC_USD, "CBBTC-USDC")
    assert native != pytest.approx(CBBTC_USD)
    assert native < 10_000.0, "an ETH-shaped fallback, not a BTC-shaped one"


def test_a_native_quoted_route_may_still_use_the_traded_price():
    """ETH-USDC really does price ETH; that shortcut must survive the fix."""
    bot = _bot(_EmptyBook())
    assert bot._estimate_native_price("base", ["ETH", "USDC"], 2500.0, "ETH-USDC") == 2500.0


def test_native_midway_through_a_route_does_not_price_the_route():
    """WETH as an intermediate hop does not make ``price`` the ETH price."""
    bot = _bot(_PriceBook())
    native = bot._estimate_native_price("base", ["CBBTC", "WETH", "USDC"], CBBTC_USD, "CBBTC-USDC")
    assert native == pytest.approx(ETH_USD)


# --------------------------------------------------------------------------
# Token pricing, same boundary
# --------------------------------------------------------------------------

def test_an_intermediate_hop_is_not_valued_at_the_pairs_price():
    bot = _bot(_PriceBook({("global", "weth"): 2496.24}))
    got = bot._estimate_token_price("base", "WETH", route=["CBBTC", "WETH", "USDC"], price=CBBTC_USD)
    assert got == pytest.approx(2496.24)


def test_the_first_leg_of_a_stable_quoted_route_is_the_pairs_price():
    bot = _bot(_EmptyBook())
    got = bot._estimate_token_price("base", "CBBTC", route=["CBBTC", "USDC"], price=CBBTC_USD)
    assert got == pytest.approx(CBBTC_USD)


def test_an_unknown_token_prices_at_zero_rather_than_at_the_pairs_price():
    bot = _bot(_EmptyBook())
    assert bot._estimate_token_price("base", "NOSUCHTOKEN", route=["CBBTC", "USDC"],
                                     price=CBBTC_USD) == 0.0


# --------------------------------------------------------------------------
# The fallback the fix newly made reachable
# --------------------------------------------------------------------------

def test_an_eth_shaped_fallback_is_not_handed_to_a_non_eth_chain(monkeypatch):
    """FALLBACK_NATIVE_PRICE is 1800.0 -- an ETH number. MATIC is ~$0.50.

    While the seed leaked, this branch was unreachable. It is reachable now,
    so it must not value MATIC at an ETH price.
    """
    monkeypatch.delenv("FALLBACK_NATIVE_PRICE_POLYGON", raising=False)
    monkeypatch.delenv("FALLBACK_NATIVE_PRICE", raising=False)
    bot = _bot(_EmptyBook())
    assert bot._estimate_native_price("polygon", ["AERO", "USDC"], 0.5017, "AERO-USDC") != 1800.0

    monkeypatch.setenv("FALLBACK_NATIVE_PRICE_POLYGON", "0.51")
    assert bot._estimate_native_price("polygon", ["AERO", "USDC"], 0.5017,
                                      "AERO-USDC") == pytest.approx(0.51)


def test_an_eth_chain_still_gets_the_eth_fallback(monkeypatch):
    monkeypatch.delenv("FALLBACK_NATIVE_PRICE_BASE", raising=False)
    monkeypatch.setenv("FALLBACK_NATIVE_PRICE", "1800.0")
    bot = _bot(_EmptyBook())
    assert bot._estimate_native_price("base", ["AERO", "USDC"], 0.5017,
                                      "AERO-USDC") == pytest.approx(1800.0)


def test_a_malformed_env_fallback_does_not_raise(monkeypatch):
    monkeypatch.setenv("FALLBACK_NATIVE_PRICE_BASE", "not-a-number")
    bot = _bot(_EmptyBook())
    got = bot._estimate_native_price("base", ["AERO", "USDC"], 0.5017, "AERO-USDC")
    assert got > 0.0 and got != pytest.approx(0.5017)
