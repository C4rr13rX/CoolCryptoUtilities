"""A trade is about a CONTRACT, never about a ticker.

Measured 2026-09-02 over 20k ``ghost_candidate`` rows in the trading cache:
131 of 408 discovered base symbols resolved to more than one contract address.
1KTO100M mapped to 57 different contracts, ANTHROPIC to 67, SPCX to 31, with
prices under a single ticker spanning seven orders of magnitude -- provably
unrelated tokens sharing a name. Six symbols already in
``data/token_addresses.json`` disagreed with what discovery was publishing for
the same ticker, BASECAT by a factor of 153.

So resolving a live swap by symbol does not merely fail to find an address; it
can find the WRONG one and spend real funds on a ticker-squatting clone. These
tests pin the two halves of the defence:

  * the contract the candidate was priced against travels on the directive, and
  * an open position is exited through the contract it was actually opened in.
"""
from __future__ import annotations

import unittest

from trading.scheduler import TradeDirective
from trading.strategies.base import Strategy, StrategyContext


#: Two of the 57 real, distinct contracts discovery published under the single
#: ticker "1KTO100M" on base. Their quoted prices differed by four orders of
#: magnitude, so buying one and selling the other is not a rounding error.
CONTRACT_A = "0x38b58732f978cd58e8a0d02f9ca66c89d2f64e70"
CONTRACT_B = "0x86b1b5340f58a21771f55a3689ee72067971fe16"


class _Probe(Strategy):
    strategy_id = "probe"

    def evaluate(self, state, ctx):  # pragma: no cover - not exercised
        return None


class _State:
    symbol = "1KTO100M-USDC"
    base_token = "1KTO100M"
    quote_token = "USDC"
    samples: list = []


def _context() -> StrategyContext:
    return StrategyContext(
        chain="base",
        last_price=1.8e-06,
        last_volume=1000.0,
        fee_rate=0.0065,
        available_quote=5.0,
        available_base=0.0,
    )


def _candidate(address):
    return _Probe().make_candidate(
        _State(),
        _context(),
        action="enter",
        expected_return=0.05,
        target_price=1.9e-06,
        confidence=0.6,
        reason="probe",
        extra_meta={"token_address": address},
    )


class DirectiveCarriesContract(unittest.TestCase):
    def test_the_priced_contract_reaches_the_directive(self):
        """Without this the bot re-resolves the ticker and picks another token."""
        directive = _candidate(CONTRACT_A)["directive"]
        self.assertEqual(directive.token_address, CONTRACT_A)

    def test_it_survives_to_dict(self):
        """``bus_plan`` is directive.to_dict(); the audit row must show the contract."""
        plan = _candidate(CONTRACT_A)["directive"].to_dict()
        self.assertEqual(plan["token_address"], CONTRACT_A)

    def test_a_pool_id_is_not_a_token(self):
        """Discovery also stores 32-byte Uniswap v4 pool ids, which are not tokens."""
        directive = _candidate("0x" + "ab" * 32)["directive"]
        self.assertEqual(directive.token_address, "")

    def test_a_native_sentinel_is_not_a_token(self):
        directive = _candidate("0x" + "0" * 40)["directive"]
        self.assertEqual(directive.token_address, "")

    def test_absent_address_is_empty_string_not_none(self):
        """Consumers do ``or ""`` on this; None would still be falsey but the
        dataclass field is typed str and asdict() feeds JSON."""
        directive = _candidate(None)["directive"]
        self.assertIsInstance(directive.token_address, str)
        self.assertEqual(directive.token_address, "")


class ResolutionPrefersTheExplicitContract(unittest.TestCase):
    """``_resolve_live_trade_asset`` must never let a lookup override a caller
    that already knows which contract the trade is about."""

    def setUp(self):
        from trading.bot import TradingBot

        self.resolve = TradingBot._resolve_live_trade_asset.__get__(
            _FakeBot(), _FakeBot
        )

    def test_explicit_contract_wins_over_the_symbol_book(self):
        symbol, token = self.resolve("base", "1KTO100M", CONTRACT_A)
        self.assertEqual(token, CONTRACT_A)

    def test_a_different_contract_for_the_same_ticker_is_honoured(self):
        """The ticker is identical; only the caller's contract may decide."""
        _, first = self.resolve("base", "1KTO100M", CONTRACT_A)
        _, second = self.resolve("base", "1KTO100M", CONTRACT_B)
        self.assertEqual(first, CONTRACT_A)
        self.assertEqual(second, CONTRACT_B)
        self.assertNotEqual(first, second)

    def test_a_bad_explicit_address_does_not_become_the_swap_token(self):
        """A pool id must fall through to the normal lookup, not be swapped."""
        _, token = self.resolve("base", "1KTO100M", "0x" + "ab" * 32)
        self.assertIsNone(token)

    def test_an_unknown_ticker_still_blocks(self):
        """The refusal that protects the wallet stays in place when nothing
        names a contract -- this is the ``token_unresolved`` path."""
        _, token = self.resolve("base", "1KTO100M", None)
        self.assertIsNone(token)


class _FakeBot:
    """Minimal stand-in for the resolution path.

    ``_resolve_token_address`` is stubbed to "this ticker is not in any book",
    which is the real state of 1KTO100M and keeps the test off the production
    ``data/token_addresses.json``. That isolation is the point: what is being
    pinned is that an explicit contract bypasses the lookup entirely, and a bad
    one falls through to it rather than reaching a swap.
    """

    class _Portfolio:
        holdings: dict = {}

    portfolio = _Portfolio()

    def _resolve_token_address(self, chain, symbol):
        return None


if __name__ == "__main__":
    unittest.main()
