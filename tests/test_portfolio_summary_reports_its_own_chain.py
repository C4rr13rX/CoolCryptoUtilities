"""PortfolioState.summary() must report the chain it is actually tracking.

`native_eth` and `stable_usd` were hardcoded to "ethereum" while the bot trades
Base (PRIMARY_CHAIN=base, LIVE_FOCUS_CHAIN=base). A Base-only portfolio was
therefore asked for its Ethereum balances and truthfully answered 0. Measured
2026-09-02 against wallet 0x291c...968ad, holding $6.98 USDC and $6.86 ETH on
Base, production printed:

    [portfolio] wallet=0x291c...968ad stable~0.00 native~0.0000 holdings=2

The line disagreed with itself -- `holdings` counted the real Base holdings while
the money came from a chain the wallet was never funded on. Nothing gates on
these two figures (every gating caller passes an explicit chain), but they are
what an operator and services/organism_state.py read to answer "is there money to
trade with", and they said no while the answer was yes.
"""

from __future__ import annotations

import unittest
from unittest import mock

from trading.portfolio import PortfolioState, TokenHolding


def _holding(chain: str, symbol: str, quantity: float, usd: float) -> TokenHolding:
    return TokenHolding(
        token=f"0x{symbol.lower():0<40}",
        symbol=symbol,
        quantity=quantity,
        usd=usd,
        chain=chain,
        raw={},
    )


class PortfolioSummaryReportsItsOwnChain(unittest.TestCase):
    def _portfolio(self, chains) -> PortfolioState:
        # get_db()/CacheBalances would open the production cache; the summary
        # path reads only the in-memory holdings this test sets directly.
        with mock.patch("trading.portfolio.get_db", return_value=object()), \
                mock.patch("trading.portfolio.CacheBalances", return_value=object()):
            return PortfolioState(wallet="0xabc", chains=chains)

    def test_base_wallet_is_not_reported_as_empty(self) -> None:
        """The exact production reading: $6.98 USDC + $6.86 ETH on Base."""
        pf = self._portfolio(("base",))
        pf.holdings = {
            ("base", "USDC"): _holding("base", "USDC", 6.977258, 6.977258),
            ("base", "ETH"): _holding("base", "ETH", 0.00285556048152249, 6.8623),
        }
        pf.native_balances = {"base": 0.00285556048152249}

        summary = pf.summary()

        self.assertEqual(summary["chain"], "base")
        self.assertAlmostEqual(summary["stable_usd"], 6.977258, places=6)
        self.assertAlmostEqual(summary["native_eth"], 0.00285556048152249, places=12)
        self.assertEqual(summary["holdings"], 2)

    def test_summary_no_longer_answers_for_ethereum(self) -> None:
        """A Base portfolio must not report an Ethereum balance as its own."""
        pf = self._portfolio(("base",))
        pf.holdings = {
            ("base", "USDC"): _holding("base", "USDC", 6.98, 6.98),
            # Funds on another chain must not leak into the Base answer.
            ("ethereum", "USDC"): _holding("ethereum", "USDC", 4242.0, 4242.0),
        }
        pf.native_balances = {"base": 0.0028, "ethereum": 9.9}

        summary = pf.summary()

        self.assertEqual(summary["chain"], "base")
        self.assertAlmostEqual(summary["stable_usd"], 6.98, places=6)
        self.assertNotAlmostEqual(summary["stable_usd"], 4242.0, places=2)
        self.assertAlmostEqual(summary["native_eth"], 0.0028, places=6)

    def test_summary_is_consistent_with_its_holdings_count(self) -> None:
        """holdings > 0 with a funded chain must not report zero money.

        This is the self-contradiction that made the log line unreadable.
        """
        pf = self._portfolio(("base",))
        pf.holdings = {
            ("base", "USDC"): _holding("base", "USDC", 6.977258, 6.977258),
            ("base", "ETH"): _holding("base", "ETH", 0.0028, 6.86),
        }
        pf.native_balances = {"base": 0.0028}

        summary = pf.summary()

        self.assertGreater(summary["holdings"], 0)
        self.assertGreater(
            summary["stable_usd"] + summary["native_eth"],
            0.0,
            "summary reports holdings but no money -- the reading that made a "
            "funded wallet look empty",
        )

    def test_an_ethereum_portfolio_still_reports_ethereum(self) -> None:
        """The fix follows the portfolio's chain rather than swapping one
        hardcoded chain for another."""
        pf = self._portfolio(("ethereum",))
        pf.holdings = {
            ("ethereum", "DAI"): _holding("ethereum", "DAI", 12.5, 12.5),
        }
        pf.native_balances = {"ethereum": 1.25}

        summary = pf.summary()

        self.assertEqual(summary["chain"], "ethereum")
        self.assertAlmostEqual(summary["stable_usd"], 12.5, places=6)
        self.assertAlmostEqual(summary["native_eth"], 1.25, places=6)


if __name__ == "__main__":
    unittest.main()
