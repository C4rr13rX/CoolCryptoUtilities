"""The GeckoTerminal branch must verify the token, rank by depth, and poll peers.

Root cause of the frozen feed, measured against the live API on 2026-09-02.
The branch was ``data[0]`` and nothing else -- no check that the pool held the
token being asked for, no liquidity ranking, no peer consensus, all three of
which the dexscreener branch beside it already did.

``search/pools?query=ARB%2FUSDC&network=base`` returns, in order:

    [0] ARBME / USDC 1%      5.33391272621165e-07    reserve $0.75
    [1] ARB / USDC 1%        2.28907708170849e-07    reserve $55.83
    [2] Arbase / USDC 1%     3.06962909444816e-05    reserve $3411.81

``market_stream`` held ``5.33391272621165e-07`` for ARB-USDC -- index [0], a
different token, out of a pool holding seventy-five cents -- unchanged from
2026-08-26 to 2026-09-02, stamped ``consensus_confidence: 1.0``.

Seven of the twenty symbols with enough ticks to score were frozen this way.
That is 18.75% of money_button's declines, and SPACEX-USDC (frozen at
1.52588956496421e-09, which is ``data[0]`` of 20 pools spanning 4.0e-9 to
4.1e-7) was one of only two symbols the live path was proposing real entries
on at the time.
"""
from __future__ import annotations

from trading.data_stream import _extract_rest_price, _gecko_pool_symbols


def _pool(name, price_usd, reserve_usd):
    return {
        "attributes": {
            "name": name,
            "base_token_price_usd": price_usd,
            "reserve_in_usd": reserve_usd,
        }
    }


# The three pools the live API returns for ARB/USDC on base, in API order.
_ARB_PAYLOAD = {
    "data": [
        _pool("ARBME / USDC 1%", "0.000000533391272621165", "0.7497"),
        _pool("ARB / USDC 1%", "0.000000228907708170849", "55.8301"),
        _pool("Arbase / USDC 1%", "0.0000306962909444816", "3411.8148"),
    ]
}

# The deepest five AERO/USDC pools, which agree with each other to 4 decimals.
_AERO_PAYLOAD = {
    "data": [
        _pool("AERO / USDC", "0.463131230938935", "28317945.9862"),
        _pool("AERO / USDC 1%", "0.463156275932694", "3205326.3316"),
        _pool("AERO / USDC 0.05%", "0.463259417741673", "539612.6502"),
        _pool("AERO / USDC 0.05%", "0.463507558180901", "188959.5526"),
        _pool("AERO / USDC 0.3%", "0.46322449642602", "107797.0429"),
    ]
}


def test_a_pool_holding_a_different_token_is_not_a_price_for_this_one() -> None:
    """ARBME is not ARB, however cheap it is and however early it is listed."""
    price = _extract_rest_price("geckoterminal", _ARB_PAYLOAD, "ARB", "USDC")

    assert price != 5.33391272621165e-07, "published ARBME's price as ARB for 7 days"
    # The one pool that actually holds ARB, even though it is neither first
    # in the list nor the deepest of the three.
    assert price == 2.28907708170849e-07


def test_list_position_does_not_outrank_liquidity() -> None:
    """``data[0]`` chose a $0.75 pool over a $3,411 one. Depth decides."""
    payload = {
        "data": [
            _pool("AERO / USDC 1%", "9.99", "0.75"),      # first, and near-empty
            _pool("AERO / USDC", "0.4631", "28317945.98"),
            _pool("AERO / USDC 0.05%", "0.4633", "539612.65"),
        ]
    }

    assert _extract_rest_price("geckoterminal", payload, "AERO", "USDC") == 0.4631


def test_the_deepest_pool_still_has_to_agree_with_its_peers() -> None:
    """Self-reported depth does not get the last word.

    A pancakeswap pool claimed $117M of liquidity on cbXRP/USDC at 0.001177
    while three other DEXes quoted 1.41. The dexscreener branch already
    refused that; geckoterminal now refuses it too, from the same code.
    """
    payload = {
        "data": [
            _pool("CBXRP / USDC", "0.001177", "117000000"),   # deepest, and absurd
            _pool("CBXRP / USDC 1%", "1.41", "250000"),
            _pool("CBXRP / USDC 0.3%", "1.40", "180000"),
            _pool("CBXRP / USDC 0.05%", "1.42", "90000"),
        ]
    }

    assert _extract_rest_price("geckoterminal", payload, "CBXRP", "USDC") == 1.41


def test_a_healthy_pair_resolves_to_the_deepest_pool() -> None:
    """The fix has to keep the good prices, or it is just an outage.

    AERO's five deepest pools agree to four decimals; the winner is the
    $28.3M one, and 0.4631 is AERO's real quote.
    """
    price = _extract_rest_price("geckoterminal", _AERO_PAYLOAD, "AERO", "USDC")

    assert price == 0.463131230938935


def test_no_verifiable_pool_means_no_price_rather_than_any_price() -> None:
    """A symbol with no pool of its own must go dark, not borrow a neighbour's.

    This is the whole failure in one assertion: the old branch always had a
    number to return because it never asked whose number it was.
    """
    payload = {
        "data": [
            _pool("SPACEXCLONE / USDC", "0.0000000015", "569.70"),
            _pool("SPCX / USDC 50%", "0.000000000027", "215.81"),
        ]
    }

    assert _extract_rest_price("geckoterminal", payload, "SPACEX", "USDC") is None


def test_an_empty_or_shapeless_response_is_not_a_price() -> None:
    for payload in ({}, {"data": []}, {"data": None}, {"data": [{}]}):
        assert _extract_rest_price("geckoterminal", payload, "AERO", "USDC") is None


def test_pool_names_carry_the_fee_tier_and_it_must_come_off() -> None:
    """``"SpaceX / USDC 50.1%"`` -- the quote is USDC, not "USDC 50.1%"."""
    assert _gecko_pool_symbols("ARBME / USDC 1%") == ("ARBME", "USDC")
    assert _gecko_pool_symbols("SpaceX / USDC 50.1%") == ("SPACEX", "USDC")
    assert _gecko_pool_symbols("AERO / USDC") == ("AERO", "USDC")
    # Not in that shape at all: unverifiable, which the caller treats as
    # unusable rather than guessing.
    assert _gecko_pool_symbols("") == ("", "")
    assert _gecko_pool_symbols("weird-name-no-slash") == ("", "")


def test_dexscreener_keeps_the_behaviour_it_already_had() -> None:
    """Both sources now share one selection rule; this pins the shared one.

    The MAMO case the dexscreener branch was written for: the same ticker on
    another chain, at 16x the price, always winning on depth.
    """
    payload = {
        "pairs": [
            {"chainId": "solana", "baseToken": {"symbol": "MAMO"},
             "quoteToken": {"symbol": "USDC"}, "priceUsd": "0.1723",
             "liquidity": {"usd": 171_000_000}},
            {"chainId": "base", "baseToken": {"symbol": "MAMO"},
             "quoteToken": {"symbol": "USDC"}, "priceUsd": "0.0103",
             "liquidity": {"usd": 363_000}},
        ]
    }

    price = _extract_rest_price("dexscreener", payload, "MAMO", "USDC", "base")

    assert price == 0.0103
