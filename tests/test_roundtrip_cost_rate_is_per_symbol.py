"""A round trip on WETH does not cost what a round trip on COMP costs.

``services/roundtrip_cost.py`` publishes the number that DECIDES ENTRIES --
``trading/swap_schedule.py`` refuses any leg whose predicted return does not
clear it, ``services/atf_static_strategy.py`` refuses to book an exit inside
it, and ``trading/bot.py`` multiplies by it. On HEAD that number is::

    roundtrip_cost_rate(notional_usd: float | None = None) -> float

A curve of NOTIONAL ALONE. It takes no symbol, so it returns the same 38.6bp
at a $6 clip whether the leg is WETH or COMP. It is not an estimate that
happens to be imprecise; it is a constant that cannot vary in the one
dimension the cost actually varies in.

The measurement that says so is committed at
``data/round_trip_quote_census_20260917T190210Z.json`` (git 1a33c9a) -- nine
Base symbols quoted across a clip grid, cost as a percent of notional. At a
$2 clip:

    symbol     measured    flat gate    gate error
    WETH          2.01bp      38.6bp    19.2x TOO STRICT
    AERO         13.03bp      38.6bp     3.0x too strict
    CBBTC        14.79bp      38.6bp     2.6x too strict
    AAVE         60.26bp      38.6bp     1.6x too loose
    CBETH        67.12bp      38.6bp     1.7x too loose
    VIRTUAL      67.62bp      38.6bp     1.8x too loose
    VVV         212.47bp      38.6bp     5.5x too loose
    DRB        1507.62bp      38.6bp    39.1x too loose
    COMP       3301.53bp      38.6bp    85.5x TOO LOOSE

The gate is wrong in BOTH directions at once, which is why "tune the
constant" cannot fix it: any single number simultaneously refuses the one
symbol this account can trade at 2bp and admits the one it cannot trade at
all. The spread is a factor of 1,642 between the cheapest and the dearest.

These tests fail on HEAD -- ``roundtrip_cost_rate`` has no ``symbol``
parameter to give them, and with no symbol every one of the nine returns an
identical number by construction.

The second half of the file guards the OTHER defect this repo has already
shipped once (``round-trip-cost-measured-its-own-default``): a symbol the
census never priced must not quietly receive the constant and be
indistinguishable from a measured one. The fallback is allowed; being SILENT
about it is not.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.roundtrip_cost import (  # noqa: E402
    DEFAULT_FIXED_USD,
    DEFAULT_RATE,
    roundtrip_cost_rate,
)

#: The clip the census grid and the live plan share.
CLIP = 2.0

#: POOL cost (DEX fee + price impact) as a FRACTION of notional at ``CLIP``,
#: read straight out of the committed census, which stores PERCENT:
#: ``0.0201`` is 2.01bp is ``0.000201``.
POOL_AT_2USD = {
    "WETH": 0.0201 / 100.0,
    "AERO": 0.1303 / 100.0,
    "CBBTC": 0.1479 / 100.0,
    "AAVE": 0.60255 / 100.0,
    "CBETH": 0.6712 / 100.0,
    "VIRTUAL": 0.67615 / 100.0,
    "VVV": 2.1247 / 100.0,
    "DRB": 15.07625 / 100.0,
    "COMP": 33.01535 / 100.0,
}

#: GAS IS NOT IN THE CENSUS. The file says so outright -- the quoted rates are
#: pool cost only, and gas is "a dollar per leg, not a rate". It is also the
#: fixed leg the existing model already charges, and the same $0.004047 median
#: round trip the census recorded from live receipts. So the per-symbol total
#: is the census pool cost with the SAME fixed leg kept:
#:
#:     rate(n, sym) = FIXED/n + pool(sym, n)
#:
#: Dropping it instead would make every symbol look nearly free at a small
#: clip, where gas ALONE is 405bp at $0.10.
GAS_AT_2USD = DEFAULT_FIXED_USD / CLIP

#: What the gate should charge each symbol at a $2 clip, in bp:
#:   WETH 22.2 | AERO 33.3 | CBBTC 35.0 | [flat gate 52.1] | AAVE 80.5
#:   CBETH 87.4 | VIRTUAL 87.9 | VVV 232.7 | DRB 1527.9 | COMP 3321.8
TOTAL_AT_2USD = {sym: pool + GAS_AT_2USD for sym, pool in POOL_AT_2USD.items()}


def _census_path() -> Path:
    found = sorted((ROOT / "data").glob("round_trip_quote_census_*.json"))
    assert found, "no committed round-trip census to price symbols from"
    return found[-1]


class TestTheCostDependsOnTheSymbol:
    """The one assertion HEAD cannot satisfy: two symbols, two costs."""

    def test_weth_and_aero_do_not_cost_the_same(self) -> None:
        weth = roundtrip_cost_rate(CLIP, symbol="WETH")
        aero = roundtrip_cost_rate(CLIP, symbol="AERO")
        assert weth != aero, (
            f"WETH and AERO both priced at {weth:.6%} on a ${CLIP:.2f} clip. "
            "The census measures 2.01bp and 13.03bp -- a factor of 6.5. A "
            "cost that cannot tell them apart is a constant, not a cost."
        )

    def test_the_cheapest_the_middle_and_the_dearest_are_ordered(self) -> None:
        weth = roundtrip_cost_rate(CLIP, symbol="WETH")
        aero = roundtrip_cost_rate(CLIP, symbol="AERO")
        comp = roundtrip_cost_rate(CLIP, symbol="COMP")
        assert weth < aero < comp, (
            f"WETH {weth * 1e4:.2f}bp, AERO {aero * 1e4:.2f}bp, "
            f"COMP {comp * 1e4:.2f}bp -- measured 2.01 / 13.03 / 3301.53."
        )

    @pytest.mark.parametrize("symbol", sorted(POOL_AT_2USD))
    def test_every_census_symbol_is_priced_at_what_was_measured(self, symbol: str) -> None:
        from services.roundtrip_cost import roundtrip_cost_breakdown

        parts = roundtrip_cost_breakdown(CLIP, symbol)
        assert parts["pool_rate"] == pytest.approx(POOL_AT_2USD[symbol], rel=1e-9), (
            f"{symbol}: gate charges {parts['pool_rate'] * 1e4:.2f}bp of pool "
            f"cost, census measured {POOL_AT_2USD[symbol] * 1e4:.2f}bp at a "
            f"${CLIP:.2f} clip."
        )
        assert parts["gas_rate"] == pytest.approx(GAS_AT_2USD, rel=1e-12), (
            "the fixed gas leg must survive per-symbol pricing; the census "
            "does not carry it and dropping it would under-charge every symbol"
        )
        got = roundtrip_cost_rate(CLIP, symbol=symbol)
        assert got == pytest.approx(TOTAL_AT_2USD[symbol], rel=1e-9)

    def test_the_flat_gate_is_wrong_in_both_directions_at_once(self) -> None:
        """Why no single constant can be repaired into correctness."""
        flat = roundtrip_cost_rate(6.0)
        weth = roundtrip_cost_rate(6.0, symbol="WETH")
        comp = roundtrip_cost_rate(6.0, symbol="COMP")
        assert weth < flat < comp, (
            f"flat {flat * 1e4:.1f}bp must sit ABOVE the cheapest symbol it "
            f"over-charges ({weth * 1e4:.2f}bp) and BELOW the dearest one it "
            f"under-charges ({comp * 1e4:.1f}bp); a constant tuned to fix "
            "either one breaks the other."
        )
        # Measured at a $6 clip: WETH 8.79bp, flat 38.62bp, COMP 7237.62bp.
        assert flat / weth > 4.0
        assert comp / flat > 100.0

    def test_the_census_is_the_source_not_a_hand_typed_table(self) -> None:
        """Every number this test asserts is in the committed artifact."""
        census = json.loads(_census_path().read_text(encoding="utf-8"))
        quoted = census["quoted"]
        for symbol, want in POOL_AT_2USD.items():
            points = {float(clip): float(pct) for clip, pct in quoted[symbol]["points"]}
            assert points[CLIP] / 100.0 == pytest.approx(want, rel=1e-12), symbol
        # ...and the artifact itself says gas is not in those numbers, which
        # is why this module still adds the fixed leg.
        assert "gas" in " ".join(census["THESE_ARE_QUOTES_NOT_FILLS"]["not_modelled"])


class TestTheNotionalStillMatters:
    """Per-symbol must not throw away the size dependence it replaces."""

    @staticmethod
    def _pool(notional: float, symbol: str) -> float:
        from services.roundtrip_cost import roundtrip_cost_breakdown

        return float(roundtrip_cost_breakdown(notional, symbol)["pool_rate"])

    def test_a_symbol_is_priced_on_its_own_grid_not_the_flat_curve(self) -> None:
        """COMP's pool cost explodes with size; WETH's does not. Both are real."""
        assert self._pool(6.0, "COMP") > 10.0 * self._pool(0.1, "COMP")
        cheap = self._pool(0.1, "WETH")
        dear = self._pool(10.0, "WETH")
        assert abs(dear - cheap) < 0.0001, "WETH's own curve is nearly flat; do not invent slope"
        # The gas leg is the only thing that moves WETH's TOTAL with size, and
        # it moves it the other way -- down. That is the old model's one true
        # size effect, and it survives.
        assert roundtrip_cost_rate(0.1, symbol="WETH") > roundtrip_cost_rate(10.0, symbol="WETH")

    def test_cost_does_not_always_fall_with_size_and_must_not_be_collapsed(self) -> None:
        """The one test a "one bp figure per symbol" shortcut cannot pass.

        ``FIXED/n + RATE`` is monotonically DECREASING in ``n`` by
        construction, so the model being replaced could only ever describe a
        cost that falls with size. The measured curves do not agree:

            clip       $0.75    $2.00    $6.00   $10.00
            WETH        2.0bp    2.0bp    2.0bp    2.1bp   flat
            AERO       24.3bp   13.0bp   10.3bp      --    DECREASING
            VIRTUAL    31.7bp   67.6bp  180.9bp  291.6bp   INCREASING 9x

        VIRTUAL-USDC is streamed (332 ticks in 24h), so this is live. Price
        impact dominates its curve and a bigger clip is DEARER, not cheaper.
        Collapsing a symbol to a single representative figure -- as a cache
        key, a convenience accessor, or a test fixture -- would be wrong by up
        to 4.6x on VIRTUAL depending which clip it was sampled at, and would
        re-introduce the "one number per symbol" defect one level below the
        one this file exists to fix.
        """
        assert self._pool(6.0, "VIRTUAL") > 2.0 * self._pool(0.75, "VIRTUAL"), (
            "VIRTUAL's pool cost RISES with clip size (31.7bp -> 180.9bp); an "
            "accessor that returns one figure per symbol cannot express that"
        )
        assert self._pool(6.0, "AERO") < self._pool(0.75, "AERO"), (
            "AERO's pool cost FALLS with clip size (24.3bp -> 10.3bp)"
        )
        # The TOTAL carries the same shapes -- the gas leg falls with size on
        # both, and on VIRTUAL the pool still overwhelms it.
        assert roundtrip_cost_rate(6.0, symbol="VIRTUAL") > roundtrip_cost_rate(0.75, symbol="VIRTUAL")
        assert roundtrip_cost_rate(6.0, symbol="AERO") < roundtrip_cost_rate(0.75, symbol="AERO")
        # ...and the legacy curve can only ever fall, which is why it could
        # not have described VIRTUAL at any constant.
        assert roundtrip_cost_rate(6.0) < roundtrip_cost_rate(0.75)

    def test_between_two_grid_points_it_interpolates(self) -> None:
        """$3 is not on the grid. It must land strictly between $2 and $4."""
        low = self._pool(2.0, "COMP")
        mid = self._pool(3.0, "COMP")
        high = self._pool(4.0, "COMP")
        assert low < mid < high
        assert mid == pytest.approx((low + high) / 2.0, rel=1e-9)
        assert roundtrip_cost_rate(2.0, symbol="COMP") < roundtrip_cost_rate(3.0, symbol="COMP")

    def test_past_the_ends_of_the_grid_it_clamps(self) -> None:
        """No extrapolation. A $10,000 clip was never quoted; do not pretend."""
        assert self._pool(10_000.0, "COMP") == pytest.approx(self._pool(10.0, "COMP"), rel=1e-12)
        assert self._pool(0.0001, "COMP") == pytest.approx(self._pool(0.1, "COMP"), rel=1e-12)
        # AERO's grid stops at $6, not $10. Clamping is per symbol, not global.
        assert self._pool(50.0, "AERO") == pytest.approx(self._pool(6.0, "AERO"), rel=1e-12)


class TestAnUnpricedSymbolIsRecordedNotSilentlyDefaulted:
    """The defect this repo already shipped once, guarded against here.

    ``round-trip-cost-measured-its-own-default``: a cost that falls back to a
    constant and says nothing is read downstream as a measurement. Falling
    back is fine. Being indistinguishable from the census is not.
    """

    def test_the_source_of_every_number_is_reportable(self) -> None:
        from services.roundtrip_cost import roundtrip_cost_rate_detail

        rate, source = roundtrip_cost_rate_detail(CLIP, "WETH")
        assert source == "census"
        assert rate == pytest.approx(TOTAL_AT_2USD["WETH"], rel=1e-9)

        rate, source = roundtrip_cost_rate_detail(CLIP, "NOTATOKEN")
        assert source == "unpriced_default", (
            "an unmeasured symbol reported itself as measured -- this is the "
            "exact shape of the bug that let a default be read as a price"
        )
        assert rate == pytest.approx(roundtrip_cost_rate(CLIP), rel=1e-12)

    def test_no_symbol_at_all_is_also_unpriced(self) -> None:
        from services.roundtrip_cost import roundtrip_cost_rate_detail

        _, source = roundtrip_cost_rate_detail(CLIP, None)
        assert source == "unpriced_default"

    @pytest.mark.parametrize("symbol", ["ZORA", "LINK", "USDC", "DEGEN", "TOSHI"])
    def test_an_unpriced_symbol_is_unchanged_never_refused(self, symbol: str) -> None:
        """THE FALLBACK IS NOT A REFUSAL, AND MUST NEVER BE TIDIED INTO ONE.

        Measured against the tape (``storage/trading_cache.db`` market_stream,
        24h to 2026-09-17): 61 symbols streamed, 6,969 ticks. The census
        prices FOUR of them -- WETH-USDC, WETH-USDT, AERO-USDC, VIRTUAL-USDC
        -- 6.6% of symbols and 20.6% of ticks. Six of the nine census symbols
        (AAVE, CBBTC, CBETH, COMP, DRB, VVV) are not streamed at all.

        So 93.4% of the streamed universe and 79.4% of all ticks reach this
        path. Making it fail closed would restrict the universe to a twentieth
        of its symbols while reading, in the diff, as a cost fix. It returns
        TODAY'S NUMBER, byte for byte, and only says so out loud.
        """
        from services.roundtrip_cost import census_symbols, roundtrip_cost_rate_detail

        assert symbol not in census_symbols(), (
            f"{symbol} is now census-priced; pick a still-unpriced symbol for "
            "this guard, do not delete it"
        )
        for notional in (0.75, 2.0, 6.0):
            rate, source = roundtrip_cost_rate_detail(notional, symbol)
            assert source == "unpriced_default"
            assert rate == pytest.approx(roundtrip_cost_rate(notional), rel=1e-12), (
                f"{symbol} at ${notional}: an unpriced symbol must get exactly "
                "the number it gets today -- not a refusal, not a penalty"
            )
            assert rate > 0.0, "a zero cost admits everything; that is not a fallback"

    def test_the_unpriced_path_is_countable(self) -> None:
        from services.roundtrip_cost import roundtrip_cost_rate_detail, unpriced_symbols

        roundtrip_cost_rate_detail(CLIP, "SOMETHINGUNMEASURED")
        assert "SOMETHINGUNMEASURED" in unpriced_symbols()

    def test_the_census_can_be_switched_off_without_a_code_change(self) -> None:
        """One env var returns the entry gate to the behaviour now in production."""
        import services.roundtrip_cost as rc

        before = os.environ.get("ROUNDTRIP_COST_CENSUS")
        os.environ["ROUNDTRIP_COST_CENSUS"] = "0"
        try:
            rate, source = rc.roundtrip_cost_rate_detail(CLIP, "WETH")
            assert source == "unpriced_default"
            assert rate == pytest.approx(rc.roundtrip_cost_rate(CLIP), rel=1e-12)
        finally:
            if before is None:
                os.environ.pop("ROUNDTRIP_COST_CENSUS", None)
            else:
                os.environ["ROUNDTRIP_COST_CENSUS"] = before
        assert rc.roundtrip_cost_rate_detail(CLIP, "WETH")[1] == "census"

    def test_the_symbol_lookup_is_case_and_pair_tolerant(self) -> None:
        """Callers hold 'WETH-USDC' and 'weth'; the census holds 'WETH'.

        A lookup that missed on punctuation would quietly demote a MEASURED
        symbol to the default -- the same failure as never measuring it, with
        none of the noise.
        """
        want = TOTAL_AT_2USD["WETH"]
        for spelling in ("WETH", "weth", "WETH-USDC", "weth-usdc", " WETH ", "WETH/USDC"):
            assert roundtrip_cost_rate(CLIP, symbol=spelling) == pytest.approx(want, rel=1e-9), spelling


class TestTheLegacyCallIsUntouched:
    """Every existing caller keeps the number it has today."""

    def test_one_argument_returns_exactly_what_it_returned_before(self) -> None:
        fixed = 0.004047
        rate = 0.003187
        for notional in (0.75, 1.0, 2.0, 6.0, 20.0):
            assert roundtrip_cost_rate(notional) == pytest.approx(
                max((fixed + rate * notional) / notional, rate), rel=1e-12
            ), notional

    def test_the_floor_still_holds_at_a_large_notional(self) -> None:
        """The unpriced default keeps its floor; a measured symbol must not.

        The floor guards a number nobody measured from amortising the fixed
        leg toward an implausibly free round trip. On WETH, which somebody DID
        measure, the same floor would re-impose the 31.87bp over-charge this
        change exists to remove.
        """
        assert roundtrip_cost_rate(10_000.0) >= DEFAULT_RATE
        assert roundtrip_cost_rate(10_000.0) == pytest.approx(DEFAULT_RATE, rel=1e-3)
        assert roundtrip_cost_rate(10_000.0, symbol="WETH") < DEFAULT_RATE / 10.0

    def test_the_environment_override_still_reaches_the_unpriced_path(self) -> None:
        before = os.environ.get("ROUNDTRIP_FEE_RATE")
        os.environ["ROUNDTRIP_FEE_RATE"] = "0.01"
        try:
            assert roundtrip_cost_rate(6.0) >= 0.01 - 1e-12
            # ...and does NOT reach a symbol the census priced.
            assert roundtrip_cost_rate(6.0, symbol="WETH") < 0.001
        finally:
            if before is None:
                os.environ.pop("ROUNDTRIP_FEE_RATE", None)
            else:
                os.environ["ROUNDTRIP_FEE_RATE"] = before


class TestTheEntryGatesAskWithASymbol:
    """A per-symbol cost nothing passes a symbol to is still a constant."""

    def test_swap_schedule_prices_each_leg_on_its_own_symbol(self) -> None:
        import time

        from trading.swap_schedule import build_schedule

        now = time.time()
        # +0.30% predicted. Dearer than WETH's 2bp round trip, far cheaper
        # than COMP's 3301bp one. Under the flat 38.6bp gate BOTH are
        # admitted; under the measured one only WETH can be.
        cands = [
            {"symbol": sym, "label": f"{sym}-15m", "predicted_return": 0.0030,
             "resolve_ts": now + 900.0, "start_price": 1.0, "target_price": 1.003}
            for sym in ("WETH", "COMP")
        ]
        sched = build_schedule(
            cands, capital_usd=100.0, clip_usd=2.0,
            roundtrip_cost_rate=0.00386, now=now,
        )
        legs = {str(leg.symbol) for leg in sched.legs}
        assert "WETH" in legs, "a 0.30% call on a 0.02% round trip must be schedulable"
        assert "COMP" not in legs, (
            "a 0.30% call on a 33.0% round trip was scheduled -- the flat gate "
            "admitted it because it charged COMP what it charges WETH"
        )
        refusals = " ".join(str(r.get("reason", "")) for r in sched.rejected)
        assert "COMP" in " ".join(str(r.get("symbol", "")) for r in sched.rejected)
        assert "round trip" in refusals

    def test_atf_static_asks_its_cost_with_the_symbol_in_hand(self) -> None:
        """The exit gate at the profit floor is inside a per-symbol loop."""
        import inspect

        import services.atf_static_strategy as atf

        src = inspect.getsource(atf)
        assert "_roundtrip_cost_rate(clip_usd, symbol)" in src or \
               "_roundtrip_cost_rate(clip_usd, symbol=symbol)" in src, (
            "services/atf_static_strategy.py still prices its exit floor "
            "without the symbol it already holds in scope"
        )
