"""L1 stickiness must be SOLVED for a change rate, not set as a band fraction.

THE BUG THESE PIN. ``L1_HYSTERESIS_MARGIN`` is a fraction of each band's own
WIDTH and the widths are terciles fitted per stream per corpus, so the same
constant buys a different amount of stickiness on every pair: 0.50 measured
37.6% alphabet turnover on AERO-USDC and 62.3% on ARB-WETH, and L2 cleared the
0.30 identifier ceiling on the first and failed it on the second for exactly
that reason ([41ca68dc], pass 116). What has to be equal across corpora is the
TURNOVER the layer above sees.

Each test below is named for the failure it prevents, and each fails against the
behaviour that shipped before ``solve_hysteresis_margin`` existed -- there was
no solver at all, so a caller had one constant and no way to ask for a rate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.omen_layers import (  # noqa: E402
    L1_MARGIN_SEARCH_CEILING, L1_STREAMS, cooccurrence_motif, l1_change_rate,
    relative_bands, solve_hysteresis_margin, sticky_motifs,
)


def _corpus(count: int, period: int, offset: int = 0):
    """A synthetic stream whose slots cross their terciles on a fixed period.

    Synthetic on purpose: the point of these tests is the SOLVER's contract, and
    a stored corpus would make them a measurement of that corpus as well -- slow,
    and red for a reason that has nothing to do with the code under test. The
    real corpora are measured by scripts/omen_margin_travel.py, which is where a
    number belongs.
    """
    frames = []
    for i in range(count):
        step = (i + offset) % period
        level = int(19 * step / max(1, period - 1))
        frames.append({
            "geometry": "geo p=q%d" % level,
            "temporal": "tmp z=u%d" % (level // 2),
            "flow": "flw v=r%d" % level,
            "volatility": "vol v=u%d" % ((19 - level) // 2),
            "cross": "crs x=d%d" % (level // 3),
        })
    return frames


def test_a_solved_margin_beats_a_constant_at_equalising_turnover():
    """The headline: one constant gives two corpora two different change rates.

    This is the whole item in one assertion. Two corpora whose scores sit
    differently against their own terciles are handed the SAME margin and land
    on materially different turnover; solved to one target they land together.
    """
    fast, slow = _corpus(400, 3), _corpus(400, 40)
    bands_fast = relative_bands(fast, streams=L1_STREAMS)
    bands_slow = relative_bands(slow, streams=L1_STREAMS)

    fixed_fast = l1_change_rate(sticky_motifs(fast, bands_fast, 0.5))
    fixed_slow = l1_change_rate(sticky_motifs(slow, bands_slow, 0.5))
    fixed_spread = abs(fixed_fast - fixed_slow)

    target = 0.30
    solved_fast = solve_hysteresis_margin(fast, bands_fast, target)
    solved_slow = solve_hysteresis_margin(slow, bands_slow, target)
    solved_spread = abs(solved_fast["achieved"] - solved_slow["achieved"])

    assert fixed_spread > 0.05, (
        "the fixture must actually reproduce the defect: a constant margin has "
        "to give these two corpora different turnover, or the test proves "
        "nothing. fast=%.4f slow=%.4f" % (fixed_fast, fixed_slow))
    assert solved_spread < fixed_spread, (
        "solving did not close the turnover spread: fixed %.4f -> solved %.4f"
        % (fixed_spread, solved_spread))


def test_the_solver_returns_the_smallest_margin_that_reaches_the_target():
    """Stickiness is not free, so the solver must not buy more than asked.

    It coarsens L1 itself -- vocabulary 119 -> 49 on p108_aero_down at 0.50 --
    so a solver that returned any margin clearing the target would be free to
    return the search ceiling and destroy the layer while passing its contract.
    """
    frames = _corpus(400, 5)
    bands = relative_bands(frames, streams=L1_STREAMS)
    result = solve_hysteresis_margin(frames, bands, 0.40)
    assert result["reached"]
    assert result["achieved"] <= 0.40 + 1e-9

    # Step BACK below the solved margin and the rate must rise above the
    # target. That is what "smallest" means, and without it the assertion above
    # is satisfied by any sufficiently large number.
    lower = max(0.0, result["margin"] - 0.05)
    assert l1_change_rate(sticky_motifs(frames, bands, lower)) > 0.40


def test_a_target_already_met_costs_no_stickiness_at_all():
    """A corpus under the target must be handed margin 0, not a token amount.

    The failure this prevents is a solver that always returns something
    positive: a calm corpus would be coarsened for nothing, and the margin-0
    control arm -- the only arm byte-identical to plain relative banding --
    would stop being reachable through the solver at all.
    """
    frames = _corpus(400, 60)
    bands = relative_bands(frames, streams=L1_STREAMS)
    plain = l1_change_rate(sticky_motifs(frames, bands, 0.0))
    result = solve_hysteresis_margin(frames, bands, plain + 0.10)
    assert result["margin"] == 0.0
    assert result["reached"]
    assert result["iterations"] == 0


def test_an_unreachable_target_is_reported_not_silently_clamped():
    """A corpus that cannot be made sticky enough must SAY so.

    Returning the ceiling with ``reached`` true would hand a caller a margin
    that looks like a solution and an ``achieved`` rate nobody checks -- which
    is precisely how the fixed 0.50 travelled unchallenged for four passes.

    THE CASE IS NARROWER THAN IT LOOKS, and this fixture is the one that
    actually produces it. A large margin drives the rate toward zero for any
    stream the margin REACHES -- a slot whose hold band is wider than the band
    itself can never leave it -- so no amount of ordinary volatility is
    unreachable. What the margin cannot touch is a stream ``relative_bands``
    OMITTED: its terciles collapsed, so ``sticky_motifs`` falls back to absolute
    sign banding for that slot, and sign banding has no hysteresis in it at all.

    These frames are built to be exactly that: ``_numeric_of`` averages to the
    same 0.0 on every bar, so the terciles collapse and the stream is omitted,
    while ``_band_of``'s sign COUNT alternates hi/lo every bar. The motif then
    changes on 100% of adjacencies at every margin up to the search ceiling,
    which is a real defect shape and not a contrived one -- it is what a
    constant-magnitude stream like volatility looked like before ``relative_bands``
    existed.
    """
    hi = {name: "s a=u10 b=d10 c=u0" for name in L1_STREAMS}
    lo = {name: "s a=u10 b=d10 c=d0" for name in L1_STREAMS}
    frames = [dict(hi if i % 2 else lo) for i in range(200)]
    bands = relative_bands(frames, streams=L1_STREAMS)
    assert bands == {}, (
        "the fixture must actually collapse the terciles, or it is testing a "
        "stream the margin CAN reach and proves nothing: %r" % (bands,))

    result = solve_hysteresis_margin(frames, bands, 0.10)
    assert result["reached"] is False
    assert result["margin"] == L1_MARGIN_SEARCH_CEILING
    assert result["achieved"] == pytest.approx(1.0)


def test_solving_to_a_rate_never_changes_what_margin_zero_encodes():
    """The baseline cannot move underneath the solver.

    ``sticky_motifs(.., 0)`` is byte-identical to per-bar ``cooccurrence_motif``
    and the whole comparison arm rests on it, so adding a solver above it must
    not have touched it. Pinned here as well as in
    tests/test_hysteresis_margin_zero_is_byte_identical.py because THIS is the
    change that could have broken it.
    """
    frames = _corpus(300, 7)
    bands = relative_bands(frames, streams=L1_STREAMS)
    solve_hysteresis_margin(frames, bands, 0.25)      # must not mutate anything
    assert sticky_motifs(frames, bands, 0.0) == [
        cooccurrence_motif(f, bands=bands) for f in frames]


def test_the_change_rate_counts_adjacencies_not_bars():
    """``n`` bars have ``n-1`` adjacencies, and holes are not adjacencies.

    The obvious ``/n`` reads a fraction of a percent low, which is invisible
    until two agents compare a solved rate against a probe's rate and the
    numbers disagree by a rounding error nobody can locate.
    """
    assert l1_change_rate(["a", "a", "b", "b"]) == pytest.approx(1 / 3)
    assert l1_change_rate(["a"]) == 0.0
    assert l1_change_rate([]) == 0.0
    # A hole -- a bar whose L0 frames could not be built -- is dropped before
    # the adjacencies are counted, so no change is claimed across a gap.
    assert l1_change_rate(["a", "", "a"]) == 0.0
    assert l1_change_rate(["a", "", "b"]) == pytest.approx(1.0)
