"""A take-profit row's overshoot is exit/TARGET, never exit/ENTRY.

THE BUG THIS PREVENTS. [c4f16946] asked for 12 ghost take-profit rows to be
annulled out of trade_outcomes AND data/strategy_ledger.json because they
"booked above their own target", and it picked them by RETURN: atf_static's
BSTONK rows at +25.35% and +17.28% were called +1.0201 of fabricated fills,
"66% of its book". atf_static is the only strategy with a live execution
branch and is judged by the re-arm rule on exactly that book, so annulling it
on a wrong predicate would have rewritten the number that decides whether real
money is armed.

Joined to the target each position was actually aiming at -- via the position
hash shared by trade_outcomes.trade_id and the action='enter' trading_ops row
-- those two rows are 1.019x and 1.074x their limits. Their returns are large
because their TARGETS were +22.96% and +9.22% above entry. A wide target is a
strategy decision; it is not the feed skipping past a narrow limit.

A return cannot distinguish the two, and that is the whole defect: BASECAT
+17.31% and atf_static BSTONK +17.28% are the same return and opposite
verdicts (1.105x vs 1.074x), because their targets were +6.17% and +9.22%.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "annul_ghost_gap_fills",
    Path(__file__).resolve().parents[1] / "scripts" / "annul_ghost_gap_fills.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MOD)

classify = _MOD.classify
GAP_FILL_RATIO = _MOD.GAP_FILL_RATIO


def _row(entry: float, return_pct: float, target_pct: float):
    """One closed take-profit row, described the way the census sees it."""
    return classify(entry, entry * (1.0 + return_pct / 100.0),
                    entry * (1.0 + target_pct / 100.0))


def test_two_rows_with_the_same_return_get_opposite_verdicts():
    """The measured pair that makes a return-based predicate indefensible.

    BASECAT +17.31% against a +6.17% target and atf_static's BSTONK +17.28%
    against a +9.22% target. Identical returns to within 0.03pp; one is 1.105x
    its limit and the other 1.074x. Any rule keyed on the return must call
    these the same, and they are not the same.
    """
    basecat = _row(0.041765565, 17.31, 6.17)
    bstonk = _row(0.0052, 17.28, 9.22)

    assert basecat["ratio"] == pytest.approx(1.105, abs=0.002)
    assert bstonk["ratio"] == pytest.approx(1.074, abs=0.002)
    assert basecat["gap_fill"] is True
    assert bstonk["gap_fill"] is False, (
        "atf_static's +17.28% row is 1.074x its own target -- inside the 1.10x "
        "bound. Annulling it would strike real ghost evidence from the book the "
        "re-arm rule reads for the ONLY strategy with a live branch."
    )
    assert abs(basecat["return_pct"] - bstonk["return_pct"]) < 0.05


def test_atf_statics_two_named_rows_are_not_gap_fills():
    """[c4f16946]'s +1.0201 of 'fabricated' fills, measured against their limits."""
    for entry, ret, tgt, want_ratio in (
        (0.0029880361, 25.35, 22.96, 1.019),
        (0.0052, 17.28, 9.22, 1.074),
    ):
        info = _row(entry, ret, tgt)
        assert info["ratio"] == pytest.approx(want_ratio, abs=0.002)
        assert info["gap_fill"] is False
        # They DID book past their limit, and the forward fix clamps them --
        # that is a different and much smaller claim than "fabricated".
        assert info["over_fee"] is True


def test_a_real_gap_fill_is_still_caught():
    """AERO's +160.99% row against a +5.00% target: 2.486x the limit."""
    info = _row(0.436805, 160.99, 5.00)
    assert info["ratio"] == pytest.approx(2.486, abs=0.002)
    assert info["gap_fill"] is True
    assert info["over_fee"] is True


def test_a_row_with_no_recoverable_target_is_unknown_not_clean():
    """UNI-USDC +122.89% has no target_price on its entry op.

    Reporting it as 'filled at its limit' would launder the worst row in the
    census; guessing its target from the return is the mistake above. It is
    UNKNOWN, and the census says so.
    """
    info = classify(2.859, 2.859 * 2.2289, None)
    assert info["known"] is False
    assert info["gap_fill"] is False
    assert info["over_fee"] is False
    assert info["ratio"] is None
    assert info["return_pct"] == pytest.approx(122.89, abs=0.01)


def test_a_fill_inside_one_legs_fee_is_slippage_not_an_overshoot():
    """The forward fix's own tolerance: target * (1 + fee_rate)."""
    info = classify(1.0, 1.0500, 1.05, fee_rate=0.003187)
    assert info["over_fee"] is False
    assert info["gap_fill"] is False
    over = classify(1.0, 1.0600, 1.05, fee_rate=0.003187)
    assert over["over_fee"] is True
