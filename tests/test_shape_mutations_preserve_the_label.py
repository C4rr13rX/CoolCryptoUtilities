"""A shape mutation that changes the label is poison.

THE FAILURE THIS PREVENTS. Training a mutated chart on the ORIGINAL label is
only sound while the mutation cannot move what the label is a function of.
``label_omen`` reads the entry close, the close ``horizon`` bars later, and the
min/max over the last ``RANGE_WINDOW`` bars. A mutation that touches any of
those keeps a label the future no longer supports, and teaches the fabric to
answer the same for genuinely different futures -- the exact opposite of the
invariance the mutation was added to buy.

These tests are written against the two mutations the item asked to be
justified or dropped, so the refusal is enforced rather than remembered:
amplitude scaling does NOT flip the label (which is why it looks safe) and
inversion flips it on most bars.
"""
from __future__ import annotations

import random

import pytest

from scripts.omen_shape_mutations import ADMITTED, MUTATIONS, mutated_sample
from trading.omen_brain import LOOKBACK_BARS, RANGE_WINDOW


def _bars(n=260, seed=11):
    rng = random.Random(seed)
    out = []
    price = 100.0
    for i in range(n):
        price = max(1.0, price * (1.0 + rng.gauss(0.0, 0.01)))
        high = price * (1.0 + abs(rng.gauss(0.0, 0.004)))
        low = price * (1.0 - abs(rng.gauss(0.0, 0.004)))
        out.append({
            "timestamp": 1_700_000_000 + i * 3600,
            "open": price * (1.0 + rng.gauss(0.0, 0.002)),
            "high": high, "low": low, "close": price,
            "volume": 1000.0 + rng.random() * 100.0,
        })
    return out


def _anchors(bars, horizon):
    return list(range(LOOKBACK_BARS, len(bars) - horizon))


@pytest.mark.parametrize("kind", ADMITTED)
def test_an_admitted_mutation_never_moves_the_label(kind):
    bars = _bars()
    horizon = 12
    rng = random.Random(3)
    checked = 0
    for index in _anchors(bars, horizon):
        sample = mutated_sample(bars, index, horizon=horizon,
                                symbol="TEST-USDC", chain="base", kind=kind,
                                rng=rng)
        if sample is None:
            continue
        checked += 1
        assert sample["mutated_label"] == sample["label"], (
            f"{kind} flipped {sample['label']} -> {sample['mutated_label']} "
            f"at bar {index}: it reached inside the last {RANGE_WINDOW} bars")
    assert checked >= 20, "too few labelable anchors to prove anything"


@pytest.mark.parametrize("kind", ADMITTED)
def test_an_admitted_mutation_actually_changes_the_frame(kind):
    """A mutation the encoder's bands swallow is the same pair taught twice."""
    bars = _bars()
    horizon = 12
    rng = random.Random(5)
    moved = total = 0
    for index in _anchors(bars, horizon):
        sample = mutated_sample(bars, index, horizon=horizon,
                                symbol="TEST-USDC", chain="base", kind=kind,
                                rng=rng)
        if sample is None:
            continue
        total += 1
        if sample["frames"] != sample["base_frames"]:
            moved += 1
    assert total >= 20
    assert moved / total > 0.5, (
        f"{kind} left the frame identical on {1 - moved / total:.0%} of bars; "
        "it buys no new training pair")


def test_inversion_is_poison_and_stays_dropped():
    """Mirroring the window turns a trough into a crest. Label cannot be kept."""
    assert MUTATIONS["invert"][2] is False
    bars = _bars()
    horizon = 12
    rng = random.Random(7)
    flips = total = 0
    for index in _anchors(bars, horizon):
        sample = mutated_sample(bars, index, horizon=horizon,
                                symbol="TEST-USDC", chain="base",
                                kind="invert", rng=rng)
        if sample is None:
            continue
        total += 1
        flips += sample["mutated_label"] != sample["label"]
    assert total >= 20
    assert flips / total > 0.25, (
        "inversion is dropped BECAUSE it moves the label; if this stops being "
        "true the reason for dropping it has changed and must be re-argued")


def test_amplitude_is_dropped_even_though_it_never_flips_the_label():
    """The trap: position-in-range is a ratio, so scaling leaves it invariant.

    The label survives and the mutation is still poison, because the label's
    FIRST test is abs(forward) against an absolute round-trip cost and this is
    the mutation that moves magnitude. The census shows it moving the temporal,
    volatility and cross slots and never the geometry ones.
    """
    assert MUTATIONS["amplitude"][2] is False
    bars = _bars()
    horizon = 12
    rng = random.Random(9)
    flips = total = 0
    geometry_moved = 0
    for index in _anchors(bars, horizon):
        sample = mutated_sample(bars, index, horizon=horizon,
                                symbol="TEST-USDC", chain="base",
                                kind="amplitude", rng=rng)
        if sample is None:
            continue
        total += 1
        flips += sample["mutated_label"] != sample["label"]
        if sample["frames"]["geometry"] != sample["base_frames"]["geometry"]:
            geometry_moved += 1
    assert total >= 20
    assert flips == 0, "amplitude scaling should be label-INVARIANT; that is the trap"
    assert geometry_moved == 0, (
        "geometry is built from ratios and must be invariant under amplitude "
        "scaling; if it moved, the mutation is not the one described")


def test_the_safe_prefix_is_derived_from_the_labeller_not_hard_coded():
    from scripts import omen_shape_mutations as mod
    assert mod.DEEP_EDGE == RANGE_WINDOW
