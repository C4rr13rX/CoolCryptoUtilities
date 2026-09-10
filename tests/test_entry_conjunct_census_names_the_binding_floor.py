"""A conjunct that never clears its floor must be named, not averaged away.

``BusScheduler.evaluate`` opens a position only when four conditions hold at
once. A four-way AND fails silently -- the tick ends at ``no_candidates`` -- so
for three passes the missing candidates were attributed to the symbol edge ban,
which had refused the worst-affected symbol exactly zero times.

Measured 2026-09-10 over 1709 snapshots in 6h, the inputs those floors read:

    direction_prob   min 0.0134  p50 0.0416  max 0.5000  floor 0.6   0/1709
    net_margin       min -2.0442 p50 -1.6645 max 0.0000  floor 0.0   4/1709

``scripts/entry_conjunct_census.py`` exists so that stays visible. These tests
pin the three properties that make it trustworthy, each of which is a way the
instrument could quietly lie:

  * a conjunct observed and never once satisfied is reported as UNSATISFIABLE,
    not as "mostly failing" -- a mean or a pass-rate hides the difference
    between a strict gate and a closed lane;
  * a conjunct whose input is not recorded reports ``observed: 0`` and is NOT
    counted as unsatisfiable, because an unmeasured condition read as a failing
    one is how a census invents a wall;
  * the floors come from the environment production actually runs with, not
    from the defaults written in the docstring.
"""

from __future__ import annotations

import scripts.entry_conjunct_census as census_mod


def _preds(**series):
    """One prediction block per index across the given keyed series."""
    length = max(len(v) for v in series.values())
    return [
        {k: v[i] for k, v in series.items() if i < len(v)}
        for i in range(length)
    ]


def test_a_conjunct_that_never_clears_its_floor_is_named_unsatisfiable() -> None:
    """0/1709 is a closed lane, and must not read as a strict gate."""
    report = census_mod.census(
        _preds(
            # The live shape: always well under the 0.6 floor, and its best
            # tick still short of it.
            direction_prob=[0.0134, 0.0416, 0.5000, 0.28],
            net_margin=[-2.0442, -1.6645, 0.0, -1.2],
        )
    )

    assert report["direction_prob"]["reachable"] == 0
    assert report["direction_prob"]["observed"] == 4
    assert "direction_prob" in census_mod.unsatisfiable(report), (
        "a conjunct that never cleared its floor was not named; the entry test "
        "is an AND, so this is the difference between strict and closed"
    )
    # net_margin touches its floor exactly once (0.0 >= 0.0): satisfiable, and
    # therefore NOT the thing to go and fix first.
    assert report["net_margin"]["reachable"] == 1
    assert "net_margin" not in census_mod.unsatisfiable(report)

    text = census_mod.render(report)
    assert "UNSATISFIABLE ON THIS FEED" in text
    assert "direction_prob" in text
    assert "0.5000" in text, "the best the input ever managed must be printed"


def test_an_unrecorded_conjunct_is_not_reported_as_a_wall() -> None:
    """An input absent from the payload is not a refusal.

    Counting an unmeasured input as a failing one would invent a third closed
    conjunct and send the next reader to fix a number nobody records.
    """
    report = census_mod.census(_preds(direction_prob=[0.9, 0.8]))

    assert report["confidence"]["observed"] == 0
    assert census_mod.unsatisfiable(report) == [], (
        "an unmeasured condition was reported as unsatisfiable"
    )
    assert "not recorded" in census_mod.render(report)


def test_the_floor_comes_from_the_environment_production_runs(monkeypatch) -> None:
    """A floor read from a default while production runs another is a lie."""
    monkeypatch.setenv("SCHEDULER_MIN_DIRECTION_PROB", "0.30")
    report = census_mod.census(_preds(direction_prob=[0.35, 0.10]))

    assert report["direction_prob"]["floor"] == 0.30
    assert report["direction_prob"]["reachable"] == 1, (
        "the census used the docstring default instead of the live setting"
    )

    # An unparseable setting falls back to the default rather than crashing the
    # census or silently reading the floor as zero, which would report every
    # conjunct as satisfiable.
    monkeypatch.setenv("SCHEDULER_MIN_DIRECTION_PROB", "not-a-number")
    fallback = census_mod.census(_preds(direction_prob=[0.35]))
    assert fallback["direction_prob"]["floor"] == 0.6
    assert fallback["direction_prob"]["reachable"] == 0


def test_a_boolean_is_not_counted_as_a_measurement() -> None:
    """`True` is an int in Python; counted as 1.0 it would clear a 0.6 floor."""
    report = census_mod.census([{"direction_prob": True}, {"direction_prob": 0.1}])
    assert report["direction_prob"]["observed"] == 1
    assert report["direction_prob"]["max"] == 0.1


def test_a_conjunct_is_read_from_the_key_the_scheduler_actually_binds() -> None:
    """`confidence` is bound from `exit_conf`, and the names do not match.

    trading/scheduler.py:634 reads
    ``confidence = float(pred_summary.get("exit_conf", 0.5))`` and then tests it
    against SCHEDULER_MIN_CONFIDENCE. A census that looked up "confidence" in
    the payload found nothing and reported the conjunct as UNMEASURED -- the
    safe direction, but it hid a second unsatisfiable floor: exit_conf runs
    0.4695 / 0.5000 / 0.5234 (min/p50/max) against 0.6, so it never clears
    either. Two of the four conjuncts are closed, not one.
    """
    report = census_mod.census([{"exit_conf": 0.5234}, {"exit_conf": 0.4695}])

    assert report["confidence"]["observed"] == 2, (
        "the census read the conjunct's NAME instead of the payload key the "
        "scheduler binds it from"
    )
    assert report["confidence"]["payload_key"] == "exit_conf"
    assert report["confidence"]["max"] == 0.5234
    assert "confidence" in census_mod.unsatisfiable(report)

    # A payload that carries a literal "confidence" key must not be picked up
    # in its place: that would be a different number under the same name.
    ignored = census_mod.census([{"confidence": 0.99}])
    assert ignored["confidence"]["observed"] == 0
