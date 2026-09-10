"""Three counts of "the strategies" existed and none of them said which it was.

THE FAILURE THIS PREVENTS. Measured across passes 106-111, a single pass quoted
72, 43 and 39 as "the strategies", and commits of that day carry "13 of 42",
"28 of 42" and "32 of 42" against a denominator nobody had defined. All three
counts were CORRECT -- they answer different questions:

    offered       72   build_default_registry().ids(); every strategy
                       StrategyRegistry.evaluate_all is asked on every tick
    commissioned  43   data/strategy_registry.json, the lifetime record
    evidenced     39   data/strategy_ledger.json, has recorded an outcome

and they do not nest the way a reader assumes. ``evidenced`` IS a subset of
``commissioned``, but ``offered`` is NOT a superset of either: 35 plugin ids
have no registry row until they first trade, and 6 registry ids are not plugins
at all. The union is 78.

WHY THIS IS A MONEY BUG AND NOT BOOKKEEPING. Every acceptance criterion the
loop steers by is a fraction of one of these -- "strategies with >=5 tradeable
trades must rise from 11 of 38", "atf_static* share must fall from 70%". A
share can be made to rise or fall purely by which count is in the denominator,
and worse: a share whose denominator is ``evidenced`` IS SELF-REFERENTIAL,
because a strategy enters that population by producing the numerator. "11 of
38" can be satisfied by strategies leaving the ledger. Against the population
that COULD produce evidence it is 1 of 78.

RULED OUT, so nobody re-checks it: the registry is NOT written non-atomically.
``services/strategy_registry._save`` goes through
``services.atomic_json.write_json``, which writes a PID+uuid unique temp and
``os.replace``s it under an O_EXCL lock. A torn read fails json parsing and
yields zero, never a plausible smaller count.

These assertions fail against the pre-fix code: ``populations()`` did not
exist, and ``graduation_status`` printed a bare "39 strategies" with no
denominator block at all.
"""

from __future__ import annotations

import unittest


class ThePopulationsAreReportedTogetherAndNamed(unittest.TestCase):
    def setUp(self) -> None:
        from services.strategy_population import populations

        self.pop = populations()

    def test_all_three_populations_are_reported_in_one_read(self) -> None:
        """One read, or two readers disagree about a moving file.

        Reading the registry and the ledger in separate calls minutes apart is
        how "42 then 24" was reported: the counts were never compared inside a
        single measurement, so no reader could tell a real change from two
        reads of different moments.
        """
        for key in ("offered", "commissioned", "evidenced", "known"):
            with self.subTest(key=key):
                self.assertIn(key, self.pop["counts"])
                self.assertIsInstance(self.pop["counts"][key], int)

    def test_every_count_names_the_source_it_came_from(self) -> None:
        """A bare number is what made three of them indistinguishable."""
        for key in ("offered", "commissioned", "evidenced", "known"):
            with self.subTest(key=key):
                source = self.pop["sources"].get(key)
                self.assertTrue(
                    source,
                    f"{key} has no named source, so a reader quoting it "
                    f"cannot say what it counts",
                )

    def test_the_known_population_is_the_union_and_bounds_the_others(self) -> None:
        """``known`` must contain every id any of the three knows about.

        This is the assertion that catches a future population being added and
        silently left out of the union -- which would reintroduce exactly the
        invisible-strategies problem, since a share against a short union
        overstates coverage.
        """
        known = set(self.pop["known"])
        for key in ("offered", "commissioned", "evidenced"):
            with self.subTest(key=key):
                self.assertTrue(
                    set(self.pop[key]) <= known,
                    f"{key} holds ids missing from 'known', so the union is "
                    f"not a bound and any share against it is overstated",
                )
        self.assertEqual(len(known), self.pop["counts"]["known"])

    def test_the_differences_between_the_populations_are_spelled_out(self) -> None:
        """"Why are these not equal" must not require a set difference.

        The set differences ARE the finding -- 35 strategies asked on every
        tick with no registry row are invisible to the status command and the
        population page. A payload that reports only counts hides them.
        """
        for key in ("offered_not_commissioned", "commissioned_not_offered",
                    "evidenced_not_commissioned"):
            with self.subTest(key=key):
                self.assertIn(key, self.pop)
                self.assertIsInstance(self.pop[key], list)

    def test_the_evidenced_population_is_a_subset_of_the_commissioned_one(self) -> None:
        """A strategy that recorded an outcome must have a lifetime record.

        ``ledger.record`` mirrors into the registry, so an id in the ledger and
        not the registry means that mirror is failing -- which is the
        json-store concurrency loss that once dropped 95% of the evidence
        gating graduation. Asserted as a live invariant, not described.
        """
        extra = self.pop["evidenced_not_commissioned"]
        self.assertEqual(
            extra,
            [],
            "these strategies recorded an outcome but have no lifetime "
            f"registry row, so the registry mirror is dropping writes: {extra}",
        )
        self.assertTrue(self.pop["evidence_is_a_subset_of_commissioned"])

    def test_the_offer_population_is_not_assumed_to_contain_the_others(self) -> None:
        """The nesting everyone assumes is false, and must stay measured.

        If this ever becomes empty it is a real change worth noticing, not a
        tidier world: it would mean every plugin now carries a registry row.
        Asserting the CURRENT shape would freeze a defect; asserting the key
        is reported keeps the question live.
        """
        self.assertIn("commissioned_not_offered", self.pop)
        offered = set(self.pop["offered"])
        commissioned = set(self.pop["commissioned"])
        self.assertEqual(
            sorted(commissioned - offered),
            self.pop["commissioned_not_offered"],
            "the reported difference does not match the sets it is derived "
            "from -- a stale field is worse than none",
        )


class TheStatusCommandReadsTheSameDenominator(unittest.TestCase):
    """Both readers, one number. Demonstrated by running both."""

    def test_the_status_payload_carries_the_same_counts(self) -> None:
        """``graduation_status`` must not compute its own population.

        A second copy of a denominator is a denominator that silently drifts,
        which is the same failure ``strategy_population`` was written to stop
        for the promotion thresholds. Run both, compare the counts.
        """
        from services.strategy_population import populations
        from scripts.graduation_status import _populations_view

        direct = populations()["counts"]
        via_status = _populations_view()["counts"]
        self.assertEqual(
            direct,
            via_status,
            "the status command and services.strategy_population disagree "
            "about the population, which is the exact condition that makes a "
            "share-phrased acceptance criterion unfalsifiable",
        )

    def test_the_rendered_status_names_which_population_it_printed(self) -> None:
        """The rendered text, not just the payload.

        The operator and every pass read the rendered status, not the JSON. A
        bare "39 strategies" is what got quoted as "the population" -- so the
        rendering is where the fix has to land, and it is asserted on the
        output rather than on a comment claiming it.
        """
        from scripts.graduation_status import render

        text = render({
            "criteria": {"min_trades": 20},
            "totals": {"strategies": 39, "ledger_span_days": 7.9},
            "populations": {"counts": {"offered": 72, "commissioned": 43,
                                       "evidenced": 39, "known": 78}},
            "strategies": [],
        })
        self.assertIn("EVIDENCED", text)
        self.assertIn("78", text)
        self.assertIn("72", text)

    def test_the_status_still_renders_when_the_population_is_unavailable(self) -> None:
        """A missing denominator must be visibly missing, never fabricated.

        Degrading to a plausible number would be worse than degrading to
        nothing: a share computed against a fabricated denominator is
        indistinguishable from a correct one.
        """
        from scripts.graduation_status import render

        text = render({
            "criteria": {"min_trades": 20},
            "totals": {"strategies": 39, "ledger_span_days": 7.9},
            "populations": {},
            "strategies": [],
        })
        self.assertIn("39 strategies", text)
        self.assertNotIn("EVIDENCED", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
