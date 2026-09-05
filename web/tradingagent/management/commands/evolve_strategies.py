"""Run the genetic search on a schedule and publish what survives.

WHY THIS IS A LOOP AND NOT A SCRIPT
-----------------------------------
The search judges rules on held-out data, and the data grows by a trade at a
time. A rule that clears the bar on 156 round trips may not clear it on 300 --
that is the point of re-running rather than trusting a finding forever. Every
pass re-searches from scratch and republishes; a rule that has stopped earning
its place simply stops being published, and the strategy registry loses it on
the next reload.

WHAT IT WILL NOT DO
-------------------
It will not lower the bar to find something. A pass that publishes nothing is
a correct outcome on a book with no signal in it, and the alternative -- a
search that always returns a rule -- is one whose output means nothing.

    python manage.py evolve_strategies              one pass
    python manage.py evolve_strategies --loop       forever, on an interval
"""

from __future__ import annotations

import time

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Search for trading rules and publish the ones that survive a holdout."

    def add_arguments(self, parser):
        parser.add_argument("--loop", action="store_true",
                            help="Keep searching on an interval.")
        parser.add_argument("--interval", type=int, default=3600,
                            help="Seconds between passes in --loop mode.")
        parser.add_argument("--seeds", type=int, default=5,
                            help="Independent searches per pass. A rule found "
                                 "under one seed only is a seed artifact.")
        parser.add_argument("--min-t", type=float, default=2.0,
                            help="Holdout t a rule must clear to be published.")
        parser.add_argument("--dry-run", action="store_true",
                            help="Search and report, but publish nothing.")

    def handle(self, *args, **options):
        while True:
            self._one_pass(options)
            if not options["loop"]:
                return
            time.sleep(max(60, int(options["interval"])))

    def _one_pass(self, options) -> None:
        from tradingagent.evolve import evolve, publish
        from tradingagent.features import closed_round_trips

        rows = closed_round_trips()
        self.stdout.write(f"closed round trips available: {len(rows)}")

        # MULTIPLE SEEDS, AND ONLY WHAT RECURS.
        #
        # One search on one seed finds whatever its random walk happened to
        # reach. Measured on the real book: five seeds produced 6, 5, 5, 0 and
        # 4 survivors, and `hurst` appeared in the top rule of four of them --
        # the recurrence is the evidence, not any single run.
        seen: dict = {}
        best: dict = {}
        for index in range(max(1, int(options["seeds"]))):
            result = evolve(rows, seed=20260905 + index * 7919)
            if not result.get("ok"):
                self.stdout.write(f"  seed {index}: {result.get('reason')}")
                continue
            for survivor in result.get("survivors") or []:
                rule = survivor["rule"]
                seen[rule] = seen.get(rule, 0) + 1
                if rule not in best or survivor["holdout_t"] > best[rule]["holdout_t"]:
                    best[rule] = survivor

        if not best:
            self.stdout.write(self.style.WARNING(
                "  nothing survived a holdout in any seed -- publishing "
                "nothing, which is the correct answer for a book with no "
                "measurable signal in it"))
            return

        seeds = max(1, int(options["seeds"]))
        # A rule found by a single seed out of several is a coincidence of
        # that seed's random walk. Require it in more than one when we ran
        # more than one.
        threshold = 2 if seeds > 1 else 1
        durable = [best[r] for r, count in seen.items() if count >= threshold]
        durable.sort(key=lambda s: s["holdout_t"], reverse=True)

        self.stdout.write(
            f"  {len(seen)} distinct rules found; {len(durable)} appeared in "
            f"{threshold}+ of {seeds} seeds")
        for survivor in durable[:6]:
            self.stdout.write(
                f"    t={survivor['holdout_t']:+.2f} "
                f"({seen[survivor['rule']]}/{seeds} seeds)  {survivor['rule']}")

        if options["dry_run"]:
            self.stdout.write("  --dry-run: nothing published")
            return

        outcome = publish({"ok": True, "survivors": durable},
                          min_t=float(options["min_t"]))
        self.stdout.write(self.style.SUCCESS(
            f"  published {outcome['published']} rule(s)"))
