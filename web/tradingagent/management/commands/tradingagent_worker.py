"""Run the reading agent on its interval.

The agent had a config flag, an engine and an API, and nothing that called
it -- enabled=True meant only that a manual trigger would work. This is the
loop that actually makes it run.

Deliberately a management command rather than a thread inside the web
process: the web server is restarted whenever static changes, and an agent
that dies with a deploy is an agent that trades in bursts nobody planned.
"""

from __future__ import annotations

import time

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Run the trading agent on its configured interval."

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true",
                            help="Run a single pass and exit.")

    def handle(self, *args, **options):
        from tradingagent.engine import run_once
        from tradingagent.models import AgentConfig

        single = bool(options.get("once"))
        self.stdout.write("trading agent worker starting")

        while True:
            config = AgentConfig.load()

            if not config.enabled:
                # Re-read every cycle rather than exiting: the switch is in
                # the database so it can be flipped from the UI without a
                # restart, and a worker that exits on disable has to be
                # remembered about later.
                self.stdout.write("agent disabled; waiting")
                if single:
                    return
                time.sleep(60)
                continue

            started = time.time()
            try:
                run = run_once(config)
                self.stdout.write(
                    f"run {run.pk} [{run.status}] {run.duration_sec:.0f}s "
                    f"opened={run.trades_opened} closed={run.trades_closed} "
                    f"tier={config.tier}")
                if run.report:
                    self.stdout.write("  " + run.report[:400])
            except Exception as exc:  # noqa: BLE001
                # A failed pass must not end the worker. The next one may
                # succeed, and a dead worker is silent in a way a failed pass
                # is not.
                self.stdout.write(self.style.ERROR(
                    f"pass failed: {type(exc).__name__}: {exc}"))

            if single:
                return

            # Sleep the remainder of the interval, so a slow pass does not
            # compound into drift.
            elapsed = time.time() - started
            time.sleep(max(30.0, float(config.interval_sec) - elapsed))
