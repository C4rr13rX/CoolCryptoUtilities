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

    #: Backoff when the CLI did not say when the quota returns. Doubling from
    #: five minutes to an hour: short enough that a brief limit costs one
    #: cycle, long enough that an exhausted daily quota is not hammered.
    QUOTA_BACKOFF_SEC = (300.0, 600.0, 1200.0, 2400.0, 3600.0)

    def _quota_wait_seconds(self, run) -> float:
        """How long to wait before trying again, or 0 to carry on normally.

        Reads the run's own report rather than a side channel, so a pass that
        hit the wall is visible in the same record everything else is.
        """
        report = getattr(run, "report", "") or ""
        status = str(getattr(run, "status", "") or "")
        haystack = f"{status} {report}".lower()
        if "quota" not in haystack and "rate limit" not in haystack:
            return 0.0

        # A stated reset time beats any guess we could make.
        resets_at = getattr(run, "resets_at", None)
        if resets_at:
            try:
                remaining = float(resets_at) - time.time()
            except (TypeError, ValueError):
                remaining = 0.0
            if remaining > 0:
                # A minute past the stated reset, because a wall clock that
                # is a few seconds fast turns "wait until" into "retry into
                # a closed door".
                return min(remaining + 60.0, 6 * 3600.0)

        index = min(self._quota_streak, len(self.QUOTA_BACKOFF_SEC) - 1)
        self._quota_streak += 1
        return self.QUOTA_BACKOFF_SEC[index]

    def handle(self, *args, **options):
        self._quota_streak = 0
        from tradingagent.engine import run_once
        from tradingagent.models import AgentConfig

        single = bool(options.get("once"))
        consecutive_quota = 0
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
            quota_wait = 0.0
            try:
                run = run_once(config)
                self.stdout.write(
                    f"run {run.pk} [{run.status}] {run.duration_sec:.0f}s "
                    f"opened={run.trades_opened} closed={run.trades_closed} "
                    f"tier={config.tier}")
                if run.report:
                    self.stdout.write("  " + run.report[:400])

                quota_wait = self._quota_wait_seconds(run)
                if quota_wait > 0:
                    consecutive_quota += 1
                else:
                    consecutive_quota = 0
            except Exception as exc:  # noqa: BLE001
                # A failed pass must not end the worker. The next one may
                # succeed, and a dead worker is silent in a way a failed pass
                # is not.
                self.stdout.write(self.style.ERROR(
                    f"pass failed: {type(exc).__name__}: {exc}"))

            if single:
                return

            if quota_wait > 0:
                # WAIT FOR THE QUOTA, NOT FOR THE INTERVAL.
                #
                # Retrying a spent session on the normal cadence spawns a
                # process per cycle and produces nothing until the window
                # reopens, and fills the run history with failures where the
                # truth is "waiting". Sleeping until the stated reset -- or
                # backing off when no reset was stated -- means the agent
                # resumes on its own the moment it can operate again, without
                # anyone restarting it.
                self.stdout.write(self.style.WARNING(
                    f"quota exhausted; waiting {quota_wait / 60:.1f} min "
                    f"before the next attempt (streak {consecutive_quota})"))
                time.sleep(quota_wait)
                continue

            # Sleep the remainder of the interval, so a slow pass does not
            # compound into drift.
            elapsed = time.time() - started
            time.sleep(max(30.0, float(config.interval_sec) - elapsed))
