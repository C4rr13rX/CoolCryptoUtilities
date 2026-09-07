"""Corpus building must not hold the live price feed hostage.

MEASURED 2026-09-06. The feed was silent for **55 of 360 minutes -- 15%
downtime** -- in two stalls of 11.4 and 43.4 minutes, while
``scripts/loop_status.py`` reported "0 ticks/10m" and production reported UP.
It looks exactly like a crashed feed and is not one: it recovers by itself,
which sends you hunting a crash that never happened.

The cause is that ``_try_cex_fallback`` runs inside ``data_ingest`` --
``Task("data_ingest", ..., timeout_sec=90.0, interval_sec=30.0)`` in
production.py, the task that feeds the live price stream. A full pass
downloads three years of hourly candles for up to 40 pairs. Observed: 27
symbols over 35 minutes, individual symbols at 12,668 / 22,099 / 26,268
candles, and a ``task_overrun`` of 2363 seconds against that 90-second budget.

The scheduler is NOT the bug. ``sequential_scheduler._execute`` joins with a
timeout, abandons the thread, records ``task_timeout`` and moves on, and
``_run_cycle`` declines to stack a second copy while the first is running.
All correct. But the abandoned worker still holds the download, so the STREAM
stays dark even though the SCHEDULER is healthy.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORKERS = ROOT / "services" / "background_workers.py"
PRODUCTION = ROOT / "production.py"


def _default_for(source: str, name: str) -> str:
    """The default in ``os.getenv("NAME", "default")``."""
    match = re.search(
        rf'os\.getenv\(\s*["\']{re.escape(name)}["\']\s*,\s*["\']([^"\']+)["\']',
        source,
    )
    assert match, f"{name} is no longer read with a literal default"
    return match.group(1)


def test_the_per_cycle_pair_count_is_small():
    """40 pairs x 3 years cannot fit in a 90-second task.

    The number is per CYCLE, not in total: run_cex_fallback_cycle skips any
    symbol whose newest candle is still fresh, so the corpus converges across
    cycles instead of in one 35-minute block.
    """
    source = WORKERS.read_text(encoding="utf-8")
    pairs = int(_default_for(source, "CEX_FALLBACK_MAX_PAIRS"))
    assert pairs <= 5, (
        f"CEX_FALLBACK_MAX_PAIRS defaults to {pairs}; at roughly 90 seconds "
        f"per symbol that is {pairs * 90}s inside a task budgeted for 90")


def test_the_budget_is_under_the_feed_tasks_own_timeout():
    """The backfill must return on its own, not be abandoned mid-download.

    An abandoned thread keeps holding whatever it is downloading, which is
    precisely how the feed went dark while the scheduler stayed healthy.
    """
    workers = WORKERS.read_text(encoding="utf-8")
    budget = float(_default_for(workers, "CEX_FALLBACK_BUDGET_SEC"))

    production = PRODUCTION.read_text(encoding="utf-8")
    match = re.search(
        r'Task\(\s*"data_ingest".*?timeout_sec\s*=\s*([0-9.]+)',
        production,
        re.S,
    )
    assert match, "data_ingest is no longer registered with a timeout_sec"
    task_timeout = float(match.group(1))

    assert budget < task_timeout, (
        f"the backfill budget ({budget}s) must stay under data_ingest's own "
        f"timeout ({task_timeout}s), or the task is abandoned while still "
        f"holding the download and the feed stays dark")


def test_an_overrun_is_reported_rather_than_silent():
    """A stall that logs nothing is a stall nobody finds.

    The first symptom of this bug was a status line reading "0 ticks/10m"
    with no error anywhere -- the backfill was working exactly as written.
    """
    source = WORKERS.read_text(encoding="utf-8")
    assert "CEX_FALLBACK_BUDGET_SEC" in source
    assert "severity=\"warning\"" in source, (
        "exceeding the budget must warn, so the next stall names itself")


def test_the_backfill_still_runs():
    """A gate that switches the corpus off entirely would be the wrong fix.

    Three years of OHLCV is the training corpus. The requirement is that it
    is built WITHOUT stopping the feed, not that it stops being built.
    """
    source = WORKERS.read_text(encoding="utf-8")
    assert "run_cex_fallback_cycle(" in source
    days = int(_default_for(source, "CEX_FALLBACK_DAYS"))
    assert days >= 365, (
        f"CEX_FALLBACK_DAYS fell to {days}; the corpus depth is the point, "
        f"the per-cycle SIZE is what had to change")


def test_the_backfill_call_is_still_guarded():
    """It must not raise into the feed task.

    An exception escaping here would take out data_ingest itself, which is a
    worse failure than the stall it replaces.
    """
    source = WORKERS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
        if "run_cex_fallback_cycle" in body:
            assert node.handlers, "the try around the backfill has no except"
            return
    raise AssertionError(
        "run_cex_fallback_cycle is no longer inside a try/except; an "
        "exception here would take down the live feed task")
