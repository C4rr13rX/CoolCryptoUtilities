"""Run production work one task at a time, paced by real machine headroom.

The problem this replaces: every cycle fired its whole task list, one slow task
(news_enrichment at its 120s timeout) pushed the backlog past the governor cap,
and the orchestrator then SKIPPED the entire cycle -- trading included. So the
work that makes money was starved by work that does not, and the backlog grew
because nothing ever drained.

Two rules fix that:

**Sequential across areas, not depth-first into one.**
Tasks run one after another in round-robin across CATEGORIES (trade, feed,
model, housekeeping) rather than draining one category before starting the
next. A stalled news feed therefore delays news, not trading.

**Priority is "can we trade profitably right now".**
Ordering is by what the money path needs: a live position that needs managing
outranks a ghost cycle, which outranks a price feed, which outranks model
training, which outranks telemetry. Under pressure the tail is deferred, never
the head.

**Polite by construction.**
Before each task the scheduler checks CPU and available RAM. Above the pressure
thresholds it sleeps instead of starting new work, so the machine stays usable;
it never spawns parallel work to "catch up", which is what locks a box up.
Deferred tasks age and gain priority so nothing starves forever.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

try:  # psutil is present in production; degrade gracefully in tests
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


# Categories, most valuable to the money path first. Round-robin visits them in
# this order, so one cycle touches every area rather than draining one.
CATEGORY_ORDER = ("trade", "feed", "model", "housekeeping")

#: Base priority per category (lower runs first).
CATEGORY_PRIORITY = {
    "trade": 0,          # manage live/ghost positions, place orders
    "feed": 1,           # prices the trade path reads
    "model": 2,          # training, GA, brain work
    "housekeeping": 3,   # telemetry, pruning, reports
}


@dataclass
class Task:
    name: str
    fn: Callable[[], Any]
    category: str = "housekeeping"
    kwargs: Dict[str, Any] = field(default_factory=dict)
    #: Minimum seconds between runs. 0 = every pass.
    interval_sec: float = 0.0
    #: Skip when free RAM is below this. Heavy tasks set it; trading does not.
    min_free_mb: float = 0.0
    #: Hard cap on one execution.
    timeout_sec: float = 120.0
    #: Never deferred for pressure. Reserved for the trade path.
    critical: bool = False

    last_run: float = 0.0
    last_ok: Optional[bool] = None
    last_error: str = ""
    runs: int = 0
    failures: int = 0
    deferrals: int = 0
    total_sec: float = 0.0

    def due(self, now: float) -> bool:
        return self.interval_sec <= 0 or (now - self.last_run) >= self.interval_sec


@dataclass
class Pressure:
    cpu: float = 0.0
    free_mb: float = 0.0
    mem_pct: float = 0.0

    @property
    def unknown(self) -> bool:
        return self.free_mb <= 0.0 and self.cpu <= 0.0


def read_pressure() -> Pressure:
    if psutil is None:
        return Pressure()
    try:
        vm = psutil.virtual_memory()
        # interval=None returns the value since the last call: non-blocking, so
        # sampling never becomes its own source of load.
        return Pressure(
            cpu=float(psutil.cpu_percent(interval=None)),
            free_mb=float(vm.available) / 1e6,
            mem_pct=float(vm.percent),
        )
    except Exception:
        return Pressure()


class SequentialScheduler:
    """Round-robin across categories, one task at a time, pressure-aware."""

    def __init__(
        self,
        *,
        cpu_pause_pct: Optional[float] = None,
        mem_pause_pct: Optional[float] = None,
        min_free_mb: Optional[float] = None,
        idle_sleep_sec: float = 1.0,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.cpu_pause_pct = float(
            cpu_pause_pct if cpu_pause_pct is not None
            else os.getenv("SEQ_SCHED_CPU_PAUSE_PCT", "88")
        )
        self.mem_pause_pct = float(
            mem_pause_pct if mem_pause_pct is not None
            else os.getenv("SEQ_SCHED_MEM_PAUSE_PCT", "92")
        )
        self.min_free_mb = float(
            min_free_mb if min_free_mb is not None
            else os.getenv("SEQ_SCHED_MIN_FREE_MB", "700")
        )
        self.idle_sleep_sec = float(idle_sleep_sec)
        self._tasks: List[Task] = []
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._cursor = 0
        self._on_event = on_event
        self.cycles = 0
        self.deferred_total = 0

    # -- registration -------------------------------------------------------

    def add(self, task: Task) -> None:
        with self._lock:
            self._tasks.append(task)

    def add_many(self, tasks: Sequence[Task]) -> None:
        for t in tasks:
            self.add(t)

    # -- pacing -------------------------------------------------------------

    def should_pause(self, pressure: Optional[Pressure] = None) -> bool:
        """Is the machine too busy to start new work?

        Politeness rule: when unsure, do NOT pause. An unreadable pressure
        sample must not silently stop trading.
        """
        p = pressure or read_pressure()
        if p.unknown:
            return False
        if p.mem_pct >= self.mem_pause_pct:
            return True
        if 0 < p.free_mb < self.min_free_mb:
            return True
        return p.cpu >= self.cpu_pause_pct

    def _runnable(self, task: Task, pressure: Pressure, now: float) -> bool:
        if not task.due(now):
            return False
        if task.critical:
            return True
        if task.min_free_mb > 0 and 0 < pressure.free_mb < task.min_free_mb:
            task.deferrals += 1
            self.deferred_total += 1
            return False
        return True

    def _ordered(self, now: float) -> List[Task]:
        """Round-robin across categories, with ageing so nothing starves.

        Rotating the category start each cycle is what makes this sequential
        ACROSS the system rather than depth-first into whichever area happens
        to sort first.
        """
        with self._lock:
            tasks = list(self._tasks)
        rotation = self.cycles % max(1, len(CATEGORY_ORDER))
        order = list(CATEGORY_ORDER[rotation:]) + list(CATEGORY_ORDER[:rotation])
        rank = {name: idx for idx, name in enumerate(order)}

        def key(t: Task):
            base = rank.get(t.category, len(order))
            # Ageing: a task waiting far past its interval climbs the order, so
            # a permanently busy machine cannot starve the tail forever.
            waited = now - t.last_run if t.last_run else float("inf")
            overdue = 0.0
            if t.interval_sec > 0 and waited != float("inf"):
                overdue = max(0.0, waited - t.interval_sec) / max(t.interval_sec, 1.0)
            return (base - min(overdue, float(len(order))), CATEGORY_PRIORITY.get(t.category, 9), t.name)

        return sorted(tasks, key=key)

    # -- execution ----------------------------------------------------------

    def run_once(self) -> Dict[str, Any]:
        """One pass: at most one execution per due task, in priority order."""
        now = time.time()
        pressure = read_pressure()
        ran: List[str] = []
        deferred: List[str] = []
        for task in self._ordered(now):
            if self._stop.is_set():
                break
            if not task.due(now):
                continue
            # Re-read pressure between tasks: the previous task may have
            # changed it, and pausing on stale numbers is how a machine either
            # locks up or idles for no reason.
            pressure = read_pressure()
            if not task.critical and self.should_pause(pressure):
                deferred.append(task.name)
                task.deferrals += 1
                self.deferred_total += 1
                continue
            if not self._runnable(task, pressure, now):
                deferred.append(task.name)
                continue
            self._execute(task)
            ran.append(task.name)
        self.cycles += 1
        return {
            "cycle": self.cycles,
            "ran": ran,
            "deferred": deferred,
            "cpu": pressure.cpu,
            "free_mb": pressure.free_mb,
        }

    def _execute(self, task: Task) -> None:
        started = time.time()
        task.last_run = started
        task.runs += 1
        try:
            result = [None]
            error = [None]

            def _target():
                try:
                    result[0] = task.fn(**task.kwargs) if task.kwargs else task.fn()
                except Exception as exc:  # noqa: BLE001
                    error[0] = exc

            thread = threading.Thread(target=_target, name="seq-%s" % task.name, daemon=True)
            thread.start()
            thread.join(timeout=max(1.0, task.timeout_sec))
            if thread.is_alive():
                # Do not kill it -- the thread is daemon and will finish or die
                # with the process. Recording the timeout keeps the SCHEDULER
                # moving, which is the whole point: one stuck task must not
                # stop the rest of the system.
                task.last_ok = False
                task.last_error = "timeout after %.0fs" % task.timeout_sec
                task.failures += 1
                self._emit("task_timeout", {"task": task.name, "timeout": task.timeout_sec})
                return
            if error[0] is not None:
                task.last_ok = False
                task.last_error = "%s: %s" % (type(error[0]).__name__, error[0])
                task.failures += 1
                self._emit("task_error", {"task": task.name, "error": task.last_error})
                return
            task.last_ok = True
            task.last_error = ""
        finally:
            task.total_sec += time.time() - started

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        if self._on_event:
            try:
                self._on_event(event, payload)
            except Exception:
                pass

    # -- loop ---------------------------------------------------------------

    def run_forever(self) -> None:
        while not self._stop.is_set():
            summary = self.run_once()
            if not summary["ran"]:
                # Nothing was due or everything was deferred: idle politely
                # rather than spinning.
                self._stop.wait(self.idle_sleep_sec)

    def stop(self) -> None:
        self._stop.set()

    # -- introspection ------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """What the dashboard shows. '--' for values that do not exist yet."""
        with self._lock:
            tasks = list(self._tasks)
        p = read_pressure()
        return {
            "cycles": self.cycles,
            "deferred_total": self.deferred_total,
            "pressure": {
                "cpu_pct": p.cpu if not p.unknown else None,
                "free_mb": p.free_mb if not p.unknown else None,
                "mem_pct": p.mem_pct if not p.unknown else None,
                "paused": self.should_pause(p),
            },
            "thresholds": {
                "cpu_pause_pct": self.cpu_pause_pct,
                "mem_pause_pct": self.mem_pause_pct,
                "min_free_mb": self.min_free_mb,
            },
            "tasks": [
                {
                    "name": t.name,
                    "category": t.category,
                    "priority": CATEGORY_PRIORITY.get(t.category, 9),
                    "critical": t.critical,
                    "runs": t.runs,
                    "failures": t.failures,
                    "deferrals": t.deferrals,
                    # None, not 0 -- "never run" is not "ran and took no time".
                    "last_run": t.last_run or None,
                    "last_ok": t.last_ok,
                    "last_error": t.last_error or None,
                    "avg_sec": (t.total_sec / t.runs) if t.runs else None,
                }
                for t in sorted(tasks, key=lambda x: (CATEGORY_PRIORITY.get(x.category, 9), x.name))
            ],
        }
