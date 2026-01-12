from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from services.adaptive_control import AdaptiveLimiter
from services.logging_utils import log_message
from services.system_profile import SystemProfile, detect_system_profile


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    depends_on: List[str]
    workers: int = 1


class DependencyGraph:
    def __init__(self, map_path: Path) -> None:
        self.map_path = map_path
        raw = json.loads(map_path.read_text(encoding="utf-8"))
        components = raw.get("components") or []
        self.nodes: Dict[str, ComponentSpec] = {}
        for item in components:
            name = str(item.get("name"))
            if not name:
                continue
            depends = [str(dep) for dep in item.get("depends_on", [])]
            workers = max(1, int(item.get("workers", 1)))
            self.nodes[name] = ComponentSpec(name=name, depends_on=depends, workers=workers)
        if not self.nodes:
            raise ValueError(f"No components defined in {map_path}")


class TaskDescriptor:
    __slots__ = ("component", "cycle_id", "callable", "args", "kwargs", "metadata")

    def __init__(
        self,
        component: str,
        cycle_id: str,
        callable: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.component = component
        self.cycle_id = cycle_id
        self.callable = callable
        self.args = args
        self.kwargs = kwargs
        self.metadata = metadata or {}


class ParallelTaskManager:
    """
    Multi-queue task coordinator that enforces dependency ordering between components.
    Intended to keep the training pipeline, news enrichment, and ghost schedulers running
    concurrently without sacrificing deterministic sequencing.
    """

    def __init__(
        self,
        map_path: Optional[Path] = None,
        limiter: Optional[AdaptiveLimiter] = None,
        system_profile: Optional[SystemProfile] = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        default_map = repo_root / "orchestration" / "dependency_map.json"
        target_map = (map_path or default_map).resolve()
        self.system_profile = system_profile or detect_system_profile()
        self.graph = DependencyGraph(target_map)
        self.queues: Dict[str, queue.Queue[Optional[TaskDescriptor]]] = {
            name: queue.Queue() for name in self.graph.nodes
        }
        self.shutdown_flag = threading.Event()
        self.threads: List[threading.Thread] = []
        self._events: Dict[Tuple[str, str], threading.Event] = {}
        self._events_lock = threading.Lock()
        self._state: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._state_lock = threading.Lock()
        self._state_path = target_map.with_name("dependency_state.json")
        # Default limiter keeps CPU below ~75% unless overridden by ADAPTIVE_* env vars.
        self.limiter = limiter or AdaptiveLimiter.from_env()
        self.max_concurrent = self._derive_max_concurrent()
        self._global_slots = threading.Semaphore(self.max_concurrent)
        self._active = 0
        self._active_lock = threading.Lock()

    def start(self) -> None:
        if self.threads:
            return
        for name, spec in self.graph.nodes.items():
            worker_cap = self._workers_for_component(spec.workers)
            for worker_idx in range(worker_cap):
                thread = threading.Thread(
                    target=self._worker,
                    args=(name,),
                    name=f"{name}-worker-{worker_idx}",
                    daemon=True,
                )
                thread.start()
                self.threads.append(thread)

    def stop(self, timeout: float = 5.0) -> None:
        self.shutdown_flag.set()
        for q in self.queues.values():
            q.put(None)
        for thread in self.threads:
            thread.join(timeout=timeout)
        self.threads.clear()

    def reset_queues(self) -> None:
        """
        Drop all pending tasks and reset internal state. Useful when a backlog
        should be flushed before restarting orchestration.
        """
        with self._state_lock:
            self._state.clear()
            try:
                self._state_path.unlink(missing_ok=True)
            except Exception:
                pass
        with self._events_lock:
            self._events.clear()
        for q in self.queues.values():
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

    def submit(
        self,
        component: str,
        func: Callable[..., Any],
        *,
        cycle_id: Optional[str] = None,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if component not in self.queues:
            raise ValueError(f"Unknown component '{component}'.")
        cycle = cycle_id or str(int(time.time()))
        task = TaskDescriptor(
            component=component,
            cycle_id=cycle,
            callable=func,
            args=args or (),
            kwargs=kwargs or {},
            metadata=metadata,
        )
        self._update_state(cycle, component, status="queued", metadata=metadata)
        self.queues[component].put(task)
        return cycle

    def mark_skipped(
        self,
        component: str,
        *,
        cycle_id: Optional[str] = None,
        reason: str = "skipped",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Mark a component as satisfied for a given cycle without enqueueing work.
        Useful when an orchestrator intentionally throttles tasks but still
        needs dependency events to release downstream components.
        """
        if component not in self.queues:
            return
        cycle = cycle_id or str(int(time.time()))
        self._update_state(cycle, component, status=reason, metadata=metadata)
        self._signal_complete(component, cycle)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _workers_for_component(self, requested: int) -> int:
        if not self.system_profile:
            return max(1, min(requested, self.max_concurrent))
        if self.system_profile.is_low_power or self.system_profile.memory_pressure:
            return 1
        budget = max(1, min(self.system_profile.max_threads // 2, self.max_concurrent))
        return max(1, min(requested, budget))

    def _worker(self, component: str) -> None:
        spec = self.graph.nodes[component]
        q = self.queues[component]
        while not self.shutdown_flag.is_set():
            try:
                task = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if task is None:
                break
            cycle_id = task.cycle_id
            self._update_state(cycle_id, component, status="waiting_deps")
            for dep in spec.depends_on:
                self._wait_for_dependency(dep, cycle_id)
            self._acquire_slot()
            if self.limiter:
                self.limiter.before_task(component)
            self._update_state(cycle_id, component, status="running")
            try:
                task.callable(*task.args, **task.kwargs)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._update_state(cycle_id, component, status="failed", error=str(exc))
                log_message(component, f"task failed during cycle {cycle_id}", severity="error", details={"error": str(exc)})
            else:
                self._update_state(cycle_id, component, status="completed")
                log_message(component, f"task completed for cycle {cycle_id}")
            finally:
                self._signal_complete(component, cycle_id)
                self._release_slot()
                q.task_done()

    def _wait_for_dependency(self, component: str, cycle_id: str) -> None:
        event = self._get_event(component, cycle_id)
        with self._state_lock:
            state = (self._state.get(cycle_id) or {}).get(component, {})
        status = str(state.get("status") or "")
        done_states = {"completed", "skipped", "failed"}
        if status in done_states or status.startswith("skip"):
            event.set()
        event.wait()

    def _signal_complete(self, component: str, cycle_id: str) -> None:
        event = self._get_event(component, cycle_id)
        event.set()

    def _get_event(self, component: str, cycle_id: str) -> threading.Event:
        key = (component, cycle_id)
        with self._events_lock:
            event = self._events.get(key)
            if event is None:
                event = threading.Event()
                # When a component has no dependencies and completes instantly, we need an event to exist.
                self._events[key] = event
            return event

    def _update_state(
        self,
        cycle_id: str,
        component: str,
        *,
        status: str,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        timestamp = time.time()
        with self._state_lock:
            cycle_entry = self._state.setdefault(cycle_id, {})
            entry = cycle_entry.setdefault(component, {})
            entry.update(
                {
                    "status": status,
                    "updated_at": timestamp,
                }
            )
            if metadata:
                entry.setdefault("metadata", {}).update(metadata)
            if error:
                entry["error"] = error
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with self._state_path.open("w", encoding="utf-8") as handle:
                json.dump(self._state, handle, indent=2)

    @property
    def pending_tasks(self) -> int:
        return sum(queue.qsize() for queue in self.queues.values())

    @property
    def active_tasks(self) -> int:
        with self._active_lock:
            return self._active

    @property
    def workload_depth(self) -> int:
        return self.pending_tasks + self.active_tasks

    # ------------------------------------------------------------------
    # Concurrency helpers
    # ------------------------------------------------------------------

    def _derive_max_concurrent(self) -> int:
        env_override = os.getenv("TASK_MAX_CONCURRENT")
        if env_override:
            try:
                return max(1, int(env_override))
            except ValueError:
                pass
        profile = self.system_profile or detect_system_profile()
        slots = 3
        if profile:
            if profile.is_low_power or profile.memory_pressure:
                slots = 2
            elif profile.cpu_count >= 12 and not profile.memory_pressure:
                slots = 4
            slots = max(1, min(slots, profile.max_threads))
        return max(1, slots)

    def _acquire_slot(self) -> None:
        self._global_slots.acquire()
        with self._active_lock:
            self._active += 1

    def _release_slot(self) -> None:
        with self._active_lock:
            self._active = max(0, self._active - 1)
        self._global_slots.release()
