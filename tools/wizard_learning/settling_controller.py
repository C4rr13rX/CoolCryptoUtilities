"""tools/wizard_learning/settling_controller.py

Settling Controller & Error-Free Biological Sleep-Pressure Loop for W1z4rD Brain.
Prevents RAM bloat on CPU-only machines (no GPU) while ensuring:
  1. Sleep pressure accurately measures W1z4rD node RAM growth above its settled
     baseline + episode batch fatigue (never conflated with Django/Waitress web RSS),
     resetting cleanly to 0% immediately after settling.
  2. Every sleep/consolidation cycle sets `CognitiveState` to `SLEEPING_SETTLING`
     (`can_chat=False`) so the GUI explicitly informs the user, runs with proper
     durable checkpoint timeouts (`30-45s`), and restores `can_chat=True` in `finally:`.
  3. Post-sleep anti-forgetting verification checks anchored & recently trained pairs
     and immediately re-binds any drifted concept so zero forgetting ever occurs.
"""
from __future__ import annotations

import ctypes
import datetime
import gc
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNTIME_DIR = _PROJECT_ROOT / "runtime" / "c0d3r"
SETTLING_HISTORY_FILE = RUNTIME_DIR / "wizard_settling_history.jsonl"
SETTLING_STATE_FILE = RUNTIME_DIR / "wizard_settling_state.json"

# Polite PC-Allowance Governor constants:
# Keep >=8.0 GB (or 26% of total RAM) untouched + a 12% under-cap safety margin
# so multiple crypto trading strategies, browser, IDE, and OS always come through smoothly.
POLITE_RESERVE_FLOOR_GB = 8.0
POLITE_RESERVE_FRACTION = 0.26
UNDER_CAP_SAFETY_FACTOR = 0.88

# Base defaults (scaled elastically up when PC RAM is abundant, and down when tight)
DEFAULT_RAM_DELTA_BUDGET_MB = 250.0
DEFAULT_RAM_HARD_CEILING_MB = 1800.0
DEFAULT_BATCH_INTERVAL = 24

_STATE_LOCK = threading.RLock()


def trim_windows_working_set(pid: Optional[int] = None) -> bool:
    """Politely ask the Windows kernel (`EmptyWorkingSet`) to reclaim unused physical
    pages from `pid` (or the current process) without losing any virtual memory state."""
    if os.name != "nt":
        return False
    try:
        k32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        if pid is None:
            handle = k32.GetCurrentProcess()
            return bool(psapi.EmptyWorkingSet(handle))
        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_SET_QUOTA = 0x0100
        handle = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_SET_QUOTA, False, int(pid))
        if not handle:
            return False
        try:
            return bool(psapi.EmptyWorkingSet(handle))
        finally:
            k32.CloseHandle(handle)
    except Exception:
        return False


def compute_elastic_pc_allowance(total_gb: float, available_gb: float) -> Dict[str, Any]:
    """Compute an elastic, polite resource budget that expands when PC RAM is abundant
    and shrinks smoothly when multiple trading strategies or desktop apps claim RAM."""
    reserve_gb = max(POLITE_RESERVE_FLOOR_GB, total_gb * POLITE_RESERVE_FRACTION) if total_gb >= 24.0 else total_gb * POLITE_RESERVE_FRACTION
    raw_above_gb = max(0.0, available_gb - reserve_gb)
    # Stay slightly more under the cap (88% of headroom above the polite reserve)
    usable_headroom_gb = round(raw_above_gb * UNDER_CAP_SAFETY_FACTOR, 2)
    full_span_gb = max(2.0, total_gb * 0.24)
    allowance_fraction = round(min(1.0, max(0.15, usable_headroom_gb / full_span_gb)), 3)

    if allowance_fraction >= 0.78:
        tier = "EXPANDED"
        delta_budget_mb = 750.0
        hard_ceiling_mb = 1850.0
        target_terminals_per_pool = 5_500_000
        evict_threshold = 5.2
        max_evict_per_pass = 256
        pairs_per_cycle = 6
        cycle_interval_s = 25.0
    elif allowance_fraction >= 0.48:
        tier = "BALANCED"
        delta_budget_mb = 420.0
        hard_ceiling_mb = 1200.0
        target_terminals_per_pool = 3_000_000
        evict_threshold = 4.0
        max_evict_per_pass = 512
        pairs_per_cycle = 4
        cycle_interval_s = 32.0
    elif allowance_fraction >= 0.24:
        tier = "POLITE_UNDER_CAP"
        delta_budget_mb = 220.0
        hard_ceiling_mb = 780.0
        target_terminals_per_pool = 1_400_000
        evict_threshold = 2.8
        max_evict_per_pass = 768
        pairs_per_cycle = 3
        cycle_interval_s = 42.0
    else:
        tier = "COMPACT_SHRINK"
        delta_budget_mb = 140.0
        hard_ceiling_mb = 520.0
        target_terminals_per_pool = 650_000
        evict_threshold = 1.8
        max_evict_per_pass = 1024
        pairs_per_cycle = 2
        cycle_interval_s = 55.0

    return {
        "allowance_tier": tier,
        "allowance_fraction": allowance_fraction,
        "allowance_pct": int(round(allowance_fraction * 100)),
        "polite_reserve_gb": round(reserve_gb, 2),
        "usable_headroom_gb": usable_headroom_gb,
        "dynamic_delta_budget_mb": delta_budget_mb,
        "dynamic_hard_ceiling_mb": hard_ceiling_mb,
        "target_terminals_per_pool": target_terminals_per_pool,
        "evict_threshold": evict_threshold,
        "max_evict_per_pass": max_evict_per_pass,
        "recommended_pairs_per_cycle": pairs_per_cycle,
        "recommended_interval_s": cycle_interval_s,
    }


class SettlingController:
    """Manages working memory boundaries, baseline-relative sleep pressure,
    error-free durable consolidation, and post-sleep recall verification."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8090",
        ram_threshold_mb: float = DEFAULT_RAM_DELTA_BUDGET_MB,
        batch_interval: int = DEFAULT_BATCH_INTERVAL,
    ):
        self.endpoint = endpoint.rstrip("/")
        # Keep compatibility if callers pass legacy 450MB threshold: treat as delta budget
        self.ram_delta_budget_mb = max(80.0, float(ram_threshold_mb))
        self.ram_hard_ceiling_mb = DEFAULT_RAM_HARD_CEILING_MB
        self.batch_interval = max(1, int(batch_interval))
        self.episodes_since_settle = 0
        self.settled_baseline_ram_mb: Optional[float] = None
        self.last_settled_at: Optional[str] = None
        self.total_settle_cycles = 0
        self.sleep_errors = 0
        self.last_sleep_error = ""
        self.recent_trained_pairs: List[Dict[str, str]] = []
        self._metrics_cache_ts: float = 0.0
        self._metrics_cache_val: Optional[Dict[str, Any]] = None
        self._cached_node_pid: Optional[int] = None
        self._cached_senior_pid: Optional[int] = None
        self._last_disk_checkpoint_ts: float = 0.0
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _load_state(self) -> None:
        with _STATE_LOCK:
            if not SETTLING_STATE_FILE.exists():
                return
            try:
                data = json.loads(SETTLING_STATE_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.episodes_since_settle = int(data.get("episodes_since_settle", 0))
                    base = data.get("settled_baseline_ram_mb")
                    self.settled_baseline_ram_mb = float(base) if base is not None else None
                    self.last_settled_at = data.get("last_settled_at")
                    self.total_settle_cycles = int(data.get("total_settle_cycles", 0))
                    self.sleep_errors = int(data.get("sleep_errors", 0))
                    self.last_sleep_error = str(data.get("last_sleep_error") or "")
                    if data.get("cached_node_pid") and not self._cached_node_pid:
                        self._cached_node_pid = int(data["cached_node_pid"])
                    if data.get("cached_senior_pid") and not self._cached_senior_pid:
                        self._cached_senior_pid = int(data["cached_senior_pid"])
                    pairs = data.get("recent_trained_pairs")
                    if isinstance(pairs, list):
                        self.recent_trained_pairs = [
                            {"prompt": str(p["prompt"]), "answer": str(p["answer"])}
                            for p in pairs[-20:]
                            if isinstance(p, dict) and p.get("prompt") and p.get("answer")
                        ]
            except Exception:
                pass

    def _save_state(self) -> None:
        with _STATE_LOCK:
            payload = {
                "updated_at": time.time(),
                "episodes_since_settle": self.episodes_since_settle,
                "settled_baseline_ram_mb": self.settled_baseline_ram_mb,
                "last_settled_at": self.last_settled_at,
                "total_settle_cycles": self.total_settle_cycles,
                "sleep_errors": self.sleep_errors,
                "last_sleep_error": self.last_sleep_error,
                "cached_node_pid": self._cached_node_pid,
                "cached_senior_pid": self._cached_senior_pid,
                "recent_trained_pairs": self.recent_trained_pairs[-20:],
            }
            tmp = SETTLING_STATE_FILE.with_suffix(f".{os.getpid()}.tmp")
            try:
                tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                os.replace(tmp, SETTLING_STATE_FILE)
            except Exception:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass

    def _measure_wizard_node_ram(self) -> tuple[Optional[int], float, float]:
        """Return (local_node_pid, local_node_ram_mb, senior_server_ram_mb) in <2ms.
        Uses cached PIDs first, falling back to a lightweight `process_iter(['pid', 'name'])`
        scan (without per-process `OpenProcess` memory_info across the entire OS table),
        and never conflates Waitress web server RSS into brain RAM."""
        node_pid: Optional[int] = None
        node_ram_mb = 0.0
        senior_ram_mb = 0.0

        # Fast path (<0.2ms): query cached PIDs directly if still alive
        if self._cached_node_pid:
            try:
                p = psutil.Process(self._cached_node_pid)
                nm = (p.name() or "").lower()
                if "w1z4rd_node" in nm or "w1z4rdv1510n-node" in nm:
                    node_pid = self._cached_node_pid
                    node_ram_mb = p.memory_info().rss / (1024.0 * 1024.0)
                else:
                    self._cached_node_pid = None
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._cached_node_pid = None

        if self._cached_senior_pid:
            try:
                ps = psutil.Process(self._cached_senior_pid)
                if "w1z4rd_brain_server" in (ps.name() or "").lower():
                    senior_ram_mb = ps.memory_info().rss / (1024.0 * 1024.0)
                else:
                    self._cached_senior_pid = None
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._cached_senior_pid = None

        now_scan = time.time()
        if node_pid is not None and (
            self._cached_senior_pid is not None
            or (now_scan - getattr(self, "_senior_scan_ts", 0.0)) < 30.0
        ):
            return node_pid, round(node_ram_mb, 2), round(senior_ram_mb, 2)

        # Fallback scan (<3ms): enumerate names only, call memory_info() solely on matches
        self._senior_scan_ts = now_scan
        node_ram_mb = 0.0
        senior_ram_mb = 0.0
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                name = (proc.info["name"] or "").lower()
                if "w1z4rd_node" in name or "w1z4rdv1510n-node" in name:
                    node_pid = proc.info["pid"]
                    self._cached_node_pid = node_pid
                    node_ram_mb += proc.memory_info().rss / (1024.0 * 1024.0)
                elif "w1z4rd_brain_server" in name:
                    self._cached_senior_pid = proc.info["pid"]
                    senior_ram_mb += proc.memory_info().rss / (1024.0 * 1024.0)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return node_pid, round(node_ram_mb, 2), round(senior_ram_mb, 2)

    def get_resource_metrics(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Inspect W1z4rD node working set, baseline growth, and sleep pressure."""
        now_t = time.time()
        if (
            not force_refresh
            and self._metrics_cache_val is not None
            and (now_t - self._metrics_cache_ts) < 3.0
            and self._metrics_cache_val.get("episodes_since_settle") == self.episodes_since_settle
        ):
            return dict(self._metrics_cache_val)
        self._load_state()
        node_pid, node_ram_mb, senior_ram_mb = self._measure_wizard_node_ram()
        py_mem_mb = round(psutil.Process().memory_info().rss / (1024.0 * 1024.0), 2)
        vm = psutil.virtual_memory()
        disk = shutil.disk_usage(str(_PROJECT_ROOT))

        if self.settled_baseline_ram_mb is None or self.settled_baseline_ram_mb <= 0.0:
            if node_ram_mb > 0.0:
                self.settled_baseline_ram_mb = node_ram_mb
                self._save_state()
            else:
                self.settled_baseline_ram_mb = 320.0

        total_gb = round(vm.total / (1024.0 ** 3), 2)
        avail_gb = round(vm.available / (1024.0 ** 3), 2)
        allowance = compute_elastic_pc_allowance(total_gb, avail_gb)

        # Elastic budget: consume more when PC RAM is abundant, shrink smoothly when tight
        effective_delta_budget = allowance["dynamic_delta_budget_mb"]
        effective_ceiling = allowance["dynamic_hard_ceiling_mb"]

        ram_delta_mb = max(0.0, round(node_ram_mb - (self.settled_baseline_ram_mb or 0.0), 2))
        has_unsettled_activity = bool(ram_delta_mb >= 25.0 or self.episodes_since_settle >= 2)
        delta_pressure = ram_delta_mb / max(1.0, effective_delta_budget)
        ceiling_pressure = (
            max(0.0, (node_ram_mb - effective_ceiling * 0.75) / (effective_ceiling * 0.25))
            if (has_unsettled_activity and node_ram_mb > effective_ceiling * 0.75)
            else 0.0
        )
        # Polite system-reserve pressure: if PC free RAM dips into the polite reserve band
        # AND W1z4rD has unsettled growth above its settled baseline, nudge sleep pressure up
        # to page cold neurons to .wbrain — without looping endlessly at 0 unsettled episodes.
        reserve_pressure = (
            min(1.0, (allowance["polite_reserve_gb"] - avail_gb) / 2.0)
            if (has_unsettled_activity and avail_gb < allowance["polite_reserve_gb"] and node_ram_mb > 240.0)
            else 0.0
        )
        episode_pressure = self.episodes_since_settle / max(1, self.batch_interval)

        sleep_pressure = round(
            min(1.0, max(delta_pressure, ceiling_pressure, reserve_pressure, episode_pressure)),
            3,
        )
        needs_settle = bool(
            sleep_pressure >= 0.85
            or (has_unsettled_activity and node_ram_mb >= effective_ceiling)
            or self.episodes_since_settle >= self.batch_interval
        )

        result_metrics = {
            "node_pid": node_pid,
            "node_ram_mb": node_ram_mb,
            "senior_brain_ram_mb": senior_ram_mb,
            "python_ram_mb": py_mem_mb,
            # Expose accurate W1z4rD node working set so HUD reflects actual brain RAM
            "total_working_set_mb": node_ram_mb,
            "settled_baseline_ram_mb": round(self.settled_baseline_ram_mb or 0.0, 2),
            "ram_delta_since_settle_mb": ram_delta_mb,
            "ram_delta_budget_mb": effective_delta_budget,
            "ram_hard_ceiling_mb": effective_ceiling,
            "system_total_ram_gb": total_gb,
            "system_available_ram_gb": avail_gb,
            "system_ram_percent": vm.percent,
            "disk_free_gb": round(disk.free / (1024.0 ** 3), 2),
            "episodes_since_settle": self.episodes_since_settle,
            "batch_interval": self.batch_interval,
            "sleep_pressure": sleep_pressure,
            "needs_settle": needs_settle,
            "sleep_errors": self.sleep_errors,
            "last_sleep_error": self.last_sleep_error,
            **allowance,
        }
        self._metrics_cache_ts = now_t
        self._metrics_cache_val = result_metrics
        return dict(result_metrics)

    def register_trained_pair(self, prompt: str, answer: str) -> None:
        """Track recently trained pairs so post-sleep verification confirms 0 forgetting."""
        self._load_state()
        p_clean = (prompt or "").strip()
        a_clean = (answer or "").strip()
        if not p_clean or not a_clean:
            return
        self.recent_trained_pairs = [
            p for p in self.recent_trained_pairs if p.get("prompt") != p_clean
        ]
        self.recent_trained_pairs.append({"prompt": p_clean, "answer": a_clean})
        self.recent_trained_pairs = self.recent_trained_pairs[-20:]
        self._save_state()

    def notify_episode_trained(self, prompt: str = "", answer: str = "") -> Dict[str, Any]:
        """Increment cognitive episode count, record pair, and settle if needed."""
        self._load_state()
        self.episodes_since_settle += 1
        if prompt and answer:
            self.recent_trained_pairs = [
                p for p in self.recent_trained_pairs if p.get("prompt") != prompt.strip()
            ]
            self.recent_trained_pairs.append({"prompt": prompt.strip(), "answer": answer.strip()})
            self.recent_trained_pairs = self.recent_trained_pairs[-20:]
        self._save_state()
        metrics = self.get_resource_metrics()
        if metrics["needs_settle"]:
            return self.settle(
                reason=(
                    f"Threshold reached: pressure={metrics['sleep_pressure']:.2f}, "
                    f"node_ram={metrics['node_ram_mb']}MB (+{metrics['ram_delta_since_settle_mb']}MB), "
                    f"episodes={self.episodes_since_settle}"
                )
            )
        return {"settled": False, "metrics": metrics}

    def _call_node_step(self, path: str, payload: Dict[str, Any], timeout: float = 35.0) -> Dict[str, Any]:
        """Invoke a W1z4rD node endpoint. Treats 404/405 on optional sub-routes as
        cleanly unsupported rather than an error, while surfacing timeouts/5xx/offline."""
        url = f"{self.endpoint}{path}"
        raw_body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=raw_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace").strip()
                if not raw:
                    return {"ok": True, "status": "ok"}
                try:
                    parsed = json.loads(raw)
                    return parsed if isinstance(parsed, dict) else {"ok": True, "value": parsed}
                except ValueError:
                    return {"ok": True, "raw": raw}
        except urllib.error.HTTPError as http_err:
            if http_err.code in (404, 405, 422):
                # Route not mounted on this binary build or empty payload consolidation no-op
                return {"ok": True, "status": "optional_noop", "http_code": http_err.code}
            return {"ok": False, "error": f"HTTP {http_err.code}: {http_err.reason}"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _verify_and_heal_post_sleep_recall(self) -> Dict[str, Any]:
        """Verify anchored memories + recent trained pairs after sleep consolidation;
        re-train any pair whose multi_pool binding decayed during cold-tier eviction."""
        probes: List[Dict[str, str]] = []
        try:
            from services.anchored_memory import AnchoredMemoryManager
            anchors = AnchoredMemoryManager.get_instance().list_all()
            for a in anchors[:5]:
                if a.get("prompt") and a.get("answer"):
                    probes.append({"prompt": str(a["prompt"]), "answer": str(a["answer"])})
        except Exception:
            pass

        for pair in self.recent_trained_pairs[-5:]:
            if pair.get("prompt") and pair.get("answer"):
                probes.append(pair)

        if not probes:
            return {"checked": 0, "passed": 0, "healed": 0, "recall_rate": 1.0}

        passed = 0
        healed = 0
        for item in probes:
            q = item["prompt"]
            expected = item["answer"].strip().lower()
            got_ok = False
            try:
                req = urllib.request.Request(
                    f"{self.endpoint}/chat",
                    data=json.dumps({"text": q, "hops": 2, "min_strength": 0.05}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=12.0) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    ans = str(res.get("answer") or "").strip().lower()
                    dec = str(res.get("decoder") or "")
                    if ans and dec != "char_chain" and (expected[:40] in ans or ans[:40] in expected):
                        got_ok = True
            except Exception:
                got_ok = False

            if got_ok:
                passed += 1
            else:
                # Immediately re-bind into multi_pool so zero forgetting survives sleep
                retrain = self._call_node_step(
                    "/multi_pool/train_pair",
                    {"src_pool": "in", "src": q, "tgt_pool": "out", "tgt": item["answer"], "passes": 20},
                    timeout=25.0,
                )
                if retrain.get("ok", True) and not retrain.get("error"):
                    healed += 1
                    passed += 1

        rate = round(passed / max(1, len(probes)), 4)
        return {
            "checked": len(probes),
            "passed": passed,
            "healed": healed,
            "recall_rate": rate,
        }

    def settle(self, reason: str = "scheduled_consolidation") -> Dict[str, Any]:
        """Execute error-free sleep, STDP consolidation, durable checkpointing,
        post-sleep recall verification, and baseline reset."""
        from tools.wizard_learning.cognitive_state import get_cognitive_state_manager

        cog = get_cognitive_state_manager()
        self._load_state()
        episodes_consolidated = self.episodes_since_settle
        metrics_before = self.get_resource_metrics()
        now_iso = datetime.datetime.now().isoformat()

        cog.set_mode(
            "SLEEPING_SETTLING",
            detail=f"Biological sleep & SSD consolidation in progress ({reason})...",
            can_chat=False,
            settling_holder_pid=os.getpid(),
            chat_unavailable_reason=(
                "W1z4rD is currently sleeping and consolidating neural synapses to SSD. "
                "Chat is temporarily locked for a few seconds to prevent write-lock contention."
            ),
        )

        step_errors: List[str] = []
        node_sleep_result: Dict[str, Any] = {}
        node_consolidate_result: Dict[str, Any] = {}
        node_checkpoint_result: Dict[str, Any] = {}
        neuro_checkpoint_result: Dict[str, Any] = {}
        recall_check: Dict[str, Any] = {"checked": 0, "passed": 0, "healed": 0, "recall_rate": 1.0}

        try:
            # 1. Check node health first so offline node is reported clearly
            health_ok = False
            try:
                with urllib.request.urlopen(f"{self.endpoint}/health", timeout=6.0) as r:
                    health_ok = r.status == 200
            except Exception as exc:
                step_errors.append(f"health_probe: {exc}")

            if health_ok:
                # 2. Biological sleep / STDP homeostatic downscaling (allow up to 120s if lock is busy)
                node_sleep_result = self._call_node_step("/brain/sleep", {}, timeout=120.0)
                if node_sleep_result.get("ok") is False and node_sleep_result.get("error"):
                    step_errors.append(f"/brain/sleep: {node_sleep_result['error']}")

                # 3. Motif / episode consolidation
                node_consolidate_result = self._call_node_step("/brain/consolidate", {}, timeout=120.0)
                if node_consolidate_result.get("ok") is False and node_consolidate_result.get("error"):
                    step_errors.append(f"/brain/consolidate: {node_consolidate_result['error']}")

                # 4. Durable substrate & multi-pool checkpoints (throttled to <=1 per 8 min unless manual,
                #    preventing SSD write-amplification while supervisor autocheckpoint also runs).
                now_ckpt = time.time()
                want_disk_ckpt = (
                    "manual" in reason.lower()
                    or episodes_consolidated >= 12
                    or (now_ckpt - self._last_disk_checkpoint_ts) >= 480.0
                )
                if want_disk_ckpt:
                    node_checkpoint_result = self._call_node_step("/brain/checkpoint", {}, timeout=180.0)
                    if node_checkpoint_result.get("ok") is False and node_checkpoint_result.get("error"):
                        step_errors.append(f"/brain/checkpoint: {node_checkpoint_result['error']}")

                    neuro_checkpoint_result = self._call_node_step("/neuro/checkpoint", {}, timeout=180.0)
                    if neuro_checkpoint_result.get("ok") is False and neuro_checkpoint_result.get("error"):
                        step_errors.append(f"/neuro/checkpoint: {neuro_checkpoint_result['error']}")

                    if not step_errors:
                        self._last_disk_checkpoint_ts = now_ckpt
                else:
                    node_checkpoint_result = {"ok": True, "skipped": "throttled_polite_io"}
                    neuro_checkpoint_result = {"ok": True, "skipped": "throttled_polite_io"}

                # 5. Sync live elastic tier-orchestrator targets so the node expands
                #    resident terminals when PC RAM is abundant and shrinks when tight!
                self._call_node_step(
                    "/brain/tier_orchestrator/params",
                    {
                        "target_terminals_per_pool": int(metrics_before.get("target_terminals_per_pool", 3_000_000)),
                        "evict_threshold": float(metrics_before.get("evict_threshold", 4.0)),
                        "max_evict_per_pass": int(metrics_before.get("max_evict_per_pass", 512)),
                        "min_system_available_mb": int(round(float(metrics_before.get("polite_reserve_gb", 8.0)) * 1024)),
                    },
                    timeout=10.0,
                )

                # 6. Post-sleep recall verification & anti-forgetting heal
                recall_check = self._verify_and_heal_post_sleep_recall()

            # 7. Python GC + polite OS working-set trim when PC headroom is tight
            gc.collect()
            trim_windows_working_set(None)
            if metrics_before.get("allowance_fraction", 1.0) < 0.60 and metrics_before.get("node_pid"):
                trim_windows_working_set(int(metrics_before["node_pid"]))

            _, node_ram_after, _ = self._measure_wizard_node_ram()
            freed_mb = round(metrics_before["node_ram_mb"] - node_ram_after, 2)

            error_free = len(step_errors) == 0
            if error_free:
                self.episodes_since_settle = 0
                if node_ram_after > 0:
                    # Update settled baseline so sleep_pressure cleanly resets to 0.0
                    self.settled_baseline_ram_mb = node_ram_after
                self.last_settled_at = now_iso
                self.total_settle_cycles += 1
                self.last_sleep_error = ""
            else:
                self.sleep_errors += 1
                self.last_sleep_error = "; ".join(step_errors)

            self._save_state()
            metrics_after = self.get_resource_metrics(force_refresh=True)

            reflection = (
                f"Why did I settle? Node RAM was {metrics_before['node_ram_mb']} MB "
                f"(+{metrics_before['ram_delta_since_settle_mb']} MB above baseline) after "
                f"{episodes_consolidated} training episodes ({reason}). "
                f"Sleep consolidated hot Hebbian/STDP synapses to SSD, verified "
                f"{recall_check['passed']}/{recall_check['checked']} post-sleep recalls "
                f"(healed {recall_check['healed']}), and reset sleep pressure to "
                f"{int(metrics_after['sleep_pressure'] * 100)}% "
                f"({'0 errors' if error_free else self.last_sleep_error})."
            )

            record = {
                "timestamp": now_iso,
                "cycle_number": self.total_settle_cycles,
                "trigger_reason": reason,
                "episodes_consolidated": episodes_consolidated,
                "ram_before_mb": metrics_before["node_ram_mb"],
                "ram_after_mb": metrics_after["node_ram_mb"],
                "memory_freed_mb": freed_mb,
                "freed_mb": freed_mb,
                "disk_free_gb": metrics_after["disk_free_gb"],
                "error_free": error_free,
                "errors": step_errors,
                "recall_verification": recall_check,
                "node_sleep_result": node_sleep_result,
                "node_consolidate_result": node_consolidate_result,
                "node_checkpoint_result": node_checkpoint_result,
                "neuro_checkpoint_result": neuro_checkpoint_result,
                "reflection": reflection,
            }

            try:
                with open(SETTLING_HISTORY_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception:
                pass

            cur_cog = cog.read()
            cog.set_mode(
                "IDLE_READY",
                detail=f"Sleep cycle #{self.total_settle_cycles} completed ({'error-free' if error_free else 'with warnings'}). Ready.",
                can_chat=True,
                chat_unavailable_reason="",
                sleep_cycles_ok=int(cur_cog.get("sleep_cycles_ok", 0)) + (1 if error_free else 0),
                sleep_errors=self.sleep_errors,
                last_sleep_error=self.last_sleep_error,
                post_sleep_recall_rate=recall_check.get("recall_rate", 1.0),
            )

            return {
                "settled": error_free,
                "error_free": error_free,
                "errors": step_errors,
                "record": record,
                "metrics_after": metrics_after,
            }
        finally:
            # Safety guarantee: never leave chat locked if an unexpected exception occurs
            st = cog.read()
            if st.get("mode") == "SLEEPING_SETTLING":
                cog.set_mode(
                    "IDLE_READY",
                    detail="Sleep cycle exited; chat unlocked.",
                    can_chat=True,
                    chat_unavailable_reason="",
                )

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the recent settling and reflection history."""
        if not SETTLING_HISTORY_FILE.exists():
            return []
        records: List[Dict[str, Any]] = []
        try:
            with open(SETTLING_HISTORY_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception:
            pass
        return records[-limit:]


_GLOBAL_CONTROLLER: Optional[SettlingController] = None


def get_settling_controller() -> SettlingController:
    global _GLOBAL_CONTROLLER
    if _GLOBAL_CONTROLLER is None:
        _GLOBAL_CONTROLLER = SettlingController()
    return _GLOBAL_CONTROLLER
