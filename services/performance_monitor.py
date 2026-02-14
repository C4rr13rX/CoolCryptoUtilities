import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import psutil
except Exception:
    psutil = None

def _memory_mb() -> float:
    if psutil is not None:
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage / 1024.0 if usage else 0.0
    except Exception:
        return 0.0


class PerformanceMonitor:
    """Empirical performance measurement and telemetry collection."""

    def __init__(self, metrics_path: Optional[Path] = None):
        self.metrics_path = metrics_path or Path("runtime/c0d3r/metrics.json")
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.start_time: Optional[float] = None
        self.metrics: Dict[str, Dict[str, Any]] = {}
    
    def start_measurement(self, operation: str) -> None:
        """Begin performance measurement for operation."""
        self.start_time = time.perf_counter()
        self.metrics[operation] = {
            "start_time": self.start_time,
            "memory_start": _memory_mb(),
        }
    
    def end_measurement(self, operation: str) -> Dict[str, Any]:
        """Complete measurement and return telemetry."""
        if operation not in self.metrics:
            raise ValueError(f"No measurement started for {operation}")
        
        end_time = time.perf_counter()
        memory_end = _memory_mb()
        
        result = {
            "duration_ms": (end_time - self.metrics[operation]["start_time"]) * 1000,
            "memory_delta_mb": memory_end - self.metrics[operation]["memory_start"],
            "timestamp": end_time
        }
        
        self.metrics[operation].update(result)
        self._persist_metrics()
        return result
    
    def _persist_metrics(self) -> None:
        """Write metrics to persistent storage."""
        with self.metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(self.metrics, fh, indent=2)
