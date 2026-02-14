import time

from services.performance_monitor import PerformanceMonitor


def test_performance_monitor_roundtrip(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    monitor = PerformanceMonitor(metrics_path=metrics_path)
    monitor.start_measurement("operation")
    time.sleep(0.005)
    result = monitor.end_measurement("operation")
    assert result["duration_ms"] > 0
    assert "memory_delta_mb" in result
    assert metrics_path.exists()
