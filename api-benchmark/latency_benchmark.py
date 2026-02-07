import requests
import time
import statistics
import json
import sys

def benchmark_latency(url, num_requests=100):
    latencies = []
    for _ in range(num_requests):
        start = time.perf_counter()
        try:
            resp = requests.get(url, timeout=30)
            end = time.perf_counter()
            if resp.status_code == 200:
                latencies.append((end - start) * 1000)
        except Exception as e:
            continue
    
    if not latencies:
        return None
    
    return {
        "avg_ms": statistics.mean(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p95_ms": sorted(latencies)[int(0.95*len(latencies))]
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python latency_benchmark.py <url> [num_requests]")
        sys.exit(1)
    
    url = sys.argv[1]
    num_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    result = benchmark_latency(url, num_requests)
    print(json.dumps(result, indent=2))
