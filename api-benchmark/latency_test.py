import requests
import time
import statistics
import json
import sys

def benchmark_latency(url, num_requests=100):
    latencies = []
    errors = 0
    
    print(f"Testing latency: {url} ({num_requests} requests)")
    
    for i in range(num_requests):
        try:
            start = time.perf_counter()
            response = requests.get(url, timeout=10)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        except Exception as e:
            errors += 1
    
    if latencies:
        results = {
            "avg_ms": statistics.mean(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies),
            "errors": errors
        }
        print(f"Results: {json.dumps(results, indent=2)}")
        return results
    return None

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://httpbin.org/get"
    benchmark_latency(url)
