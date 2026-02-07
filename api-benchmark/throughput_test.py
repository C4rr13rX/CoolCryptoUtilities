import requests
import time
import threading
import json
import sys
from concurrent.futures import ThreadPoolExecutor

def single_request(url):
    try:
        start = time.perf_counter()
        response = requests.get(url, timeout=10)
        end = time.perf_counter()
        return {"success": True, "duration": end - start, "size": len(response.content)}
    except:
        return {"success": False, "duration": 0, "size": 0}

def benchmark_throughput(url, duration=30, workers=5):
    print(f"Testing throughput: {url} ({duration}s, {workers} workers)")
    
    results = []
    end_time = time.time() + duration
    
    def worker():
        while time.time() < end_time:
            results.append(single_request(url))
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker) for _ in range(workers)]
        time.sleep(duration)
    
    successful = [r for r in results if r["success"]]
    if successful:
        metrics = {
            "total_requests": len(results),
            "successful_requests": len(successful),
            "requests_per_second": len(successful) / duration,
            "avg_response_ms": sum(r["duration"] for r in successful) / len(successful) * 1000,
            "total_mb": sum(r["size"] for r in successful) / (1024*1024)
        }
        print(f"Results: {json.dumps(metrics, indent=2)}")
        return metrics
    return None

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://httpbin.org/get"
    benchmark_throughput(url)
