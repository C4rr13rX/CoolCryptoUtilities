#!/usr/bin/env python3
import requests
import time
import statistics
import sys
from datetime import datetime

def latency_test(url, count=5):
    times = []
    successes = 0
    print(f"Latency test: {count} requests to {url}")
    
    for i in range(count):
        try:
            start = time.time()
            response = requests.get(url, timeout=3)
            end = time.time()
            
            if response.status_code == 200:
                times.append((end - start) * 1000)
                successes += 1
                print(f"  Request {i+1}: {times[-1]:.1f}ms")
        except Exception as e:
            print(f"  Request {i+1} failed: {e}")
    
    if times:
        return {
            "avg_ms": round(statistics.mean(times), 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
            "success_rate": round(successes / count * 100, 1)
        }
    return None

def throughput_test(url, duration=3):
    print(f"Throughput test: {duration}s to {url}")
    start_time = time.time()
    end_time = start_time + duration
    requests_made = 0
    successes = 0
    
    while time.time() < end_time:
        try:
            response = requests.get(url, timeout=2)
            requests_made += 1
            if response.status_code == 200:
                successes += 1
        except:
            requests_made += 1
    
    actual_duration = time.time() - start_time
    return {
        "requests_per_second": round(requests_made / actual_duration, 2),
        "success_rate": round(successes / requests_made * 100, 1) if requests_made > 0 else 0
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python benchmark.py <URL>")
        sys.exit(1)
    
    url = sys.argv[1]
    print(f"REST API Benchmark - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: {url}\n")
    
    latency = latency_test(url)
    throughput = throughput_test(url)
    
    print("\n=== RESULTS ===")
    if latency:
        print(f"Latency - Avg: {latency['avg_ms']}ms, Min: {latency['min_ms']}ms, Max: {latency['max_ms']}ms")
        print(f"Success Rate: {latency['success_rate']}%")
    
    if throughput:
        print(f"Throughput: {throughput['requests_per_second']} req/s")
        print(f"Success Rate: {throughput['success_rate']}%")

if __name__ == "__main__":
    main()