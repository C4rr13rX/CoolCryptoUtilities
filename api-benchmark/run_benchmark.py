import subprocess
import json
import sys
import os
from datetime import datetime

def run_benchmark(url):
    print(f"Benchmarking API: {url}")
    print("=" * 50)
    
    # Run latency test
    print("\nRunning latency test (100 requests)...")
    latency_result = subprocess.run([sys.executable, "latency_benchmark.py", url, "100"], 
                                   capture_output=True, text=True)
    
    if latency_result.returncode == 0:
        latency_data = json.loads(latency_result.stdout)
        print(f"Average latency: {latency_data["avg_ms"]:.2f}ms")
        print(f"P95 latency: {latency_data["p95_ms"]:.2f}ms")
    else:
        print("Latency test failed")
        latency_data = None
    
    # Run throughput test
    print("\nRunning throughput test (1000 requests, 50 concurrent)...")
    throughput_result = subprocess.run([sys.executable, "throughput_benchmark.py", url, "1000", "50"], 
                                      capture_output=True, text=True)
    
    if throughput_result.returncode == 0:
        throughput_data = json.loads(throughput_result.stdout)
        print(f"Requests per second: {throughput_data["requests_per_second"]:.2f}")
        print(f"Success rate: {throughput_data["success_rate"]:.1f}%")
    else:
        print("Throughput test failed")
        throughput_data = None
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"results/benchmark_{timestamp}.json"
    
    results = {
        "url": url,
        "timestamp": timestamp,
        "latency": latency_data,
        "throughput": throughput_data
    }
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {result_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_benchmark.py <url>")
        sys.exit(1)
    
    run_benchmark(sys.argv[1])
