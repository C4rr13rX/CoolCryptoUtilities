import requests
import time
import statistics
import concurrent.futures
import json
from datetime import datetime

class APIBenchmark:
    def __init__(self, base_url, endpoints):
        self.base_url = base_url
        self.endpoints = endpoints
        self.results = {}
    
    def test_latency(self, endpoint, num_requests=100):
        url = f"{self.base_url}{endpoint}"
        latencies = []
        
        for _ in range(num_requests):
            start = time.perf_counter()
            try:
                response = requests.get(url, timeout=30)
                end = time.perf_counter()
                if response.status_code == 200:
                    latencies.append((end - start) * 1000)
            except Exception as e:
                print(f"Request failed: {e}")
        
        return {
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0,
            "successful_requests": len(latencies),
            "total_requests": num_requests
        }
    
    def test_throughput(self, endpoint, concurrent_users=10, duration_seconds=30):
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        end_time = start_time + duration_seconds
        successful_requests = 0
        total_requests = 0
        
        def make_request():
            nonlocal successful_requests, total_requests
            while time.time() < end_time:
                try:
                    response = requests.get(url, timeout=10)
                    total_requests += 1
                    if response.status_code == 200:
                        successful_requests += 1
                except:
                    total_requests += 1
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(make_request) for _ in range(concurrent_users)]
            concurrent.futures.wait(futures)
        
        actual_duration = time.time() - start_time
        return {
            "requests_per_second": successful_requests / actual_duration,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
            "duration_seconds": actual_duration,
            "concurrent_users": concurrent_users
        }
    
    def run_benchmark(self, latency_requests=100, throughput_users=10, throughput_duration=30):
        print(f"Starting API benchmark at {datetime.now()}")
        print(f"Target: {self.base_url}")
        
        for endpoint in self.endpoints:
            print(f"Testing endpoint: {endpoint}")
            latency_results = self.test_latency(endpoint, latency_requests)
            throughput_results = self.test_throughput(endpoint, throughput_users, throughput_duration)
            
            self.results[endpoint] = {
                "latency": latency_results,
                "throughput": throughput_results
            }
        
        return self.results
    
    def save_results(self, filename="benchmark_results.json"):
        with open(filename, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "base_url": self.base_url,
                "results": self.results
            }, f, indent=2)
        print(f"Results saved to {filename}")
    
    def print_summary(self):
        print("BENCHMARK SUMMARY")
        for endpoint, data in self.results.items():
            print(f"Endpoint: {endpoint}")
            print(f"  Avg Latency: {data["latency"]["avg_latency_ms"]:.2f}ms")
            print(f"  P95 Latency: {data["latency"]["p95_latency_ms"]:.2f}ms")
            print(f"  Throughput: {data["throughput"]["requests_per_second"]:.2f} RPS")
            print(f"  Success Rate: {data["throughput"]["success_rate"]:.1f}%")

if __name__ == "__main__":
    BASE_URL = "https://jsonplaceholder.typicode.com"
    ENDPOINTS = ["/posts/1", "/users/1"]
    
    benchmark = APIBenchmark(BASE_URL, ENDPOINTS)
    results = benchmark.run_benchmark(latency_requests=50, throughput_users=5, throughput_duration=10)
    benchmark.print_summary()
    benchmark.save_results()
