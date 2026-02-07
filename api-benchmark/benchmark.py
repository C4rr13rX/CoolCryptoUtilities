import requests
import time
import statistics
import concurrent.futures
import json
from datetime import datetime

class APIBenchmark:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.results = {}
    
    def measure_latency(self, endpoint="/", num_requests=50):
        """Measure average response latency"""
        url = f"{self.base_url}{endpoint}"
        latencies = []
        
        print(f"Testing latency: {url} ({num_requests} requests)")
        
        for i in range(num_requests):
            start = time.perf_counter()
            try:
                response = requests.get(url, timeout=10)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
            except Exception as e:
                print(f"Request failed: {e}")
        
        if latencies:
            return {
                "avg_ms": round(statistics.mean(latencies), 2),
                "min_ms": round(min(latencies), 2),
                "max_ms": round(max(latencies), 2),
                "p95_ms": round(statistics.quantiles(latencies, n=20)[18], 2) if len(latencies) >= 20 else None,
                "success_rate": len(latencies) / num_requests
            }
        return None
    
    def measure_throughput(self, endpoint="/", duration_seconds=30, max_workers=10):
        """Measure requests per second under load"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        end_time = start_time + duration_seconds
        completed_requests = 0
        errors = 0
        
        print(f"Testing throughput: {url} ({duration_seconds}s, {max_workers} workers)")
        
        def make_request():
            try:
                response = requests.get(url, timeout=10)
                return response.status_code
            except:
                return None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            while time.time() < end_time:
                if len(futures) < max_workers:
                    future = executor.submit(make_request)
                    futures.append(future)
                
                # Check completed futures
                done_futures = [f for f in futures if f.done()]
                for future in done_futures:
                    result = future.result()
                    if result:
                        completed_requests += 1
                    else:
                        errors += 1
                    futures.remove(future)
                
                time.sleep(0.01)
            
            # Wait for remaining futures
            for future in futures:
                result = future.result()
                if result:
                    completed_requests += 1
                else:
                    errors += 1
        
        actual_duration = time.time() - start_time
        return {
            "requests_per_second": round(completed_requests / actual_duration, 2),
            "total_requests": completed_requests,
            "errors": errors,
            "duration_seconds": round(actual_duration, 2)
        }
    
    def run_benchmark(self, endpoint="/"):
        """Run complete benchmark suite"""
        print(f"Starting benchmark for {self.base_url}{endpoint}")
        print("=" * 50)
        
        # Test latency
        latency_results = self.measure_latency(endpoint)
        
        # Test throughput
        throughput_results = self.measure_throughput(endpoint)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": f"{self.base_url}{endpoint}",
            "latency": latency_results,
            "throughput": throughput_results
        }
        
        # Save results
        with open(f"benchmark_results_{int(time.time())}.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\nBenchmark Results:")
        print(f"Average Latency: {latency_results["avg_ms"]}ms")
        print(f"95th Percentile: {latency_results["p95_ms"]}ms")
        print(f"Throughput: {throughput_results["requests_per_second"]} req/s")
        print(f"Success Rate: {latency_results["success_rate"] * 100:.1f}%")
        
        return results

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <api_url> [endpoint]")
        print("Example: python benchmark.py https://api.example.com /health")
        sys.exit(1)
    
    api_url = sys.argv[1]
    endpoint = sys.argv[2] if len(sys.argv) > 2 else "/"
    
    benchmark = APIBenchmark(api_url)
    benchmark.run_benchmark(endpoint)
