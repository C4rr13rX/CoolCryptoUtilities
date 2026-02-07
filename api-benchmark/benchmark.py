import requests
import time
import statistics
import concurrent.futures
import json
from datetime import datetime

class APIBenchmark:
    def __init__(self, base_url, endpoints=None):
        self.base_url = base_url.rstrip('/')
        self.endpoints = endpoints or ['/health', '/api/v1/status']
        self.results = {}
    
    def measure_latency(self, endpoint, num_requests=100):
        """Measure average latency for sequential requests"""
        url = f"{self.base_url}{endpoint}"
        latencies = []
        
        for _ in range(num_requests):
            start = time.perf_counter()
            try:
                response = requests.get(url, timeout=10)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms
            except Exception as e:
                print(f"Request failed: {e}")
                continue
        
        return {
            'avg_latency_ms': statistics.mean(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'p95_latency_ms': statistics.quantiles(latencies, n=20)[18],
            'successful_requests': len(latencies)
        }
    
    def measure_throughput(self, endpoint, concurrent_users=10, duration_seconds=30):
        """Measure throughput with concurrent requests"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        end_time = start_time + duration_seconds
        request_count = 0
        errors = 0
        
        def make_request():
            nonlocal request_count, errors
            while time.time() < end_time:
                try:
                    response = requests.get(url, timeout=5)
                    request_count += 1
                except:
                    errors += 1
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(make_request) for _ in range(concurrent_users)]
            concurrent.futures.wait(futures)
        
        actual_duration = time.time() - start_time
        return {
            'requests_per_second': request_count / actual_duration,
            'total_requests': request_count,
            'errors': errors,
            'duration_seconds': actual_duration,
            'concurrent_users': concurrent_users
        }
    
    def run_benchmark(self, latency_requests=100, throughput_users=10, throughput_duration=30):
        """Run complete benchmark suite"""
        print(f"Starting benchmark for {self.base_url}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        for endpoint in self.endpoints:
            print(f"\nTesting endpoint: {endpoint}")
            
            # Latency test
            print("  Running latency test...")
            latency_results = self.measure_latency(endpoint, latency_requests)
            
            # Throughput test
            print("  Running throughput test...")
            throughput_results = self.measure_throughput(endpoint, throughput_users, throughput_duration)
            
            self.results[endpoint] = {
                'latency': latency_results,
                'throughput': throughput_results
            }
            
            # Print results
            print(f"  Latency - Avg: {latency_results['avg_latency_ms']:.2f}ms, P95: {latency_results['p95_latency_ms']:.2f}ms")
            print(f"  Throughput - RPS: {throughput_results['requests_per_second']:.2f}, Errors: {throughput_results['errors']}")
        
        return self.results
    
    def save_results(self, filename='benchmark_results.json'):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'base_url': self.base_url,
                'results': self.results
            }, f, indent=2)
        print(f"Results saved to {filename}")

if __name__ == '__main__':
    # Example usage
    api_url = input("Enter API base URL (e.g., https://api.example.com): ")
    endpoints = input("Enter endpoints to test (comma-separated, or press Enter for defaults): ").strip()
    
    if endpoints:
        endpoints = [ep.strip() for ep in endpoints.split(',')]
    else:
        endpoints = None
    
    benchmark = APIBenchmark(api_url, endpoints)
    results = benchmark.run_benchmark()
    benchmark.save_results()
