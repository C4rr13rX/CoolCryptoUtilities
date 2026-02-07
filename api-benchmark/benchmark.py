import requests
import time
import statistics
import concurrent.futures
import json
import argparse
from datetime import datetime

class APIBenchmark:
    def __init__(self, base_url, endpoints=None):
        self.base_url = base_url.rstrip('/')
        self.endpoints = endpoints or ['/health', '/api/v1/status']
        self.results = {}
    
    def measure_latency(self, endpoint, num_requests=100):
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
        
        if latencies:
            return {
                'avg': statistics.mean(latencies),
                'p95': statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0],
                'min': min(latencies),
                'max': max(latencies)
            }
        return {'avg': 0, 'p95': 0, 'min': 0, 'max': 0}
    
    def measure_throughput(self, endpoint, duration=30, max_workers=10):
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        end_time = start_time + duration
        successful_requests = 0
        total_requests = 0
        
        def make_request():
            try:
                response = requests.get(url, timeout=10)
                return response.status_code == 200
            except:
                return False
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            while time.time() < end_time:
                if len(futures) < max_workers:
                    future = executor.submit(make_request)
                    futures.append(future)
                
                # Check completed futures
                completed = [f for f in futures if f.done()]
                for future in completed:
                    total_requests += 1
                    if future.result():
                        successful_requests += 1
                    futures.remove(future)
                
                time.sleep(0.01)  # Small delay to prevent overwhelming
            
            # Wait for remaining futures
            for future in concurrent.futures.as_completed(futures):
                total_requests += 1
                if future.result():
                    successful_requests += 1
        
        actual_duration = time.time() - start_time
        rps = total_requests / actual_duration if actual_duration > 0 else 0
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'rps': rps,
            'success_rate': success_rate,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'duration': actual_duration
        }
    
    def run_benchmark(self, latency_requests=100, throughput_duration=30):
        print(f"Starting benchmark for {self.base_url}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print()
        
        for endpoint in self.endpoints:
            print(f"Testing endpoint: {endpoint}")
            
            # Latency test
            print("  Running latency test...")
            latency_results = self.measure_latency(endpoint, latency_requests)
            
            # Throughput test
            print("  Running throughput test...")
            throughput_results = self.measure_throughput(endpoint, throughput_duration)
            
            # Store results
            self.results[endpoint] = {
                'latency': latency_results,
                'throughput': throughput_results,
                'timestamp': datetime.now().isoformat()
            }
            
            # Print results
            print(f"  Latency - Avg: {latency_results['avg']:.2f}ms, P95: {latency_results['p95']:.2f}ms")
            print(f"  Throughput - RPS: {throughput_results['rps']:.2f}, Success Rate: {throughput_results['success_rate']:.2f}%")
            print()
        
        # Save results to file
        with open('benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("Results saved to benchmark_results.json")
        return self.results

def main():
    parser = argparse.ArgumentParser(description='REST API Benchmark Tool')
    parser.add_argument('url', help='Base URL of the API to benchmark (e.g., https://api.example.com)')
    parser.add_argument('--endpoints', nargs='+', default=['/health', '/api/v1/status'],
                       help='List of endpoints to test (default: /health /api/v1/status)')
    parser.add_argument('--latency-requests', type=int, default=100,
                       help='Number of requests for latency test (default: 100)')
    parser.add_argument('--throughput-duration', type=int, default=30,
                       help='Duration in seconds for throughput test (default: 30)')
    
    args = parser.parse_args()
    
    benchmark = APIBenchmark(args.url, args.endpoints)
    benchmark.run_benchmark(args.latency_requests, args.throughput_duration)

if __name__ == '__main__':
    main()
