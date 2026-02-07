# REST API Benchmark Tool

A minimal Python tool for benchmarking REST API latency and throughput.

## Features

- **Latency Testing**: Measures average, P95, min, and max response times
- **Throughput Testing**: Measures requests per second and success rates
- **Concurrent Testing**: Uses thread pools for realistic load simulation
- **JSON Output**: Saves detailed results for analysis
- **Flexible Configuration**: Customizable endpoints, request counts, and test duration

## Installation

```bash
pip install requests
```

## Usage

### Basic Usage
```bash
python benchmark.py https://api.example.com
```

### Custom Endpoints
```bash
python benchmark.py https://api.example.com --endpoints /health /api/v1/users /api/v1/data
```

### Quick Test (5 requests, 5 seconds)
```bash
python benchmark.py https://httpbin.org --endpoints /status/200 --latency-requests 5 --throughput-duration 5
```

### Production Load Test
```bash
python benchmark.py https://api.example.com --latency-requests 1000 --throughput-duration 60
```

## Output

The tool provides:
- Console output with real-time results
- JSON file (`benchmark_results.json`) with detailed metrics
- Timestamp and configuration details

## Metrics Explained

- **Latency**: Response time in milliseconds
  - Avg: Average response time
  - P95: 95th percentile (95% of requests faster than this)
  - Min/Max: Fastest and slowest requests

- **Throughput**: Request handling capacity
  - RPS: Requests per second
  - Success Rate: Percentage of successful requests

## Best Practices

1. **Start Small**: Begin with low request counts to avoid overwhelming the API
2. **Monitor Resources**: Watch CPU and network usage during tests
3. **Test Realistic Scenarios**: Use actual endpoints your application calls
4. **Consider Rate Limits**: Respect API rate limiting policies
5. **Multiple Runs**: Run tests multiple times for consistent results

## Example Results

```
Starting benchmark for https://httpbin.org
Timestamp: 2026-02-06T20:17:28.898669

Testing endpoint: /status/200
  Running latency test...
  Running throughput test...
  Latency - Avg: 1007.34ms, P95: 1467.10ms
  Throughput - RPS: 4.73, Success Rate: 100.00%

Results saved to benchmark_results.json
```
