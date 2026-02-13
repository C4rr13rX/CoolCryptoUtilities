# REST API Benchmark Plan

## Overview
Minimal benchmarking tool to measure REST API latency and throughput.

## Quick Start
```bash
python benchmark.py <API_URL>
```

## Example
```bash
python benchmark.py https://httpbin.org/get
```

## Metrics Measured

### Latency Test
- **Method**: Sequential requests (default: 5)
- **Metrics**: Average, Min, Max response time in milliseconds
- **Success Rate**: Percentage of successful requests

### Throughput Test
- **Method**: Concurrent requests for fixed duration (default: 3s)
- **Metrics**: Requests per second
- **Success Rate**: Percentage of successful requests

## Customization
Edit `benchmark.py` to adjust:
- Request count for latency test
- Duration for throughput test
- Timeout values
- Add authentication headers
- Test different HTTP methods

## Dependencies
- Python 3.6+
- requests library

## Installation
```bash
pip install requests
```

## Sample Output
```
REST API Benchmark - 2026-02-12 20:57:49
Target: https://httpbin.org/get

Latency test: 5 requests to https://httpbin.org/get
  Request 1: 635.5ms
  Request 2: 812.2ms
  Request 3: 601.7ms
  Request 4: 571.1ms
  Request 5: 794.1ms
Throughput test: 3s to https://httpbin.org/get

=== RESULTS ===
Latency - Avg: 682.9ms, Min: 571.12ms, Max: 812.2ms
Success Rate: 100.0%
Throughput: 1.49 req/s
Success Rate: 100.0%
```