# REST API Benchmark Suite

Minimal tools for benchmarking REST API latency and throughput.

## Quick Start

```bash
# Install dependencies
pip install requests

# Test latency (default: 100 requests)
python latency_test.py
python latency_test.py https://api.example.com/endpoint 50

# Test throughput (default: 10 concurrent, 100 total)
python throughput_test.py
python throughput_test.py https://api.example.com/endpoint 20 200
```

## Output Metrics

### Latency Test
- Average, min, max response times
- Median and 95th percentile
- Error count

### Throughput Test
- Requests per second
- Total duration
- Success/error rates
- Concurrent connection performance

## Example Results

```json
{
  "avg_ms": 863.30,
  "min_ms": 638.17,
  "max_ms": 3901.82,
  "median_ms": 730.37,
  "p95_ms": 1647.43,
  "errors": 0
}
```

## Customization

- Modify request headers, methods, or payloads in the scripts
- Adjust timeout values for slow APIs
- Add authentication as needed
- Export results to CSV/JSON for analysis
