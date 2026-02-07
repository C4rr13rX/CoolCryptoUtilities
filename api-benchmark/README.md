# REST API Benchmark Tool

Minimal Python tool to benchmark REST API latency and throughput.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python benchmark.py
```

You will be prompted for:
- API base URL (e.g., https://api.example.com)
- Endpoints to test (comma-separated, or use defaults)

## Metrics Collected

### Latency Metrics
- Average latency (ms)
- Min/Max latency (ms) 
- 95th percentile latency (ms)
- Successful request count

### Throughput Metrics
- Requests per second (RPS)
- Total requests completed
- Error count
- Test duration
- Concurrent users

## Configuration

Default settings:
- Latency test: 100 sequential requests
- Throughput test: 10 concurrent users for 30 seconds
- Request timeout: 10s (latency), 5s (throughput)

## Output

Results are saved to `benchmark_results.json` with timestamp and detailed metrics.

## Example Output

```
Starting benchmark for https://api.example.com
Timestamp: 2026-02-06T22:58:35

Testing endpoint: /health
  Running latency test...
  Running throughput test...
  Latency - Avg: 45.23ms, P95: 78.91ms
  Throughput - RPS: 156.78, Errors: 0
```
