import asyncio
import aiohttp
import time
import json
import sys

async def make_request(session, url):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            return response.status == 200
    except:
        return False

async def benchmark_throughput(url, num_requests=1000, concurrency=50):
    start_time = time.perf_counter()
    
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request():
            async with semaphore:
                return await make_request(session, url)
        
        tasks = [bounded_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    successful = sum(results)
    
    return {
        "total_requests": num_requests,
        "successful_requests": successful,
        "failed_requests": num_requests - successful,
        "duration_s": duration,
        "requests_per_second": num_requests / duration,
        "success_rate": successful / num_requests * 100
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python throughput_benchmark.py <url> [num_requests] [concurrency]")
        sys.exit(1)
    
    url = sys.argv[1]
    num_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    concurrency = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    result = asyncio.run(benchmark_throughput(url, num_requests, concurrency))
    print(json.dumps(result, indent=2))
