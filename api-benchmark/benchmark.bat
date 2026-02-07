@echo off
echo REST API Benchmark Plan
echo Usage: benchmark.bat [URL] [requests]
echo.
echo Running latency test...
python latency_test.py %1 %2
echo.
echo Running throughput test...
python throughput_test.py %1 %2
echo.
echo Results saved to results/ directory
