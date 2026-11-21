"""tools/wizard_learning/build_massive_context_corpus.py

Compiles and permanently saves a 1,250+ item English context-aware training &
evaluation corpus under `data/wizard_learning/massive_context_corpus/`:
  1. time_date_location.jsonl   — live time, date, calendar, world clock, location, env & negative controls
  2. context_data_json.jsonl    — inline JSON/CSV extraction, filtering, aggregation, sorting, format adherence & caller context
  3. polyglot_code.jsonl        — Python, Rust, TypeScript, JavaScript, SQL, Go, Bash, PowerShell, HTML/CSS + paraphrases
  4. tool_routing.jsonl         — C0d3rV2 Agent Mode & tool-routing prompts
  5. english_qa_paraphrased.jsonl — 550+ paraphrased/rephrased English science, CS, math, engineering & conversational Q&A
  6. online_lookup_cases.jsonl  — live online search + automatic W1z4rD self-training + fast recall checks
  7. corpus_manifest.json       — durable index & stats for future training runs
"""
from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CORPUS_DIR = PROJECT_ROOT / "data" / "wizard_learning" / "massive_context_corpus"


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_time_date_location() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    time_phrasings = [
        "What time is it?",
        "What's the current time?",
        "Tell me the time",
        "Got the time?",
        "Current time?",
        "What time is it right now?",
        "Give me the local time",
        "What's the clock say?",
        "What is the current time in JSON?",
        "Return the current time as JSON",
        "Give me the current ISO timestamp",
        "What is the current Unix timestamp?",
        "Tell me the current time and date",
        "Give me today's date and current time as JSON",
    ]
    for i, q in enumerate(time_phrasings):
        rows.append({
            "id": f"tdl_time_{i:03d}",
            "category": "time_date_location",
            "endpoint": "/api/wizard-chat/message/",
            "question": q,
            "dynamic": True,
            "expect_json": "json" in q.lower(),
            "expect_type": "time",
            "context": {"location": "Columbus, Ohio, USA", "project": "R3V3N!R"},
        })

    date_phrasings = [
        ("What's today's date?", "date"),
        ("What date is today?", "date"),
        ("Tell me today's date", "date"),
        ("Current date?", "date"),
        ("What day of the week is today?", "weekday"),
        ("Which day is today?", "weekday"),
        ("What month is it?", "month"),
        ("What year is it?", "year"),
        ("What is tomorrow's date?", "tomorrow"),
        ("What day is tomorrow?", "tomorrow"),
        ("What was yesterday's date?", "yesterday"),
        ("What day was yesterday?", "yesterday"),
        ("What date is it in 7 days?", "offset_7"),
        ("What date will it be in 14 days?", "offset_14"),
        ("What date is it in 30 days?", "offset_30"),
        ("How many days until Christmas?", "days_until"),
        ("How many days until New Year?", "days_until"),
        ("How many days until 2027-01-01?", "days_until"),
        ("Return today's date as valid JSON", "date"),
        ("Give me tomorrow's date in JSON format", "tomorrow"),
    ]
    for i, (q, etype) in enumerate(date_phrasings):
        rows.append({
            "id": f"tdl_date_{i:03d}",
            "category": "time_date_location",
            "endpoint": "/api/wizard-chat/message/",
            "question": q,
            "dynamic": True,
            "expect_json": "json" in q.lower(),
            "expect_type": etype,
            "context": {"location": "Columbus, Ohio, USA"},
        })

    world_cities = [
        "UTC", "London", "Paris", "Berlin", "Cairo", "Dubai", "India",
        "Singapore", "Beijing", "Tokyo", "Seoul", "Sydney", "New York",
        "Chicago", "Denver", "Los Angeles", "Hawaii", "Toronto", "EST", "PST",
    ]
    for i, city in enumerate(world_cities):
        q1 = f"What time is it in {city}?"
        q2 = f"Current time in {city} as JSON"
        rows.append({"id": f"tdl_tz_{i:02d}a", "category": "time_date_location", "endpoint": "/api/wizard-chat/message/", "question": q1, "dynamic": True, "expect_json": False, "expect_type": "world_time", "expect_contains": [":"]})
        rows.append({"id": f"tdl_tz_{i:02d}b", "category": "time_date_location", "endpoint": "/api/wizard-chat/message/", "question": q2, "dynamic": True, "expect_json": True, "expect_type": "world_time", "expect_keys": ["zone", "time_24h"]})

    loc_env_phrasings = [
        ("Where am I?", "Columbus, Ohio, USA", False),
        ("What is my current location?", "Columbus, Ohio, USA", False),
        ("Tell me my location as JSON", "Columbus, Ohio, USA", True),
        ("What timezone am I in?", "UTC", False),
        ("What is my local timezone?", "UTC", False),
        ("Return my current timezone in JSON", "timezone", True),
        ("What operating system am I running?", "Windows", False),
        ("What OS is this running on?", "Windows", False),
        ("What workspace am I in?", "CoolCryptoUtilities", False),
        ("What Python version is running?", "3.", False),
        ("Return current status and context as JSON", "status", True),
    ]
    for i, (q, needle, is_j) in enumerate(loc_env_phrasings):
        rows.append({
            "id": f"tdl_loc_{i:03d}",
            "category": "time_date_location",
            "endpoint": "/api/wizard-chat/message/",
            "question": q,
            "dynamic": True,
            "expect_json": is_j,
            "expect_contains": [needle],
            "context": {"location": "Columbus, Ohio, USA"},
        })

    # Negative-control silence checks: context IS provided, but prompt does NOT ask for time/location!
    neg_controls = [
        ("Calculate 19 * 23", "437"),
        ("Calculate 128 + 256", "384"),
        ("What is Ohm's law?", "V = I * R"),
        ("Hello", "W1z4rD"),
        ("solve x^2 - 9 = 0", "3"),
    ]
    for i, (q, expected) in enumerate(neg_controls):
        rows.append({
            "id": f"tdl_neg_{i:03d}",
            "category": "time_date_location",
            "endpoint": "/api/wizard-chat/message/",
            "question": q,
            "dynamic": True,
            "expect_contains": [expected],
            "forbid_contains": ["Columbus, Ohio, USA"],
            "context": {"location": "Columbus, Ohio, USA", "secret_tag": "DO_NOT_LEAK"},
        })
    return rows


def build_context_data_json() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # 1. Inline JSON dict key extraction (60 cases)
    sample_dicts = [
        ({"user": "alice", "role": "engineer", "tier": "pro", "score": 94}, "user", "alice"),
        ({"user": "bob", "role": "architect", "tier": "enterprise", "score": 88}, "role", "architect"),
        ({"host": "node-04", "region": "us-east", "latency_ms": 4.2, "healthy": True}, "region", "us-east"),
        ({"symbol": "ETH", "chain": "base", "fee_bps": 5, "active": True}, "chain", "base"),
        ({"robot": "Mark-V", "alloy": "Ti-6Al-4V", "payload_kg": 25, "reach_m": 2.0}, "alloy", "Ti-6Al-4V"),
        ({"order_id": "ORD-991", "status": "shipped", "carrier": "UPS", "items": 4}, "status", "shipped"),
        ({"sensor": "lidar-2", "fps": 60, "range_m": 120, "mode": "3d"}, "mode", "3d"),
        ({"model": "W1z4rD", "pools": 8, "ram_mb": 235, "store_gb": 87.55}, "model", "W1z4rD"),
        ({"city": "Austin", "temp_c": 24, "humidity": 45, "wind_kph": 12}, "city", "Austin"),
        ({"database": "sqlite", "journal": "WAL", "threads": 48, "port": 8001}, "journal", "WAL"),
    ]
    templates = [
        ("Extract {k} from {j}", False),
        ("What is the value of {k} in {j}?", False),
        ("Given {j}, return {k} as valid JSON", True),
        ("From this JSON {j}, get {k} in JSON format", True),
        ("Given {j}, what is {k}?", False),
        ("Return only {k} as JSON from {j}", True),
    ]
    idx = 0
    for obj, k, v in sample_dicts:
        js = json.dumps(obj)
        for tmpl, is_j in templates:
            rows.append({
                "id": f"data_ext_{idx:03d}",
                "category": "context_data_json",
                "endpoint": "/api/wizard-chat/message/",
                "question": tmpl.format(k=k, j=js),
                "dynamic": True,
                "expect_json": is_j,
                "expect_contains": [str(v)],
            })
            idx += 1

    # 2. Price * qty totals & dict aggregations (40 cases)
    for i in range(20):
        price = 10 + i * 5
        qty = 2 + (i % 7)
        tot = price * qty
        js = json.dumps({"item": f"sku-{i}", "price": price, "qty": qty})
        rows.append({
            "id": f"data_tot_{i:03d}a",
            "category": "context_data_json",
            "endpoint": "/api/wizard-chat/message/",
            "question": f"Given {js}, compute the total as JSON",
            "dynamic": True,
            "expect_json": True,
            "expect_contains": [str(tot)],
        })
        rows.append({
            "id": f"data_tot_{i:03d}b",
            "category": "context_data_json",
            "endpoint": "/api/wizard-chat/message/",
            "question": f"What are the keys in {js}? Return as JSON",
            "dynamic": True,
            "expect_json": True,
            "expect_contains": ["price", "qty"],
        })

    # 3. Numeric array operations (sort, sum, average, min, max, unique, count) (60 cases)
    arrays = [
        [9, 2, 7, 4, 1], [10, 20, 30, 40], [5, 5, 2, 8, 2, 9], [100, 25, 75, 50],
        [3, 1, 4, 1, 5, 9], [12, 18, 6, 24], [42, 7, 19, 88, 3], [8, 16, 32, 64],
        [15, 30, 45, 60, 75], [11, 22, 33, 44],
    ]
    for i, arr in enumerate(arrays):
        js = json.dumps(arr)
        rows.append({"id": f"data_arr_{i:02d}_sum", "category": "context_data_json", "endpoint": "/api/wizard-chat/message/", "question": f"Sum {js} and return as JSON", "dynamic": True, "expect_json": True, "expect_contains": [str(sum(arr))]})
        rows.append({"id": f"data_arr_{i:02d}_max", "category": "context_data_json", "endpoint": "/api/wizard-chat/message/", "question": f"Find the max of {js} in JSON format", "dynamic": True, "expect_json": True, "expect_contains": [str(max(arr))]})
        rows.append({"id": f"data_arr_{i:02d}_min", "category": "context_data_json", "endpoint": "/api/wizard-chat/message/", "question": f"What is the min of {js}?", "dynamic": True, "expect_json": False, "expect_contains": [str(min(arr))]})
        rows.append({"id": f"data_arr_{i:02d}_srt", "category": "context_data_json", "endpoint": "/api/wizard-chat/message/", "question": f"Sort {js} ascending and output as JSON", "dynamic": True, "expect_json": True, "expect_contains": [str(sorted(arr)[0]), str(sorted(arr)[-1])]})
        rows.append({"id": f"data_arr_{i:02d}_unq", "category": "context_data_json", "endpoint": "/api/wizard-chat/message/", "question": f"Return unique values from {js} as JSON", "dynamic": True, "expect_json": True, "expect_contains": [str(arr[0])]})
        rows.append({"id": f"data_arr_{i:02d}_cnt", "category": "context_data_json", "endpoint": "/api/wizard-chat/message/", "question": f"Count items in {js} as JSON", "dynamic": True, "expect_json": True, "expect_contains": [str(len(arr))]})

    # 4. List of records filtering, sorting, CSV/Markdown table conversion (30 cases)
    records = [
        [{"name": "alpha", "status": "active", "score": 90}, {"name": "beta", "status": "paused", "score": 60}, {"name": "gamma", "status": "active", "score": 80}],
        [{"id": 1, "role": "admin", "budget": 500}, {"id": 2, "role": "viewer", "budget": 100}, {"id": 3, "role": "admin", "budget": 700}],
        [{"city": "NYC", "tier": "gold", "rev": 1200}, {"city": "LA", "tier": "silver", "rev": 800}, {"city": "CHI", "tier": "gold", "rev": 950}],
        [{"bot": "M1", "state": "online", "watts": 120}, {"bot": "M2", "state": "offline", "watts": 0}, {"bot": "M3", "state": "online", "watts": 180}],
        [{"pkg": "core", "env": "prod", "ms": 4}, {"pkg": "ui", "env": "dev", "ms": 12}, {"pkg": "api", "env": "prod", "ms": 6}],
    ]
    for i, recs in enumerate(records):
        js = json.dumps(recs)
        rows.append({"id": f"data_rec_{i:02d}_tbl", "category": "context_data_json", "endpoint": "/api/wizard-chat/message/", "question": f"Format {js} as a markdown table", "dynamic": True, "expect_contains": ["|", "---"]})
        rows.append({"id": f"data_rec_{i:02d}_csv", "category": "context_data_json", "endpoint": "/api/wizard-chat/message/", "question": f"Convert {js} to CSV", "dynamic": True, "expect_contains": [","]})
        rows.append({"id": f"data_rec_{i:02d}_cnt", "category": "context_data_json", "endpoint": "/api/wizard-chat/message/", "question": f"How many records are in {js}? Return as JSON", "dynamic": True, "expect_json": True, "expect_contains": ["3"]})
        k2 = list(recs[0].keys())[1]
        v2 = recs[0][k2]
        rows.append({"id": f"data_rec_{i:02d}_flt", "category": "context_data_json", "endpoint": "/api/wizard-chat/message/", "question": f"Filter {js} where {k2}={v2} and output as JSON", "dynamic": True, "expect_json": True, "expect_contains": [str(v2)]})
        k3 = list(recs[0].keys())[2]
        rows.append({"id": f"data_rec_{i:02d}_srt", "category": "context_data_json", "endpoint": "/api/wizard-chat/message/", "question": f"Sort {js} by {k3} descending as JSON", "dynamic": True, "expect_json": True, "expect_contains": [str(max(r[k3] for r in recs))]})
        rows.append({"id": f"data_rec_{i:02d}_sum", "category": "context_data_json", "endpoint": "/api/wizard-chat/message/", "question": f"Sum {k3} in {js} as JSON", "dynamic": True, "expect_json": True, "expect_contains": [str(sum(r[k3] for r in recs))]})

    # 5. Caller-supplied `context` / `data` envelope Q&A (30 cases)
    caller_cases = [
        ({"project_name": "Revenir", "owner": "Adam", "target_latency": "5ms"}, "What is the project_name in context?", "Revenir", False),
        ({"project_name": "Revenir", "owner": "Adam", "target_latency": "5ms"}, "Who is the owner in context?", "Adam", False),
        ({"project_name": "Revenir", "owner": "Adam", "target_latency": "5ms"}, "Return target_latency from context as JSON", "5ms", True),
        ({"cluster": "prod-us-1", "replicas": 6, "version": "2.4.1"}, "What cluster am I on?", "prod-us-1", False),
        ({"cluster": "prod-us-1", "replicas": 6, "version": "2.4.1"}, "What is the current version in context as JSON?", "2.4.1", True),
        ({"wallet_mode": "ghost", "chain_name": "base", "slippage_bps": 15}, "What is the wallet_mode?", "ghost", False),
        ({"wallet_mode": "ghost", "chain_name": "base", "slippage_bps": 15}, "Tell me the chain_name from context", "base", False),
        ({"robot_mark": "Mark-VII", "battery_pct": 92, "habitat": "Lunar-01"}, "What is the robot_mark?", "Mark-VII", False),
        ({"robot_mark": "Mark-VII", "battery_pct": 92, "habitat": "Lunar-01"}, "What is the battery_pct as JSON?", "92", True),
        ({"robot_mark": "Mark-VII", "battery_pct": 92, "habitat": "Lunar-01"}, "What habitat is active in context?", "Lunar-01", False),
    ]
    for i, (cdict, q, needle, is_j) in enumerate(caller_cases):
        rows.append({"id": f"data_ctx_{i:03d}a", "category": "context_data_json", "endpoint": "/api/wizard-chat/message/", "question": q, "dynamic": True, "expect_json": is_j, "expect_contains": [needle], "context": cdict})
        # Also test passing via `data` key
        rows.append({"id": f"data_ctx_{i:03d}b", "category": "context_data_json", "endpoint": "/api/wizard-chat/message/", "question": q, "dynamic": True, "expect_json": is_j, "expect_contains": [needle], "data": cdict})
        # Also test in Agent Mode!
        rows.append({"id": f"data_ctx_{i:03d}c", "category": "context_data_json", "endpoint": "/api/wizard-chat/agent/", "question": q, "dynamic": True, "expect_json": is_j, "expect_contains": [needle], "context": cdict})

    return rows


def build_polyglot_code() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    specs = [
        # Python
        ("Write a Python function to compute Fibonacci numbers", "def fib(", ["Implement Fibonacci in Python", "Python function for Fibonacci sequence", "Give me Python Fibonacci code", "Code a fib function in Python"]),
        ("Write a Python function for binary search", "def binary_search(", ["Implement binary search in Python", "Python binary search algorithm", "Code binary search in Python", "Show me a Python binary search function"]),
        ("Write a Python function for merge sort", "def merge_sort(", ["Implement merge sort in Python", "Python merge sort algorithm", "Write quicksort or merge sort in Python", "Show me sorting in Python using merge sort"]),
        ("Write a Python PID controller", "class PIDController:", ["Implement a PID controller in Python", "Python anti-windup PID controller code", "Code a discrete PID controller in Python"]),
        ("Write a Python function to check if a number is prime", "def is_prime(", ["Python prime checker function", "Implement is_prime in Python", "Check prime numbers in Python"]),
        ("Write a Python function for factorial", "def factorial(", ["Implement factorial in Python", "Python iterative factorial function", "Calculate factorial in Python code"]),
        ("Write a Python Dijkstra shortest path algorithm", "def dijkstra(", ["Implement Dijkstra in Python", "Python shortest path Dijkstra code", "Graph Dijkstra algorithm in Python"]),
        ("Write a Python 1D Kalman filter", "class ScalarKalmanFilter:", ["Implement a Kalman filter in Python", "Python scalar Kalman filter code", "Code a Kalman filter in Python"]),
        ("Write a Python matrix multiplication function", "def matmul(", ["Implement matrix multiplication in Python", "Python matmul function", "Multiply two matrices in Python code"]),
        # Rust
        ("Write a Rust function to compute Fibonacci numbers", "pub fn fib(", ["Implement Fibonacci in Rust", "Rust Fibonacci code", "Code fib in Rust", "Give me a Rust fib function"]),
        ("Write a Rust function for binary search", "pub fn binary_search", ["Implement binary search in Rust", "Rust generic binary search", "Binary search code in Rust"]),
        ("Write a Rust function to reverse a string", "pub fn reverse_str", ["Reverse a string in Rust", "Rust string reversal function", "Unicode string reverse in Rust"]),
        ("Write a Rust function to check prime numbers", "pub fn is_prime", ["Prime number check in Rust", "Rust is_prime function", "Implement primality test in Rust"]),
        ("Write a Rust struct with serde JSON support", "Serialize, Deserialize", ["Rust serde JSON struct", "Define a serializable Rust struct", "Rust JSON data struct"]),
        # TypeScript
        ("Write a TypeScript debounce function", "export function debounce", ["Implement debounce in TypeScript", "TypeScript debounce utility", "Type-safe debounce in TypeScript"]),
        ("Write a TypeScript interface for an API record", "export interface ApiRecord", ["TypeScript user API interface", "Define a TypeScript interface", "Strict TypeScript record type"]),
        ("Write a TypeScript async fetch JSON helper", "export async function fetchJson", ["TypeScript typed fetch API function", "HTTP JSON fetcher in TypeScript", "Async fetchJson in TypeScript"]),
        ("Write a TypeScript groupBy utility", "export function groupBy", ["Implement groupBy in TypeScript", "TypeScript array groupBy function"]),
        # JavaScript
        ("Write a JavaScript debounce helper", "function debounce", ["Implement debounce in JavaScript", "JS debounce function", "Debounce utility in JavaScript"]),
        ("Write a JavaScript Fibonacci function", "function fib", ["Implement Fibonacci in JavaScript", "JS fib function", "Fibonacci code in JavaScript"]),
        ("Write a JavaScript palindrome checker", "isPalindrome", ["Check palindrome in JavaScript", "JS palindrome function", "Palindrome test in JavaScript"]),
        ("Write a JavaScript array deduplication helper", "new Set(", ["Remove duplicates in JavaScript", "JS unique array function", "Deduplicate array in JavaScript"]),
        # SQL
        ("Write a SQL query to find top categories by total amount", "GROUP BY category", ["SQL top 10 aggregation query", "Rank categories in SQL", "SQL query for highest total by category"]),
        ("Write a SQL left join query between users and orders", "LEFT JOIN orders", ["SQL join users and orders", "Lifetime spend per user in SQL", "Left join summary query in SQL"]),
        ("Write a SQL query to find duplicate emails", "HAVING COUNT(*) > 1", ["Find duplicates in SQL", "SQL duplicate email query", "Detect duplicate rows in SQL"]),
        # Go / Bash / PowerShell / HTML
        ("Write a Go worker pool with goroutines", "sync.WaitGroup", ["Implement a concurrent worker pool in Go", "Golang goroutine worker pool", "Go concurrency pool example"]),
        ("Write a Go function to sum integers", "func Sum(", ["Sum a slice in Go", "Golang integer sum function"]),
        ("Write a PowerShell command to list largest files", "Get-ChildItem", ["PowerShell top 10 largest files", "pwsh find biggest files"]),
        ("Write a Bash script to find largest files", "#!/usr/bin/env bash", ["Bash top 10 largest files", "Shell script list biggest files in bash"]),
        ("Write an HTML CSS status card", "<section class=\"card\"", ["Responsive HTML card", "HTML and CSS status card snippet"]),
    ]
    idx = 0
    for canonical, needle, variants in specs:
        for q in [canonical, *variants]:
            rows.append({
                "id": f"code_{idx:03d}",
                "category": "polyglot_code",
                "endpoint": "/api/wizard-chat/message/",
                "question": q,
                "dynamic": True,
                "expect_contains": [needle],
            })
            idx += 1
    return rows


def build_tool_routing() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cases = [
        ("read AGENTS.md", "Working style"),
        ("show file AGENTS.md", "Working style"),
        ("inspect AGENTS.md", "Working style"),
        ("read tools/wizard_learning/runtime_context.py", "build_runtime_context"),
        ("read tools/c0d3rV2/context_tools.py", "ContextClockTool"),
        ("list files in tools/wizard_learning", "fast_conversational_engine.py"),
        ("list directory web/wizard_chat", "agent_runner.py"),
        ("list files in tools/c0d3rV2", "context_tools.py"),
        ("search workspace for probe_unified_eye_dynamics", "neural_eye_dynamics.py"),
        ("grep build_runtime_context", "runtime_context.py"),
        ("search workspace for ParaphraseCorpusIndex", "context_and_data_engine.py"),
        ("search workspace for ContextClockTool", "context_tools.py"),
        ('python -c "print(2**16)"', "65536"),
        ('python -c "print(sum(range(1, 101)))"', "5050"),
        ('python -c "print(len(\'W1z4rD\'))"', "6"),
        ('python -c "import json; print(json.dumps({\'ok\': True}))"', '"ok": true'),
        ("Write a Python function fib(n) and test fib(10)", "55"),
        ("Write a Python function is_prime(n) and test is_prime(97)", "True"),
        ("Write a Python function factorial(n) and test factorial(6)", "720"),
        ("```python\nprint(7 * 8 * 9)\n```", "504"),
        ("```python\nprint(','.join(str(x*x) for x in range(1, 6)))\n```", "1,4,9,16,25"),
        ("What time is it?", ":"),
        ("What's today's date?", "-"),
        ("Where am I?", "Columbus, Ohio, USA"),
        ("Given {\"a\": 12, \"b\": 8}, compute sum as JSON", "20"),
        ("Write a Rust function to compute Fibonacci numbers", "pub fn fib("),
        ("Write a SQL query to find duplicate emails", "HAVING COUNT(*) > 1"),
        ("Calculate (25 * 40) + 123", "1123"),
        ("solve x^2 - 7*x + 12 = 0", "3"),
        ("Deduce robot arm for 20 kg payload and 1.5m reach", "Titanium"),
    ]
    for rep in range(3):
        for i, (q, needle) in enumerate(cases):
            rows.append({
                "id": f"tool_{rep}_{i:03d}",
                "category": "tool_routing",
                "endpoint": "/api/wizard-chat/agent/",
                "question": q,
                "dynamic": True,
                "expect_contains": [needle],
                "context": {"location": "Columbus, Ohio, USA"},
            })
    return rows


def build_english_qa_paraphrased() -> List[Dict[str, Any]]:
    """550+ verified English Q&A items (canonical + 4–6 natural paraphrases each)
    covering CS, networking, databases, math, physics, robotics, engineering & dialogue."""
    knowledge_bank: List[tuple[str, str, str, List[str]]] = [
        # (category, canonical_question, verified_answer, paraphrases)
        ("cs", "What is a hash table?", "A hash table maps keys to values using a hash function to compute bucket indices, achieving average O(1) lookup, insertion, and deletion.", ["Explain hash tables", "How does a hash map work?", "Define hash table data structure", "What makes hash maps O(1)?"]),
        ("cs", "What is Big O notation?", "Big O notation describes the asymptotic upper bound on an algorithm's time or space growth as input size n approaches infinity.", ["Explain Big-O complexity", "Define Big O notation", "What does Big O measure in algorithms?", "How is Big O used for time complexity?"]),
        ("cs", "What is recursion in programming?", "Recursion is a technique where a function solves a problem by calling itself on smaller subproblems until reaching a base case.", ["Explain recursion", "How does recursive function calling work?", "Define recursion and base case", "What is a recursive algorithm?"]),
        ("cs", "What is the difference between a process and a thread?", "A process has its own isolated virtual address space and OS resources, whereas threads share the same process memory space while maintaining separate execution stacks.", ["Process vs thread difference", "How do threads differ from processes?", "Compare OS processes and threads", "Explain processes versus threads"]),
        ("cs", "What is a deadlock in concurrency?", "A deadlock occurs when two or more threads wait indefinitely for resources held by each other, forming a circular wait cycle.", ["Explain concurrency deadlock", "Define deadlock in operating systems", "How do deadlocks happen in multithreading?", "What causes a deadlock?"]),
        ("cs", "What is a binary search tree?", "A binary search tree (BST) is a rooted binary tree where every node's left subtree holds smaller keys and its right subtree holds larger keys.", ["Explain BST data structure", "How does a binary search tree work?", "Define binary search tree", "What is the invariant of a BST?"]),
        ("cs", "What is dynamic programming?", "Dynamic programming solves complex problems with overlapping subproblems and optimal substructure by caching (memoizing or tabulating) intermediate results.", ["Explain dynamic programming", "How does DP memoization work?", "Define dynamic programming in algorithms", "When should you use dynamic programming?"]),
        ("cs", "What is a garbage collector?", "A garbage collector automatically reclaims heap memory occupied by objects that are no longer reachable from live program roots.", ["Explain garbage collection", "How does automatic memory garbage collection work?", "Define GC in programming languages", "What does a garbage collector do?"]),
        ("cs", "What is immutability?", "Immutability means an object's state cannot be modified after creation; updates produce a new value, eliminating shared-state race conditions.", ["Explain immutable data", "Define immutability in software", "Why use immutable objects?", "What does immutable mean in programming?"]),
        ("cs", "What is an abstract syntax tree?", "An abstract syntax tree (AST) is a hierarchical tree representation of source code's grammatical structure used by parsers, compilers, and linters.", ["Explain AST in compilers", "What does AST stand for in programming?", "Define abstract syntax tree", "How do compilers use an AST?"]),
        ("cs", "What is memoization?", "Memoization caches the return values of pure function calls keyed by their arguments so repeated evaluations return immediately in O(1).", ["Explain memoization", "How does function memoization work?", "Define memoization caching", "Why memoize recursive functions?"]),
        ("cs", "What is a race condition?", "A race condition occurs when program correctness depends on the nondeterministic interleaving or timing of concurrent threads accessing shared state.", ["Explain race conditions", "Define a race condition in concurrency", "What causes race conditions?", "How do race conditions happen?"]),
        ("cs", "What is dependency injection?", "Dependency injection supplies an object's required collaborators from outside rather than hard-coding their construction, improving modularity and testability.", ["Explain dependency injection", "Define DI in software architecture", "Why use dependency injection?", "How does dependency injection help testing?"]),
        ("cs", "What is idempotency in APIs?", "An idempotent operation produces the same server state whether executed once or multiple identical times (for example HTTP GET, PUT, and DELETE).", ["Explain idempotency", "Define idempotent API requests", "What makes an HTTP endpoint idempotent?", "Why is idempotency important in distributed systems?"]),
        ("cs", "What is a Bloom filter?", "A Bloom filter is a compact probabilistic bit-array data structure that tests set membership with zero false negatives and a tunable false-positive rate.", ["Explain Bloom filters", "How does a Bloom filter work?", "Define Bloom filter data structure", "Can a Bloom filter have false negatives?"]),
        ("net", "What is the difference between TCP and UDP?", "TCP is connection-oriented with reliable ordered delivery and congestion control, whereas UDP is connectionless, minimal-overhead, and does not guarantee delivery or order.", ["TCP vs UDP comparison", "Compare TCP and UDP protocols", "When to use UDP instead of TCP?", "Explain TCP versus UDP"]),
        ("net", "What is DNS?", "The Domain Name System (DNS) translates human-readable hostnames like example.com into IP addresses through a hierarchical distributed resolver network.", ["Explain DNS", "How does the Domain Name System work?", "Define DNS resolution", "What does DNS do on the internet?"]),
        ("net", "What is HTTPS and TLS?", "HTTPS layers HTTP over TLS encryption, providing confidentiality, data integrity, and server identity authentication via X.509 certificates.", ["Explain HTTPS and TLS", "How does TLS secure HTTP?", "Define HTTPS encryption", "Why is HTTPS safer than HTTP?"]),
        ("net", "What is a REST API?", "A REST API exposes resources at uniform URLs manipulated through stateless HTTP verbs (GET, POST, PUT, PATCH, DELETE) using standard status codes and representations like JSON.", ["Explain RESTful APIs", "Define REST architecture", "How does a REST API work?", "What are the principles of REST?"]),
        ("net", "What is a WebSocket?", "A WebSocket upgrades an HTTP connection into a persistent, full-duplex TCP channel allowing low-latency bidirectional messaging between client and server.", ["Explain WebSockets", "How does WebSocket differ from HTTP?", "Define WebSocket protocol", "When should you use WebSockets?"]),
        ("net", "What is CORS?", "Cross-Origin Resource Sharing (CORS) is a browser security mechanism that uses HTTP response headers like Access-Control-Allow-Origin to control cross-origin requests.", ["Explain CORS in browsers", "Define Cross-Origin Resource Sharing", "Why do CORS errors happen?", "How does CORS work?"]),
        ("net", "What is an HTTP 429 status code?", "HTTP 429 Too Many Requests indicates the client has exceeded a server's rate limit and should back off, typically honoring the Retry-After header.", ["Explain HTTP 429", "What does status code 429 mean?", "Define HTTP 429 Too Many Requests", "How to handle HTTP 429 rate limits?"]),
        ("net", "What is a reverse proxy?", "A reverse proxy sits in front of backend servers to handle TLS termination, load balancing, caching, compression, and request routing.", ["Explain reverse proxies", "What does a reverse proxy like Nginx do?", "Define reverse proxy server", "Why put a reverse proxy in front of an app?"]),
        ("db", "What is SQLite WAL mode?", "Write-Ahead Logging (WAL) appends SQLite mutations to a separate -wal file so concurrent readers never block writers and writers never block readers.", ["Explain SQLite WAL", "Why enable PRAGMA journal_mode=WAL in SQLite?", "How does Write-Ahead Logging work in SQLite?", "SQLite WAL mode benefits"]),
        ("db", "What is ACID in databases?", "ACID stands for Atomicity, Consistency, Isolation, and Durability — the four guarantees ensuring reliable database transactions even across crashes.", ["Explain database ACID properties", "Define ACID transactions", "What does ACID stand for in SQL?", "Why are ACID guarantees important?"]),
        ("db", "What is a database index?", "A database index (commonly a B-tree or hash structure) speeds up row lookups and range scans by maintaining a sorted auxiliary map at the cost of extra write/storage overhead.", ["Explain SQL database indexes", "How does a B-tree index speed up queries?", "Define database indexing", "What is the tradeoff of adding a database index?"]),
        ("db", "What is the CAP theorem?", "The CAP theorem states that during a network partition, a distributed data store must choose between strong Consistency and high Availability.", ["Explain CAP theorem", "Define CAP in distributed systems", "Consistency vs Availability under partition", "What does CAP stand for?"]),
        ("db", "What is SQL injection?", "SQL injection occurs when untrusted user input is concatenated directly into SQL queries; prevent it strictly with parameterized queries or prepared statements.", ["Explain SQL injection", "How do you prevent SQL injection?", "Define SQLi vulnerability", "Why use parameterized SQL queries?"]),
        ("db", "What is database normalization?", "Normalization structures relational tables to eliminate redundant data and update anomalies, commonly up to Third Normal Form (3NF).", ["Explain SQL normalization", "Define 3NF database normalization", "Why normalize relational schemas?", "What problem does database normalization solve?"]),
        ("sys", "What is Docker?", "Docker packages applications and their dependencies into lightweight, reproducible Linux/OS containers sharing the host kernel via namespaces and cgroups.", ["Explain Docker containers", "How does Docker differ from a VM?", "Define Docker containerization", "What problem does Docker solve?"]),
        ("sys", "What is Git rebase vs merge?", "Git merge preserves branch history by creating a merge commit, while Git rebase replays commits onto a new base commit to produce a linear history.", ["Compare git rebase and git merge", "Git merge vs rebase difference", "When should you use git rebase?", "Explain git rebase versus merge"]),
        ("sys", "What is CI/CD?", "Continuous Integration automatically builds and tests every code change, while Continuous Delivery/Deployment automates releasing validated artifacts to staging or production.", ["Explain CI/CD pipelines", "Define Continuous Integration and Deployment", "What does CI CD stand for?", "Why use CI/CD in software engineering?"]),
        ("sys", "What is the Python GIL?", "The Global Interpreter Lock (GIL) is a mutex in standard CPython ensuring only one thread executes Python bytecode at a time; CPU-bound parallelism uses multiprocessing or C/Rust extensions.", ["Explain Python GIL", "What is the Global Interpreter Lock in CPython?", "How does the GIL affect multithreading in Python?", "Why does Python have a GIL?"]),
        ("sys", "What is memory-mapped I/O (mmap)?", "Memory-mapped I/O maps a file directly into virtual address space so the OS kernel pages chunks on demand without copying the entire file into heap RAM.", ["Explain mmap", "How does memory-mapped file I/O work?", "Why use mmap for large files?", "Define memory-mapped I/O"]),
        ("sys", "What is a mutex vs a semaphore?", "A mutex is a binary ownership lock allowing only the locking thread to unlock a critical section, whereas a semaphore is a counter permitting up to N concurrent holders.", ["Mutex vs semaphore difference", "Compare mutex and semaphore", "Explain mutex versus semaphore", "When to use a semaphore instead of a mutex?"]),
        ("phys", "What is the speed of light in vacuum?", "The speed of light in vacuum is exactly c = 299,792,458 meters per second (~3.00 × 10^8 m/s).", ["Exact value of speed of light", "How fast is light in a vacuum?", "Value of constant c in physics", "Speed of light in meters per second"]),
        ("phys", "What is Newton's first law of motion?", "An object remains at rest or in uniform straight-line motion unless acted upon by a non-zero net external force (law of inertia).", ["Explain Newton's first law", "Define the law of inertia", "State Newton's 1st law", "What does Newton's first law say?"]),
        ("phys", "What is Newton's third law of motion?", "For every force exerted by body A on body B, body B exerts an equal-magnitude, opposite-direction force on body A (F_AB = -F_BA).", ["Explain Newton's third law", "Define action and reaction law", "State Newton's 3rd law", "What is Newton's action-reaction principle?"]),
        ("phys", "What is kinetic energy?", "Translational kinetic energy is KE = 0.5 * m * v^2 (Joules), where m is mass in kg and v is speed in m/s.", ["Formula for kinetic energy", "Explain kinetic energy in physics", "How do you calculate kinetic energy?", "Define translational kinetic energy"]),
        ("phys", "What is the Ideal Gas Law?", "The Ideal Gas Law is P * V = n * R * T, relating pressure P (Pa), volume V (m^3), moles n, gas constant R = 8.314 J/(mol·K), and absolute temperature T (K).", ["Explain the ideal gas law", "Formula for PV = nRT", "Define ideal gas equation", "What does PV = nRT mean?"]),
        ("phys", "What is the First Law of Thermodynamics?", "Energy is conserved: the change in internal energy equals heat added minus work done by the system (ΔU = Q - W).", ["Explain first law of thermodynamics", "State conservation of energy in thermodynamics", "Define ΔU = Q - W", "What does the 1st law of thermodynamics say?"]),
        ("phys", "What is the Second Law of Thermodynamics?", "In any spontaneous process, the total entropy of an isolated system never decreases (ΔS_total >= 0), bounding heat-engine efficiency by the Carnot limit.", ["Explain second law of thermodynamics", "Define entropy increase law", "What does the 2nd law of thermodynamics state?", "Why can't heat engines be 100% efficient?"]),
        ("phys", "What is Bernoulli's principle?", "Along a steady, incompressible, inviscid streamline, P + 0.5*rho*v^2 + rho*g*h = constant, so faster fluid speed corresponds to lower static pressure.", ["Explain Bernoulli's principle", "State Bernoulli's equation in fluid dynamics", "How does Bernoulli's law relate pressure and velocity?", "Define Bernoulli's principle"]),
        ("phys", "What is Faraday's law of induction?", "A time-varying magnetic flux Φ_B through a closed loop induces an electromotive force EMF = -dΦ_B/dt opposing the flux change (Lenz's law).", ["Explain Faraday's law", "How does electromagnetic induction work?", "Define Faraday's law of electromagnetic induction", "Formula for induced EMF"]),
        ("phys", "What is Snell's law of refraction?", "Snell's law relates incident and refracted light angles across media of refractive indices n1 and n2: n1 * sin(theta1) = n2 * sin(theta2).", ["Explain Snell's law", "Formula for light refraction", "Define Snell's law in optics", "How does light bend between media?"]),
        ("math", "What is the Pythagorean theorem?", "In a right triangle with legs a, b and hypotenuse c, a^2 + b^2 = c^2.", ["Explain Pythagorean theorem", "State a^2 + b^2 = c^2", "Formula for hypotenuse of a right triangle", "Define the Pythagorean theorem"]),
        ("math", "What is the quadratic formula?", "For a*x^2 + b*x + c = 0 (a != 0), the roots are x = (-b ± sqrt(b^2 - 4*a*c)) / (2*a).", ["Explain the quadratic formula", "How do you solve ax^2 + bx + c = 0?", "Formula for quadratic roots", "State the quadratic equation solution"]),
        ("math", "What is Bayes' theorem?", "Bayes' theorem updates posterior probability given evidence: P(A|B) = P(B|A) * P(A) / P(B).", ["Explain Bayes theorem", "Formula for Bayesian probability", "Define Bayes' rule", "How do you compute posterior probability with Bayes' theorem?"]),
        ("math", "What is Shannon entropy?", "Shannon entropy quantifies average information uncertainty in bits: H(X) = -sum(p_i * log2(p_i)).", ["Explain Shannon entropy", "Formula for information entropy", "Define entropy in information theory", "How is Shannon entropy calculated?"]),
        ("math", "What is Euler's identity?", "Euler's identity is e^(i*pi) + 1 = 0, uniting e, i, pi, 1, and 0 from e^(i*theta) = cos(theta) + i*sin(theta).", ["Explain Euler's identity", "State e^(i*pi) + 1 = 0", "What is Euler's formula?", "Why is Euler's identity famous?"]),
        ("math", "What is a derivative in calculus?", "A derivative f'(x) = lim_{h->0} (f(x+h) - f(x))/h measures the instantaneous rate of change (slope of the tangent line) of f at x.", ["Explain derivatives in calculus", "Define the derivative of a function", "What does a derivative represent?", "Geometric meaning of a derivative"]),
        ("math", "What is an eigenvalue?", "For a square matrix A, a non-zero vector v is an eigenvector with eigenvalue λ if A*v = λ*v, meaning A only scales v without rotating it.", ["Explain eigenvalues and eigenvectors", "Define Av = lambda v", "What is an eigenvalue in linear algebra?", "How do eigenvectors work?"]),
        ("math", "What is standard deviation?", "Standard deviation σ = sqrt(E[(X - μ)^2]) measures the dispersion of data points around their mean in the original units of the data.", ["Explain standard deviation", "Define variance and standard deviation", "What does standard deviation measure?", "Formula for standard deviation"]),
        ("robo", "What is a PID controller?", "A PID controller computes control effort u(t) = Kp*e(t) + Ki*∫e(t)dt + Kd*de(t)/dt from proportional, integral, and derivative error terms.", ["Explain PID control", "How does a PID loop work?", "Define proportional integral derivative controller", "What do Kp Ki and Kd do in PID?"]),
        ("robo", "What is inverse kinematics in robotics?", "Inverse kinematics computes the joint angles needed to place a robot's end-effector at a desired 3D position and orientation.", ["Explain inverse kinematics", "Define IK in robotics", "Difference between forward and inverse kinematics", "How does a robot arm compute joint angles?"]),
        ("robo", "What is a Kalman filter?", "A Kalman filter is an optimal recursive state estimator for linear-Gaussian systems that alternates prediction and measurement-update steps weighted by Kalman gain.", ["Explain Kalman filtering", "How does a Kalman filter fuse noisy sensors?", "Define Kalman filter in robotics", "What is a Kalman filter used for?"]),
        ("robo", "What is SLAM in robotics?", "Simultaneous Localization and Mapping (SLAM) estimates a robot's pose while incrementally building a consistent map of an unknown environment.", ["Explain SLAM", "What does SLAM stand for in robotics?", "Define Simultaneous Localization and Mapping", "How do autonomous robots map rooms with SLAM?"]),
        ("robo", "What is a planetary gearbox?", "A planetary (epicyclic) gearbox transmits high torque compactly by sharing load across multiple planet gears orbiting between a central sun gear and an outer ring gear.", ["Explain planetary gearboxes", "How does an epicyclic gearhead work?", "Why use planetary gears in robot actuators?", "Define planetary gear reducer"]),
        ("robo", "What is Hebbian learning?", "Hebbian learning strengthens synaptic weights between neurons that fire together coherently ('cells that fire together, wire together'), enabling associative memory without backpropagation.", ["Explain Hebbian learning", "Define Hebb's rule in neural networks", "How do Hebbian neural pools learn?", "What does fire together wire together mean?"]),
        ("sci", "What is photosynthesis?", "Photosynthesis converts sunlight, water (H2O), and carbon dioxide (CO2) inside chloroplasts into glucose (C6H12O6) and oxygen (O2).", ["Explain photosynthesis", "What is the chemical equation of photosynthesis?", "How do plants make energy from sunlight?", "Define photosynthesis in biology"]),
        ("sci", "What is DNA?", "Deoxyribonucleic acid (DNA) is a double-helix polynucleotide (adenine-thymine, cytosine-guanine base pairs) encoding hereditary genetic instructions.", ["Explain DNA structure", "What does DNA stand for?", "Define deoxyribonucleic acid", "What are the four base pairs in DNA?"]),
        ("sci", "What is mitochondria's role in a cell?", "Mitochondria generate most of a eukaryotic cell's chemical energy (ATP) through oxidative phosphorylation and the citric acid cycle.", ["What do mitochondria do?", "Explain the function of mitochondria", "Why are mitochondria called the powerhouse of the cell?", "Role of mitochondria in cellular respiration"]),
        ("sci", "What is the boiling point of water?", "At standard sea-level atmospheric pressure (101.325 kPa / 1 atm), pure water boils at 100 °C (212 °F / 373.15 K).", ["At what temperature does water boil?", "Boiling point of water in Celsius and Fahrenheit", "Water boiling temperature at 1 atm", "How hot does water need to be to boil?"]),
        ("sci", "What is absolute zero?", "Absolute zero is 0 Kelvin (-273.15 °C / -459.67 °F), the theoretical lower limit of thermodynamic temperature where classical thermal motion ceases.", ["Define absolute zero", "What is 0 Kelvin in Celsius?", "Lowest possible temperature in physics", "Explain absolute zero temperature"]),
        ("sci", "What is a light-year?", "A light-year is a unit of distance equal to how far light travels in vacuum in one Julian year (~9.461 × 10^12 km or ~5.879 trillion miles).", ["Explain what a light-year measures", "How many kilometers is a light-year?", "Is a light-year distance or time?", "Define light-year in astronomy"]),
        ("sci", "What is the chemical symbol for gold?", "The chemical symbol for gold is Au (atomic number 79, from Latin aurum).", ["Chemical symbol of gold", "Atomic number of gold", "What is Au on the periodic table?", "Periodic table symbol for gold"]),
        ("sci", "What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system, a gas giant with roughly 318 Earth masses.", ["Which planet is the biggest in the solar system?", "Largest solar system planet", "What is the most massive planet orbiting the Sun?", "Name the biggest planet in our solar system"]),
        ("sci", "What causes Earth's seasons?", "Earth's seasons are caused by its 23.44-degree axial tilt relative to its orbital plane, changing solar elevation angle and daylight duration throughout the year.", ["Why do we have seasons on Earth?", "Explain the cause of Earth's seasons", "What creates summer and winter?", "Is Earth's distance or tilt responsible for seasons?"]),
        ("eng", "What is Young's modulus?", "Young's modulus E = stress / strain (Pascals) measures a solid material's axial elastic stiffness under tension or compression.", ["Explain Young's modulus", "Define elastic modulus E", "Formula for Young's modulus", "What does Young's modulus measure in materials?"]),
        ("eng", "What is a safety factor in engineering?", "Factor of Safety FoS = ultimate_or_yield_strength / working_stress quantifies structural load margin above expected operating conditions.", ["Explain Factor of Safety", "Define engineering safety factor", "How do you calculate safety factor?", "Why design with FoS >= 2?"]),
        ("eng", "What is LiFePO4 battery chemistry?", "Lithium Iron Phosphate (LiFePO4) offers 3.2 V nominal per cell, high thermal runaway stability, and 2,000–5,000+ charge cycles for robotics and energy storage.", ["Explain LiFePO4 batteries", "What is the nominal voltage of a LiFePO4 cell?", "Benefits of Lithium Iron Phosphate batteries", "Define LiFePO4 chemistry"]),
        ("eng", "What is PWM in electronics?", "Pulse-Width Modulation (PWM) varies the duty cycle of a fixed-frequency square wave to efficiently control average power delivered to motors, LEDs, or servos.", ["Explain PWM signal", "What does Pulse-Width Modulation do?", "How does PWM control motor speed?", "Define PWM duty cycle"]),
        ("eng", "What is an op-amp?", "An operational amplifier (op-amp) is a high-gain differential voltage amplifier used with negative feedback for filtering, buffering, summing, and signal conditioning.", ["Explain operational amplifiers", "What does an op-amp do in circuits?", "Define op-amp", "Common uses of operational amplifiers"]),
        ("eng", "What is Nyquist-Shannon sampling theorem?", "To reconstruct a band-limited signal without aliasing, the sampling rate must be strictly greater than twice its highest frequency component (f_s > 2 * f_max).", ["Explain Nyquist theorem", "Define Nyquist sampling rate", "How do you avoid aliasing when sampling?", "State the Nyquist-Shannon sampling theorem"]),
        ("fin", "What is compound interest?", "Compound interest earns return on both principal and accumulated prior interest: A = P * (1 + r/n)^(n*t).", ["Explain compound interest", "Formula for compounding interest", "How does compound interest work?", "Define compound interest equation"]),
        ("fin", "What is slippage in trading?", "Slippage is the difference between the expected execution price of an order and the actual fill price due to order-book depth, latency, or volatility.", ["Explain trading slippage", "Define slippage in crypto and stocks", "What causes execution slippage?", "How does slippage affect net profit?"]),
        ("fin", "What is Sharpe ratio?", "Sharpe ratio = (R_p - R_f) / sigma_p measures excess return per unit of portfolio volatility (standard deviation).", ["Explain the Sharpe ratio", "Formula for Sharpe ratio", "How do you measure risk-adjusted return?", "Define Sharpe ratio in finance"]),
        ("fin", "What is an automated market maker (AMM)?", "An AMM prices on-chain token swaps algorithmically from liquidity pools (such as Uniswap's constant-product x * y = k or concentrated liquidity ticks) without a central order book.", ["Explain AMM in DeFi", "How does an automated market maker work?", "Define constant product AMM", "What is an AMM liquidity pool?"]),
        ("geo", "What is the capital of Japan?", "The capital of Japan is Tokyo.", ["Capital city of Japan", "Which city is Japan's capital?", "Name Japan's capital", "What city serves as the capital of Japan?"]),
        ("geo", "What is the capital of France?", "The capital of France is Paris.", ["Capital city of France", "Which city is France's capital?", "Name France's capital", "What city serves as the capital of France?"]),
        ("geo", "What is the longest river in Africa?", "The Nile River (approximately 6,650 km) is the longest river in Africa.", ["Longest river on the African continent", "Which river is longest in Africa?", "How long is the Nile River?", "Name Africa's longest river"]),
        ("geo", "What is the tallest mountain on Earth above sea level?", "Mount Everest in the Himalayas is the tallest mountain above sea level at 8,848.86 meters (29,031.7 feet).", ["Highest mountain above sea level", "How tall is Mount Everest?", "Tallest peak on Earth", "Name the highest mountain on Earth"]),
        ("gen", "How many bits are in a byte?", "There are 8 bits in one byte (representing 256 distinct values from 0 to 255).", ["Bits per byte", "How many bits make a byte?", "Number of bits in a single byte", "Define a byte in bits"]),
        ("gen", "How many seconds are in a day?", "There are 86,400 seconds in a standard 24-hour day (24 × 60 × 60).", ["Seconds in one day", "How many seconds in 24 hours?", "Calculate seconds per day", "Total seconds in a day"]),
        ("gen", "How many centimeters are in a meter?", "There are 100 centimeters in 1 meter.", ["Centimeters per meter", "How many cm in a meter?", "Convert 1 meter to centimeters", "Number of centimeters in a meter"]),
        ("gen", "How many grams are in a kilogram?", "There are 1,000 grams in 1 kilogram.", ["Grams per kilogram", "How many grams make 1 kg?", "Convert 1 kg to grams", "Number of grams in a kilogram"]),
        ("gen", "What is the freezing point of water?", "Pure water freezes at 0 °C (32 °F / 273.15 K) at standard atmospheric pressure.", ["At what temperature does water freeze?", "Water freezing point in Celsius and Fahrenheit", "Freezing temperature of water", "When does water turn to ice?"]),
        ("gen", "What are the primary colors of light?", "In additive RGB light mixing (screens and optics), the primary colors are red, green, and blue.", ["RGB primary colors of light", "What are additive primary colors?", "Primary light colors in displays", "Name the three primary colors of light"]),
        ("gen", "What is JSON?", "JSON (JavaScript Object Notation) is a lightweight, human-readable text format for structured key-value objects and arrays.", ["Explain JSON format", "What does JSON stand for?", "Define JavaScript Object Notation", "Why is JSON used for APIs?"]),
        ("gen", "What is UTF-8?", "UTF-8 is a variable-width (1–4 byte) Unicode character encoding that is backward-compatible with ASCII and standard across the web.", ["Explain UTF-8 encoding", "Define UTF-8 Unicode", "Why use UTF-8 text encoding?", "Is UTF-8 compatible with ASCII?"]),
        ("gen", "What is Markdown?", "Markdown is a lightweight plain-text markup syntax for formatting headings, lists, links, tables, and code blocks cleanly into HTML.", ["Explain Markdown syntax", "Define Markdown", "What is Markdown used for?", "Why write docs in Markdown?"]),
        ("gen", "What is a regular expression?", "A regular expression (regex) is a concise pattern-matching language used to search, validate, and transform strings.", ["Explain regex", "Define regular expressions", "What are regex patterns used for?", "How do regular expressions work?"]),
        ("gen", "What is SSH?", "Secure Shell (SSH) is an encrypted network protocol for remote terminal access, command execution, port forwarding, and secure file transfer.", ["Explain SSH protocol", "What does SSH stand for?", "Define Secure Shell", "What is SSH used for?"]),
        ("gen", "What is a CPU cache?", "A CPU cache (L1/L2/L3 SRAM) stores frequently accessed instructions and data close to processor cores to hide main DRAM latency.", ["Explain L1 L2 L3 CPU cache", "Define processor cache memory", "Why is CPU cache faster than RAM?", "How does CPU caching speed up programs?"]),
        ("gen", "What is virtual memory?", "Virtual memory maps per-process virtual addresses to physical RAM frames or disk swap via page tables, providing isolation and larger address spaces.", ["Explain virtual memory in operating systems", "How do page tables work?", "Define virtual memory and paging", "Why do OS kernels use virtual memory?"]),
        ("gen", "What is a compiler vs an interpreter?", "A compiler translates source code ahead-of-time into machine/bytecode before execution, whereas an interpreter evaluates source or bytecode instructions directly at runtime.", ["Compiler vs interpreter difference", "Compare compiled and interpreted languages", "Explain compiler versus interpreter", "How does compilation differ from interpretation?"]),
        ("gen", "What is unit testing?", "Unit testing verifies individual functions or classes in isolation with fast, deterministic assertions to catch regressions early.", ["Explain unit tests", "Define unit testing in software", "Why write unit tests?", "What is the purpose of unit testing?"]),
        ("gen", "What is semantic versioning?", "Semantic Versioning (MAJOR.MINOR.PATCH) increments MAJOR for breaking changes, MINOR for backward-compatible features, and PATCH for bug fixes.", ["Explain SemVer", "Define Semantic Versioning MAJOR MINOR PATCH", "How does semantic versioning work?", "What do version numbers 1.2.3 mean?"]),
        ("gen", "What is OAuth 2.0?", "OAuth 2.0 is an authorization framework that lets applications obtain scoped access tokens on a user's behalf without handling the user's raw password.", ["Explain OAuth 2", "Define OAuth 2.0 authorization", "How do OAuth access tokens work?", "What problem does OAuth 2.0 solve?"]),
        ("gen", "What is a JWT?", "A JSON Web Token (JWT) is a compact, URL-safe, signed (header.payload.signature) token used to transmit verifiable claims between parties.", ["Explain JWT tokens", "What does JWT stand for?", "Define JSON Web Token", "What are the three parts of a JWT?"]),
        ("gen", "What is GraphQL?", "GraphQL is a strongly typed API query language and runtime that lets clients request the exact fields and nested relationships they need in a single request.", ["Explain GraphQL", "How does GraphQL differ from REST?", "Define GraphQL API", "Why use GraphQL?"]),
        ("gen", "What is WebAssembly (Wasm)?", "WebAssembly is a portable, sandboxed binary instruction format enabling near-native execution of Rust, C/C++, and Go code in browsers and edge runtimes.", ["Explain WebAssembly", "What is Wasm used for?", "Define WebAssembly binary format", "Why compile Rust to WebAssembly?"]),
    ]

    # Also include 25 EMPIRICAL_CURRICULUM items with paraphrases
    from tools.wizard_learning.empirical_material import EMPIRICAL_CURRICULUM
    for item in EMPIRICAL_CURRICULUM[:25]:
        ans = f"{item.topic}: {item.formula} ({item.units}). {item.answer}"
        knowledge_bank.append((
            f"empirical_{item.category}",
            item.question,
            ans,
            [
                f"Explain {item.topic}",
                f"What is the formula for {item.topic}?",
                f"Tell me about {item.topic}",
                f"Give me the equation for {item.topic}",
            ],
        ))

    rows: List[Dict[str, Any]] = []
    idx = 0
    for cat, q, a, paraphrases in knowledge_bank:
        # Store the canonical record with all paraphrases (for ParaphraseCorpusIndex)
        # AND emit individual test rows for canonical + every paraphrase (5x coverage -> ~630 items!)
        import re as _re
        alpha_toks = _re.findall(r"[A-Za-z]{4,}", a)
        key_token = alpha_toks[0] if alpha_toks else a[:10]
        rows.append({
            "id": f"eng_{idx:04d}",
            "category": cat,
            "endpoint": "/api/wizard-chat/message/",
            "question": q,
            "answer": a,
            "paraphrases": paraphrases,
            "dynamic": False,
            "expect_contains": [key_token],
        })
        idx += 1
        for p_i, pq in enumerate(paraphrases):
            rows.append({
                "id": f"eng_{idx:04d}",
                "category": cat,
                "endpoint": "/api/wizard-chat/message/",
                "question": pq,
                "answer": a,
                "paraphrases": [],
                "dynamic": False,
                "is_paraphrase": True,
                "expect_contains": [key_token],
            })
            idx += 1

    # Add 40 arithmetic & symbolic algebra/calculus battery items (dynamic <1ms)
    for i in range(1, 21):
        a_n, b_n = 11 * i, 7 * i + 3
        rows.append({
            "id": f"math_ar_{i:03d}",
            "category": "math_dynamic",
            "endpoint": "/api/wizard-chat/message/",
            "question": f"Calculate ({a_n} * {b_n}) + {i}",
            "dynamic": True,
            "expect_contains": [str(a_n * b_n + i)],
        })
    for r1, r2 in [(1, 2), (2, 3), (2, 5), (3, 4), (3, 7), (4, 5), (1, 6), (2, 8), (5, 6), (4, 9)]:
        b_c = r1 + r2
        c_c = r1 * r2
        rows.append({
            "id": f"math_alg_{r1}_{r2}",
            "category": "math_dynamic",
            "endpoint": "/api/wizard-chat/message/",
            "question": f"solve x^2 - {b_c}*x + {c_c} = 0",
            "dynamic": True,
            "expect_contains": [str(r1), str(r2)],
        })
    for n in range(2, 12):
        rows.append({
            "id": f"math_der_{n:02d}",
            "category": "math_dynamic",
            "endpoint": "/api/wizard-chat/message/",
            "question": f"derivative of x^{n} + {n}*x",
            "dynamic": True,
            "expect_contains": [f"{n}*x" if n > 2 else "2*x"],
        })

    return rows


def build_online_lookup_cases() -> List[Dict[str, Any]]:
    return [
        {
            "id": "web_001",
            "category": "online_lookup",
            "endpoint": "/api/wizard-chat/message/",
            "question": "Search online for Nikola Tesla",
            "dynamic": True,
            "allow_web_lookup": True,
            "expect_contains": ["Tesla"],
        },
        {
            "id": "web_002",
            "category": "online_lookup",
            "endpoint": "/api/wizard-chat/message/",
            "question": "Look up Ada Lovelace",
            "dynamic": True,
            "allow_web_lookup": True,
            "expect_contains": ["Lovelace"],
        },
        {
            "id": "web_003",
            "category": "online_lookup",
            "endpoint": "/api/wizard-chat/message/",
            "question": "Search online for Alan Turing",
            "dynamic": True,
            "allow_web_lookup": True,
            "expect_contains": ["Turing"],
        },
        {
            "id": "web_004",
            "category": "online_lookup",
            "endpoint": "/api/wizard-chat/message/",
            "question": "Look up Marie Curie",
            "dynamic": True,
            "allow_web_lookup": True,
            "expect_contains": ["Curie"],
        },
    ]


def compile_and_save_all() -> Dict[str, Any]:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {
        "time_date_location.jsonl": build_time_date_location(),
        "context_data_json.jsonl": build_context_data_json(),
        "polyglot_code.jsonl": build_polyglot_code(),
        "tool_routing.jsonl": build_tool_routing(),
        "english_qa_paraphrased.jsonl": build_english_qa_paraphrased(),
        "online_lookup_cases.jsonl": build_online_lookup_cases(),
    }

    counts: Dict[str, int] = {}
    total = 0
    for fname, rows in datasets.items():
        _write_jsonl(CORPUS_DIR / fname, rows)
        counts[fname] = len(rows)
        total += len(rows)

    manifest = {
        "version": 1,
        "compiled_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "language": "English",
        "total_items": total,
        "files": counts,
        "corpus_dir": str(CORPUS_DIR),
    }
    (CORPUS_DIR / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    t0 = time.perf_counter()
    m = compile_and_save_all()
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"Compiled & saved {m['total_items']} corpus items in {dt:.1f} ms -> {CORPUS_DIR}")
    for k, v in m["files"].items():
        print(f"  - {k:32s}: {v:4d} items")
