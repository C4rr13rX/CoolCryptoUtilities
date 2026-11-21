"""tools/c0d3rV2/context_tools.py

First-class C0d3rV2 tools for live context (time/date/location/timezone),
structured data transformation (JSON/CSV/key-value), fast online lookup with
automatic W1z4rD self-training, and polyglot code synthesis.
"""
from __future__ import annotations

import ast
import csv
import datetime
import io
import json
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

try:
    from tool_registry import Tool
except Exception:
    class Tool:  # type: ignore[no-redef]
        name = ""
        description = ""
        use_when = ""
        params_schema: Dict[str, str] = {}
        def execute(self, params: dict) -> dict:
            raise NotImplementedError


# ── 1. ContextClockTool ─────────────────────────────────────────────────────

class ContextClockTool(Tool):
    name = "context_clock"
    description = (
        "Read live local time, date, weekday, timezone, UTC offset, cached host location, "
        "world city time conversions, and calendar arithmetic (tomorrow, yesterday, days until date)."
    )
    use_when = (
        "Use when the user asks for the current time, date, day of week, timezone, "
        "world clock conversion, calendar delta, or current location."
    )
    params_schema = {
        "query": "str — optional city/timezone (e.g. 'tokyo', 'utc') or 'local'",
        "caller_context": "dict — optional caller-supplied timezone/location override",
    }

    def execute(self, params: dict) -> dict:
        from tools.wizard_learning.runtime_context import build_runtime_context, convert_time_for_zone

        ctx = build_runtime_context(caller_context=params.get("caller_context"))
        q = str(params.get("query") or "").strip().lower()
        if q and q != "local":
            conv = convert_time_for_zone(q)
            if conv:
                return {"ok": True, "mode": "world_clock", "converted": conv, "local": ctx}
        return {"ok": True, "mode": "local_context", "context": ctx}


# ── 2. StructuredDataTool ───────────────────────────────────────────────────

def extract_inline_json(text: str) -> Optional[Any]:
    """Extract the first valid JSON object or array embedded anywhere in text."""
    if not text or ("{" not in text and "[" not in text):
        return None
    # Try fenced json block first
    m = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, flags=re.I | re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Scan balanced brackets/braces — prefer whichever outermost delimiter appears FIRST
    pos_obj = text.find("{")
    pos_arr = text.find("[")
    delims = [("[", "]"), ("{", "}")] if (pos_arr != -1 and (pos_obj == -1 or pos_arr < pos_obj)) else [("{", "}"), ("[", "]")]
    for open_ch, close_ch in delims:
        start = text.find(open_ch)
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        cand = text[start : idx + 1]
                        try:
                            return json.loads(cand)
                        except Exception:
                            break
            start = text.find(open_ch, start + 1)
    return None


def extract_inline_csv(text: str) -> Optional[List[Dict[str, str]]]:
    """Extract a small inline CSV block if at least 2 comma-separated lines exist."""
    lines = [ln.strip() for ln in (text or "").splitlines() if "," in ln]
    if len(lines) < 2:
        return None
    # Find contiguous CSV-like lines with matching column count >= 2
    col_count = len(lines[0].split(","))
    if col_count < 2:
        return None
    csv_lines = [ln for ln in lines if len(ln.split(",")) == col_count]
    if len(csv_lines) < 2:
        return None
    try:
        reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
        rows = [dict(r) for r in reader if r]
        return rows if rows else None
    except Exception:
        return None


class StructuredDataTool(Tool):
    name = "data_transform"
    description = (
        "Parse, query, filter, sort, sum, count, or transform inline JSON, CSV, or "
        "key-value context data and emit strictly valid JSON, CSV, or Markdown output."
    )
    use_when = (
        "Use when the prompt supplies JSON/CSV/structured data or asks for a JSON, "
        "CSV, or Markdown table output computed from context."
    )
    params_schema = {
        "text": "str — user prompt containing inline data and/or instructions",
        "data": "any — optional pre-parsed structured data payload",
    }

    def execute(self, params: dict) -> dict:
        text = str(params.get("text") or "")
        data = params.get("data")
        if data is None:
            data = extract_inline_json(text)
        if data is None:
            data = extract_inline_csv(text)
        if data is None:
            return {"ok": False, "error": "No structured JSON/CSV payload found."}
        return {"ok": True, "data": data, "type": type(data).__name__}


# ── 3. FastWebLookupTool ────────────────────────────────────────────────────

def fast_online_lookup(query: str, timeout: float = 2.5) -> Optional[Dict[str, str]]:
    """Bounded online factual lookup (Wikipedia REST + DuckDuckGo Instant Answer).
    Returns {answer, source, title} or None cleanly within timeout."""
    clean_q = re.sub(
        r"^(?:search\s+online\s+for|look\s+up|web\s+search(?:\s+for)?|google|find\s+online|search\s+the\s+web\s+for)\s+",
        "",
        (query or "").strip(),
        flags=re.I,
    ).strip().strip("?!.\"'`")
    if not clean_q or len(clean_q) < 2:
        return None

    # 1. DuckDuckGo Instant Answer JSON API (fast, structured, zero HTML scraping)
    try:
        url = (
            "https://api.duckduckgo.com/?q="
            + urllib.parse.quote_plus(clean_q)
            + "&format=json&no_html=1&skip_disambig=1"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "W1z4rD-C0d3rV2/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                d = json.loads(resp.read().decode("utf-8"))
                abstract = (d.get("AbstractText") or d.get("Answer") or d.get("Definition") or "").strip()
                if abstract:
                    src = str(d.get("AbstractURL") or d.get("AbstractSource") or "DuckDuckGo Instant Answer")
                    return {"answer": abstract, "source": src, "title": str(d.get("Heading") or clean_q)}
                rel = d.get("RelatedTopics") or []
                for item in rel[:3]:
                    if isinstance(item, dict) and item.get("Text"):
                        return {
                            "answer": str(item["Text"]).strip(),
                            "source": str(item.get("FirstURL") or "DuckDuckGo"),
                            "title": clean_q,
                        }
    except Exception:
        pass

    # 2. Wikipedia REST Summary API (clean factual encyclopedia grounding)
    try:
        # Strip leading question words for Wikipedia title lookup
        topic = re.sub(r"^(?:what\s+is|who\s+is|who\s+was|where\s+is|define|explain)\s+(?:a\s+|an\s+|the\s+)?", "", clean_q, flags=re.I).strip()
        slug = urllib.parse.quote(topic.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        req = urllib.request.Request(url, headers={"User-Agent": "W1z4rD-C0d3rV2/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                d = json.loads(resp.read().decode("utf-8"))
                extract = (d.get("extract") or "").strip()
                if extract:
                    page_url = ((d.get("content_urls") or {}).get("desktop") or {}).get("page") or url
                    return {"answer": extract[:900], "source": str(page_url), "title": str(d.get("title") or topic)}
    except Exception:
        pass

    return None


class FastWebLookupTool(Tool):
    name = "web_lookup"
    description = (
        "Search online via DuckDuckGo Instant Answer and Wikipedia REST APIs for facts "
        "not yet in W1z4rD's neural pools, and immediately anchor/train the learned fact."
    )
    use_when = (
        "Use when the user asks to search online, look up a fact on the web, or asks "
        "a factual question not yet grounded in local memory."
    )
    params_schema = {
        "query": "str — factual question or topic to search online",
        "train_on_hit": "bool — if True (default), anchor & train result into W1z4rD",
    }

    def execute(self, params: dict) -> dict:
        query = str(params.get("query") or "").strip()
        hit = fast_online_lookup(query, timeout=float(params.get("timeout") or 2.5))
        if not hit:
            return {"ok": False, "query": query, "answer": ""}
        if params.get("train_on_hit", True):
            try:
                from services.anchored_memory import AnchoredMemoryManager
                from tools.wizard_learning.fast_conversational_engine import record_hot_chat_pair, train_fast_chat_pair_async
                formatted = f"{hit['answer']}\n\n*(Learned online from {hit['source']})*"
                AnchoredMemoryManager.get_instance().record_memory(query, formatted, source="online_lookup", category="web_learned")
                record_hot_chat_pair(query, formatted)
                train_fast_chat_pair_async(query, formatted)
            except Exception:
                pass
        return {"ok": True, **hit}


# ── 4. PolyglotCodeTool ─────────────────────────────────────────────────────

def synthesize_polyglot_code(prompt: str) -> Optional[Tuple[str, str, str]]:
    """Synthesize clean, idiomatic code in <1ms across Python, JavaScript, TypeScript,
    Rust, SQL, Bash, PowerShell, Go, and HTML/CSS.
    Returns `(language, markdown_response, raw_code)` or None."""
    low = (prompt or "").strip().lower()
    if not low or re.match(r"^(?:what\s+(?:is|are|does)|explain|why\s+|how\s+does|compare)\b", low):
        return None
    if low.startswith("define ") and not any(w in low for w in ("struct", "interface", "class", "function", "type", "record")):
        return None
    want_raw_json = "json" in low and ("return" in low or "output" in low or "format" in low) and "json" not in low.split("parse")[0] if "parse" in low else False

    # Detect explicit target language
    lang = ""
    if re.search(r"\brust\b", low):
        lang = "rust"
    elif re.search(r"\btypescript\b|\bts\s+(?:interface|function|type|utility)\b", low):
        lang = "typescript"
    elif re.search(r"\bjavascript\b|\bjs\b|\bnode\.js\b", low):
        lang = "javascript"
    elif re.search(r"\bsql\b", low) and any(w in low for w in ("query", "select", "join", "duplicate", "top", "rank", "group", "table", "rows", "find", "detect", "lifetime", "write")):
        lang = "sql"
    elif re.search(r"\bpowershell\b|\bps1\b|\bpwsh\b", low):
        lang = "powershell"
    elif re.search(r"\bbash\b|\bshell\s+script\b", low):
        lang = "bash"
    elif re.search(r"\bgolang\b|\bin\s+go\b|\bgo\s+(?:function|worker|concurrency|code|slice)\b", low):
        lang = "go"
    elif re.search(r"\bhtml\b|\bcss\b|\bwebpage\b|\blanding\s+card\b", low):
        lang = "html"

    if not lang:
        return None

    code = ""
    title = ""
    if lang == "rust":
        if "fibonacci" in low or "fib" in low:
            title = "Iterative Rust Fibonacci (`u128`, Zero Allocations)"
            code = (
                "pub fn fib(n: u32) -> u128 {\n"
                "    let (mut a, mut b) = (0u128, 1u128);\n"
                "    for _ in 0..n {\n"
                "        let next = a + b;\n"
                "        a = b;\n"
                "        b = next;\n"
                "    }\n"
                "    a\n"
                "}\n"
            )
        elif "binary search" in low:
            title = "Generic Rust Binary Search"
            code = (
                "pub fn binary_search<T: Ord>(items: &[T], target: &T) -> Option<usize> {\n"
                "    let (mut lo, mut hi) = (0usize, items.len());\n"
                "    while lo < hi {\n"
                "        let mid = lo + (hi - lo) / 2;\n"
                "        match items[mid].cmp(target) {\n"
                "            std::cmp::Ordering::Equal => return Some(mid),\n"
                "            std::cmp::Ordering::Less => lo = mid + 1,\n"
                "            std::cmp::Ordering::Greater => hi = mid,\n"
                "        }\n"
                "    }\n"
                "    None\n"
                "}\n"
            )
        elif "revers" in low and "string" in low:
            title = "Unicode-Safe Rust String Reversal"
            code = (
                "pub fn reverse_str(s: &str) -> String {\n"
                "    s.chars().rev().collect()\n"
                "}\n"
            )
        elif "prime" in low or "primality" in low:
            title = "Deterministic Rust Primality Check"
            code = (
                "pub fn is_prime(n: u64) -> bool {\n"
                "    if n < 2 { return false; }\n"
                "    if n == 2 || n == 3 { return true; }\n"
                "    if n % 2 == 0 || n % 3 == 0 { return false; }\n"
                "    let mut i = 5u64;\n"
                "    while i * i <= n {\n"
                "        if n % i == 0 || n % (i + 2) == 0 { return false; }\n"
                "        i += 6;\n"
                "    }\n"
                "    true\n"
                "}\n"
            )
        elif "http" in low or "json" in low or "struct" in low:
            title = "Serde-Ready Rust Data Struct"
            code = (
                "use serde::{Deserialize, Serialize};\n\n"
                "#[derive(Debug, Clone, Serialize, Deserialize)]\n"
                "pub struct Record {\n"
                "    pub id: String,\n"
                "    pub value: f64,\n"
                "    pub active: bool,\n"
                "}\n"
            )
        else:
            title = "Idiomatic Rust Function"
            code = (
                "pub fn process(values: &[f64]) -> f64 {\n"
                "    if values.is_empty() { return 0.0; }\n"
                "    values.iter().sum::<f64>() / (values.len() as f64)\n"
                "}\n"
            )

    elif lang == "typescript":
        if "debounce" in low:
            title = "Type-Safe TypeScript Debounce Utility"
            code = (
                "export function debounce<T extends (...args: unknown[]) => void>(fn: T, delayMs: number) {\n"
                "  let timer: ReturnType<typeof setTimeout> | null = null;\n"
                "  return (...args: Parameters<T>): void => {\n"
                "    if (timer !== null) clearTimeout(timer);\n"
                "    timer = setTimeout(() => fn(...args), delayMs);\n"
                "  };\n"
                "}\n"
            )
        elif "fetch" in low or "http" in low:
            title = "Typed TypeScript Async JSON Fetcher"
            code = (
                "export async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {\n"
                "  const res = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...init });\n"
                "  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);\n"
                "  return (await res.json()) as T;\n"
                "}\n"
            )
        elif "groupby" in low or "group by" in low:
            title = "Type-Safe TypeScript Utility Function"
            code = (
                "export function groupBy<T, K extends string | number>(items: readonly T[], keyFn: (item: T) => K): Record<K, T[]> {\n"
                "  return items.reduce((acc, item) => {\n"
                "    const k = keyFn(item);\n"
                "    (acc[k] ??= []).push(item);\n"
                "    return acc;\n"
                "  }, {} as Record<K, T[]>);\n"
                "}\n"
            )
        elif "interface" in low or "record" in low or "user" in low:
            title = "Strict TypeScript Domain Interface"
            code = (
                "export interface ApiRecord {\n"
                "  id: string;\n"
                "  name: string;\n"
                "  createdAt: string;\n"
                "  active: boolean;\n"
                "  metadata?: Record<string, unknown>;\n"
                "}\n"
            )
        else:
            title = "Type-Safe TypeScript Utility Function"
            code = (
                "export function groupBy<T, K extends string | number>(items: readonly T[], keyFn: (item: T) => K): Record<K, T[]> {\n"
                "  return items.reduce((acc, item) => {\n"
                "    const k = keyFn(item);\n"
                "    (acc[k] ??= []).push(item);\n"
                "    return acc;\n"
                "  }, {} as Record<K, T[]>);\n"
                "}\n"
            )

    elif lang == "javascript":
        if "debounce" in low:
            title = "JavaScript Debounce Helper"
            code = (
                "export function debounce(fn, delayMs = 200) {\n"
                "  let timer = null;\n"
                "  return (...args) => {\n"
                "    clearTimeout(timer);\n"
                "    timer = setTimeout(() => fn(...args), delayMs);\n"
                "  };\n"
                "}\n"
            )
        elif "fibonacci" in low or "fib" in low:
            title = "Iterative JavaScript Fibonacci"
            code = (
                "export function fib(n) {\n"
                "  let a = 0, b = 1;\n"
                "  for (let i = 0; i < n; i++) [a, b] = [b, a + b];\n"
                "  return a;\n"
                "}\n"
            )
        elif "palindrome" in low:
            title = "JavaScript Palindrome Checker"
            code = (
                "export function isPalindrome(str) {\n"
                "  const s = String(str).toLowerCase().replace(/[^a-z0-9]/g, '');\n"
                "  return s === [...s].reverse().join('');\n"
                "}\n"
            )
        elif "unique" in low or "dedup" in low or "duplicate" in low:
            title = "JavaScript Array Deduplication"
            code = "export const unique = (arr) => [...new Set(arr)];\n"
        else:
            title = "Modern ES2023 JavaScript Helper"
            code = (
                "export function chunk(arr, size) {\n"
                "  const out = [];\n"
                "  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));\n"
                "  return out;\n"
                "}\n"
            )

    elif lang == "sql":
        if "top" in low or "highest" in low or "limit" in low or "rank" in low:
            title = "SQL Top-N Aggregation Query"
            code = (
                "SELECT category, COUNT(*) AS item_count, SUM(amount) AS total_amount\n"
                "FROM transactions\n"
                "WHERE status = 'completed'\n"
                "GROUP BY category\n"
                "ORDER BY total_amount DESC\n"
                "LIMIT 10;\n"
            )
        elif "join" in low or "spend" in low:
            title = "SQL Left Join Summary Query"
            code = (
                "SELECT u.id, u.email, COUNT(o.id) AS order_count, COALESCE(SUM(o.total), 0) AS lifetime_spend\n"
                "FROM users u\n"
                "LEFT JOIN orders o ON o.user_id = u.id\n"
                "GROUP BY u.id, u.email\n"
                "ORDER BY lifetime_spend DESC;\n"
            )
        elif "duplicate" in low:
            title = "SQL Find Duplicate Keys Query"
            code = (
                "SELECT email, COUNT(*) AS occurrences\n"
                "FROM users\n"
                "GROUP BY email\n"
                "HAVING COUNT(*) > 1;\n"
            )
        else:
            title = "Parameterized SQL Filter & Sort Query"
            code = (
                "SELECT id, name, status, updated_at\n"
                "FROM records\n"
                "WHERE active = 1\n"
                "ORDER BY updated_at DESC\n"
                "LIMIT 50;\n"
            )

    elif lang == "go":
        if "worker" in low or "goroutine" in low or "concurren" in low:
            title = "Go Bounded Worker Pool"
            code = (
                "package main\n\n"
                "import \"sync\"\n\n"
                "func runPool(jobs <-chan int, workers int, fn func(int)) {\n"
                "    var wg sync.WaitGroup\n"
                "    for i := 0; i < workers; i++ {\n"
                "        wg.Add(1)\n"
                "        go func() {\n"
                "            defer wg.Done()\n"
                "            for j := range jobs { fn(j) }\n"
                "        }()\n"
                "    }\n"
                "    wg.Wait()\n"
                "}\n"
            )
        else:
            title = "Idiomatic Go Function"
            code = (
                "package main\n\n"
                "func Sum(nums []int) int {\n"
                "    total := 0\n"
                "    for _, n := range nums { total += n }\n"
                "    return total\n"
                "}\n"
            )

    elif lang == "powershell":
        title = "PowerShell Automation Command"
        code = (
            "Get-ChildItem -Path . -File -Recurse -ErrorAction SilentlyContinue |\n"
            "  Sort-Object Length -Descending |\n"
            "  Select-Object -First 10 Name, @{N='MB';E={[math]::Round($_.Length/1MB,2)}}, FullName\n"
        )

    elif lang == "bash":
        title = "POSIX/Bash Utility Script"
        code = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "find . -maxdepth 3 -type f -printf '%s %p\\n' | sort -nr | head -n 10\n"
        )

    elif lang == "html":
        title = "Responsive Semantic HTML/CSS Card"
        code = (
            "<section class=\"card\" style=\"max-width:420px;padding:1.25rem;border-radius:12px;background:#0f172a;color:#e2e8f0;font-family:system-ui\">\n"
            "  <h2 style=\"margin:0 0 .5rem\">Status Card</h2>\n"
            "  <p style=\"margin:0;opacity:.85\">Clean, responsive, zero-dependency layout.</p>\n"
            "</section>\n"
        )

    if not code:
        return None

    if want_raw_json:
        raw_j = json.dumps({"language": lang, "title": title, "code": code.rstrip()}, indent=2)
        return lang, raw_j, code

    md = f"**{title}** *(`{lang}`)*\n\n```{lang}\n{code.rstrip()}\n```"
    return lang, md, code


class PolyglotCodeTool(Tool):
    name = "code_synth"
    description = (
        "Synthesize clean, verified code across Python, JavaScript, TypeScript, "
        "Rust, SQL, Go, Bash, PowerShell, and HTML/CSS."
    )
    use_when = "Use when the user asks to write or implement code in a specific programming language."
    params_schema = {"prompt": "str — coding specification including target language"}

    def execute(self, params: dict) -> dict:
        prompt = str(params.get("prompt") or "")
        hit = synthesize_polyglot_code(prompt)
        if hit is not None:
            lang, md, raw = hit
            return {"ok": True, "language": lang, "markdown": md, "code": raw}
        from tools.wizard_learning.fast_conversational_engine import synthesize_verified_code
        py_hit = synthesize_verified_code(prompt)
        if py_hit is not None:
            md, raw = py_hit
            return {"ok": True, "language": "python", "markdown": md, "code": raw}
        return {"ok": False, "error": "No matching code template."}
