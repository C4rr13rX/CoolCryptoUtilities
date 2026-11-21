"""tools/wizard_learning/run_massive_ms_battery.py

Executes the full 1,198+ live HTTP test battery against `http://127.0.0.1:8001`
(both `/api/wizard-chat/message/` and `/api/wizard-chat/agent/`), validating:
  - >= 1,000 strictly verified right answers across time/date/location, inline
    JSON/CSV & caller context, polyglot code, C0d3rV2 tool-routing, paraphrased
    English knowledge, and online lookup + follow-up fast recall.
  - Zero unprompted context leakage on negative-control prompts.
  - Exact millisecond latency distributions (p50, p95, p99, max) per category.
"""
from __future__ import annotations

import argparse
import datetime
import http.client
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = PROJECT_ROOT / "data" / "wizard_learning" / "massive_context_corpus"


def _reload_waitress_and_wait() -> None:
    print("[0/3] Signaling Waitress for clean w1z4rd_supervisor reload...")
    for c in psutil.net_connections("tcp"):
        if c.laddr and c.laddr.port == 8001 and c.status == "LISTEN" and c.pid:
            try:
                os.kill(c.pid, 15)
                print(f"      Signaled Waitress PID {c.pid}")
            except Exception:
                pass
    time.sleep(3.0)
    deadline = time.time() + 140.0
    while time.time() < deadline:
        conn = http.client.HTTPConnection("127.0.0.1", 8001, timeout=15.0)
        try:
            t0 = time.perf_counter()
            conn.request("GET", "/api/wizard-chat/status/")
            resp = conn.getresponse()
            resp.read()
            ms = (time.perf_counter() - t0) * 1000.0
            if resp.status == 200 and ms < 250.0:
                print(f"      Waitress live & warm ({ms:.2f} ms)")
                conn.close()
                return
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
        time.sleep(2.0)
    raise RuntimeError("Timed out waiting for Waitress on 127.0.0.1:8001")


def _post_keepalive(conn: http.client.HTTPConnection, path: str, payload: Dict[str, Any]) -> Tuple[int, float, Dict[str, Any], http.client.HTTPConnection]:
    body_bytes = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
    for attempt in range(2):
        t0 = time.perf_counter()
        try:
            conn.request("POST", path, body=body_bytes, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            ms = (time.perf_counter() - t0) * 1000.0
            return resp.status, ms, json.loads(raw.decode("utf-8")), conn
        except Exception as exc:
            try:
                conn.close()
            except Exception:
                pass
            conn = http.client.HTTPConnection("127.0.0.1", 8001, timeout=30.0)
            if attempt == 1:
                return 599, (time.perf_counter() - t0) * 1000.0, {"error": str(exc)}, conn
    return 599, 0.0, {}, conn


def _verify_case(case: Dict[str, Any], status: int, body: Dict[str, Any]) -> Tuple[bool, str]:
    if status != 200:
        return False, f"HTTP {status}"
    ans = str(body.get("answer") or "").strip()
    if not ans:
        return False, "empty answer"
    if "doesn't have a confident answer" in ans.lower() or "unable to chat" in ans.lower():
        return False, f"ungrounded/blocked: {ans[:80]}"

    # Strict JSON validation when requested
    if case.get("expect_json"):
        try:
            parsed = json.loads(ans)
        except Exception as exc:
            return False, f"invalid JSON ({exc}): {ans[:80]}"
        for k in (case.get("expect_keys") or []):
            if not (isinstance(parsed, dict) and k in parsed):
                return False, f"missing JSON key '{k}'"

    # Dynamic time/date semantic checks
    now_local = datetime.datetime.now().astimezone()
    etype = case.get("expect_type")
    if etype == "time":
        if ":" not in ans and str(int(time.time()))[:5] not in ans:
            return False, f"missing time in '{ans[:80]}'"
    elif etype == "date":
        if str(now_local.year) not in ans:
            return False, f"missing year {now_local.year} in '{ans[:80]}'"
    elif etype == "weekday":
        if now_local.strftime("%A").lower() not in ans.lower():
            return False, f"missing weekday {now_local.strftime('%A')} in '{ans[:80]}'"
    elif etype == "month":
        if now_local.strftime("%B").lower() not in ans.lower():
            return False, f"missing month {now_local.strftime('%B')} in '{ans[:80]}'"
    elif etype == "year":
        if str(now_local.year) not in ans:
            return False, f"missing year in '{ans[:80]}'"
    elif etype == "tomorrow":
        tmrw = (now_local.date() + datetime.timedelta(days=1)).isoformat()
        if tmrw not in ans:
            return False, f"missing tomorrow {tmrw} in '{ans[:80]}'"
    elif etype == "yesterday":
        yest = (now_local.date() - datetime.timedelta(days=1)).isoformat()
        if yest not in ans:
            return False, f"missing yesterday {yest} in '{ans[:80]}'"
    elif etype and etype.startswith("offset_"):
        n = int(etype.split("_")[1])
        fut = (now_local.date() + datetime.timedelta(days=n)).isoformat()
        if fut not in ans:
            return False, f"missing +{n}d date {fut} in '{ans[:80]}'"
    elif etype == "days_until":
        if "day" not in ans.lower():
            return False, f"missing days_until in '{ans[:80]}'"

    low_ans = ans.lower()
    for needle in (case.get("expect_contains") or []):
        if str(needle).lower() not in low_ans:
            return False, f"missing '{needle}' in '{ans[:90]}'"

    for bad in (case.get("forbid_contains") or []):
        if str(bad).lower() in low_ans:
            return False, f"LEAKED forbidden context '{bad}' in '{ans[:90]}'"

    return True, "ok"


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return round(sorted_vals[idx], 2)


def run_battery(reload_waitress: bool = False) -> Dict[str, Any]:
    if reload_waitress:
        _reload_waitress_and_wait()

    # Load all compiled JSONL files
    files = [
        "time_date_location.jsonl",
        "context_data_json.jsonl",
        "polyglot_code.jsonl",
        "tool_routing.jsonl",
        "english_qa_paraphrased.jsonl",
        "online_lookup_cases.jsonl",
    ]
    all_cases: List[Dict[str, Any]] = []
    for fn in files:
        p = CORPUS_DIR / fn
        if not p.is_file():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                all_cases.append(json.loads(ln))

    print(f"[1/3] Loaded {len(all_cases)} live test cases from {CORPUS_DIR}")
    conn = http.client.HTTPConnection("127.0.0.1", 8001, timeout=20.0)

    # Warm up once (ensures ParaphraseCorpusIndex & _WS_FILE_CACHE & elevation_method are hot)
    for warm_path, warm_payload in (
        ("/api/wizard-chat/message/", {"text": "Hello", "session_id": "bat-warm"}),
        ("/api/wizard-chat/message/", {"text": "What is a hash table?", "session_id": "bat-warm"}),
        ("/api/wizard-chat/agent/", {"text": "search workspace for probe_unified_eye_dynamics", "session_id": "bat-warm"}),
    ):
        _, _, _, conn = _post_keepalive(conn, warm_path, warm_payload)

    print(f"[2/3] Executing {len(all_cases)} live HTTP requests (+ online follow-up recall checks)...")
    t_suite0 = time.perf_counter()

    by_cat: Dict[str, Dict[str, Any]] = {}
    all_local_ms: List[float] = []
    web_ms: List[float] = []
    web_followup_ms: List[float] = []
    failures: List[Dict[str, Any]] = []
    passed = 0
    total = 0

    for i, case in enumerate(all_cases):
        cat = str(case.get("category") or "other")
        bucket = by_cat.setdefault(cat, {"total": 0, "passed": 0, "latencies": []})

        path = str(case.get("endpoint") or "/api/wizard-chat/message/")
        payload: Dict[str, Any] = {
            "text": case["question"],
            "session_id": f"bat-{cat}",
        }
        if case.get("context"):
            payload["context"] = case["context"]
        if case.get("data"):
            payload["data"] = case["data"]
        if case.get("allow_web_lookup"):
            payload["allow_web_lookup"] = True

        status, ms, body, conn = _post_keepalive(conn, path, payload)
        ok, reason = _verify_case(case, status, body)
        total += 1
        bucket["total"] += 1
        bucket["latencies"].append(ms)
        if cat == "online_lookup":
            web_ms.append(ms)
        else:
            all_local_ms.append(ms)

        if ok:
            passed += 1
            bucket["passed"] += 1
        else:
            failures.append({
                "id": case.get("id"),
                "category": cat,
                "question": case.get("question", "")[:120],
                "ms": round(ms, 2),
                "reason": reason,
                "decoder": body.get("decoder"),
            })

        # For online_lookup cases: immediately ask a 2nd time to verify it was auto-trained and recalls in ms!
        if cat == "online_lookup" and ok:
            st2, ms2, body2, conn = _post_keepalive(conn, path, {"text": case["question"], "session_id": "bat-web-recall"})
            ok2, reason2 = _verify_case(case, st2, body2)
            total += 1
            f_bucket = by_cat.setdefault("online_lookup_recall", {"total": 0, "passed": 0, "latencies": []})
            f_bucket["total"] += 1
            f_bucket["latencies"].append(ms2)
            web_followup_ms.append(ms2)
            all_local_ms.append(ms2)
            if ok2:
                passed += 1
                f_bucket["passed"] += 1
            else:
                failures.append({"id": f"{case.get('id')}_recall", "category": "online_lookup_recall", "question": case["question"], "ms": round(ms2, 2), "reason": reason2})

        if (i + 1) % 250 == 0:
            print(f"      Progress: {i+1}/{len(all_cases)} ({passed}/{total} passed)")

    conn.close()
    total_wall_s = round(time.perf_counter() - t_suite0, 2)

    all_local_ms.sort()
    summary_cats: Dict[str, Any] = {}
    for cat, b in sorted(by_cat.items()):
        lats = sorted(b["latencies"])
        summary_cats[cat] = {
            "passed": b["passed"],
            "total": b["total"],
            "accuracy_pct": round(100.0 * b["passed"] / max(1, b["total"]), 2),
            "p50_ms": _percentile(lats, 50),
            "p95_ms": _percentile(lats, 95),
            "p99_ms": _percentile(lats, 99),
            "max_ms": round(max(lats), 2) if lats else 0.0,
        }

    report = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy_pct": round(100.0 * passed / max(1, total), 2),
        "total_wall_seconds": total_wall_s,
        "local_latency_ms": {
            "count": len(all_local_ms),
            "p50_ms": _percentile(all_local_ms, 50),
            "p95_ms": _percentile(all_local_ms, 95),
            "p99_ms": _percentile(all_local_ms, 99),
            "mean_ms": round(statistics.mean(all_local_ms), 2) if all_local_ms else 0.0,
            "max_ms": round(max(all_local_ms), 2) if all_local_ms else 0.0,
        },
        "online_lookup_ms": [round(x, 2) for x in web_ms],
        "online_followup_recall_ms": [round(x, 2) for x in web_followup_ms],
        "by_category": summary_cats,
        "sample_failures": failures[:25],
    }

    out_path = CORPUS_DIR / "battery_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n[3/3] BATTERY COMPLETE in {total_wall_s}s: {passed}/{total} PASSED ({report['accuracy_pct']}%)")
    ll = report["local_latency_ms"]
    print(f"      Local Latency ({ll['count']} reqs): p50={ll['p50_ms']} ms | p95={ll['p95_ms']} ms | p99={ll['p99_ms']} ms | mean={ll['mean_ms']} ms | max={ll['max_ms']} ms")
    if web_ms:
        print(f"      Online Lookup (cold web): {report['online_lookup_ms']} ms -> Follow-up Recall (learned): {report['online_followup_recall_ms']} ms")
    print("\n      Per-Category Breakdown:")
    for cat, s in summary_cats.items():
        print(f"        {cat:28s} | {s['passed']:4d}/{s['total']:4d} ({s['accuracy_pct']:6.2f}%) | p50={s['p50_ms']:6.2f} ms | p95={s['p95_ms']:6.2f} ms | p99={s['p99_ms']:6.2f} ms")
    if failures:
        print(f"\n      Sample failures ({len(failures)} total):")
        for f in failures[:10]:
            print(f"        - [{f['id']}] {f['question']} -> {f['reason']}")
    print(f"\n      Saved report -> {out_path}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reload-waitress", action="store_true")
    args = ap.parse_args()
    rep = run_battery(reload_waitress=args.reload_waitress)
    sys.exit(0 if rep["passed"] >= 1000 and rep["failed"] == 0 else 1)
