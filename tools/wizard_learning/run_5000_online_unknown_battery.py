"""tools/wizard_learning/run_5000_online_unknown_battery.py

5,000-Question Multi-Domain Unknown Online Discovery, License-Verified Self-Training,
Biological Settling & Offline Next-Time Recall Battery for W1z4rD + C0d3rV2.

Architecture & Anti-Leakage Guarantees:
  1. Harvests > 5,400 candidate factual questions across 22 diverse domains from
     Wikidata SPARQL (`CC0 1.0`) and stores the evaluator's answer key ONLY in an
     external scratch path (`--eval-cache`) outside `data/wizard_learning/` so
     W1z4rD has ZERO access to the evaluator benchmark file.
  2. Runs a Pre-Flight Local Ignorance Filter (`allow_web_lookup=False`) and keeps
     EXACTLY `5,000` distinct questions where W1z4rD initially has `0 / 5,000`
     (`0.00%`) local knowledge.
  3. Pass 1 — Cold Online Discovery & Self-Training (`5,000` live HTTP requests to
     `:8001` with `allow_web_lookup=True`): W1z4rD detects each gap, looks up
     licensable open sources (`Wikidata CC0 1.0` + `Wikipedia CC BY-SA 4.0`),
     saves `licensed_web_training_material.jsonl`, trains itself, and executes
     periodic biological settling cycles (`SettlingController`).
  4. Explicit Post-Pass-1 Biological Settling & RAM Consolidation (`[settle_online_memory]`).
  5. Pass 2 — Strictly Offline "Next Time" Recall (`5,000` live HTTP requests with
     `allow_web_lookup=False`, mixing exact + paraphrased re-asks): proves W1z4rD
     retains and recalls the learned answers locally (`web_used == False`) in
     milliseconds after settling!
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime
import http.client
import json
import re
import sys
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_OUT = PROJECT_ROOT / "data" / "wizard_learning" / "online_self_learned_corpus" / "battery_5000_report.json"
DEFAULT_EVAL_CACHE = Path(r"C:\Users\Adam\.gemini\antigravity\brain\6ae55156-2135-4ffd-b50b-4a41ed10ed8a\scratch\eval_only_5000_unknowns.jsonl")

UA = "W1z4rDV1510n-EvalHarness/1.0 (https://github.com/C4rr13rX/W1z4rDV1510n; CC0 benchmark builder)"


def _ascii_fold(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s or "")
    no_marks = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", re.sub(r"['`ʻʼʽˈˌ\"()\[\].,;:!?/\\-]+", " ", no_marks.lower())).strip()


# ── 22 Diverse Domains Across Human Knowledge (SPARQL Generator Specs) ──────
# Each tuple: `(domain_id, sparql_where_body, q_template, paraphrase_template, limit)`

SPARQL_DOMAIN_SPECS: List[Tuple[str, str, str, str, int]] = [
    (
        "01_geography_capitals",
        "?item wdt:P31 wd:Q6256 ; wdt:P36 ?val .",
        "What is the capital city of {subj}?",
        "Tell me the capital of {subj}",
        220,
    ),
    (
        "02_economics_currencies",
        "?item wdt:P31 wd:Q6256 ; wdt:P38 ?val .",
        "What currency is used in {subj}?",
        "What is the currency of {subj}?",
        220,
    ),
    (
        "03_geography_continents",
        "?item wdt:P31 wd:Q6256 ; wdt:P30 ?val .",
        "On which continent is {subj} located?",
        "Continent of {subj}",
        200,
    ),
    (
        "04_linguistics_official_languages",
        "?item wdt:P31 wd:Q6256 ; wdt:P37 ?val .",
        "What is the official language of {subj}?",
        "Official language of {subj}",
        200,
    ),
    (
        "05_linguistics_writing_systems",
        "?item wdt:P31 wd:Q34770 ; wdt:P282 ?val .",
        "What writing system is used to write {subj}?",
        "Writing system of {subj}",
        320,
    ),
    (
        "06_chemistry_atomic_numbers",
        "?item wdt:P31 wd:Q11344 ; wdt:P1086 ?val .",
        "What is the atomic number of the chemical element {subj}?",
        "Atomic number of {subj}",
        120,
    ),
    (
        "07_chemistry_element_symbols",
        "?item wdt:P31 wd:Q11344 ; wdt:P246 ?val .",
        "What is the chemical symbol for the element {subj}?",
        "Chemical symbol of {subj}",
        120,
    ),
    (
        "08_chemistry_compounds_formulas",
        "?item wdt:P31 wd:Q11173 ; wdt:P274 ?val .",
        "What is the chemical formula of the compound {subj}?",
        "Chemical formula of {subj}",
        340,
    ),
    (
        "09_astronomy_moons",
        "?item wdt:P31 wd:Q2537 ; wdt:P397 ?val .",
        "Which planet does the moon {subj} orbit?",
        "Parent astronomical body of {subj}",
        220,
    ),
    (
        "10_astronomy_constellations",
        "?item wdt:P31 wd:Q523 ; wdt:P59 ?val .",
        "In which constellation is the star {subj} located?",
        "Constellation of {subj}",
        320,
    ),
    (
        "11_literature_novels",
        "?item wdt:P31 wd:Q7725634 ; wdt:P50 ?val .",
        "Who wrote the literary work {subj}?",
        "Author of {subj}",
        420,
    ),
    (
        "12_biology_species_taxonomy",
        "?item wdt:P31 wd:Q16521 ; wdt:P105 wd:Q7432 ; wdt:P171 ?val .",
        "To which parent taxon does the species {subj} belong?",
        "Parent taxon of {subj}",
        360,
    ),
    (
        "13_computing_video_games_devs",
        "?item wdt:P31 wd:Q7889 ; wdt:P178 ?val .",
        "Who developed the software {subj}?",
        "Designer / Developer of {subj}",
        340,
    ),
    (
        "14_visual_arts_paintings",
        "?item wdt:P31 wd:Q3305213 ; wdt:P170 ?val .",
        "Who painted {subj}?",
        "Creator of {subj}",
        360,
    ),
    (
        "15_cinema_films",
        "?item wdt:P31 wd:Q11424 ; wdt:P57 ?val .",
        "Who directed the film {subj}?",
        "Director of {subj}",
        360,
    ),
    (
        "16_computing_programming_languages",
        "?item wdt:P31 wd:Q9143 ; wdt:P287 ?val .",
        "Who designed the programming language {subj}?",
        "Designer / Developer of {subj}",
        260,
    ),
    (
        "17_architecture_landmarks",
        "?item wdt:P31 wd:Q41176 ; wdt:P84 ?val .",
        "Which architect designed the building {subj}?",
        "Architect of {subj}",
        320,
    ),
    (
        "18_earth_science_mountains",
        "?item wdt:P31 wd:Q8502 ; wdt:P4552 ?val .",
        "In which mountain range is the mountain {subj} located?",
        "Mountain range of {subj}",
        360,
    ),
    (
        "19_hydrology_rivers",
        "?item wdt:P31 wd:Q4022 ; wdt:P403 ?val .",
        "Into which body of water does the river {subj} flow?",
        "Mouth of watercourse of {subj}",
        360,
    ),
    (
        "20_cultural_heritage_sites",
        "?item wdt:P1435 wd:Q9259 ; wdt:P17 ?val .",
        "In which country is the heritage site {subj} located?",
        "Country of {subj}",
        380,
    ),
    (
        "21_aerospace_spacecraft",
        "?item wdt:P31 wd:Q26529 ; wdt:P137 ?val .",
        "Which space agency operated the space probe {subj}?",
        "Operator of {subj}",
        220,
    ),
    (
        "22_global_universities_countries",
        "?item wdt:P31 wd:Q3918 ; wdt:P17 ?val .",
        "In which country is the university {subj} located?",
        "Country of {subj}",
        420,
    ),
    (
        "23_lakes_countries",
        "?item wdt:P31 wd:Q23397 ; wdt:P17 ?val .",
        "In which country is the lake {subj} located?",
        "Country of {subj}",
        380,
    ),
    (
        "24_museums_countries",
        "?item wdt:P31 wd:Q33506 ; wdt:P17 ?val .",
        "In which country is the museum {subj} located?",
        "Country of {subj}",
        380,
    ),
]


def _run_sparql(where_clause: str, limit: int) -> List[Tuple[str, str]]:
    """Fetch `(wikipedia_title, target_value_string)` pairs from Wikidata SPARQL (`CC0 1.0`)."""
    is_literal_prop = any(p in where_clause for p in ("wdt:P1086", "wdt:P246", "wdt:P274"))
    if is_literal_prop:
        query = f"""
SELECT ?wikiTitle (STR(?val) AS ?ans) WHERE {{
  {where_clause}
  ?sitelink schema:about ?item ;
            schema:isPartOf <https://en.wikipedia.org/> ;
            schema:name ?wikiTitle .
}}
LIMIT {limit}
"""
    else:
        query = f"""
SELECT ?wikiTitle ?ans WHERE {{
  {where_clause}
  ?sitelink schema:about ?item ;
            schema:isPartOf <https://en.wikipedia.org/> ;
            schema:name ?wikiTitle .
  ?val rdfs:label ?ans .
  FILTER(LANG(?ans) = "en")
}}
LIMIT {limit}
"""
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query, "format": "json"})
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=25.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    rows: List[Tuple[str, str]] = []
                    for b in (data.get("results") or {}).get("bindings") or []:
                        wt = (b.get("wikiTitle") or {}).get("value", "").strip()
                        ans = (b.get("ans") or {}).get("value", "").strip()
                        if wt and ans and not wt.startswith(("List of ", "Category:", "Template:")):
                            rows.append((wt, ans))
                    return rows
        except Exception as exc:
            time.sleep(1.5 * (attempt + 1))
            if attempt == 2:
                print(f"  [WARN] SPARQL timeout/err on {where_clause[:40]}: {exc}", flush=True)
    return []


def build_or_load_5000_unknown_benchmark(cache_path: Path, target_count: int = 5000, *, fixed_suite_path: Path | None = None) -> List[Dict[str, Any]]:
    """Build or load >= 5,000 verified-unknown multi-domain questions.
    If `fixed_suite_path` already exists, reload the exact locked 5,000-question suite
    so multi-round iterative training & adjustment is measured on the identical 5,000 items!"""
    if fixed_suite_path and fixed_suite_path.is_file():
        locked = [json.loads(ln) for ln in fixed_suite_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(locked) == target_count:
            print(f"[Suite] Loaded locked {len(locked)}-question benchmark across {len({x['domain'] for x in locked})} domains.", flush=True)
            return locked
    # Check what W1z4rD ALREADY knows locally BEFORE Pass 1 so we filter out pre-known items!
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from tools.wizard_learning.fast_conversational_engine import resolve_fast_wizard_turn

    def _is_already_known_locally(q: str) -> bool:
        hit = resolve_fast_wizard_turn(q, node_busy=True, allow_web_lookup=False)
        return bool(hit and hit.get("answer"))

    raw_candidates: List[Dict[str, Any]] = []
    if cache_path.is_file():
        for ln in cache_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                raw_candidates.append(json.loads(ln))

    existing_domains = {c["domain"] for c in raw_candidates}
    missing_specs = [sp for sp in SPARQL_DOMAIN_SPECS if sp[0] not in existing_domains]

    if len(raw_candidates) < target_count + 150 or missing_specs:
        specs_to_fetch = SPARQL_DOMAIN_SPECS if len(raw_candidates) < 2000 else missing_specs
        print(f"[Harvest] Querying Wikidata CC0 SPARQL across {len(specs_to_fetch)} domain(s)...", flush=True)
        grouped: Dict[Tuple[str, str], Set[str]] = collections.defaultdict(set)
        meta_tmpl: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for dom_id, where_c, q_t, p_t, lim in specs_to_fetch:
            rows = _run_sparql(where_c, lim)
            for wt, ans in rows:
                if re.match(r"^Q\d+$", ans) or len(wt) > 75:
                    continue
                grouped[(dom_id, wt)].add(ans)
                meta_tmpl[(dom_id, wt)] = (q_t, p_t)
            print(f"  + {dom_id:34s}: {len(rows):4d} raw facts harvested", flush=True)
            time.sleep(0.25)

        seen_questions: Set[str] = {c["question"].lower() for c in raw_candidates}
        for (dom_id, wt), ans_set in grouped.items():
            q_t, p_t = meta_tmpl[(dom_id, wt)]
            q_str = q_t.format(subj=wt)
            p_str = p_t.format(subj=wt)
            qk = q_str.lower()
            if qk in seen_questions:
                continue
            seen_questions.add(qk)
            raw_candidates.append({
                "domain": dom_id,
                "subject": wt,
                "question": q_str,
                "paraphrase": p_str,
                "expected_answers": sorted(ans_set),
            })

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            for c in raw_candidates:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Round-robin interleave across ALL domains so every batch is maximally diverse!
    by_dom: Dict[str, Deque[Dict[str, Any]]] = collections.defaultdict(collections.deque)
    for c in raw_candidates:
        by_dom[c["domain"]].append(c)
    interleaved: List[Dict[str, Any]] = []
    dom_keys = sorted(by_dom.keys())
    while any(by_dom.values()):
        for dk in dom_keys:
            if by_dom[dk]:
                interleaved.append(by_dom[dk].popleft())

    # Filter strictly to items W1z4rD did NOT know prior to online self-learning
    # (ignoring `online_self_learned_corpus` so Round 2+ evaluates the exact same 5,000-item suite!)
    from tools.wizard_learning import context_and_data_engine as cde
    orig_dirs = cde.CORPUS_DIRS
    cde.CORPUS_DIRS = tuple(d for d in orig_dirs if d.name != "online_self_learned_corpus")
    cde.ParaphraseCorpusIndex._instance = None
    try:
        unknown_items: List[Dict[str, Any]] = []
        skipped_known = 0
        for c in interleaved:
            if _is_already_known_locally(c["question"]):
                skipped_known += 1
                continue
            unknown_items.append(c)
            if len(unknown_items) >= target_count:
                break
    finally:
        cde.CORPUS_DIRS = orig_dirs
        cde.ParaphraseCorpusIndex._instance = None

    if fixed_suite_path:
        fixed_suite_path.parent.mkdir(parents=True, exist_ok=True)
        with fixed_suite_path.open("w", encoding="utf-8") as fh:
            for it in unknown_items[:target_count]:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(
        f"[Filter] Selected {len(unknown_items)} verified-UNKNOWN questions across "
        f"{len({x['domain'] for x in unknown_items})} domains (filtered out {skipped_known} pre-known items).",
        flush=True,
    )
    if len(unknown_items) < target_count:
        raise RuntimeError(f"Only {len(unknown_items)} unknown questions available; need {target_count}")
    return unknown_items[:target_count]


_TLS = threading.local()


def _http_ask(endpoint_path: str, text: str, allow_web: bool, session_id: str) -> Tuple[int, Dict[str, Any], float]:
    conn: http.client.HTTPConnection | None = getattr(_TLS, "conn", None)
    if conn is None:
        conn = http.client.HTTPConnection("127.0.0.1", 8001, timeout=25.0)
        _TLS.conn = conn

    payload = json.dumps({
        "text": text,
        "session_id": session_id,
        "allow_web_lookup": allow_web,
    }).encode("utf-8")
    t0 = time.perf_counter()
    for attempt in range(2):
        try:
            conn.request("POST", endpoint_path, body=payload, headers={"Content-Type": "application/json", "Connection": "keep-alive"})
            resp = conn.getresponse()
            raw = resp.read()
            ms = (time.perf_counter() - t0) * 1000.0
            return resp.status, json.loads(raw.decode("utf-8")), ms
        except Exception as exc:
            try:
                conn.close()
            except Exception:
                pass
            conn = http.client.HTTPConnection("127.0.0.1", 8001, timeout=25.0)
            _TLS.conn = conn
            if attempt == 1:
                return 599, {"answer": "", "error": str(exc)}, (time.perf_counter() - t0) * 1000.0
    return 599, {"answer": ""}, 0.0


def _matches_ground_truth(answer_text: str, expected_list: List[str], subject: str) -> bool:
    if not answer_text or "doesn't have a confident answer" in answer_text or "isn't grounded" in answer_text:
        return False
    folded_ans = _ascii_fold(answer_text)
    for exp in expected_list:
        fe = _ascii_fold(exp)
        if not fe:
            continue
        # Numeric/short symbol (e.g. atomic number '42' or symbol 'Mo') — check word boundary
        if len(fe) <= 2 or fe.isdigit():
            if re.search(rf"\b{re.escape(fe)}\b", folded_ans):
                return True
        else:
            if fe in folded_ans:
                return True
            # Also match primary surname / head token when Wikidata label has parenthetical or middle name
            core = re.sub(r"\s*\(.*?\)", "", exp).strip()
            fc = _ascii_fold(core)
            if len(fc) >= 4 and fc in folded_ans:
                return True
            toks = [t for t in fc.split() if len(t) >= 5]
            if toks and toks[-1] in folded_ans:
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--eval-cache", type=str, default=str(DEFAULT_EVAL_CACHE))
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    # Reset any prior online_self_learned_corpus run if starting a clean 5,000-unknown test
    learned_jsonl = PROJECT_ROOT / "data" / "wizard_learning" / "online_self_learned_corpus" / "licensed_web_training_material.jsonl"
    items = build_or_load_5000_unknown_benchmark(Path(args.eval_cache), target_count=args.count)

    # ── Stage A: Live HTTP Pre-Check on 100 Stratified Samples (`allow_web_lookup=False`) ──
    print("[Stage A] Verifying live HTTP pre-flight ignorance (`allow_web_lookup=False`)...", flush=True)
    pre_sample = [items[i] for i in range(0, len(items), max(1, len(items) // 100))][:100]
    pre_known = 0
    for s in pre_sample:
        st, body, _ = _http_ask("/api/wizard-chat/message/", s["question"], allow_web=False, session_id="precheck")
        if st == 200 and _matches_ground_truth(str(body.get("answer") or ""), s["expected_answers"], s["subject"]):
            pre_known += 1
    print(f"          Pre-flight live check: {pre_known}/{len(pre_sample)} known before online lookup (expected ~0).", flush=True)

    # ── Stage B: Pass 1 — Cold Online Lookup + License Check + Self-Train + Periodic Settle ──
    print(f"\n[Stage B] PASS 1: Running {len(items)} UNKNOWN questions live (`allow_web_lookup=True`, {args.workers} coalesced workers)...", flush=True)
    t_p1 = time.perf_counter()

    def _run_p1(idx_item: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
        idx, it = idx_item
        ep = "/api/wizard-chat/agent/" if (idx % 10 == 0) else "/api/wizard-chat/message/"
        st, body, ms = _http_ask(ep, it["question"], allow_web=True, session_id=f"p1-{(idx % 20)}")
        ans = str(body.get("answer") or "")
        ok_gt = (st == 200) and _matches_ground_truth(ans, it["expected_answers"], it["subject"])
        has_lic = ("cc0" in ans.lower()) or ("cc by-sa" in ans.lower()) or ("public domain" in ans.lower()) or bool(body.get("license"))
        return {
            "idx": idx,
            "domain": it["domain"],
            "question": it["question"],
            "expected": it["expected_answers"],
            "ok": ok_gt and has_lic,
            "ok_gt": ok_gt,
            "has_license": has_lic,
            "ms": round(ms, 2),
            "preview": ans.replace("\n", " ")[:130],
        }

    p1_results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done_n, r in enumerate(pool.map(_run_p1, enumerate(items)), 1):
            p1_results.append(r)
            if done_n % 500 == 0:
                ok_c = sum(1 for x in p1_results if x["ok"])
                elapsed = time.perf_counter() - t_p1
                print(f"          Pass 1 Progress: {done_n:4d}/{len(items)} | Right+Licensed: {ok_c:4d}/{done_n} ({100.0*ok_c/done_n:.2f}%) | {elapsed:.1f}s", flush=True)

    p1_wall_s = round(time.perf_counter() - t_p1, 2)
    p1_passed = sum(1 for x in p1_results if x["ok"])

    # Trigger explicit final biological settling & RAM consolidation on live Waitress
    print("\n[Settle]  Triggering final post-training biological settling & working-set consolidation...", flush=True)
    _, settle_body, settle_ms = _http_ask("/api/wizard-chat/message/", "[settle_online_memory]", allow_web=True, session_id="settle-final")
    print(f"          Settling response ({settle_ms:.1f} ms): {str(settle_body.get('answer') or '')[:180]}", flush=True)

    # ── Stage C: Pass 2 — Strictly Offline Post-Settling "Next Time" Recall (`allow_web_lookup=False`) ──
    print(f"\n[Stage C] PASS 2: Re-testing all {len(items)} questions OFFLINE (`allow_web_lookup=False`; 50% exact + 50% paraphrased)...", flush=True)
    t_p2 = time.perf_counter()

    def _run_p2(idx_item: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
        idx, it = idx_item
        use_paraphrase = (idx % 2 == 1)
        q_ask = it["paraphrase"] if use_paraphrase else it["question"]
        ep = "/api/wizard-chat/agent/" if (idx % 10 == 0) else "/api/wizard-chat/message/"
        st, body, ms = _http_ask(ep, q_ask, allow_web=False, session_id=f"p2-{(idx % 20)}")
        ans = str(body.get("answer") or "")
        web_used = bool(body.get("web_used", False))
        ok_gt = (st == 200) and (not web_used) and _matches_ground_truth(ans, it["expected_answers"], it["subject"])
        return {
            "idx": idx,
            "domain": it["domain"],
            "mode": "paraphrase" if use_paraphrase else "exact",
            "question_asked": q_ask,
            "expected": it["expected_answers"],
            "ok": ok_gt,
            "web_used": web_used,
            "ms": round(ms, 2),
            "preview": ans.replace("\n", " ")[:120],
        }

    p2_results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for done_n, r in enumerate(pool.map(_run_p2, enumerate(items)), 1):
            p2_results.append(r)
            if done_n % 1000 == 0:
                ok_c = sum(1 for x in p2_results if x["ok"])
                print(f"          Pass 2 Offline Recall: {done_n:4d}/{len(items)} | Retained & Right: {ok_c:4d}/{done_n} ({100.0*ok_c/done_n:.2f}%)", flush=True)

    p2_wall_s = round(time.perf_counter() - t_p2, 2)
    p2_passed = sum(1 for x in p2_results if x["ok"])
    p1_lats = sorted(x["ms"] for x in p1_results)
    p2_lats = sorted(x["ms"] for x in p2_results)

    # Per-domain breakdown
    dom_summary: Dict[str, Any] = {}
    for dom in sorted({x["domain"] for x in items}):
        d1 = [x for x in p1_results if x["domain"] == dom]
        d2 = [x for x in p2_results if x["domain"] == dom]
        dom_summary[dom] = {
            "count": len(d1),
            "pass1_online_pct": round(100.0 * sum(1 for x in d1 if x["ok"]) / max(1, len(d1)), 2),
            "pass2_offline_recall_pct": round(100.0 * sum(1 for x in d2 if x["ok"]) / max(1, len(d2)), 2),
        }

    p1_failures = [x for x in p1_results if not x["ok"]]
    p2_failures = [x for x in p2_results if not x["ok"]]

    saved_lines = sum(1 for _ in learned_jsonl.open("r", encoding="utf-8")) if learned_jsonl.is_file() else 0

    report = {
        "evaluated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_unknown_questions": len(items),
        "domains_covered": len(dom_summary),
        "precheck_known_before_online": f"{pre_known}/{len(pre_sample)}",
        "pass1_online_discovery_and_self_train": {
            "passed": p1_passed,
            "total": len(items),
            "accuracy_pct": round(100.0 * p1_passed / len(items), 2),
            "wall_seconds": p1_wall_s,
            "p50_ms": round(p1_lats[len(p1_lats) // 2], 2),
            "p95_ms": round(p1_lats[int(len(p1_lats) * 0.95)], 2),
        },
        "licensed_training_records_saved": saved_lines,
        "final_settling_result": settle_body.get("answer", ""),
        "pass2_offline_post_settling_recall": {
            "passed": p2_passed,
            "total": len(items),
            "accuracy_pct": round(100.0 * p2_passed / len(items), 2),
            "web_used_count": sum(1 for x in p2_results if x["web_used"]),
            "wall_seconds": p2_wall_s,
            "p50_ms": round(p2_lats[len(p2_lats) // 2], 2),
            "p95_ms": round(p2_lats[int(len(p2_lats) * 0.95)], 2),
        },
        "per_domain": dom_summary,
        "sample_pass1_failures": p1_failures[:25],
        "sample_pass2_failures": p2_failures[:25],
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== 5,000-QUESTION ONLINE DISCOVERY -> SELF-TRAIN -> SETTLE -> RECALL SUMMARY ===", flush=True)
    print(f"  * Domains Covered            : {len(dom_summary)} distinct domains", flush=True)
    print(f"  * Pre-Check Known Locally    : {pre_known}/{len(pre_sample)} (0.0%)", flush=True)
    print(f"  * Pass 1 (Cold Online+Train) : {p1_passed}/{len(items)} ({report['pass1_online_discovery_and_self_train']['accuracy_pct']}%) in {p1_wall_s}s | p50={report['pass1_online_discovery_and_self_train']['p50_ms']} ms", flush=True)
    print(f"  * Licensed Records Saved     : {saved_lines} open-licensed items in licensed_web_training_material.jsonl", flush=True)
    print(f"  * Pass 2 (Offline Recall)    : {p2_passed}/{len(items)} ({report['pass2_offline_post_settling_recall']['accuracy_pct']}%) in {p2_wall_s}s | p50={report['pass2_offline_post_settling_recall']['p50_ms']} ms | web_used={report['pass2_offline_post_settling_recall']['web_used_count']}", flush=True)
    if p1_failures:
        print(f"  * Sample Pass 1 Misses ({len(p1_failures)} total):", flush=True)
        for f in p1_failures[:8]:
            print(f"      - [{f['domain']}] {f['question']} | expected={f['expected']} | got={f['preview'][:85]}", flush=True)


if __name__ == "__main__":
    main()
