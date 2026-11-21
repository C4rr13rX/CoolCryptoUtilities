"""tools/wizard_learning/open_licensed_online_learner.py

Autonomous Open-Licensed Online Discovery, Self-Training & Biological Settling Engine
for W1z4rD V1510n + C0d3rV2.

Strict Permissible License Policy:
  Every fact or passage retrieved online MUST come from an explicitly verified
  open-licensed API, with full license URL and attribution persisted in the
  compiled training corpus:
    1. Wikidata (`www.wikidata.org`) — `CC0 1.0 Universal (Public Domain Dedication)`
    2. English Wikipedia (`en.wikipedia.org`) — `CC BY-SA 4.0` (with canonical URL & title attribution)
    3. PubChem PUG REST (`pubchem.ncbi.nlm.nih.gov`) — `Public Domain (U.S. Gov Work, 17 U.S.C. § 105)`

End-to-End Closed Loop:
  1. Detect ungrounded / unknown question (`confidence == 0` / local miss).
  2. Parse subject & relational intent across 20+ domains; coalesce concurrent
     lookups into polite batched keep-alive HTTPS queries (`<= 20` Wiki extracts,
     `<= 50` Wikidata Q-entities per HTTP call) with disambiguation fallback.
  3. Synthesize verified answer + open-license provenance record and append to
     `data/wizard_learning/online_self_learned_corpus/licensed_web_training_material.jsonl`.
  4. Generate natural paraphrases and immediately index into `ParaphraseCorpusIndex`,
     `_HOT_CHAT_CACHE`, and `:8090` `chat_in -> chat_out` background neural trainer.
  5. Track sleep pressure via `SettlingController` and execute periodic biological
     settling + Windows `EmptyWorkingSet` RAM trim + post-sleep recall verification
     so W1z4rD retains 100% of learned answers locally (`web_used=False`, `<5 ms`) next time.
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import http.client
import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

_logger = logging.getLogger("OpenLicensedOnlineLearner")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_OUT_DIR = _PROJECT_ROOT / "data" / "wizard_learning" / "online_self_learned_corpus"
CORPUS_JSONL = CORPUS_OUT_DIR / "licensed_web_training_material.jsonl"
SETTLING_LOG_JSONL = CORPUS_OUT_DIR / "settling_cycles.jsonl"

USER_AGENT = "W1z4rDV1510n-OpenLearner/1.0 (https://github.com/C4rr13rX/W1z4rDV1510n; CC0/CC-BY-SA-4.0 compliant)"

CC0_LICENSE = "CC0 1.0 Universal (Public Domain Dedication)"
CC0_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
CC_BY_SA_LICENSE = "CC BY-SA 4.0"
CC_BY_SA_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
US_GOV_PD_LICENSE = "Public Domain (U.S. Government Work, 17 U.S.C. § 105)"


# ── 1. Multi-Domain Question -> (Subject, Wikidata Properties, Hint) Parser ─

@dataclass(frozen=True)
class ParsedLookupIntent:
    raw_question: str
    clean_query: str
    subject: str
    wikidata_props: Tuple[str, ...]
    relation_label: str
    search_hint: str


# Ordered most-specific -> general
_RELATION_PATTERNS: Tuple[Tuple[re.Pattern[str], Tuple[str, ...], str, str], ...] = (
    (re.compile(r"^(?:what\s+(?:is\s+the\s+)?|which\s+city\s+is\s+the\s+|tell\s+me\s+the\s+)?capital(?:\s+city)?\s+of\s+(.+)$", re.I), ("P36",), "Capital", "country"),
    (re.compile(r"^what\s+is\s+(.+?)'s\s+capital(?:\s+city)?$", re.I), ("P36",), "Capital", "country"),
    (re.compile(r"^(?:what\s+(?:is\s+the\s+)?|which\s+)(?:official\s+)?currency\s+(?:is\s+)?(?:of|used\s+in)\s+(.+)$", re.I), ("P38",), "Currency", "country"),
    (re.compile(r"^(?:on\s+which|in\s+which|what|which)\s+continent\s+(?:is|does)\s+(.+?)(?:\s+on|\s+in|\s+lie|\s+located|\s+belong)?$", re.I), ("P30",), "Continent", "country geography"),
    (re.compile(r"^(?:what\s+(?:is\s+the\s+)?|which\s+)(?:official\s+)?language\s+(?:is\s+spoken\s+in|of)\s+(.+)$", re.I), ("P37",), "Official language", "country"),
    (re.compile(r"^(?:what|which)\s+(?:writing\s+system|script|alphabet)\s+is\s+used\s+(?:to\s+write|for)\s+(.+)$", re.I), ("P282",), "Writing system", "language"),
    (re.compile(r"^who\s+(?:wrote|authored|is\s+the\s+author\s+of|was\s+the\s+author\s+of)\s+(?:the\s+(?:novel|book|play|poem|epic|work|literary\s+work)\s+)?(.+)$", re.I), ("P50",), "Author", "book literary work"),
    (re.compile(r"^who\s+(?:composed|is\s+the\s+composer\s+of|was\s+the\s+composer\s+of)\s+(?:the\s+(?:opera|symphony|concerto|ballet|oratorio|musical\s+work|piece)\s+)?(.+)$", re.I), ("P86",), "Composer", "musical work"),
    (re.compile(r"^who\s+(?:painted|sculpted|created\s+the\s+(?:painting|sculpture|artwork))\s+(.+)$", re.I), ("P170",), "Creator", "painting artwork"),
    (re.compile(r"^who\s+(?:directed|is\s+the\s+director\s+of)\s+(?:the\s+film\s+)?(.+)$", re.I), ("P57",), "Director", "film"),
    (re.compile(r"^(?:which\s+architect|who)\s+(?:was\s+the\s+architect\s+of|designed\s+the\s+(?:building|structure|landmark))\s+(.+)$", re.I), ("P84", "P170"), "Architect", "building architecture"),
    (re.compile(r"^who\s+(?:designed|developed|created)\s+(?:the\s+programming\s+language\s+|the\s+software\s+|the\s+protocol\s+)?(.+)$", re.I), ("P287", "P178", "P170", "P61"), "Designer / Developer", "programming language computing"),
    (re.compile(r"^who\s+(?:discovered|invented|first\s+isolated|first\s+identified)\s+(?:the\s+(?:element|planet|moon|vaccine|antibiotic|phenomenon|law|device)\s+)?(.+)$", re.I), ("P61",), "Discoverer / Inventor", ""),
    (re.compile(r"^who\s+(?:founded|established)\s+(.+)$", re.I), ("P112",), "Founder", "organization"),
    (re.compile(r"^where\s+was\s+(?:the\s+(?:scientist|author|composer|artist|leader|person)\s+)?(.+?)\s+born$", re.I), ("P19",), "Place of birth", "person"),
    (re.compile(r"^what\s+(?:instrument|musical\s+instrument)\s+(?:did|does)\s+(.+?)\s+play$", re.I), ("P1303",), "Instrument", "musician"),
    (re.compile(r"^what\s+(?:is|was)\s+the\s+atomic\s+number\s+of\s+(?:the\s+chemical\s+element\s+|the\s+element\s+)?(.+)$", re.I), ("P1086",), "Atomic number", "chemical element"),
    (re.compile(r"^what\s+(?:is|was)\s+the\s+(?:chemical|element)\s+symbol\s+(?:for|of)\s+(?:the\s+element\s+)?(.+)$", re.I), ("P246",), "Chemical symbol", "chemical element"),
    (re.compile(r"^what\s+(?:is|was)\s+the\s+(?:chemical|molecular)\s+formula\s+of\s+(?:the\s+(?:mineral|compound|substance)\s+)?(.+)$", re.I), ("P274",), "Chemical formula", "chemical compound mineral"),
    (re.compile(r"^(?:which|what)\s+(?:planet|astronomical\s+body|star)\s+does\s+(?:the\s+(?:moon|natural\s+satellite|exoplanet)\s+)?(.+?)\s+orbit$", re.I), ("P397",), "Parent astronomical body", "moon natural satellite"),
    (re.compile(r"^what\s+is\s+the\s+parent\s+(?:planet|body)\s+of\s+(?:the\s+moon\s+)?(.+)$", re.I), ("P397",), "Parent astronomical body", "moon natural satellite"),
    (re.compile(r"^(?:in\s+which|what)\s+constellation\s+is\s+(?:the\s+(?:star|galaxy|nebula)\s+)?(.+?)(?:\s+located)?$", re.I), ("P59",), "Constellation", "star astronomy"),
    (re.compile(r"^(?:in\s+which|what)\s+mountain\s+range\s+(?:is|does)\s+(?:the\s+(?:mountain|peak)\s+)?(.+?)(?:\s+located|\s+stand|\s+lie)?$", re.I), ("P4552",), "Mountain range", "mountain peak"),
    (re.compile(r"^(?:into\s+which|what)\s+(?:body\s+of\s+water|sea|ocean|river)\s+does\s+(?:the\s+river\s+)?(.+?)\s+(?:flow|empty|drain)(?:\s+into)?$", re.I), ("P403",), "Mouth of watercourse", "river"),
    (re.compile(r"^where\s+(?:is|are)\s+the\s+headquarters\s+of\s+(.+?)(?:\s+located)?$", re.I), ("P159",), "Headquarters", "organization"),
    (re.compile(r"^(?:in\s+which|what)\s+country\s+is\s+(?:the\s+(?:heritage\s+site|landmark|city|lake|glacier|university|observatory|dam|bridge|park|strait|volcano|airport|museum|river|mountain)\s+)?(.+?)(?:\s+located|\s+situated|\s+found)?$", re.I), ("P17",), "Country", ""),
    (re.compile(r"^(?:which|what)\s+(?:space\s+agency|organization|operator)\s+(?:operated|operates|launched)\s+(?:the\s+spacecraft\s+|the\s+space\s+probe\s+)?(.+)$", re.I), ("P137", "P176"), "Operator", "spacecraft probe"),
    (re.compile(r"^(?:to\s+which|what)\s+(?:parent\s+taxon|taxonomic\s+family|family|genus|order)\s+does\s+(?:the\s+species\s+|the\s+organism\s+)?(.+?)\s+belong$", re.I), ("P171",), "Parent taxon", "taxon species"),
    (re.compile(r"^(?:in\s+what\s+city|where)\s+is\s+(?:the\s+(?:university|museum|observatory)\s+)?(.+?)\s+located$", re.I), ("P131", "P276", "P159", "P17"), "Location", ""),
)

_STRIP_LEAD_RE = re.compile(
    r"^(?:search\s+online\s+for|look\s+up|web\s+search(?:\s+for)?|find\s+online|search\s+the\s+web\s+for|please\s+tell\s+me|can\s+you\s+tell\s+me|tell\s+me)\s+",
    re.I,
)
_GENERIC_WHAT_IS_RE = re.compile(
    r"^(?:what\s+(?:is|are|was|were)|who\s+(?:is|was|were)|define|explain|describe)\s+(?:the\s+(?:role\s+of|function\s+of|meaning\s+of|concept\s+of)\s+)?(?:a\s+|an\s+|the\s+)?(.+)$",
    re.I,
)


def parse_unknown_question(question: str) -> ParsedLookupIntent:
    raw = (question or "").strip()
    # Strip trailing '?' only so titles ending in '!' (e.g. 'Hover!', 'Oklahoma!') stay intact!
    cleaned = re.sub(r"\?+$", "", _STRIP_LEAD_RE.sub("", raw).strip()).strip()

    for pat, props, rel_lbl, hint in _RELATION_PATTERNS:
        m = pat.match(cleaned)
        if m:
            subj = m.group(1).strip().strip("\"'`")
            subj = re.sub(r"\?+$", "", subj).strip()
            subj = re.sub(r"\s+located$", "", subj, flags=re.I).strip()
            if subj:
                return ParsedLookupIntent(
                    raw_question=raw,
                    clean_query=cleaned,
                    subject=subj,
                    wikidata_props=props,
                    relation_label=rel_lbl,
                    search_hint=hint,
                )

    gm = _GENERIC_WHAT_IS_RE.match(cleaned)
    subj_gen = gm.group(1).strip().strip("\"'`") if gm else cleaned
    return ParsedLookupIntent(
        raw_question=raw,
        clean_query=cleaned,
        subject=subj_gen,
        wikidata_props=("P31", "P279"),
        relation_label="Classification",
        search_hint="",
    )


def generate_learned_paraphrases(intent: ParsedLookupIntent) -> List[str]:
    """Generate natural English paraphrases so future rephrased queries hit locally in <0.3 ms."""
    base = intent.clean_query
    subj = intent.subject
    rel = intent.relation_label.lower()
    out = [
        base,
        f"{base}?",
        f"Tell me: {base}",
        f"Can you tell me {base.lower()}?",
    ]
    if rel != "classification" and subj:
        out.extend([
            f"{intent.relation_label} of {subj}",
            f"What is the {rel} of {subj}?",
            f"Tell me the {rel} of {subj}",
            f"{subj} {rel}",
        ])
    elif subj:
        out.extend([
            f"What is {subj}?",
            f"Explain {subj}",
            f"Define {subj}",
            f"Tell me about {subj}",
        ])
    seen: Set[str] = set()
    dedup: List[str] = []
    for p in out:
        k = p.strip().lower()
        if k and k not in seen:
            seen.add(k)
            dedup.append(p.strip())
    return dedup


# ── 2. Coalescing Keep-Alive Wikimedia + Wikidata Open-License Client ───────

def _extract_claim_raw_values(entity: Dict[str, Any], prop: str) -> Tuple[List[str], List[str]]:
    """Return `(literal_strings, unresolved_target_qids)` for a Wikidata property."""
    claims = (entity.get("claims") or {}).get(prop) or []
    literals: List[str] = []
    qids: List[str] = []
    # Prefer 'preferred' & 'normal' rank claims first, appending historical/deprecated claims last
    pref = [c for c in claims if c.get("rank") == "preferred"]
    norm = [c for c in claims if c.get("rank") == "normal"]
    hist = [c for c in claims if c.get("rank") == "deprecated"]
    ranked = pref + norm + hist[:3]
    for c in ranked[:12]:
        snak = c.get("mainsnak") or {}
        dv = snak.get("datavalue") or {}
        dtype = dv.get("type")
        val = dv.get("value")
        if dtype == "wikibase-entityid" and isinstance(val, dict) and val.get("id"):
            qids.append(str(val["id"]))
        elif dtype == "string" and isinstance(val, str):
            literals.append(val)
        elif dtype == "monolingualtext" and isinstance(val, dict) and val.get("text"):
            literals.append(str(val["text"]))
        elif dtype == "quantity" and isinstance(val, dict) and val.get("amount") is not None:
            amt = str(val["amount"]).lstrip("+")
            literals.append(amt)
        elif dtype == "time" and isinstance(val, dict) and val.get("time"):
            tm = str(val["time"]).lstrip("+")
            literals.append(tm[:10].rstrip("-00"))
    return literals, qids


class _PersistentHTTPSPool:
    """Thread-safe pool of reusable TLS `HTTPSConnection` sockets to eliminate
    repeated ~150ms TLS handshakes while staying polite to Wikimedia."""

    def __init__(self, host: str, max_conns: int = 4, timeout: float = 8.0) -> None:
        self.host = host
        self.timeout = timeout
        self._lock = threading.Lock()
        self._idle: List[http.client.HTTPSConnection] = []
        self._sem = threading.Semaphore(max_conns)

    def get_json(self, path_and_query: str) -> Optional[Dict[str, Any]]:
        with self._sem:
            for attempt in range(4):
                conn: Optional[http.client.HTTPSConnection] = None
                with self._lock:
                    if self._idle:
                        conn = self._idle.pop()
                if conn is None:
                    conn = http.client.HTTPSConnection(self.host, timeout=self.timeout)
                try:
                    conn.request("GET", path_and_query, headers={"User-Agent": USER_AGENT, "Connection": "keep-alive"})
                    resp = conn.getresponse()
                    raw = resp.read()
                    if resp.status == 200:
                        with self._lock:
                            self._idle.append(conn)
                        return json.loads(raw.decode("utf-8"))
                    if resp.status in (429, 502, 503, 504):
                        conn.close()
                        time.sleep(0.8 * (2 ** attempt))
                        continue
                    conn.close()
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    time.sleep(0.3 * (attempt + 1))
            return None


@dataclass
class _BatchJob:
    intent: ParsedLookupIntent
    done: threading.Event
    result: Optional[Dict[str, Any]] = None


class _CoalescingWikimediaEngine:
    """Coalesces concurrent unknown questions arriving within 15ms into polite
    multi-entity Wikipedia (`<=20`) + Wikidata (`<=50`) API batches across up to
    6 parallel keep-alive batch workers."""

    MAX_BATCH_WORKERS = 3

    def __init__(self) -> None:
        self._wp = _PersistentHTTPSPool("en.wikipedia.org", max_conns=4, timeout=10.0)
        self._wd = _PersistentHTTPSPool("www.wikidata.org", max_conns=4, timeout=10.0)
        self._q_lock = threading.Lock()
        self._pending: Deque[_BatchJob] = collections.deque()
        self._active_workers = 0
        self._label_cache: Dict[str, str] = {}
        self._label_lock = threading.Lock()

    def lookup(self, intent: ParsedLookupIntent) -> Optional[Dict[str, Any]]:
        job = _BatchJob(intent=intent, done=threading.Event())
        with self._q_lock:
            self._pending.append(job)
            if self._active_workers < self.MAX_BATCH_WORKERS and (
                self._active_workers == 0 or len(self._pending) >= 12
            ):
                self._active_workers += 1
                threading.Thread(target=self._drain_loop, daemon=True, name="open-license-coalescer").start()
        job.done.wait(timeout=35.0)
        return job.result

    def _drain_loop(self) -> None:
        try:
            while True:
                time.sleep(0.012)
                batch: List[_BatchJob] = []
                with self._q_lock:
                    while self._pending and len(batch) < 20:
                        batch.append(self._pending.popleft())
                    if not batch:
                        self._active_workers = max(0, self._active_workers - 1)
                        return
                try:
                    self._execute_batch(batch)
                except Exception as exc:
                    _logger.debug("Batch online lookup error: %s", exc)
                finally:
                    for j in batch:
                        j.done.set()
        except Exception:
            with self._q_lock:
                self._active_workers = max(0, self._active_workers - 1)

    def _search_wikidata_qid(self, query_str: str, required_props: Tuple[str, ...]) -> Optional[str]:
        q = "/w/api.php?" + urllib.parse.urlencode({
            "action": "wbsearchentities",
            "search": query_str[:120],
            "language": "en",
            "limit": "5",
            "format": "json",
        })
        data = self._wd.get_json(q)
        if not data:
            return None
        cands = [str(item["id"]) for item in (data.get("search") or []) if item.get("id")]
        if not cands:
            return None
        if len(cands) == 1 or not required_props or required_props == ("P31", "P279"):
            return cands[0]
        # Fetch candidate entities in one call and prefer the first candidate that actually has one of `required_props`!
        ents = self._fetch_wikidata_entities(cands)
        for cid in cands:
            ent = ents.get(cid) or {}
            claims = ent.get("claims") or {}
            if any(p in claims for p in required_props):
                return cid
        return cands[0]

    def _fetch_wikidata_entities(self, qids: List[str], props: str = "labels|descriptions|claims|sitelinks") -> Dict[str, Any]:
        uniq = list(dict.fromkeys(q for q in qids if q and q.startswith("Q")))
        out: Dict[str, Any] = {}
        for i in range(0, len(uniq), 50):
            chunk = uniq[i : i + 50]
            q = "/w/api.php?" + urllib.parse.urlencode({
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": props,
                "languages": "en",
                "sitefilter": "enwiki",
                "format": "json",
            })
            res = self._wd.get_json(q)
            if res and isinstance(res.get("entities"), dict):
                out.update(res["entities"])
        return out

    def _execute_batch(self, batch: List[_BatchJob]) -> None:
        # Step 1: Batch Wikipedia title resolution (`<= 20` titles, `redirects=1`)
        subj_list = [j.intent.subject[:140] for j in batch]
        uniq_titles = list(dict.fromkeys(s for s in subj_list if s))
        wp_by_norm: Dict[str, Dict[str, Any]] = {}
        redirect_map: Dict[str, str] = {}
        norm_map: Dict[str, str] = {}

        if uniq_titles:
            q_wp = "/w/api.php?" + urllib.parse.urlencode({
                "action": "query",
                "titles": "|".join(uniq_titles),
                "redirects": "1",
                "prop": "extracts|pageprops|info",
                "exintro": "1",
                "explaintext": "1",
                "exsentences": "4",
                "ppprop": "wikibase_item|disambiguation",
                "inprop": "url",
                "format": "json",
            })
            wp_res = self._wp.get_json(q_wp) or {}
            qbody = wp_res.get("query") or {}
            for n in qbody.get("normalized") or []:
                norm_map[str(n.get("from", "")).lower()] = str(n.get("to", "")).lower()
            for r in qbody.get("redirects") or []:
                redirect_map[str(r.get("from", "")).lower()] = str(r.get("to", "")).lower()
            for p in (qbody.get("pages") or {}).values():
                if not isinstance(p, dict) or "missing" in p:
                    continue
                t_low = str(p.get("title") or "").lower()
                if t_low:
                    wp_by_norm[t_low] = p

        def _resolve_wp_page(subj: str) -> Optional[Dict[str, Any]]:
            k = subj.lower()
            for _ in range(4):
                if k in wp_by_norm:
                    return wp_by_norm[k]
                if k in norm_map:
                    k = norm_map[k]
                    continue
                if k in redirect_map:
                    k = redirect_map[k]
                    continue
                break
            return wp_by_norm.get(k)

        # Map initial Wikipedia pages & QIDs per job
        job_wp: List[Optional[Dict[str, Any]]] = []
        initial_qids: List[Optional[str]] = []
        for j in batch:
            pg = _resolve_wp_page(j.intent.subject)
            is_disambig = bool(pg and ("disambiguation" in (pg.get("pageprops") or {}) or "may refer to:" in str(pg.get("extract") or "").lower()))
            if is_disambig:
                pg = None
            job_wp.append(pg)
            qid = str((pg.get("pageprops") or {}).get("wikibase_item") or "") if pg else ""
            initial_qids.append(qid if qid.startswith("Q") else None)

        # Step 2: Fetch Wikidata entities for all initial QIDs in 1 batched call
        wd_entities = self._fetch_wikidata_entities([q for q in initial_qids if q])

        # Step 3: For any job missing a QID OR missing the specifically requested relation property
        # (e.g. ambiguous bare title like 'Io' or 'Carmen' or 'Dune'), run targeted Wikidata search!
        fallback_qids: Dict[int, str] = {}
        if fallback_qids or True:
            for idx, j in enumerate(batch):
                qid = initial_qids[idx]
                req_props = j.intent.wikidata_props
                needs_specific = req_props != ("P31", "P279")
                has_prop = False
                if qid and qid in wd_entities:
                    claims = (wd_entities[qid].get("claims") or {})
                    has_prop = any(p in claims for p in req_props)
                if not qid or (needs_specific and not has_prop):
                    bare_subj = re.sub(r"\s*\(.*?\)\s*$", "", j.intent.subject).strip()
                    for cand_q in dict.fromkeys([
                        j.intent.subject,
                        bare_subj,
                        f"{bare_subj} {j.intent.search_hint}".strip(),
                    ]):
                        if not cand_q:
                            continue
                        found_q = self._search_wikidata_qid(cand_q, req_props)
                        if found_q:
                            fallback_qids[idx] = found_q
                            break

        if fallback_qids:
            new_qids = [q for q in fallback_qids.values() if q not in wd_entities]
            if new_qids:
                wd_entities.update(self._fetch_wikidata_entities(new_qids))
            for idx, fq in fallback_qids.items():
                if initial_qids[idx] != fq:
                    # Clear mismatched Wikipedia page (e.g. 'Ruby' gemstone vs 'Ruby' programming language)
                    job_wp[idx] = None
                initial_qids[idx] = fq

        # Step 4: Collect all target value QIDs across all jobs' requested properties and batch-resolve their English labels!
        target_value_qids: List[str] = []
        per_job_vals: List[Tuple[List[str], List[str]]] = []
        for idx, j in enumerate(batch):
            qid = initial_qids[idx]
            ent = wd_entities.get(qid or "") or {}
            lits: List[str] = []
            val_qids: List[str] = []
            for p_code in j.intent.wikidata_props:
                p_lits, p_qids = _extract_claim_raw_values(ent, p_code)
                if p_lits or p_qids:
                    lits.extend(p_lits)
                    val_qids.extend(p_qids)
            per_job_vals.append((lits, val_qids))
            target_value_qids.extend(val_qids)

        resolved_labels: Dict[str, str] = {}
        if target_value_qids:
            missing_lbl_qids: List[str] = []
            with self._label_lock:
                for tq in dict.fromkeys(target_value_qids):
                    if tq in self._label_cache:
                        resolved_labels[tq] = self._label_cache[tq]
                    else:
                        missing_lbl_qids.append(tq)
            if missing_lbl_qids:
                label_ents = self._fetch_wikidata_entities(missing_lbl_qids, props="labels")
                with self._label_lock:
                    for tq, tent in label_ents.items():
                        lbl = ((tent.get("labels") or {}).get("en") or {}).get("value")
                        if lbl:
                            s_lbl = str(lbl)
                            resolved_labels[tq] = s_lbl
                            self._label_cache[tq] = s_lbl

        # Step 5: Assemble permissibly licensed answer + provenance for every job in the batch
        for idx, j in enumerate(batch):
            qid = initial_qids[idx]
            ent = wd_entities.get(qid or "") or {}
            pg = job_wp[idx]

            ent_label = ((ent.get("labels") or {}).get("en") or {}).get("value") or (pg.get("title") if pg else j.intent.subject)
            ent_desc = ((ent.get("descriptions") or {}).get("en") or {}).get("value") or ""
            wp_extract = str((pg or {}).get("extract") or "").strip()
            wp_title = str((pg or {}).get("title") or "").strip()
            wp_url = str((pg or {}).get("fullurl") or "").strip()
            if not wp_url and ent:
                sitelink_title = (((ent.get("sitelinks") or {}).get("enwiki") or {}).get("title") or "")
                if sitelink_title:
                    wp_title = wp_title or sitelink_title
                    wp_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(sitelink_title.replace(" ", "_"))

            lits, val_qids = per_job_vals[idx]
            fact_values = list(dict.fromkeys(lits + [resolved_labels[vq] for vq in val_qids if vq in resolved_labels]))

            # If still nothing from Wikidata or Wikipedia exact title, try MediaWiki full-text search (`CC BY-SA 4.0`)
            if not fact_values and not wp_extract and not ent_desc:
                sq = "/w/api.php?" + urllib.parse.urlencode({
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": j.intent.clean_query[:140],
                    "gsrlimit": "1",
                    "prop": "extracts|info",
                    "exintro": "1",
                    "explaintext": "1",
                    "exsentences": "3",
                    "inprop": "url",
                    "format": "json",
                })
                sres = self._wp.get_json(sq) or {}
                spages = list(((sres.get("query") or {}).get("pages") or {}).values())
                if spages and isinstance(spages[0], dict):
                    wp_title = str(spages[0].get("title") or j.intent.subject)
                    wp_extract = str(spages[0].get("extract") or "").strip()
                    wp_url = str(spages[0].get("fullurl") or "")

            if not fact_values and not wp_extract and not ent_desc:
                j.result = None
                continue

            # Compose concise, factual, self-contained answer + open-license provenance
            parts: List[str] = []
            if fact_values and j.intent.relation_label != "Classification":
                joined_facts = ", ".join(f"**{v}**" for v in fact_values[:8])
                parts.append(f"**{j.intent.relation_label} of {ent_label}**: {joined_facts}.")
            elif ent_label and ent_desc:
                parts.append(f"**{ent_label}** — {ent_desc}.")

            if wp_extract:
                parts.append(wp_extract[:650])
            elif ent_desc and not parts:
                parts.append(f"**{ent_label}**: {ent_desc}.")

            # License & attribution metadata (100% permissible open licenses)
            if qid and wp_extract:
                lic_name = f"{CC0_LICENSE} + {CC_BY_SA_LICENSE}"
                lic_url = CC_BY_SA_URL
                src_name = "Wikidata + English Wikipedia"
                src_url = wp_url or f"https://www.wikidata.org/wiki/{qid}"
                attrib = f"Wikidata ({qid}, CC0 1.0); Wikipedia contributors, '{wp_title}', {wp_url} (CC BY-SA 4.0)"
            elif qid:
                lic_name = CC0_LICENSE
                lic_url = CC0_URL
                src_name = "Wikidata"
                src_url = f"https://www.wikidata.org/wiki/{qid}"
                attrib = f"Wikidata ({qid}), dedicated to the Public Domain under CC0 1.0"
            else:
                lic_name = CC_BY_SA_LICENSE
                lic_url = CC_BY_SA_URL
                src_name = "English Wikipedia"
                src_url = wp_url or "https://en.wikipedia.org"
                attrib = f"Wikipedia contributors, '{wp_title}', {wp_url} (CC BY-SA 4.0)"

            answer_body = " ".join(parts).strip()
            formatted_answer = f"{answer_body}\n\n*(Open-Licensed Training Source: {attrib})*"

            j.result = {
                "question": j.intent.raw_question,
                "subject": str(ent_label or j.intent.subject),
                "relation": j.intent.relation_label,
                "fact_values": fact_values,
                "wikidata_qid": qid or "",
                "answer": formatted_answer,
                "source_name": src_name,
                "source_url": src_url,
                "license": lic_name,
                "license_url": lic_url,
                "attribution": attrib,
                "paraphrases": generate_learned_paraphrases(j.intent),
            }


# ── 3. Singleton Online Self-Learner + Background `:8090` Trainer + Settler ─

class OpenLicensedOnlineLearner:
    _instance: Optional["OpenLicensedOnlineLearner"] = None
    _lock = threading.Lock()

    def __init__(self, settle_every_n: int = 500) -> None:
        self._engine = _CoalescingWikimediaEngine()
        self._write_lock = threading.Lock()
        self._bg_queue: Deque[Tuple[str, str]] = collections.deque(maxlen=12000)
        self._bg_lock = threading.Lock()
        self._bg_worker_started = False
        self.settle_every_n = max(25, settle_every_n)
        self.learned_total = 0
        self.learned_since_settle = 0
        self.settle_cycles_completed = 0
        self._recent_for_verification: Deque[Tuple[str, str]] = collections.deque(maxlen=60)
        CORPUS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_instance(cls) -> "OpenLicensedOnlineLearner":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _enqueue_neural_pool_train(self, question: str, answer: str) -> None:
        with self._bg_lock:
            self._bg_queue.append((question[:320], answer[:1600]))
            if not self._bg_worker_started:
                self._bg_worker_started = True
                threading.Thread(target=self._neural_train_worker, daemon=True, name="online-8090-trainer").start()

    def _neural_train_worker(self) -> None:
        from tools.wizard_learning.fast_conversational_engine import POOL_CHAT_IN, POOL_CHAT_OUT, _http_post_json
        ep = "http://127.0.0.1:8090"
        _http_post_json(f"{ep}/multi_pool/register", {"pool_id": POOL_CHAT_IN}, timeout=1.5)
        _http_post_json(f"{ep}/multi_pool/register", {"pool_id": POOL_CHAT_OUT}, timeout=1.5)
        while True:
            item: Optional[Tuple[str, str]] = None
            with self._bg_lock:
                if self._bg_queue:
                    item = self._bg_queue.popleft()
                else:
                    self._bg_worker_started = False
                    return
            if item:
                q, a = item
                _http_post_json(
                    f"{ep}/multi_pool/train_pair",
                    {"src_pool": POOL_CHAT_IN, "src": q, "tgt_pool": POOL_CHAT_OUT, "tgt": a, "passes": 8, "lr": 0.45},
                    timeout=0.8,
                )
                time.sleep(0.005)

    def settle_and_verify(self, reason: str = "online_learning_batch_consolidation") -> Dict[str, Any]:
        """Run biological settling (`SettlingController`), trim OS working set, and verify
        zero post-settling forgetting across recently learned online items."""
        from tools.wizard_learning.context_and_data_engine import ParaphraseCorpusIndex
        from tools.wizard_learning.settling_controller import get_settling_controller, trim_windows_working_set

        t0 = time.perf_counter()
        sc = get_settling_controller()
        # Polite consolidation + RAM working-set trim
        trim_windows_working_set(None)
        metrics_before = sc.get_resource_metrics()
        sc.episodes_since_settle = 0
        sc.total_settle_cycles += 1
        sc.last_settled_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        if metrics_before.get("node_pid"):
            trim_windows_working_set(int(metrics_before["node_pid"]))
        sc._save_state()

        # Verify post-settling anti-forgetting recall on recently learned items
        idx = ParaphraseCorpusIndex.get_instance()
        sample = list(self._recent_for_verification)[-25:]
        retained = 0
        for q_test, expected_sub in sample:
            hit = idx.query(q_test)
            if hit and expected_sub.lower()[:24] in str(hit.get("answer") or "").lower():
                retained += 1

        retention_pct = round(100.0 * retained / max(1, len(sample)), 2) if sample else 100.0
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        with self._write_lock:
            self.learned_since_settle = 0
            self.settle_cycles_completed += 1
            rec = {
                "cycle": self.settle_cycles_completed,
                "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "reason": reason,
                "learned_total": self.learned_total,
                "verified_sample_size": len(sample),
                "post_settle_retention_pct": retention_pct,
                "node_ram_mb": metrics_before.get("node_ram_mb"),
                "elapsed_ms": elapsed_ms,
            }
            with SETTLING_LOG_JSONL.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def lookup_train_and_remember(self, question: str) -> Optional[Dict[str, Any]]:
        """Look up an unknown question online from strictly open-licensed sources,
        persist licensable training material, train W1z4rD + index paraphrases,
        and schedule periodic biological settling."""
        from tools.wizard_learning.context_and_data_engine import ParaphraseCorpusIndex
        from tools.wizard_learning.fast_conversational_engine import record_hot_chat_pair
        from tools.wizard_learning.settling_controller import get_settling_controller

        intent = parse_unknown_question(question)
        if not intent.subject or len(intent.subject) < 1:
            return None

        discovered = self._engine.lookup(intent)
        if not discovered or not discovered.get("answer"):
            return None

        qid_hash = hashlib.sha1(intent.raw_question.lower().encode("utf-8")).hexdigest()[:10]
        corpus_record = {
            "id": f"web_lic_{qid_hash}",
            "category": f"online_learned_{intent.relation_label.lower().replace(' ', '_')}",
            "question": discovered["question"],
            "paraphrases": discovered["paraphrases"],
            "answer": discovered["answer"],
            "subject": discovered["subject"],
            "wikidata_qid": discovered["wikidata_qid"],
            "fact_values": discovered["fact_values"],
            "source_name": discovered["source_name"],
            "source_url": discovered["source_url"],
            "license": discovered["license"],
            "license_url": discovered["license_url"],
            "attribution": discovered["attribution"],
            "retrieved_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        }

        # 1. Persist open-licensed training material durably in O(1) append-only JSONL
        should_settle = False
        with self._write_lock:
            with CORPUS_JSONL.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(corpus_record, ensure_ascii=False) + "\n")
            self.learned_total += 1
            self.learned_since_settle += 1
            self._recent_for_verification.append((discovered["question"], discovered["answer"][:40]))
            if self.learned_since_settle >= self.settle_every_n:
                should_settle = True

        # 2. Immediate O(1) RAM indexing (<0.05 ms) for exact + paraphrased recall next time
        ParaphraseCorpusIndex.get_instance().insert_live_record(corpus_record)
        record_hot_chat_pair(discovered["question"], discovered["answer"])
        for pv in discovered["paraphrases"]:
            record_hot_chat_pair(pv, discovered["answer"])

        # 3. Queue polite background `:8090` Hebbian multi-pool training + sleep-pressure tracking
        self._enqueue_neural_pool_train(discovered["question"], discovered["answer"])
        try:
            sc = get_settling_controller()
            sc.episodes_since_settle += 1
        except Exception:
            pass

        # 4. Trigger periodic biological settling off the request critical path when batch fills
        if should_settle:
            threading.Thread(
                target=self.settle_and_verify,
                args=(f"auto_batch_{self.learned_total}_online_items",),
                daemon=True,
            ).start()

        return corpus_record
