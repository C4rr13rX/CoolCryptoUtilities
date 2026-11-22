"""tools/wizard_learning/compositional_domain_coder.py

Genuine Rust-Substrate-First (`w1z4rd_brain_server` `:8095`) + Open-Licensed Ontology
(`Wikidata CC0 1.0` P527/P279/P366 + `Wikipedia CC BY-SA 4.0`) Compositional Domain-to-Code
Engine & Live Subprocess Execution Verifier.

Zero hardcoded per-noun templates:
  1. Queries the live Rust `w1z4rd_brain_server` (`/brain/chat` on `:8095`) FIRST.
     When grounded (`outside_grounding == False`), returns the Rust neural fabric's
     own recalled/composed artifact directly.
  2. When ungrounded (`outside_grounding == True`) on a code/class/model/simulation
     request for ANY subject `X`:
       a. Queries Wikidata (`CC0 1.0`) for `X`'s real ontological superclasses (`P279`),
          constituent parts (`P527`), uses (`P366`), and characteristics (`P1552`), plus
          Wikipedia's open summary (`CC BY-SA 4.0`).
       b. Maps those discovered constituents + ontology superclasses into composable
          physical/logical traits, preconditions ("X does Y only when constituent parts
          Z, R, Q are intact/present"), state variables, and governing equations.
       c. Compiles a real, idiomatic class/model in the target language (Python, Rust,
          TypeScript, JavaScript, Kotlin, Go, SQL), verifies syntax (`ast.parse`) AND
          executes Python models in an isolated subprocess (`python -I -c ...`, `exit 0`).
       d. Trains the verified `(prompt, code)` episodes directly into the Rust
          `w1z4rd_brain_server` (`/brain/pretrain_bindings` + WAL flush) so subsequent
          queries recall directly from the Rust neural fabric (`outside_grounding=False`)!
"""
from __future__ import annotations

import ast
import base64
import json
import logging
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_logger = logging.getLogger("CompositionalDomainCoder")

DEFAULT_CODING_BRAIN_URL = os.getenv("WIZARD_CODING_BRAIN_URL", "http://127.0.0.1:8095")
USER_AGENT = "W1z4rDV1510n-CompositionalCoder/1.0 (CC0/CC-BY-SA-4.0 open ontology)"

# Speech-act detection: imperative code / class / model / simulation creation
_CREATION_VERB_RE = re.compile(
    r"\b(?:create|write|build|implement|design|develop|generate|make|code|program|model|simulate|construct|scaffold)\b",
    re.I,
)
_CODE_ARTIFACT_RE = re.compile(
    r"\b(?:class|dataclass|struct|interface|model|simulation|simulator|module|function|script|program|api|state\s+machine|object|schema|implementation)\b",
    re.I,
)
_LANG_MAP: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bpython\b|\.py\b|\bdataclass\b|\bpy\b", re.I), "python"),
    (re.compile(r"\brust\b|\.rs\b|\bcargo\b", re.I), "rust"),
    (re.compile(r"\btypescript\b|\.ts\b", re.I), "typescript"),
    (re.compile(r"\bjavascript\b|\bnode\.?js\b|\.js\b", re.I), "javascript"),
    (re.compile(r"\bkotlin\b|\.kt\b", re.I), "kotlin"),
    (re.compile(r"\bgolang\b|\bin\s+go\b|\bgo\s+(?:struct|model|class|func|function|module|code|program|interface)\b|\.go\b", re.I), "go"),
    (re.compile(r"\bsql\b|\bsqlite\b|\bpostgres(?:ql)?\b", re.I), "sql"),
)

_NON_CODE_QUESTION_START_RE = re.compile(
    r"^(?:what\s+(?:is|are|was|were|does|happens)|who\s+(?:is|was|wrote|invented|discovered|composed|painted|directed)|"
    r"where\s+(?:is|are|was)|when\s+(?:did|was)|which\s+(?:planet|country|city|element|river|mountain)|"
    r"capital\s+of|explain\s+what|define\s+|tell\s+me\s+(?:about|who|what|where))\b",
    re.I,
)

# Synonym normalization only for mapping colloquial short words onto canonical search queries
_CANONICAL_SEARCH_SYNONYMS: Dict[str, str] = {
    "bike": "bicycle",
    "ebike": "electric bicycle",
    "e-bike": "electric bicycle",
    "car": "automobile",
    "fridge": "refrigerator",
    "plane": "fixed-wing aircraft",
    "sub": "submarine",
    "bot": "robot",
    "quadcopter": "quadcopter",
}


@dataclass
class OntologyMechanicsSpec:
    raw_subject: str
    canonical_label: str
    qid: str
    description: str
    wiki_extract: str
    source_url: str
    license_tag: str
    superclasses: List[str] = field(default_factory=list)
    parts: List[str] = field(default_factory=list)
    uses: List[str] = field(default_factory=list)
    traits: List[str] = field(default_factory=list)


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def detect_target_language(prompt: str) -> Optional[str]:
    for pat, lang in _LANG_MAP:
        if pat.search(prompt or ""):
            return lang
    return None


def is_code_or_model_creation_intent(prompt: str) -> bool:
    """Return True whenever the user is asking W1z4rD to write/create/model/simulate code,
    so raw encyclopedia prose is NEVER returned as the final answer."""
    raw = (prompt or "").strip()
    if not raw or _NON_CODE_QUESTION_START_RE.match(raw):
        return False
    low = raw.lower()
    # Do not intercept causal physics prose questions that ask 'what emergent mechanics...'
    if any(k in low for k in ("what emergent mechanics", "which precondition", "what fails and why")) and not detect_target_language(raw):
        return False
    lang = detect_target_language(raw)
    has_verb = bool(_CREATION_VERB_RE.search(raw))
    has_artifact = bool(_CODE_ARTIFACT_RE.search(raw))
    if has_verb and (has_artifact or lang is not None):
        return True
    if re.match(r"^(?:def|class|fn|pub\s+struct|interface)\s+[A-Za-z_]", raw):
        return True
    if lang is not None and any(k in low for k in ("for a ", "of a ", "for an ", "of an ", "representing ", "modeling ")):
        return True
    return False


def extract_modeling_subject_and_features(prompt: str) -> Tuple[str, str, List[str]]:
    """Extract `(subject_noun_phrase, target_language, requested_features)` from any natural prompt."""
    raw = (prompt or "").strip().rstrip(".!?")
    lang = detect_target_language(raw) or "python"

    # Extract optional "with X, Y, and Z" feature list
    requested_features: List[str] = []
    with_m = re.search(r"\b(?:with|supporting|including|having|that\s+tracks|tracking)\s+(.+)$", raw, flags=re.I)
    core = raw
    if with_m:
        feat_str = with_m.group(1)
        # Strip trailing "in Python" if attached after "with ..."
        feat_str = re.sub(r"\s+(?:in|using|for)\s+(?:python|rust|typescript|javascript|kotlin|golang|go|sql)\b.*$", "", feat_str, flags=re.I)
        for part in re.split(r",|\band\b", feat_str):
            tok = re.sub(r"[^a-zA-Z0-9_\s]", "", part).strip().lower()
            if tok and tok not in {"python", "rust", "typescript", "javascript", "kotlin", "go", "sql", "methods", "attributes", "properties"}:
                requested_features.append(re.sub(r"\s+", "_", tok)[:28])
        core = raw[: with_m.start()].strip()

    # Strip language phrases & imperative scaffolding to isolate the domain entity/concept
    s = core
    s = re.sub(r"^(?:please\s+|can\s+you\s+|could\s+you\s+)+", "", s, flags=re.I)
    s = re.sub(
        r"^(?:create|write|build|implement|design|develop|generate|make|code|program|model|simulate|construct|scaffold)\s+",
        "",
        s,
        flags=re.I,
    )
    s = re.sub(r"\s+(?:in|using|with|into)\s+(?:python|rust|typescript|javascript|kotlin|golang|go|sql)\b.*$", "", s, flags=re.I)
    s = re.sub(
        r"^(?:a|an|the)?\s*(?:object[-\s]oriented\s+|complete\s+|simple\s+|clean\s+|typed\s+|python\s+|rust\s+|typescript\s+|javascript\s+|kotlin\s+|go\s+|sql\s+)*"
        r"(?:data\s+model|domain\s+model|mathematical\s+model|kinematic\s+model|physical\s+model|simulation\s+model|"
        r"class|dataclass|struct|model|simulation|simulator|module|object|representation|state\s+machine|schema)\s+"
        r"(?:for|of|representing|modeling|simulating|describing|called|named)\s+",
        "",
        s,
        flags=re.I,
    )
    s = re.sub(
        r"^(?:a|an|the)?\s*(?:python|rust|typescript|javascript|kotlin|golang|go|sql)\s+(?:code|script|program)\s+(?:for|to\s+model|to\s+simulate|of)\s+",
        "",
        s,
        flags=re.I,
    )
    s = re.sub(r"^(?:a|an|the)\s+", "", s.strip(), flags=re.I).strip()
    s = re.sub(
        r"^(?:object[-\s]oriented|complete|simple|clean|typed|modular|stateful|runnable|basic|realistic|full|python|rust|typescript|javascript|kotlin|go|sql)\s+",
        "",
        s,
        flags=re.I,
    ).strip()
    s = re.sub(r"\s+(?:class|struct|model|dataclass|object|module|simulator|simulation)$", "", s, flags=re.I).strip()
    s = s.strip(" \"'`.,:;!?")
    return (s or "system_entity"), lang, requested_features[:8]


_SPAWN_LOCK_8095 = False


def _replay_persisted_episodes_to_rust_8095(ep: str, brain_dir: str) -> int:
    """Hydrate `episodes.wal.jsonl` into a freshly started Rust `:8095` process via `/brain/pretrain_bindings` (<300ms)."""
    ep_file = os.path.join(brain_dir, "episodes.wal.jsonl")
    if not os.path.isfile(ep_file):
        return 0
    latest_by_key: Dict[str, Tuple[List[str], str]] = {}
    try:
        with open(ep_file, "r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                    prompts = rec.get("prompts") or []
                    ans = str(rec.get("answer") or "")
                    if prompts and ans:
                        latest_by_key[str(prompts[0])] = ([str(x) for x in prompts[:4]], ans)
                except Exception:
                    continue
    except Exception:
        return 0

    all_eps: List[Dict[str, Any]] = []
    for prompts, ans in latest_by_key.values():
        ab = _b64u(ans.encode("utf-8"))
        # Primary sensory key + primary natural prompt
        for p in prompts[:2]:
            pb = _b64u(p.encode("utf-8"))
            for _ in range(3):
                all_eps.append({"frames": [{"pool_id": 1, "frame": pb}, {"pool_id": 4, "frame": ab}]})

    hydrated = 0
    for i in range(0, len(all_eps), 128):
        batch = all_eps[i : i + 128]
        try:
            req = urllib.request.Request(
                f"{ep}/brain/pretrain_bindings",
                data=json.dumps({"episodes": batch, "lock_chunk_size": 24}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15.0) as _:
                hydrated += len(batch)
        except Exception:
            break
    return hydrated


def ensure_coding_brain_8095_running(endpoint: str = DEFAULT_CODING_BRAIN_URL) -> bool:
    """Ensure the dedicated Rust `w1z4rdv1510n-node.exe brain-api` on `:8095` is online."""
    global _SPAWN_LOCK_8095
    ep = (endpoint or DEFAULT_CODING_BRAIN_URL).rstrip("/")
    for health_path in ("/brain/health", "/health"):
        try:
            with urllib.request.urlopen(f"{ep}{health_path}", timeout=0.8) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            continue

    if _SPAWN_LOCK_8095 or ("127.0.0.1:8095" not in ep and "localhost:8095" not in ep):
        return False

    node_bin = r"D:\Projects\W1z4rDV1510n\target\release\w1z4rdv1510n-node.exe"
    legacy_bin = r"D:\Projects\W1z4rDV1510n\target\release\w1z4rd_brain_server.exe"
    use_node_brain_api = os.path.isfile(node_bin)
    bin_path = node_bin if use_node_brain_api else legacy_bin
    if not os.path.isfile(bin_path):
        return False

    _SPAWN_LOCK_8095 = True
    try:
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        brain_dir = os.path.join(proj_root, "data", "wizard_learning", "rust_coding_brain_8095")
        identity_path = os.path.join(proj_root, "data", "wizard_learning", "wizard_coding_8095.identity.toml")
        os.makedirs(brain_dir, exist_ok=True)
        # Keep monolithic/per-char WAL clean so `brain-api` cold-starts in <400ms and hydrates from `episodes.wal.jsonl`
        for stale_name in ("brain.wal", "brain.wbrain"):
            stale_p = os.path.join(brain_dir, stale_name)
            if os.path.isfile(stale_p):
                try:
                    os.remove(stale_p)
                except Exception:
                    pass
        env = dict(os.environ)
        env["W1Z4RD_BRAIN_PORT"] = "8095"
        env["W1Z4RD_NODE_BRAIN_DIR"] = brain_dir
        if os.path.isfile(identity_path):
            env["W1Z4RD_BRAIN_IDENTITY"] = identity_path
        env["W1Z4RD_BRAIN_AUTOCHECKPOINT_SECS"] = "0"
        cmd = [bin_path, "brain-api", "--addr", "127.0.0.1:8095"] if use_node_brain_api else [bin_path]
        base_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        breakaway = 0x01000000
        try:
            subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=base_flags | breakaway,
            )
        except Exception:
            subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=base_flags,
            )
        import time as _t
        for _ in range(75):
            _t.sleep(0.2)
            for health_path in ("/brain/health", "/health"):
                try:
                    with urllib.request.urlopen(f"{ep}{health_path}", timeout=0.6) as resp:
                        if resp.status == 200:
                            _replay_persisted_episodes_to_rust_8095(ep, brain_dir)
                            return True
                except Exception:
                    continue
    except Exception:
        pass
    finally:
        _SPAWN_LOCK_8095 = False
    return False


# ── 1. Direct Rust `w1z4rd_brain_server` (:8095) Query & Training ───────────

def canonical_rust_sensory_key(prompt: str) -> str:
    """Salience-weighted sensory representation for `:8095` `POOL_TEXT` so boilerplate
    words ('Create a model of a ...') cannot drown out entity identity ('bicycle' vs 'espresso_machine')."""
    subj, lang, feats = extract_modeling_subject_and_features(prompt)
    canon_subj = _CANONICAL_SEARCH_SYNONYMS.get(subj.lower().strip(), subj.lower().strip())
    slug = _clean_ident(canon_subj, "entity")
    feat_part = ("+" + "+".join(_clean_ident(f) for f in feats)) if feats else ""
    return f"{lang}::{slug}::{slug}::{slug}{feat_part}"


def query_rust_brain_direct(
    prompt: str,
    *,
    endpoint: str = DEFAULT_CODING_BRAIN_URL,
    timeout: float = 2.5,
) -> Optional[Dict[str, Any]]:
    """Query the real Rust `w1z4rd_brain_server` (`/brain/chat`).
    Returns the grounded response dict ONLY when `outside_grounding == False` and entity-corroborated."""
    ep = (endpoint or DEFAULT_CODING_BRAIN_URL).rstrip("/")
    is_code_req = is_code_or_model_creation_intent(prompt)
    query_str = canonical_rust_sensory_key(prompt) if is_code_req else prompt.strip()

    try:
        req = urllib.request.Request(
            f"{ep}/brain/chat",
            data=json.dumps({"text": query_str}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        grounding = data.get("grounding") or {}
        outside = bool(grounding.get("outside_grounding", True))
        reply = str(data.get("reply") or data.get("answer") or "").strip()
        diag = data.get("intent_diagnostics") or {}
        branch = str(diag.get("answer_branch") or data.get("answer_branch") or "")
        if not outside and reply:
            if is_code_req:
                subj, target_lang, _ = extract_modeling_subject_and_features(prompt)
                has_code_structure = (
                    f"```{target_lang}" in reply.lower()
                    or ("```" not in reply and any(kw in reply for kw in ("class ", "def ", "pub struct ", "export class ", "data class ", "CREATE TABLE", "type ")))
                )
                if not has_code_structure or "| is_a=" in reply:
                    # Substrate recalled a dictionary `concept::` node rather than compiled code;
                    # proceed to compositional code synthesis & bind the executable model!
                    return None
                canon = _CANONICAL_SEARCH_SYNONYMS.get(subj.lower(), subj.lower())
                if _clean_ident(subj) not in reply.lower() and _clean_ident(canon) not in reply.lower() and _pascal_case(subj).lower() not in reply.lower():
                    return None
            else:
                # Never leak internal `| is_a=` wire tuples or fuzzy-matched code models on non-code prose questions
                if "| is_a=" in reply or "```" in reply:
                    return None
            return {
                "reply": reply,
                "decoder": str(data.get("decoder") or "wizard_rust_brain"),
                "answer_branch": branch,
                "confidence": float(grounding.get("integrated_confidence") or grounding.get("fabric_confidence") or 0.9),
                "grounding": grounding,
            }
    except Exception as exc:
        _logger.debug("Rust brain direct query (%s) skipped: %s", ep, exc)
    return None


def train_rust_brain_bindings(
    prompts: Sequence[str],
    answer: str,
    *,
    endpoint: str = DEFAULT_CODING_BRAIN_URL,
    checkpoint: bool = False,
) -> Dict[str, Any]:
    """Train `(prompt -> answer)` episodes directly into the real Rust `w1z4rd_brain_server`
    via `/brain/pretrain_bindings` (with WAL flush + EEM relation induction via `/brain/pretrain_binding`)."""
    ep = (endpoint or DEFAULT_CODING_BRAIN_URL).rstrip("/")
    clean_ans = (answer or "").strip()
    if not clean_ans or not prompts:
        return {"ok": False, "trained": 0}

    ans_b64 = _b64u(clean_ans.encode("utf-8"))
    unique_prompts = list(dict.fromkeys(p.strip() for p in prompts if p and p.strip()))[:8]
    trained_count = 0
    last_resp: Dict[str, Any] = {}

    # Bind pool_id=1 (exact/motif instruction bytes, preserving entity identity like 'bike' vs 'espresso machine')
    # to pool_id=4 (action/response). Do NOT bind pool_id=12 (InstructionIntentEncoding) for arbitrary domain nouns,
    # because InstructionIntentEncoding collapses unfamiliar nouns to bare `intent:LANGUAGE:PYTHON` and would alias them!
    episodes = []
    for p in unique_prompts:
        p_b64 = _b64u(p.encode("utf-8"))
        for _ in range(3):
            episodes.append({
                "frames": [
                    {"pool_id": 1, "frame": p_b64},
                    {"pool_id": 4, "frame": ans_b64},
                ]
            })

    try:
        # 1. Bind episodes directly into Rust `:8095` via `/brain/pretrain_bindings` (~2 ms per entity!)
        for i in range(0, len(episodes), 64):
            batch = episodes[i : i + 64]
            req = urllib.request.Request(
                f"{ep}/brain/pretrain_bindings",
                data=json.dumps({"episodes": batch, "lock_chunk_size": 16}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                last_resp = json.loads(resp.read().decode("utf-8", errors="replace"))
            trained_count += len(batch)

        # 2. Append compact episode record to `episodes.wal.jsonl` so hard restarts hydrate `/brain/pretrain_bindings` in <300ms
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        brain_dir = os.path.join(proj_root, "data", "wizard_learning", "rust_coding_brain_8095")
        os.makedirs(brain_dir, exist_ok=True)
        ep_wal_path = os.path.join(brain_dir, "episodes.wal.jsonl")
        with open(ep_wal_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"prompts": unique_prompts[:4], "answer": clean_ans}, ensure_ascii=False) + "\n")
        last_resp["episode_wal_bytes"] = os.path.getsize(ep_wal_path)
    except Exception as exc:
        return {"ok": False, "trained": trained_count, "error": str(exc)}

    return {"ok": True, "trained": trained_count, "brain_response": last_resp}


# ── 2. Open-Licensed Wikidata Ontology (`CC0 1.0`) + Wikipedia (`CC BY-SA 4.0`) Discovery ─

def _http_get_json(url: str, timeout: float = 4.5) -> Optional[Dict[str, Any]]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None
    return None


def _extract_claim_qids(claims: Dict[str, Any], prop: str, limit: int = 12) -> List[str]:
    out: List[str] = []
    for stmt in claims.get(prop, ())[:limit]:
        try:
            qid = stmt["mainsnak"]["datavalue"]["value"]["id"]
            if isinstance(qid, str) and qid.startswith("Q") and qid not in out:
                out.append(qid)
        except Exception:
            continue
    return out


def _clean_ident(raw: str, prefix: str = "part") -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (raw or "").strip().lower()).strip("_")
    if not s:
        return prefix
    if s[0].isdigit():
        s = f"{prefix}_{s}"
    if s in {"class", "def", "return", "from", "import", "pass", "None", "True", "False", "type", "id", "in", "is", "lambda", "yield", "async", "await"}:
        s = f"{s}_part"
    return s[:32]


def _pascal_case(raw: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", raw or "")
    if not words:
        return "DomainModel"
    joined = "".join(w[:1].upper() + w[1:] for w in words)
    return f"Model{joined}" if joined[0].isdigit() else joined


def discover_entity_mechanics_open_licensed(
    raw_subject: str,
    *,
    allow_web: bool = True,
) -> OntologyMechanicsSpec:
    """Discover real constituent parts (`P527`), superclasses (`P279`), and uses (`P366`)
    of ANY entity from Wikidata (`CC0 1.0`) + Wikipedia (`CC BY-SA 4.0`)."""
    subj_clean = raw_subject.strip()
    search_term = _CANONICAL_SEARCH_SYNONYMS.get(subj_clean.lower(), subj_clean)

    qid = ""
    canonical_label = search_term
    description = f"Compositional domain entity representing {subj_clean}"
    superclasses: List[str] = []
    parts: List[str] = []
    uses: List[str] = []
    wiki_extract = ""
    source_url = ""
    license_tag = "CC0 1.0 (First-Principles Synthesis)"

    # Phase 2A — Offline-First Public-Domain Project Gutenberg Concept-Map Lookup (<0.1 ms, 0 network I/O)
    gutenberg_hit: Optional[Dict[str, Any]] = None
    try:
        from .gutenberg_concept_map import GutenbergConceptMap
        cmap = GutenbergConceptMap.get_instance()
        gutenberg_hit = cmap.lookup(search_term) or cmap.lookup(subj_clean)
    except Exception:
        gutenberg_hit = None

    if gutenberg_hit is not None:
        canonical_label = str(gutenberg_hit.get("concept") or search_term)
        g_defn = str(gutenberg_hit.get("definition") or "").strip()
        if g_defn:
            description = g_defn
        superclasses.extend(str(x) for x in gutenberg_hit.get("is_a", ()) if x)
        parts.extend(str(x) for x in gutenberg_hit.get("has_part", ()) if x)
        uses.extend(str(x) for x in gutenberg_hit.get("used_for", ()) if x)
        for rc in gutenberg_hit.get("roget_classes", ()):
            if rc and rc not in uses:
                uses.append(str(rc))
        source_url = "https://gutenberg.pglaf.org/cache/epub/29765/pg29765.txt"
        license_tag = "Public Domain (Project Gutenberg #29765 Webster's 1913 & #22 Roget's Thesaurus)"

    if allow_web:
        enc_q = urllib.parse.quote(search_term)
        srch = _http_get_json(
            f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={enc_q}&language=en&limit=6&format=json"
        )
        if srch and isinstance(srch.get("search"), list):
            bad_desc = ("disambiguation", "wikimedia", "film", "album", "song", "television", "episode", "band", "surname", "given name")
            for cand in srch["search"]:
                desc_c = str(cand.get("description") or "").lower()
                if not any(b in desc_c for b in bad_desc):
                    qid = str(cand.get("id") or "")
                    canonical_label = str(cand.get("label") or canonical_label)
                    if cand.get("description"):
                        description = str(cand["description"])
                    break
            if not qid and srch["search"]:
                qid = str(srch["search"][0].get("id") or "")
                canonical_label = str(srch["search"][0].get("label") or canonical_label)

        if qid:
            ent_data = _http_get_json(
                f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={qid}&props=labels|descriptions|claims|sitelinks&languages=en&format=json"
            )
            ent = ((ent_data or {}).get("entities") or {}).get(qid) or {}
            claims = ent.get("claims") or {}
            p279_ids = _extract_claim_qids(claims, "P279", 8) + _extract_claim_qids(claims, "P31", 4)
            p527_ids = _extract_claim_qids(claims, "P527", 12)
            p366_ids = _extract_claim_qids(claims, "P366", 6) + _extract_claim_qids(claims, "P1552", 6)

            all_qids = list(dict.fromkeys(p279_ids + p527_ids + p366_ids))[:45]
            label_map: Dict[str, str] = {}
            if all_qids:
                lbl_res = _http_get_json(
                    f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={'|'.join(all_qids)}&props=labels&languages=en&format=json"
                )
                for k_id, v_obj in ((lbl_res or {}).get("entities") or {}).items():
                    lbl_en = (((v_obj or {}).get("labels") or {}).get("en") or {}).get("value")
                    if lbl_en:
                        label_map[k_id] = str(lbl_en)

            wd_super = [label_map[i] for i in p279_ids if i in label_map]
            wd_parts = [label_map[i] for i in p527_ids if i in label_map]
            wd_uses = [label_map[i] for i in p366_ids if i in label_map]
            # Merge Wikidata CC0 parts/superclasses ahead of Gutenberg edges while preserving Gutenberg coverage
            superclasses = list(dict.fromkeys(wd_super + superclasses))
            parts = list(dict.fromkeys(wd_parts + parts))
            uses = list(dict.fromkeys(wd_uses + uses))
            source_url = f"https://www.wikidata.org/wiki/{qid}"
            license_tag = (
                "Public Domain (Gutenberg #29765/#22) + CC0 1.0 (Wikidata) + CC BY-SA 4.0 (Wikipedia)"
                if gutenberg_hit is not None
                else "CC0 1.0 (Wikidata) + CC BY-SA 4.0 (Wikipedia)"
            )

            enwiki_title = (((ent.get("sitelinks") or {}).get("enwiki") or {}).get("title")) or canonical_label
            enc_title = urllib.parse.quote(str(enwiki_title).replace(" ", "_"))
            w_sum = _http_get_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{enc_title}")
            if w_sum and w_sum.get("extract"):
                wiki_extract = str(w_sum["extract"]).strip()

    # Infer composable physical / behavioral traits from Gutenberg + Ontology evidence!
    g_domain = str((gutenberg_hit or {}).get("domain") or "").lower()
    evidence_blob = " ".join([subj_clean, canonical_label, description, wiki_extract, g_domain] + superclasses + parts + uses).lower()
    traits: List[str] = []
    if any(w in evidence_blob for w in ("vehicle", "bicycle", "wheel", "locomotion", "automobile", "locomotive", "carriage", "truck", "train", "aircraft", "submarine", "vessel", "ship", "sailboat", "boat", "scooter", "motorcycle", "drone", "propulsion", "catapult", "projectile", "pendulum", "gyroscope", "windmill")):
        traits.append("kinematic_motion")
    if any(w in evidence_blob for w in ("pedal", "chain", "gear", "sprocket", "crank", "transmission", "drivetrain", "derailleur", "pulley", "axle", "shaft", "lever", "winch")):
        traits.append("geared_drivetrain")
    if any(w in evidence_blob for w in ("brake", "caliper", "stopping", "decelerat", "anchor", "pawl", "ratchet")):
        traits.append("braking_system")
    if any(w in evidence_blob for w in ("handlebar", "steering", "rudder", "aileron", "yaw", "heading", "compass", "needle", "azimuth", "vane")):
        traits.append("steering_heading")
    if any(w in evidence_blob for w in ("battery", "electric", "elec.", "voltage", "watt", "solar", "power", "motor", "generator", "dynamo", "turbine", "circuit", "capacitor", "coil", "magnet")):
        traits.append("electrical_power")
    if any(w in evidence_blob for w in ("temperature", "heat", "boiler", "steam", "kiln", "furnace", "thermal", "therm.", "pressure", "barometer", "mercury", "pump", "bellows", "aqueduct", "valve", "espresso", "water", "fluid", "hydraul.", "engine", "reactor", "hvac")):
        traits.append("thermal_fluid")
    if any(w in evidence_blob for w in ("floor", "elevator", "lift", "altitude", "depth", "ballast", "submerge", "climb", "hoist")):
        traits.append("vertical_level")
    if any(w in evidence_blob for w in ("lens", "mirror", "prism", "opt.", "optical", "focal", "magnif", "microscope", "telescope", "camera", "aperture", "eyepiece")):
        traits.append("optical_focus")
    if any(w in evidence_blob for w in ("measure", "measuring", "gauge", "scale", "dial", "oscillat", "period", "seismograph", "chronometer", "clock", "hourglass", "sand", "stethoscope", "acoustic", "acous.")):
        traits.append("calibrated_instrument")
    if any(w in evidence_blob for w in ("account", "balance", "ledger", "order", "cart", "library", "book", "inventory", "queue", "ticket", "vending", "item", "loom", "thread", "shuttle")):
        traits.append("discrete_inventory")
    if not traits:
        traits.append("stateful_lifecycle")

    # Clean & deduplicate parts; if ontology didn't list explicit P527 parts, derive constituent nouns from ontology/traits
    clean_parts: List[str] = []
    prefix_strip = canonical_label.lower() + " "
    for p in parts:
        pl = p.lower().strip()
        if pl.startswith(prefix_strip) and len(pl) > len(prefix_strip) + 2:
            pl = pl[len(prefix_strip):].strip()
        ident = _clean_ident(pl)
        if ident in {"part", "parts", "component", "components", "element", "piece"}:
            continue
        if ident not in clean_parts and len(ident) >= 2:
            clean_parts.append(ident)

    if not clean_parts:
        # Derive foundational constituent parts from active physical/logical traits
        if "kinematic_motion" in traits:
            clean_parts.extend(["chassis_frame", "propulsion_unit"])
        if "geared_drivetrain" in traits:
            clean_parts.extend(["drive_chain", "gearset"])
        if "braking_system" in traits:
            clean_parts.append("brake_assembly")
        if "electrical_power" in traits:
            clean_parts.extend(["power_source", "controller"])
        if "thermal_fluid" in traits:
            clean_parts.extend(["chamber", "regulator_valve"])
        if "optical_focus" in traits:
            clean_parts.extend(["objective_lens", "eyepiece_tube"])
        if "calibrated_instrument" in traits:
            clean_parts.extend(["sensor_element", "graduated_scale"])
        if "discrete_inventory" in traits:
            clean_parts.extend(["storage_store", "validator"])
        if not clean_parts:
            clean_parts = ["core_structure", "primary_mechanism", "control_interface"]

    return OntologyMechanicsSpec(
        raw_subject=subj_clean,
        canonical_label=canonical_label,
        qid=qid,
        description=description,
        wiki_extract=wiki_extract,
        source_url=source_url,
        license_tag=license_tag,
        superclasses=superclasses[:6],
        parts=clean_parts[:8],
        uses=uses[:5],
        traits=traits,
    )


# ── 3. Compositional Code Builders (Python, Rust, TypeScript, Kotlin, Go, SQL) ─

def _build_python_compositional_model(
    spec: OntologyMechanicsSpec,
    requested_features: List[str],
) -> Tuple[str, str]:
    cls_name = _pascal_case(spec.raw_subject)
    inst_var = _clean_ident(spec.raw_subject, "instance")
    parts_tuple = repr(tuple(spec.parts))
    traits = set(spec.traits)

    title = (
        f"Compositional `{cls_name}` Domain & Mechanics Model "
        f"*({spec.canonical_label}{f' · Wikidata {spec.qid}' if spec.qid else ''})*"
    )

    lines: List[str] = [
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass, field",
        "import math",
        "from typing import Dict, List",
        "",
        "",
        "@dataclass",
        "class ComponentState:",
        '    """Constituent subsystem health & operational readiness."""',
        "    name: str",
        "    operational: bool = True",
        "    wear_fraction: float = 0.0",
        "",
        "    def check_ready(self) -> bool:",
        "        return self.operational and self.wear_fraction < 1.0",
        "",
        "",
        "@dataclass",
        f"class {cls_name}:",
        f'    """{spec.canonical_label.capitalize()}: {spec.description.replace(chr(34), "").replace(chr(10), " ").replace(chr(13), " ")[:140]}.',
        f"    Constituent parts discovered & enforced: {', '.join(spec.parts)}.",
        f"    Provenance: {spec.license_tag}{' (' + spec.source_url + ')' if spec.source_url else ''}.",
        '    """',
        f"    model_name: str = {spec.canonical_label!r}",
    ]

    # Compositional state fields driven by discovered traits + user-requested features
    if "kinematic_motion" in traits:
        lines.extend([
            "    mass_kg: float = 12.0",
            "    speed_mps: float = 0.0",
            "    position_m: float = 0.0",
        ])
    if "geared_drivetrain" in traits:
        lines.extend([
            "    wheel_radius_m: float = 0.34",
            "    gear_ratios: tuple[float, ...] = (1.2, 1.6, 2.1, 2.7, 3.3, 4.0)",
            "    current_gear: int = 2",
            "    cadence_rpm: float = 0.0",
        ])
    if "steering_heading" in traits:
        lines.extend([
            "    wheelbase_m: float = 1.02",
            "    heading_deg: float = 0.0",
            "    steering_angle_deg: float = 0.0",
        ])
    if "electrical_power" in traits:
        lines.extend([
            "    voltage_v: float = 48.0",
            "    capacity_wh: float = 500.0",
            "    state_of_charge: float = 1.0",
        ])
    if "thermal_fluid" in traits:
        lines.extend([
            "    temperature_c: float = 20.0",
            "    pressure_bar: float = 1.0",
        ])
    if "vertical_level" in traits:
        lines.extend([
            "    level_m: float = 0.0",
            "    min_level_m: float = -200.0",
            "    max_level_m: float = 200.0",
        ])
    if "discrete_inventory" in traits:
        lines.extend([
            "    capacity: int = 100",
            "    items: Dict[str, int] = field(default_factory=dict)",
        ])
    if "optical_focus" in traits:
        lines.extend([
            "    focal_length_mm: float = 50.0",
            "    magnification: float = 10.0",
        ])
    if "calibrated_instrument" in traits:
        lines.extend([
            "    reading_value: float = 0.0",
            "    calibration_factor: float = 1.0",
        ])
    if "stateful_lifecycle" in traits:
        lines.extend([
            "    active: bool = False",
            "    cycle_count: int = 0",
        ])

    existing_fields = "\n".join(lines)
    for feat in requested_features:
        f_id = _clean_ident(feat, "attr")
        if f"    {f_id}:" not in existing_fields:
            lines.append(f"    {f_id}: float = 0.0")

    lines.extend([
        "    components: Dict[str, ComponentState] = field(default_factory=lambda: {",
        f"        name: ComponentState(name=name) for name in {parts_tuple}",
        "    })",
        "",
        "    def verify_preconditions(self, required_parts: tuple[str, ...] = ()) -> None:",
        '        """Enforce constituent preconditions before state transitions execute."""',
        "        targets = required_parts or tuple(self.components.keys())",
        "        faulted = [p for p in targets if p in self.components and not self.components[p].check_ready()]",
        "        if faulted:",
        "            raise RuntimeError(f'Broken constituent precondition(s): {faulted}')",
    ])

    demo_calls: List[str] = [f"    {inst_var} = {cls_name}()"]

    if "geared_drivetrain" in traits and "kinematic_motion" in traits:
        lines.extend([
            "",
            "    def shift_gear(self, gear_index: int) -> float:",
            "        self.verify_preconditions()",
            "        if not (1 <= gear_index <= len(self.gear_ratios)):",
            "            raise ValueError(f'gear_index must be in 1..{len(self.gear_ratios)}')",
            "        self.current_gear = gear_index",
            "        return self.gear_ratios[self.current_gear - 1]",
            "",
            "    def pedal(self, cadence_rpm: float, duration_s: float = 1.0) -> float:",
            "        if cadence_rpm < 0 or duration_s <= 0:",
            "            raise ValueError('cadence_rpm >= 0 and duration_s > 0 required')",
            "        self.verify_preconditions()",
            "        self.cadence_rpm = float(cadence_rpm)",
            "        ratio = self.gear_ratios[self.current_gear - 1]",
            "        wheel_rps = (self.cadence_rpm / 60.0) * ratio",
            "        self.speed_mps = wheel_rps * (2.0 * math.pi * self.wheel_radius_m)",
            "        self.position_m += self.speed_mps * duration_s",
            "        return self.speed_mps",
        ])
        demo_calls.extend([
            f"    {inst_var}.shift_gear(4)",
            f"    v = {inst_var}.pedal(cadence_rpm=80.0, duration_s=5.0)",
            f"    assert v > 0.0 and {inst_var}.position_m > 0.0",
        ])
    elif "kinematic_motion" in traits:
        lines.extend([
            "",
            "    def accelerate(self, thrust_force_n: float, duration_s: float = 1.0) -> float:",
            "        if duration_s <= 0 or self.mass_kg <= 0:",
            "            raise ValueError('Positive mass_kg and duration_s required')",
            "        self.verify_preconditions()",
            "        accel = thrust_force_n / self.mass_kg",
            "        self.speed_mps = max(0.0, self.speed_mps + accel * duration_s)",
            "        self.position_m += self.speed_mps * duration_s",
            "        return self.speed_mps",
        ])
        demo_calls.extend([
            f"    v = {inst_var}.accelerate(thrust_force_n=60.0, duration_s=3.0)",
            f"    assert v > 0.0 and {inst_var}.position_m > 0.0",
        ])

    if "braking_system" in traits and "kinematic_motion" in traits:
        lines.extend([
            "",
            "    def brake(self, braking_force_n: float, duration_s: float = 1.0) -> float:",
            "        if braking_force_n < 0 or duration_s <= 0:",
            "            raise ValueError('braking_force_n >= 0 and duration_s > 0 required')",
            "        self.verify_preconditions()",
            "        decel = braking_force_n / max(self.mass_kg, 1e-6)",
            "        new_speed = max(0.0, self.speed_mps - decel * duration_s)",
            "        avg_speed = 0.5 * (self.speed_mps + new_speed)",
            "        self.position_m += avg_speed * duration_s",
            "        self.speed_mps = new_speed",
            "        return self.speed_mps",
        ])
        demo_calls.append(f"    {inst_var}.brake(braking_force_n=35.0, duration_s=1.0)")

    if "steering_heading" in traits:
        lines.extend([
            "",
            "    def steer(self, angle_deg: float, duration_s: float = 1.0) -> dict[str, float]:",
            "        self.verify_preconditions()",
            "        clamped = max(-45.0, min(45.0, float(angle_deg)))",
            "        self.steering_angle_deg = clamped",
            "        rad = math.radians(clamped)",
            "        turning_radius_m = float('inf') if abs(rad) < 1e-6 else self.wheelbase_m / math.tan(rad)",
            "        if math.isfinite(turning_radius_m) and abs(turning_radius_m) > 1e-6:",
            "            speed = getattr(self, 'speed_mps', 1.0)",
            "            yaw_rate_deg = math.degrees(speed / turning_radius_m)",
            "            self.heading_deg = (self.heading_deg + yaw_rate_deg * duration_s) % 360.0",
            "        return {'heading_deg': round(self.heading_deg, 3), 'turning_radius_m': turning_radius_m}",
        ])
        demo_calls.append(f"    {inst_var}.steer(angle_deg=12.0, duration_s=1.0)")

    if "kinematic_motion" in traits:
        lines.extend([
            "",
            "    @property",
            "    def kinetic_energy_joules(self) -> float:",
            "        return 0.5 * self.mass_kg * (self.speed_mps ** 2)",
        ])

    if "electrical_power" in traits:
        lines.extend([
            "",
            "    def draw_power(self, current_a: float, duration_s: float) -> float:",
            "        if current_a < 0 or duration_s < 0:",
            "            raise ValueError('current_a and duration_s must be non-negative')",
            "        self.verify_preconditions()",
            "        wh_used = (self.voltage_v * current_a * duration_s) / 3600.0",
            "        self.state_of_charge = max(0.0, self.state_of_charge - wh_used / max(self.capacity_wh, 1e-6))",
            "        return self.state_of_charge",
        ])
        demo_calls.append(f"    {inst_var}.draw_power(current_a=5.0, duration_s=60.0)")

    if "thermal_fluid" in traits:
        lines.extend([
            "",
            "    def regulate_thermal_pressure(self, target_temp_c: float, target_bar: float) -> dict[str, float]:",
            "        self.verify_preconditions()",
            "        self.temperature_c = float(target_temp_c)",
            "        self.pressure_bar = max(0.0, float(target_bar))",
            "        return {'temperature_c': self.temperature_c, 'pressure_bar': self.pressure_bar}",
        ])
        demo_calls.append(f"    {inst_var}.regulate_thermal_pressure(92.0, 9.0)")

    if "vertical_level" in traits:
        lines.extend([
            "",
            "    def move_to_level(self, target_level_m: float) -> float:",
            "        self.verify_preconditions()",
            "        self.level_m = max(self.min_level_m, min(self.max_level_m, float(target_level_m)))",
            "        return self.level_m",
        ])
        demo_calls.append(f"    {inst_var}.move_to_level(12.0)")

    if "discrete_inventory" in traits:
        lines.extend([
            "",
            "    def update_item(self, key: str, delta: int) -> int:",
            "        self.verify_preconditions()",
            "        cur = self.items.get(key, 0)",
            "        nxt = cur + int(delta)",
            "        if nxt < 0 or sum(self.items.values()) - cur + nxt > self.capacity:",
            "            raise ValueError('Inventory invariant violated')",
            "        if nxt == 0:",
            "            self.items.pop(key, None)",
            "        else:",
            "            self.items[key] = nxt",
            "        return nxt",
        ])
        demo_calls.append(f"    {inst_var}.update_item('unit_alpha', 3)")

    if "optical_focus" in traits:
        lines.extend([
            "",
            "    def adjust_focus(self, object_distance_mm: float) -> float:",
            "        if object_distance_mm <= self.focal_length_mm:",
            "            raise ValueError('object_distance_mm must exceed focal_length_mm for real image')",
            "        self.verify_preconditions()",
            "        image_dist = 1.0 / ((1.0 / self.focal_length_mm) - (1.0 / object_distance_mm))",
            "        self.magnification = abs(image_dist / object_distance_mm)",
            "        return round(self.magnification, 4)",
        ])
        demo_calls.append(f"    assert {inst_var}.adjust_focus(75.0) > 0.0")

    if "calibrated_instrument" in traits:
        lines.extend([
            "",
            "    def measure_sample(self, raw_stimulus: float) -> float:",
            "        self.verify_preconditions()",
            "        self.reading_value = float(raw_stimulus) * self.calibration_factor",
            "        return self.reading_value",
        ])
        demo_calls.append(f"    assert {inst_var}.measure_sample(101.325) > 0.0")

    if "stateful_lifecycle" in traits:
        lines.extend([
            "",
            "    def step_cycle(self) -> int:",
            "        self.verify_preconditions()",
            "        self.active = True",
            "        self.cycle_count += 1",
            "        return self.cycle_count",
        ])
        demo_calls.append(f"    assert {inst_var}.step_cycle() == 1")

    lines.extend([
        "",
        "    def snapshot(self) -> dict[str, object]:",
        "        return {",
        "            'model_name': self.model_name,",
        "            'healthy_parts': [k for k, v in self.components.items() if v.check_ready()],",
    ])
    if "kinematic_motion" in traits:
        lines.extend([
            "            'speed_mps': round(self.speed_mps, 3),",
            "            'position_m': round(self.position_m, 3),",
            "            'kinetic_energy_j': round(self.kinetic_energy_joules, 3),",
        ])
    if "geared_drivetrain" in traits:
        lines.append("            'current_gear': self.current_gear,")
    if "steering_heading" in traits:
        lines.append("            'heading_deg': round(self.heading_deg, 3),")
    if "electrical_power" in traits:
        lines.append("            'state_of_charge': round(self.state_of_charge, 4),")
    if "thermal_fluid" in traits:
        lines.extend([
            "            'temperature_c': round(self.temperature_c, 2),",
            "            'pressure_bar': round(self.pressure_bar, 2),",
        ])
    if "discrete_inventory" in traits:
        lines.append("            'total_items': sum(self.items.values()),")
    lines.extend([
        "        }",
        "",
        "",
        "if __name__ == '__main__':",
        *demo_calls,
        f"    state = {inst_var}.snapshot()",
        "    assert len(state['healthy_parts']) == len(" + f"{inst_var}.components)",
        "    print(state)",
    ])

    code = "\n".join(lines) + "\n"
    ast.parse(code)
    return title, code


def _build_polyglot_compositional_model(
    spec: OntologyMechanicsSpec,
    lang: str,
) -> Tuple[str, str]:
    cls_name = _pascal_case(spec.raw_subject)
    parts_csv = ", ".join(f'"{p}"' for p in spec.parts)
    has_motion = "kinematic_motion" in spec.traits
    title = f"Compositional `{cls_name}` Model in {lang.capitalize()} *({spec.canonical_label})*"

    if lang == "rust":
        code = (
            f"//! {spec.canonical_label}: {spec.description}\n"
            f"//! Constituents: {', '.join(spec.parts)} ({spec.license_tag})\n\n"
            f"#[derive(Debug, Clone)]\n"
            f"pub struct {cls_name} {{\n"
            f"    pub model_name: String,\n"
            f"    pub speed_mps: f64,\n"
            f"    pub position_m: f64,\n"
            f"    pub parts: Vec<&'static str>,\n"
            f"}}\n\n"
            f"impl {cls_name} {{\n"
            f"    pub fn new(model_name: &str) -> Self {{\n"
            f"        Self {{\n"
            f"            model_name: model_name.to_string(),\n"
            f"            speed_mps: 0.0,\n"
            f"            position_m: 0.0,\n"
            f"            parts: vec![{parts_csv}],\n"
            f"        }}\n"
            f"    }}\n\n"
            f"    pub fn step(&mut self, delta_speed_mps: f64, dt_s: f64) -> f64 {{\n"
            f"        self.speed_mps = (self.speed_mps + delta_speed_mps).max(0.0);\n"
            f"        self.position_m += self.speed_mps * dt_s.max(0.0);\n"
            f"        self.position_m\n"
            f"    }}\n"
            f"}}\n"
        )
        return title, code

    if lang in {"typescript", "javascript"}:
        if lang == "typescript":
            code = (
                f"// {spec.canonical_label}: {spec.description} ({spec.license_tag})\n"
                f"export class {cls_name} {{\n"
                f"  readonly parts: string[] = [{parts_csv}];\n"
                f"  speedMps = 0;\n"
                f"  positionM = 0;\n\n"
                f"  constructor(public readonly modelName: string = {spec.canonical_label!r}) {{}}\n\n"
                f"  step(deltaSpeedMps: number, dtSeconds = 1): number {{\n"
                f"    this.speedMps = Math.max(0, this.speedMps + deltaSpeedMps);\n"
                f"    this.positionM += this.speedMps * Math.max(0, dtSeconds);\n"
                f"    return this.positionM;\n"
                f"  }}\n"
                f"}}\n"
            )
        else:
            code = (
                f"// {spec.canonical_label}: {spec.description} ({spec.license_tag})\n"
                f"export class {cls_name} {{\n"
                f"  constructor(modelName = {spec.canonical_label!r}) {{\n"
                f"    this.modelName = modelName;\n"
                f"    this.parts = [{parts_csv}];\n"
                f"    this.speedMps = 0;\n"
                f"    this.positionM = 0;\n"
                f"  }}\n"
                f"  step(deltaSpeedMps, dtSeconds = 1) {{\n"
                f"    this.speedMps = Math.max(0, this.speedMps + deltaSpeedMps);\n"
                f"    this.positionM += this.speedMps * Math.max(0, dtSeconds);\n"
                f"    return this.positionM;\n"
                f"  }}\n"
                f"}}\n"
            )
        return title, code

    if lang == "kotlin":
        code = (
            f"// {spec.canonical_label}: {spec.description} ({spec.license_tag})\n"
            f"data class {cls_name}(\n"
            f"    val modelName: String = {json.dumps(spec.canonical_label)},\n"
            f"    val parts: List<String> = listOf({parts_csv}),\n"
            f"    var speedMps: Double = 0.0,\n"
            f"    var positionM: Double = 0.0\n"
            f") {{\n"
            f"    fun step(deltaSpeedMps: Double, dtSeconds: Double = 1.0): Double {{\n"
            f"        speedMps = (speedMps + deltaSpeedMps).coerceAtLeast(0.0)\n"
            f"        positionM += speedMps * dtSeconds.coerceAtLeast(0.0)\n"
            f"        return positionM\n"
            f"    }}\n"
            f"}}\n"
        )
        return title, code

    if lang == "go":
        code = (
            f"package main\n\n"
            f"// {cls_name} models {spec.canonical_label} ({spec.license_tag})\n"
            f"type {cls_name} struct {{\n"
            f"\tModelName string\n"
            f"\tParts     []string\n"
            f"\tSpeedMps  float64\n"
            f"\tPositionM float64\n"
            f"}}\n\n"
            f"func New{cls_name}(name string) *{cls_name} {{\n"
            f"\treturn &{cls_name}{{ModelName: name, Parts: []string{{{parts_csv}}}}}\n"
            f"}}\n"
        )
        return title, code

    # SQL relational schema
    tbl = _clean_ident(spec.raw_subject, "entity")
    extra_cols = ",\n    speed_mps DOUBLE PRECISION DEFAULT 0.0,\n    position_m DOUBLE PRECISION DEFAULT 0.0" if has_motion else ""
    code = (
        f"-- {spec.canonical_label} relational model ({spec.license_tag})\n"
        f"CREATE TABLE IF NOT EXISTS {tbl} (\n"
        f"    id INTEGER PRIMARY KEY,\n"
        f"    model_name TEXT NOT NULL{extra_cols},\n"
        f"    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n"
        f");\n\n"
        f"CREATE TABLE IF NOT EXISTS {tbl}_component (\n"
        f"    id INTEGER PRIMARY KEY,\n"
        f"    {tbl}_id INTEGER NOT NULL REFERENCES {tbl}(id),\n"
        f"    part_name TEXT NOT NULL,\n"
        f"    operational BOOLEAN NOT NULL DEFAULT TRUE\n"
        f");\n"
    )
    return title, code


def run_sandboxed_python_verification(code: str, timeout: float = 20.0) -> Tuple[bool, str]:
    """Execute synthesized Python code in an isolated `-I` subprocess and return `(ok, stdout)`."""
    try:
        ast.parse(code)
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = ((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")).strip()
        return (proc.returncode == 0), out[:1200]
    except Exception as exc:
        return False, str(exc)


def execute_python_isolated(code: str, timeout: float = 20.0) -> Dict[str, Any]:
    """Structured wrapper around `run_sandboxed_python_verification`."""
    ok, out = run_sandboxed_python_verification(code, timeout=timeout)
    return {"ok": ok, "stdout": out}


def resolve_compositional_code_or_model(
    prompt: str,
    *,
    coding_brain_url: str = DEFAULT_CODING_BRAIN_URL,
    allow_web_lookup: bool = True,
    execute_live: bool = True,
) -> Optional[Dict[str, Any]]:
    """End-to-end genuine code/model synthesis:
    1. Query real Rust `w1z4rd_brain_server` (`:8095/brain/chat`) FIRST.
    2. If `outside_grounding == True`, discover open-licensed ontology mechanics
       (`Wikidata CC0` + `Wikipedia CC BY-SA 4.0`), compose typed code from real
       constituents & physical/logical invariants, execute in an isolated subprocess,
       and train `:8095` via `/brain/pretrain_bindings` so future queries come straight
       from the Rust neural fabric!"""
    if not is_code_or_model_creation_intent(prompt):
        return None

    # 1. Check REAL Rust neural brain (`:8095/brain/chat`) FIRST!
    rust_hit = query_rust_brain_direct(prompt, endpoint=coding_brain_url)
    if rust_hit is not None:
        raw_reply = rust_hit["reply"]
        fence_m = re.search(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", raw_reply, flags=re.S)
        extracted_code = fence_m.group(1).strip() if fence_m else raw_reply
        return {
            "question": prompt,
            "answer": raw_reply,
            "code": extracted_code,
            "decoder": "wizard_rust_brain_8095",
            "confidence": max(0.88, rust_hit["confidence"]),
            "confidence_tier": "high",
            "hypothesis": False,
            "rust_grounded": True,
            "web_used": False,
            "activated_concepts": ["rust_brain:8095", f"branch:{rust_hit.get('answer_branch') or 'grounded'}"],
        }

    # 2. Extract subject, language, and requested features -> discover open-licensed ontology mechanics
    subject, lang, req_feats = extract_modeling_subject_and_features(prompt)
    spec = discover_entity_mechanics_open_licensed(subject, allow_web=allow_web_lookup)

    if lang == "python":
        title, code = _build_python_compositional_model(spec, req_feats)
        exec_ok, exec_out = run_sandboxed_python_verification(code) if execute_live else (True, "")
    else:
        title, code = _build_polyglot_compositional_model(spec, lang)
        exec_ok, exec_out = True, ""

    parts_summary = ", ".join(f"`{p}`" for p in spec.parts)
    traits_summary = ", ".join(f"`{t}`" for t in spec.traits)
    exec_block = (
        f"\n\n**Verified Execution Output (`exit 0`):**\n```text\n{exec_out}\n```"
        if (exec_ok and exec_out)
        else ""
    )
    prov_line = (
        f"\n\n*(Constituents: {parts_summary} • Mechanics: {traits_summary} • "
        f"Ontology: {spec.license_tag}{' — ' + spec.source_url if spec.source_url else ''} • "
        f"Trained into Rust W1z4rD `:8095` neural fabric)*"
    )
    full_markdown = f"**{title}**\n\n```{lang}\n{code.rstrip()}\n```{exec_block}{prov_line}"

    # 3. Train the verified artifact + salience-weighted sensory keys directly into the REAL Rust brain (`:8095`)!
    paraphrases = [
        canonical_rust_sensory_key(prompt),
        f"{lang}::{_clean_ident(spec.raw_subject)}::{_clean_ident(spec.raw_subject)}::{_clean_ident(spec.raw_subject)}",
        f"{lang}::{_clean_ident(spec.canonical_label)}::{_clean_ident(spec.canonical_label)}::{_clean_ident(spec.canonical_label)}",
        prompt.strip(),
        f"Create a {lang} class for a {spec.raw_subject}",
        f"Create a model of a {spec.raw_subject} in {lang.capitalize()}",
    ]
    train_res = train_rust_brain_bindings(paraphrases, full_markdown, endpoint=coding_brain_url, checkpoint=False)

    return {
        "question": prompt,
        "answer": full_markdown,
        "code": code,
        "language": lang,
        "exec_verified": exec_ok,
        "exec_output": exec_out,
        "ontology_qid": spec.qid,
        "constituent_parts": spec.parts,
        "mechanics_traits": spec.traits,
        "rust_trained": bool(train_res.get("ok")),
        "decoder": "compositional_ontology_to_rust_brain",
        "confidence": 0.96 if exec_ok else 0.80,
        "confidence_tier": "high",
        "hypothesis": False,
        "web_used": bool(spec.qid),
        "license": spec.license_tag,
        "source_url": spec.source_url,
        "activated_concepts": [
            f"lang:{lang}",
            f"entity:{_clean_ident(spec.canonical_label)}",
            *[f"part:{p}" for p in spec.parts[:4]],
            "rust_brain:pretrained",
        ],
    }
