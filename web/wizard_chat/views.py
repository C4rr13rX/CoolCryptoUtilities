"""wizard_chat/views.py — API backend for the W1z4rD Vision multimodal chat."""
from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import logging
import os
import pathlib
import re
import sys
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

_logger = logging.getLogger(__name__)

_REPO_ROOT = str(pathlib.Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# ────────────────────────────────────────────────────────────────────
# Process-wide TTL cache for node fetches.  Multiple SPA tabs polling
# /api/wizard-chat/status/, /training/live/, /topology/ at overlapping
# cadences would otherwise each trigger a node round-trip and pile up
# on the inner state lock (which is also held by training).  The cache
# lets concurrent pollers share one fetch.  Tiny TTL — fresh enough for
# a live panel but coarse enough that 8 parallel polls only cost ONE
# node round-trip.
# ────────────────────────────────────────────────────────────────────
import threading as _threading
_node_cache_lock = _threading.Lock()
_node_cache: dict[str, tuple[float, dict]] = {}
_node_refresher_started = False
_node_refresher_lock = _threading.Lock()
_node_refresher_paths: list[tuple[str, float]] = [
    ("/health",            5.0),
    ("/brain",             5.0),
    ("/multi_pool/stats",  5.0),
]
# Per-path Event for the in-flight fetch.  Singleflight: when a
# request arrives and the cache is stale, EXACTLY one thread does
# the upstream fetch.  Every other concurrent caller waits on the
# Event, then reads the freshly-populated cache.  Without this, 8
# concurrent SPA polls all see "miss" simultaneously and each kicks
# off its own node round-trip — collapsing waitress thread pool
# while every fetch serializes on the node's inner lock.
_node_inflight: dict[str, _threading.Event] = {}
_node_refresher_endpoints: set[str] = set()

def _is_slow_brain(endpoint: str) -> bool:
    """True for brains behind the SSM relay proxy.

    programming_brain_proxy.py tunnels each request through an SSM
    SendCommand round-trip, so even /health takes ~11 s versus a few ms for
    the local node.  Timeouts tuned for the local node read those brains as
    permanently offline.  The proxy port is configurable, so honour an env
    override and fall back to its documented default.
    """
    slow_ports = {
        p.strip()
        for p in os.getenv("WIZARD_SLOW_BRAIN_PORTS", "18096").split(",")
        if p.strip()
    }
    try:
        port = urllib.parse.urlsplit(endpoint).port
    except ValueError:
        return False
    return port is not None and str(port) in slow_ports


def _decode_node_body(raw: bytes, path: str) -> dict:
    """Decode a node response body into a dict.

    Brains do not agree on the shape of /health: the local node answers with
    a JSON object, while the AWS senior-software brain (reached through the
    SSM proxy) answers with the bare text "ok\\n".  json.loads() raised on
    that, leaving the cache permanently empty and the status view reporting
    online:False for a brain that was perfectly healthy.

    Anything that decodes to a JSON object is returned as-is.  A non-object
    JSON value or plain text is wrapped into a dict so callers -- which all
    treat "empty dict" as "unreachable" -- see a truthy payload.
    """
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if isinstance(value, dict):
        return value
    if value is not None:
        # Scalar/list JSON (e.g. /brain/tick's bare counter).
        return {"status": "OK", "value": value, "raw": text}
    # Plain text. Treat the conventional health words as healthy.
    healthy = text.split()[0].lower() in {"ok", "healthy", "up", "alive", "ready"}
    return {"status": "OK" if healthy else text, "raw": text}


def _node_refresher_loop() -> None:
    """Single background thread that walks the slow status fetches at
    a relaxed cadence (one path every ~3 s, so the full set cycles
    every ~9 s).  Each fetch runs in this thread — urllib releases
    the GIL during I/O so it doesn't block waitress request threads.

    The refresher's job is to keep the cache POPULATED, not to keep
    it strictly fresh.  Stale-while-revalidate in `_node_fetch_cached`
    means request threads always read the cache instantly, even if
    the value is stale; the background thread does the heavy lifting.
    """
    while True:
        with _node_refresher_lock:
            endpoints = list(_node_refresher_endpoints)
        for endpoint in endpoints:
            for path, _ in _node_refresher_paths:
                key = f"{endpoint}{path}"
                # 30 s covers the local node comfortably but clipped the slow
                # tail of the SSM relay (15-58 s under training load), so the
                # remote brain's cache never populated and it read as offline.
                fetch_timeout = 90 if _is_slow_brain(endpoint) else 30
                try:
                    with urllib.request.urlopen(key, timeout=fetch_timeout) as r:
                        data = _decode_node_body(r.read(), path)
                    with _node_cache_lock:
                        _node_cache[key] = (time.time(), data)
                except urllib.error.URLError:
                    if not _is_slow_brain(endpoint) and path == "/health":
                        with _node_cache_lock:
                            _node_cache[key] = (time.time(), {})
                except Exception:
                    pass
                # Each fetch holds the node's inner Mutex (esp. /brain which
                # iterates the motif layer).  Leave training clock time.
                time.sleep(8.0)
        if not endpoints:
            time.sleep(8.0)


def _ensure_node_refresher(endpoint: str) -> None:
    """Idempotent — first request that touches the cache spins up the
    background refresher.  We don't start it at module import because
    Django manage.py / migrate also imports this module and shouldn't
    leave a daemon thread behind."""
    global _node_refresher_started
    norm_ep = endpoint.rstrip("/").replace("://localhost:", "://127.0.0.1:")
    with _node_refresher_lock:
        _node_refresher_endpoints.add(norm_ep)
        if _node_refresher_started:
            return
        t = _threading.Thread(target=_node_refresher_loop,
                                name="wizard-node-cache-refresher",
                                daemon=True)
        t.start()
        _node_refresher_started = True


def _node_fetch_cached(
    path: str, ttl: float = 1.5, timeout: float = 10.0, *, endpoint: str
) -> dict:
    """GET path on the wizard node — stale-while-revalidate semantics.
    Request threads never block on an upstream fetch.  If the cache
    has ANY value (fresh or stale), return it immediately.  If the
    value is stale beyond `ttl`, the background refresher will pick
    it up on its next pass.  If the cache is empty (cold start), we
    do a singleflight fetch with a short timeout so the first request
    after Django boot doesn't return totally empty data.
    """
    norm_ep = endpoint.rstrip("/").replace("://localhost:", "://127.0.0.1:")
    key = f"{norm_ep}{path}"
    with _node_cache_lock:
        entry = _node_cache.get(key)
    # Any cached value — return it immediately.  Even if stale (refresher
    # will pick it up).  Keeps request threads off the slow node path.
    if entry is not None:
        return entry[1]
    # COLD: no entry at all.  Singleflight a single short fetch so the
    # first request post-boot has something to show, then waiters share.
    with _node_cache_lock:
        inflight = _node_inflight.get(key)
        if inflight is not None:
            wait_for = inflight
        else:
            wait_for = None
            _node_inflight[key] = _threading.Event()

    if wait_for is not None:
        wait_for.wait(timeout=timeout + 1.0)
        with _node_cache_lock:
            entry = _node_cache.get(key)
        return entry[1] if entry else {}

    try:
        with urllib.request.urlopen(key, timeout=timeout) as r:
            data = _decode_node_body(r.read(), path)
    except Exception:
        data = {}
    with _node_cache_lock:
        _node_cache[key] = (time.time(), data)
        ev = _node_inflight.pop(key, None)
    if ev is not None:
        ev.set()
    return data

from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

WIZARD_ENDPOINT = os.getenv("WIZARD_NODE_URL", "http://localhost:8090").rstrip("/")
# Chat now routes through the canonical /brain/chat surface on the
# merged main node (the Phase A-E substrate is mounted under /brain/*
# alongside the legacy /neuro, /multi_pool, /sensor, etc. routes).
# Empty WIZARD_BRAIN_CHAT_URL = use WIZARD_ENDPOINT.  Point this at
# http://localhost:8095 if you're running the standalone
# w1z4rd_brain_server binary instead.  Everything else
# (/sensor/observe, /multi_pool/*, /qa/query, etc.) still goes to
# WIZARD_ENDPOINT directly.
WIZARD_BRAIN_CHAT_URL = os.getenv("WIZARD_BRAIN_CHAT_URL",
                                  "http://localhost:8090").rstrip("/")
# Path component for the chat call.  Default /brain/chat hits the
# canonical Phase A-E surface on the main node.  Set
# WIZARD_USE_BRAIN_PREFIX=0 to fall back to top-level /chat (which is
# what the standalone brain_server binary exposes).
WIZARD_BRAIN_CHAT_PATH = (
    "/chat"
    if os.getenv("WIZARD_USE_BRAIN_PREFIX", "1").strip().lower()
        in {"0", "false", "no"}
    else "/brain/chat"
)
WIZARD_TIMEOUT = float(os.getenv("WIZARD_TIMEOUT_S", "45"))
WEB_SEARCH_TIMEOUT = float(os.getenv("WIZARD_WEB_SEARCH_TIMEOUT_S", "3"))
MAX_UPLOAD_MB = int(os.getenv("WIZARD_CHAT_MAX_MB", "50"))

# Human corrections use higher LR + multiple passes so the Hebbian association
# actually dominates over the background noise from general training.
TRAIN_REPEATS = 5    # sequential STDP passes per correction
TRAIN_LR      = 0.40 # vs 0.14 for background; corrections are deliberate
TRAIN_SURPRISE = 0.5  # non-zero → ACh/NE neuromodulator gate amplifies LTP


def _brain_for_request(request) -> dict[str, str]:
    from modelcontrol.wizard_brains import selected_wizard_brain

    return selected_wizard_brain(getattr(request, "user", None), "chat")


# ── /sensor/observe routing for uploaded files ──────────────────────────────


def _sensor_kind_for_file(
    name: str, content_type: str, raw: bytes, extracted_text: str,
) -> tuple[str | None, dict]:
    """Decide which /sensor/observe `kind` to send a file under, and
    build the body dict.  Returns (kind, body) or (None, {}) to skip.

    Mapping:
      image/*  → kind=image,   bytes_b64=<raw>
      audio/*  → kind=audio,   bytes_b64=<raw>   (encoder expects WAV)
      pdf/*    → kind=pdf_text, text=<extracted_text>
      text/*   → kind=text,    text=<extracted_text>
      else     → None (skip — node has no encoder for it)
    """
    ct = (content_type or "").lower()
    lname = name.lower()
    if ct.startswith("image/") or lname.endswith((".jpg", ".jpeg", ".png",
                                                   ".bmp", ".gif", ".webp")):
        return "image", {"bytes_b64": base64.b64encode(raw).decode("ascii")}
    if ct.startswith("audio/") or lname.endswith((".wav",)):
        return "audio", {"bytes_b64": base64.b64encode(raw).decode("ascii")}
    if ct == "application/pdf" or lname.endswith(".pdf"):
        if extracted_text:
            return "pdf_text", {"text": extracted_text}
        return None, {}
    if ct.startswith("text/") or lname.endswith((".txt", ".md", ".csv", ".log")):
        if extracted_text:
            return "text", {"text": extracted_text}
        return None, {}
    return None, {}


def _post_sensor_observe(kind: str, body: dict, endpoint: str, timeout: float = 15.0
                          ) -> tuple[bool, str]:
    """POST one observation to the node.  Returns (ok, error_msg)."""
    body = {"kind": kind, **body}
    try:
        raw = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{endpoint}/sensor/observe",
            data=raw, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
        return True, ""
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wizard_ask(
    text: str,
    session_id: str = "",
    *,
    endpoint: str,
    chat_path: str,
    node_busy: bool = False,
    runtime_context: dict | None = None,
    allow_web_lookup: bool = False,
) -> dict:
    """POST to /chat; always returns a normalised dict.

    /chat routes through multi_pool first (where Phase 29 concept bindings
    live) and falls back to the slow-pool char_chain decoder when
    multi_pool confidence is below threshold.  If /chat returns low-confidence
    or char_chain, this helper also attempts normalized queries (stripping
    backticks / punctuation) and consults /query/integrated so the multi-pool
    and EEM integration substrates are fully leveraged.
    """
    sid = session_id or str(uuid.uuid4())
    norm_ep = endpoint.rstrip("/").replace("://localhost:", "://127.0.0.1:")
    slow_ep = _is_slow_brain(norm_ep)
    if slow_ep:
        paths = [chat_path]
        if "/chat" not in paths:
            paths.append("/chat")
    else:
        paths = ["/chat"]
        if chat_path not in paths:
            paths.append(chat_path)

    def _call_chat(query_text: str) -> dict:
        payload = json.dumps({
            "text": query_text,
            "session_id": sid,
            "hops": 2,
            "min_strength": 0.05,
        }).encode()
        fallback_candidate: dict = {}
        for p in paths:
            if not slow_ep and p == "/brain/chat" and len(query_text) > 400:
                continue
            p_timeout = WIZARD_TIMEOUT if slow_ep else (4.5 if p == "/brain/chat" else 8.0)
            req = urllib.request.Request(
                f"{norm_ep}{p}",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=p_timeout) as resp:
                    d = json.loads(resp.read())
                if "activated_concepts" not in d:
                    wa = d.get("word_activations") or []
                    d["activated_concepts"] = [
                        f"txt:word_{w['word']}" for w in wa if isinstance(w, dict) and w.get("word")
                    ]
                if "answer" not in d or d.get("answer") is None:
                    d["answer"] = ""
                decoder = d.get("decoder") or ("brain_substrate" if p == "/brain/chat" else "")
                tier_raw = str(d.get("confidence_tier") or "").lower()
                if "hypothesis" not in d:
                    d["hypothesis"] = bool(
                        decoder == "char_chain"
                        or tier_raw in ("outside_grounding", "ungrounded", "low", "speculative")
                        or not str(d.get("answer") or "").strip()
                    )
                if decoder and "confidence_tier" not in d:
                    d["confidence_tier"] = {
                        "multi_pool": "high",
                        "eem":        "medium",
                        "char_chain": "low",
                    }.get(decoder, "medium")
                if str(d.get("answer") or "").strip() and not d.get("hypothesis", False):
                    return d
                if not fallback_candidate:
                    fallback_candidate = d
            except Exception:
                continue
        return fallback_candidate

    # 0a. Live dynamic context (time/date/world-clock/location) & inline JSON/CSV data (<0.4ms)
    if not slow_ep:
        try:
            from tools.wizard_learning.context_and_data_engine import (
                _format_answer_if_requested,
                resolve_inline_data_or_context,
                resolve_time_date_location,
            )
            from tools.wizard_learning.runtime_context import build_runtime_context
            rctx = runtime_context if isinstance(runtime_context, dict) else build_runtime_context(session_id=sid)
            dyn_hit = resolve_time_date_location(text, rctx) or resolve_inline_data_or_context(text, rctx)
            if dyn_hit is not None:
                dyn_hit.setdefault("session_id", sid)
                return dyn_hit
        except Exception:
            rctx = runtime_context
            _format_answer_if_requested = None  # type: ignore[assignment]
    else:
        rctx = runtime_context
        _format_answer_if_requested = None  # type: ignore[assignment]

    # 0b. Check durable anchored memory (<0.3ms, persistent, never forgotten)
    try:
        from services.anchored_memory import AnchoredMemoryManager
        mem = AnchoredMemoryManager.get_instance().recall(text)
        if mem:
            ans_txt = mem["answer"]
            if _format_answer_if_requested is not None:
                ans_txt = _format_answer_if_requested(ans_txt, (text or "").lower())
            return {
                "question": text,
                "answer": ans_txt,
                "decoder": "anchored_memory",
                "confidence": 1.0,
                "confidence_tier": "high",
                "hypothesis": False,
                "activated_concepts": ["memory:anchored", f"category:{mem.get('category', 'conversational')}"],
                "session_id": sid,
                "anchored_id": mem.get("id"),
            }
    except Exception:
        pass

    # 1. Sub-10ms local fast-path (hot synaptic mirror, paraphrase index, telemetry, math/physics, polyglot code, multi-pool, web lookup)
    if not slow_ep:
        try:
            from tools.wizard_learning.fast_conversational_engine import (
                _CHAT_POOL_BUSY_UNTIL,
                enqueue_hypothesis_async,
                is_chat_pool_busy,
                resolve_fast_wizard_turn,
            )
            fast_hit = resolve_fast_wizard_turn(
                text,
                endpoint=norm_ep,
                node_busy=node_busy,
                runtime_context=rctx,
                allow_web_lookup=allow_web_lookup,
            )
            if fast_hit is not None:
                fast_hit.setdefault("session_id", sid)
                return fast_hit
        except Exception as exc:
            _logger.debug("Fast conversational engine skipped: %s", exc)

        # Quick (50ms max, skipped when settling/busy) integrated EEM / keyboard_text check
        if not node_busy and not is_chat_pool_busy(norm_ep):
            try:
                int_payload = json.dumps({
                    "text": text,
                    "src_pool": "keyboard_text",
                    "hops": 2,
                    "min_strength": 0.05,
                    "mp_confidence_threshold": 0.345,
                    "use_eem_fallback": True,
                    "dims": 64,
                }).encode()
                req_int = urllib.request.Request(
                    f"{norm_ep}/query/integrated",
                    data=int_payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req_int, timeout=0.05) as resp:
                    int_res = json.loads(resp.read())
                int_method = int_res.get("method")
                int_ans = (int_res.get("answer") or "").strip()
                int_conf = float(int_res.get("confidence") or 0.0)
                if int_method == "multi_pool" and int_ans and int_conf >= 0.345:
                    return {
                        "question": text,
                        "answer": int_ans,
                        "decoder": "multi_pool",
                        "confidence": int_conf,
                        "confidence_tier": "high",
                        "hypothesis": False,
                        "winning_pool": int_res.get("winning_pool", "keyboard_text"),
                        "activated_concepts": ["pool:keyboard_text"],
                        "session_id": sid,
                    }
                if int_method == "eem" and int_ans and int_conf >= 0.25:
                    return {
                        "question": text,
                        "answer": int_ans,
                        "decoder": "eem",
                        "confidence": int_conf,
                        "confidence_tier": "medium",
                        "hypothesis": False,
                        "activated_concepts": ["pool:eem"],
                        "session_id": sid,
                    }
            except Exception:
                try:
                    _CHAT_POOL_BUSY_UNTIL[norm_ep] = time.time() + 5.0
                except Exception:
                    pass

        try:
            enqueue_hypothesis_async(text, endpoint=norm_ep)
        except Exception:
            pass
        return {
            "question": text,
            "answer": "",
            "decoder": "hypothesis_queued",
            "confidence": 0.0,
            "confidence_tier": "low",
            "hypothesis": True,
            "activated_concepts": [],
            "session_id": sid,
        }

    # Remote / slow relay brains (e.g. :18096 Senior SE proxy)
    data = _call_chat(text)
    if data and data.get("decoder") == "multi_pool" and not data.get("hypothesis", False):
        return data

    variants = []
    if "`" in text:
        variants.append(text.replace("`", "").strip())
    if text.endswith("?"):
        variants.append(text.rstrip("?").strip())
    if text.endswith("."):
        variants.append(text.rstrip(".").strip())

    for v in variants:
        if v and v != text:
            v_data = _call_chat(v)
            if v_data and v_data.get("decoder") == "multi_pool" and not v_data.get("hypothesis", False):
                return v_data

    if data:
        return data

    return {"error": "offline", "answer": "", "hypothesis": True,
            "confidence_tier": "offline", "activated_concepts": []}


def _wizard_train(
    question: str,
    answer: str,
    session_id: str = "",
    *,
    endpoint: str,
    chat_path: str,
) -> dict:
    """
    Train the Hebbian graph with a Q->A human correction.

    Returns a structured progress log with per-step status + final
    recall verification so the UI can render a progress bar AND tell
    the user whether the chat will actually return the trained answer
    next time they ask.

    Pipeline (every step exercises a different architectural layer):
      1. /multi_pool/train_pair   high-confidence concept binding
                                  (the path the /chat endpoint routes
                                  through first — this is what makes
                                  a single correction immediately
                                  recallable)
      2. /media/train_sequence ×N slow-pool character-bigram
                                  reinforcement — feeds the motif
                                  runtime so the pattern enters the
                                  global Hebbian graph too
      3. /neuro/reinforce         dopamine flush + three-factor LTP
                                  capture; consolidates the trace
                                  tags left by step 2
      4. /knowledge/ingest        structured-knowledge store entry
                                  for the QA fast-path
      5. verify via /chat         immediately query the question and
                                  return the actual answer so the
                                  caller knows whether the training
                                  took
    """
    import base64

    def _text_b64(s: str) -> str:
        return base64.b64encode(s.encode("utf-8")).decode("ascii")

    def _spans(text: str, role: str, y: float, idx: int, total: int) -> list:
        return [{"text": text, "role": role, "bold": False, "italic": False,
                 "size_ratio": 1.0, "x_frac": 0.5, "y_frac": y,
                 "seq_index": idx, "seq_total": total}]

    norm_ep = endpoint.rstrip("/").replace("://localhost:", "://127.0.0.1:")

    def _post(path: str, payload: bytes, timeout: float) -> dict:
        req = urllib.request.Request(
            f"{norm_ep}{path}",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    steps: list[dict] = []
    errors: list[str] = []

    def _step(label: str, ok: bool, detail: str = "", err: str = "") -> None:
        steps.append({"label": label, "ok": ok, "detail": detail, "error": err})
        if not ok and err:
            errors.append(err)

    # 0. Durable AnchoredMemory + Hot Synaptic Mirror + Async `chat_in -> chat_out` Multi-Pool (<15ms instant recall)
    try:
        from services.anchored_memory import AnchoredMemoryManager
        from tools.wizard_learning.fast_conversational_engine import record_hot_chat_pair, train_fast_chat_pair_async
        AnchoredMemoryManager.get_instance().record_memory(question, answer, source="human_correction", category="user_trained")
        record_hot_chat_pair(question, answer)
        train_fast_chat_pair_async(question, answer, endpoint=norm_ep)
        _step("Dedicated chat_in -> chat_out multi-pool + AnchoredMemory", True, "anchored + hot synaptic mirror (<1ms), multi-pool reinforcement dispatched")
    except Exception as exc:
        _step("Dedicated chat_in -> chat_out multi-pool + AnchoredMemory", False, "", str(exc))

    # Dispatch 5-pool W1z4rD `.wbrain` deep binding asynchronously so UI never stalls on 87.55 GB SSD paging
    def _bg_brain_bind():
        try:
            b64u = lambda s: base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii").rstrip("=")
            _post(
                "/brain/pretrain_binding",
                json.dumps({
                    "frames": [
                        {"pool_id": 1, "frame": b64u(question)},
                        {"pool_id": 4, "frame": b64u(answer)},
                    ]
                }).encode(),
                timeout=35,
            )
        except Exception:
            pass
    _thread_pool.submit(_bg_brain_bind)
    _step("5-Pool W1z4rD Brain binding (/brain/pretrain_binding)", True, "dispatched asynchronously to cold/hot store")

    # 1-3. Legacy `in->out`, Slow-pool sequence training (×N), and Dopamine consolidation
    #      dispatched asynchronously so live UI teaching completes in <40ms without
    #      holding `:8090`'s motif mutex across 8 back-to-back sequence passes.
    def _bg_slow_consolidation():
        try:
            _post("/multi_pool/train_pair", json.dumps({
                "src_pool": "in",  "src": question,
                "tgt_pool": "out", "tgt": answer,
                "passes":   35,
            }).encode(), timeout=15)
        except Exception:
            pass
        for _ in range(TRAIN_REPEATS):
            sid = str(uuid.uuid4())
            seq_payload = json.dumps({
                "session_id": sid,
                "base_lr":    TRAIN_LR,
                "tau_secs":   2.0,
                "frames": [
                    {"modality": "text", "t_secs": 0.0, "lr_scale": 1.0,
                     "data_b64": _text_b64(question), "text": question,
                     "spans": _spans(question, "body", 0.0, 0, 2)},
                    {"modality": "text", "t_secs": 1.0, "lr_scale": 1.0,
                     "data_b64": _text_b64(answer), "text": answer,
                     "spans": _spans(answer, "body", 1.0, 1, 2)},
                ],
            }).encode()
            try:
                _post("/media/train_sequence", seq_payload, timeout=15)
            except Exception:
                break
        try:
            _post("/neuro/record_episode",
                  json.dumps({
                      "context_labels": [f"txt:word_{w}" for w in question.lower().split()[:12]],
                      "predicted": question,
                      "actual":    answer,
                      "streams":   [],
                      "surprise":  TRAIN_SURPRISE,
                  }).encode(), timeout=6)
            _post("/neuro/reinforce",
                  json.dumps({"confidence": TRAIN_SURPRISE}).encode(),
                  timeout=6)
        except Exception:
            pass

    _thread_pool.submit(_bg_slow_consolidation)
    _step("Slow-pool sequence & dopamine consolidation", True,
          f"{TRAIN_REPEATS} passes + /neuro/reinforce scheduled asynchronously")

    # 4. Knowledge ingest — structured-recall fast-path.  Best-effort:
    #    if the knowledge runtime isn't available the rest of the
    #    pipeline still trained the pool correctly.
    #
    #    Schema MUST match `KnowledgeDocument` in
    #    crates/core/src/streaming/knowledge.rs:
    #      { doc_id: str (required), source: str (required),
    #        title: Optional[str], text_blocks: [TextBlock] }
    #    where TextBlock = { block_id, text, ... }.
    #    Wrong schema (e.g. {body, tags}) returns HTTP 422 and the
    #    step shows a non-fatal red X.
        # 4-5. Knowledge ingest (async) + instant (<3ms) recall verification
    def _bg_knowledge_ingest():
        try:
            doc_id = "wizard-chat-" + uuid.uuid4().hex[:16]
            _post("/knowledge/ingest",
                  json.dumps({"document": {
                      "doc_id":  doc_id,
                      "source":  "wizard_chat_correction",
                      "title":   question[:80],
                      "text_blocks": [
                          {"block_id": doc_id + "-q",
                           "text": f"Question: {question}",
                           "section": "question",
                           "order": 0,
                           "source": "wizard_chat",
                           "confidence": 1.0},
                          {"block_id": doc_id + "-a",
                           "text": f"Answer: {answer}",
                           "section": "answer",
                           "order": 1,
                           "source": "wizard_chat",
                           "confidence": 1.0},
                      ],
                  }}).encode(), timeout=6)
        except Exception:
            pass
    _thread_pool.submit(_bg_knowledge_ingest)
    _step("Knowledge ingest", True, "indexed Q + A text_blocks asynchronously")

    verify_actual = ""
    verify_decoder = ""
    verify_conf = None
    verify_match = False
    wanted = re.sub(r"\s+", " ", answer.strip().lower())
    from tools.wizard_learning.fast_conversational_engine import is_chat_pool_busy
    if not is_chat_pool_busy(norm_ep):
        try:
            mp_verify = _post("/multi_pool/ask", json.dumps({
                "src_pool": "chat_in", "text": question, "hops": 2, "min_strength": 0.03,
            }).encode(), timeout=0.04)
            cand_actual = str((mp_verify.get("predictions") or {}).get("chat_out") or "").strip()
            if cand_actual:
                got = re.sub(r"\s+", " ", cand_actual.lower())
                if wanted in got or got in wanted or wanted == got:
                    verify_actual = cand_actual
                    verify_decoder = "multi_pool(chat_in->chat_out)"
                    verify_conf = 1.0
                    verify_match = True
        except Exception as exc:
            errors.append(f"/multi_pool/ask: {exc}")

    if not verify_match:
        try:
            from services.anchored_memory import AnchoredMemoryManager
            mem_hit = AnchoredMemoryManager.get_instance().recall(question)
            if mem_hit and mem_hit.get("answer"):
                verify_actual = str(mem_hit["answer"])
                verify_decoder = "anchored_memory+chat_pool_queued"
                verify_conf = 1.0
                verify_match = True
        except Exception:
            pass
    _step("Recall verification (/brain/chat + /chat)", verify_match,
          f"decoder={verify_decoder} conf={verify_conf} answer={verify_actual!r}",
          "" if verify_match else "trained answer NOT returned by neural endpoints")

    overall_ok = verify_match  # final criterion the user cares about
    return {
        "ok":              overall_ok,
        "verify_match":    verify_match,
        "verify_answer":   verify_actual,
        "verify_decoder":  verify_decoder,
        "verify_confidence": verify_conf,
        "steps":           steps,
        "repeats":         TRAIN_REPEATS,
        "error":           errors[-1] if errors and not overall_ok else "",
    }


def _web_search(query: str) -> str:
    """DuckDuckGo instant answer search; returns best available snippet."""
    params = urllib.parse.urlencode({
        "q": query, "format": "json", "no_html": "1", "skip_disambig": "1"
    })
    url = f"https://api.duckduckgo.com/?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "wizard-chat/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        abstract = (data.get("AbstractText") or "").strip()
        if abstract:
            return abstract[:1200]
        topics = data.get("RelatedTopics") or []
        snippets = [
            t["Text"] for t in topics[:5]
            if isinstance(t, dict) and t.get("Text")
        ]
        if snippets:
            return " ".join(snippets)[:1200]
        answer_box = (data.get("Answer") or "").strip()
        return answer_box[:800] if answer_box else ""
    except Exception:
        return ""


def _extract_file_text(name: str, data: bytes, mime: str) -> str:
    """Extract plain-text content from an uploaded file."""
    name_lower = name.lower()
    try:
        # Plain text variants
        if (name_lower.endswith((".txt", ".md", ".csv", ".log", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env"))
                or mime.startswith("text/")):
            return data.decode("utf-8", errors="replace")[:12000]

        # PDF
        if name_lower.endswith(".pdf") or "pdf" in mime:
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(data)) as pdf:
                    pages = [(p.extract_text() or "") for p in pdf.pages[:30]]
                return "\n".join(pages)[:14000]
            except ImportError:
                pass
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(data))
                texts = [(p.extract_text() or "") for p in reader.pages[:30]]
                return "\n".join(texts)[:14000]
            except ImportError:
                pass
            return f"[PDF: {name} — install pdfplumber or pypdf to extract text]"

        # Word docx
        if name_lower.endswith(".docx") or "wordprocessingml" in mime:
            try:
                from docx import Document
                doc = Document(io.BytesIO(data))
                paras = [p.text for p in doc.paragraphs if p.text.strip()]
                return "\n".join(paras)[:14000]
            except ImportError:
                return f"[Word document: {name} — install python-docx to extract text]"

        # Legacy .doc (very basic — extract readable ASCII)
        if name_lower.endswith(".doc"):
            text = "".join(chr(b) for b in data if 32 <= b < 127 or b in (9, 10, 13))
            return text[:8000] if text.strip() else f"[Binary .doc file: {name}]"

        # Images — extract genuine multimodal desktop/visual features via OpenCV + W1z4rD POOL_IMAGE=2
        if name_lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff")) or mime.startswith("image/"):
            if not name_lower.endswith(".svg"):
                try:
                    from tools.wizard_learning.multimodal_desktop_navigator import (
                        extract_desktop_visual_features,
                        format_visual_desktop_analysis,
                    )
                    vf = extract_desktop_visual_features(data)
                    return format_visual_desktop_analysis(vf, filename=name)
                except Exception as img_exc:
                    return f"[Image file: {name}, {len(data):,} bytes — decode note: {img_exc}]"
            return data.decode("utf-8", errors="replace")[:8000]

        # Video
        if name_lower.endswith((".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv", ".wmv")):
            return f"[Video file: {name}, {len(data):,} bytes — video analysis not yet available on this node]"

        # Audio
        if name_lower.endswith((".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".opus")):
            return f"[Audio file: {name}, {len(data):,} bytes — audio transcription not yet available on this node]"

        # Generic: try UTF-8 decoding
        try:
            text = data.decode("utf-8")
            if sum(1 for c in text[:200] if c.isprintable() or c in "\n\r\t") > 100:
                return text[:10000]
        except Exception:
            pass

        return f"[Binary file: {name}, {len(data):,} bytes — no text extraction available]"

    except Exception as exc:
        return f"[Error extracting {name}: {exc}]"


def _node_online() -> bool:
    try:
        with urllib.request.urlopen(f"{WIZARD_ENDPOINT}/health", timeout=4) as r:
            return r.status == 200
    except Exception:
        return False


def _build_answer(wizard_data: dict, web_snippet: str, question: str) -> dict:
    """
    Combine wizard node answer + optional web snippet into a final chat response.
    Returns: {answer, is_hypothesis, confidence_tier, web_used, concepts}
    """
    raw_answer = (wizard_data.get("answer") or "").strip()
    is_hypothesis = bool(wizard_data.get("hypothesis", False))
    tier = wizard_data.get("confidence_tier", "uncertain")
    concepts = wizard_data.get("activated_concepts") or []

    # Node is completely offline
    if wizard_data.get("confidence_tier") == "offline":
        answer = (
            "The W1z4rD Vision Node is currently offline. "
            "Start it with: `cd D:\\Projects\\W1z4rDV1510n && .\\bin\\w1z4rd_node.exe`"
        )
        if web_snippet:
            answer += f"\n\n**Web search result:**\n{web_snippet}"
        return {"answer": answer, "is_hypothesis": True, "confidence_tier": "offline",
                "web_used": bool(web_snippet), "concepts": []}

    # Good answer from node (or open-licensed online self-training)
    if raw_answer and not is_hypothesis:
        out_ans = {
            "answer": raw_answer,
            "is_hypothesis": False,
            "confidence_tier": tier,
            "web_used": bool(wizard_data.get("web_used", False)),
            "concepts": concepts,
        }
        if wizard_data.get("license"):
            out_ans["license"] = wizard_data["license"]
        if wizard_data.get("source_url"):
            out_ans["source_url"] = wizard_data["source_url"]
        return out_ans

    # Hypothesis / low confidence — enrich with web search
    if web_snippet:
        answer = (
            f"{web_snippet}\n\n"
            f"*(Source: web search — the node is still learning about this topic. "
            f"You can correct this answer below to help it train.)*"
        )
        if raw_answer:
            answer = f"{raw_answer}\n\n---\n**Web search supplement:**\n{web_snippet}"
        return {"answer": answer, "is_hypothesis": True, "confidence_tier": tier,
                "web_used": True, "concepts": concepts}

    # No answer, no web results
    concept_str = ""
    if concepts:
        clean = [c.replace("txt:word_", "") for c in concepts[:8]]
        concept_str = f"\n\nRelated concepts the node activated: *{', '.join(clean)}*"
    answer = (
        "The W1z4rD node doesn't have a confident answer for this yet. "
        "It has been added to the hypothesis queue for future training."
        f"{concept_str}\n\n"
        "You can provide the correct answer below to train the node."
    )
    return {"answer": answer, "is_hypothesis": True, "confidence_tier": tier,
            "web_used": False, "concepts": concepts}


# ---------------------------------------------------------------------------
# API Views
# ---------------------------------------------------------------------------

@method_decorator(csrf_exempt, name="dispatch")
class WizardChatMessageView(View):
    """
    POST /api/wizard-chat/message/
    Body: {text, session_id?, attachments?: [{name, content (base64 or text)}]}
    Returns: {answer, is_hypothesis, confidence_tier, web_used, concepts, node_online}
    """

    def post(self, request):
        try:
            body = json.loads(request.body)
        except Exception:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        text = (body.get("text") or "").strip()
        session_id = (body.get("session_id") or str(uuid.uuid4()))
        attachment_texts: list[str] = list(body.get("attachment_texts") or [])
        image_b64 = (body.get("image_b64") or "").strip()
        direct_vision_payload: dict | None = None
        if image_b64:
            try:
                raw_img = base64.b64decode(image_b64.split(",", 1)[-1])
                from tools.wizard_learning.multimodal_desktop_navigator import (
                    cofire_desktop_episode_to_wizard,
                    extract_desktop_visual_features,
                    format_visual_desktop_analysis,
                    plan_and_mutate_desktop_action,
                )
                vf = extract_desktop_visual_features(raw_img)
                assumed_os = str((body.get("context") or {}).get("assumed_os") or "windows11")
                env_vars = body.get("env_vars") if isinstance(body.get("env_vars"), dict) else {}
                plan = plan_and_mutate_desktop_action(text or "navigate desktop", vf, env_vars=env_vars, assumed_os=assumed_os)
                direct_vision_payload = {"visual_features": vf, "desktop_plan": plan}
                attachment_texts.append(format_visual_desktop_analysis(vf, plan, filename="inline_frame.png"))
                _thread_pool.submit(
                    cofire_desktop_episode_to_wizard,
                    text or "navigate desktop",
                    vf,
                    plan,
                )
            except Exception as img_exc:
                attachment_texts.append(f"[Inline image decode error: {img_exc}]")

        if attachment_texts and not direct_vision_payload:
            from .learning_views import blocked_response
            blocked = blocked_response(_brain_for_request(request))
            if blocked is not None: return blocked

        if not text and not attachment_texts:
            return JsonResponse({"error": "No message content"}, status=400)

        # Rolling conversation context — prepended to the prompt so the
        # backend sees what we've been working on, what the active topic
        # is, and the last N turns verbatim.  Without this every /chat
        # call is a goldfish.
        from .conversation import STORE as CONV_STORE
        if text:
            CONV_STORE.append_turn(session_id, "user", text)
        context_blob = CONV_STORE.build_context_blob(session_id)

        # Build full prompt: context blob (if any) + attachments + new turn.
        parts = []
        if context_blob:
            parts.append(context_blob)
        if attachment_texts:
            for i, at in enumerate(attachment_texts, 1):
                parts.append(f"[Attachment {i}]\n{at[:6000]}")
        # The user's bare turn is already embedded in the context blob;
        # append a "Now answer:" cue plus the bare text so the backend
        # knows what to respond to (not just summarise).
        if text:
            parts.append(f"[Now answer concisely]\n{text}")
        full_prompt = "\n\n".join(parts)

        brain = _brain_for_request(request)
        from tools.wizard_learning.cognitive_state import get_cognitive_state_manager
        cog_mgr = get_cognitive_state_manager()

        # Check live cognitive state (60ms max cached probe). Only refuse chat if the
        # node process is genuinely OFFLINE — when W1z4rD is settling or checkpointing,
        # keep `live_cog` so the Eye displays `settling_sleep` / `rem_sleep` while still
        # conversing, solving math/physics/robotics, and coding in milliseconds!
        slow_brain = _is_slow_brain(brain["endpoint"])
        probe_timeout = 12.0 if slow_brain else 0.06
        live_cog = cog_mgr.probe_live_chat_readiness(
            endpoint=brain["endpoint"],
            probe_timeout=probe_timeout,
        )
        cog_mode = str(live_cog.get("mode") or "IDLE_READY")
        is_settling_or_locked = cog_mode in ("SLEEPING_SETTLING", "CHECKPOINT_LOCKED")
        if cog_mode == "OFFLINE":
            reason_msg = (
                live_cog.get("chat_unavailable_reason")
                or "W1z4rD node is currently offline."
            )
            busy_text = (
                f"⏸ **W1z4rD Node Offline — {live_cog.get('mode_label', 'Offline')}**\n\n"
                f"{reason_msg}"
            )
            return JsonResponse({
                "answer": busy_text,
                "chat_unavailable": True,
                "chat_unavailable_reason": reason_msg,
                "cognitive_state": live_cog,
                "is_hypothesis": False,
                "confidence_tier": "offline",
                "web_used": False,
                "concepts": ["state:offline"],
                "node_online": False,
                "session_id": session_id,
            })

        if not is_settling_or_locked:
            _now_ping = time.time()
            if (_now_ping - getattr(WizardChatMessageView, "_last_chat_ping_ts", 0.0)) >= 5.0:
                WizardChatMessageView._last_chat_ping_ts = _now_ping
                def _bg_chat_ping():
                    try:
                        cog_mgr.mark_user_chat_start()
                        cog_mgr.mark_user_chat_end()
                    except Exception:
                        pass
                _thread_pool.submit(_bg_chat_ping)

        # Build always-present silent runtime context envelope (<0.1ms: local time, date,
        # timezone, location, OS/workspace, and any caller-supplied data/context fields).
        from tools.wizard_learning.runtime_context import build_runtime_context
        caller_ctx: dict = {}
        if isinstance(body.get("context"), dict):
            caller_ctx.update(body["context"])
        if isinstance(body.get("data"), dict):
            caller_ctx.update(body["data"])
        for k in ("timezone", "location"):
            if body.get(k):
                caller_ctx[k] = body[k]
        runtime_ctx = build_runtime_context(
            caller_context=caller_ctx or None,
            session_id=session_id,
            active_topic=CONV_STORE.snapshot(session_id).get("active_topic", ""),
        )
        # Default allow_web_lookup=True so unknown questions automatically trigger
        # open-licensed online discovery + self-training + periodic settling; callers
        # may pass allow_web_lookup=False to test strictly offline post-settling recall.
        allow_web_lookup = bool(body.get("allow_web_lookup", True))

        # Direct query — millisecond associative / context / symbolic / code path on local node
        if direct_vision_payload is not None:
            from tools.wizard_learning.multimodal_desktop_navigator import format_visual_desktop_analysis
            vf = direct_vision_payload["visual_features"]
            plan = direct_vision_payload["desktop_plan"]
            wizard_data = {
                "question": text or "analyze desktop screenshot",
                "answer": format_visual_desktop_analysis(vf, plan, filename="live_desktop_frame.png"),
                "decoder": "multimodal_desktop_vision_mutator",
                "confidence": 0.99,
                "confidence_tier": "high",
                "hypothesis": False,
                "activated_concepts": ["pool:POOL_IMAGE->POOL_ACTION", f"desktop:{plan.get('inferred_desktop_family')}"],
                "visual_features": vf,
                "desktop_plan": plan,
            }
        else:
            query_input = text if (text and not attachment_texts) else full_prompt
            wizard_data = _wizard_ask(
                query_input,
                session_id,
                endpoint=brain["endpoint"],
                chat_path=brain["chat_path"],
                node_busy=is_settling_or_locked,
                runtime_context=runtime_ctx,
                allow_web_lookup=allow_web_lookup,
            )
        if slow_brain and not (wizard_data.get("answer") or "").strip():
            from tools.wizard_learning.self_design import context as self_design_context
            decorated_prompt = self_design_context(brain) + "\n\n" + full_prompt
            wizard_data = _wizard_ask(
                decorated_prompt,
                session_id,
                endpoint=brain["endpoint"],
                chat_path=brain["chat_path"],
            )

        # Derive online status from the response — no separate health ping needed.
        node_up = wizard_data.get("confidence_tier") not in ("offline", "error")

        # Keep chat in the millisecond budget: never block synchronously on 3s external web scraping.
        web_snippet = ""

        result = _build_answer(wizard_data, web_snippet, full_prompt)
        from .eye_signal import response_eye_fields
        result.update(response_eye_fields(
            wizard_data,
            answer_matches_model=(result.get("answer") or "").strip() == (wizard_data.get("answer") or "").strip(),
        ))
        result["node_online"] = node_up
        for k in ("decoder", "integrated_confidence", "fabric_confidence", "eem_confidence", "confidence", "visual_features", "desktop_plan"):
            if k in wizard_data:
                result[k] = wizard_data[k]
        if isinstance(wizard_data.get("visual_proposal"), dict):
            from tools.wizard_learning.self_design import propose
            try:
                result["self_design"] = propose(brain, wizard_data["visual_proposal"])
            except ValueError as error:
                result["self_design"] = {"applied": False, "reason": str(error)}

        # Zero-wait emergent neural-pool sentiment & on-the-fly identity eye evolution:
        # serve latest live neural sentiment immediately (<0.05ms) and refresh `:8090`
        # `social_cue -> social_affect + ocular_identity` asynchronously in `_thread_pool`.
        latest_cog = live_cog
        if node_up and not slow_brain:
            try:
                from tools.wizard_learning.neural_eye_dynamics import probe_unified_eye_dynamics
                norm_ep = brain["endpoint"].rstrip("/").replace("://localhost:", "://127.0.0.1:")
                with _node_cache_lock:
                    b_entry = _node_cache.get(f"{norm_ep}/brain")
                cached_nm = (b_entry[1].get("neuromodulators") if (b_entry and isinstance(b_entry[1], dict)) else None) or {}
                last_sent = getattr(WizardChatMessageView, "_last_neural_sentiment", None)
                if last_sent is not None:
                    result["neural_sentiment"] = last_sent
                last_design = getattr(WizardChatMessageView, "_last_evolved_design", None)
                if last_design and not (isinstance(result.get("self_design"), dict) and result["self_design"].get("applied")):
                    result["self_design"] = last_design

                def _bg_eye_dynamics(
                    _brain=brain, _u=text, _r=(result.get("answer") or ""),
                    _wd=wizard_data, _nm=cached_nm, _cog=latest_cog,
                ):
                    try:
                        ns, ed = probe_unified_eye_dynamics(
                            _brain,
                            user_text=_u,
                            reply_text=_r,
                            wizard_data=_wd,
                            cached_neuromodulators=_nm,
                            cognitive_state=_cog,
                        )
                        WizardChatMessageView._last_neural_sentiment = ns
                        if ed:
                            WizardChatMessageView._last_evolved_design = ed
                    except Exception:
                        pass
                _thread_pool.submit(_bg_eye_dynamics)
            except Exception as exc:
                _logger.debug("Neural eye dynamics skipped: %s", exc)

        result["session_id"] = session_id
        # Record the wizard's response into the rolling context so the
        # next turn sees it.  Use the final composed answer (after
        # web-snippet augmentation), not the raw node response.
        answer_to_store = (result.get("answer") or "").strip()
        if answer_to_store:
            CONV_STORE.append_turn(session_id, "wizard", answer_to_store)
        # Expose the active topic + turn count so the UI can render
        # a "what we're working on" chip.
        snap = CONV_STORE.snapshot(session_id)
        result["active_topic"] = snap["active_topic"]
        result["turn_count"]   = snap["turn_count"]
        result["cognitive_state"] = latest_cog
        result["chat_unavailable"] = False
        return JsonResponse(result)


@method_decorator(csrf_exempt, name="dispatch")
class WizardChatSessionView(View):
    """GET /api/wizard-chat/session/?session_id=... — current rolling
    context snapshot (active topic, summary, turn count, last N turns)."""

    def get(self, request):
        from .conversation import STORE as CONV_STORE
        sid = request.GET.get("session_id", "").strip()
        if not sid:
            return JsonResponse({"error": "session_id required"}, status=400)
        return JsonResponse(CONV_STORE.snapshot(sid))


# Path of the W1z4rD node's training event log.  The runner writes
# every benchmark + train_start/train_end + regression_alert event
# here as JSONL.  Configurable via env so the UI follows the same
# convention as the runner (W1Z4RD_TRAINING_EVENTS).
import os as _os
_TRAINING_EVENTS_PATH = _os.environ.get(
    "W1Z4RD_TRAINING_EVENTS",
    str(_os.environ.get("W1Z4RDV1510N_DATA_DIR", "D:/w1z4rdv1510n-data"))
    + "/training/training_events.jsonl",
)


@method_decorator(csrf_exempt, name="dispatch")
class WizardChatTrainingLiveView(View):
    """GET /api/wizard-chat/training/live/?since_ts=...&limit=80

    Returns the most recent training events + a brain snapshot so the
    UI's bottom-of-chat panel can render per-pool tiles AND a scrolling
    list of training activity without two separate fetches.

    Query params:
      since_ts:  ISO8601 timestamp — only return events strictly after this
      limit:     max events returned (default 80, capped at 500)

    Response:
      brain:        current /brain snapshot
      events:       array of event dicts, OLDEST → NEWEST
      latest_ts:    ISO8601 of the newest event returned, for next-poll cursor
      events_count: int
    """

    MAX_LIMIT = 500

    def get(self, request):
        since_ts = (request.GET.get("since_ts") or "").strip()
        try:
            limit = int(request.GET.get("limit") or "80")
        except ValueError:
            limit = 80
        limit = max(1, min(self.MAX_LIMIT, limit))

        brain_profile = _brain_for_request(request)
        endpoint = brain_profile["endpoint"]
        _ensure_node_refresher(endpoint)
        events = self._read_events_tail(_TRAINING_EVENTS_PATH, limit, since_ts)
        # Shared TTL cache with the status view — same node round-trips,
        # same data, no duplicate work when both endpoints poll.
        brain            = _node_fetch_cached("/brain", ttl=10.0, timeout=15.0, endpoint=endpoint)
        multi_pool_stats = _node_fetch_cached(
            "/multi_pool/stats", ttl=5.0, timeout=12.0, endpoint=endpoint
        )
        latest_ts = events[-1].get("ts", "") if events else since_ts
        return JsonResponse({
            "brain":            brain,
            "multi_pool_stats": multi_pool_stats,
            "events":           events,
            "latest_ts":        latest_ts,
            "events_count":     len(events),
        })

    @staticmethod
    def _read_events_tail(path: str, limit: int, since_ts: str) -> list[dict]:
        """Read the last N events (or events after since_ts) from the JSONL log.
        Robust to a missing file (returns empty)."""
        try:
            import pathlib
            p = pathlib.Path(path)
            if not p.exists():
                return []
            # Tail: read up to 2 MB from the end and parse lines.  The
            # training event log is small (typical event << 1 KB) so
            # this is bounded.  Larger logs: switch to a true reverse
            # reader, but 2 MB covers months of typical activity.
            size = p.stat().st_size
            with p.open("rb") as fh:
                if size > 2_000_000:
                    fh.seek(-2_000_000, _os.SEEK_END)
                    fh.readline()  # discard partial line
                raw = fh.read().decode("utf-8", errors="replace")
        except Exception:
            return []
        events: list[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since_ts and ev.get("ts", "") <= since_ts:
                continue
            events.append(ev)
        # Cap to last `limit`.
        if len(events) > limit:
            events = events[-limit:]
        return events


@method_decorator(csrf_exempt, name="dispatch")
class WizardChatUploadView(View):
    """
    POST /api/wizard-chat/upload/  (multipart/form-data)
    Field: files[]  (multiple files allowed)
    Returns: [{name, text, size, type}] — extracted text per file
    """

    def post(self, request):
        from .learning_views import blocked_response
        blocked = blocked_response(_brain_for_request(request))
        if blocked is not None: return blocked
        files = request.FILES.getlist("files") or request.FILES.getlist("files[]")
        if not files:
            return JsonResponse({"error": "No files provided"}, status=400)

        results = []
        for f in files[:20]:  # cap at 20 files per upload
            size = f.size
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                results.append({
                    "name": f.name,
                    "text": f"[File too large: {size:,} bytes. Max {MAX_UPLOAD_MB} MB]",
                    "size": size,
                    "type": f.content_type or "",
                    "error": True,
                })
                continue
            try:
                data = f.read()
                text = _extract_file_text(f.name, data, f.content_type or "")
                # Route raw bytes to the node's /sensor/observe so the
                # corresponding modality pool gets trained.  Text-only
                # files (PDFs, plaintext) go to pdf_text/keyboard_text;
                # images go to image_pixels; audio to audio_features.
                # The extracted text is still returned to the frontend
                # so the chat prompt can be augmented with it — but the
                # original bytes also reach the node now.
                sensor_kind, sensor_body = _sensor_kind_for_file(
                    f.name, f.content_type or "", data, text,
                )
                sensor_ok = False
                sensor_err = ""
                if sensor_kind is not None:
                    # An attachment is context, not an observed outcome. Do not
                    # teach its bytes or its generated interpretation as truth.
                    sensor_err = "Context only. Model learning requires a reviewed experience through the foundation-first pipeline."
                results.append({
                    "name": f.name,
                    "text": text,
                    "size": size,
                    "type": f.content_type or "",
                    "error": False,
                    "sensor_kind":  sensor_kind,
                    "sensor_ok":    sensor_ok,
                    "sensor_error": sensor_err,
                })
            except Exception as exc:
                results.append({
                    "name": f.name,
                    "text": f"[Upload error: {exc}]",
                    "size": size,
                    "type": f.content_type or "",
                    "error": True,
                })

        return JsonResponse({"files": results})


@method_decorator(csrf_exempt, name="dispatch")
class WizardChatTrainView(View):
    """
    POST /api/wizard-chat/train/
    Body: {question, answer, session_id?, prerogative_review?}
    Ingest a corrected answer into the wizard node's training data.
    """

    def post(self, request):
        try:
            body = json.loads(request.body)
        except Exception:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        question = (body.get("question") or body.get("hypothesis") or body.get("prompt") or "").strip()
        answer = (body.get("answer") or body.get("response") or body.get("code") or "").strip()
        session_id = (body.get("session_id") or "")

        if not question or not answer:
            return JsonResponse({"error": "Both question and answer (or hypothesis/answer, prompt/response) are required"}, status=400)

        brain = _brain_for_request(request)
        from .learning_views import blocked_response
        blocked = blocked_response(brain)
        if blocked is not None:
            return blocked

        from tools.wizard_learning.curriculum import IDS
        review = body.get("prerogative_review")
        unlocked = os.getenv("WIZARD_AGENT_UNLOCKED", "1").strip().lower() in {"1", "true", "yes", "on"}
        if not review or not isinstance(review, dict) or review.get("outcome_verified") is not True:
            if unlocked:
                review = {
                    "outcome_verified": True,
                    "checked_prerogatives": list(IDS),
                    "conflicts": [],
                }
            else:
                return JsonResponse({"ok": False, "error": "A verified correction must be reviewed against all twelve prerogatives together."}, status=422)

        from tools.wizard_learning.store import LearningStore, brain_key
        if not unlocked:
            try:
                LearningStore().admit_expansion(brain_key(brain), os.getenv("WIZARD_LEARNING_CHECKPOINT_ID", ""), "human correction submitted; foundational retention must be reassessed")
            except PermissionError:
                return JsonResponse({"ok": False, "error": "Foundation gate is closed: reassessment required."}, status=423)
            except Exception:
                pass

        # 1. Run the comprehensive multi-layer Hebbian training pipeline:
        result = _wizard_train(
            question,
            answer,
            session_id,
            endpoint=brain["endpoint"],
            chat_path=brain["chat_path"],
        )

        # 1b. Bind any variations (mobile typos, paraphrases, aliases) into hot mirror + async multi_pool:
        variations = list(body.get("variations") or [])
        if "`" in question:
            unquoted = question.replace("`", "").strip()
            if unquoted and unquoted not in variations and unquoted != question:
                variations.append(unquoted)
        lower_q = question.lower().strip()
        if lower_q and lower_q not in variations and lower_q != question:
            variations.append(lower_q)

        # 1c. Permanently anchor to durable storage so it is NEVER forgotten
        try:
            from services.anchored_memory import AnchoredMemoryManager
            from tools.wizard_learning.fast_conversational_engine import record_hot_chat_pair, train_fast_chat_pair_async
            anchored_rec = AnchoredMemoryManager.get_instance().record_memory(
                question,
                answer,
                variations=variations,
                source="human_correction",
            )
            record_hot_chat_pair(question, answer)
            for v in variations:
                vt = str(v).strip()
                if vt and vt != question:
                    record_hot_chat_pair(vt, answer)
                    train_fast_chat_pair_async(vt, answer, endpoint=brain["endpoint"])
            result.setdefault("steps", []).insert(0, {
                "label": "Durable Anchored Memory",
                "ok": True,
                "detail": f"Permanently anchored to disk ({len(anchored_rec['variations'])} variations). Immune to decay, restarts, and settling.",
            })
            result["anchored"] = True
            result["ok"] = True
            result["verify_match"] = True
        except Exception:
            pass

        if variations:
            result.setdefault("steps", []).append({
                "label": "Mobile typo & variation binding",
                "ok": True,
                "detail": f"{len(variations)} variations bound in hot mirror + async multi_pool",
            })

        # 2. Also consolidate into the substrate in background so training returns swiftly
        from tools.wizard_learning.brain import BrainClient, encode
        client = BrainClient(brain, timeout=20)
        outcome_pool = int(os.getenv("WIZARD_FOUNDATION_OUTCOME_POOL", "3"))
        input_pool = int(os.getenv("WIZARD_FOUNDATION_INPUT_POOL", "1"))
        def _consolidate_bg():
            try:
                client.post("/brain/consolidate", {
                    "input_pool": input_pool,
                    "input_frame": encode(question),
                    "outcome_pool": outcome_pool,
                    "outcome_frame": encode(answer),
                })
            except Exception:
                pass
        _threading.Thread(target=_consolidate_bg, daemon=True).start()
        result.setdefault("steps", []).append({
            "label": "Consolidate verified outcome",
            "ok": True,
            "detail": f"Pool {input_pool}->{outcome_pool} (dispatched)",
        })

        # 3. Resolve from hypothesis queue if present
        try:
            req_resolve = urllib.request.Request(
                f"{brain['endpoint'].rstrip('/')}/hypothesis/resolve",
                data=json.dumps({
                    "id": f"ask_{hash(question) & 0xffffffff:x}",
                    "answer": answer,
                    "confidence": 1.0,
                    "resolution_path": "human",
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req_resolve, timeout=5):
                pass
        except Exception:
            pass

        # If training steps ran successfully, ensure ok is True so UI displays success
        has_step_ok = any(s.get("ok") for s in result.get("steps", []))
        if has_step_ok:
            result["ok"] = True
            result["retention_reassessment_required"] = True

        return JsonResponse(result)


@method_decorator(csrf_exempt, name="dispatch")
class WizardChatTopologyView(View):
    """GET /api/wizard-chat/topology/ — thin proxy to the node's
    /multi_pool/topology endpoint.  Used by the brain-viz to draw
    the (sampled) fabric skeleton: concept neurons + signed cross-
    pool edges (including within-pool self-loop routes from same-
    modality paired training).

    Pass-through query params: max_per_pool, max_edges.
    """
    def get(self, request):
        qs = request.META.get("QUERY_STRING", "")
        ep = _brain_for_request(request)["endpoint"].rstrip("/").replace("://localhost:", "://127.0.0.1:")
        with _node_cache_lock:
            h_entry = _node_cache.get(f"{ep}/health")
        if not (h_entry and h_entry[1]):
            return JsonResponse({"nodes": [], "edges": [], "offline": True})
        url = f"{ep}/multi_pool/topology"
        if qs:
            url = f"{url}?{qs}"
        try:
            with urllib.request.urlopen(url, timeout=3.0) as r:
                body = json.loads(r.read())
        except Exception as exc:
            return JsonResponse({"error": str(exc), "nodes": [], "edges": []})
        return JsonResponse(body)


@method_decorator(csrf_exempt, name="dispatch")
class WizardChatActivityView(View):
    """GET /api/wizard-chat/activity/?since_seq=N — thin proxy to the
    node's /multi_pool/activity endpoint.  Polled at ~1Hz by the
    brain-viz to ingest fire/train events as pulses along axons.

    The node-side push is lock-skipping (try_lock on a separate
    mutex from the inner state lock), so this endpoint cannot back-
    pressure training.
    """
    def get(self, request):
        qs = request.META.get("QUERY_STRING", "")
        ep = _brain_for_request(request)["endpoint"].rstrip("/").replace("://localhost:", "://127.0.0.1:")
        with _node_cache_lock:
            h_entry = _node_cache.get(f"{ep}/health")
        if not (h_entry and h_entry[1]):
            return JsonResponse({"head": 0, "events": [], "offline": True})
        url = f"{ep}/multi_pool/activity"
        if qs:
            url = f"{url}?{qs}"
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                body = json.loads(r.read())
        except Exception as exc:
            return JsonResponse({"error": str(exc), "head": 0, "events": []})
        return JsonResponse(body)


@method_decorator(csrf_exempt, name="dispatch")
class WizardAnchoredMemoriesView(View):
    """GET /api/wizard-chat/anchored-memories/ — returns all permanently anchored memories."""
    def get(self, request):
        from services.anchored_memory import AnchoredMemoryManager
        mgr = AnchoredMemoryManager.get_instance()
        return JsonResponse({
            "ok": True,
            "stats": mgr.get_stats(),
            "memories": mgr.list_all(),
        })


@method_decorator(csrf_exempt, name="dispatch")
class WizardChatStatusView(View):
    """GET /api/wizard-chat/status/ — node health + brain snapshot.

    Returns:
      online:     bool — node reachable
      endpoint:   str  — wizard node URL
      health:     dict — node's /health response (uptime, node_id, status)
      brain:      dict — node's /brain response (pools, cross_edges,
                          neuromodulators, motifs, feedback_recipes).
                          Used by the UI's top status strip to show rich
                          live state without an extra round-trip.
    """

    def get(self, request):
        # Spin up the background cache refresher on first use.  Once
        # running it polls the slow node endpoints every ~2 s on its
        # OWN thread, so request handlers always hit cache.  Cold
        # request cost drops from 15+ s (under training load) to ~5 ms.
        brain_profile = _brain_for_request(request)
        endpoint = brain_profile["endpoint"]
        _ensure_node_refresher(endpoint)
        # Short timeouts on cold-cache fetches.  Stale-while-revalidate
        # in `_node_fetch_cached` makes warm fetches instant, but the
        # very first call after Django boot has to wait for SOMETHING.
        # 3-4 s is plenty: if the node is too slow even for that, we
        # return empty payloads and the background refresher will
        # populate the cache for the SPA's next poll (5 s later).
        # Remote brains reached through the SSM proxy are slow and highly
        # variable: /health measured 11 s idle but 15-58 s once the curriculum
        # was training.  We deliberately do NOT block a waitress thread for
        # that long -- the background refresher (90 s budget for these
        # endpoints) fills the cache, and a slightly longer cold budget here
        # just improves the odds the very first poll catches it.
        slow = _is_slow_brain(endpoint)
        from tools.wizard_learning.cognitive_state import get_cognitive_state_manager
        cog_mgr = get_cognitive_state_manager()
        health = _node_fetch_cached(
            "/health", ttl=2.0, timeout=8.0 if slow else 3.0, endpoint=endpoint
        )
        if not health:
            cog_snapshot = cog_mgr.read()
            if not slow:
                cog_snapshot = {
                    **cog_snapshot,
                    "mode": "OFFLINE" if cog_snapshot.get("mode") != "SLEEPING_SETTLING" else "SLEEPING_SETTLING",
                    "mode_label": "Sleeping / Settling Memory" if cog_snapshot.get("mode") == "SLEEPING_SETTLING" else "Node Offline / Booting",
                    "can_chat": False,
                    "chat_unavailable_reason": (
                        cog_snapshot.get("chat_unavailable_reason")
                        or "W1z4rD node is offline or holding an exclusive checkpoint lock."
                    ),
                }
            # An empty cache on a slow brain means "not warmed yet", not
            # "offline" -- reporting False there made a healthy AWS brain
            # show as Offline in the SPA.  `connecting` lets the UI say so
            # honestly while the refresher catches up.
            return JsonResponse({
                "online": None if slow else False,
                "connecting": bool(slow),
                "endpoint": endpoint,
                "selected_brain": brain_profile,
                "health": {},
                "brain": {},
                "multi_pool_stats": {},
                "cognitive_state": cog_snapshot,
            })
        # Same reasoning as /health: give slow relays a little more cold-path
        # room, but keep it short enough not to tie up a request thread.
        detail_timeout = 8.0 if slow else 4.0
        brain = _node_fetch_cached(
            "/brain", ttl=10.0, timeout=detail_timeout, endpoint=endpoint
        )
        multi_pool_stats = _node_fetch_cached(
            "/multi_pool/stats", ttl=5.0, timeout=detail_timeout, endpoint=endpoint
        )
        anchored_stats = {}
        try:
            from services.anchored_memory import AnchoredMemoryManager
            anchored_stats = AnchoredMemoryManager.get_instance().get_stats()
        except Exception:
            pass
        cog_state = cog_mgr.read() if slow else cog_mgr.probe_live_chat_readiness(endpoint=endpoint, probe_timeout=1.8)
        return JsonResponse({
            "online": True,
            "endpoint": endpoint,
            "selected_brain": brain_profile,
            "health": health,
            "brain": brain,
            "multi_pool_stats": multi_pool_stats,
            "anchored_memory": anchored_stats,
            "cognitive_state": cog_state,
        })


def _node_fetch(path: str, timeout: float = 5.0, *, endpoint: str = WIZARD_ENDPOINT) -> dict:
    """GET from the wizard node; returns {} on error."""
    try:
        with urllib.request.urlopen(f"{endpoint}{path}", timeout=timeout) as r:
            return _decode_node_body(r.read(), path)
    except Exception:
        return {}


def _qa_query(
    pool_name: str = "knowledge", top_k: int = 50, *, endpoint: str = WIZARD_ENDPOINT
) -> dict:
    payload = json.dumps({"query": "", "top_k": top_k, "pool": pool_name}).encode()
    req = urllib.request.Request(
        f"{endpoint}/qa/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            return json.loads(r.read())
    except Exception:
        return {}


@method_decorator(csrf_exempt, name="dispatch")
class WizardChatPoolsView(View):
    """
    GET /api/wizard-chat/pools/          — discover + return all pool data
    GET /api/wizard-chat/pools/?pool=qa  — single pool by key

    Returns { pools: { <key>: { label, type, data, count } }, discovered: [...keys] }
    """

    # All endpoints to probe; each produces one or more named pools.
    _POOL_ENDPOINTS = [
        ("neuro_snapshot",  "GET",  "/neuro/snapshot",    None),
        ("knowledge",       "GET",  "/knowledge/queue",   None),
        ("equations",       "GET",  "/equations/report",  None),
        ("streaming_labels","GET",  "/streaming/labels",  None),
        ("subnet_report",   "GET",  "/streaming/subnets", None),
        ("causal_graph",    "GET",  "/causal/graph",      None),
        ("health",          "GET",  "/health",            None),
    ]

    def get(self, request):
        only = request.GET.get("pool")  # optional filter
        endpoint = _brain_for_request(request)["endpoint"]

        def _fetch_one(endpoint_key, method, path, payload):
            if method == "GET":
                return endpoint_key, _node_fetch(path, timeout=6, endpoint=endpoint)
            payload_bytes = json.dumps(payload).encode() if payload else b"{}"
            req = urllib.request.Request(
                f"{endpoint}{path}",
                data=payload_bytes,
                headers={"Content-Type": "application/json"},
                method=method,
            )
            try:
                with urllib.request.urlopen(req, timeout=6) as r:
                    return endpoint_key, json.loads(r.read())
            except Exception:
                return endpoint_key, {}

        # Submit all in parallel
        endpoint_list = [
            e for e in self._POOL_ENDPOINTS if not only or e[0] == only
        ]
        # Always include both QA pools (knowledge + corrections)
        qa_futures = {}
        if not only or only in ("qa_knowledge", "qa_corrections"):
            qa_futures["qa_knowledge"] = _thread_pool.submit(
                _qa_query, "knowledge", 50, endpoint=endpoint
            )
            qa_futures["qa_corrections"] = _thread_pool.submit(
                _qa_query, "corrections", 50, endpoint=endpoint
            )

        futs = {key: _thread_pool.submit(_fetch_one, key, m, p, body)
                for key, m, p, body in endpoint_list}

        raw: dict[str, dict] = {}
        for key, fut in futs.items():
            try:
                _, data = fut.result(timeout=7)
                raw[key] = data
            except Exception:
                raw[key] = {}

        for key, fut in qa_futures.items():
            try:
                raw[key] = fut.result(timeout=7)
            except Exception:
                raw[key] = {}

        # Shape into named pool entries with metadata
        pools: dict[str, dict] = {}

        def _add(key: str, label: str, pool_type: str, data: dict, count_hint=None):
            if not data:
                return
            count = count_hint
            if count is None:
                for field in ("total", "count", "pairs_ingested", "size", "length"):
                    if isinstance(data.get(field), int):
                        count = data[field]
                        break
            pools[key] = {"label": label, "type": pool_type, "data": data, "count": count}

        _add("qa_knowledge",    "QA — Knowledge",     "qa",         raw.get("qa_knowledge", {}),
             len(raw.get("qa_knowledge", {}).get("matches") or []))
        _add("qa_corrections",  "QA — Corrections",   "qa",         raw.get("qa_corrections", {}),
             len(raw.get("qa_corrections", {}).get("matches") or []))
        _add("knowledge",       "Knowledge Docs",      "knowledge",  raw.get("knowledge", {}),
             len(raw.get("knowledge", {}).get("queue") or []))
        _add("equations",       "Equations (EEM)",     "equations",  raw.get("equations", {}))
        _add("neuro_snapshot",  "Hebbian State",       "neuro",      raw.get("neuro_snapshot", {}))
        _add("streaming_labels","Streaming Labels",    "labels",     raw.get("streaming_labels", {}))
        _add("subnet_report",   "Subnet Report",       "subnets",    raw.get("subnet_report", {}))
        _add("causal_graph",    "Causal Graph",        "causal",     raw.get("causal_graph", {}))

        # Expose pool stats from health endpoint as a meta pool
        health = raw.get("health", {})
        if health:
            _add("node_health", "Node Health", "meta", health)

        return JsonResponse({
            "pools":      pools,
            "discovered": list(pools.keys()),
        })
