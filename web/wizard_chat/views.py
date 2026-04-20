"""wizard_chat/views.py — API backend for the W1z4rD Vision multimodal chat."""
from __future__ import annotations

import concurrent.futures
import io
import json
import os
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

WIZARD_ENDPOINT = os.getenv("WIZARD_NODE_URL", "http://localhost:8090").rstrip("/")
WIZARD_TIMEOUT = float(os.getenv("WIZARD_TIMEOUT_S", "8"))
WEB_SEARCH_TIMEOUT = float(os.getenv("WIZARD_WEB_SEARCH_TIMEOUT_S", "3"))
MAX_UPLOAD_MB = int(os.getenv("WIZARD_CHAT_MAX_MB", "50"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wizard_ask(text: str, session_id: str = "") -> dict:
    """POST to /neuro/pipeline; always returns a dict."""
    payload = json.dumps({
        "text": text,
        "session_id": session_id or str(uuid.uuid4()),
        "hops": 3,
        "top_k": 30,
    }).encode()
    req = urllib.request.Request(
        f"{WIZARD_ENDPOINT}/neuro/pipeline",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=WIZARD_TIMEOUT) as resp:
            data = json.loads(resp.read())
            # Normalise pipeline response shape to match what the view expects.
            # /neuro/pipeline returns `answer` at top level; older /neuro/ask
            # returned `answer` inside `activated_concepts` — keep compat.
            if "answer" not in data:
                data["answer"] = ""
            if "confidence_tier" not in data:
                data["confidence_tier"] = data.get("confidence_tier", "low")
            if "activated_concepts" not in data:
                data["activated_concepts"] = []
            return data
    except urllib.error.URLError as exc:
        return {"error": str(exc), "answer": "", "hypothesis": True,
                "confidence_tier": "offline", "activated_concepts": []}
    except Exception as exc:
        return {"error": str(exc), "answer": "", "hypothesis": True,
                "confidence_tier": "error", "activated_concepts": []}


def _wizard_train(question: str, answer: str, session_id: str = "") -> dict:
    """POST a corrected QA pair to /qa/ingest for training."""
    candidate = {
        "qa_id": str(uuid.uuid4()),
        "question": question,
        "answer": answer,
        "book_id": f"wizard_chat_{session_id[:8] if session_id else 'manual'}",
        "confidence": 0.95,
        "evidence": "User-corrected answer from wizard chat",
        "review_status": "approved",
    }
    payload = json.dumps({"candidates": [candidate]}).encode()
    req = urllib.request.Request(
        f"{WIZARD_ENDPOINT}/qa/ingest",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return {"ok": True, "response": json.loads(resp.read())}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


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

        # Images — no vision on the wizard node yet
        if name_lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg")):
            return f"[Image file: {name}, {len(data):,} bytes — image analysis not yet available on this node]"

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

    # Good answer from node
    if raw_answer and not is_hypothesis:
        return {"answer": raw_answer, "is_hypothesis": False,
                "confidence_tier": tier, "web_used": False, "concepts": concepts}

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
        attachment_texts: list[str] = body.get("attachment_texts") or []

        if not text and not attachment_texts:
            return JsonResponse({"error": "No message content"}, status=400)

        # Build full prompt
        parts = []
        if attachment_texts:
            for i, at in enumerate(attachment_texts, 1):
                parts.append(f"[Attachment {i}]\n{at[:6000]}")
        if text:
            parts.append(text)
        full_prompt = "\n\n".join(parts)

        wizard_data = _wizard_ask(full_prompt, session_id)

        # Derive online status from the response — no separate health ping needed.
        node_up = wizard_data.get("confidence_tier") not in ("offline", "error")

        # Auto web-search when hypothesis/offline — run in a thread so it can't
        # block longer than WEB_SEARCH_TIMEOUT seconds regardless of DDG latency.
        web_snippet = ""
        is_hypothesis = bool(wizard_data.get("hypothesis", False)) or not (wizard_data.get("answer") or "").strip()
        if is_hypothesis and node_up:
            search_query = text if text else full_prompt[:200]
            fut = _thread_pool.submit(_web_search, search_query)
            try:
                web_snippet = fut.result(timeout=WEB_SEARCH_TIMEOUT)
            except Exception:
                web_snippet = ""

        result = _build_answer(wizard_data, web_snippet, full_prompt)
        result["node_online"] = node_up
        result["session_id"] = session_id
        return JsonResponse(result)


@method_decorator(csrf_exempt, name="dispatch")
class WizardChatUploadView(View):
    """
    POST /api/wizard-chat/upload/  (multipart/form-data)
    Field: files[]  (multiple files allowed)
    Returns: [{name, text, size, type}] — extracted text per file
    """

    def post(self, request):
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
                results.append({
                    "name": f.name,
                    "text": text,
                    "size": size,
                    "type": f.content_type or "",
                    "error": False,
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
    Body: {question, answer, session_id?}
    Ingest a corrected answer into the wizard node's training data.
    """

    def post(self, request):
        try:
            body = json.loads(request.body)
        except Exception:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        question = (body.get("question") or "").strip()
        answer = (body.get("answer") or "").strip()
        session_id = (body.get("session_id") or "")

        if not question or not answer:
            return JsonResponse({"error": "Both question and answer are required"}, status=400)

        result = _wizard_train(question, answer, session_id)
        return JsonResponse(result)


@method_decorator(csrf_exempt, name="dispatch")
class WizardChatStatusView(View):
    """GET /api/wizard-chat/status/ — node health + training stats."""

    def get(self, request):
        try:
            with urllib.request.urlopen(f"{WIZARD_ENDPOINT}/health", timeout=5) as r:
                health = json.loads(r.read())
            return JsonResponse({
                "online": True,
                "endpoint": WIZARD_ENDPOINT,
                "health": health,
            })
        except Exception as exc:
            return JsonResponse({
                "online": False,
                "endpoint": WIZARD_ENDPOINT,
                "error": str(exc),
                "health": {},
            })
