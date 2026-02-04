#!/usr/bin/env python3
"""
Build KnowledgeDocument entries from OCR textbook pages and enqueue them.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_django() -> None:
    if not os.getenv("DJANGO_SETTINGS_MODULE"):
        os.environ["DJANGO_SETTINGS_MODULE"] = "coolcrypto_dashboard.settings"
    import django
    django.setup()


def _build_default_citation(source) -> str:
    if source.citation_apa:
        return source.citation_apa
    author = source.authors or "Unknown"
    year = source.year or "n.d."
    title = source.title or "Untitled"
    publisher = source.publisher or ""
    url = source.url or ""
    parts = [f"{author} ({year}). {title}."]
    if publisher:
        parts.append(publisher + ".")
    if url:
        parts.append(url)
    return " ".join(parts).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build KnowledgeDocument rows from textbook OCR pages.")
    parser.add_argument("--limit", type=int, default=500, help="Max pages to ingest in this run.")
    parser.add_argument("--min-chars", type=int, default=200, help="Minimum OCR chars to store.")
    parser.add_argument("--status", default="pending", help="Queue status to set.")
    parser.add_argument("--visual-queue", action="store_true", help="Also enqueue VisualLabelQueueItem entries.")
    args = parser.parse_args()

    _ensure_django()
    from core.models import TextbookPage, KnowledgeDocument, KnowledgeQueueItem, VisualLabelQueueItem

    pages = TextbookPage.objects.select_related("textbook").order_by("id")
    count = 0
    created = 0
    queued = 0
    for page in pages:
        if count >= args.limit:
            break
        text = (page.ocr_text or "").strip()
        if len(text) < args.min_chars:
            continue
        source = page.textbook
        title = f"{source.title} p{page.page_num}"
        doc, was_created = KnowledgeDocument.objects.get_or_create(
            source=f"textbook:{source.id}",
            title=title,
            defaults={
                "abstract": "",
                "body": text,
                "url": source.url or "",
                "citation_apa": _build_default_citation(source),
                "metadata": {
                    "textbook_id": source.id,
                    "page_num": page.page_num,
                    "image_path": page.image_path,
                    "sha256": source.sha256,
                },
            },
        )
        if not was_created and doc.body != text:
            doc.body = text
            doc.metadata = doc.metadata or {}
            doc.metadata.update({"page_num": page.page_num, "image_path": page.image_path})
            doc.save(update_fields=["body", "metadata"])
        if was_created:
            created += 1
        KnowledgeQueueItem.objects.get_or_create(
            document=doc,
            defaults={"status": args.status, "confidence": 0.0, "label": ""},
        )
        if args.visual_queue and page.image_path:
            VisualLabelQueueItem.objects.get_or_create(
                document=doc,
                image_path=page.image_path,
                defaults={"status": args.status, "confidence": 0.0, "label": ""},
            )
        queued += 1
        count += 1
    print(f"processed={count} created={created} queued={queued}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
