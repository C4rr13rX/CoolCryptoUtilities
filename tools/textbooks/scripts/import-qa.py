import os
import json
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")

import django

django.setup()

from core.models import TextbookSource, TextbookQACandidate


def main():
    base_dir = Path(os.getenv("TEXTBOOKS_DIR", ".")).resolve()
    qa_path = base_dir / "data" / "textbooks" / "qa_candidates.jsonl"
    if not qa_path.exists():
        print("qa_candidates.jsonl not found")
        return 1
    inserted = 0
    for line in qa_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        book_id = row.get("book_id") or ""
        tb = TextbookSource.objects.filter(file_path__icontains=book_id).first()
        if not tb:
            tb = TextbookSource.objects.filter(title__icontains=book_id.replace("_", " ")).first()
        if not tb:
            continue
        obj, created = TextbookQACandidate.objects.get_or_create(
            textbook=tb,
            page_num=int(row.get("page") or 0),
            question=row.get("question") or "",
            defaults={
                "answer": row.get("answer") or "",
                "confidence": float(row.get("confidence") or 0.0),
                "metadata": row,
            },
        )
        if created:
            inserted += 1
    print(f"Imported {inserted} QA candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
