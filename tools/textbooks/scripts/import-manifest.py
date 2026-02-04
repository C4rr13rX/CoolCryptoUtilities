import hashlib
import json
import os
from pathlib import Path
from datetime import datetime

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")

import django

django.setup()

from core.models import TextbookSource


def apa_citation(title, authors, year, publisher, url):
    author_part = authors.strip() or "Unknown Author"
    year_part = str(year) if year else "n.d."
    title_part = title.strip() or "Untitled"
    publisher_part = publisher.strip() or "LibreTexts"
    url_part = url.strip()
    if url_part:
        return f"{author_part}. ({year_part}). {title_part}. {publisher_part}. {url_part}"
    return f"{author_part}. ({year_part}). {title_part}. {publisher_part}."


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(manifest_path: Path):
    if not manifest_path.exists():
        return []
    rows = []
    for line in manifest_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def main():
    base_dir = Path(os.getenv("TEXTBOOKS_DIR", ".")).resolve()
    manifest = base_dir / "textbooks" / "manifest.jsonl"
    rows = load_manifest(manifest)
    if not rows:
        print("manifest.jsonl not found or empty")
        return 1
    inserted = 0
    for row in rows:
        title = row.get("title") or row.get("printId") or "Untitled"
        print_id = row.get("printId") or ""
        url = row.get("sourceUrl") or row.get("href") or ""
        file_path = row.get("filePath") or ""
        publisher = "LibreTexts"
        authors = row.get("authors") or ""
        year = row.get("year")
        if file_path:
            pdf_path = Path(file_path)
            if not pdf_path.is_absolute():
                pdf_path = (base_dir / file_path).resolve()
            if pdf_path.exists():
                row["sha256"] = sha256_file(pdf_path)
                file_path = str(pdf_path)
        citation = apa_citation(title, authors, year, publisher, url)
        obj, created = TextbookSource.objects.get_or_create(
            print_id=print_id,
            defaults={
                "title": title,
                "authors": authors,
                "year": year,
                "publisher": publisher,
                "url": url,
                "file_path": file_path,
                "sha256": row.get("sha256", ""),
                "citation_apa": citation,
                "metadata": row,
            },
        )
        if not created:
            # update missing fields deterministically
            changed = False
            for field, value in {
                "title": title,
                "authors": authors,
                "year": year,
                "publisher": publisher,
                "url": url,
                "file_path": file_path,
                "sha256": row.get("sha256", ""),
                "citation_apa": citation,
            }.items():
                if value and not getattr(obj, field):
                    setattr(obj, field, value)
                    changed = True
            if changed:
                obj.save()
        else:
            inserted += 1
    print(f"Imported {inserted} new textbooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
