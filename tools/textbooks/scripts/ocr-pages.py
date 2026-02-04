import os
import re
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")

import django

django.setup()

from core.models import TextbookSource, TextbookPage

try:
    import pytesseract
    from PIL import Image
except Exception as exc:
    print(f"Missing OCR deps: {exc}")
    raise SystemExit(1)


def parse_page_num(name: str) -> int:
    m = re.search(r"page-(\d+)", name)
    return int(m.group(1)) if m else 0


def main():
    base_dir = Path(os.getenv("TEXTBOOKS_DIR", ".")).resolve()
    output_dir = base_dir / "textbooks" / "output"
    if not output_dir.exists():
        print("textbooks/output not found")
        return 1

    for book_dir in output_dir.iterdir():
        if not book_dir.is_dir():
            continue
        book_name = book_dir.name
        tb = TextbookSource.objects.filter(file_path__icontains=book_name).first()
        if not tb:
            # fallback by title matching
            tb = TextbookSource.objects.filter(title__icontains=book_name.replace("-full", "")).first()
        if not tb:
            print(f"No textbook source found for {book_name}; skipping")
            continue
        pngs = sorted(book_dir.glob("page-*.png"))
        for png in pngs:
            page_num = parse_page_num(png.name)
            if TextbookPage.objects.filter(textbook=tb, page_num=page_num).exists():
                continue
            try:
                text = pytesseract.image_to_string(Image.open(png))
                conf = None
                TextbookPage.objects.create(
                    textbook=tb,
                    page_num=page_num,
                    image_path=str(png),
                    ocr_text=text,
                    ocr_confidence=conf,
                )
                print(f"OCR stored {book_name} page {page_num}")
            except Exception as exc:
                print(f"OCR failed {book_name} page {page_num}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
