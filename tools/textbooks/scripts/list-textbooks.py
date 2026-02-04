import os
import json
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")

import django

django.setup()

from core.models import TextbookSource


def main():
    out = []
    for tb in TextbookSource.objects.all().order_by("-downloaded_at"):
        out.append({
            "title": tb.title,
            "print_id": tb.print_id,
            "url": tb.url,
            "file_path": tb.file_path,
            "citation_apa": tb.citation_apa,
        })
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
