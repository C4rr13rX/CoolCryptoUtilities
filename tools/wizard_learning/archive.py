"""Bounded Internet Archive discovery. Collection availability is not a reuse grant."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

from .store import LearningStore

COLLECTION = "marionstokesvideo"

def discover_stokes(store: LearningStore, limit: int = 10) -> dict:
    if not 1 <= limit <= 25: raise ValueError("Use 1–25 results per discovery batch.")
    query = urllib.parse.urlencode({"q": f"collection:{COLLECTION} AND mediatype:movies", "output": "json", "rows": limit,
                                   "fl[]": ["identifier", "title", "licenseurl", "rights"]}, doseq=True)
    with urllib.request.urlopen("https://archive.org/advancedsearch.php?" + query, timeout=15) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000: raise ValueError("Archive discovery response is too large")
    data = json.loads(raw)
    entries = []
    for item in data.get("response", {}).get("docs", [])[:limit]:
        identifier = item.get("identifier", "")
        if not isinstance(identifier, str) or not identifier or len(identifier) > 200: continue
        page = "https://archive.org/details/" + urllib.parse.quote(identifier, safe="")
        payload = {"title": str(item.get("title", identifier))[:200], "source_page": page,
                   "collection": COLLECTION, "reported_license": item.get("licenseurl"), "reported_rights": item.get("rights"),
                   "rights_status": "unverified", "real_footage_status": "needs-segment-review",
                   "note": "TV recordings may include cartoons, advertisements, copyrighted inserts and inaccurate claims. No collection-wide commercial-training clearance is inferred."}
        entries.append({"id": store.enqueue("source-discovery", payload), "source_page": page, "admitted": False})
    return {"collection": "https://archive.org/details/" + COLLECTION, "discovered": entries,
            "rights_policy": "https://archivesupport.zendesk.com/hc/en-us/articles/360014759692-Rights", "learning": False}
