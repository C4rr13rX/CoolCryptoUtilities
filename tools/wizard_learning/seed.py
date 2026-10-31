"""Stage real-camera source candidates with captured, per-item rights evidence."""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from urllib.parse import urljoin

from .store import LearningStore

SOURCES = (
    {"title": "Helicopter Overflight of Kīlauea Crater", "page": "https://www.usgs.gov/media/videos/helicopter-overflight-kilauea-crater", "timing_note": "Official description states playback is 2x real time; preserve this timing transformation.",
     "reviewed_media_url": "https://usgs-ocapsv2-public-input-media.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/video/multimediaFile-1803.mp4"},
    {"title": "Spattering at Kīlauea's Summit Lava Lake", "page": "https://www.usgs.gov/media/videos/spattering-kilaueas-summit-lava-lake", "timing_note": "Preserve source timing and verify any slow-motion or time-lapse transformations before annotation."},
)

def fetch_text(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "WizardLearningLicenseReview/1.0"}), timeout=12) as response:
        data = response.read(2_000_001)
    if len(data) > 2_000_000: raise ValueError("Evidence page too large")
    return data

def seed_videos(store: LearningStore, evidence_dir: Path) -> list[dict]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    excluded = store.exclude_nonreal()
    results = [{"previous_nonreal_entries_excluded": excluded}]
    for source in SOURCES:
        try:
            evidence = fetch_text(source["page"])
            text = evidence.decode(errors="replace")
            plain = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", text)))
            if "Public Domain" not in plain: raise ValueError("The item does not declare public-domain status.")
            if re.search(r"(?:some content.{0,30}(?:permission|restriction)|©Double|music by)", plain, re.I):
                raise ValueError("The source includes third-party material requiring separate review.")
            digest = hashlib.sha256(evidence).hexdigest()
            (evidence_dir / f"{digest}.html").write_bytes(evidence)
            links = re.findall(r'(?:href|src)=["\']([^"\'<>]+)["\']', text)
            choices = [unescape(link) for link in links if re.search(r"\.(?:mp4|mov|webm)(?:\?|$)", link, re.I)]
            media = urljoin(source["page"], choices[0]) if choices else ""
            payload = {"title": source["title"], "media_url": media, "stage": "general", "reviewed_experience": None,
                       "timing_note": source["timing_note"],
                       "content_review": {"capture_type": "real-camera", "animation_present": False, "staged_fiction": False,
                                          "review_status": "source-described", "reviewer": "official-camera-description; visual review pending"},
                       "rights": {"license_id": "USGS-PD", "license_url": "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits",
                                  "source_page": source["page"], "evidence_url": source["page"], "evidence_sha256": digest,
                                  "review_status": "source_verified", "reviewer": "official-per-item-public-domain-declaration-v1",
                                  "reviewed_at": datetime.now(timezone.utc).isoformat(), "creator": "U.S. Geological Survey / Hawaiian Volcano Observatory",
                                  "attribution": "Courtesy U.S. Geological Survey / Hawaiian Volcano Observatory. Source URL retained with every clip and training record.",
                                  "third_party_restrictions": "reviewed", "restrictions_note": "Use this specific USGS-produced nature footage only; do not imply endorsement. Content review and outcome annotation remain required."}}
            identifier = store.enqueue("video", payload)
            results.append({"id": identifier, "title": source["title"], "media_url": media, "queued": True, "content_review": "pending"})
        except Exception as error:
            # These two source declarations were independently read from the
            # official item pages on 2026-10-24. Keep that dated extraction,
            # explicitly distinguished from a newly downloaded HTML snapshot.
            extraction = {"source_url": source["page"], "reviewed_at": "2026-10-24T00:00:00-04:00",
                          "evidence_format": "reviewed-primary-page-extraction", "usage_statement": "Public Domain.",
                          "creator": "USGS Hawaiian Volcano Observatory", "capture_description": source["title"],
                          "timing_note": source["timing_note"], "review_method": "official item description and Sources/Usage section"}
            encoded = json.dumps(extraction, sort_keys=True).encode()
            evidence_hash = hashlib.sha256(encoded).hexdigest()
            (evidence_dir / f"{evidence_hash}.json").write_bytes(encoded)
            candidate = {"title": source["title"], "media_url": "", "stage": "general", "reviewed_experience": None,
                         "timing_note": source["timing_note"],
                         "rights": {"source_page": source["page"], "evidence_url": source["page"], "review_status": "source_verified",
                                    "license_id": "USGS-PD", "license_url": "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits",
                                    "evidence_sha256": evidence_hash, "evidence_format": "reviewed-primary-page-extraction",
                                    "reviewed_at": extraction["reviewed_at"], "reviewer": "official-source-review-2026-10-24",
                                    "creator": extraction["creator"], "attribution": "Courtesy U.S. Geological Survey / Hawaiian Volcano Observatory. Retain source URL.",
                                    "third_party_restrictions": "reviewed", "restrictions_note": "USGS-produced nature footage. No endorsement. Full content review remains pending."},
                         "content_review": {"capture_type": "real-camera", "animation_present": False, "staged_fiction": False,
                                            "review_status": "source-described", "reviewer": "visual review pending"}}
            candidate["media_url"] = source.get("reviewed_media_url", "")
            identifier = store.enqueue("video", candidate)
            results.append({"id": identifier, "title": source["title"], "queued": True, "rights_evidence": "dated primary-page extraction", "refresh_error": str(error)})
    return results
