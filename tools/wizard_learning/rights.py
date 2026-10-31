"""Conservative commercial-training intake policy with source-specific evidence."""
from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlsplit

ALLOWED_LICENSES = {
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY-3.0": "https://creativecommons.org/licenses/by/3.0/",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "USGS-PD": "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits",
}

def https_url(value) -> bool:
    if not isinstance(value, str): return False
    try:
        parsed = urlsplit(value)
        return parsed.scheme == "https" and bool(parsed.hostname) and not parsed.username and not parsed.password and not parsed.fragment
    except ValueError: return False

def license_blockers(item: dict, *, now: datetime | None = None) -> list[str]:
    rights = item.get("rights")
    if not isinstance(rights, dict): return ["License provenance is missing."]
    errors = []
    license_id = rights.get("license_id")
    if license_id not in ALLOWED_LICENSES:
        errors.append("This queue accepts CC0, CC BY 3.0/4.0, or specifically verified USGS public-domain footage; other grants require review.")
    elif rights.get("license_url") != ALLOWED_LICENSES[license_id]:
        errors.append("License identifier and license URL disagree.")
    for field in ("source_page", "evidence_url"):
        if not https_url(rights.get(field)): errors.append(f"A valid HTTPS {field} is required.")
    if not rights.get("creator") or not rights.get("attribution"):
        errors.append("Creator and attribution are required.")
    if rights.get("review_status") != "source_verified" or not rights.get("reviewer"):
        errors.append("Source-specific license evidence has not been reviewed.")
    digest = rights.get("evidence_sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        errors.append("A SHA-256 digest of the captured license evidence is required.")
    try:
        reviewed = datetime.fromisoformat(rights.get("reviewed_at", "").replace("Z", "+00:00"))
        if reviewed.tzinfo is None: raise ValueError("Timezone required")
        age = ((now or datetime.now(timezone.utc)) - reviewed).total_seconds()
        if age < -300 or age > 30 * 86400: errors.append("Recheck license provenance; the review is stale or future-dated.")
    except (ValueError, TypeError, AttributeError): errors.append("A dated license review is required.")
    if rights.get("third_party_restrictions") != "reviewed":
        errors.append("Underlying works, music, likeness, and source-specific exclusions need review.")
    if not https_url(item.get("media_url")): errors.append("A direct HTTPS media URL is required.")
    content = item.get("content_review")
    if not isinstance(content, dict) or content.get("capture_type") != "real-camera" or content.get("animation_present") is not False or content.get("staged_fiction") is not False:
        errors.append("Only real camera footage is permitted; cartoons, animation, synthetic footage, and staged fiction are excluded.")
    elif content.get("review_status") != "verified" or not content.get("reviewer"):
        errors.append("The footage still needs a real-world-content review before training.")
    return errors
