"""Bounded local review assets for licensed real-camera queue entries.

Review downloads never reach a learning endpoint. Decoding is optional and uses
the local OpenCV/Pillow installation, not an online vision or captioning service.
"""
from __future__ import annotations

import hashlib
import base64
import json
import math
import os
import urllib.request
from pathlib import Path

from .rights import license_blockers
from .store import LearningStore


def review_video(store: LearningStore, identifier: str, output_dir: Path, *, max_bytes: int = 128 * 1024 * 1024) -> dict:
    with store.connect() as db:
        row = db.execute("SELECT kind,payload,state FROM queue WHERE id=?", (identifier,)).fetchone()
    if not row or row["kind"] != "video" or row["state"] != "queued": raise ValueError("Select a queued video")
    payload = json.loads(row["payload"])
    problems = [reason for reason in license_blockers(payload) if reason != "The footage still needs a real-world-content review before training."]
    if problems: raise PermissionError("Video cannot be downloaded for review: " + "; ".join(problems))
    output_dir = output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / (identifier + ".part")
    target = output_dir / (identifier + ".mp4")
    if not target.exists():
        try:
            request = urllib.request.Request(payload["media_url"], headers={"User-Agent": "WizardLearningVideoReview/1.0"})
            with urllib.request.urlopen(request, timeout=15) as response, temporary.open("wb") as stream:
                length = response.headers.get("Content-Length")
                if length and int(length) > max_bytes: raise ValueError("Video exceeds the bounded review download limit")
                total = 0
                while True:
                    block = response.read(65536)
                    if not block: break
                    total += len(block)
                    if total > max_bytes: raise ValueError("Video exceeds the bounded review download limit")
                    stream.write(block)
            os.replace(temporary, target)
        finally:
            if temporary.exists(): temporary.unlink()
    if target.stat().st_size > max_bytes: raise ValueError("Existing review file exceeds limit")
    import cv2
    from PIL import Image, ImageDraw
    capture = cv2.VideoCapture(str(target))
    if not capture.isOpened(): raise ValueError("Could not decode the video")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS); count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count <= 0: raise ValueError("Invalid video timing")
        duration = count / fps
        if duration > 300: raise ValueError("Review one clip of at most five minutes at a time")
        width, height = 240, 150
        sheet = Image.new("RGB", (6 * width, 4 * (height + 22)), "#111018")
        draw = ImageDraw.Draw(sheet)
        timestamps = []
        for index in range(24):
            stamp = min(duration - 1 / fps, index * duration / 24)
            capture.set(cv2.CAP_PROP_POS_MSEC, stamp * 1000)
            ok, frame = capture.read()
            if not ok: continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((width, height))
            x, y = index % 6 * width, index // 6 * (height + 22)
            sheet.paste(image, (x + (width - image.width) // 2, y))
            draw.text((x + 4, y + height + 3), f"{stamp:.2f} sec", fill="white")
            timestamps.append(stamp)
        if len(timestamps) < 12: raise ValueError("Insufficient decoded review frames")
        contact = output_dir / (identifier + "-contact.png"); sheet.save(contact)
    finally: capture.release()
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"queue_id": identifier, "video": str(target), "sha256": digest, "contact_sheet": str(contact),
              "duration_seconds": duration, "sampled_timestamps": timestamps, "timing_note": payload.get("timing_note"),
              "source_page": payload["rights"]["source_page"], "source_license_evidence_sha256": payload["rights"]["evidence_sha256"],
              "review_status": "needs-review", "learning": False}
    (output_dir / (identifier + "-review.json")).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def confirm_video_review(store: LearningStore, identifier: str, review: dict) -> None:
    """Persist an operator/agent review after inspecting source and decoded frames."""
    from .curriculum import IDS
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip(): raise ValueError("Reviewer is required")
    if review.get("capture_type") != "real-camera" or review.get("animation_present") is not False or review.get("staged_fiction") is not False:
        raise ValueError("The reviewed segment must be real camera footage")
    if set(review.get("checked_prerogatives", [])) != set(IDS) or review.get("conflicts") != [] or review.get("outcome_verified") is not True:
        raise ValueError("Verify the observations and assess all twelve prerogatives together")
    if not isinstance(review.get("observed_outcome"), dict) or not review["observed_outcome"]: raise ValueError("Record the verified visual observations")
    path = Path(review["video"]).resolve()
    with path.open("rb") as stream: digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != review.get("sha256"): raise ValueError("Reviewed video digest mismatch")
    start, end = review.get("start_seconds"), review.get("end_seconds")
    if not all(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) for value in (start, end)) or not 0 <= start < end <= 300:
        raise ValueError("Review one bounded segment of at most five minutes")
    with store.connect() as db:
        row = db.execute("SELECT kind,payload,state FROM queue WHERE id=?", (identifier,)).fetchone()
        if not row or row["kind"] != "video" or row["state"] != "queued": raise ValueError("Select a queued video")
        payload = json.loads(row["payload"])
        payload["content_review"] = {"capture_type": "real-camera", "animation_present": False, "staged_fiction": False,
                                     "review_status": "verified", "reviewer": review["reviewer"], "method": review.get("method", "operator review"),
                                     "start_seconds": start, "end_seconds": end, "video_sha256": digest}
        payload["reviewed_experience"] = review
        payload["next_seconds"] = start
        db.execute("UPDATE queue SET payload=? WHERE id=?", (json.dumps(payload, allow_nan=False), identifier))


def train_video_batch(store: LearningStore, identifier: str, client, checkpoint: str,
                      *, video_pool: int = 12, context_pool: int = 1, outcome_pool: int = 4, max_frames: int = 4) -> dict:
    """Actual quantized camera frames -> multisensory consolidation, after admission.

    Every bounded batch consumes the assessment; another batch needs new retention
    evidence. A lost acknowledgement quarantines the item instead of blind replay.
    """
    import cv2
    from .brain import encode
    from .curriculum import MANIFEST
    from .store import brain_key
    if not 1 <= max_frames <= 8 or len({video_pool, context_pool, outcome_pool}) != 3: raise ValueError("Use separate pools and 1–8 frames per batch")
    key = brain_key(client.profile)
    rows = store.items(key, checkpoint, limit=200)
    item = next((item for item in rows if item["id"] == identifier), None)
    if not item or not item["eligible"]: raise PermissionError("Video is not admitted: " + "; ".join(item["blockers"] if item else ["not found"]))
    payload = item["payload"]; review = payload["reviewed_experience"]
    path = Path(review["video"])
    with path.open("rb") as stream: digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != review["sha256"]: raise ValueError("Training video differs from the reviewed video")
    capture = cv2.VideoCapture(str(path)); frames = []; start = payload.get("next_seconds", review["start_seconds"])
    try:
        for index in range(max_frames):
            stamp = start + index * .5
            if stamp >= review["end_seconds"]: break
            capture.set(cv2.CAP_PROP_POS_MSEC, stamp * 1000); ok, frame = capture.read()
            if not ok: raise ValueError("Could not decode a reviewed training frame")
            frame = cv2.cvtColor(cv2.resize(frame, (128, 72)), cv2.COLOR_BGR2RGB)
            # Store each channel in two bits; use uint16 during arithmetic.
            levels = ((frame.astype('uint16') + 42) // 85).astype('uint8')
            packed = (levels[:, :, 0] << 4) | (levels[:, :, 1] << 2) | levels[:, :, 2]
            pixels = base64.urlsafe_b64encode(packed.tobytes()).decode().rstrip('=')
            frames.append({"timestamp_seconds": stamp, "width": 128, "height": 72, "encoding": "rgb222-u8-v1", "pixels": pixels})
    finally: capture.release()
    if not frames: raise ValueError("No unprocessed reviewed frames remain")
    store.admit_expansion(key, checkpoint, "reviewed real-video batch", queue_id=identifier)
    try:
        for frame in frames:
            result = client.post("/brain/consolidate/multi", {"streams": [
                {"pool_id": video_pool, "frame": encode(frame)},
                {"pool_id": context_pool, "frame": encode({"core_prerogatives": MANIFEST, "source": payload["rights"]["source_page"], "video_sha256": digest, "timing_note": payload.get("timing_note"), "timestamp_seconds": frame["timestamp_seconds"]})}],
                "outcome_pool": outcome_pool, "outcome_frame": encode(review["observed_outcome"])})
            if result.get("consolidated") is not True: raise RuntimeError("Brain did not acknowledge consolidation")
        payload["next_seconds"] = frames[-1]["timestamp_seconds"] + .5
        state = "completed" if payload["next_seconds"] >= review["end_seconds"] else "queued"
        with store.connect() as db: db.execute("UPDATE queue SET state=?,payload=? WHERE id=?", (state, json.dumps(payload), identifier))
        return {"frames_consolidated": len(frames), "next_seconds": payload["next_seconds"], "state": state, "retention_reassessment_required": True}
    except Exception as error:
        with store.connect() as db: db.execute("UPDATE queue SET state='uncertain',error=? WHERE id=?", (str(error)[:400], identifier))
        raise
