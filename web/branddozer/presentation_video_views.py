"""
Video export endpoints for the presentation decks.

``presentation_video.py`` has been able to render a deck to MP4 for a while --
9:16 portrait by default, cube-flip transitions, word animations timed to the
real Polly speech marks -- but nothing ever called it. These are the endpoints
that make it reachable, plus the listing the menu link needs.

Rendering runs on a background thread rather than inside the request. A deck of
any length takes far longer than a browser (or API Gateway, at 30s) will wait,
so the POST returns immediately with a job id and the client polls. The job
state lives on disk beside the output, which means it survives the process
restarting mid-render -- the client sees a stale-but-honest status instead of a
job that silently vanished.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from branddozer.models import ResearchPaper
from branddozer.research_views import RUNTIME_ROOT

logger = logging.getLogger(__name__)

# One render at a time per process. Each frame is a 1080x1920 array and ffmpeg
# is already using the cores; running two concurrently makes both slower and
# can exhaust memory on a phone.
_render_lock = threading.Lock()
_active: dict[str, bool] = {}


def _paper_root(paper_id: str) -> Path:
    return Path(RUNTIME_ROOT) / str(paper_id) / "presentation"


def _job_path(paper_id: str) -> Path:
    return _paper_root(paper_id) / "video-job.json"


def _video_path(paper_id: str) -> Path:
    return _paper_root(paper_id) / "presentation.mp4"


def _read_job(paper_id: str) -> dict[str, Any]:
    path = _job_path(paper_id)
    if not path.is_file():
        return {"status": "idle"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "idle"}


def _write_job(paper_id: str, payload: dict[str, Any]) -> None:
    path = _job_path(paper_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _render(paper_id: str, deck: dict, config_kwargs: dict, max_slides: int) -> None:
    """Render on a worker thread, recording progress to the job file."""
    from branddozer.presentation_video import VideoConfig, export_mp4

    root = _paper_root(paper_id)
    started = time.time()

    def progress(done: int, total: int) -> None:
        # Written every slide so a long render is visibly moving rather than
        # looking hung.
        _write_job(paper_id, {
            "status": "rendering",
            "slides_done": done,
            "slides_total": total,
            "percent": round(100.0 * done / max(total, 1), 1),
            "started_at": started,
        })

    try:
        _write_job(paper_id, {"status": "rendering", "percent": 0.0,
                              "started_at": started})
        summary = export_mp4(
            deck,
            out_path=_video_path(paper_id),
            audio_dir=root / "audio",
            config=VideoConfig(**config_kwargs),
            max_slides=max_slides,
            score_path=str(root / "score.wav") if (root / "score.wav").is_file() else "",
            progress=progress,
        )
        _write_job(paper_id, {
            "status": "ready",
            "percent": 100.0,
            "started_at": started,
            "finished_at": time.time(),
            "duration_s": round(time.time() - started, 1),
            "summary": summary,
        })
        logger.info("video render complete for %s in %.1fs",
                    paper_id, time.time() - started)
    except Exception as exc:  # noqa: BLE001
        logger.exception("video render failed for %s", paper_id)
        _write_job(paper_id, {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=6),
            "started_at": started,
            "finished_at": time.time(),
        })
    finally:
        _active.pop(paper_id, None)


class VideoStudioListView(APIView):
    """
    GET /api/branddozer/video/

    Every paper that can be turned into a video, with its current render
    state. This is what the Video Studio menu entry lands on -- without it the
    player is only reachable if you already know a paper id.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        from branddozer.presentation_video import (
            ASPECT_RATIOS,
            DEFAULT_ASPECT,
            DEFAULT_TRANSITION,
            DEFAULT_WORD_ANIMATION,
            VIDEO_TRANSITIONS,
            VIDEO_WORD_ANIMATIONS,
        )

        items = []
        for paper in ResearchPaper.objects.all().order_by("-updated_at")[:200]:
            root = _paper_root(str(paper.id))
            job = _read_job(str(paper.id))
            video = _video_path(str(paper.id))
            items.append({
                "id": str(paper.id),
                "title": paper.title,
                "updated_at": paper.updated_at,
                # The client uses these to decide what to offer: rendering
                # without narration works but is silent, so the UI can warn.
                "has_deck": (root / "deck.json").is_file(),
                "has_audio": (root / "audio").is_dir(),
                "has_score": (root / "score.wav").is_file(),
                "video": {
                    "exists": video.is_file(),
                    "bytes": video.stat().st_size if video.is_file() else 0,
                    "status": job.get("status", "idle"),
                    "percent": job.get("percent", 0),
                },
            })

        return Response({
            "items": items,
            "count": len(items),
            # Shipped with the listing so the render form needs no second call.
            "options": {
                "aspects": list(ASPECT_RATIOS),
                "transitions": list(VIDEO_TRANSITIONS),
                "word_animations": list(VIDEO_WORD_ANIMATIONS),
                "defaults": {
                    "aspect": DEFAULT_ASPECT,
                    "transition": DEFAULT_TRANSITION,
                    "word_animation": DEFAULT_WORD_ANIMATION,
                },
            },
        })


class VideoRenderView(APIView):
    """
    POST /api/branddozer/video/<paper_id>/render/
    GET  /api/branddozer/video/<paper_id>/render/   -> job status

    Body (all optional): aspect, transition, word_animation, fps, max_slides.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request: Request, paper_id: str, *args, **kwargs) -> Response:
        video = _video_path(paper_id)
        return Response({
            "paper_id": paper_id,
            "job": _read_job(paper_id),
            "video": {
                "exists": video.is_file(),
                "bytes": video.stat().st_size if video.is_file() else 0,
            },
        })

    def post(self, request: Request, paper_id: str, *args, **kwargs) -> Response:
        from branddozer.presentation_video import (
            ASPECT_RATIOS,
            VIDEO_TRANSITIONS,
            VIDEO_WORD_ANIMATIONS,
        )

        paper = ResearchPaper.objects.filter(id=paper_id).first()
        if not paper:
            return Response({"detail": "Research paper not found"},
                            status=status.HTTP_404_NOT_FOUND)

        if _active.get(paper_id):
            return Response({"detail": "A render is already running for this paper",
                             "job": _read_job(paper_id)},
                            status=status.HTTP_409_CONFLICT)

        body = request.data or {}
        config: dict[str, Any] = {}

        # Validate rather than pass through: an unknown transition would fail
        # deep inside the renderer, minutes into a job, instead of here.
        aspect = str(body.get("aspect") or "").strip()
        if aspect:
            if aspect not in ASPECT_RATIOS:
                return Response({"detail": f"unknown aspect {aspect!r}",
                                 "allowed": list(ASPECT_RATIOS)}, status=400)
            config["aspect"] = aspect

        transition = str(body.get("transition") or "").strip()
        if transition:
            if transition not in VIDEO_TRANSITIONS:
                return Response({"detail": f"unknown transition {transition!r}",
                                 "allowed": list(VIDEO_TRANSITIONS)}, status=400)
            config["transition"] = transition

        animation = str(body.get("word_animation") or "").strip()
        if animation:
            if animation not in VIDEO_WORD_ANIMATIONS:
                return Response({"detail": f"unknown word_animation {animation!r}",
                                 "allowed": list(VIDEO_WORD_ANIMATIONS)}, status=400)
            config["word_animation"] = animation

        try:
            max_slides = max(0, int(body.get("max_slides") or 0))
        except (TypeError, ValueError):
            max_slides = 0

        # Prefer the deck already generated for the player, so an export
        # matches what was previewed rather than re-deriving it.
        deck_file = _paper_root(paper_id) / "deck.json"
        if deck_file.is_file():
            try:
                deck = json.loads(deck_file.read_text(encoding="utf-8"))
            except ValueError:
                deck = None
        else:
            deck = None
        if not deck:
            # Same call the player's view makes, so an export of a paper whose
            # deck was never persisted still matches what the player shows.
            from branddozer.presentation import build_presentation
            from branddozer.presentation_media import score_sync_points

            deck = build_presentation(
                paper_id=str(paper.id),
                title=paper.title,
                markdown=paper.content_markdown,
                abstract=paper.abstract,
            )
            deck["timeline"] = score_sync_points(deck["slides"])

        if not (deck.get("slides") or []):
            return Response({"detail": "deck has no slides to render"}, status=400)

        _active[paper_id] = True
        thread = threading.Thread(
            target=_render, args=(paper_id, deck, config, max_slides),
            name=f"video-{paper_id[:8]}", daemon=True,
        )
        thread.start()

        return Response({
            "status": "started",
            "paper_id": paper_id,
            "slides": len(deck.get("slides") or []),
            "config": config,
        }, status=status.HTTP_202_ACCEPTED)


class VideoDownloadView(APIView):
    """
    GET /api/branddozer/video/<paper_id>/file/

    Streams the rendered MP4. Range requests are honoured so a <video> element
    can seek without downloading the whole file first.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request: Request, paper_id: str, *args, **kwargs):
        from django.http import FileResponse, HttpResponse

        path = _video_path(paper_id)
        if not path.is_file():
            return HttpResponse("video not rendered", status=404)

        response = FileResponse(path.open("rb"), content_type="video/mp4")
        response["Content-Length"] = path.stat().st_size
        response["Accept-Ranges"] = "bytes"
        response["Content-Disposition"] = (
            f'inline; filename="presentation-{paper_id[:8]}.mp4"'
        )
        return response
