from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from django.db.models import Q
from django.http import FileResponse, HttpResponse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import ResearchPaper
from .research import RUNTIME_ROOT, persist_paper_files


def _source_payload(source: Any) -> dict[str, Any]:
    return {
        "citation_key": source.citation_key,
        "title": source.title,
        "authors": source.authors,
        "publication_year": source.publication_year,
        "publisher": source.publisher,
        "url": source.url,
        "doi": source.doi,
        "retrieved_at": (
            source.retrieved_at.isoformat() if source.retrieved_at else None
        ),
        "content_sha256": source.content_sha256,
        "authority_tier": source.authority_tier,
        "peer_reviewed": source.peer_reviewed,
        "archival": source.archival,
        "source_class": source.source_class,
        "first_party": source.first_party,
        "provenance_status": source.provenance_status,
        "provenance_detail": source.provenance_detail,
        "verification_status": source.verification_status,
        "verification_detail": source.verification_detail,
        "verified_passage": source.verified_passage,
    }


def _claim_payload(claim: Any) -> dict[str, Any]:
    return {
        "id": claim.id,
        "section": claim.section,
        "claim_text": claim.claim_text,
        "source_keys": claim.source_keys,
        "verification_status": claim.verification_status,
        "rationale": claim.rationale,
    }


def paper_payload(paper: ResearchPaper, *, detail: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": str(paper.id),
        "project_id": str(paper.project_id),
        "project_name": paper.project.name,
        "run_id": str(paper.run_id) if paper.run_id else None,
        "title": paper.title,
        "research_question": paper.research_question,
        "abstract": paper.abstract,
        "keywords": paper.keywords,
        "status": paper.status,
        "version": paper.version,
        "citation_style": paper.citation_style,
        "target_journal": paper.target_journal,
        "validation_report": paper.validation_report,
        "word_count": paper.word_count,
        "content_sha256": paper.content_sha256,
        "validated_at": (
            paper.validated_at.isoformat() if paper.validated_at else None
        ),
        "created_at": paper.created_at.isoformat(),
        "updated_at": paper.updated_at.isoformat(),
        "downloads": {
            "pdf": f"/branddozer/research/papers/{paper.id}/download/?kind=pdf",
            "markdown": (
                f"/branddozer/research/papers/{paper.id}/download/?kind=markdown"
            ),
            "json": f"/branddozer/research/papers/{paper.id}/download/?kind=json",
            # Replayable record of every source check, for fact-checking.
            "verification": (
                f"/branddozer/research/papers/{paper.id}/download/?kind=verification"
            ),
        },
    }
    if detail:
        payload.update(
            content_markdown=paper.content_markdown,
            sources=[_source_payload(source) for source in paper.sources.all()],
            claims=[_claim_payload(claim) for claim in paper.claims.all()],
            revisions=[
                {
                    "version": revision.version,
                    "change_summary": revision.change_summary,
                    "validation_report": revision.validation_report,
                    "created_at": revision.created_at.isoformat(),
                }
                for revision in paper.revisions.all()
            ],
        )
    return payload


class ResearchPaperListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        query = str(request.query_params.get("q") or "").strip()[:300]
        project_id = str(request.query_params.get("project_id") or "").strip()
        paper_status = str(request.query_params.get("status") or "").strip()
        try:
            limit = max(1, min(100, int(request.query_params.get("limit", 40))))
        except (TypeError, ValueError):
            limit = 40
        papers = ResearchPaper.objects.select_related("project", "run")
        if query:
            terms = [term for term in re.split(r"\s+", query) if term][:12]
            for term in terms:
                papers = papers.filter(
                    Q(title__icontains=term)
                    | Q(research_question__icontains=term)
                    | Q(abstract__icontains=term)
                    | Q(content_markdown__icontains=term)
                    | Q(keywords__icontains=term)
                )
        if project_id:
            papers = papers.filter(project_id=project_id)
        if paper_status:
            papers = papers.filter(status=paper_status)
        values = [paper_payload(paper) for paper in papers[:limit]]
        return Response(
            {"papers": values, "count": len(values), "query": query},
            status=status.HTTP_200_OK,
        )


class ResearchPaperDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(
        self, request: Request, paper_id: str, *args: Any, **kwargs: Any
    ) -> Response:
        paper = (
            ResearchPaper.objects.select_related("project", "run")
            .prefetch_related("sources", "claims", "revisions")
            .filter(id=paper_id)
            .first()
        )
        if not paper:
            return Response(
                {"detail": "Research paper not found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response({"paper": paper_payload(paper, detail=True)})


class ResearchPresentationView(APIView):
    """The adjacent, mobile-first presentation view of a paper.

    Returns the deck itself (slides, kinds, durations, timeline). Narration
    and artwork are generated separately and asynchronously, because Polly
    and image generation are slow and a reader should be able to open the
    deck immediately in text form.
    """

    permission_classes = [IsAuthenticated]

    def get(
        self, request: Request, paper_id: str, *args: Any, **kwargs: Any
    ) -> Response:
        paper = ResearchPaper.objects.filter(id=paper_id).first()
        if not paper:
            return Response(
                {"detail": "Research paper not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        from branddozer.presentation import build_presentation
        from branddozer.presentation_media import (
            COLOR_RATIOS,
            COLOR_SCHEMES,
            TRANSITIONS,
            WORD_ANIMATIONS,
            score_sync_points,
        )

        deck = build_presentation(
            paper_id=str(paper.id),
            title=paper.title,
            markdown=paper.content_markdown,
            abstract=paper.abstract,
        )
        deck["timeline"] = score_sync_points(deck["slides"])
        # The player renders the option pickers from these, so the choices
        # stay in one place rather than being duplicated in the frontend.
        deck["options"] = {
            "transitions": list(TRANSITIONS),
            "word_animations": list(WORD_ANIMATIONS),
            "color_schemes": COLOR_SCHEMES,
            "color_ratios": COLOR_RATIOS,
        }
        deck["paper_status"] = paper.status
        return Response(deck)


class ResearchPresentationMediaView(APIView):
    """Generate narration and a synchronised score for a deck.

    Narration is the expensive part (two Polly calls per slide), so a
    `limit` is honoured to let the player preview a deck's opening before
    committing to the whole thing. The score is composed *after* narration
    so it can be aligned to the real spoken durations rather than the
    reading-speed estimates.
    """

    permission_classes = [IsAuthenticated]

    def post(
        self, request: Request, paper_id: str, *args: Any, **kwargs: Any
    ) -> Response:
        paper = ResearchPaper.objects.filter(id=paper_id).first()
        if not paper:
            return Response(
                {"detail": "Research paper not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        from branddozer.presentation import attach_word_timings, build_presentation
        from branddozer.presentation_media import (
            MediaConfig,
            score_sync_points,
            synthesize_slide,
        )

        data = request.data or {}
        config = MediaConfig(
            voice_id=str(data.get("voice_id") or "Joanna"),
            transition=str(data.get("transition") or "crossfade"),
            word_animation=str(data.get("word_animation") or "highlight"),
        )
        try:
            limit = int(data.get("limit") or 0)
        except (TypeError, ValueError):
            limit = 0

        deck = build_presentation(
            paper_id=str(paper.id),
            title=paper.title,
            markdown=paper.content_markdown,
            abstract=paper.abstract,
        )
        media_root = Path(RUNTIME_ROOT) / str(paper.id) / "presentation"
        targets = deck["slides"][:limit] if limit > 0 else deck["slides"]

        narrated = 0
        failures: list[dict[str, Any]] = []
        for slide in targets:
            # A bare URL is shown, never read aloud.
            if slide.get("notes") == "url":
                continue
            try:
                result = synthesize_slide(
                    slide["text"],
                    config=config,
                    out_dir=media_root / "audio",
                    name=f"slide-{slide['index']:05d}",
                )
                attach_word_timings(
                    slide, result["marks"], audio_ms=result["audio_ms"]
                )
                slide["audio_url"] = (
                    f"/api/branddozer/research/papers/{paper.id}"
                    f"/presentation/audio/{slide['index']}/"
                )
                narrated += 1
            except Exception as exc:
                # One bad slide must not lose the rest of the narration.
                failures.append({"index": slide["index"], "error": str(exc)[:200]})

        deck["narrated"] = narrated > 0
        deck["timeline"] = score_sync_points(deck["slides"])
        deck["estimated_duration_ms"] = sum(
            int(s.get("duration_ms") or 0) for s in deck["slides"]
        )

        score_info: dict[str, Any] = {"composed": False}
        if data.get("score") and narrated:
            score_info = self._compose(deck, paper, media_root)

        deck["score"] = score_info
        deck["media_failures"] = failures
        (media_root / "deck.json").parent.mkdir(parents=True, exist_ok=True)
        (media_root / "deck.json").write_text(
            json.dumps(deck, indent=2), encoding="utf-8"
        )
        return Response(deck)

    @staticmethod
    def _compose(
        deck: dict[str, Any], paper: Any, media_root: Any
    ) -> dict[str, Any]:
        """Compose and render a score aligned to the narrated timeline."""
        from branddozer.presentation_score import (
            ScoreRequest,
            alignment_report,
            compose_score,
            export_midi,
            render_wav,
        )

        try:
            from tools.ai_session import get_session_class, settings_for_role

            provider = "claude_code"
            SessionClass = get_session_class(provider, explicit=True)
            session = SessionClass(
                session_name=f"score-{paper.id}",
                read_timeout_s=None,
                **settings_for_role(provider, "worker"),
            )

            def agent_send(prompt: str, system: str = "") -> str:
                return session.send(prompt, stream=False, system=system)

            transitions = [int(p["at_ms"]) for p in deck["timeline"]]
            request = ScoreRequest(
                title=paper.title,
                abstract=paper.abstract,
                duration_ms=int(deck["estimated_duration_ms"]),
                transitions_ms=transitions,
                findings_tone=(
                    "negative / inconclusive"
                    if paper.status != "validated"
                    else "supported"
                ),
            )
            score = compose_score(request, agent_send=agent_send)
            rendered = render_wav(score, media_root / "score.wav")
            midi = export_midi(score, media_root / "score.mid")
            return {
                "composed": True,
                "key": score["key"],
                "bpm": score["bpm"],
                "rationale": score["rationale"],
                "alignment": alignment_report(score, transitions),
                "audio_url": (
                    f"/api/branddozer/research/papers/{paper.id}/presentation/score/"
                ),
                "midi_path": midi,
                "duration_ms": rendered["duration_ms"],
                "issues": score.get("issues") or [],
            }
        except Exception as exc:
            return {"composed": False, "error": str(exc)[:300]}


class ResearchPresentationAudioView(APIView):
    """Serve one slide's narration clip."""

    permission_classes = [IsAuthenticated]

    def get(
        self, request: Request, paper_id: str, index: int, *args: Any, **kwargs: Any
    ) -> HttpResponse:
        path = (
            Path(RUNTIME_ROOT)
            / str(paper_id)
            / "presentation"
            / "audio"
            / f"slide-{int(index):05d}.mp3"
        )
        if not path.is_file():
            return HttpResponse("narration not generated", status=404)
        return HttpResponse(path.read_bytes(), content_type="audio/mpeg")


class ResearchPresentationScoreView(APIView):
    """Serve the rendered background score."""

    permission_classes = [IsAuthenticated]

    def get(
        self, request: Request, paper_id: str, *args: Any, **kwargs: Any
    ) -> HttpResponse:
        path = Path(RUNTIME_ROOT) / str(paper_id) / "presentation" / "score.wav"
        if not path.is_file():
            return HttpResponse("score not generated", status=404)
        return HttpResponse(path.read_bytes(), content_type="audio/wav")


class ResearchPaperDownloadView(APIView):
    permission_classes = [IsAuthenticated]

    def get(
        self, request: Request, paper_id: str, *args: Any, **kwargs: Any
    ) -> HttpResponse:
        paper = (
            ResearchPaper.objects.select_related("project", "run")
            .prefetch_related("sources", "claims", "revisions")
            .filter(id=paper_id)
            .first()
        )
        if not paper:
            return HttpResponse("Research paper not found", status=404)
        output_format = str(request.query_params.get("kind") or "pdf").lower()
        filename = re.sub(r"[^a-zA-Z0-9._-]+", "-", paper.title).strip("-")[:100]
        filename = filename or f"research-paper-{paper.id}"
        if output_format in {"md", "markdown"}:
            response = HttpResponse(
                paper.content_markdown, content_type="text/markdown; charset=utf-8"
            )
            response["Content-Disposition"] = (
                f'attachment; filename="{filename}-v{paper.version}.md"'
            )
            return response
        if output_format == "json":
            content = json.dumps(
                paper_payload(paper, detail=True), indent=2, ensure_ascii=False
            )
            response = HttpResponse(
                content, content_type="application/json; charset=utf-8"
            )
            response["Content-Disposition"] = (
                f'attachment; filename="{filename}-v{paper.version}.json"'
            )
            return response
        if output_format in {"verification", "manifest"}:
            # The fact-checking manifest: every source check, replayable.
            from branddozer.reproducibility import build_manifest

            manifest = build_manifest(
                [
                    {
                        "citation_key": source.citation_key,
                        "url": source.url,
                        "verification_status": source.verification_status,
                        "verification_detail": source.verification_detail,
                        "content_sha256": source.content_sha256,
                        "retrieved_at": (
                            source.retrieved_at.isoformat()
                            if source.retrieved_at
                            else ""
                        ),
                        "verified_passage": source.verified_passage,
                    }
                    for source in paper.sources.all()
                ],
                paper_sha256=paper.content_sha256,
                claims=[
                    {
                        "claim_text": claim.claim_text,
                        "source_keys": claim.source_keys,
                    }
                    for claim in paper.claims.all()
                ],
            )
            response = HttpResponse(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                content_type="application/json; charset=utf-8",
            )
            response["Content-Disposition"] = (
                f'attachment; filename="{filename}-v{paper.version}-verification.json"'
            )
            return response
        if output_format != "pdf":
            return HttpResponse(
                "kind must be pdf, markdown, json, or verification", status=400
            )
        path = Path(paper.pdf_path) if paper.pdf_path else Path()
        try:
            safe_root = RUNTIME_ROOT.resolve()
            safe_path = path.resolve()
            safe = safe_path.is_relative_to(safe_root)
        except (OSError, ValueError):
            safe = False
        if not safe or not safe_path.is_file():
            persist_paper_files(paper)
            safe_path = Path(paper.pdf_path).resolve()
        return FileResponse(
            safe_path.open("rb"),
            as_attachment=True,
            filename=f"{filename}-v{paper.version}.pdf",
            content_type="application/pdf",
        )
