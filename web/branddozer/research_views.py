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
        if output_format != "pdf":
            return HttpResponse("kind must be pdf, markdown, or json", status=400)
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
