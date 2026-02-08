from __future__ import annotations

from urllib.parse import urlparse

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from services.web_research import CrawlPolicy, WebResearcher

from .models import (
    InvestigationArticle,
    InvestigationEntity,
    InvestigationEvidence,
    InvestigationProject,
    InvestigationRelation,
    InvestigationTarget,
)


def _get_project(project_id: int, request: Request) -> InvestigationProject | None:
    try:
        return InvestigationProject.objects.get(id=project_id, user=request.user)
    except InvestigationProject.DoesNotExist:
        return None


class ProjectListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        items = (
            InvestigationProject.objects.filter(user=request.user)
            .order_by("-updated_at")
        )
        payload = [
            {
                "id": item.id,
                "name": item.name,
                "description": item.description,
                "status": item.status,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "updated_at": item.updated_at.isoformat() if item.updated_at else None,
            }
            for item in items
        ]
        return Response({"items": payload, "count": len(payload)}, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        payload = request.data or {}
        name = str(payload.get("name") or "").strip()
        if not name:
            return Response({"detail": "name is required"}, status=status.HTTP_400_BAD_REQUEST)
        description = str(payload.get("description") or "").strip()
        project = InvestigationProject.objects.create(
            user=request.user,
            name=name,
            description=description,
            status=payload.get("status") or "active",
        )
        return Response(
            {
                "item": {
                    "id": project.id,
                    "name": project.name,
                    "description": project.description,
                    "status": project.status,
                    "created_at": project.created_at.isoformat() if project.created_at else None,
                    "updated_at": project.updated_at.isoformat() if project.updated_at else None,
                }
            },
            status=status.HTTP_201_CREATED,
        )


class ProjectDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request: Request, project_id: int, *args, **kwargs) -> Response:
        project = _get_project(project_id, request)
        if not project:
            return Response({"detail": "project not found"}, status=status.HTTP_404_NOT_FOUND)
        payload = request.data or {}
        if "name" in payload:
            project.name = str(payload.get("name") or "").strip() or project.name
        if "description" in payload:
            project.description = str(payload.get("description") or "").strip()
        if "status" in payload:
            project.status = str(payload.get("status") or project.status)
        project.save(update_fields=["name", "description", "status", "updated_at"])
        return Response(
            {
                "item": {
                    "id": project.id,
                    "name": project.name,
                    "description": project.description,
                    "status": project.status,
                    "created_at": project.created_at.isoformat() if project.created_at else None,
                    "updated_at": project.updated_at.isoformat() if project.updated_at else None,
                }
            },
            status=status.HTTP_200_OK,
        )

    def delete(self, request: Request, project_id: int, *args, **kwargs) -> Response:
        project = _get_project(project_id, request)
        if not project:
            return Response({"detail": "project not found"}, status=status.HTTP_404_NOT_FOUND)
        project.delete()
        return Response({"deleted": True}, status=status.HTTP_200_OK)


class TargetListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, project_id: int, *args, **kwargs) -> Response:
        project = _get_project(project_id, request)
        if not project:
            return Response({"detail": "project not found"}, status=status.HTTP_404_NOT_FOUND)
        targets = project.targets.order_by("-updated_at")
        payload = [
            {
                "id": t.id,
                "url": t.url,
                "requires_login": t.requires_login,
                "login_url": t.login_url,
                "notes": t.notes,
                "crawl_policy": t.crawl_policy or {},
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "updated_at": t.updated_at.isoformat() if t.updated_at else None,
            }
            for t in targets
        ]
        return Response({"items": payload, "count": len(payload)}, status=status.HTTP_200_OK)

    def post(self, request: Request, project_id: int, *args, **kwargs) -> Response:
        project = _get_project(project_id, request)
        if not project:
            return Response({"detail": "project not found"}, status=status.HTTP_404_NOT_FOUND)
        payload = request.data or {}
        url = str(payload.get("url") or "").strip()
        if not url:
            return Response({"detail": "url is required"}, status=status.HTTP_400_BAD_REQUEST)
        target = InvestigationTarget.objects.create(
            project=project,
            url=url,
            requires_login=bool(payload.get("requires_login")),
            login_url=str(payload.get("login_url") or ""),
            notes=str(payload.get("notes") or ""),
            crawl_policy=payload.get("crawl_policy") or {},
        )
        return Response(
            {
                "item": {
                    "id": target.id,
                    "url": target.url,
                    "requires_login": target.requires_login,
                    "login_url": target.login_url,
                    "notes": target.notes,
                    "crawl_policy": target.crawl_policy or {},
                    "created_at": target.created_at.isoformat() if target.created_at else None,
                    "updated_at": target.updated_at.isoformat() if target.updated_at else None,
                }
            },
            status=status.HTTP_201_CREATED,
        )


class TargetCrawlView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, target_id: int, *args, **kwargs) -> Response:
        try:
            target = InvestigationTarget.objects.get(id=target_id, project__user=request.user)
        except InvestigationTarget.DoesNotExist:
            return Response({"detail": "target not found"}, status=status.HTTP_404_NOT_FOUND)
        payload = request.data or {}
        policy_payload = payload.get("policy") or target.crawl_policy or {}
        parsed = urlparse(target.url)
        allowed_domain = parsed.netloc
        policy = CrawlPolicy(
            max_depth=int(policy_payload.get("max_depth", 2)),
            max_pages=int(policy_payload.get("max_pages", 12)),
            allowed_domains=policy_payload.get("allowed_domains") or ([allowed_domain] if allowed_domain else []),
        )
        researcher = WebResearcher()
        results = researcher.crawl(target.url, policy=policy)
        created = 0
        for url, text in results:
            InvestigationEvidence.objects.create(
                project=target.project,
                target=target,
                url=url,
                title="",
                content=text[:8000],
                metadata={"source": "crawl"},
            )
            created += 1
        return Response({"saved": created, "pages": len(results)}, status=status.HTTP_200_OK)


class EvidenceListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, project_id: int, *args, **kwargs) -> Response:
        project = _get_project(project_id, request)
        if not project:
            return Response({"detail": "project not found"}, status=status.HTTP_404_NOT_FOUND)
        limit = int(request.query_params.get("limit", 50))
        entries = project.evidence.order_by("-captured_at")[:limit]
        payload = [
            {
                "id": e.id,
                "url": e.url,
                "title": e.title,
                "content": e.content,
                "metadata": e.metadata or {},
                "captured_at": e.captured_at.isoformat() if e.captured_at else None,
            }
            for e in entries
        ]
        return Response({"items": payload, "count": len(payload)}, status=status.HTTP_200_OK)


class ArticleListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, project_id: int, *args, **kwargs) -> Response:
        project = _get_project(project_id, request)
        if not project:
            return Response({"detail": "project not found"}, status=status.HTTP_404_NOT_FOUND)
        entries = project.articles.order_by("-updated_at")
        payload = [
            {
                "id": item.id,
                "title": item.title,
                "status": item.status,
                "updated_at": item.updated_at.isoformat() if item.updated_at else None,
            }
            for item in entries
        ]
        return Response({"items": payload, "count": len(payload)}, status=status.HTTP_200_OK)

    def post(self, request: Request, project_id: int, *args, **kwargs) -> Response:
        project = _get_project(project_id, request)
        if not project:
            return Response({"detail": "project not found"}, status=status.HTTP_404_NOT_FOUND)
        payload = request.data or {}
        title = str(payload.get("title") or "Untitled")
        article = InvestigationArticle.objects.create(
            project=project,
            title=title,
            body=str(payload.get("body") or ""),
            status=str(payload.get("status") or "draft"),
        )
        return Response(
            {
                "item": {
                    "id": article.id,
                    "title": article.title,
                    "status": article.status,
                    "updated_at": article.updated_at.isoformat() if article.updated_at else None,
                }
            },
            status=status.HTTP_201_CREATED,
        )


class ArticleDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, article_id: int, *args, **kwargs) -> Response:
        try:
            article = InvestigationArticle.objects.get(id=article_id, project__user=request.user)
        except InvestigationArticle.DoesNotExist:
            return Response({"detail": "article not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(
            {
                "item": {
                    "id": article.id,
                    "title": article.title,
                    "body": article.body,
                    "status": article.status,
                    "updated_at": article.updated_at.isoformat() if article.updated_at else None,
                }
            },
            status=status.HTTP_200_OK,
        )

    def patch(self, request: Request, article_id: int, *args, **kwargs) -> Response:
        try:
            article = InvestigationArticle.objects.get(id=article_id, project__user=request.user)
        except InvestigationArticle.DoesNotExist:
            return Response({"detail": "article not found"}, status=status.HTTP_404_NOT_FOUND)
        payload = request.data or {}
        if "title" in payload:
            article.title = str(payload.get("title") or article.title)
        if "body" in payload:
            article.body = str(payload.get("body") or "")
        if "status" in payload:
            article.status = str(payload.get("status") or article.status)
        article.save(update_fields=["title", "body", "status", "updated_at"])
        return Response(
            {
                "item": {
                    "id": article.id,
                    "title": article.title,
                    "body": article.body,
                    "status": article.status,
                    "updated_at": article.updated_at.isoformat() if article.updated_at else None,
                }
            },
            status=status.HTTP_200_OK,
        )


class EntityListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, project_id: int, *args, **kwargs) -> Response:
        project = _get_project(project_id, request)
        if not project:
            return Response({"detail": "project not found"}, status=status.HTTP_404_NOT_FOUND)
        entries = project.entities.order_by("name")
        payload = [
            {
                "id": item.id,
                "kind": item.kind,
                "name": item.name,
                "aliases": item.aliases or [],
                "metadata": item.metadata or {},
            }
            for item in entries
        ]
        return Response({"items": payload, "count": len(payload)}, status=status.HTTP_200_OK)

    def post(self, request: Request, project_id: int, *args, **kwargs) -> Response:
        project = _get_project(project_id, request)
        if not project:
            return Response({"detail": "project not found"}, status=status.HTTP_404_NOT_FOUND)
        payload = request.data or {}
        name = str(payload.get("name") or "").strip()
        if not name:
            return Response({"detail": "name is required"}, status=status.HTTP_400_BAD_REQUEST)
        entity = InvestigationEntity.objects.create(
            project=project,
            kind=str(payload.get("kind") or "person"),
            name=name,
            aliases=payload.get("aliases") or [],
            metadata=payload.get("metadata") or {},
        )
        return Response(
            {
                "item": {
                    "id": entity.id,
                    "kind": entity.kind,
                    "name": entity.name,
                    "aliases": entity.aliases or [],
                    "metadata": entity.metadata or {},
                }
            },
            status=status.HTTP_201_CREATED,
        )


class RelationListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, project_id: int, *args, **kwargs) -> Response:
        project = _get_project(project_id, request)
        if not project:
            return Response({"detail": "project not found"}, status=status.HTTP_404_NOT_FOUND)
        entries = project.relations.select_related("source", "target").order_by("-created_at")
        payload = [
            {
                "id": item.id,
                "source": item.source.name,
                "target": item.target.name,
                "relation_type": item.relation_type,
                "notes": item.notes,
            }
            for item in entries
        ]
        return Response({"items": payload, "count": len(payload)}, status=status.HTTP_200_OK)

    def post(self, request: Request, project_id: int, *args, **kwargs) -> Response:
        project = _get_project(project_id, request)
        if not project:
            return Response({"detail": "project not found"}, status=status.HTTP_404_NOT_FOUND)
        payload = request.data or {}
        try:
            source = InvestigationEntity.objects.get(id=payload.get("source_id"), project=project)
            target = InvestigationEntity.objects.get(id=payload.get("target_id"), project=project)
        except InvestigationEntity.DoesNotExist:
            return Response({"detail": "invalid source/target"}, status=status.HTTP_400_BAD_REQUEST)
        relation = InvestigationRelation.objects.create(
            project=project,
            source=source,
            target=target,
            relation_type=str(payload.get("relation_type") or "associated"),
            notes=str(payload.get("notes") or ""),
        )
        return Response(
            {
                "item": {
                    "id": relation.id,
                    "source": source.name,
                    "target": target.name,
                    "relation_type": relation.relation_type,
                    "notes": relation.notes,
                }
            },
            status=status.HTTP_201_CREATED,
        )
