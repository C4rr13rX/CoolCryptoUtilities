from __future__ import annotations

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from investigations import views as investigations_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/streams/", include("streams.urls")),
    path("api/telemetry/", include("telemetry.urls")),
    path("api/console/", include("opsconsole.urls")),
    path("api/datalab/", include("datalab.urls")),
    path("api/lab/", include("lab.urls")),
    path("api/guardian/", include("guardianpanel.urls")),
    path("api/cron/", include("cronpanel.urls")),
    path("api/secure/", include("securevault.urls")),
    path("api/wallet/", include("walletpanel.urls")),
    path("api/addressbook/", include("addressbook.urls")),
    path("api/integrations/", include("integrations.urls")),
    path("api/branddozer/", include("branddozer.urls")),
    path("api/u53rxr080t/", include("u53rxr080t.urls")),
    path("api/investigations/", include("investigations.urls")),
    # Legacy alias: allow /investigations/... API calls if a client misses /api prefix.
    path("investigations/projects/", investigations_views.ProjectListView.as_view()),
    path("investigations/projects", investigations_views.ProjectListView.as_view()),
    path("investigations/projects/<int:project_id>/", investigations_views.ProjectDetailView.as_view()),
    path("investigations/projects/<int:project_id>", investigations_views.ProjectDetailView.as_view()),
    path("investigations/projects/<int:project_id>/targets/", investigations_views.TargetListView.as_view()),
    path("investigations/projects/<int:project_id>/targets", investigations_views.TargetListView.as_view()),
    path("investigations/targets/<int:target_id>/crawl/", investigations_views.TargetCrawlView.as_view()),
    path("investigations/targets/<int:target_id>/crawl", investigations_views.TargetCrawlView.as_view()),
    path("investigations/projects/<int:project_id>/evidence/", investigations_views.EvidenceListView.as_view()),
    path("investigations/projects/<int:project_id>/evidence", investigations_views.EvidenceListView.as_view()),
    path("investigations/projects/<int:project_id>/articles/", investigations_views.ArticleListView.as_view()),
    path("investigations/projects/<int:project_id>/articles", investigations_views.ArticleListView.as_view()),
    path("investigations/articles/<int:article_id>/", investigations_views.ArticleDetailView.as_view()),
    path("investigations/articles/<int:article_id>", investigations_views.ArticleDetailView.as_view()),
    path("investigations/projects/<int:project_id>/entities/", investigations_views.EntityListView.as_view()),
    path("investigations/projects/<int:project_id>/entities", investigations_views.EntityListView.as_view()),
    path("investigations/projects/<int:project_id>/relations/", investigations_views.RelationListView.as_view()),
    path("investigations/projects/<int:project_id>/relations", investigations_views.RelationListView.as_view()),
    path("", include("core.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

handler500 = "core.views.guardian_failure_response"
