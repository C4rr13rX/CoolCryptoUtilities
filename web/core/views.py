from __future__ import annotations

import os

from django.conf import settings
from django.views.generic import TemplateView


class IndexView(TemplateView):
    template_name = "core/index.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update(
            {
                "debug": settings.DEBUG,
                "vite_dev_server": os.getenv("VITE_DEV_SERVER", "http://localhost:5173"),
            }
        )
        return context
