"""
Every SPA route must survive a refresh / deep link.

The recurring bug this prevents: `SpaRouteView` used a hardcoded allowlist, so
a new page worked when navigated to inside the app and then silently bounced to
the dashboard when refreshed or opened by URL. Because the redirect looks like
a working app rather than an error, it was found by hand each time and fixed
one route at a time -- /video-studio was the latest.

The allowlist is now derived from the frontend router, so this test is the
guard that it stays derived rather than drifting back to a manual list.
"""

from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")
os.environ.setdefault("DJANGO_DB_VENDOR", "sqlite")
os.environ.setdefault("ALLOW_SQLITE_FALLBACK", "1")
os.environ.setdefault("SECURE_ENV_HYDRATED", "1")
os.environ.setdefault("GUARDIAN_AUTO_DISABLED", "1")
os.environ.setdefault("CRON_AUTO_DISABLED", "1")
os.environ.setdefault("PRODUCTION_AUTO_DISABLED", "1")

import django  # noqa: E402

django.setup()

from core.views import SpaRouteView  # noqa: E402

ROUTER = ROOT / "web" / "frontend" / "src" / "router" / "index.ts"


def router_routes() -> set[str]:
    """Single-segment top-level routes declared by the Vue router."""
    source = ROUTER.read_text(encoding="utf-8")
    return {m.lower() for m in re.findall(r"path:\s*'/([A-Za-z0-9_-]+)'", source)}


class SpaRouteRefresh(unittest.TestCase):
    def test_every_router_route_survives_refresh(self):
        """
        The headline guard: no route may be missing from the allowlist.

        A miss here is the exact symptom -- refresh redirects to dashboard.
        """
        known = SpaRouteView.known_routes()
        missing = sorted(r for r in router_routes() if r not in known)
        self.assertFalse(
            missing,
            "these routes redirect to the dashboard on refresh: "
            + ", ".join(missing),
        )

    def test_video_studio_specifically(self):
        """Regression: the route that exposed the hardcoded allowlist."""
        self.assertIn("video-studio", SpaRouteView.known_routes())

    def test_hyphenated_routes_are_handled(self):
        """
        Hyphens are the common case for new pages and were the risky one.

        Django's slug converter does accept them, so the URL matched -- the
        redirect came from the allowlist, one layer further in. Worth pinning
        so a future 'fix' does not swap the converter and call it solved.
        """
        known = SpaRouteView.known_routes()
        for route in ("model-control", "wizard-chat", "video-studio"):
            self.assertIn(route, known)

    def test_unknown_routes_still_redirect(self):
        """
        The allowlist must still be an allowlist.

        Deriving it must not turn the catch-all into "render the shell for
        anything", which would serve the SPA for typos and probes.
        """
        known = SpaRouteView.known_routes()
        for bogus in ("wp-admin", "..", "definitely-not-a-page"):
            self.assertNotIn(bogus, known)

    def test_allowlist_is_derived_not_hardcoded(self):
        """
        The list must come from the router, or this all recurs.

        Checks the source: a literal set of route names inside dispatch() is
        what made every new page a two-place edit.
        """
        source = (ROOT / "web" / "core" / "views.py").read_text(encoding="utf-8")
        start = source.index("class SpaRouteView")
        end = source.index("class PaperPresentView")
        block = source[start:end]
        self.assertIn("known_routes", block)
        self.assertIn("router", block.lower())

    def test_dashboard_is_always_available(self):
        """The default route must never depend on parsing succeeding."""
        self.assertIn("dashboard", SpaRouteView._EXTRA_ROUTES)


if __name__ == "__main__":
    unittest.main()
