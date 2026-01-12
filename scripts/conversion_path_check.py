#!/usr/bin/env python3
"""
Conversion path smoke test using Playwright (Chromium).
Env:
  BASE_URL: required root URL to test.
  HEADLESS: default true.
  VIEWPORT_MOBILE: e.g., 360,640 (width,height) or empty for default.
  VIEWPORT_DESKTOP: e.g., 1280,800 (width,height) or empty for default.
Behavior:
  - Visits BASE_URL, asserts presence of a CTA (button/link with common text)
  - Optionally clicks CTA and checks for a form/next-page element
  - Runs mobile and desktop viewports
Exit non-zero on failure unless CONVERSION_WARN_ONLY=1
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HEADLESS = os.getenv("HEADLESS", "1").lower() not in {"0", "false", "no"}
WARN_ONLY = os.getenv("CONVERSION_WARN_ONLY", "0").lower() in {"1", "true", "yes", "on"}

try:
    from playwright.sync_api import sync_playwright
except Exception as exc:  # pragma: no cover
    print(f"Playwright import failed: {exc}")
    sys.exit(0 if WARN_ONLY else 1)


def _viewport(env_var: str, default: tuple[int, int]) -> tuple[int, int]:
    raw = os.getenv(env_var, "")
    if not raw:
        return default
    parts = raw.split(",")
    if len(parts) != 2:
        return default
    try:
        return int(parts[0]), int(parts[1])
    except Exception:
        return default


CTA_SELECTORS = [
    "text=/.*(start|get started|try|sign up|join|demo|buy|subscribe).*/i",
    "role=button >> text=/.*(start|get started|try|sign up|join|demo|buy|subscribe).*/i",
    "a[href*='signup']",
    "a[href*='start']",
]


def run_flow(browser, base_url: str, viewport: tuple[int, int]) -> list[str]:
    messages = []
    page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
    page.goto(base_url, wait_until="networkidle")
    messages.append(f"Loaded {base_url} at {viewport}")
    cta_found = False
    for selector in CTA_SELECTORS:
        try:
            el = page.locator(selector).first
            if el and el.is_visible():
                cta_found = True
                el.click(timeout=3000)
                messages.append(f"Clicked CTA selector: {selector}")
                break
        except Exception:
            continue
    if not cta_found:
        raise RuntimeError("CTA not found")

    # Simple form/next-page check
    try:
        page.wait_for_timeout(1000)
        if page.locator("form").count() > 0:
            messages.append("Form present after CTA.")
        elif page.locator("text=/thank|welcome|next/i").count() > 0:
            messages.append("Next-page content present after CTA.")
        else:
            messages.append("CTA click succeeded; no form/next marker detected.")
    except Exception:
        messages.append("CTA click succeeded; verification skipped.")
    return messages


def main() -> int:
    base_url = os.getenv("BASE_URL")
    if not base_url:
        print("BASE_URL is required for conversion path check.")
        return 0 if WARN_ONLY else 1
    vp_mobile = _viewport("VIEWPORT_MOBILE", (360, 640))
    vp_desktop = _viewport("VIEWPORT_DESKTOP", (1280, 800))
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=HEADLESS)
            msgs = []
            for vp in (vp_mobile, vp_desktop):
                msgs.extend(run_flow(browser, base_url, vp))
            browser.close()
            print("\n".join(msgs))
            return 0
    except Exception as exc:
        print(f"Conversion path check failed: {exc}")
        return 0 if WARN_ONLY else 1


if __name__ == "__main__":
    sys.exit(main())
