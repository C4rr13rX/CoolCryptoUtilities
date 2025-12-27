from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.branddozer_ui import capture_ui_screenshots


def _parse_routes(raw: str) -> List[Dict[str, str]]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture UI screenshots for BrandDozer.")
    parser.add_argument("--root", default=".", help="Project root path")
    parser.add_argument("--base-url", default=None, help="Base URL to capture")
    parser.add_argument("--output-dir", default="runtime/branddozer/ui-capture", help="Output directory")
    parser.add_argument(
        "--routes",
        default="",
        help='JSON array of routes, e.g. \'[{"name":"branddozer","path":"/branddozer"}]\'',
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    routes = _parse_routes(args.routes)
    result = capture_ui_screenshots(root, output_dir=output_dir, base_url=args.base_url, routes=routes or None)

    payload = {
        "base_url": result.base_url,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "screenshots": [str(p) for p in result.screenshots],
        "server_started": result.server_started,
        "server_log": str(result.server_log) if result.server_log else "",
        "meta": result.meta,
    }
    print(json.dumps(payload, indent=2))
    return 0 if result.exit_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
