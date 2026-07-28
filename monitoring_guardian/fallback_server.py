from __future__ import annotations

import html
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional


class GuardianFallbackServer:
    """Dependency-free status page used only while Django is unavailable."""

    def __init__(self, state_path: Path, log_path: Path, *, host: str = "127.0.0.1", port: int = 8000) -> None:
        self.state_path, self.log_path = state_path, log_path
        self.host, self.port = host, int(port)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> bool:
        if self.running:
            return True
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                body = owner.render().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                return

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError:
            self._server = None
            return False
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, name="guardian-fallback-http", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        server, thread = self._server, self._thread
        self._server = self._thread = None
        if server:
            server.shutdown()
            server.server_close()
        if thread and thread is not threading.current_thread():
            thread.join(timeout=3)

    def _state(self) -> Dict[str, Any]:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def _tail(self, limit: int = 120) -> str:
        try:
            text = "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:])
            text = re.sub(r"(?i)(api[_-]?key|token|secret|private[_-]?key|password)(\s*[:=]\s*)[^\s,;]+", r"\1\2[redacted]", text)
            text = re.sub(r"0x[a-fA-F0-9]{40,}", "[redacted-hex]", text)
            return text
        except Exception:
            return "Guardian is starting; no recovery output has been written yet."

    def render(self) -> str:
        state = self._state()
        rows = "".join(
            f"<tr><td>{html.escape(str(name))}</td><td>{html.escape(str(value.get('status', 'unknown')))}</td>"
            f"<td>{html.escape(str(value.get('attempts', 0)))}</td><td>{html.escape(str(value.get('detail', '')))}</td></tr>"
            for name, value in (state.get("components") or {}).items()
            if isinstance(value, dict)
        )
        return f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<meta http-equiv='refresh' content='5'><title>Guardian Recovery</title><style>
body{{margin:0;background:#07111f;color:#e8f0ff;font:16px system-ui}}main{{max-width:1050px;margin:auto;padding:30px 18px}}.panel{{background:#101f34;border:1px solid #315878;padding:20px;margin:14px 0}}h1{{color:#9bdcff}}table{{width:100%;border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid #29445d;text-align:left}}pre{{white-space:pre-wrap;max-height:50vh;overflow:auto;color:#b9ffd0}}.pulse{{color:#7dffa8}}
</style></head><body><main><h1>Guardian is restoring the control tower</h1><p class='pulse'>Guardian is running independently of Django. This page refreshes every five seconds.</p>
<section class='panel'><h2>Components</h2><table><thead><tr><th>Component</th><th>Status</th><th>Attempts</th><th>Detail</th></tr></thead><tbody>{rows}</tbody></table></section>
<section class='panel'><h2>Recovery output</h2><pre>{html.escape(self._tail())}</pre></section></main></body></html>"""
