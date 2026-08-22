"""
Android entry point: a local API Gateway in front of the Lambda handlers.

The Vue GUI speaks HTTP to ``baseURL: '/api'``, so something must accept a
socket.  That something is deliberately as thin as possible: it translates a
request into a Lambda event, calls ``lambda_runtime.invoke("http", event)``,
and translates the response back.  It is API Gateway, not a web server.

Why not just run Django directly
--------------------------------
Because then Django is *resident* — holding memory and threads whether or not
the user is looking at the screen.  Routing through the handler keeps one
contract for every entry point (tap, schedule, or cloud), so the same code
paths run on the phone and in AWS and cannot drift apart.  The Django app
itself is imported once and kept warm inside the runtime, which is what makes
a warm request 8 ms rather than 1.5 s.

Three sandbox facts this has to paper over, each of which otherwise breaks
Django outright:

1. **No writable working directory.**  The APK is read-only; only
   ``getFilesDir()`` accepts writes.  The project derives paths from
   ``Path(__file__).parents[N]``, which now points inside the APK, so the same
   ``WRITABLE_ROOT`` indirection the Lambda build uses is reused here.
2. **No process supervisor.**  ``core/apps.py`` starts guardian/cron/production
   threads at import.  Those belong to ``ScheduledJobService`` now, so they are
   disabled here.
3. **No TensorFlow wheel.**  ``tf_stub`` absorbs the import so a missing
   package cannot take down unrelated views.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

logger = logging.getLogger("android.bootstrap")

_server = None
_thread = None
_state = {"status": "stopped", "port": None, "error": None}

# Requests larger than this are refused rather than buffered. Nothing the GUI
# sends is close; an oversized body here means a bug or an attack.
MAX_BODY = 8 * 1024 * 1024


def _configure(files_dir: str) -> Path:
    root = Path(files_dir)
    for name in ("storage", "runtime", "logs", "data", "config"):
        (root / name).mkdir(parents=True, exist_ok=True)

    os.environ["WRITABLE_ROOT"] = str(root)
    os.environ.setdefault("DJANGO_SQLITE_PATH", str(root / "storage" / "app.db"))
    os.environ.setdefault("TRADING_DB_PATH", str(root / "storage" / "trading_cache.db"))
    os.environ.setdefault("CRON_STATE_PATH", str(root / "runtime" / "cron" / "state.json"))
    os.environ.setdefault("HOME", str(root))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / ".cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".mpl"))

    os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                          "coolcrypto_dashboard.settings_android")
    os.environ.setdefault("DJANGO_DB_VENDOR", "sqlite")
    os.environ.setdefault("ALLOW_SQLITE_FALLBACK", "1")
    os.environ.setdefault("SECURE_ENV_HYDRATED", "1")

    # Long-running work is owned by ScheduledJobService, not by import side
    # effects. Without these, importing Django starts daemon threads the OS
    # cannot see, the user cannot stop, and Doze suspends mid-work.
    os.environ["GUARDIAN_AUTO_DISABLED"] = "1"
    os.environ["CRON_AUTO_DISABLED"] = "1"
    os.environ["PRODUCTION_AUTO_DISABLED"] = "1"
    os.environ["WIZARD_DISABLE_REFRESHER"] = "1"
    os.environ["WIZARD_BRAIN_FEEDER_ENABLED"] = "0"
    os.environ.pop("WAITRESS_PORT", None)
    return root


def _to_event(handler: BaseHTTPRequestHandler, body: bytes) -> dict:
    """Build the API Gateway v2 event the handlers already expect."""
    split = urlsplit(handler.path)
    return {
        "version": "2.0",
        "rawPath": split.path,
        "rawQueryString": split.query,
        "headers": {k.lower(): v for k, v in handler.headers.items()},
        "requestContext": {
            "http": {
                "method": handler.command,
                "path": split.path,
                "protocol": handler.request_version,
                "sourceIp": "127.0.0.1",
            },
            "stage": "$default",
            "domainName": "127.0.0.1",
        },
        "body": body.decode("utf-8", "replace") if body else "",
        "isBase64Encoded": False,
    }


class _Gateway(BaseHTTPRequestHandler):
    """Translates HTTP <-> Lambda event. Holds no application state."""

    # HTTP/1.0 semantics: one request per connection, closed when the body is
    # written. HTTP/1.1 keep-alive requires the response to advertise it
    # consistently, and a mismatch leaves the WebView waiting on a socket that
    # will never produce another byte. The client is a single local WebView,
    # so connection reuse buys nothing worth that risk.
    protocol_version = "HTTP/1.0"
    server_version = "CoolCryptoGateway/1.0"

    def log_message(self, fmt, *args):     # noqa: A003
        logger.debug(fmt, *args)

    def _handle(self) -> None:
        import lambda_runtime

        # The event loop is owned by lambda_runtime, which keeps exactly one
        # for the process. Creating one here per request thread is what made
        # Mangum's executor deadlock after the first invocation.

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length > MAX_BODY:
            self.send_error(413, "request too large")
            return
        body = self.rfile.read(length) if length else b""

        # Route to the handler that owns the path, mirroring the API Gateway
        # mounts in serverless/local/deploy_local.sh (/auth, /hybrid, /market
        # are separate functions; everything else is the Django app). Keeping
        # the split identical means a request behaves the same on-device as it
        # does deployed.
        path = urlsplit(self.path).path
        if path.startswith(("/api/auth/", "/auth/")):
            target = "auth"
        elif path.startswith(("/api/hybrid/", "/hybrid/")):
            target = "hybrid"
        elif path.startswith(("/api/market/", "/market/")):
            target = "market"
        else:
            target = "http"

        try:
            result = lambda_runtime.invoke(target, _to_event(self, body))
        except Exception as exc:  # noqa: BLE001
            logger.exception("gateway invoke failed")
            self.send_error(500, f"invoke failed: {exc}")
            return

        status = int(result.get("statusCode", 500))
        payload = result.get("body") or ""
        if result.get("isBase64Encoded"):
            import base64

            data = base64.b64decode(payload)
        else:
            data = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)

        self.send_response(status)
        # Be explicit about closing. Without this header the client cannot
        # tell whether more is coming and waits on the socket until its own
        # timeout -- which presents as a dashboard that loads one panel and
        # then hangs.
        self.send_header("Connection", "close")
        for key, value in (result.get("headers") or {}).items():
            # Content-Length is set below from the real body length; echoing
            # the handler's value risks a mismatch that hangs keep-alive.
            if key.lower() != "content-length":
                self.send_header(key, str(value))
        for cookie in result.get("cookies") or []:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)
        try:
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The WebView navigated away mid-response; not an error worth
            # logging on every page change.
            pass
        self.close_connection = True

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = _handle
    do_OPTIONS = _handle


class _ThreadingGateway(ThreadingHTTPServer):
    # stdlib ThreadingHTTPServer rather than a hand-rolled ThreadingMixIn
    # subclass: the mixin's MRO has to come before HTTPServer to take effect,
    # and getting that wrong silently yields a server that accepts exactly one
    # connection and then stops responding.
    daemon_threads = True
    allow_reuse_address = True


def start_server(files_dir: str, port: int = 8765) -> dict:
    """Bind the local gateway. Returns immediately; serving runs on a thread."""
    global _server, _thread

    if _state["status"] == "running":
        return dict(_state)
    try:
        _configure(files_dir)

        # Before anything imports Django: several modules import TensorFlow at
        # module scope and there is no Android wheel.
        import tf_stub  # noqa: F401

        _server = _ThreadingGateway(("127.0.0.1", port), _Gateway)
        _thread = threading.Thread(target=_server.serve_forever,
                                   name="local-gateway", daemon=True)
        _thread.start()
        _state.update(status="running", port=port, error=None)
        logger.info("gateway listening on 127.0.0.1:%s", port)
    except Exception as exc:  # noqa: BLE001
        _state.update(status="error", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    return dict(_state)


def warm_handlers() -> str:
    """
    Pre-import the request-path handlers.

    The one deliberate exception to "nothing runs when idle": a few hundred
    milliseconds of import, once, so the user's first tap is 8 ms instead of
    1.5 s.
    """
    import lambda_runtime

    return lambda_runtime.warm(["http", "auth"])


def migrate() -> str:
    """Create the schema on first launch. A fresh install has no database."""
    try:
        import django

        django.setup()
        from django.core.management import call_command

        call_command("migrate", "--noinput", "--run-syncdb", verbosity=0)
        return json.dumps({"status": "ok"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("migrate failed")
        return json.dumps({"status": "error", "error": str(exc)})


def stop_server() -> dict:
    global _server, _thread
    if _server is not None:
        try:
            _server.shutdown()
            _server.server_close()
        except Exception:  # noqa: BLE001
            pass
    _server = None
    _thread = None
    _state.update(status="stopped", port=None)
    return dict(_state)


def status() -> dict:
    return dict(_state)
