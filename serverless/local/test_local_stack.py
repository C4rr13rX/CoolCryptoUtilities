#!/usr/bin/env python3
"""
Integration tests against the LocalStack deployment.

These invoke the *deployed* Lambda functions through the real AWS APIs, so
they exercise the actual event shapes API Gateway and EventBridge produce --
not a mocked request factory.  A pass here means the handler contract, the
settings overlay, the bundle layout and the DB wiring all work together.

Run:  python serverless/local/test_local_stack.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.config import Config

ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")
REGION = os.getenv("AWS_REGION", "us-east-1")
HERE = Path(__file__).resolve().parent

_cfg = Config(retries={"max_attempts": 3}, read_timeout=310, connect_timeout=10)
_kw = dict(
    endpoint_url=ENDPOINT,
    region_name=REGION,
    aws_access_key_id="test",
    aws_secret_access_key="test",
    config=_cfg,
)
lam = boto3.client("lambda", **_kw)
sched = boto3.client("scheduler", **_kw)
apigw_v2 = boto3.client("apigatewayv2", **_kw)
apigw_v1 = boto3.client("apigateway", **_kw)

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> bool:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def invoke(fn: str, payload: dict) -> dict:
    resp = lam.invoke(
        FunctionName=fn,
        Payload=json.dumps(payload).encode(),
    )
    raw = resp["Payload"].read().decode()
    if resp.get("FunctionError"):
        return {"__error__": resp["FunctionError"], "raw": raw}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"__raw__": raw}


def http_event(path: str, method: str = "GET", headers: dict | None = None) -> dict:
    """Build the API Gateway HTTP API v2 payload-format-2.0 event."""
    return {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": path,
        "rawQueryString": "",
        "headers": {"host": "localhost", "x-forwarded-proto": "https",
                    **(headers or {})},
        "requestContext": {
            "http": {"method": method, "path": path, "protocol": "HTTP/1.1",
                     "sourceIp": "127.0.0.1"},
            "stage": "$default", "domainName": "localhost",
        },
        "isBase64Encoded": False,
    }


def main() -> int:
    print("=" * 62)
    print("  LocalStack integration tests")
    print("=" * 62)

    # -- 1. functions exist and are active ------------------------------
    print("\n[1] Lambda functions")
    names = {f["FunctionName"] for f in lam.list_functions()["Functions"]}
    for fn in ("coolcrypto-http", "coolcrypto-cron", "coolcrypto-ws",
               "coolcrypto-ws-push", "coolcrypto-admin"):
        check(f"{fn} deployed", fn in names)

    # -- 2. migrations --------------------------------------------------
    print("\n[2] Database migrations")
    r = invoke("coolcrypto-admin", {"command": "migrate", "args": ["--noinput"]})
    check("migrate runs", r.get("status") == "ok",
          str(r.get("error") or r.get("__error__") or "")[:300])
    if r.get("status") == "ok":
        out = r.get("output", "")
        check("serverless app migrated",
              "serverless" in out or "No migrations to apply" in out)

    # -- 3. the command whitelist actually blocks -----------------------
    print("\n[3] Admin command whitelist")
    r = invoke("coolcrypto-admin", {"command": "shell"})
    check("non-whitelisted command rejected", r.get("status") == "rejected")

    # -- 4. Django serves HTTP through Mangum ---------------------------
    print("\n[4] HTTP handler")
    r = invoke("coolcrypto-http", http_event("/api/console/status/"))
    status = r.get("statusCode")
    # Any real HTTP status proves the whole chain worked: event parsed,
    # Django booted, URL resolved, response serialised. 500 does not.
    check("returns an HTTP response", isinstance(status, int),
          str(r.get("__error__") or r.get("raw", ""))[:400])
    check("not a server error", status != 500, f"status={status}")

    # An unknown *page* redirects to the dashboard -- core.urls has a catch-all
    # that predates this work. An unknown *API* path must still 404 rather than
    # redirect, or clients would parse an HTML dashboard as JSON.
    r = invoke("coolcrypto-http", http_event("/api/definitely-not-real/"))
    check("404 for unknown API path", r.get("statusCode") == 404,
          f"status={r.get('statusCode')}")

    r = invoke("coolcrypto-http", http_event("/"))
    check("home page renders", r.get("statusCode") == 200,
          f"status={r.get('statusCode')}")

    r = invoke("coolcrypto-http", http_event("/admin/login/"))
    check("admin login reachable", r.get("statusCode") in (200, 301, 302),
          f"status={r.get('statusCode')}")

    # WhiteNoise serves assets from inside the bundle; a 404 here means
    # collectstatic did not run during packaging.
    r = invoke("coolcrypto-http", http_event("/static/admin/css/base.css"))
    check("static asset served by WhiteNoise", r.get("statusCode") == 200,
          f"status={r.get('statusCode')}")

    # A cookieless POST must be rejected by CSRF, not crash. This confirms
    # SECURE_PROXY_SSL_HEADER/CSRF_TRUSTED_ORIGINS are coherent behind the
    # gateway rather than producing a 500.
    r = invoke("coolcrypto-http",
               http_event("/api/console/start/", method="POST"))
    check("POST handled (no 5xx)", int(r.get("statusCode", 500)) < 500,
          f"status={r.get('statusCode')}")

    # -- 5. cron handler ------------------------------------------------
    print("\n[5] Cron handler (EventBridge)")
    r = invoke("coolcrypto-cron", {"task_id": "__nonexistent__"})
    check("unknown task reported cleanly", r.get("status") == "not_found",
          str(r.get("__error__") or "")[:300])

    # -- 6. websocket lifecycle ----------------------------------------
    print("\n[6] WebSocket handler")
    cid = f"test-conn-{int(time.time())}"
    ctx = {"requestContext": {"routeKey": "$connect", "connectionId": cid,
                              "domainName": "localhost", "stage": "prod"},
           "queryStringParameters": {"channel": "console.logs"}}
    r = invoke("coolcrypto-ws", ctx)
    check("$connect accepted", r.get("statusCode") == 200,
          str(r.get("__error__") or r.get("raw", ""))[:300])

    bad = dict(ctx, queryStringParameters={"channel": "bogus"})
    r = invoke("coolcrypto-ws", bad)
    check("unknown channel rejected", r.get("statusCode") == 400)

    r = invoke("coolcrypto-ws-push", {})
    check("broadcast runs", r.get("status") == "ok",
          str(r.get("__error__") or "")[:300])

    dis = {"requestContext": {"routeKey": "$disconnect", "connectionId": cid}}
    r = invoke("coolcrypto-ws", dis)
    check("$disconnect cleans up", r.get("statusCode") == 200)

    # -- 7. schedules ---------------------------------------------------
    print("\n[7] EventBridge schedules")
    got = {s["Name"] for s in sched.list_schedules()["Schedules"]}
    for s in ("coolcrypto-auto-pipeline", "coolcrypto-weekly-bootstrap",
              "coolcrypto-ws-broadcast"):
        check(f"{s} exists", s in got)

    # -- 8. API Gateway -------------------------------------------------
    print("\n[8] API Gateway")
    rest = {a["name"]: a for a in apigw_v1.get_rest_apis()["items"]}
    check("HTTP REST API exists", "coolcrypto-http" in rest)
    if "coolcrypto-http" in rest:
        api_id = rest["coolcrypto-http"]["id"]
        paths = {r.get("path") for r in
                 apigw_v1.get_resources(restApiId=api_id)["items"]}
        # Both are required: {proxy+} does not match the root path, so without
        # "/" the site would 403 on its own home page.
        check("root resource present", "/" in paths, str(paths))
        check("greedy proxy resource present", "/{proxy+}" in paths, str(paths))

    # apigatewayv2 is a LocalStack Pro feature; on Community it raises rather
    # than returning an empty list, so a failure here is not a code defect.
    try:
        apis = {a["Name"]: a for a in apigw_v2.get_apis()["Items"]}
        check("WebSocket API exists", "coolcrypto-ws" in apis)
        if "coolcrypto-ws" in apis:
            check("WS protocol correct",
                  apis["coolcrypto-ws"]["ProtocolType"] == "WEBSOCKET")
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIP] WebSocket API -- apigatewayv2 unavailable "
              f"({type(exc).__name__}); handlers covered in section 6")

    print("\n" + "=" * 62)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"    FAILED: {f}")
    print("=" * 62)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
