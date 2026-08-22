#!/usr/bin/env python3
"""
Integration tests for the hybrid data layer and quantum-safe auth.

Invokes the *deployed* Lambdas through the AWS APIs with the event shapes API
Gateway actually produces, so a pass here exercises the real handler contract
rather than a mocked request.

Run:  python serverless/local/test_hybrid_stack.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from serverless.hybrid.pq_auth import AuthService, seal_password  # noqa: E402

ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")
REGION = os.getenv("AWS_REGION", "us-east-1")

lam = boto3.client(
    "lambda", endpoint_url=ENDPOINT, region_name=REGION,
    aws_access_key_id="test", aws_secret_access_key="test",
    config=Config(retries={"max_attempts": 3}, read_timeout=310),
)

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def http_event(path: str, method: str = "POST", body: dict | None = None,
               token: str | None = None) -> dict:
    headers = {"host": "localhost", "content-type": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    return {
        "version": "2.0",
        "rawPath": path,
        "headers": headers,
        "requestContext": {"http": {"method": method, "path": path},
                           "stage": "prod", "domainName": "localhost"},
        "body": json.dumps(body or {}),
        "isBase64Encoded": False,
    }


def invoke(fn: str, event: dict) -> dict:
    resp = lam.invoke(FunctionName=fn, Payload=json.dumps(event).encode())
    raw = resp["Payload"].read().decode()
    if resp.get("FunctionError"):
        return {"__error__": resp["FunctionError"], "raw": raw[:400]}
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return {"__raw__": raw[:400]}
    if isinstance(out, dict) and "body" in out:
        try:
            out["parsed"] = json.loads(out["body"])
        except (json.JSONDecodeError, TypeError):
            out["parsed"] = {}
    return out


def main() -> int:
    print("=" * 62)
    print("  Hybrid storage + quantum-safe auth")
    print("=" * 62)

    # -- 1. login as the migrated admin account -------------------------
    print("\n[1] Quantum-safe login (admin)")
    r = invoke("coolcrypto-auth", http_event("/auth/challenge"))
    challenge = r.get("parsed", {})
    check("challenge issued", r.get("statusCode") == 200 and "server_key" in challenge,
          str(r.get("__error__") or r.get("raw", ""))[:300])
    if "server_key" not in challenge:
        print("cannot continue without a challenge")
        return 1

    check("uses ML-KEM-768", challenge.get("kem") == "ML-KEM-768")

    # Seal with the same construction the browser uses.
    transcript = AuthService.transcript_for(
        challenge["challenge_id"], challenge["server_key"]
    )
    envelope = seal_password("admin", challenge["server_key"], transcript)

    r = invoke("coolcrypto-auth", http_event("/auth/login", body={
        "challenge_id": challenge["challenge_id"],
        "email": "admin",
        "envelope": envelope,
    }))
    login = r.get("parsed", {})
    check("admin/admin logs in", r.get("statusCode") == 200 and "token" in login,
          json.dumps(login)[:300])
    token = login.get("token", "")

    # -- 2. replay + wrong password -------------------------------------
    print("\n[2] Credential handling")
    r = invoke("coolcrypto-auth", http_event("/auth/login", body={
        "challenge_id": challenge["challenge_id"],
        "email": "admin", "envelope": envelope,
    }))
    check("challenge is single-use", r.get("statusCode") in (400, 401),
          f"status={r.get('statusCode')}")

    r2 = invoke("coolcrypto-auth", http_event("/auth/challenge"))
    ch2 = r2.get("parsed", {})
    tr2 = AuthService.transcript_for(ch2["challenge_id"], ch2["server_key"])
    bad = seal_password("not-the-password", ch2["server_key"], tr2)
    r = invoke("coolcrypto-auth", http_event("/auth/login", body={
        "challenge_id": ch2["challenge_id"], "email": "admin", "envelope": bad,
    }))
    check("wrong password rejected", r.get("statusCode") == 401,
          f"status={r.get('statusCode')}")

    # A plaintext password must never be accepted, even with a valid challenge.
    r3 = invoke("coolcrypto-auth", http_event("/auth/challenge"))
    ch3 = r3.get("parsed", {})
    r = invoke("coolcrypto-auth", http_event("/auth/login", body={
        "challenge_id": ch3["challenge_id"], "email": "admin",
        "envelope": {"kem_ct": "AA==", "nonce": "AA==", "sealed": "AA=="},
    }))
    check("malformed envelope rejected", r.get("statusCode") in (400, 401),
          f"status={r.get('statusCode')}")

    # -- 3. session -----------------------------------------------------
    print("\n[3] Sessions")
    r = invoke("coolcrypto-auth", http_event("/auth/session", method="GET", token=token))
    check("session endpoint validates token", r.get("statusCode") == 200,
          json.dumps(r.get("parsed", {}))[:200])

    r = invoke("coolcrypto-auth", http_event("/auth/session", method="GET",
                                             token=token[:-3] + "aaa"))
    check("forged token rejected", r.get("statusCode") == 401)

    # -- 4. hybrid data API --------------------------------------------
    print("\n[4] Hybrid data API")
    r = invoke("coolcrypto-hybrid", http_event("/hybrid/notes", method="GET"))
    check("unauthenticated read blocked", r.get("statusCode") == 401,
          f"status={r.get('statusCode')}")

    r = invoke("coolcrypto-hybrid",
               http_event("/hybrid/auth_users", method="GET", token=token))
    check("credential table not routable", r.get("statusCode") == 404,
          f"status={r.get('statusCode')}")

    r = invoke("coolcrypto-hybrid", http_event(
        "/hybrid/notes", method="POST", token=token,
        body={"title": "hybrid test", "body": "written through to S3"},
    ))
    created = r.get("parsed", {})
    check("insert allocates an id", r.get("statusCode") == 201 and created.get("id"),
          json.dumps(created)[:200])

    if created.get("id"):
        r = invoke("coolcrypto-hybrid", http_event(
            f"/hybrid/notes/{created['id']}", method="GET", token=token))
        check("row reads back", r.get("statusCode") == 200
              and r.get("parsed", {}).get("title") == "hybrid test")

        r = invoke("coolcrypto-hybrid", http_event(
            f"/hybrid/notes/{created['id']}", method="PUT", token=token,
            body={"title": "updated"}))
        check("row updates", r.get("statusCode") == 200
              and r.get("parsed", {}).get("title") == "updated")

    r = invoke("coolcrypto-hybrid", http_event("/hybrid/notes", method="GET", token=token))
    listing = r.get("parsed", {})
    check("list returns items + seq", r.get("statusCode") == 200
          and "items" in listing and "seq" in listing,
          json.dumps(listing)[:200])

    r = invoke("coolcrypto-hybrid",
               http_event("/hybrid/notes/_change", method="GET", token=token))
    check("change feed readable", r.get("statusCode") == 200
          and isinstance(r.get("parsed", {}).get("seq"), int))

    if created.get("id"):
        r = invoke("coolcrypto-hybrid", http_event(
            f"/hybrid/notes/{created['id']}", method="DELETE", token=token))
        check("row deletes", r.get("statusCode") == 204)

    # -- 5. migrated data ----------------------------------------------
    print("\n[5] Migrated data reachable")
    os.environ.setdefault("AWS_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
    os.environ.setdefault("HYBRID_BUCKET", "coolcrypto-hybrid")
    from serverless.hybrid.smart_storage import SmartStorage

    st = SmartStorage(bucket="coolcrypto-hybrid",
                      endpoint_url="http://localhost:9000")
    for table, expected in (
        ("securevault_securesetting", 15),
        ("branddozer_deliverysession", 221),   # UUID primary keys
        ("branddozer_researchsource", 119),
    ):
        rows = st.list_table(table)
        check(f"{table} has {expected} rows", len(rows) == expected,
              f"got {len(rows)}")

    print("\n" + "=" * 62)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"    FAILED: {f}")
    print("=" * 62)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
