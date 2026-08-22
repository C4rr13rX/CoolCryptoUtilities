"""
Quantum-safe, password-based authentication.

Replaces the emailed magic link.  A magic link has two properties that are bad
here: it puts a live credential in an inbox (anyone reading the mailbox is
authenticated), and it makes login depend on SES deliverability and cost.  A
password never leaves the client in usable form and costs nothing to verify.

Threat model
------------
The concrete worry is **harvest-now-decrypt-later**: an adversary records TLS
traffic today and decrypts it once a cryptographically-relevant quantum
computer exists.  TLS 1.3's key exchange (X25519/ECDHE) is broken by Shor's
algorithm; a password captured that way is still valid years later, because
people do not rotate passwords on a quantum-computing schedule.

So the password is *never* protected only by TLS.  Every login wraps it in a
second, post-quantum layer inside the TLS tunnel.

Construction
------------
1. **ML-KEM-768** (FIPS 203, formerly Kyber) for transport.  The server
   publishes an encapsulation key; the client encapsulates a shared secret and
   sends the ciphertext.  Recovering the secret requires breaking the
   Module-LWE problem, for which no efficient quantum algorithm is known.
   768 is the level-3 parameter set -- the one NIST recommends as the default.

2. **HKDF-SHA-384** to derive an AES key from that shared secret, bound to a
   per-attempt transcript so a ciphertext replayed against a different
   challenge derives a different key and fails to decrypt.

3. **AES-256-GCM** to seal the password.  Symmetric primitives are only
   weakened quadratically by Grover's algorithm, so AES-256 keeps ~128-bit
   post-quantum strength.

4. **Argon2id** for the stored verifier.  Memory-hard, so a stolen database
   resists GPU and ASIC cracking; Grover offers no useful speedup against a
   memory-bound function.  Parameters follow OWASP: 64 MiB, t=3, p=4.

5. **HMAC-SHA-384 session tokens** rather than JWT.  This deliberately differs
   from C4rr13rX's `jsonwebtoken`: JWT's algorithm agility is a footgun (the
   `alg: none` and RS256->HS256 confusion classes), and we need none of it for
   a single-issuer, single-audience token.  A fixed-algorithm HMAC has no
   negotiable parameters to confuse.

What this does NOT claim
------------------------
The server still terminates ordinary TLS, so metadata (timing, IP, request
sizes) has classical protection only.  This protects the *password and session
secret*, which are the parts whose disclosure is catastrophic later.  It also
does not defend a compromised client: malware on the device sees the password
before it is ever encrypted.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("serverless.hybrid.pq_auth")

# --- tunables ---------------------------------------------------------------
# A challenge only has to survive one round trip. Two minutes covers a slow
# network and clock skew without leaving a long replay window.
CHALLENGE_TTL_SECONDS = int(os.getenv("PQ_CHALLENGE_TTL", "120"))
SESSION_TTL_SECONDS = int(os.getenv("PQ_SESSION_TTL", "86400"))

# OWASP's Argon2id baseline. Lambda charges by GB-second, so 64 MiB for ~100ms
# is a fraction of a cent per login while costing an attacker the same memory
# per guess -- the asymmetry that makes it worth paying for.
ARGON2_MEMORY_KIB = int(os.getenv("PQ_ARGON2_MEMORY_KIB", "65536"))
ARGON2_TIME_COST = int(os.getenv("PQ_ARGON2_TIME", "3"))
ARGON2_PARALLELISM = int(os.getenv("PQ_ARGON2_PARALLELISM", "4"))

# Lock an account after repeated failures. Counted server-side against the
# stored record, so it survives the attacker rotating IPs.
MAX_FAILED_ATTEMPTS = int(os.getenv("PQ_MAX_FAILED", "8"))
LOCKOUT_SECONDS = int(os.getenv("PQ_LOCKOUT_SECONDS", "900"))

TABLE_USERS = "auth_users"
TABLE_CHALLENGES = "auth_challenges"
TABLE_SESSIONS = "auth_sessions"


class AuthError(Exception):
    """Authentication failure that is safe to report to the caller."""

    def __init__(self, message: str, code: str = "auth_failed") -> None:
        super().__init__(message)
        self.code = code


# --- encoding helpers -------------------------------------------------------
def b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


# --- post-quantum KEM -------------------------------------------------------
def _kem():
    """ML-KEM-768 (FIPS 203). Imported lazily so this module loads without it."""
    try:
        from kyber_py.ml_kem import ML_KEM_768

        return ML_KEM_768
    except ImportError as exc:  # pragma: no cover
        raise AuthError(
            "post-quantum KEM unavailable (kyber-py not installed)",
            code="kem_unavailable",
        ) from exc


def generate_server_keypair() -> tuple[str, str]:
    """Return (encapsulation_key_b64, decapsulation_key_b64)."""
    ek, dk = _kem().keygen()
    return b64e(ek), b64e(dk)


def _derive_key(shared_secret: bytes, transcript: bytes) -> bytes:
    """
    HKDF-SHA-384 -> 32-byte AES key.

    ``transcript`` binds the key to this specific attempt (challenge id +
    encapsulation key). Without it, a captured ciphertext could be replayed
    against a different challenge and still derive a usable key.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(
        algorithm=hashes.SHA384(),
        length=32,
        salt=b"coolcrypto-pq-auth-v1",
        info=transcript,
    ).derive(shared_secret)


def seal_password(password: str, server_ek_b64: str, transcript: bytes) -> dict:
    """
    Client-side: encapsulate to the server key and seal the password.

    Ships here so the test suite and any Python client exercise byte-for-byte
    the same construction the browser implements.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    shared, ct = _kem().encaps(b64d(server_ek_b64))
    key = _derive_key(shared, transcript)
    nonce = secrets.token_bytes(12)
    sealed = AESGCM(key).encrypt(nonce, password.encode("utf-8"), transcript)
    return {"kem_ct": b64e(ct), "nonce": b64e(nonce), "sealed": b64e(sealed)}


def open_password(envelope: dict, server_dk_b64: str, transcript: bytes) -> str:
    """Server-side inverse of :func:`seal_password`."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    try:
        shared = _kem().decaps(b64d(server_dk_b64), b64d(envelope["kem_ct"]))
        key = _derive_key(shared, transcript)
        plain = AESGCM(key).decrypt(
            b64d(envelope["nonce"]), b64d(envelope["sealed"]), transcript
        )
        return plain.decode("utf-8")
    except AuthError:
        raise
    except Exception as exc:  # noqa: BLE001
        # Never leak which step failed: distinguishing "bad ciphertext" from
        # "bad tag" hands an attacker an oracle.
        raise AuthError("could not open credential envelope") from exc


# --- password verifier ------------------------------------------------------
def _hasher():
    from argon2 import PasswordHasher

    return PasswordHasher(
        memory_cost=ARGON2_MEMORY_KIB,
        time_cost=ARGON2_TIME_COST,
        parallelism=ARGON2_PARALLELISM,
    )


def hash_password(password: str) -> str:
    return _hasher().hash(password)


def verify_password(stored_hash: str, password: str) -> bool:
    from argon2.exceptions import VerificationError, VerifyMismatchError

    try:
        return _hasher().verify(stored_hash, password)
    except (VerifyMismatchError, VerificationError):
        return False
    except Exception:  # noqa: BLE001
        return False


def password_strength_error(password: str) -> str | None:
    """
    Reject weak passwords at registration.

    Length carries far more entropy than character-class rules, so the floor is
    12 characters rather than a symbol/digit checklist that mostly produces
    'Password1!'.
    """
    if len(password) < 12:
        return "Password must be at least 12 characters."
    if len(password) > 1024:
        # Argon2 cost is bounded by input size; refuse a memory-exhaustion knob.
        return "Password must be at most 1024 characters."
    # Reject the password *being* one of these (with trivial decoration),
    # not merely containing them. A substring test fails a genuinely strong
    # passphrase that happens to include the word "password", which pushes
    # users toward shorter, weaker choices.
    stripped = "".join(ch for ch in password.lower() if ch.isalnum())
    for bad in ("password", "123456", "qwerty", "letmein", "admin", "welcome"):
        if stripped == bad or stripped.strip("0123456789") == bad:
            return "Password is a common, easily-guessed sequence."
    if len(set(password)) < 5:
        return "Password must use at least 5 distinct characters."
    return None


# --- session tokens ---------------------------------------------------------
def _session_secret() -> bytes:
    secret = os.getenv("PQ_SESSION_SECRET", "")
    if not secret:
        raise AuthError("PQ_SESSION_SECRET is not configured", code="misconfigured")
    if len(secret) < 32:
        raise AuthError(
            "PQ_SESSION_SECRET must be at least 32 characters", code="misconfigured"
        )
    return secret.encode("utf-8")


def issue_session_token(user_id: int, email: str) -> tuple[str, dict]:
    """
    Mint an HMAC-SHA-384 session token.

    Format: base64url(payload).base64url(mac) -- JWT-shaped but with the
    algorithm fixed in code rather than declared in a header the caller
    controls, which removes the `alg` confusion attacks entirely.
    """
    now = int(time.time())
    payload = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + SESSION_TTL_SECONDS,
        "jti": secrets.token_urlsafe(16),
    }
    body = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).rstrip(b"=")
    mac = hmac.new(_session_secret(), body, hashlib.sha384).digest()
    token = f"{body.decode()}.{base64.urlsafe_b64encode(mac).rstrip(b'=').decode()}"
    return token, payload


def verify_session_token(token: str) -> dict:
    """Validate a session token and return its payload, or raise AuthError."""
    try:
        body_s, mac_s = token.split(".", 1)
    except ValueError as exc:
        raise AuthError("malformed session token") from exc

    body = body_s.encode()
    expected = hmac.new(_session_secret(), body, hashlib.sha384).digest()
    given = base64.urlsafe_b64decode(mac_s + "=" * (-len(mac_s) % 4))
    # compare_digest: a byte-wise comparison leaks the correct prefix through
    # timing, which is enough to forge a MAC one byte at a time.
    if not hmac.compare_digest(expected, given):
        raise AuthError("invalid session signature")

    payload = json.loads(base64.urlsafe_b64decode(body_s + "=" * (-len(body_s) % 4)))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise AuthError("session expired", code="session_expired")
    return payload


# --- flows ------------------------------------------------------------------
@dataclass
class AuthService:
    """Registration and login over the S3-backed store."""

    storage: Any

    # -- registration --
    def register(self, email: str, password: str) -> dict:
        email = (email or "").strip().lower()
        if "@" not in email or len(email) > 254:
            raise AuthError("A valid email address is required.", code="bad_email")

        # Existence is checked before strength so a duplicate signup reports
        # the real reason. Reversing these makes a second attempt on an
        # existing account complain about the password instead, which sends
        # the user off changing something that was never the problem.
        if self.storage.find_by(TABLE_USERS, "email", email):
            # An explicit conflict is standard for registration, and the
            # address is already known to whoever is typing it.
            raise AuthError("That account already exists.", code="exists")

        problem = password_strength_error(password or "")
        if problem:
            raise AuthError(problem, code="weak_password")

        record = self.storage.insert_row(
            TABLE_USERS,
            {
                "email": email,
                "password_hash": hash_password(password),
                "created_at": time.time(),
                "failed_attempts": 0,
                "locked_until": 0,
                "is_active": True,
            },
        )
        self.storage.set_index(TABLE_USERS, "email", email, record["id"])
        return {"id": record["id"], "email": email}

    # -- login step 1: challenge --
    def begin_login(self) -> dict:
        """
        Issue a per-attempt ML-KEM keypair.

        A fresh keypair per attempt (rather than one long-lived server key)
        means compromising a decapsulation key exposes only the attempts still
        inside the 2-minute window, not every past login.
        """
        ek, dk = generate_server_keypair()
        challenge_id = secrets.token_urlsafe(24)
        self.storage.put_row(
            TABLE_CHALLENGES,
            challenge_id,
            {
                "id": challenge_id,
                "dk": dk,
                "ek": ek,
                "created_at": time.time(),
                "expires_at": time.time() + CHALLENGE_TTL_SECONDS,
                "used": False,
            },
        )
        return {
            "challenge_id": challenge_id,
            "server_key": ek,
            "kem": "ML-KEM-768",
            "expires_in": CHALLENGE_TTL_SECONDS,
        }

    @staticmethod
    def transcript_for(challenge_id: str, server_ek_b64: str) -> bytes:
        """Bytes both sides bind into the KDF and use as AEAD associated data."""
        return f"{challenge_id}|{server_ek_b64}".encode()

    # -- login step 2: complete --
    def complete_login(self, challenge_id: str, email: str, envelope: dict) -> dict:
        challenge = self.storage.get_row(TABLE_CHALLENGES, challenge_id)
        if not challenge:
            raise AuthError("Unknown or expired challenge.", code="bad_challenge")

        # Single-use: without this, a captured envelope replays forever inside
        # the TTL window.
        if challenge.get("used"):
            raise AuthError("Challenge already used.", code="bad_challenge")
        if float(challenge.get("expires_at", 0)) < time.time():
            raise AuthError("Challenge expired.", code="bad_challenge")

        challenge["used"] = True
        self.storage.put_row(TABLE_CHALLENGES, challenge_id, challenge)

        email = (email or "").strip().lower()
        user = self.storage.find_by(TABLE_USERS, "email", email)

        transcript = self.transcript_for(challenge_id, challenge["ek"])
        password = open_password(envelope, challenge["dk"], transcript)

        if user is None:
            # Spend comparable work on an unknown address so response time does
            # not reveal which emails are registered.
            hash_password(password)
            raise AuthError("Invalid email or password.")

        locked_until = float(user.get("locked_until") or 0)
        if locked_until > time.time():
            raise AuthError(
                "Account temporarily locked. Try again later.", code="locked"
            )

        if not user.get("is_active", True):
            raise AuthError("Account is disabled.", code="disabled")

        if not verify_password(user.get("password_hash", ""), password):
            failed = int(user.get("failed_attempts", 0)) + 1
            user["failed_attempts"] = failed
            if failed >= MAX_FAILED_ATTEMPTS:
                user["locked_until"] = time.time() + LOCKOUT_SECONDS
                user["failed_attempts"] = 0
            self.storage.put_row(TABLE_USERS, user["id"], user)
            raise AuthError("Invalid email or password.")

        if user.get("failed_attempts") or user.get("locked_until"):
            user["failed_attempts"] = 0
            user["locked_until"] = 0
        user["last_login"] = time.time()
        self.storage.put_row(TABLE_USERS, user["id"], user)

        token, payload = issue_session_token(user["id"], email)
        self.storage.put_row(
            TABLE_SESSIONS,
            payload["jti"],
            {
                "id": payload["jti"],
                "user_id": user["id"],
                "issued_at": payload["iat"],
                "expires_at": payload["exp"],
                "revoked": False,
            },
        )
        return {"token": token, "expires_at": payload["exp"], "user": {
            "id": user["id"], "email": email,
        }}

    # -- session --
    def check_session(self, token: str) -> dict:
        """Verify a token *and* confirm the server-side session is still live."""
        payload = verify_session_token(token)
        session = self.storage.get_row(TABLE_SESSIONS, payload.get("jti"))
        # The MAC alone cannot express revocation -- a stolen token would stay
        # valid until expiry. The stored row is what makes logout real.
        if session is None or session.get("revoked"):
            raise AuthError("Session revoked.", code="session_revoked")
        return payload

    def revoke_session(self, token: str) -> None:
        try:
            payload = verify_session_token(token)
        except AuthError:
            return  # logging out an invalid token is a no-op, not an error
        jti = payload.get("jti")
        session = self.storage.get_row(TABLE_SESSIONS, jti)
        if session:
            session["revoked"] = True
            self.storage.put_row(TABLE_SESSIONS, jti, session)
