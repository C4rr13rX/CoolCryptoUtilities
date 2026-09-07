"""Is the wizard brain able to answer at all -- and can you tell if it isn't?

``trading.brain_bridge`` returns ``(None, 0.0)`` from ``predict_outcome`` and
``query_confidence`` for EVERY failure: a refused TCP connect, a 30-second
timeout, a malformed body, a node serving an empty fabric. Its callers -- the
money path at ``trading/bot.py:5014`` and ``services/ga_service.py:296`` --
document that contract as "the caller already treats a None answer as no
opinion". So a brain that is DOWN and a brain that is THINKING ABOUT THIS INPUT
AND HAS NOTHING TO SAY produce the same value, and the down case raises nothing.

Measured 2026-09-07 08:35, which is why this module exists:

    :8090  (the node every BRAIN_ENDPOINT default points at)
        GET  /health          -> 200, uptime 116292s (32.3 hours)
        GET  /brain/stats     -> 521224 concepts / 10725783 terminals
        POST /brain/predict   -> TIMED OUT at 25s
        POST /brain/consolidate -> TIMED OUT at 30s
        process RSS           -> 34 MB against a 15.7 GB brain.wbrain

    :8091  (same binary, launched with --config node_config.json api)
        POST /brain/predict   -> answered in 0.00s
        ...with known_atom_count 0, outside_grounding true -- an EMPTY fabric

Two different broken states, both of which reach the trade path as "no
opinion", and neither of which logged anything. The 32 hours are the cost: for
that entire window every brain query in the live lane returned nothing and no
counter, log line or status field said so. This is the same shape as the status
line reading "0 ticks/10m with no error anywhere" -- nothing was FAILING, so
nothing reported a failure.

The distinction this module draws is the one the callers actually need:

    unreachable  no TCP listener              -- the node is not running
    blocked      listener answers, fabric call exceeded the deadline
    empty        fabric answers, holds no atoms to answer FROM
    ready        answers inside the deadline against a non-empty fabric

Only ``ready`` makes a ``(None, 0.0)`` reply mean "no opinion about this
input". Under every other state a None is an OUTAGE, and ``usable`` is False so
a caller can refuse to treat silence as a forecast.

Deliberately dependency-free (``socket`` + ``urllib``) and transport-injectable
so the test suite exercises all four states without a node. Never raises: a
health probe that can itself throw is one more silent failure.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

#: States, most-broken first. Byte-disjoint words so a state can never be
#: read as a prefix of another when it lands in a log line or a status field.
UNREACHABLE = "unreachable"
BLOCKED = "blocked"
EMPTY = "empty"
READY = "ready"

#: Order matters: a probe reports the FIRST state it can prove, so a node that
#: is both blocked and empty reports blocked (you cannot measure emptiness
#: through a call that never returned).
STATES = (UNREACHABLE, BLOCKED, EMPTY, READY)

#: How long a fabric call may take before the brain counts as blocked, in
#: seconds. This is NOT the bridge's 30s transport timeout -- that ceiling
#: exists so a slow call still lands. This is the deadline past which the brain
#: is no use to a trade decision. Measured for scale: /brain/predict runs ~5ms
#: on a small fabric and ~211ms mean / 276ms p95 at 6.7M terminals (exp15), so
#: a healthy node at production scale sits two orders of magnitude inside 8s.
DEFAULT_DEADLINE = float(os.getenv("BRAIN_HEALTH_DEADLINE_SECONDS", "8.0"))

#: Probe frame. Intentionally nonsense: it must not collide with a trained
#: situation, because a probe that could be ANSWERED would make "the fabric
#: replied" and "the fabric knows this input" the same measurement again. We
#: are asking whether the node can complete a fabric round trip, not what it
#: thinks.
PROBE_FRAME = "brainhealth probe zzzz9 qqqq7"


def _b64url(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


@dataclass
class BrainHealth:
    """What the brain can do right now, with the numbers behind it."""

    state: str
    endpoint: str
    #: Seconds the fabric round trip took, or the deadline it blew through.
    latency_seconds: float = 0.0
    #: Atoms the node recognised in the probe frame. Zero on an empty fabric.
    known_atom_count: int = 0
    #: Fabric size from /brain/stats. Zero or None when stats did not answer.
    total_terminals: Optional[int] = None
    resident_terminals: Optional[int] = None
    total_concepts: Optional[int] = None
    #: Human-readable cause, always populated -- a state with no reason is
    #: the silence this module was written to end.
    reason: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """True only when a ``(None, 0.0)`` reply means 'no opinion'.

        Every caller that sizes, gates or scores on a brain answer should read
        this before reading the answer. Under any other state the brain is not
        declining to forecast, it is failing to reply.
        """
        return self.state == READY

    @property
    def silence_is_an_outage(self) -> bool:
        """The inverse of :attr:`usable`, named for the decision it drives."""
        return not self.usable

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["usable"] = self.usable
        return payload

    def summary(self) -> str:
        return (f"brain {self.state} at {self.endpoint} "
                f"({self.latency_seconds:.2f}s): {self.reason}")


def _default_stats(endpoint: str, timeout: float) -> Optional[Dict[str, Any]]:
    """GET /brain/stats. None when it does not answer inside `timeout`.

    Kept separate from the fabric probe on purpose: on the 2026-09-07 node
    stats answered instantly for hours while every fabric call blocked, so
    stats ALONE proves only that the HTTP server is alive. It is read for the
    fabric-size numbers, never as evidence of readiness.
    """
    try:
        with urllib.request.urlopen(f"{endpoint}/brain/stats", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def _default_predict(endpoint: str, frame: str, timeout: float) -> Optional[Dict[str, Any]]:
    """POST /brain/predict -- the read-only fabric round trip.

    /brain/predict is chosen over /brain/observe or /brain/consolidate because
    it takes the fabric lock without writing: a health probe must never train
    the brain it is measuring.
    """
    payload = json.dumps({
        "query_pool": int(os.getenv("WIZARD_MARKET_INPUT_POOL", "1")),
        "target_pool": int(os.getenv("WIZARD_MARKET_OUTCOME_POOL", "3")),
        "frame": _b64url(frame),
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint}/brain/predict", data=payload,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def _listening(endpoint: str, timeout: float) -> bool:
    """Is anything accepting TCP at the endpoint's host:port?

    Separates 'the node is not running' from 'the node is running and stuck',
    which are different operator actions: start it, versus find what holds the
    fabric lock.
    """
    parsed = urlparse(endpoint)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=min(timeout, 5.0)):
            return True
    except OSError:
        return False


def probe(
    endpoint: Optional[str] = None,
    *,
    deadline: Optional[float] = None,
    predict: Optional[Callable[[str, str, float], Optional[Dict[str, Any]]]] = None,
    stats: Optional[Callable[[str, float], Optional[Dict[str, Any]]]] = None,
    listening: Optional[Callable[[str, float], bool]] = None,
) -> BrainHealth:
    """Measure what the brain can do, without raising and without training it.

    Reads the same ``BRAIN_ENDPOINT`` default the bridge uses, so a probe and a
    trade query always talk to the same node -- a health check aimed somewhere
    else is worse than none.

    The transports are injectable so every state is reachable in a test with no
    node running. Nothing here writes to the fabric.
    """
    target = (endpoint or os.getenv("BRAIN_ENDPOINT", "http://127.0.0.1:8090")).rstrip("/")
    limit = float(deadline if deadline is not None else DEFAULT_DEADLINE)
    predict_fn = predict or _default_predict
    stats_fn = stats or _default_stats
    listening_fn = listening or _listening

    try:
        alive = listening_fn(target, limit)
    except Exception:
        alive = False
    if not alive:
        return BrainHealth(
            state=UNREACHABLE, endpoint=target,
            reason="nothing is accepting TCP at the endpoint -- the node is not running",
        )

    # Fabric size is read first but judged last: it is diagnostic colour, not
    # evidence. On the node that motivated this module these numbers were
    # healthy (521224 concepts) while the fabric could not answer at all.
    raw_stats = None
    try:
        raw_stats = stats_fn(target, limit)
    except Exception:
        raw_stats = None
    stats_payload = raw_stats if isinstance(raw_stats, dict) else {}

    def _int_or_none(key: str) -> Optional[int]:
        value = stats_payload.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    total_terminals = _int_or_none("total_terminals")
    resident_terminals = _int_or_none("resident_terminals")
    total_concepts = _int_or_none("total_concepts")

    started = time.perf_counter()
    try:
        answer = predict_fn(target, PROBE_FRAME, limit)
    except Exception:
        answer = None
    elapsed = time.perf_counter() - started

    common = dict(
        endpoint=target, latency_seconds=elapsed, total_terminals=total_terminals,
        resident_terminals=resident_terminals, total_concepts=total_concepts,
        detail={"stats_answered": bool(stats_payload)},
    )

    if answer is None:
        # The listener answered but the fabric call did not come back. Whether
        # it timed out or errored, the brain cannot serve a decision, and a
        # caller must not read the resulting None as an abstention.
        served_stats = "; /brain/stats answered, so the HTTP server is alive and the fabric lock is the suspect" if stats_payload else ""
        return BrainHealth(
            state=BLOCKED,
            reason=(f"fabric call did not return within {limit:.1f}s{served_stats}"),
            **common,
        )

    if elapsed > limit:
        return BrainHealth(
            state=BLOCKED,
            reason=f"fabric answered but took {elapsed:.2f}s, past the {limit:.1f}s decision deadline",
            **common,
        )

    try:
        known = int(answer.get("known_atom_count") or 0)
    except (TypeError, ValueError, AttributeError):
        known = 0
    common["known_atom_count"] = known

    # An empty fabric answers FASTER than a loaded one -- :8091 replied in
    # 0.00s -- so latency alone would grade it healthiest of all. Emptiness has
    # to be read off the fabric's own size, not off how quickly it said nothing.
    if not total_terminals:
        return BrainHealth(
            state=EMPTY,
            reason=("fabric answers but holds no terminals -- an unloaded or fresh brain; "
                    "every prediction it returns is groundless"),
            **common,
        )

    return BrainHealth(
        state=READY,
        reason=(f"answered in {elapsed:.2f}s against {total_terminals} terminals -- "
                f"a null answer from this node means no opinion, not an outage"),
        **common,
    )


def main() -> int:
    """CLI: ``python -m services.brain_health``. Exit 0 only when READY.

    Non-zero exit on every broken state so a supervisor, a pass, or a shell
    check can gate on the brain being able to answer rather than on the node
    merely being up -- which is what /health has been reporting all along.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Probe whether the wizard brain can answer.")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--deadline", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    health = probe(args.endpoint, deadline=args.deadline)
    print(json.dumps(health.as_dict(), indent=2) if args.json else health.summary())
    return 0 if health.usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
