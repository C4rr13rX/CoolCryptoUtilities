"""Send a short milestone alert to a phone via a carrier email-to-SMS gateway.

Carriers accept mail at <number>@<gateway> and deliver the body as a text.
Verizon's SMS gateway is vtext.com (vzwpix.com is MMS). The gateway keys off
the number alone -- billing address and location are irrelevant.

Sends through AWS SES using the FountainServer profile, from c4rr13rx.com.
That domain is verified in us-east-1 WITH DKIM, which matters: carrier
gateways silently drop unauthenticated mail, so an unsigned sender looks
like it worked and never arrives. The account has production access
(50k/day), not sandbox, so it can send to arbitrary recipients.

Configure by env (all have working defaults; override in .env if needed):

    SMS_TO           9194957881@vtext.com
    SMS_FROM         revenir@c4rr13rx.com
    AWS_SES_PROFILE  FountainServer
    AWS_SES_REGION   us-east-1

Usage:
    python scripts/notify_sms.py "first live trade: +$0.04"
    python scripts/notify_sms.py --body-file notice.txt   # body from a FILE
    python scripts/notify_sms.py --check       # config check, sends nothing
    python scripts/notify_sms.py --test        # sends a real test message

PREFER --body-file FOR ANYTHING LONGER THAN A SENTENCE, and never build a
notice on the command line. Measured 2026-09-10: a notice of roughly 2,400
characters arrived as 978, and one of roughly 1,450 arrived as 331 cut
mid-word at "Median price move is 0.0783 perce". Reading the body from a
file takes it off the command line entirely, and that is still the right
habit -- but it was NOT the whole story, and the rest of it cost four
agents a notice each.

THE SEGMENT CAP WAS THE SECOND CUT, AND IT REPORTED SUCCESS. Measured
2026-09-26 by Cove against this file unchanged since 2026-09-10 18:44
(1f08755): a 3,977-character body segments to 15 parts totalling 4,074
characters, ``MAX_SEGMENTS`` kept 12 of them totalling 3,600, and ``send``
returned 0. So 3 segments and 474 characters were dropped and the caller
was told the notice went out. Four reports blamed "a hard cap of six in the
send loop"; that cap does not exist -- stubbing ``_send_one`` and calling
``send`` on the same body performs 12 sends, and on a 3,001-character body
performs 11 sends of 3,069 characters, which is every segment there is.
The observable half of those reports -- twelve segments needed, a short
delivery, exit 0 -- is entirely explained by the cap binding at 12. The
"six" is not reproducible here and no code path in this module can produce
it.

So the cap now SPEAKS: a notice that cannot be delivered whole exits 3 and
says on stderr how many characters and segments were lost. Raise the cap
with ``SMS_MAX_SEGMENTS`` when a long notice genuinely has to go.

It matters more than a formatting nit: the standing orders put the ASK LAST
in every notice, so either cut removes precisely the decision request and
leaves the evidence that motivated it.

Exit codes: 0 sent whole, 1 not configured/skipped, 2 send failed,
3 SENT BUT INCOMPLETE -- part of the notice was not delivered.
A missing config is not worth crashing a trading loop over, so callers
should treat exit 1 as "skip" -- which is why it differs from 2. Treat 3 as
"the operator did not read all of this".
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

# A CARRIER SEGMENT IS ~160 CHARS. THAT IS A REASON TO SPLIT, NOT TO TRUNCATE.
#
# This was `MAX_BODY = 300` applied as `body[:MAX_BODY]` -- a silent slice that
# cut every notice mid-phrase and reported success. The standing orders blame
# the resulting fragments on agents abbreviating by hand ("no [:90], no
# trailing ...") and state that "the transport no longer truncates anything".
# It did. Measured 2026-09-10: a 1,791-character pass-109 notice went out as
# 300 characters, stopping inside a sentence, and the send printed OK.
#
# A notice must stand alone for someone who has not read the log, so the fix is
# to SEGMENT it: numbered parts, split on whitespace so no word is cut, each
# small enough to survive the gateway. Nothing is dropped.
SEGMENT_BODY = 300
MAX_SEGMENTS = 12

# The cap exists so a runaway body cannot text the operator two hundred times.
# It is NOT a length policy, and it must never be the reason a notice arrives
# half-finished with exit 0 -- see the docstring. Raise it deliberately.
MAX_SEGMENTS_ENV = "SMS_MAX_SEGMENTS"

# segments() takes ``limit`` as a plain int, so "do not cap" needs a number
# rather than None. Anything above the longest notice anyone will ever write.
NO_LIMIT = 1_000_000

DEFAULTS = {
    "SMS_TO": "9194957881@vtext.com",
    "SMS_FROM": "revenir@c4rr13rx.com",
    "AWS_SES_PROFILE": "FountainServer",
    "AWS_SES_REGION": "us-east-1",
}


def _load_env() -> None:
    """Read .env without a hard dependency on python-dotenv."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


def config() -> dict:
    _load_env()
    return {k: (os.getenv(k) or v).strip() for k, v in DEFAULTS.items()}


def _room(size: int, limit: int) -> int:
    """Characters of CONTENT a part may hold once numbering is reserved.

    It depends on ``limit`` -- "(12/12) " is eight characters and
    "(1000000/1000000) " is eighteen -- which is why the full and the capped
    segmentation of one body must be taken at ONE room. Splitting twice with
    two limits gives two different part boundaries, and then "how much did the
    cap drop" is answered against a notice that was never going to be sent.
    """
    return max(32, size - len("(%d/%d) " % (limit, limit)))


def _split(body: str, room: int) -> list:
    """The unnumbered parts, breaking on whitespace only. No cap applied."""
    parts, cur = [], ""
    for word in body.split():
        candidate = (cur + " " + word) if cur else word
        if len(candidate) <= room:
            cur = candidate
            continue
        if cur:
            parts.append(cur)
            cur = ""
        # A single word longer than one segment is cut, because nothing else
        # can be done with it -- but that is a hard break on one token, not a
        # silent slice through the message.
        while len(word) > room:
            parts.append(word[:room])
            word = word[room:]
        cur = word
    if cur:
        parts.append(cur)
    return parts


def _number(parts: list) -> list:
    total = len(parts)
    return ["(%d/%d) %s" % (i, total, p) for i, p in enumerate(parts, 1)]


def segments(body: str, *, size: int = SEGMENT_BODY,
             limit: int = MAX_SEGMENTS) -> list:
    """Split a notice into numbered parts, breaking on whitespace only.

    Pure, so the transport's own truncation is testable without SES. Returns
    ["(1/3) ...", "(2/3) ...", ...], or a single UNPREFIXED part when the whole
    notice already fits -- a one-part notice must not be labelled "(1/1)".

    IT RETURNS THE CAPPED LIST. What the cap dropped is not recoverable from
    the return value, so do not size a notice off ``len(segments(body))`` --
    use ``plan``, which returns the loss alongside it.
    """
    parts, _dropped_chars, _dropped_segs = plan(body, size=size, limit=limit)
    return parts


def max_segments() -> int:
    """The cap, overridable, so a long notice can go whole when it must."""
    _load_env()
    raw = (os.getenv(MAX_SEGMENTS_ENV) or "").strip()
    try:
        val = int(raw)
    except ValueError:
        val = 0
    return val if val > 0 else MAX_SEGMENTS


def plan(body: str, *, size: int = SEGMENT_BODY,
         limit: int = None) -> tuple:  # noqa: RUF013
    """What will be sent, and what the cap will drop -- both, explicitly.

    Returns ``(parts, dropped_chars, dropped_segments)``. ``segments`` alone
    cannot answer the second and third: it returns the capped list, so a
    caller that only counts what it got back cannot tell a complete notice
    from a decapitated one. That is exactly how ``send`` used to return 0
    having delivered 12 of 15 segments.
    """
    limit = max_segments() if limit is None else limit
    body = (body or "").strip()
    if not body:
        return [], 0, 0
    if len(body) <= size:
        return [body], 0, 0

    raw = _split(body, _room(size, limit))
    if len(raw) <= limit:
        return _number(raw), 0, 0

    dropped = raw[limit:]
    kept = raw[:limit]
    # Say what was dropped IN the notice as well as on stderr -- the operator
    # holding the phone is the one person stderr never reaches.
    kept[-1] += " [TRUNCATED: notice exceeded %d segments]" % limit
    return _number(kept), sum(len(p) for p in dropped), len(dropped)


def send(body: str) -> int:
    cfg = config()

    parts, dropped_chars, dropped_segs = plan(body)
    if not parts:
        sys.stderr.write("notify_sms: empty message; nothing to send\n")
        return 1

    aws = shutil.which("aws")
    if not aws:
        sys.stderr.write("notify_sms: aws CLI not found; skipping\n")
        return 1

    rc = 0
    for part in parts:
        rc = _send_one(aws, cfg, part) or rc

    if dropped_segs:
        # LOUD, AND NONZERO. An agent must not be able to write "notice sent"
        # off this call when the ask at the bottom of the notice never left.
        sys.stderr.write(
            "notify_sms: INCOMPLETE -- delivered %d of %d segments; "
            "%d characters (%d segments) were NOT delivered, and they are the "
            "END of the notice, which is where the ask is.\n"
            "  Raise the cap for this send:  %s=%d\n"
            "  Or split the notice.  Exit code 3 means SENT BUT INCOMPLETE.\n"
            % (len(parts), len(parts) + dropped_segs, dropped_chars,
               dropped_segs, MAX_SEGMENTS_ENV, len(parts) + dropped_segs)
        )
        return rc or 3
    return rc


def _send_one(aws: str, cfg: dict, body: str) -> int:

    # Subject stays empty: carrier gateways prepend it, so a subject shows up
    # as a duplicated first line in the text.
    cmd = [
        aws, "--profile", cfg["AWS_SES_PROFILE"], "--region", cfg["AWS_SES_REGION"],
        "ses", "send-email",
        "--from", cfg["SMS_FROM"],
        "--destination", json.dumps({"ToAddresses": [cfg["SMS_TO"]]}),
        "--message", json.dumps({
            "Subject": {"Data": "", "Charset": "UTF-8"},
            "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
        }),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("notify_sms: send failed: %s: %s\n" % (type(exc).__name__, exc))
        return 2

    if out.returncode != 0:
        sys.stderr.write("notify_sms: SES rejected: %s\n" % (out.stderr or "").strip()[:400])
        return 2

    mid = ""
    try:
        mid = (json.loads(out.stdout) or {}).get("MessageId", "")
    except Exception:
        pass
    print("notify_sms: sent to %s (%d chars) %s" % (cfg["SMS_TO"], len(body), mid))
    return 0


def read_body_file(path: str) -> str:
    """The notice, read from a file so it never passes through ``argv``."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def main() -> int:
    args = sys.argv[1:]
    if not args:
        sys.stderr.write(__doc__ or "")
        return 1

    if args[0] == "--check":
        cfg = config()
        for k in DEFAULTS:
            print("%-16s = %s" % (k, cfg[k]))
        print("\naws CLI: %s" % (shutil.which("aws") or "NOT FOUND"))
        return 0 if shutil.which("aws") else 1

    if args[0] == "--test":
        return send("R3V3N!R test message. If you got this, milestone alerts work.")

    if args[0] == "--body-file":
        if len(args) < 2 or not args[1].strip():
            sys.stderr.write("notify_sms: --body-file needs a path\n")
            return 1
        try:
            body = read_body_file(args[1])
        except OSError as exc:
            sys.stderr.write("notify_sms: cannot read %s: %s\n" % (args[1], exc))
            return 1
        if not body.strip():
            sys.stderr.write("notify_sms: %s is empty; nothing to send\n" % args[1])
            return 1
        return send(body)

    # AN UNRECOGNISED FLAG WAS TEXTED TO THE OPERATOR AS THE MESSAGE.
    #
    # This function matched only --check and --test and let everything else
    # fall through to ``send(" ".join(args))``. So `notify_sms.py --body-file
    # notice.txt` -- a plausible guess at an interface that did not exist yet --
    # sent the literal string "--body-file notice.txt" to a real phone and
    # printed "notify_sms: sent to 9194957881@vtext.com (23 chars)" with exit
    # 0. It happened twice on 2026-09-10 before anyone read the source.
    #
    # Reporting SUCCESS for delivering the wrong thing is the worst shape a
    # failure can take, because nothing downstream can detect it: the caller
    # sees 0, the log says sent, and only the person holding the phone knows.
    # So an unknown option is refused rather than transmitted.
    #
    # "--" ends option parsing, for the genuine case of a notice that has to
    # begin with two dashes.
    if args[0] == "--":
        return send(" ".join(args[1:]))

    if args[0].startswith("--"):
        sys.stderr.write(
            "notify_sms: unknown option %s -- refusing to send it as a message.\n"
            "Known options: --check, --test, --body-file <path>.\n"
            "To send a message that really does start with '--', use: "
            "notify_sms.py -- <message>\n" % args[0]
        )
        return 1

    return send(" ".join(args))


if __name__ == "__main__":
    raise SystemExit(main())
