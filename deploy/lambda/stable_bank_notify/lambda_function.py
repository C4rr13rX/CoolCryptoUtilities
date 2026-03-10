"""
Lambda function: Stable Bank Threshold Notification

Receives a POST request via API Gateway when the trading pipeline's
stable bank crosses a user-defined USD threshold.  Sends a formatted
email via Amazon SES.

Environment variables (set during deployment):
  AWS_SES_REGION      – SES region (default: us-east-1)
  DEFAULT_SENDER      – Verified SES sender address (optional fallback)

The request body must include:
  recipient_email     – Where to send the alert
  sender_email        – Verified SES sender (or uses DEFAULT_SENDER)
  stable_bank_usd     – Current stable bank value in USD
  threshold_usd       – The threshold that was crossed
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

_SES_REGION = os.environ.get("AWS_SES_REGION", "us-east-1")
_DEFAULT_SENDER = os.environ.get("DEFAULT_SENDER", "")

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def _validate_email(addr: str) -> bool:
    return bool(addr and _EMAIL_RE.match(addr))


def _format_usd(value: float) -> str:
    return f"${value:,.2f}"


def _build_html(stable_bank_usd: float, threshold_usd: float) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""\
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
             background: #0a1628; color: #dce6f5; padding: 2rem;">
  <div style="max-width: 520px; margin: 0 auto; background: #111d30;
              border: 1px solid rgba(111,167,255,0.25); border-radius: 16px; padding: 2rem;">
    <h2 style="color: #6fa7ff; margin-top: 0;">Stable Bank Threshold Reached</h2>
    <p style="font-size: 1.1rem; line-height: 1.6;">
      Your trading pipeline's <strong>stable bank</strong> has reached
      <span style="color: #34d399; font-size: 1.3rem; font-weight: 700;">
        {_format_usd(stable_bank_usd)}
      </span>
      &mdash; crossing your configured threshold of
      <strong>{_format_usd(threshold_usd)}</strong>.
    </p>
    <p style="color: rgba(255,255,255,0.6); font-size: 0.85rem;">
      This value represents accumulated stablecoin profits (converted to USD)
      that the pipeline has checkpointed during ghost and live trading.
    </p>
    <hr style="border: none; border-top: 1px solid rgba(111,167,255,0.2); margin: 1.5rem 0;" />
    <p style="color: rgba(255,255,255,0.45); font-size: 0.75rem; margin-bottom: 0;">
      Sent by CoolCryptoUtilities &middot; {ts}
    </p>
  </div>
</body>
</html>"""


def _build_text(stable_bank_usd: float, threshold_usd: float) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"Stable Bank Threshold Reached\n\n"
        f"Your trading pipeline's stable bank has reached "
        f"{_format_usd(stable_bank_usd)}, crossing your configured "
        f"threshold of {_format_usd(threshold_usd)}.\n\n"
        f"This value represents accumulated stablecoin profits (converted "
        f"to USD) that the pipeline has checkpointed during ghost and "
        f"live trading.\n\n"
        f"— CoolCryptoUtilities · {ts}"
    )


def lambda_handler(event, context):
    # Support both direct invocation and API Gateway proxy.
    if isinstance(event.get("body"), str):
        try:
            body = json.loads(event["body"])
        except (json.JSONDecodeError, TypeError):
            return _response(400, {"error": "Invalid JSON body"})
    else:
        body = event

    recipient = (body.get("recipient_email") or "").strip()
    sender = (body.get("sender_email") or _DEFAULT_SENDER or "").strip()
    stable_bank_usd = float(body.get("stable_bank_usd", 0))
    threshold_usd = float(body.get("threshold_usd", 0))

    if not _validate_email(recipient):
        return _response(400, {"error": "Invalid or missing recipient_email"})
    if not _validate_email(sender):
        return _response(400, {"error": "Invalid or missing sender_email — set DEFAULT_SENDER env var or pass sender_email"})
    if stable_bank_usd <= 0:
        return _response(400, {"error": "stable_bank_usd must be positive"})

    subject = f"Stable Bank Reached {_format_usd(stable_bank_usd)}"
    html_body = _build_html(stable_bank_usd, threshold_usd)
    text_body = _build_text(stable_bank_usd, threshold_usd)

    ses = boto3.client("ses", region_name=_SES_REGION)
    try:
        result = ses.send_email(
            Source=sender,
            Destination={"ToAddresses": [recipient]},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {
                    "Text": {"Data": text_body, "Charset": "UTF-8"},
                    "Html": {"Data": html_body, "Charset": "UTF-8"},
                },
            },
        )
        message_id = result.get("MessageId", "unknown")
        return _response(200, {"status": "sent", "message_id": message_id})
    except ClientError as exc:
        return _response(502, {"error": f"SES send failed: {exc.response['Error']['Message']}"})


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }
