"""
Lambda function: CoolCryptoUtilities Notification Service

Receives POST requests via API Gateway for various alert types:

1. **stable_bank** (default) — stable bank crossed a USD threshold
2. **gas_unsat** — wallet cannot cover gas on a chain; tells user exactly
   what to buy, how much, and which wallet to send it to

Environment variables (set during deployment):
  AWS_SES_REGION      – SES region (default: us-east-1)
  DEFAULT_SENDER      – Verified SES sender address (optional fallback)

Common request body fields:
  recipient_email     – Where to send the alert
  sender_email        – Verified SES sender (or uses DEFAULT_SENDER)
  alert_type          – "stable_bank" | "gas_unsat"  (default: stable_bank)

stable_bank fields:
  stable_bank_usd     – Current stable bank value in USD
  threshold_usd       – The threshold that was crossed

gas_unsat fields:
  wallet_address      – The wallet that needs funding
  chain               – Chain where gas is needed (e.g. "base")
  native_symbol       – Native token symbol (e.g. "ETH")
  deficit_native      – How much native token is needed
  deficit_usd         – USD equivalent of the deficit
  native_price_usd    – Current price of the native token
  total_available_usd – What's available across all chains
  recommendation      – Human-readable recommendation string
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


# ---------------------------------------------------------------------------
# Stable Bank alert
# ---------------------------------------------------------------------------

def _build_html_stable_bank(stable_bank_usd: float, threshold_usd: float) -> str:
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


def _build_text_stable_bank(stable_bank_usd: float, threshold_usd: float) -> str:
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


# ---------------------------------------------------------------------------
# Gas UNSAT alert
# ---------------------------------------------------------------------------

def _build_html_gas_unsat(body: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    wallet = body.get("wallet_address", "unknown")
    chain = body.get("chain", "unknown").capitalize()
    native_symbol = body.get("native_symbol", "ETH")
    deficit_native = float(body.get("deficit_native", 0))
    deficit_usd = float(body.get("deficit_usd", 0))
    native_price = float(body.get("native_price_usd", 0))
    total_available = float(body.get("total_available_usd", 0))
    recommendation = body.get("recommendation", "")

    # Truncate wallet for display but show full in monospace
    wallet_short = f"{wallet[:6]}...{wallet[-4:]}" if len(wallet) > 12 else wallet

    return f"""\
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
             background: #0a1628; color: #dce6f5; padding: 2rem;">
  <div style="max-width: 560px; margin: 0 auto; background: #111d30;
              border: 1px solid rgba(255,100,100,0.4); border-radius: 16px; padding: 2rem;">
    <h2 style="color: #ff6b6b; margin-top: 0;">&#9888; Gas Funding Required</h2>
    <p style="font-size: 1.05rem; line-height: 1.6;">
      Wallet <code style="background: #1a2942; padding: 2px 6px; border-radius: 4px;
      font-size: 0.9rem;">{wallet_short}</code> on <strong>{chain}</strong>
      cannot cover gas fees. Trading is paused until funded.
    </p>

    <div style="background: #1a2942; border-radius: 12px; padding: 1.2rem; margin: 1rem 0;">
      <table style="width: 100%; border-collapse: collapse; font-size: 0.95rem;">
        <tr>
          <td style="padding: 6px 0; color: rgba(255,255,255,0.6);">Buy:</td>
          <td style="padding: 6px 0; text-align: right; font-weight: 700; color: #ff9f43;">
            {deficit_native:.6f} {native_symbol}
          </td>
        </tr>
        <tr>
          <td style="padding: 6px 0; color: rgba(255,255,255,0.6);">USDC equivalent:</td>
          <td style="padding: 6px 0; text-align: right; font-weight: 700; color: #34d399;">
            {_format_usd(deficit_usd)}
          </td>
        </tr>
        <tr>
          <td style="padding: 6px 0; color: rgba(255,255,255,0.6);">{native_symbol} price:</td>
          <td style="padding: 6px 0; text-align: right;">
            {_format_usd(native_price)}
          </td>
        </tr>
        <tr>
          <td style="padding: 6px 0; color: rgba(255,255,255,0.6);">Chain:</td>
          <td style="padding: 6px 0; text-align: right;">{chain}</td>
        </tr>
        <tr>
          <td style="padding: 6px 0; color: rgba(255,255,255,0.6);">Available (all chains):</td>
          <td style="padding: 6px 0; text-align: right;">{_format_usd(total_available)}</td>
        </tr>
      </table>
    </div>

    <p style="font-size: 0.9rem; line-height: 1.5; color: rgba(255,255,255,0.8);">
      <strong>Send to:</strong>
    </p>
    <code style="display: block; background: #0d1520; padding: 10px 14px; border-radius: 8px;
                 font-size: 0.85rem; word-break: break-all; color: #6fa7ff; letter-spacing: 0.3px;">
      {wallet}
    </code>

    {f'<p style="margin-top: 1rem; font-size: 0.9rem; color: rgba(255,255,255,0.7);">{recommendation}</p>' if recommendation else ''}

    <hr style="border: none; border-top: 1px solid rgba(255,100,100,0.2); margin: 1.5rem 0;" />
    <p style="color: rgba(255,255,255,0.45); font-size: 0.75rem; margin-bottom: 0;">
      Sent by CoolCryptoUtilities &middot; {ts}
    </p>
  </div>
</body>
</html>"""


def _build_text_gas_unsat(body: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    wallet = body.get("wallet_address", "unknown")
    chain = body.get("chain", "unknown").capitalize()
    native_symbol = body.get("native_symbol", "ETH")
    deficit_native = float(body.get("deficit_native", 0))
    deficit_usd = float(body.get("deficit_usd", 0))
    native_price = float(body.get("native_price_usd", 0))
    total_available = float(body.get("total_available_usd", 0))
    recommendation = body.get("recommendation", "")

    return (
        f"GAS FUNDING REQUIRED\n\n"
        f"Wallet: {wallet}\n"
        f"Chain: {chain}\n\n"
        f"Buy: {deficit_native:.6f} {native_symbol} ({_format_usd(deficit_usd)} USDC equivalent)\n"
        f"{native_symbol} price: {_format_usd(native_price)}\n"
        f"Available across all chains: {_format_usd(total_available)}\n\n"
        f"Send {native_symbol} to:\n{wallet}\n\n"
        f"{recommendation}\n\n"
        f"— CoolCryptoUtilities · {ts}"
    )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

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
    alert_type = (body.get("alert_type") or "stable_bank").strip().lower()

    if not _validate_email(recipient):
        return _response(400, {"error": "Invalid or missing recipient_email"})
    if not _validate_email(sender):
        return _response(400, {"error": "Invalid or missing sender_email — set DEFAULT_SENDER env var or pass sender_email"})

    if alert_type == "gas_unsat":
        wallet = body.get("wallet_address", "")
        chain = body.get("chain", "unknown").capitalize()
        native_symbol = body.get("native_symbol", "ETH")
        deficit_usd = float(body.get("deficit_usd", 0))

        subject = f"[ACTION] Buy {_format_usd(deficit_usd)} of {native_symbol} for gas on {chain}"
        html_body = _build_html_gas_unsat(body)
        text_body = _build_text_gas_unsat(body)
    else:
        # Default: stable_bank
        stable_bank_usd = float(body.get("stable_bank_usd", 0))
        threshold_usd = float(body.get("threshold_usd", 0))

        if stable_bank_usd <= 0:
            return _response(400, {"error": "stable_bank_usd must be positive"})

        subject = f"Stable Bank Reached {_format_usd(stable_bank_usd)}"
        html_body = _build_html_stable_bank(stable_bank_usd, threshold_usd)
        text_body = _build_text_stable_bank(stable_bank_usd, threshold_usd)

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
