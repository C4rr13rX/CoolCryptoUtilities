#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# deploy.sh — Deploy the Stable Bank Notification stack on YOUR AWS account.
#
# Prerequisites:
#   1. AWS CLI v2 installed and configured (aws configure)
#   2. An SES-verified domain or email address in your region
#   3. Python 3 (for zipping the Lambda)
#
# Usage:
#   ./deploy.sh                              # uses default AWS profile
#   AWS_PROFILE=MyProfile ./deploy.sh        # uses a named profile
#   ./deploy.sh --region eu-west-1           # override region
#
# What it creates:
#   • IAM Role: CoolCrypto-StableBankNotify-Lambda
#   • Lambda:   CoolCrypto-StableBankNotify
#   • API GW:   CoolCrypto-Notifications  (/notify/stable-bank POST)
#   • API Key + Usage Plan (100 calls/day, 2/sec rate)
#
# After deployment, store these in your SecureVault settings:
#   STABLE_BANK_NOTIFY_ENDPOINT  → the API Gateway URL printed at the end
#   STABLE_BANK_NOTIFY_API_KEY   → the API key value printed at the end
#   STABLE_BANK_NOTIFY_EMAIL     → your email address
#   STABLE_BANK_NOTIFY_SENDER    → a verified SES sender address
#   STABLE_BANK_THRESHOLD_USD    → the USD threshold (e.g. 100)
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

REGION="${1:-${AWS_DEFAULT_REGION:-us-east-1}}"
if [[ "${1:-}" == "--region" ]]; then REGION="${2:-us-east-1}"; fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAMBDA_NAME="CoolCrypto-StableBankNotify"
ROLE_NAME="CoolCrypto-StableBankNotify-Lambda"
API_NAME="CoolCrypto-Notifications"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Stable Bank Notification — AWS Deployment                  ║"
echo "║  Region: $REGION                                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── 1. Get account info ──────────────────────────────────────────
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION")
echo "Account: $ACCOUNT_ID"

# ── 2. Create IAM Role ──────────────────────────────────────────
echo "[1/6] Creating IAM role..."
ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text 2>/dev/null || true)
if [[ -z "$ROLE_ARN" || "$ROLE_ARN" == "None" ]]; then
  ROLE_ARN=$(aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    --query 'Role.Arn' --output text)
  aws iam put-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-name ses-send-and-logs \
    --policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["ses:SendEmail","ses:SendRawEmail"],"Resource":"*"},{"Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],"Resource":"arn:aws:logs:*:*:*"}]}'
  echo "  Created role: $ROLE_ARN"
  echo "  Waiting 10s for IAM propagation..."
  sleep 10
else
  echo "  Role exists: $ROLE_ARN"
fi

# ── 3. Package & deploy Lambda ───────────────────────────────────
echo "[2/6] Packaging Lambda..."
ZIPFILE="$(mktemp -d)/stable_bank_notify.zip"
python3 -c "
import zipfile
with zipfile.ZipFile('$ZIPFILE', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('$SCRIPT_DIR/lambda_function.py', 'lambda_function.py')
print('  Packaged:', '$ZIPFILE')
"

echo "[3/6] Deploying Lambda..."
EXISTING=$(aws lambda get-function --function-name "$LAMBDA_NAME" --region "$REGION" 2>/dev/null || true)
if [[ -z "$EXISTING" ]]; then
  read -rp "Enter a verified SES sender email (e.g. noreply@yourdomain.com): " SENDER_EMAIL
  aws lambda create-function \
    --function-name "$LAMBDA_NAME" \
    --runtime python3.12 \
    --role "$ROLE_ARN" \
    --handler lambda_function.lambda_handler \
    --zip-file "fileb://$ZIPFILE" \
    --timeout 15 --memory-size 128 \
    --environment "Variables={AWS_SES_REGION=$REGION,DEFAULT_SENDER=$SENDER_EMAIL}" \
    --description "Sends email when stable bank crosses threshold" \
    --region "$REGION" > /dev/null
  echo "  Created Lambda: $LAMBDA_NAME"
else
  aws lambda update-function-code \
    --function-name "$LAMBDA_NAME" \
    --zip-file "fileb://$ZIPFILE" \
    --region "$REGION" > /dev/null
  echo "  Updated Lambda: $LAMBDA_NAME"
fi
LAMBDA_ARN="arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$LAMBDA_NAME"

# ── 4. Create API Gateway ───────────────────────────────────────
echo "[4/6] Setting up API Gateway..."
API_ID=$(aws apigateway get-rest-apis --region "$REGION" \
  --query "items[?name=='$API_NAME'].id | [0]" --output text 2>/dev/null || true)

if [[ -z "$API_ID" || "$API_ID" == "None" ]]; then
  API_ID=$(aws apigateway create-rest-api \
    --name "$API_NAME" \
    --description "Notification endpoints for CoolCryptoUtilities" \
    --endpoint-configuration types=REGIONAL \
    --region "$REGION" --query 'id' --output text)
  echo "  Created API: $API_ID"
fi

ROOT_ID=$(aws apigateway get-resources --rest-api-id "$API_ID" --region "$REGION" \
  --query 'items[0].id' --output text)

# Create /notify
NOTIFY_ID=$(aws apigateway get-resources --rest-api-id "$API_ID" --region "$REGION" \
  --query "items[?pathPart=='notify'].id | [0]" --output text 2>/dev/null || true)
if [[ -z "$NOTIFY_ID" || "$NOTIFY_ID" == "None" ]]; then
  NOTIFY_ID=$(aws apigateway create-resource --rest-api-id "$API_ID" \
    --parent-id "$ROOT_ID" --path-part "notify" --region "$REGION" \
    --query 'id' --output text)
fi

# Create /notify/stable-bank
SB_ID=$(aws apigateway get-resources --rest-api-id "$API_ID" --region "$REGION" \
  --query "items[?pathPart=='stable-bank'].id | [0]" --output text 2>/dev/null || true)
if [[ -z "$SB_ID" || "$SB_ID" == "None" ]]; then
  SB_ID=$(aws apigateway create-resource --rest-api-id "$API_ID" \
    --parent-id "$NOTIFY_ID" --path-part "stable-bank" --region "$REGION" \
    --query 'id' --output text)
fi

# POST method + Lambda integration
aws apigateway put-method --rest-api-id "$API_ID" --resource-id "$SB_ID" \
  --http-method POST --authorization-type NONE --api-key-required \
  --region "$REGION" > /dev/null 2>&1 || true

aws apigateway put-integration --rest-api-id "$API_ID" --resource-id "$SB_ID" \
  --http-method POST --type AWS_PROXY --integration-http-method POST \
  --uri "arn:aws:apigateway:$REGION:lambda:path/2015-03-31/functions/$LAMBDA_ARN/invocations" \
  --region "$REGION" > /dev/null

# Lambda permission
aws lambda add-permission --function-name "$LAMBDA_NAME" \
  --statement-id apigateway-invoke --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:$REGION:$ACCOUNT_ID:$API_ID/*/POST/notify/stable-bank" \
  --region "$REGION" > /dev/null 2>&1 || true

# Deploy to prod stage
aws apigateway create-deployment --rest-api-id "$API_ID" \
  --stage-name prod --region "$REGION" > /dev/null
echo "  Deployed API to prod stage"

# ── 5. Create API Key + Usage Plan ──────────────────────────────
echo "[5/6] Creating API key..."
KEY_RESULT=$(aws apigateway create-api-key \
  --name "CoolCrypto-Notify-Key-$(date +%s)" \
  --enabled --region "$REGION" --query '{id:id,value:value}' --output json)
KEY_ID=$(echo "$KEY_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
KEY_VALUE=$(echo "$KEY_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['value'])")

PLAN_ID=$(aws apigateway create-usage-plan \
  --name "CoolCrypto-Notify-Plan" \
  --throttle burstLimit=5,rateLimit=2 \
  --quota limit=100,period=DAY \
  --api-stages "apiId=$API_ID,stage=prod" \
  --region "$REGION" --query 'id' --output text 2>/dev/null || true)

if [[ -n "$PLAN_ID" && "$PLAN_ID" != "None" ]]; then
  aws apigateway create-usage-plan-key \
    --usage-plan-id "$PLAN_ID" --key-id "$KEY_ID" --key-type API_KEY \
    --region "$REGION" > /dev/null
fi

# ── 6. Print summary ────────────────────────────────────────────
ENDPOINT="https://$API_ID.execute-api.$REGION.amazonaws.com/prod/notify/stable-bank"

echo ""
echo "[6/6] Done!  Add these to your SecureVault settings:"
echo ""
echo "  STABLE_BANK_NOTIFY_ENDPOINT = $ENDPOINT"
echo "  STABLE_BANK_NOTIFY_API_KEY  = $KEY_VALUE"
echo "  STABLE_BANK_NOTIFY_EMAIL    = <your-email>"
echo "  STABLE_BANK_NOTIFY_SENDER   = <verified-ses-sender>"
echo "  STABLE_BANK_THRESHOLD_USD   = 100"
echo ""
echo "Test with:"
echo "  curl -X POST '$ENDPOINT' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -H 'x-api-key: $KEY_VALUE' \\"
echo "    -d '{\"recipient_email\":\"you@example.com\",\"sender_email\":\"noreply@yourdomain.com\",\"stable_bank_usd\":150.00,\"threshold_usd\":100.00}'"
