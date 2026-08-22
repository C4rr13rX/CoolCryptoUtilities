#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Deploy the whole stack into LocalStack.
#
# Creates the same resources the real deployment would, using the same AWS API
# calls -- LocalStack implements the AWS API, so these commands are the ones
# you would run against a real account with AWS_ENDPOINT_URL removed.
#
#   * IAM role for the functions
#   * Lambda: http, cron, websocket, ws-broadcast, admin
#   * API Gateway REST API     -> http function ({proxy+} catch-all)
#   * API Gateway WebSocket    -> websocket function (Pro only; skipped locally)
#   * EventBridge Scheduler    -> cron + broadcast functions
#   * S3 bucket for media
#
# The dummy credentials below are LocalStack's fixed test values. The real
# AWS profile (FountainServer) is never referenced.
# ---------------------------------------------------------------------------
set -euo pipefail

ENDPOINT="${AWS_ENDPOINT_URL:-http://localhost:4566}"
REGION="${AWS_REGION:-us-east-1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ZIP="$ROOT/dist/coolcrypto-lambda.zip"
# The AWS CLI here is the native Windows build, which cannot resolve Git Bash's
# /d/... paths. cygpath hands it a real Windows path; on Linux/macOS the
# command is absent and the POSIX path is already correct.
if command -v cygpath >/dev/null 2>&1; then
  ZIP="$(cygpath -w "$ZIP")"
fi

export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION="$REGION"
# Never let an ambient profile (e.g. FountainServer) redirect these calls.
unset AWS_PROFILE || true

aws() { command aws --endpoint-url "$ENDPOINT" --region "$REGION" "$@"; }

ROLE_ARN="arn:aws:iam::000000000000:role/coolcrypto-lambda"
MEDIA_BUCKET="coolcrypto-media"
CODE_BUCKET="coolcrypto-lambda-code"
HYBRID_BUCKET="coolcrypto-hybrid"
# Local-only. On real AWS this must come from Secrets Manager -- a shared
# static secret would let anyone who reads the repo mint session tokens.
PQ_SESSION_SECRET="${PQ_SESSION_SECRET:-local-dev-session-secret-32-chars-min!}"
RUNTIME="python3.12"

# Lambda containers are siblings of the compose network, so MinIO and
# LocalStack are reached via the host gateway rather than compose DNS names.
DB_HOST="host.docker.internal"

# No Postgres. Under the hybrid model the durable store is S3 and Django's
# own database is a scratch SQLite file in /tmp used only by the parts of
# contrib that insist on one (admin, auth tables for management commands).
COMMON_ENV="DJANGO_SETTINGS_MODULE=coolcrypto_dashboard.settings_lambda,\
DJANGO_DB_VENDOR=sqlite,\
DJANGO_SECRET_KEY=local-dev-not-a-real-secret,\
DJANGO_DEBUG=0,\
DJANGO_SECURE_SSL_REDIRECT=0,\
DJANGO_ALLOWED_HOSTS=*,\
MEDIA_BUCKET=$MEDIA_BUCKET,\
AWS_S3_ENDPOINT_URL=http://$DB_HOST:9000,\
AWS_ACCESS_KEY_ID=minioadmin,\
AWS_SECRET_ACCESS_KEY=minioadmin,\
AWS_SESSION_TOKEN=,\
SECURE_ENV_HYDRATED=1,\
ALLOW_SQLITE_FALLBACK=0,\
API_GATEWAY_STRIP_STAGE=1,\
HYBRID_DB=1,\
HYBRID_BUCKET=$HYBRID_BUCKET,\
DJANGO_SQLITE_PATH=/tmp/django-scratch.db,\
PQ_SESSION_SECRET=$PQ_SESSION_SECRET,\
PQ_REGISTRATION_OPEN=1,\
PQ_ALLOWED_ORIGIN=http://localhost:5173"

echo "=== 0/6  preflight ==="
[[ -f "$ZIP" ]] || { echo "missing $ZIP -- run build_package.py first"; exit 1; }
aws sts get-caller-identity >/dev/null || { echo "LocalStack unreachable"; exit 1; }
echo "  LocalStack reachable at $ENDPOINT"

echo "=== 1/6  IAM role ==="
aws iam create-role --role-name coolcrypto-lambda \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":["lambda.amazonaws.com","scheduler.amazonaws.com"]},"Action":"sts:AssumeRole"}]}' \
  >/dev/null 2>&1 || echo "  role exists"

echo "=== 2/6  S3 buckets + code upload ==="
aws s3api create-bucket --bucket "$MEDIA_BUCKET" >/dev/null 2>&1 || echo "  media bucket exists"
aws s3api create-bucket --bucket "$CODE_BUCKET" >/dev/null 2>&1 || echo "  code bucket exists"
aws s3api create-bucket --bucket "$HYBRID_BUCKET" >/dev/null 2>&1 || echo "  hybrid bucket exists"
# Lambda rejects a direct --zip-file upload over 50 MB (the bundle carries
# collected static files and cryptography's native wheel). Publishing from S3
# is the supported path and raises the ceiling to 250 MB unzipped.
CODE_KEY="coolcrypto-lambda.zip"
aws s3api put-object --bucket "$CODE_BUCKET" --key "$CODE_KEY" --body "$ZIP" >/dev/null
echo "  uploaded bundle to s3://$CODE_BUCKET/$CODE_KEY"

echo "=== 3/6  Lambda functions ==="
deploy_fn() {
  local name="$1" handler="$2" timeout="$3" memory="$4"
  if aws lambda get-function --function-name "$name" >/dev/null 2>&1; then
    aws lambda update-function-code --function-name "$name" \
      --s3-bucket "$CODE_BUCKET" --s3-key "$CODE_KEY" >/dev/null
    # A code update leaves the function in LastUpdateStatus=InProgress, and a
    # configuration update during that window is rejected with
    # ResourceConflictException. Wait for the code update to settle first.
    aws lambda wait function-updated-v2 --function-name "$name" 2>/dev/null || true
    aws lambda update-function-configuration --function-name "$name" \
      --handler "$handler" --timeout "$timeout" --memory-size "$memory" \
      --environment "Variables={$COMMON_ENV}" >/dev/null
    echo "  updated $name"
  else
    aws lambda create-function --function-name "$name" \
      --runtime "$RUNTIME" --role "$ROLE_ARN" --handler "$handler" \
      --code "S3Bucket=$CODE_BUCKET,S3Key=$CODE_KEY" --timeout "$timeout" --memory-size "$memory" \
      --environment "Variables={$COMMON_ENV}" >/dev/null
    echo "  created $name"
  fi
  aws lambda wait function-active-v2 --function-name "$name" 2>/dev/null || true
}

# 30s matches API Gateway's hard integration timeout -- a longer Lambda would
# keep burning time on a request the client already gave up on.
deploy_fn coolcrypto-http      serverless.handlers.http.lambda_handler          30  1024
# Cron tasks do real pipeline work; 15min is the Lambda maximum.
deploy_fn coolcrypto-cron      serverless.handlers.cron.lambda_handler          900 2048
deploy_fn coolcrypto-ws        serverless.handlers.websocket.lambda_handler     30  512
deploy_fn coolcrypto-ws-push   serverless.handlers.websocket.broadcast_handler  60  512
deploy_fn coolcrypto-admin     serverless.handlers.admin_tasks.lambda_handler   900 1024
# Auth and the hybrid data API import no Django, so they cold-start in ~0.3s
# instead of ~4s and stay small enough to sit in the free tier. 512 MB is
# chosen for Argon2id: the hash is configured for 64 MiB, and the rest is
# headroom for the ML-KEM operations.
deploy_fn coolcrypto-auth      serverless.handlers.auth.lambda_handler          15  512
deploy_fn coolcrypto-hybrid    serverless.handlers.hybrid_api.lambda_handler    15  512
# Serves the shared Parquet partitions. Presigns rather than proxies, so
# it stays small regardless of how large a partition gets.
deploy_fn coolcrypto-market    serverless.handlers.market_api.lambda_handler    15  512

echo "=== 4/6  HTTP API (REST v1) ==="
# LocalStack Community implements API Gateway v1 (REST) but not v2 (HTTP API).
# A REST API with a {proxy+} greedy resource and AWS_PROXY integration produces
# the same Lambda proxy contract, so the handler code is unchanged between
# local and real deployments. On real AWS you would create an HTTP API here --
# Mangum accepts both payload formats.
API_ID=$(aws apigateway get-rest-apis --query "items[?name=='coolcrypto-http'].id | [0]" --output text)
if [[ -z "$API_ID" || "$API_ID" == "None" ]]; then
  API_ID=$(aws apigateway create-rest-api --name coolcrypto-http     --endpoint-configuration types=REGIONAL --query id --output text)
  ROOT_ID=$(aws apigateway get-resources --rest-api-id "$API_ID" --query 'items[0].id' --output text)
  PROXY_ID=$(aws apigateway create-resource --rest-api-id "$API_ID"     --parent-id "$ROOT_ID" --path-part '{proxy+}' --query id --output text)
  URI="arn:aws:apigateway:$REGION:lambda:path/2015-03-31/functions/arn:aws:lambda:$REGION:000000000000:function:coolcrypto-http/invocations"
  # Both the root and the greedy child need a method: {proxy+} does not match
  # the empty path, so without the root ANY the site would 403 on "/".
  for pair in "$ROOT_ID" "$PROXY_ID"; do
    aws apigateway put-method --rest-api-id "$API_ID" --resource-id "$pair"       --http-method ANY --authorization-type NONE >/dev/null
    aws apigateway put-integration --rest-api-id "$API_ID" --resource-id "$pair"       --http-method ANY --type AWS_PROXY --integration-http-method POST       --uri "$URI" >/dev/null
  done
  aws apigateway create-deployment --rest-api-id "$API_ID" --stage-name prod >/dev/null
  echo "  created HTTP API $API_ID"
else
  echo "  HTTP API exists: $API_ID"
fi
aws lambda add-permission --function-name coolcrypto-http   --statement-id apigw-invoke --action lambda:InvokeFunction   --principal apigateway.amazonaws.com >/dev/null 2>&1 || true

# --- /auth/* and /hybrid/* -> their own functions ------------------------
# These are mounted as sibling {proxy+} resources rather than being folded
# into the Django catch-all, so a login never pays Django's ~4s cold start
# and cannot be reached through the dashboard's middleware stack.
mount_subtree() {
  local segment="$1" fn="$2"
  local parent child uri
  parent=$(aws apigateway get-resources --rest-api-id "$API_ID" \
    --query "items[?pathPart=='$segment'].id | [0]" --output text 2>/dev/null)
  if [[ -z "$parent" || "$parent" == "None" ]]; then
    parent=$(aws apigateway create-resource --rest-api-id "$API_ID" \
      --parent-id "$ROOT_RESOURCE_ID" --path-part "$segment" \
      --query id --output text)
  fi
  # A greedy child under the segment: {proxy+} does not match the segment
  # itself, so /auth/login needs the child and /auth needs the parent.
  child=$(aws apigateway get-resources --rest-api-id "$API_ID" \
    --query "items[?parentId=='$parent'&&pathPart=='{proxy+}'].id | [0]" \
    --output text 2>/dev/null)
  if [[ -z "$child" || "$child" == "None" ]]; then
    child=$(aws apigateway create-resource --rest-api-id "$API_ID" \
      --parent-id "$parent" --path-part '{proxy+}' --query id --output text)
  fi
  uri="arn:aws:apigateway:$REGION:lambda:path/2015-03-31/functions/arn:aws:lambda:$REGION:000000000000:function:$fn/invocations"
  for rid in "$parent" "$child"; do
    aws apigateway put-method --rest-api-id "$API_ID" --resource-id "$rid" \
      --http-method ANY --authorization-type NONE >/dev/null 2>&1 || true
    aws apigateway put-integration --rest-api-id "$API_ID" --resource-id "$rid" \
      --http-method ANY --type AWS_PROXY --integration-http-method POST \
      --uri "$uri" >/dev/null 2>&1 || true
  done
  aws lambda add-permission --function-name "$fn" \
    --statement-id "apigw-$segment" --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com >/dev/null 2>&1 || true
  echo "  mounted /$segment -> $fn"
}

ROOT_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id "$API_ID" \
  --query "items[?path=='/'].id | [0]" --output text)
mount_subtree auth   coolcrypto-auth
mount_subtree hybrid coolcrypto-hybrid
mount_subtree market coolcrypto-market

# Routes only take effect once the stage is redeployed.
aws apigateway create-deployment --rest-api-id "$API_ID" --stage-name prod >/dev/null
echo "  redeployed prod stage"

echo "=== 5/6  WebSocket API ==="
# apigatewayv2 (the only way to get a WebSocket API) is a LocalStack Pro
# feature. The WebSocket *handlers* are still tested end-to-end by invoking
# them with the exact $connect/$disconnect/$default event shapes API Gateway
# emits -- see test_local_stack.py section 6. Set WS_ID to deploy for real.
WS_ID=""
if aws apigatewayv2 get-apis >/dev/null 2>&1; then
  WS_ID=$(aws apigatewayv2 get-apis --query "Items[?Name=='coolcrypto-ws'].ApiId | [0]" --output text)
  if [[ -z "$WS_ID" || "$WS_ID" == "None" ]]; then
    WS_ID=$(aws apigatewayv2 create-api --name coolcrypto-ws --protocol-type WEBSOCKET       --route-selection-expression '$request.body.action' --query ApiId --output text)
    WS_INT=$(aws apigatewayv2 create-integration --api-id "$WS_ID"       --integration-type AWS_PROXY --integration-method POST       --integration-uri "arn:aws:lambda:$REGION:000000000000:function:coolcrypto-ws"       --query IntegrationId --output text)
    for route in '$connect' '$disconnect' '$default'; do
      aws apigatewayv2 create-route --api-id "$WS_ID" --route-key "$route"         --target "integrations/$WS_INT" >/dev/null
    done
    aws apigatewayv2 create-stage --api-id "$WS_ID" --stage-name prod --auto-deploy >/dev/null
    echo "  created WebSocket API $WS_ID"
  else
    echo "  WebSocket API exists: $WS_ID"
  fi
  aws lambda update-function-configuration --function-name coolcrypto-ws-push     --environment "Variables={$COMMON_ENV,WEBSOCKET_MANAGEMENT_ENDPOINT=http://$DB_HOST:4566/_aws/execute-api/$WS_ID/prod}" >/dev/null
  aws lambda wait function-updated-v2 --function-name coolcrypto-ws-push 2>/dev/null || true
else
  echo "  SKIPPED: apigatewayv2 unavailable (LocalStack Pro feature)."
  echo "           WebSocket handlers are still covered by direct invocation."
fi

echo "=== 6/6  EventBridge schedules ==="
mk_schedule() {
  local name="$1" expr="$2" fn="$3" payload="$4"
  aws scheduler delete-schedule --name "$name" >/dev/null 2>&1 || true
  aws scheduler create-schedule --name "$name" \
    --schedule-expression "$expr" \
    --flexible-time-window '{"Mode":"OFF"}' \
    --target "{\"Arn\":\"arn:aws:lambda:$REGION:000000000000:function:$fn\",\"RoleArn\":\"$ROLE_ARN\",\"Input\":\"$payload\"}" \
    >/dev/null
  echo "  scheduled $name ($expr)"
}

# Intervals mirror services/cron_profile.py so behaviour matches the threaded
# supervisor it replaces: auto_pipeline every 3h, weekly_bootstrap every 7d.
mk_schedule coolcrypto-auto-pipeline "rate(180 minutes)" coolcrypto-cron '{\"task_id\":\"auto_pipeline\"}'
mk_schedule coolcrypto-weekly-bootstrap "rate(10080 minutes)" coolcrypto-cron '{\"task_id\":\"weekly_bootstrap\"}'
# Replaces the consumers' 2s asyncio push loop. 1 minute is EventBridge's
# finest granularity -- a UI needing sub-minute updates should poll the HTTP
# API instead of relying on this fan-out.
mk_schedule coolcrypto-ws-broadcast "rate(1 minute)" coolcrypto-ws-push '{}'

cat <<EOF

──────────────────────────────────────────────────────────────
  Deployed to LocalStack

  HTTP API   http://localhost:4566/_aws/execute-api/$API_ID/prod/
  WebSocket  ${WS_ID:+ws://localhost:4566/_aws/execute-api/$WS_ID/prod}${WS_ID:-(not deployed - LocalStack Pro)}
  MinIO      http://localhost:9001  (minioadmin/minioadmin)
  Postgres   localhost:55432

  Run migrations:
    aws --endpoint-url $ENDPOINT lambda invoke \\
      --function-name coolcrypto-admin \\
      --payload '{"command":"migrate","args":["--noinput"]}' \\
      --cli-binary-format raw-in-base64-out /dev/stdout
──────────────────────────────────────────────────────────────
EOF

echo "$API_ID" > "$SCRIPT_DIR/.http_api_id"
echo "$WS_ID" > "$SCRIPT_DIR/.ws_api_id"
