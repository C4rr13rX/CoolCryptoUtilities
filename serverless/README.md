# Serverless deployment

The Django dashboard, packaged for AWS Lambda + API Gateway + EventBridge, and
tested locally against open-source equivalents. **No real AWS account is
touched** — the local stack replaces every managed service, and the deploy
script explicitly unsets `AWS_PROFILE` so an ambient `FountainServer` profile
cannot redirect a call to a live account.

## There is no "Lambda version of Django"

Worth stating plainly, because it shapes everything below: Lambda runs ordinary
Django. This project already had Django 6.0.3 and it needed **no downgrade** —
Lambda supplies a Python runtime, and [Mangum](https://mangum.io) adapts the
ASGI app to Lambda's `(event, context)` calling convention.

What actually had to change was **architecture**, not version. The site was
built around a long-lived Waitress process, and Lambda gives you the opposite:
a frozen sandbox, a read-only filesystem, and no threads between invocations.

## Service mapping

| AWS | Local equivalent | Notes |
|---|---|---|
| Lambda | LocalStack | Runs the real `public.ecr.aws/lambda/python:3.12` image |
| API Gateway (HTTP) | LocalStack | REST API + `{proxy+}`; v2 HTTP API is Pro-only |
| API Gateway (WebSocket) | *not emulated* | Pro-only; handlers tested by direct invocation |
| EventBridge Scheduler | LocalStack | Same `rate()` / `cron()` expressions |
| S3 (data + media) | MinIO | Holds the database; there is no RDS |
| CloudWatch Logs | LocalStack | |

## The four conversion problems

### 1. Background threads (the blocker)

`core/apps.py` spawned four daemon threads at Django boot — guardian, market
brain, production manager, cron. On Lambda the sandbox freezes when the handler
returns, so a sleeping thread is suspended mid-work and resumes at an arbitrary
later time, or never.

`settings_lambda.py` sets the same env switches `manage.py` already understood
(`GUARDIAN_AUTO_DISABLED`, `CRON_AUTO_DISABLED`, …) *before* importing the base
settings, so the threads never start. The cron workload moves to
`handlers/cron.py`, invoked by EventBridge.

### 2. Read-only filesystem

Only `/tmp` is writable. Several modules ran `mkdir(parents=True)` **at import
time** on a path derived from `Path(__file__).parents[N]` —
`services/wallet_state.py` is the one the telemetry URLconf reaches first, and
it took down all 40 URL patterns with a `PermissionError` before any view ran.

Fixed with `services/writable_paths.ensure_dir`, which keeps the original path
when writable and falls back to a `/tmp` equivalent when not. Five call sites
use it. `serverless/bootstrap.py` sets `WRITABLE_ROOT=/tmp` so that fallback is
deterministic.

A symlink (`/var/task/storage` → `/tmp/storage`) would have avoided touching
source, but creating one means writing inside `/var/task` — the exact
permission Lambda withholds.

### 3. WebSockets

The three `opsconsole` consumers accept a socket then loop forever, sleeping
and pushing snapshots. That needs a process that outlives the request.

Inverted: API Gateway holds the connection, the connection IDs are persisted
**in the S3 hybrid store** (`ws_connections`), and a scheduled
`broadcast_handler` does what `_stream()` used to — building the same payloads
and posting them to every live connection. The wire format is unchanged, so the
frontend needs no edits.

The IDs cannot live in Django's database: under the hybrid model that is a
scratch SQLite file in `/tmp`, private to one sandbox and empty on every cold
start, so the broadcaster would never see the connections `$connect`
registered. `serverless/models.py` keeps the old model only so the existing
migration still applies on the Postgres fallback path.

Because EventBridge's finest granularity is 1 minute, the console's old 2-second
cadence is gone. A UI needing sub-minute updates should poll the HTTP API.

### 4. Bundle size

The full dependency set came to 546 MB unzipped against Lambda's 250 MB limit.
`tools/c0d3rV2/native_os_service` alone was 229 MB of build artifacts that
nothing in the request path imports. Excluding it plus pruning vendored test
suites and headers (91.7 MB) brings it to **241 MB — under the limit, but not
by much.**

pandas and numpy account for ~142 MB and *are* required: the telemetry and
wallet views import them while the URLconf is built. Adding another large
dependency will exceed the ceiling; move heavy work to a container-image
Lambda or a layer at that point.

## The hybrid database (no RDS)

Ported from C4rr13rX. **S3 is the database; the browser is the query engine.**

```
Browser: AllezORM over IndexedDB     <- real SQLite, offline, survives reload
   |   REST over API Gateway
   v
Lambda: stateless S3 facade
   |
   v
S3: database/tables/<table>/<id>.json
```

There is no RDS and no DynamoDB. RDS bills per hour whether traffic arrives or
not (~$15-30/month floor) for a dataset of ~1,400 rows; S3 bills per request
and scales to zero, with no VPC, NAT gateway, or RDS Proxy to pay for. Every
read the browser answers from IndexedDB is an API Gateway request, a Lambda
invocation, and an S3 GET that never happen -- the local tier is the cost
control, not just a latency trick.

**Nothing is cached server-side.** An earlier draft mirrored the data into
SQLite in `/tmp`; that was wrong. A Lambda sandbox is disposable, so such a
cache dies on every cold start, forks into N inconsistent copies under
concurrency, and inverts the point of the design. The only server state is a
per-invocation memo, cleared at the top of every handler -- without that reset
a warm sandbox could serve one user a row it read for another.

Layout, matching C4rr13rX's `database/tables` prefix:

| Key | Purpose |
|---|---|
| `<table>/<id>.json` | one row |
| `<table>/total.txt` | id allocator (compare-and-set) |
| `<table>/_keys.json` | explicit key list for UUID-keyed tables |
| `<table>/_index/<field>.json` | secondary lookup (e.g. email -> id) |
| `<table>/change.txt` | monotonic sequence; the browser polls this |

`_keys.json` exists because several branddozer models use 32-char UUID primary
keys, which a `total.txt` range cannot address. Without it those 260+ rows are
written but unreadable.

`_index/<field>.json` matters for cost as much as speed: login resolves an
email on every attempt, and a table scan would be O(n) S3 GETs per try.

**Consistency:** S3 is strongly consistent per object, but there are no
cross-object transactions. That is fine for these tables (configuration and
admin data) and is the same trade C4rr13rX makes. A table needing multi-row
atomicity does not belong here.

### Migration

`serverless/hybrid/migrate_to_s3.py` moved the real data: **815 rows** across
14 tables, plus the `admin` account. It deliberately skips the trading
telemetry (`metrics`, `feedback_events`, `market_stream` -- 3.4M rows, 27 GB):
that is append-only instrumentation read through the trading pipeline, not the
Django ORM, and per-row S3 objects would cost a fortune in PUTs for no benefit.

## Quantum-safe login (replaces the magic link)

A magic link puts a live credential in an inbox and makes login depend on SES
deliverability and cost. This replaces it with a password that never leaves the
client in usable form.

The threat is **harvest-now-decrypt-later**: TLS 1.3 key exchange is
X25519/ECDHE, which Shor's algorithm breaks. Traffic recorded today can be
decrypted later, and a password recovered that way is probably still valid. So
the password is never protected by TLS alone:

1. **ML-KEM-768** (FIPS 203) -- server publishes a per-attempt encapsulation
   key; the client encapsulates. A fresh keypair per attempt means a
   compromised decapsulation key exposes only a 2-minute window.
2. **HKDF-SHA-384** -- derives the AES key, bound to a transcript
   (`challenge_id|server_key`) so a captured envelope cannot be replayed
   against a different challenge.
3. **AES-256-GCM** -- seals the password. Grover only weakens symmetric
   primitives quadratically, so AES-256 keeps ~128-bit post-quantum strength.
4. **Argon2id** (64 MiB, t=3, p=4) -- memory-hard verifier; ~183 ms on the
   Lambda runtime.
5. **HMAC-SHA-384 session tokens**, not JWT. JWT's algorithm agility is a
   footgun (`alg: none`, RS256->HS256 confusion) and we need none of it for a
   single-issuer token. Revocation is real: the token's `jti` is checked
   against a stored session row, so logout actually invalidates.

Verified cross-language: the browser's `@noble/post-quantum` seals an envelope
that Python's `kyber-py` opens correctly.

**This does not claim** to protect metadata (timing, IP, sizes) -- those keep
classical protection only. Nor does it help a compromised client: malware sees
the password before it is encrypted.

### Credentials

The migrated account is **`admin` / `admin`**, as requested, seeded directly by
the migration (Django's PBKDF2 hash cannot be converted to Argon2id, so it is
re-created rather than moved). That password bypasses the 12-character policy
that applies to every other account, and the record carries
`must_change_password: true`.

**Change it before this is reachable from anything but localhost.**

## Shared market data

Market data is **shared across accounts** — the price of WETH at a given second
is not per-user — so one copy serves everyone, under its own prefix with no
owner column and no per-account filtering.

It could not use the per-row `<table>/<id>.json` layout. That shape is right
for ~1,400 rows of config; for 3.4M rows of time series it means 3.4M PUTs to
write and 3.4M GETs on every sync. So market data is **Parquet, partitioned by
month**:

```
database/market/<table>/<YYYY-MM>.parquet   ← columnar, zstd
database/market/<table>/_manifest.json      ← partitions, counts, time ranges
```

Measured result: **3,386,863 rows → 32 objects, 28.7 MB** (~16x smaller than
the source SQLite pages). Parquet is columnar, so a price chart transfers `ts`
and `price` without ever pulling the `raw` JSON column, and `hyparquet` reads
it directly in the browser where AllezORM does the searching.

Closed months are immutable, which is what makes caching safe — only the
current month is ever refetched.

### organism_snapshots is the exception

54,447 rows averaging ~430 KB of JSON: **22.3 GB raw**, ~3 GB compressed
(measured 7.4x). Columnar storage buys nothing on opaque blobs, so each row is
a separately gzipped object plus a small Parquet index.

The browser mirrors **only the index** (~54k rows of id/ts/size) and fetches a
payload when the user opens one. Mirroring the bodies would blow past every
browser's IndexedDB quota.

### Partitions are presigned, not proxied

`GET /market/<table>/partition/<YYYY-MM>` returns a **302 to a presigned S3
URL**. A month of metrics exceeds API Gateway's 10 MB response cap (6 MB for
the Lambda payload), and proxying would bill for transferring every byte twice.

### pyarrow is deliberately not in the bundle

It is 136 MB and alone pushes the deployment past Lambda's 250 MB ceiling
(measured: 378 MB with it, 242 MB without). Nothing in the request path needs
it — the handler presigns URLs and the browser parses the Parquet. Only
`migrate_market.py` reads Parquet in Python, and that runs on a workstation.

## No Postgres

There is no database container and no RDS. The compose stack is LocalStack +
MinIO only.

Django still needs *a* database for the parts of contrib that assume one
(admin, the auth tables management commands touch), so under `HYBRID_DB=1`
that is a scratch SQLite file in `/tmp` — per-sandbox, holding no application
data, and expendable on every cold start. Sessions use signed cookies rather
than the database for exactly that reason.

Anything durable is in S3. Set `HYBRID_DB=0` to restore the Postgres path.

## Source data removed (no duplicates)

The migration is complete and the source copies are gone. `storage/trading_cache.db`
went from **25.5 GB to 1.5 GB** (3,445,781 rows dropped across 23 tables, then
VACUUM to return the space), and the Postgres container and its volume are
deleted. S3 is the only copy.

The deletion was gated, not assumed:

1. `verify_migration.py` compares S3 against the source **per partition** (a
   table total can match while individual months are wrong), samples values
   byte-for-byte, and confirms every blob uploaded. It exits non-zero on any
   gap, and `cleanup_source.py` refuses to run unless it passes.
2. The first run **failed by design** — 3 partitions were short by 654, 16 and
   36 rows because the live trading system was still writing. That drift never
   converges while writers run, so production and its supervisor
   (`W1z4rDV1510n/scripts/w1z4rd_supervisor.py`, which respawns it) were
   stopped, the current month re-synced, and verification re-run to 40/40.
3. Only then were the tables dropped, and production restarted.

`cleanup_source.py` deletes **only** the tables the migration copied. Tables
that were never migrated (`kv_store`, `experiments`, `system_logs`,
`pair_adjustments`, ...) are untouched — they exist nowhere else. Django's
contrib scaffolding is also kept, since it is recreated locally anyway.

The trading system recreated its tables on restart and resumed writing with no
manual repair. SQLite is now a small working buffer; the history is in S3.

A manifest of what was removed is written to `storage/migration-cleanup.json`.

### The ORM tables have to be rehydrated

A gap worth stating plainly, because it broke BrandDozer and SecureVault once:

**The browser's AllezORM tier serves the frontend. It does not serve
server-side Django code.** The admin, the DRF views and the management
commands all still issue SQL. So when the cleanup dropped the 14 migrated
tables, every BrandDozer and SecureVault view failed with "no such table" even
though the data was safe in S3.

`restore_django_tables.py` fixes that by rebuilding those tables from S3, and
`cleanup_source.py` now runs it automatically as step 4. The local database is
a **projection of S3, not a second source of truth** — re-running is safe
because rows are replaced, not appended.

Three things it has to handle, all found the hard way:

* `migrate` alone is a no-op — dropping a table does not clear its
  `django_migrations` row, so Django believes it still exists. The schema is
  built from the model definitions via Django's schema editor instead.
* A table can exist and still be **older than its model**
  (`branddozer_brandproject` was missing four columns the model declares).
  Missing columns are added with plain `ALTER TABLE`; SQLite's schema editor
  rebuilds the table and loses them.
* Rows are loaded with `PRAGMA foreign_keys = OFF`, because a child table is
  often restored before its parent. Enforcement is re-enabled afterwards and
  `foreign_key_check` reports anything genuinely dangling. Current state: 815
  rows restored, **0 violations**.

## Usage

```bash
# 1. Start the local stack (Postgres + MinIO + LocalStack)
cd serverless/local && docker compose up -d

# 2. Build the bundle (resolves deps for manylinux, not Windows)
python serverless/local/build_package.py

# 3. Deploy: 5 functions, REST API, 3 schedules, S3 buckets
bash serverless/local/deploy_local.sh

# 4. Migrate, then test
aws --endpoint-url http://localhost:4566 lambda invoke \
  --function-name coolcrypto-admin \
  --payload '{"command":"migrate","args":["--noinput"]}' \
  --cli-binary-format raw-in-base64-out /dev/stdout

python serverless/local/test_local_stack.py
```

`build_package.py --skip-deps` reuses the resolved dependencies and only
re-copies source — much faster when iterating on handler code.

## Functions

| Function | Handler | Timeout | Trigger |
|---|---|---|---|
| `coolcrypto-http` | `handlers.http` | 30s | API Gateway `{proxy+}` |
| `coolcrypto-cron` | `handlers.cron` | 900s | EventBridge (3h / 7d) |
| `coolcrypto-ws` | `handlers.websocket` | 30s | WebSocket routes |
| `coolcrypto-ws-push` | `…websocket.broadcast_handler` | 60s | EventBridge (1 min) |
| `coolcrypto-admin` | `handlers.admin_tasks` | 900s | Manual invoke |
| `coolcrypto-auth` | `handlers.auth` | 15s | API Gateway `/auth/*` |
| `coolcrypto-hybrid` | `handlers.hybrid_api` | 15s | API Gateway `/hybrid/*` |
| `coolcrypto-market` | `handlers.market_api` | 15s | API Gateway `/market/*` |

`coolcrypto-auth` and `coolcrypto-hybrid` import **no Django**: a login
cold-starts in ~0.3s instead of ~4s, cannot be reached through the dashboard's
middleware, and stays small enough to sit in the free tier.

`coolcrypto-admin` runs management commands from a **whitelist**
(`migrate`, `showmigrations`, `collectstatic`, `check`, `createcachetable`).
It holds database credentials and is invocable by anything with
`lambda:InvokeFunction`, so allowing arbitrary commands (`shell`, `dumpdata`)
would make it a remote-code-execution and data-exfiltration path.

## Going to real AWS

The deploy script uses the real AWS APIs, so it is close to production-ready,
but do not point it at an account without changing these:

1. **`DJANGO_SECRET_KEY`** is a hardcoded local value. Use Secrets Manager.
2. **`DJANGO_ALLOWED_HOSTS=*`** — set the real API Gateway domain.
3. **Postgres credentials** are `postgres/postgres` in plaintext. Use RDS with
   Secrets Manager, and put RDS Proxy in front: `CONN_MAX_AGE=0` means every
   invocation opens its own connection, and concurrent Lambdas will exhaust
   `max_connections` without pooling.
4. **The IAM role is a stub** with no policies. It needs, at minimum, RDS
   access, S3 read/write on the media bucket,
   `execute-api:ManageConnections` for the WebSocket push, and CloudWatch Logs.
5. **API Gateway has no auth** (`--authorization-type NONE`). The Django
   session layer is the only thing in front of the admin.
6. Swap the REST API for an HTTP API (cheaper, lower latency) — Mangum handles
   both payload formats. Set `API_GATEWAY_STRIP_STAGE` correctly for the stage
   style you choose.

## Known limitations

- **Cold start ~4s** (measured on container-local disk). Django with 30 apps
  plus pandas is not fast to import. Use provisioned concurrency if that
  matters. Measuring this over a Docker bind mount reports ~160s — an artifact
  of Docker Desktop's filesystem, not the bundle.
- **`LAMBDA_RUNTIME_ENVIRONMENT_TIMEOUT=300`** in the compose file is a
  workaround for that same slow volume, not a property of real Lambda.
- **`/tmp` is per-sandbox and cleared between cold starts.** Fine for the
  caches and scratch state relocated there; anything durable belongs in
  Postgres or S3.
- **WebSocket API is untested end-to-end** locally (LocalStack Pro). The
  handlers are exercised with the exact `$connect`/`$disconnect`/`$default`
  event shapes API Gateway emits.
- **The trading/ML stack is not bundled** (tensorflow, opencv, web3, kuzu).
  Scheduled tasks that need it must run as a container-image Lambda or on
  Fargate.
- **`channels` is dropped from `INSTALLED_APPS`** under Lambda. The Waitress
  deployment is unaffected — `settings.py` is untouched.
