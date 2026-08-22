# CoolCrypto — native Android app

The dashboard, the trading services and the W1z4rDV1510n node, packaged as one
APK. The Vue GUI is used **exactly as built** — no frontend changes.

## The idea: Lambda as a mobile efficiency model

This is the part worth reading, because it drives every other decision.

**Lambda's value was never "the cloud". It is that nothing runs when nothing is
happening.** On a server that saves money; on a phone it saves battery and RAM,
which is the scarcer resource. So the same handlers that run in AWS run here,
invoked the same way, through the same `(event, context)` contract:

```
User taps  ──> local API Gateway ──> lambda_runtime.invoke("http", event)
                (android_bootstrap)        │
Schedule fires ─> JobScheduler ──────────> lambda_runtime.invoke("cron", event)
                                           │
                                    serverless/handlers/*.py
                                    (the same code deployed to AWS)
```

Between invocations, **nothing runs**. No polling threads, no wake locks, no
resident web server.

### What this replaced

The first draft of this port ran guardian, cron, pipeline and production as
four `while True` threads holding a `PARTIAL_WAKE_LOCK`. That is the always-on
server model, and on a phone it is worse than on a server: the CPU never idles,
the radio never sleeps, and the OS cannot reclaim any of it. It was deleted.

`JobScheduler` now owns the clock, exactly as EventBridge owns it in AWS. Two
things follow that a private timer can never achieve:

* the OS **batches** our wakeups with every other app's, so the device wakes
  once for many jobs instead of once for ours;
* Doze and app-standby are respected rather than fought.

### Measured

Taken on the desktop through the real gateway and the real handlers:

| | |
|---|---|
| Cold start (first invocation, imports Django + 30 apps) | ~3.0 s |
| Warm `GET /api/console/status/` | **18 ms** |
| Warm `GET /api/telemetry/dashboard/` (27 KB) | **58 ms** |
| `POST /api/auth/challenge` (ML-KEM-768 keygen) | 651 ms |
| Dashboard root (30 KB) | 348 ms |

The 165x gap between cold and warm is why `warm_handlers()` runs once at
startup: it buys a responsive first tap for a few hundred milliseconds of
import, and is the one deliberate exception to "nothing runs when idle".

## Components

| Piece | What it is | Why |
|---|---|---|
| `DjangoService` | Foreground service hosting the local API Gateway on `127.0.0.1:8765` | The WebView is useless without it, and Android kills background processes holding a socket |
| `ScheduledJobService` | `JobService` running scheduled invocations | Batched, deferrable, survives reboot (`setPersisted`) |
| `WizardNodeService` | Foreground service running the Rust node on `127.0.0.1:8090` | Separate from Django so it can be hot-swapped by the updater |
| `WizardUpdateWorker` | WorkManager job, every 6 h | Pulls node builds from C4rr13rX |
| `MainActivity` | Full-screen WebView | Loads the Vue GUI unchanged |

## Wizard node updates from C4rr13rX

Both a scheduled worker and a route, as requested:

```
GET  /api/wizard-chat/node/status/    what build is installed
POST /api/wizard-chat/node/update/    check and install  {"force": true} to reinstall
```

The flow is manifest → compare version → download to temp → **verify sha256** →
atomic `os.replace` → restart the service.

Verifying before installing is not ceremony. This project has already been
bitten by a node process running a stale binary whose routes 404'd — the
symptom was 4,952 of 5,000 pushes failing with accuracy at zero. Recording the
version and hash makes *"which build is actually running?"* answerable.

Tested: install → unchanged on re-run → **corrupt hash rejected without
touching the working binary**.

## What does not run on-device

**TensorFlow.** There is no Android wheel. Several `services/` and `trading/`
modules import it at module scope, and on the URLconf import path a single
missing package takes down the whole site — the same failure the Lambda build
hit with `bs4`.

`tf_stub.py` registers a stub so `import tensorflow` succeeds and *using* it
raises `TensorFlowUnavailable`. The distinction is deliberate:

* importing must succeed, or unrelated views 500;
* calling must fail loudly, or a caller believes it trained a model that never
  existed.

`tf_available()` lets callers branch, and status endpoints report
`tf_unavailable` so the UI can say so instead of showing a blank chart. For
real on-device inference, export the models to TFLite.

## Build status

**The APK builds and is signed.** Verified output:

```
app-debug.apk            70.5 MB    v2 signature: Verifies
  assets/  60.1 MB       CPython 3.12 + Django 5.1.4 + numpy + pandas
  lib/     11.8 MB       arm64-v8a natives
  4 dex files             8.7 MB
package: com.coolcrypto.dashboard.debug
```

Every dependency the request path needs is present in the APK: `numpy`,
`pandas`, `kyber_py`, `cryptography`, `rest_framework`, `whitenoise`,
`corsheaders`, `feedparser`, `yaml`, `requests`.

### Four build problems and their fixes

Each of these failed the build outright; recording them so the next person does
not rediscover them:

1. **`sdk.dir` in `local.properties`** — Windows backslashes must be escaped or
   written as forward slashes, or AGP fails in `SdkLocator.validateSdkPath`
   with the unhelpful "The filename, directory name, or volume label syntax is
   incorrect".
2. **Gradle hashing live runtime files** — pointing `python.srcDirs` at the
   repo made Gradle try to MD5 `runtime/guardian/internal-cron.lock`, held open
   by the running trading system. Fixed with the `stagePython` task, which
   copies only source (22 MB) into `build/pythonStage`.
3. **`--` inside XML comments is illegal.** The manifest comments used it as an
   em dash and the merger died with a bare "Error parsing AndroidManifest.xml".
4. **No launcher icon.** `mipmap-anydpi-v26` adaptive icon, vector only, so
   there are no per-density rasters to maintain.

## Build

```powershell
# 1. Cross-compile the Rust node (pure Rust + rustls, so no C deps)
cd android
./build_wizard_node.ps1

# 2. Build the frontend the app serves
cd ../web/frontend && npm run build

# 3. Build the APK
cd ../../android
./gradlew assembleDebug
```

Requires: JDK 17, Android SDK 36, NDK 27.1.12297006, Rust with the
`aarch64-linux-android` target (the script adds it).

### Why `libw1z4rd_node.so`

The node is an executable, not a library, but it ships in `jniLibs` under that
name because **Android only extracts and grants execute permission to files
matching `lib*.so`** in the native library directory. A binary placed in
`assets/` lands on a `noexec` mount and fails with "Permission denied".

## Notes and limits

* **arm64 only.** Every shipping Android phone is arm64, and each extra ABI
  duplicates CPython, numpy and pandas — roughly +40 MB for hardware nobody
  runs this on.
* **Invocations are serialised.** One at a time, mirroring a Lambda sandbox.
  Concurrent Django requests here contend on one SQLite file and on
  module-level state that was never written to be re-entrant; that showed up
  as requests that hang rather than fail.
* **One process-wide event loop, created but not running.** Mangum drives the
  ASGI app with `run_until_complete()`, which raises "This event loop is
  already running" against a `run_forever()` loop, and binds a
  `ThreadPoolExecutor` to whichever loop is current — so a fresh loop per
  request thread deadlocks on the second invocation.
* **HTTP/1.0 with explicit `Connection: close`.** Keep-alive requires
  consistent advertisement; a mismatch leaves the WebView waiting on a socket
  that never produces another byte.
* **Cleartext is permitted only to loopback** (`network_security_config.xml`).
  Everything else keeps the platform TLS default.
* **`BootReceiver` does not start the gateway or the node.** Those serve the
  UI; starting them when the user has not opened the app would burn battery
  for nothing.
