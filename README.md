# R3V3N!R

**An autonomous crypto trading system that refuses to trade until it has earned the right.**

Most trading bots ask you to trust a backtest. This one makes every strategy prove itself in ghost mode — 20 closed trades, 55% win rate, positive P&L — before a single real dollar moves. Strategies graduate independently. Strategies that stop performing get demoted automatically. The gate is the product.

Behind it: 70 strategies across 6 time horizons, a post-quantum login, a database that costs cents instead of hundreds, and the whole thing runs on your phone.

906 tests. One command to start.

---

## The trading engine

### Ghost-to-live graduation that cannot be fooled

Every strategy runs in simulation first, keeping its own independent ledger. Promotion requires evidence, not optimism:

| Gate | Requirement |
|---|---|
| Sample size | 20 closed ghost trades |
| Win rate | ≥ 55% |
| Profitability | Net positive P&L |
| Demotion | 4 consecutive live losses → straight back to ghost |

Demotion **wipes the ghost record**. Re-graduation demands fresh evidence, never stale pre-demotion stats.

Why this matters: a contaminated feed once produced 969 ghost trades at a 0% win rate. The gate caught it. The ledger was reset and rebuilt from zero — which is exactly what a gate you can trust is supposed to cost you.

```bash
python scripts/readiness_report.py     # what has actually been proven, per strategy
```

The report computes the **Wilson lower bound** on every win rate, because 7 wins from 7 trades looks like 100% and is only evidence of 49%. That number is why trade count can't be waived.

### 70 strategies, 6 horizons, one ledger each

Mean reversion, momentum, OBV accumulation, RSI reversal, VWAP bands, breakout, arbitrage cells — each replicated across **5h, 12h, 1d, 3d, 5d and 1w** horizons and graded separately. A signal that works weekly but not hourly gets promoted weekly and blocked hourly. No blending, no averaging away the truth.

### Guardrails that assume things go wrong

- **SAT/UNSAT gas solver** — proves a trade is affordable before attempting it, and tells you exactly what to fund when it isn't
- **VWAP deviation guard** — rejects cross-exchange denomination mixing before it poisons a window
- **Resource governor + reflex blocks** — halts on anomaly, not after the loss
- **Stablecoin decimal integrity** — the bug class that quietly loses $50 and never explains why

---

## Post-quantum authentication

**The threat is real and it is already happening: an adversary records your TLS traffic today and decrypts it in ten years.**

TLS 1.3's key exchange is X25519, which Shor's algorithm breaks. A password captured that way stays valid, because nobody rotates credentials on a quantum-computing schedule.

So the password never relies on TLS alone:

```
ML-KEM-768 (FIPS 203)  →  HKDF-SHA-384  →  AES-256-GCM  →  Argon2id
   encapsulation           transcript-bound     sealed        64 MiB verifier
```

- **Per-attempt keypairs** — a compromised key exposes a 2-minute window, not your history
- **Transcript binding** — a captured envelope replayed against a different challenge simply fails
- **HMAC-SHA-384 sessions, not JWT** — no `alg` header to confuse, no algorithm agility to exploit
- **Real revocation** — logout invalidates server-side, not just in the browser

Verified interoperable: the browser's `@noble/post-quantum` seals an envelope Python's `kyber-py` opens.

---

## The hybrid database

**S3 is the database. The browser is the query engine. There is no RDS bill.**

```
AllezORM / IndexedDB          ← real SQL in the tab, offline, survives reload
      ↓  REST over API Gateway
Lambda (stateless)
      ↓
S3: database/tables/<table>/<id>.json
```

RDS charges by the hour whether traffic arrives or not. S3 charges per request and scales to zero — no VPC, no NAT gateway, no proxy. Every read the browser answers locally is a request, an invocation, and a GET that never happen.

**Measured on the real migration:**

| | |
|---|---|
| Shared market data | 3,386,863 rows → **25 Parquet objects, 29.5 MB** |
| Snapshot archive | 22.3 GB → **1.5 GB** (14.9x) |
| Source database | 25.5 GB → **1.5 GB** |

Market data is columnar and month-partitioned, so a price chart transfers `ts` and `price` and never touches the raw JSON column. Closed months are immutable — only the current month is ever refetched.

Deletion was gated, not assumed: a per-partition verifier compared every row against the source and **failed twice** before passing 40/40. It refused to delete until it did.

---

## Runs on Lambda. Runs on your phone.

The same handlers serve both. Not a port — the same code.

**Lambda's value was never "the cloud." It's that nothing runs when nothing is happening.** On a server that saves money; on a phone it saves battery, which is scarcer.

```
User taps      → local API Gateway → invoke("http", event)
Schedule fires → JobScheduler ─────→ invoke("cron", event)
                                          ↓
                              the same code deployed to AWS
```

Between invocations, **nothing runs**. No polling threads. No wake locks. No resident server.

| | |
|---|---|
| Cold start | ~3.0 s |
| Warm request | **18 ms** |
| Signed APK | 115 MB, arm64 |

Android's `JobScheduler` batches the app's wakeups with every other app's and defers under Doze — something a private timer can never do. The Rust wizard node ships as a native binary and updates itself over the air, verifying SHA-256 before replacing anything.

---

## BrandDozer: evidence-first research

Runs software-delivery projects **or** archival research that refuses to overclaim.

- **Epistemic gates** — a paper that cannot support a claim reports the gap instead of filling it
- **Reproducibility manifest** — every figure traceable to the data that produced it
- **Retained revisions** — the full history, searchable
- **Agent auto-resume** — long research survives restarts

Full workflow: [`docs/BRANDDOZER_RESEARCH_WORKFLOW.md`](docs/BRANDDOZER_RESEARCH_WORKFLOW.md)

### Video Studio

Compiles a research deck into a narrated MP4 — 9:16 portrait, cube-flip transitions, words revealed on the real speech-mark timings. 6 aspect ratios, 9 transitions, 6 word animations. Frames composed in PIL, piped straight to ffmpeg; no system install, no moviepy.

---

## c0d3r: a senior engineer that shows its work

Turns an objective into disciplined engineering: clarify, plan, execute, inspect, iterate. It runs commands, reads the output, and adapts — then records what it did so the work is explainable and repeatable.

- **Evidence-driven planning** with explicit acceptance criteria
- **Scientific-method loop** — hypothesis, experiment, measurement, correction
- **Verification hooks** — micro-checks and full lint/build/test before declaring success
- **Filesystem safety** — project-root-only mutation under strict mode
- **Audit-friendly** — transcripts, exit codes, output slices

**Turing rubric + memory harness** probes STM/LTM recall, multi-turn consistency and tool use:

```powershell
powershell -ExecutionPolicy Bypass -File runtime\c0d3r\obstacle_course.ps1
```

**Graph store:** the equation matrix lives in Django's DB *and* mirrors into an embedded Kùzu graph for traversal. Path `storage/graph/kuzu` (`GRAPH_DB_DIR`).

---

## The W1z4rD brain

A Rust learning node on `:8090` that reads market regime and feeds the trading loop as a **bounded second opinion** — maximum 0.06 influence on any decision.

That bound is deliberate. It can tip a call the model already nearly made; it cannot manufacture one. A confidently wrong node cannot drag a neutral prediction across the entry threshold.

Its confidence scoring measures **one-sidedness × evidence**, never keyword density — because density measures verbosity. Under the old scheme the word "bull" scored 1.000 while a thorough hedged analysis scored 0.120. The tersest reply carried the most weight. That is fixed, and the floor is calibrated from the measured distribution rather than intuition.

---

## Everything else in the box

- **19 Django apps, 27 UI routes** — wallet, telemetry, streams, guardian, cron, datalab, model lab, investigations, address book, code graph
- **Secure vault** — Kyber-encrypted per-user secrets, never plaintext at rest
- **Wallet tooling** — balances, transfers, NFTs, swaps, optional bridging across 8 chains (Ethereum, Base, Arbitrum, Optimism, Polygon, BSC, Avalanche, zkSync)
- **Guardian supervisor** — scheduled monitoring with leases, so two processes never fight
- **42 languages** in the UI
- **Multi-provider AI** — Codex CLI, AWS Bedrock, or the local wizard node

---

## Quickstart

```powershell
scripts\quickstart.ps1            # Windows PowerShell
scripts\quickstart.cmd            # Windows CMD
bash scripts/quickstart.sh        # macOS / Linux
```

Installs dependencies, defaults to SQLite, adds the repo CLIs to PATH, migrates, and starts the server.

<details>
<summary>Platform-specific</summary>

**Windows**
```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd web; python manage.py migrate; python run_waitress.py
```

**macOS / Linux**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd web && python manage.py migrate && python run_waitress.py
```

**Frontend**
```bash
cd web/frontend && npm install && npm run build
```

**Android APK**
```powershell
cd android
./build_wizard_node.ps1          # cross-compile the Rust node
./gradlew assembleDebug
```
Requires JDK 17, Android SDK 36, NDK 27, Rust with `aarch64-linux-android`.

</details>

Dashboard: **http://127.0.0.1:8001/**

**Requirements:** Python 3.11+ · Node 18+ (frontend) · Docker (serverless testing)

---

## Honest limitations

Stated plainly, because a system that hides these is not one you should trust with money:

- **Organism, Code Graph and U53R xR080T are rough** and under active development
- **TensorFlow has no Android wheel** — on-device model scoring is skipped and reported as `tf_unavailable` rather than silently returning nothing
- **The APK is unproven on hardware** — it builds, signs and verifies; it has not been installed on a phone
- **The trading system has not graduated a strategy yet.** At the current rate of evidence, the nearest is weeks away. That is the gate doing its job, not a defect.

---

## License

MIT — see [LICENSE](LICENSE).
