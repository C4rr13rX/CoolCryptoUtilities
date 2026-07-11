# AgentTheFreeloader

`AgentTheFreeloader` is a peer C0d3rV2 AI backend. Select it explicitly in
the same place that selects Wizard, Bedrock, or Codex:

```powershell
$env:C0D3R_BACKEND = "freeloader"
```

ATF is exclusive. Creating an ATF session disables Wizard probes and sends in
that process. Setting `C0D3R_BACKEND=freeloader` makes the exclusivity visible
to every web/background worker started with that environment and forces the
C0d3rV2 factories to resolve to ATF instead of probing Wizard or falling
through Wizard to Bedrock. Restart workers when switching back to another
backend; this avoids mixed-backend jobs during an in-place mode change.

It implements the existing C0d3rV2 session contract:

```python
send(prompt, stream=False, system="") -> str
```

Every `send()` is one agent iteration. The plugin reads the iteration prompt
and system/tool schema, classifies the work, and ranks the callable chat models
in `docs/free_ai_model_catalog.csv`.

## Selection order

1. Capability fit for the current iteration: coding, tool use, reasoning,
   structured output, multimodal input, multilingual work, or speed.
2. Remaining quota headroom across every quota pool attached to the model.
3. Recent endpoint health.
4. Least-used model within an equivalent quality tier.

Capability scores are grouped into quality tiers. Quota and least-used state
rotate requests only among models in the best available tier; they do not make
a materially weaker model win merely because it has more quota.

## Shared quota pools

A model can belong to multiple quota pools. Local reservation succeeds only if
all pools have room. The initial catalog applies shared pools to providers whose
documented free allowance spans a model family or account, including:

- `openrouter:free`
- `sambanova:free-tier`
- `cohere:trial`
- `huggingface:monthly-credit`
- `cloudflare:neurons`
- `mistral:organization`

If one model returns HTTP 402/429, its shared pool is cooled down and sibling
models in that pool are skipped. The router then tries the next eligible model
from another pool. Published RPM/RPD/TPM/TPD/monthly limits are reserved before
the call and reconciled with provider usage metadata afterward. Provider rate
limit headers refine the remaining-headroom score.

Quota state persists globally for the repository at
`runtime/agent_the_freeloader/quota.json` by default, so different C0d3rV2
workdirs consume the same provider allowance. Override the location with
`AGENT_FREELOADER_STATE_PATH`.

## Credentials

Add one or more keys to the encrypted SecureSetting vault or environment:

- `GEMINI_API_KEY`
- `GROQ_API_KEY`
- `CEREBRAS_API_KEY`
- `SAMBANOVA_API_KEY`
- `OPENROUTER_API_KEY`
- `MISTRAL_API_KEY`
- `COHERE_API_KEY`
- `HF_TOKEN`
- `GITHUB_TOKEN`
- `CLOUDFLARE_API_TOKEN` plus `CLOUDFLARE_ACCOUNT_ID`
- `AI_GATEWAY_API_KEY` for Vercel AI Gateway
- `MODELSCOPE_ACCESS_TOKEN` for ModelScope API-Inference
- `SPEKA_API_KEY` for Speka
- `POLLINATIONS_API_KEY` optionally raises Pollinations to its registered free tier
- `SCW_SECRET_KEY` for Scaleway Generative APIs
- `ZHIPU_API_KEY` for Zhipu BigModel free Flash models
- `IOINTELLIGENCE_API_KEY` for IO Intelligence
- `DASHSCOPE_API_KEY` for Alibaba Model Studio
- `SILICONFLOW_API_KEY` for SiliconFlow
- `HYPERBOLIC_API_KEY` for Hyperbolic

Only configured providers enter the candidate set. Trial-credit providers can
remain in the research CSV without becoming runnable wildcard entries.

BlockRun's documented free NVIDIA-hosted models require no key and provide the
zero-setup fallback pool. They share `blockrun:free`; a capacity response cools
the family down before ATF negotiates another provider.

Pollinations also provides a zero-key pool through `openai-fast`. Anonymous
access is limited to one request every 15 seconds; a free Seed registration
raises that to one every 5 seconds. ATF uses the conservative anonymous quota
locally and automatically sends `POLLINATIONS_API_KEY` when configured.

The additional independent pools researched on 2026-06-28 are:

- Vercel AI Gateway: $5 of credit every 30 days on free accounts. All catalog
  entries share `vercel:monthly-credit`.
- ModelScope API-Inference: free registered developer inference with dynamic
  capacity and a documented 20-request daily model quota. It requires Alibaba
  Cloud account binding and real-name verification.
- Speka: $1 of usage included monthly and 10 RPM on the no-card free plan. All
  Speka models share `speka:monthly-credit`.
- GitHub Models: current callable model IDs replace the old placeholders.
  GitHub documents free limits per model/tier, so each model has its own quota
  pool rather than one provider-wide pool.
- Scaleway Generative APIs: new-customer token allowance shared across its
  catalog; all routes use `scaleway:new-customer`.
- Zhipu BigModel: the documented Flash routes use independent model pools.
- IO Intelligence: daily free credits are represented by one shared provider
  pool so exhaustion of one route blocks its siblings.
- Alibaba Model Studio: time-limited free token grants are tracked per model.
- SiliconFlow: fixed free models use independent model pools.
- Hyperbolic: promotional credit is one shared provider pool; HTTP 402 cools
  the entire pool for 30 days unless the provider supplies a retry interval.

Credit-denominated pools cannot be precisely reserved locally because request
cost varies by model and token mix. ATF treats them as shared provider pools,
uses returned usage and rate-limit headers when present, and cools the complete
pool on credit/quota errors.

OpenRouter defaults to the documented 50 free requests/day allowance. Set
`AGENT_FREELOADER_OPENROUTER_1K_RPD=1` only after the account has purchased the
required lifetime credits for the 1,000 requests/day allowance.

Optional filters:

```powershell
$env:AGENT_FREELOADER_PROVIDERS = "Groq,OpenRouter,Google Gemini API"
$env:AGENT_FREELOADER_DENY_PROVIDERS = "Cohere"
$env:AGENT_FREELOADER_MODELS = "openai/gpt-oss-120b,qwen/qwen3-32b"
```

## Observability

`session.last_route` records each attempted model, capability score, quality
tier, quota headroom, and outcome. `session.get_model_id()` returns the model
that served the last successful iteration.

The current repository still maps the legacy `codex` backend name to Wizard.
This plugin does not change that existing behavior.

## Unattended workday supervisor

ATF includes a standalone durable queue for leaving C0d3rV2 working for a
bounded shift. It does not require Django. Jobs, checkpoints, attempts,
heartbeats, cancellation state, and expiring leases are stored in SQLite at
`runtime/agent_the_freeloader/workday.sqlite3` by default.

Queue small, independently verifiable jobs. A validation command is strongly
recommended because its exit status feeds semantic success/failure back into
future model ranking:

```powershell
$env:C0D3R_BACKEND = "freeloader"

python scripts/atf_workday.py enqueue `
  "Implement the parser change and its focused tests." `
  --workdir D:\Projects\MyProject `
  --validate "python -m pytest tests/test_parser.py -q" `
  --max-attempts 4 `
  --timeout 1800

python scripts/atf_workday.py run --hours 8
```

The supervisor runs each job in a separate process. It renews a lease while
the worker is alive, kills the full child process tree on cancellation or
timeout, requeues expired leases after a crash, and applies exponential or
quota-specific retry delays. Wizard remains disabled throughout the worker.

Operational commands:

```powershell
python scripts/atf_workday.py status
python scripts/atf_workday.py list --status retry
python scripts/atf_workday.py show JOB_ID
python scripts/atf_workday.py cancel JOB_ID
python scripts/atf_workday.py retry JOB_ID --extra-attempts 1
python scripts/atf_workday.py report --hours 24
python scripts/atf_workday.py run --until-empty
```

Shift reports are written as JSON and Markdown under
`runtime/agent_the_freeloader/reports/`. Reports include completed, failed,
retrying, and cancelled jobs plus rolling 24-hour request/token usage.

### Workday controls

| Environment variable | Default | Purpose |
| --- | ---: | --- |
| `ATF_WORKDAY_CONCURRENCY` | `1` | Maximum isolated workers. Keep this low for free tiers. |
| `ATF_WORKDAY_SHIFT_HOURS` | `8` | Default bounded supervisor runtime. |
| `ATF_WORKDAY_MAX_REQUESTS` | `200` | Rolling 24-hour request budget; `0` disables. |
| `ATF_WORKDAY_MAX_TOKENS` | `2000000` | Rolling 24-hour token budget; `0` disables. |
| `ATF_WORKDAY_JOB_TIMEOUT_SECONDS` | `1800` | Default hard job timeout. |
| `ATF_WORKDAY_LEASE_SECONDS` | `90` | Crash-recovery lease duration. |
| `ATF_WORKDAY_HEARTBEAT_SECONDS` | `15` | Worker lease-renewal interval. |
| `ATF_WORKDAY_RETRY_SECONDS` | `60` | Base exponential retry delay. |
| `ATF_WORKDAY_QUOTA_RETRY_SECONDS` | `300` | Delay after quota/rate-limit exhaustion. |
| `AGENT_FREELOADER_WORKDAY_DB` | runtime path | Override the queue database. |
| `AGENT_FREELOADER_FEEDBACK_PATH` | runtime path | Override persistent model-quality feedback. |

Successful or failed validation updates a persistent semantic-health score for
the final serving model. This complements endpoint health: a model that returns
HTTP 200 but repeatedly produces code that fails tests is gradually demoted,
while one that passes validation is promoted. The prior prevents a single
noisy result from permanently excluding a model.

Free-provider availability remains best effort. Configure providers with
independent quota pools if the queue must survive one provider family becoming
unavailable. A job with no validation command can still run, but only transport
completion can be measured automatically; such a job provides weaker semantic
feedback.

Capacity-only failures do not consume a job attempt. The worker returns the
job to `retry`, retains its checkpoint, and waits for
`ATF_WORKDAY_QUOTA_RETRY_SECONDS`. Corrective retries include the prior model
output, validation error, and validator evidence. A later passing retry marks
the prior correction event resolved.

## Hallucination telemetry and local retrieval

ATF stores correction events in the feedback SQLite database with provider,
model, classification, trigger, failed output, correction, resolution state,
and branch metadata. Operational endpoint failures are not labeled as model
hallucinations. The Model Control page displays recent events.

Before each C0d3rV2 branch iteration, relevant prior corrections are retrieved
with the CPU-only `BAAI/bge-small-en-v1.5` embedding model through FastEmbed
and ONNX Runtime. Set `ATF_LOCAL_VERIFIER=0` for deterministic lexical-only
retrieval. The local model ranks evidence; it does not decide factual truth.
Use `deploy/environment-atf-cpu.yml` with Conda/Miniforge, or install
`deploy/requirements-atf-cpu.txt` into a Python environment.

Tool errors involving versions, packages, APIs, imports, or unsupported
arguments automatically trigger C0d3rV2's `web_search` before the corrective
model call. Web results become task-tree evidence available to later steps.

## Benchmarks

Scientific GUI artifact benchmarks run in isolated runtime directories and
are built and repaired only through C0d3rV2+ATF:

```powershell
python scripts/atf_benchmarks.py --list
python scripts/atf_benchmarks.py --case django-spectrum-instrument --hours 2
```

The catalog covers Django, DearPyGui, Ionic 8/Angular, Qt 6/C++20, and
Tkinter. Validators reject empty test suites and fabricated native-build
claims. On a host without Qt or a C++ compiler, the Qt case requires explicit
disclosure and static CMake/test-target evidence.

Complex planning benchmarks score sequential dependencies, constraint
coverage, measurable acceptance criteria, and recovery/reconvergence policy:

```powershell
python scripts/atf_planning_benchmarks.py --list
python scripts/atf_planning_benchmarks.py --case blockchain-hybrid-cluster
```
