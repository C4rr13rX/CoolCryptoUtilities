# U53RxR080T agent protocol (draft)

A minimal loop for browser extensions or the Rust desktop helper to exercise the UX and report back.

## Auth
Use the same Django session auth as the dashboard (or a DRF token if enabled). All endpoints live under `/api/u53rxr080t/` and expect JSON.

## Heartbeat
Register/update an agent every ~15s. The `id` is optional; the server will mint one.

```
POST /api/u53rxr080t/heartbeat/
{
  "id": "<agent-uuid>",
  "name": "edge-plugin",
  "kind": "browser",
  "platform": "windows",
  "browser": "edge",
  "status": "idle",
  "meta": {"version": "0.1.0", "viewport": [1440, 900]}
}
```
Response: `{ "id": "<uuid>", "status": "idle" }`

## Claim work
Fetch the next pending task (oldest first) and claim it for the agent.

```
POST /api/u53rxr080t/tasks/next/
{ "agent_id": "<uuid>" }
```
Response: `{ "task": { ...serialized task... } }` or `{ "task": null }` if the queue is empty.

## Update task status
When starting/finishing a task, report the status and any metadata (timings, URLs, hashes).

```
POST /api/u53rxr080t/tasks/<task_id>/
{
  "status": "in_progress" | "done" | "error",
  "meta": {"elapsed_ms": 1200, "note": "clicked all nav items"},
  "assigned_to": "<agent-uuid>"   // optional, keeps assignment sticky
}
```

## Report findings
Send UX observations, errors, or screenshots with context.

```
POST /api/u53rxr080t/findings/
{
  "session": "<agent-uuid>",
  "title": "Navigation buttons unclickable",
  "summary": "JSON snapshot parse failed in main.js, navigation disabled",
  "severity": "warn",   // info | warn | error
  "screenshot_url": "https://.../nav-error.png",
  "context": {"route": "/", "stack": "..."}
}
```

## Listings (used by the dashboard view)
- `GET /api/u53rxr080t/agents/` → `{ agents: [...] }`
- `GET /api/u53rxr080t/tasks/?status=pending` → `{ tasks: [...] }`
- `GET /api/u53rxr080t/findings/?limit=100` → `{ findings: [...] }`

## Sending updates to Codex CLI
`services/u53_agent.py::send_codex_update` writes transcripts to `runtime/u53rxr080t/transcripts` and routes through the CodexSession integration so operator updates land in the CLI.

## Suggested agent loop
1. `heartbeat`
2. `tasks/next` → if task
3. Drive browser/desktop, capture screenshots/har files as needed
4. `tasks/<id>` status → `in_progress`/`done`/`error`
5. `findings` for any observations
6. `heartbeat` again

This is intentionally small; extend the payloads as we add richer UX flows.
