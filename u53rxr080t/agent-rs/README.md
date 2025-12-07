# U53RxR080T Rust Agent (daemon)

Headless companion to the browser extension. Runs locally, talks to the Django API, captures screenshots, and can enqueue Guardian tickets.

## Features
- Local HTTP API (defaults to `127.0.0.1:36279`): `/health`, `/screenshot`, `/sequence`, `/task/once`.
- Task loop (opt-in) that heartbeats, claims `/api/u53rxr080t/tasks/next/`, captures a screenshot, calls `/suggest/`, posts `/findings/`, and updates the task.
- Fallback screenshot if OS capture fails (real capture when `screenshots` crate works on the host).
- Extension can call `/sequence` for multi-shot captures and `/health` to detect the daemon.

## Build
Requires Rust toolchain (`cargo`).

```bash
cd u53rxr080t/agent-rs
cargo build --release           # linux
# for windows (on Windows host): cargo build --release --target x86_64-pc-windows-msvc
```

Outputs will be in `target/release/u53rx-agent` (or `.exe`). Package the binary into:
- `web/static/u53rxr080t/rust-agent-linux.zip`
- `web/static/u53rxr080t/rust-agent-windows.zip`

## Run
```bash
./u53rx-agent \
  --server http://127.0.0.1:8000 \
  --token "<optional bearer token>" \
  --name rust-daemon \
  --enable-loop
```
Endpoints then live on `http://127.0.0.1:36279`.

## Extension handshake
- Extension polls `/health`; if reachable, it can ask the daemon for `/sequence` captures instead of `captureVisibleTab`.
- Daemon posts findings and suggestions to Django and enqueues Guardian tickets when suggestions are present.

## Notes
- Screenshot capture depends on OS support (`screenshots` crate). If it fails, the agent returns a placeholder image but still reports tasks.
- This repo does not include the compiled binaries yet; build on the target OS and drop the zips into `web/static/u53rxr080t/`.
