# U53RxR080T Rust Agent Build Kit

This kit includes the agent source (`agent-rs`) and simple build scripts so you can compile a binary for your OS and re-upload it to the dashboard static downloads.

## Requirements
- Rust toolchain (`cargo`) installed on your OS.
- For Windows: build on a Windows machine with the MSVC toolchain.
- For Linux/macOS: standard Rust install is enough.

## Build (Linux/macOS)
```bash
cd agent-rs
cargo build --release
zip ../rust-agent-$(uname | tr '[:upper:]' '[:lower:]').zip target/release/u53rx-agent
```
Then upload `rust-agent-<os>.zip` to `web/static/u53rxr080t/` on the server (replacing the placeholder). The dashboard download button will serve your binary.

## Build (Windows, PowerShell)
```powershell
cd agent-rs
cargo build --release
Compress-Archive -Path target/release/u53rx-agent.exe -DestinationPath ../rust-agent-windows.zip -Force
```
Upload `rust-agent-windows.zip` to `web/static/u53rxr080t/` on the server.

## Notes
- The agent listens on 127.0.0.1:36279 by default. Run with: `./u53rx-agent --server http://127.0.0.1:8000 --name rust-daemon --enable-loop`.
- The browser extension Options page has a `Daemon URL` field; set it to `http://127.0.0.1:36279`.
- If you need cross-compilation, build on the target OS for best results.
