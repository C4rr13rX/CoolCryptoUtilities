use std::{
    fs, io,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::OnceLock,
    time::Duration,
};

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::{net::TcpListener, process::Command, sync::oneshot, time};
use tracing::{error, info};

#[cfg(windows)]
use windows_service::{
    define_windows_service,
    service::{
        ServiceControl, ServiceControlAccept, ServiceExitCode, ServiceState, ServiceStatus,
        ServiceType,
    },
    service_control_handler::{self, ServiceControlHandlerResult},
    service_dispatcher,
};

const SERVICE_NAME: &str = "C0d3rNativeOsService";
static SERVICE_ARGS: OnceLock<Args> = OnceLock::new();

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Privileged loopback OS service for C0D3R V2")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8765")]
    listen: String,

    #[arg(long, default_value = "")]
    token: String,

    #[arg(long, default_value = "")]
    token_file: String,

    #[arg(long, default_value_t = false)]
    service: bool,
}

#[derive(Clone)]
struct AppState {
    token: String,
}

#[derive(Debug, Deserialize)]
struct ExecRequest {
    command: String,
    cwd: Option<String>,
    shell: Option<String>,
    timeout_seconds: Option<u64>,
}

#[derive(Debug, Serialize)]
struct ExecResponse {
    return_code: Option<i32>,
    stdout: String,
    stderr: String,
    timed_out: bool,
}

#[derive(Debug, Deserialize)]
struct FsRequest {
    action: String,
    path: Option<String>,
    target: Option<String>,
    content: Option<String>,
    encoding: Option<String>,
    recursive: Option<bool>,
    overwrite: Option<bool>,
    limit: Option<usize>,
}

#[derive(Debug, Serialize)]
struct FsEntry {
    name: String,
    path: String,
    kind: String,
    bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct FsResponse {
    status: String,
    path: Option<String>,
    target: Option<String>,
    content: Option<String>,
    entries: Option<Vec<FsEntry>>,
    bytes: Option<usize>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let args = Args::parse();

    #[cfg(windows)]
    if args.service {
        let _ = SERVICE_ARGS.set(args.clone());
        if let Err(err) = service_dispatcher::start(SERVICE_NAME, ffi_service_main) {
            error!("service dispatcher failed: {err}");
        }
        return;
    }

    if let Err(err) = run_server(args, async {
        let _ = tokio::signal::ctrl_c().await;
    })
    .await
    {
        error!("{err}");
        std::process::exit(1);
    }
}

#[cfg(windows)]
define_windows_service!(ffi_service_main, service_main);

#[cfg(windows)]
fn service_main(_arguments: Vec<std::ffi::OsString>) {
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let shutdown_tx = std::sync::Mutex::new(Some(shutdown_tx));
    let status_handle = service_control_handler::register(SERVICE_NAME, move |event| match event {
        ServiceControl::Stop | ServiceControl::Shutdown => {
            if let Some(tx) = shutdown_tx.lock().ok().and_then(|mut guard| guard.take()) {
                let _ = tx.send(());
            }
            ServiceControlHandlerResult::NoError
        }
        _ => ServiceControlHandlerResult::NotImplemented,
    });

    let Ok(status_handle) = status_handle else {
        return;
    };
    let _ = status_handle.set_service_status(ServiceStatus {
        service_type: ServiceType::OWN_PROCESS,
        current_state: ServiceState::Running,
        controls_accepted: ServiceControlAccept::STOP | ServiceControlAccept::SHUTDOWN,
        exit_code: ServiceExitCode::Win32(0),
        checkpoint: 0,
        wait_hint: Duration::from_secs(10),
        process_id: None,
    });

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return,
    };
    let args = SERVICE_ARGS
        .get()
        .cloned()
        .unwrap_or_else(|| Args::parse_from(["c0d3r-native-os-service", "--service"]));
    rt.block_on(async {
        let _ = run_server(args, async {
            let _ = shutdown_rx.await;
        })
        .await;
    });

    let _ = status_handle.set_service_status(ServiceStatus {
        service_type: ServiceType::OWN_PROCESS,
        current_state: ServiceState::Stopped,
        controls_accepted: ServiceControlAccept::empty(),
        exit_code: ServiceExitCode::Win32(0),
        checkpoint: 0,
        wait_hint: Duration::from_secs(5),
        process_id: None,
    });
}

async fn run_server<F>(args: Args, shutdown: F) -> Result<(), String>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let token = resolve_token(&args).map_err(|err| format!("token error: {err}"))?;
    if token.trim().is_empty() {
        return Err("empty token is not allowed".to_string());
    }
    let addr: SocketAddr = args
        .listen
        .parse()
        .map_err(|err| format!("invalid listen address {}: {err}", args.listen))?;
    if !addr.ip().is_loopback() {
        return Err("refusing to bind privileged OS service to a non-loopback address".to_string());
    }

    let state = AppState { token };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/exec", post(exec_command))
        .route("/v1/fs", post(fs_op))
        .with_state(state);
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|err| format!("bind failed: {err}"))?;
    info!("listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|err| format!("server failed: {err}"))
}

fn resolve_token(args: &Args) -> io::Result<String> {
    if !args.token.trim().is_empty() {
        return Ok(args.token.trim().to_string());
    }
    let token_file = if !args.token_file.trim().is_empty() {
        PathBuf::from(args.token_file.trim())
    } else if let Ok(path) = std::env::var("C0D3R_NATIVE_OS_TOKEN_FILE") {
        PathBuf::from(path)
    } else {
        std::env::current_dir()?
            .join("runtime")
            .join("native_os_service")
            .join("token.txt")
    };
    fs::read_to_string(token_file).map(|s| s.trim().to_string())
}

async fn health(State(_state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({"status":"ok","service":"c0d3r-native-os-service"}))
}

fn authorized(headers: &HeaderMap, state: &AppState) -> bool {
    let expected = state.token.trim();
    let direct = headers
        .get("x-c0d3r-native-token")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .trim();
    if !direct.is_empty() && direct == expected {
        return true;
    }
    let bearer = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .trim();
    bearer
        .strip_prefix("Bearer ")
        .map(|value| value.trim() == expected)
        .unwrap_or(false)
}

fn unauthorized() -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::UNAUTHORIZED,
        Json(ErrorResponse {
            error: "unauthorized".to_string(),
        }),
    )
}

async fn exec_command(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ExecRequest>,
) -> impl IntoResponse {
    if !authorized(&headers, &state) {
        return unauthorized().into_response();
    }
    if req.command.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "command is required".to_string(),
            }),
        )
            .into_response();
    }
    match run_command(req).await {
        Ok(response) => Json(response).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: err }),
        )
            .into_response(),
    }
}

async fn run_command(req: ExecRequest) -> Result<ExecResponse, String> {
    let shell = req.shell.unwrap_or_else(|| "powershell".to_string());
    let mut command = match shell.to_lowercase().as_str() {
        "cmd" | "cmd.exe" => {
            let mut c = Command::new("cmd.exe");
            c.arg("/C").arg(&req.command);
            c
        }
        "none" | "direct" => {
            let mut parts = req.command.split_whitespace();
            let exe = parts.next().ok_or_else(|| "empty command".to_string())?;
            let mut c = Command::new(exe);
            c.args(parts);
            c
        }
        _ => {
            let mut c = Command::new("powershell.exe");
            c.arg("-NoProfile")
                .arg("-ExecutionPolicy")
                .arg("Bypass")
                .arg("-Command")
                .arg(&req.command);
            c
        }
    };
    if let Some(cwd) = req.cwd.as_deref().filter(|s| !s.trim().is_empty()) {
        command.current_dir(cwd);
    }
    command.kill_on_drop(true);
    let timeout = Duration::from_secs(req.timeout_seconds.unwrap_or(120).clamp(1, 3600));
    let child = command.output();
    match time::timeout(timeout, child).await {
        Ok(Ok(output)) => Ok(ExecResponse {
            return_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            timed_out: false,
        }),
        Ok(Err(err)) => Err(err.to_string()),
        Err(_) => Ok(ExecResponse {
            return_code: None,
            stdout: String::new(),
            stderr: format!("command timed out after {} seconds", timeout.as_secs()),
            timed_out: true,
        }),
    }
}

async fn fs_op(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<FsRequest>,
) -> impl IntoResponse {
    if !authorized(&headers, &state) {
        return unauthorized().into_response();
    }
    match do_fs_op(req) {
        Ok(response) => Json(response).into_response(),
        Err(err) => (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: err })).into_response(),
    }
}

fn required_path(req: &FsRequest) -> Result<PathBuf, String> {
    req.path
        .as_deref()
        .filter(|p| !p.trim().is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| "path is required".to_string())
}

fn do_fs_op(req: FsRequest) -> Result<FsResponse, String> {
    let action = req.action.trim().to_lowercase();
    match action.as_str() {
        "list" => {
            let path = required_path(&req)?;
            let limit = req.limit.unwrap_or(500).min(5000);
            let mut entries = Vec::new();
            for item in fs::read_dir(&path)
                .map_err(|err| err.to_string())?
                .take(limit)
            {
                let item = item.map_err(|err| err.to_string())?;
                let meta = item.metadata().map_err(|err| err.to_string())?;
                entries.push(FsEntry {
                    name: item.file_name().to_string_lossy().to_string(),
                    path: item.path().to_string_lossy().to_string(),
                    kind: if meta.is_dir() { "dir" } else { "file" }.to_string(),
                    bytes: if meta.is_file() {
                        Some(meta.len())
                    } else {
                        None
                    },
                });
            }
            Ok(FsResponse {
                status: "listed".to_string(),
                path: Some(path.to_string_lossy().to_string()),
                target: None,
                content: None,
                entries: Some(entries),
                bytes: None,
            })
        }
        "read" => {
            let path = required_path(&req)?;
            let bytes = fs::read(&path).map_err(|err| err.to_string())?;
            let text = if req
                .encoding
                .as_deref()
                .unwrap_or("utf-8")
                .eq_ignore_ascii_case("base64")
            {
                base64_encode(&bytes)
            } else {
                String::from_utf8_lossy(&bytes).to_string()
            };
            Ok(FsResponse {
                status: "read".to_string(),
                path: Some(path.to_string_lossy().to_string()),
                target: None,
                content: Some(text),
                entries: None,
                bytes: Some(bytes.len()),
            })
        }
        "write" => {
            let path = required_path(&req)?;
            if path.exists() && !req.overwrite.unwrap_or(true) {
                return Err("target exists and overwrite=false".to_string());
            }
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|err| err.to_string())?;
            }
            let content = req.content.unwrap_or_default();
            fs::write(&path, content.as_bytes()).map_err(|err| err.to_string())?;
            Ok(simple_fs("written", &path, None, Some(content.len())))
        }
        "mkdir" => {
            let path = required_path(&req)?;
            fs::create_dir_all(&path).map_err(|err| err.to_string())?;
            Ok(simple_fs("created", &path, None, None))
        }
        "copy" => {
            let path = required_path(&req)?;
            let target = req
                .target
                .as_deref()
                .filter(|p| !p.trim().is_empty())
                .map(PathBuf::from)
                .ok_or_else(|| "target is required".to_string())?;
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent).map_err(|err| err.to_string())?;
            }
            let bytes = copy_path(&path, &target, req.overwrite.unwrap_or(true))?;
            Ok(simple_fs("copied", &path, Some(&target), Some(bytes)))
        }
        "move" | "rename" => {
            let path = required_path(&req)?;
            let target = req
                .target
                .as_deref()
                .filter(|p| !p.trim().is_empty())
                .map(PathBuf::from)
                .ok_or_else(|| "target is required".to_string())?;
            if target.exists() && !req.overwrite.unwrap_or(false) {
                return Err("target exists and overwrite=false".to_string());
            }
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent).map_err(|err| err.to_string())?;
            }
            if target.exists() {
                remove_path(&target, true)?;
            }
            fs::rename(&path, &target).map_err(|err| err.to_string())?;
            Ok(simple_fs("moved", &path, Some(&target), None))
        }
        "delete" | "remove" => {
            let path = required_path(&req)?;
            remove_path(&path, req.recursive.unwrap_or(false))?;
            Ok(simple_fs("deleted", &path, None, None))
        }
        _ => Err(format!("unknown fs action: {action}")),
    }
}

fn simple_fs(status: &str, path: &Path, target: Option<&Path>, bytes: Option<usize>) -> FsResponse {
    FsResponse {
        status: status.to_string(),
        path: Some(path.to_string_lossy().to_string()),
        target: target.map(|p| p.to_string_lossy().to_string()),
        content: None,
        entries: None,
        bytes,
    }
}

fn copy_path(source: &Path, target: &Path, overwrite: bool) -> Result<usize, String> {
    if source.is_dir() {
        fs::create_dir_all(target).map_err(|err| err.to_string())?;
        let mut total = 0usize;
        for item in fs::read_dir(source).map_err(|err| err.to_string())? {
            let item = item.map_err(|err| err.to_string())?;
            total += copy_path(&item.path(), &target.join(item.file_name()), overwrite)?;
        }
        return Ok(total);
    }
    if target.exists() && !overwrite {
        return Err(format!("target exists: {}", target.display()));
    }
    fs::copy(source, target)
        .map(|n| n as usize)
        .map_err(|err| err.to_string())
}

fn remove_path(path: &Path, recursive: bool) -> Result<(), String> {
    if path.is_dir() {
        if recursive {
            fs::remove_dir_all(path).map_err(|err| err.to_string())
        } else {
            fs::remove_dir(path).map_err(|err| err.to_string())
        }
    } else {
        fs::remove_file(path).map_err(|err| err.to_string())
    }
}

fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    let mut i = 0;
    while i < bytes.len() {
        let b0 = bytes[i];
        let b1 = if i + 1 < bytes.len() { bytes[i + 1] } else { 0 };
        let b2 = if i + 2 < bytes.len() { bytes[i + 2] } else { 0 };
        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[(((b0 & 0b0000_0011) << 4) | (b1 >> 4)) as usize] as char);
        if i + 1 < bytes.len() {
            out.push(TABLE[(((b1 & 0b0000_1111) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if i + 2 < bytes.len() {
            out.push(TABLE[(b2 & 0b0011_1111) as usize] as char);
        } else {
            out.push('=');
        }
        i += 3;
    }
    out
}
