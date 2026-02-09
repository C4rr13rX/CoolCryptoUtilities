from __future__ import annotations
import os, sys, json, time
from pathlib import Path
from typing import List, Optional
from dotenv_fallback import load_dotenv, dotenv_values, find_dotenv


def is_test_env() -> bool:
    if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("PYTEST_RUNNING"):
        return True
    argv = " ".join(sys.argv).lower()
    if "pytest" in argv or "py.test" in argv:
        return True
    return "test" in sys.argv

_ALLOW_FLAG = "ALLOW_DOTENV_LOADING"
_SECURE_ENV_FLAG = "SECURE_ENV_HYDRATED"


def _bool_env(value: str | None) -> bool:
    return (value or "0").lower() in {"1", "true", "yes", "on"}


def _postgres_connectable() -> tuple[bool, str]:
    try:
        import psycopg
    except Exception as exc:
        return False, f"psycopg unavailable: {exc}"
    try:
        timeout = int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "3") or "3")
    except Exception:
        timeout = 3
    try:
        conn = psycopg.connect(
            dbname=os.getenv("POSTGRES_DB", "coolcrypto"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            sslmode=os.getenv("POSTGRES_SSLMODE", "prefer"),
            connect_timeout=timeout,
        )
        conn.close()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _maybe_enable_sqlite_fallback() -> None:
    prefer_sqlite = _bool_env(os.getenv("DJANGO_PREFER_SQLITE_FALLBACK"))
    db_vendor_env = os.getenv("DJANGO_DB_VENDOR")
    if prefer_sqlite and not db_vendor_env:
        os.environ["DJANGO_DB_VENDOR"] = "sqlite"
        return

    db_vendor = (db_vendor_env or "postgres").lower()
    if db_vendor != "postgres":
        return

    if _bool_env(os.getenv("REQUIRE_POSTGRES") or os.getenv("STRICT_POSTGRES")):
        return

    allow_fallback = _bool_env(os.getenv("ALLOW_SQLITE_FALLBACK")) or prefer_sqlite
    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    local_host = host in {"127.0.0.1", "localhost"}
    if not (allow_fallback or local_host):
        return

    ok, error = _postgres_connectable()
    if ok:
        return
    os.environ["DJANGO_DB_VENDOR"] = "sqlite"
    os.environ["ALLOW_SQLITE_FALLBACK"] = "1"
    # Always silence fallback warnings to keep CLI output clean.


class EnvLoader:
    """Robust .env loader you can import anywhere."""
    @staticmethod
    def load() -> None:
        testing = is_test_env()
        # ensure the fallback shim is allowed to touch .env files
        os.environ.setdefault(_ALLOW_FLAG, "1")
        if testing:
            os.environ.setdefault("DJANGO_DB_VENDOR", "sqlite")
            os.environ.setdefault("TRADING_DB_VENDOR", "sqlite")
            os.environ.setdefault("ALLOW_SQLITE_FALLBACK", "1")
            os.environ.setdefault(_SECURE_ENV_FLAG, "1")
        repo_root = Path(__file__).resolve().parents[1]
        web_dir = repo_root / "web"
        for path in (repo_root, web_dir):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

        loaded = False

        # 1) try an auto-discovered .env in current working dir
        path = find_dotenv(usecwd=True)
        if path:
            try:
                load_dotenv(path, override=False)
                loaded = True
            except Exception:
                pass

        # 2) common fallbacks
        cands: List[Path] = []
        for p in (
            Path.cwd() / ".env",
            Path(sys.argv[0]).resolve().parent / ".env" if sys.argv and sys.argv[0] else None,
            Path(__file__).resolve().parent / ".env" if "__file__" in globals() else None,
            Path.home() / ".env",
        ):
            if p:
                cands.append(p)

        # 2a) load_dotenv on first readable candidate
        for p in cands:
            try:
                if p.is_file():
                    load_dotenv(p, override=False)
                    loaded = True
                    break
            except Exception:
                pass

        # 2b) last-resort: parse and set os.environ without load_dotenv
        for p in cands:
            try:
                if p.is_file():
                    for k, v in (dotenv_values(p) or {}).items():
                        os.environ.setdefault(k, v or "")
                    loaded = True
                    break
            except Exception:
                pass

        if not testing:
            _maybe_enable_sqlite_fallback()

        # 3) hydrate from secure settings (Postgres-backed secrets) if available.
        #    Do this once per process to avoid repeated DB hits.
        if not testing and os.environ.get(_SECURE_ENV_FLAG) != "1":
            try:
                from services.secure_settings import build_process_env
                env = build_process_env()
                env.update(_derive_stream_env(env))
                os.environ.update(env)
                os.environ[_SECURE_ENV_FLAG] = "1"
                _debug_env(env)
            except Exception:
                # Silent fail: fallback to plain .env-only mode
                pass

        _apply_default_env()


def _apply_default_env() -> None:
    """
    Set non-secret defaults so the pipeline has sensible baselines even when
    env files are minimal. Only applies when values are missing.
    """
    defaults = {
        "ALCHEMY_TIMEOUT_SEC": "10",
        "HTTP_TIMEOUT_SEC": "8",
        "TX_TIMEOUT_SEC": "120",
        "PORTFOLIO_REORG_SAFETY": "12",
        "MARKET_DATA_COINGECKO_IDS": "bitcoin,ethereum",
        "MARKET_DATA_TOP_N": "25",
        "MARKET_DATA_REFRESH_SEC": "1200",
        "MARKET_DATA_SNAPSHOT_PATH": "data/market_snapshots.json",
        "PRICE_UNISWAP": "1",
        "PRICE_TTL_SEC": "300",
        "PRICE_MAX_TOKENS": "25",
        "PRICE_WORKERS": "8",
        "CG_TIMEOUT": "6",
        "DS_TIMEOUT": "6",
        "FILTER_SCAMS": "1",
        "SCAM_STRICT": "1",
        "SCAM_MIN_LIQ_USD": "2000",
        "SCAM_MAX_TAX_PCT": "25",
        "SCAM_USE_GOPLUS": "1",
        "BLOCKSCOUT_INCLUDE_UNVERIFIED": "1",
        "REPRICE_COVALENT": "0",
        "AUTO_WRAP_NATIVE": "1",
        "APPROVE_MODE": "u",
        "GAS_BASE_MULT": "2.0",
        "LIVE_ALLOW_MICRO": "0",
        "LIVE_MICRO_MIN_CAPITAL_USD": "5",
        "LIVE_MICRO_NATIVE_BUFFER_USD": "1.5",
        "LIVE_MICRO_MIN_CLIP_USD": "2",
        "LIVE_MICRO_RISK_BUDGET": "0.2",
        "LIVE_MICRO_MAX_TRADE_SHARE": "0.08",
        "LIVE_MICRO_AUTO_PROMOTE": "0",
        "LIVE_FOCUS_CHAINS": "base,arbitrum,optimism,polygon",
        "SIMULATE_BRIDGE_TOPUP": "1",
        "ENABLE_BRIDGE_TOPUP": "0",
        "BRIDGE_FEE_USD": "1.5",
        "BRIDGE_FEE_RATIO": "0.001",
        "BRIDGE_MIN_PROFIT_USD": "1.0",
        "LOG_API_REQUESTS": "1",
        "C0D3R_SESSION_BUCKET": "c0dersessions",
        "C0D3R_SESSION_REGION": "us-east-1",
        "C0D3R_SESSION_PREFIX": "sessions",
        "C0D3R_SESSION_MODE": "auto",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    if not os.getenv("PYTHON_BIN"):
        repo_root = Path(__file__).resolve().parents[1]
        repo_venv = repo_root / ".venv"
        if repo_venv.exists():
            candidate = repo_venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if candidate.exists():
                os.environ["PYTHON_BIN"] = str(candidate)


def resolve_python_bin() -> str:
    candidate = os.getenv("PYTHON_BIN")
    if candidate:
        try:
            path = Path(candidate)
            if path.exists():
                return str(path)
        except Exception:
            pass
    repo_root = Path(__file__).resolve().parents[1]
    repo_venv = repo_root / ".venv"
    if repo_venv.exists():
        repo_bin = repo_venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if repo_bin.exists():
            return str(repo_bin)
    virtual_env = os.getenv("VIRTUAL_ENV")
    if virtual_env:
        venv_bin = Path(virtual_env) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if venv_bin.exists():
            return str(venv_bin)
    return sys.executable


def _derive_stream_env(env: dict) -> dict:
    """
    Build chain-specific RPC/WSS defaults from vault-stored Alchemy keys so
    market streams and on-chain listeners come up without manual env wiring.
    Falls back to public RPC/WSS if no key is present.
    """
    updates: dict = {}
    chain = (env.get("PRIMARY_CHAIN") or os.getenv("PRIMARY_CHAIN", "base")).lower()
    try:
        from balances import CHAIN_CONFIG
        cfg = CHAIN_CONFIG.get(chain)
        if not cfg:
            return updates
        env_alc = cfg.get("env_alchemy_url")
        api_key = (env.get("ALCHEMY_API_KEY") or os.getenv("ALCHEMY_API_KEY") or "").strip()
        slug = cfg.get("alchemy_slug")
        if api_key and slug:
            rpc = f"https://{slug}.g.alchemy.com/v2/{api_key}"
            wss = f"wss://{slug}.g.alchemy.com/v2/{api_key}"
            chain_prefix = chain.upper()
            updates.setdefault(f"{chain_prefix}_RPC_URL", rpc)
            updates.setdefault(f"{chain_prefix}_WSS_URL", wss)
            updates.setdefault("GLOBAL_RPC_URL", rpc)
            updates.setdefault("GLOBAL_WSS_URL", wss)
            if env_alc:
                updates.setdefault(env_alc, rpc)
        if not updates:
            # Public fallback if no Alchemy key/slug available
            public_rpcs = cfg.get("public_rpcs") or []
            if public_rpcs:
                rpc = str(public_rpcs[0]).strip()
                if rpc:
                    wss = rpc.replace("https://", "wss://").replace("http://", "ws://") if rpc.startswith("http") else ""
                    chain_prefix = chain.upper()
                    updates.setdefault(f"{chain_prefix}_RPC_URL", rpc)
                    if wss:
                        updates.setdefault(f"{chain_prefix}_WSS_URL", wss)
                    updates.setdefault("GLOBAL_RPC_URL", rpc)
                    if wss:
                        updates.setdefault("GLOBAL_WSS_URL", wss)
                    if env_alc:
                        updates.setdefault(env_alc, rpc)
    except Exception:
        return updates
    return updates


def _debug_env(env: dict) -> None:
    path = Path("logs/stream_debug.log")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    payload = {
        "ts": time.time(),
        "label": "env_load",
        "BASE_WSS_URL": env.get("BASE_WSS_URL") or env.get("GLOBAL_WSS_URL"),
        "GLOBAL_WSS_URL": env.get("GLOBAL_WSS_URL"),
        "MARKET_WS_TEMPLATE": env.get("MARKET_WS_TEMPLATE"),
        "MARKET_WS_SUBSCRIBE": env.get("MARKET_WS_SUBSCRIBE"),
    }
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except Exception:
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(str(payload) + "\n")
        except Exception:
            return
