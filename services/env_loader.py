from __future__ import annotations
import os, sys, json, time
from pathlib import Path
from typing import List, Optional
from dotenv_fallback import load_dotenv, dotenv_values, find_dotenv

_ALLOW_FLAG = "ALLOW_DOTENV_LOADING"
_SECURE_ENV_FLAG = "SECURE_ENV_HYDRATED"

class EnvLoader:
    """Robust .env loader you can import anywhere."""
    @staticmethod
    def load() -> None:
        # ensure the fallback shim is allowed to touch .env files
        os.environ.setdefault(_ALLOW_FLAG, "1")
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

        # 3) hydrate from secure settings (Postgres-backed secrets) if available.
        #    Do this once per process to avoid repeated DB hits.
        if os.environ.get(_SECURE_ENV_FLAG) != "1":
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
