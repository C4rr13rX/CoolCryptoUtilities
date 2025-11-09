from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from filter_scams import FilterScamTokens
from services.wallet_logger import wallet_log

Spec = str


def _normalize_chain(value: str) -> str:
    return str(value or "").strip().lower()


def _normalize_addr(value: str) -> str:
    val = str(value or "").strip().lower()
    if val.startswith("0x"):
        return val
    if val:
        return "0x" + val
    return val


def _spec(chain: str, token: str) -> Spec:
    return f"{_normalize_chain(chain)}:{_normalize_addr(token)}"


class TokenSafetyRegistry:
    """
    Caches scam-filter verdicts (GoPlus via FilterScamTokens) so wallet flows can
    reuse the results without repeatedly hitting the remote API.
    """

    _CRITICAL_REASONS = {"is_honeypot", "cannot_sell_all"}
    _HIGH_REASONS = {"is_blacklisted", "transfer_pausable"}
    _MEDIUM_REASONS = {"high_tax", "personal_slippage_modifiable"}

    def __init__(
        self,
        *,
        ttl_sec: Optional[int] = None,
        cache_path: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        self.ttl_sec = int(ttl_sec if ttl_sec is not None else int(os.getenv("TOKEN_SAFETY_TTL", "3600") or "3600"))
        base_path = cache_path or os.getenv("TOKEN_SAFETY_CACHE", "storage/token_safety.json")
        self.cache_path = Path(base_path).expanduser()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[Spec, Dict[str, object]] = self._load_cache()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ disk io
    def _load_cache(self) -> Dict[Spec, Dict[str, object]]:
        if not self.cache_path.exists():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text())
            if isinstance(payload, dict):
                normalized: Dict[Spec, Dict[str, object]] = {}
                for spec, data in payload.items():
                    entry = dict(data)
                    if "severity" not in entry:
                        entry["severity"] = self.classify_reasons(entry.get("reasons") or [])
                    normalized[str(spec)] = entry
                return normalized
        except Exception:
            pass
        return {}

    def _save_cache(self) -> None:
        try:
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache, indent=2, sort_keys=True))
            tmp.replace(self.cache_path)
        except Exception:
            pass

    # ------------------------------------------------------------------ helpers
    def _needs_refresh(self, spec: Spec) -> bool:
        if self.ttl_sec <= 0:
            return True
        entry = self._cache.get(spec)
        if not entry:
            return True
        ts = float(entry.get("ts") or 0.0)
        return (time.time() - ts) > self.ttl_sec

    @staticmethod
    def classify_reasons(reasons: Sequence[str]) -> str:
        lowered = {str(reason).strip().lower() for reason in (reasons or []) if str(reason).strip()}
        if not lowered:
            return "info"
        if lowered & TokenSafetyRegistry._CRITICAL_REASONS:
            return "critical"
        if lowered & TokenSafetyRegistry._HIGH_REASONS:
            return "high"
        if lowered & TokenSafetyRegistry._MEDIUM_REASONS:
            return "medium"
        return "low"

    def _cache_result(self, spec: Spec, status: str, reasons: Optional[Sequence[str]]) -> None:
        severity = self.classify_reasons(reasons or [])
        self._cache[spec] = {
            "status": status,
            "reasons": list(reasons or []),
            "ts": time.time(),
            "severity": severity,
        }

    def _refresh_specs(self, specs: Sequence[Spec]) -> None:
        if not specs:
            return
        try:
            filt = FilterScamTokens()
        except Exception as exc:
            wallet_log("token_safety.fallback", reason=str(exc))
            return

        try:
            result = filt.filter(specs)  # type: ignore[arg-type]
        except Exception as exc:
            wallet_log("token_safety.filter_error", reason=str(exc))
            return

        allowed = {str(token).lower() for token in (result.tokens or [])}
        flagged = getattr(result, "reasons", {}) or {}
        with self._lock:
            for spec in specs:
                spec_l = spec.lower()
                if spec_l in allowed:
                    self._cache_result(spec_l, "ok", [])
                else:
                    reasons = flagged.get(spec_l.split(":", 1)[-1], [])
                    self._cache_result(spec_l, "flagged", reasons)
            self._save_cache()

    # ------------------------------------------------------------------ public
    def filter_pairs(self, pairs: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
        normalized: List[Tuple[str, str, Optional[Spec]]] = []
        to_refresh: List[Spec] = []
        for chain, token in pairs:
            tok = str(token or "").strip()
            if not tok:
                continue
            lower = tok.lower()
            if not lower.startswith("0x"):
                normalized.append((chain, tok, None))
                continue
            spec = _spec(chain, tok)
            normalized.append((chain, tok, spec))
            if self._needs_refresh(spec):
                to_refresh.append(spec)

        if to_refresh:
            self._refresh_specs(to_refresh)

        survivors: List[Tuple[str, str]] = []
        for chain, token, spec in normalized:
            if not spec:
                survivors.append((chain, token))
                continue
            entry = self._cache.get(spec)
            if entry and entry.get("status") == "flagged":
                wallet_log(
                    "token_safety.filtered",
                    chain=chain,
                    token=token,
                    reasons=entry.get("reasons"),
                )
                continue
            survivors.append((chain, token))
        return survivors

    def ensure_safe(self, chain: str, token: str) -> None:
        tok = str(token or "").strip()
        if not tok or not tok.lower().startswith("0x"):
            return
        spec = _spec(chain, tok)
        if self._needs_refresh(spec):
            self._refresh_specs([spec])
        entry = self._cache.get(spec)
        if entry and entry.get("status") == "flagged":
            reasons = ", ".join(entry.get("reasons", [])) or "flagged"
            raise RuntimeError(f"{chain}:{token} blocked by scam filter ({reasons})")

    def verdict_for(self, chain: str, token: str) -> Dict[str, object]:
        tok = _normalize_addr(token)
        spec = _spec(chain, tok)
        entry = self._cache.get(spec)
        if not entry:
            return {"status": "unknown", "severity": "unknown", "reasons": []}
        return {
            "status": entry.get("status", "unknown"),
            "severity": entry.get("severity", "info"),
            "reasons": list(entry.get("reasons") or []),
            "ts": entry.get("ts"),
        }

    def report(self) -> Dict[str, object]:
        """Expose cache contents for debugging/telemetry."""
        return {
            "entries": len(self._cache),
            "flagged": sum(1 for v in self._cache.values() if v.get("status") == "flagged"),
        }

    def reasons_for(self, chain: str, token: str) -> List[str]:
        """
        Surface cached reasons for a flagged token so higher layers can
        display meaningful errors without re-querying the remote API.
        Returns an empty list when the token is unknown or currently allowed.
        """
        spec = _spec(chain, token)
        entry = self._cache.get(spec)
        if not entry:
            return []
        reasons = entry.get("reasons") or []
        return list(reasons)


def filter_token_pairs(
    pairs: Iterable[Tuple[str, str]],
    *,
    registry: Optional[TokenSafetyRegistry] = None,
) -> List[Tuple[str, str]]:
    if registry is None:
        registry = TokenSafetyRegistry()
    return registry.filter_pairs(pairs)


def enforce_token_safety(
    chain: str,
    token: str,
    registry: Optional[TokenSafetyRegistry],
) -> None:
    """
    Helper used by wallet actions to prevent interacting with flagged tokens.
    - No-op when registry is disabled or token is not a hex address.
    - Converts registry RuntimeError into ValueError for user-friendly surfaces.
    """
    if registry is None:
        return
    tok = str(token or "").strip()
    if not tok or not tok.lower().startswith("0x"):
        return
    try:
        registry.ensure_safe(chain, tok)
    except RuntimeError as exc:
        raise ValueError(f"{chain}:{tok} blocked by token safety registry ({exc})") from exc
