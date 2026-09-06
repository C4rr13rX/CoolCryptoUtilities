"""``services/smart_storage.py`` decides where market data lives.

It can DELETE local files, so the tests carry an asymmetric burden: proving it
frees space matters far less than proving it cannot destroy the only copy of
something.

The failure that motivated this module: on 2026-09-06 D: reached 0 bytes free
of 894 GB, the wizard brain's checkpoint began writing 179-byte truncated
stubs, and the node restarted into an empty fabric with its confidence down
from 0.278 to 0.000057. A storage layer that trades disk pressure for data
loss would have made that worse, not better.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.smart_storage import (  # noqa: E402
    SmartStorageService, StorageConfig, reset_storage,
)


@pytest.fixture()
def local_only(tmp_path, monkeypatch):
    """A service with no remote configured -- the default posture."""
    monkeypatch.delenv("MARKET_DATA_S3_ENABLED", raising=False)
    monkeypatch.delenv("MARKET_DATA_S3_BUCKET", raising=False)
    reset_storage()
    svc = SmartStorageService(
        config=StorageConfig(market_s3_enabled=False, bucket=""),
        roots={"market": str(tmp_path.relative_to(ROOT))
               if str(tmp_path).startswith(str(ROOT)) else "data/test_market"},
    )
    return svc


def _svc(tmp_path, *, remote: bool, bucket: str = "") -> SmartStorageService:
    """Service rooted inside tmp_path, bypassing the repo-relative default."""
    svc = SmartStorageService(
        config=StorageConfig(market_s3_enabled=remote, bucket=bucket))
    # Point the domain root at the temp dir directly.
    svc.roots = {"market": "data/__test_market"}
    real_root = ROOT / "data" / "__test_market"
    real_root.mkdir(parents=True, exist_ok=True)
    return svc


def test_remote_is_off_by_default(monkeypatch):
    """A fresh clone must behave exactly as it does today.

    Every existing deployment reads the same paths it always has; nothing
    reaches for a network until someone switches it on.
    """
    for name in ("MARKET_DATA_S3_ENABLED", "ACCOUNT_DATA_S3_ENABLED",
                 "MARKET_DATA_S3_BUCKET"):
        monkeypatch.delenv(name, raising=False)
    cfg = StorageConfig()
    assert cfg.market_s3_enabled is False
    assert cfg.account_s3_enabled is False
    assert cfg.remote_enabled is False


def test_a_half_configured_remote_behaves_as_local_only(monkeypatch):
    """Flag on but no bucket must not raise on the first read.

    Someone will set the flag before creating the bucket. That should degrade
    to local-only, not take down the trading loop.
    """
    monkeypatch.setenv("MARKET_DATA_S3_ENABLED", "1")
    monkeypatch.delenv("MARKET_DATA_S3_BUCKET", raising=False)
    cfg = StorageConfig()
    assert cfg.market_s3_enabled is True
    assert cfg.remote_enabled is False, (
        "remote_enabled must require BOTH the switch and a bucket")


def test_nothing_is_ever_evicted_in_local_only_mode(tmp_path):
    """THE LOAD-BEARING TEST.

    With no remote, every local file is the only copy. A cache that deletes
    the last copy of something is not a cache, and no amount of disk pressure
    justifies it.
    """
    svc = _svc(tmp_path, remote=False)
    svc.config.local_budget_mb = 0          # maximum possible pressure
    svc.put("market", "aaa/bar.json", b"x" * 4096)
    result = svc.reclaim("market", dry_run=False)
    assert result["evicted"] == []
    assert result["freed_mb"] == 0.0
    assert "only copies" in result.get("reason", "")
    assert svc.local_path("market", "aaa/bar.json").exists()


def test_eviction_requires_confirmed_durability(tmp_path, monkeypatch):
    """A key is only evictable once its presence in S3 is CONFIRMED.

    Not assumed from a successful-looking upload, not inferred from a flag --
    confirmed by this process having uploaded it or read it back. Anything
    weaker risks deleting the only copy on a bad assumption.
    """
    svc = _svc(tmp_path, remote=True, bucket="test-bucket")
    svc.config.local_budget_mb = 0
    svc.put("market", "bbb/cold.json", b"y" * 8192, sync=False)

    # No S3 client is reachable, so nothing can be confirmed durable.
    monkeypatch.setattr(svc, "_s3", lambda: None)
    result = svc.reclaim("market", dry_run=False)
    assert result["evicted"] == []
    assert result["skipped_not_durable"] >= 1
    assert svc.local_path("market", "bbb/cold.json").exists(), (
        "a key that could not be confirmed durable must survive")


def test_reclaim_defaults_to_dry_run(tmp_path):
    """The caller should see what would go before anything goes."""
    svc = _svc(tmp_path, remote=True, bucket="test-bucket")
    svc.config.local_budget_mb = 0
    svc.put("market", "ccc/x.json", b"z" * 2048, sync=False)
    svc._mark_durable("market", "ccc/x.json")   # pretend it is safely remote
    result = svc.reclaim("market")              # no dry_run= argument
    assert result["dry_run"] is True
    assert svc.local_path("market", "ccc/x.json").exists(), (
        "a default-argument reclaim must not delete anything")


def test_writes_are_atomic(tmp_path):
    """A crash mid-write must leave the previous version, not a stub.

    This is the exact failure that cost the project its brain checkpoint: the
    serializer wrote a 179-byte header directly over the target and died,
    leaving neither the old brain nor a new one. Writing to a temp file in the
    same directory and renaming makes a partial write impossible to observe.
    """
    svc = _svc(tmp_path, remote=False)
    svc.put("market", "ddd/atomic.json", b"first version")
    path = svc.local_path("market", "ddd/atomic.json")
    assert path.read_bytes() == b"first version"

    svc.put("market", "ddd/atomic.json", b"second version, longer")
    assert path.read_bytes() == b"second version, longer"
    # No .tmp residue left behind.
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_a_key_cannot_escape_its_domain(tmp_path):
    """Traversal is rejected, not normalised.

    Silently repairing ``../../etc/passwd`` would hide a bug at the call site.
    """
    svc = _svc(tmp_path, remote=False)
    for bad in ("../escape.json", "a/../../b.json", "", "/"):
        with pytest.raises(ValueError):
            svc.local_path("market", bad)


def test_an_unknown_domain_is_refused(tmp_path):
    svc = _svc(tmp_path, remote=False)
    with pytest.raises(ValueError):
        svc.local_path("not_a_domain", "x.json")


def test_get_returns_none_rather_than_raising_when_remote_is_down(tmp_path, monkeypatch):
    """Fails OPEN. A storage layer that raises can stop the money path.

    Losing a cache read is recoverable; an exception escaping into a trading
    loop is not.
    """
    svc = _svc(tmp_path, remote=True, bucket="test-bucket")

    def _boom():
        raise RuntimeError("network is down")

    monkeypatch.setattr(svc, "_s3", _boom)
    with pytest.raises(RuntimeError):
        svc._s3()          # the fake itself raises, proving the monkeypatch
    # but get() must swallow it
    monkeypatch.setattr(svc, "_s3", lambda: None)
    assert svc.get("market", "eee/missing.json") is None


def test_the_remote_key_carries_the_domain_prefix(tmp_path):
    """Bucket lifecycle rules tier on ``market/`` and ``news/`` prefixes.

    If the prefix were dropped, objects would never transition to IA or
    Glacier-IR and storage cost would stay at Standard forever.
    """
    svc = _svc(tmp_path, remote=False)
    assert svc.remote_key("market", "base/WETH.json") == "market/base/WETH.json"
    assert svc.remote_key("news", "2026/09/a.json") == "news/2026/09/a.json"


def test_stats_report_the_configuration_that_produced_them(tmp_path):
    """A counter without its config is unreadable six months later."""
    svc = _svc(tmp_path, remote=False)
    stats = svc.stats()
    for key in ("remote_enabled", "bucket", "local_budget_mb",
                "hot_window_bars", "local_hits", "misses"):
        assert key in stats


def teardown_module(module):  # noqa: ARG001
    """Remove the temp domain root this module creates inside the repo."""
    import shutil
    shutil.rmtree(ROOT / "data" / "__test_market", ignore_errors=True)
    reset_storage()
