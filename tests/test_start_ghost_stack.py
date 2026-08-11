from scripts import start_ghost_stack


def test_safe_environment_forces_all_live_interlocks_off(monkeypatch):
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "1")
    monkeypatch.setenv("EXECUTE_LIVE_TRADES", "1")
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "0")
    monkeypatch.setenv("AUTO_PROMOTE_LIVE", "1")
    env = start_ghost_stack.safe_environment()
    assert env["ENABLE_LIVE_TRADING"] == "0"
    assert env["EXECUTE_LIVE_TRADES"] == "0"
    assert env["LIVE_TRADES_DRY_RUN"] == "1"
    assert env["AUTO_PROMOTE_LIVE"] == "0"
    assert env["PRODUCTION_AUTO_DISABLED"] == "1"
    assert env["PAIR_INDEX_MAX_AGE_DAYS"] == "30"


def test_fresh_heartbeat_requires_running_and_recent(tmp_path, monkeypatch):
    monkeypatch.setattr(start_ghost_stack, "LOGS", tmp_path)
    path = tmp_path / "production_manager_heartbeat.json"
    path.write_text('{"timestamp": 0, "status": "running"}', encoding="utf-8")
    assert not start_ghost_stack.fresh_production_heartbeat()
