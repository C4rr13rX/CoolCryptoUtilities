from __future__ import annotations

import os

from services.system_profile import SystemProfile, detect_system_profile


def test_detect_system_profile_defaults(monkeypatch):
    profile = detect_system_profile()
    assert isinstance(profile, SystemProfile)
    assert profile.cpu_count >= 1
    assert profile.total_memory_gb > 0
    assert profile.max_threads >= 1


def test_detect_system_profile_env_override(monkeypatch):
    monkeypatch.setenv("SYSTEM_MEMORY_GB", "10")
    profile = detect_system_profile()
    assert profile.memory_pressure is True
    assert profile.is_low_power is True
    monkeypatch.delenv("SYSTEM_MEMORY_GB", raising=False)
