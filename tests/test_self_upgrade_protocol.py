import sys
from pathlib import Path


def test_self_upgrade_guidance_protocol():
    from tools import c0d3r_cli

    text = c0d3r_cli._self_upgrade_guidance(
        python_exec=sys.executable,
        project_root=Path(c0d3r_cli.PROJECT_ROOT),
    )
    lowered = text.lower()
    assert "at least 10" in lowered
    assert "feedback" in lowered
    assert "feedback control" in lowered
    assert "rigid" in lowered
    assert "deterministic" in lowered
