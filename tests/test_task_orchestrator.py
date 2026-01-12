import json
from pathlib import Path


def test_ux_audit_report_template_exists():
    # Basic smoke to ensure report script/templates present
    assert Path("scripts/ux_audit_report.py").exists()

