"""A profitability lint is only useful if its findings are believable.

The first version of this checker reported 15 findings on the money path and
ALL FIFTEEN were false. A lint that is wrong every time gets muted, and a
muted lint is worse than none because it looks like coverage. So half these
tests are about what it must NOT flag.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from services.profit_logic_audit import audit, audit_file


def _write(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    return path


class TestCatchesRealBugs:
    """Every case here shipped in this repo and cost money."""

    def test_a_return_compared_to_zero(self, tmp_path):
        """Admits every trade between zero and the round-trip cost."""
        findings = audit_file(_write(tmp_path, """
def gate(gross_return):
    if gross_return > 0:
        return "enter"
"""))
        assert any(f.rule == "profit_compared_to_zero" for f in findings)

    def test_a_rate_compared_against_dollars(self, tmp_path):
        """This shape barred every small-clip trade until it was found."""
        findings = audit_file(_write(tmp_path, """
def gate(expectancy_usd, fee_rate):
    if expectancy_usd > fee_rate:
        return "enter"
"""))
        assert any(f.rule == "units_mismatch" for f in findings)

    def test_a_cost_billed_twice(self, tmp_path):
        """Charging both legs to a forward decision made 209 of 400 hold-negative."""
        findings = audit_file(_write(tmp_path, """
def cost(fee_cost, gas_usd):
    return fee_cost * 2 + gas_usd
"""))
        assert any(f.rule == "cost_doubled" for f in findings)

    def test_an_unmeasurable_cost_defaulted_to_zero(self, tmp_path):
        """Makes every trade look free."""
        findings = audit_file(_write(tmp_path, """
def size(fee_cost):
    if fee_cost is None:
        fee_cost = 0
    return fee_cost
"""))
        assert any(f.rule == "unmeasurable_as_permission" for f in findings)


class TestDoesNotCryWolf:
    """The false positives that made version one useless."""

    def test_a_net_figure_may_be_compared_to_zero(self, tmp_path):
        """economic_profit is ALREADY net of fees; zero is the right bar."""
        findings = audit_file(_write(tmp_path, """
def book(economic_profit):
    if economic_profit > 0:
        return "checkpoint"
"""))
        assert findings == []

    def test_a_divide_guard_is_not_a_profit_gate(self, tmp_path):
        """`gross_loss > 0` guards a division, it does not admit a trade."""
        findings = audit_file(_write(tmp_path, """
def factor(wins, gross_loss):
    return (wins / gross_loss) if gross_loss > 0 else 0.0
"""))
        assert findings == []

    def test_a_sign_test_is_not_a_profit_gate(self, tmp_path):
        """`expected_return > 0` in a boolean names a DIRECTION."""
        findings = audit_file(_write(tmp_path, """
def fading(expected_return, accel):
    return (expected_return > 0 and accel < 0) or (expected_return < 0 and accel > 0)
"""))
        assert findings == []

    def test_a_running_total_is_not_a_return(self, tmp_path):
        findings = audit_file(_write(tmp_path, """
def drawdown(peak_pnl, total_pnl):
    if peak_pnl > 0:
        return peak_pnl - total_pnl
    return 0.0
"""))
        assert findings == []

    def test_a_token_quantity_is_not_a_return(self, tmp_path):
        findings = audit_file(_write(tmp_path, """
def clamp(expected_base, max_adverse):
    if expected_base > 0:
        return expected_base * (1.0 - max_adverse)
    return None
"""))
        assert findings == []


class TestRobustness:
    def test_a_file_outside_the_repo_does_not_crash(self, tmp_path):
        """Every test fixture lives outside the repo.

        relative_to RAISES rather than returning an absolute path, which made
        the module impossible to test on planted bugs.
        """
        assert audit_file(_write(tmp_path, "x = 1\n")) == []

    def test_an_unparseable_file_yields_nothing(self, tmp_path):
        path = tmp_path / "broken.py"
        path.write_text("def (:::\n", encoding="utf-8")
        assert audit_file(path) == []

    def test_a_missing_file_yields_nothing(self, tmp_path):
        assert audit_file(tmp_path / "absent.py") == []


class TestVerdict:
    def test_the_real_money_path_is_clean(self):
        """A regression alarm: if this fails, something on the money path
        acquired one of the shapes that has cost money here before."""
        result = audit()
        assert result["counts"]["loses_money"] == 0, [
            f"{f['file']}:{f['line']} {f['detail']}"
            for f in result["findings"] if f["severity"] == "loses-money"]

    def test_the_verdict_never_claims_profitability(self, tmp_path):
        """Reading cannot answer whether an edge exists."""
        result = audit([_write(tmp_path, "x = 1\n")])
        assert "not proof of an edge" in result["summary"]
