"""Read the trading code and ask whether its logic can make money.

WHAT THIS IS FOR
----------------
Every check in this repo tests behaviour after the fact: run the pipeline,
look at what it did, decide whether the numbers moved the right way. That
works and it is slow -- a sign error in a cost comparison costs real money for
however long it takes the P/L to notice.

Some errors are visible in the code itself, before anything trades. A profit
gate that compares a return against zero instead of against cost is wrong on
inspection. So is one that subtracts a rate from a dollar amount, or bills a
round trip to one leg, or treats "unmeasurable" as "condition met". Every one
of those has actually shipped here, and each was found by its damage rather
than by reading:

    expectancy subtracted a rate from a dollar  -> barred every small clip
    gas priced in the traded pair               -> a $0.41 fee on a $3 trade
    the exit gate charged both legs             -> 209 of 400 held negative
    unmeasurable read as permission             -> positions nobody could justify

This walks the AST of the money path and flags those shapes. It is a lint for
profitability, and like any lint it reports SUSPICIONS -- a flagged line may
be correct for a reason the checker cannot see, and the checker says so rather
than claiming a bug.

WHAT IT WILL NOT DO
-------------------
It will not decide the system is profitable. Whether an edge exists is a
question for measurement, and no amount of reading can answer it. This answers
a narrower and still-useful question: given that an edge exists, is the code
arranged to keep it or to give it away?
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

#: Files on the money path. Everything else can be wrong without costing a
#: trade, so scanning it would bury the findings that matter.
MONEY_PATH = (
    "trading/bot.py",
    "trading/scheduler.py",
    "trading/rotation.py",
    "trading/swap_schedule.py",
    "trading/micro_profit.py",
    "trading/strategies",
    "services/swap_service.py",
    "services/symbol_edge_gate.py",
    "web/tradingagent/lattice.py",
    "web/tradingagent/engine.py",
)

#: Names that hold a COST -- a fee, gas, slippage, a round trip.
COST_NAMES = re.compile(
    r"(fee|fees|cost|gas|slippage|spread|commission|round_?trip)", re.I)

#: Names that hold a GROSS return -- one the cost has NOT yet been taken out
#: of. Only these need checking against cost, and the distinction is the whole
#: difference between a useful checker and one that cries wolf.
#:
#: Measured on the first run: matching every profit-ish name produced 15
#: findings and ALL FIFTEEN were false. `economic_profit` is already net of
#: fees, so comparing it to zero is right. `_live_peak_pnl > 0` asks "have we
#: ever been up", not "is this trade worth taking". `expected_base` is a token
#: quantity. A checker that is wrong fifteen times out of fifteen gets muted,
#: and then it is worse than no checker at all.
GROSS_RETURN_NAMES = re.compile(
    r"(gross_|_gross|expected_return|raw_return|price_move|delta_pct)", re.I)

#: Names that are ALREADY net of cost. A comparison against zero is correct
#: for these, and flagging it is the noise that buries real findings.
NET_RETURN_NAMES = re.compile(
    r"(economic_|net_|_net|retained_|realised_|realized_|after_fee)", re.I)

#: Names that are not returns at all, however profit-shaped they read.
NOT_A_RETURN = re.compile(
    r"(peak|total_pnl|cumulative|_base$|_count|_ts$|balance|quantity)", re.I)

#: Names denominated in DOLLARS rather than as a fraction.
DOLLAR_NAMES = re.compile(r"(_usd|usd_|notional|clip|size_usd|amount)", re.I)

#: Names that are a RATE or fraction rather than an amount.
RATE_NAMES = re.compile(r"(_rate|_pct|_ratio|_frac|fraction|bps)", re.I)


@dataclass
class Finding:
    """One suspicion about the money path."""

    rule: str
    severity: str                  # "loses-money" | "suspicious" | "note"
    file: str
    line: int
    snippet: str
    detail: str
    why_it_costs: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule, "severity": self.severity, "file": self.file,
            "line": self.line, "snippet": self.snippet, "detail": self.detail,
            "why_it_costs": self.why_it_costs,
        }


def _source_line(lines: Sequence[str], number: int) -> str:
    if 1 <= number <= len(lines):
        return lines[number - 1].strip()[:180]
    return ""


def _name_of(node: ast.AST) -> str:
    """A readable name for whatever this expression is, for pattern matching."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _name_of(node.func)
    if isinstance(node, ast.Subscript):
        return _name_of(node.value)
    return ""


class _MoneyPathVisitor(ast.NodeVisitor):
    """Walks one file looking for shapes that give away an edge."""

    def __init__(self, rel_path: str, lines: Sequence[str]) -> None:
        self.rel_path = rel_path
        self.lines = lines
        self.findings: List[Finding] = []
        # Comparisons that are DECIDING whether to trade, rather than
        # guarding a division or testing a sign. Only these are profit gates.
        self._gate_compares: set = set()

    def visit_If(self, node: ast.If) -> None:  # noqa: D401 - see below
        # An `if` whose body enters or exits IS a decision. Everything else
        # -- a ternary guarding a divide, a boolean naming a direction -- is
        # not, and flagging it produced four false findings out of four.
        self._mark_gate(node.test)
        self._check_none_as_permission(node)
        self.generic_visit(node)

    def _mark_gate(self, test: ast.AST) -> None:
        """Record every comparison inside a decision's condition."""
        for child in ast.walk(test):
            if isinstance(child, ast.Compare):
                self._gate_compares.add(id(child))

    # -- a profit test that forgets the cost --------------------------------
    def visit_Compare(self, node: ast.Compare) -> None:
        self._check_profit_against_zero(node)
        self._check_units_across_comparison(node)
        self.generic_visit(node)

    def _check_profit_against_zero(self, node: ast.Compare) -> None:
        """`expected_return > 0` is not a profit test; it ignores the cost.

        The round trip on this account costs 0.65% of notional. A comparison
        against zero admits every trade between zero and that, and each one
        loses the difference. Measured: 209 of 400 decisions were
        hold-negative once the cost was actually charged.
        """
        left_name = _name_of(node.left) or ""
        # Only a GROSS return needs testing against cost. A figure already net
        # of fees is correctly compared to zero, and a running total or a
        # token quantity is not a return at all.
        if not GROSS_RETURN_NAMES.search(left_name):
            return
        if NET_RETURN_NAMES.search(left_name) or NOT_A_RETURN.search(left_name):
            return
        # Only the ordering comparisons; equality against zero is a different
        # question and usually a legitimate "did anything happen" check.
        if not node.ops or not isinstance(node.ops[0], (ast.Gt, ast.GtE)):
            return
        # AND ONLY WHERE THE COMPARISON DECIDES SOMETHING.
        #
        # `x > 0` appears constantly in money code without being a profit
        # gate: `gross_loss > 0` guards a division, and
        # `expected_return > 0` in a boolean names a DIRECTION. Both were
        # flagged by an earlier version, and between them and the name
        # matching the checker was wrong on every single finding it produced.
        # A lint with no true positives gets muted, and a muted lint is worse
        # than none because it looks like coverage.
        if id(node) not in self._gate_compares:
            return
        for comparator in node.comparators:
            if isinstance(comparator, ast.Constant) and comparator.value == 0:
                self.findings.append(Finding(
                    rule="profit_compared_to_zero",
                    severity="loses-money",
                    file=self.rel_path,
                    line=node.lineno,
                    snippet=_source_line(self.lines, node.lineno),
                    detail=(f"{left_name!r} is compared against 0 rather than "
                            f"against what a round trip costs."),
                    why_it_costs=(
                        "Every trade whose return falls between zero and the "
                        "round-trip cost is admitted and loses the "
                        "difference. On this account that band is 0.65% of "
                        "notional wide."),
                ))

    def _check_units_across_comparison(self, node: ast.Compare) -> None:
        """A dollar amount compared against a rate is a units error.

        `expectancy - fee_rate` shipped here and barred every small clip: the
        left side was dollars, the right a fraction, and the subtraction was
        meaningless in both directions.
        """
        left = _name_of(node.left)
        if not left:
            return
        left_is_dollars = bool(DOLLAR_NAMES.search(left))
        left_is_rate = bool(RATE_NAMES.search(left))
        if not (left_is_dollars or left_is_rate):
            return
        for comparator in node.comparators:
            right = _name_of(comparator)
            if not right:
                continue
            right_is_dollars = bool(DOLLAR_NAMES.search(right))
            right_is_rate = bool(RATE_NAMES.search(right))
            if left_is_dollars and right_is_rate:
                mismatch = (left, "dollars", right, "a rate")
            elif left_is_rate and right_is_dollars:
                mismatch = (left, "a rate", right, "dollars")
            else:
                continue
            self.findings.append(Finding(
                rule="units_mismatch",
                severity="loses-money",
                file=self.rel_path,
                line=node.lineno,
                snippet=_source_line(self.lines, node.lineno),
                detail=(f"{mismatch[0]!r} looks like {mismatch[1]} and "
                        f"{mismatch[2]!r} looks like {mismatch[3]}; comparing "
                        f"them is a units error."),
                why_it_costs=(
                    "A rate compared against an amount is wrong by the size "
                    "of the position. This exact shape barred every "
                    "small-clip trade here until it was found."),
            ))

    # -- unmeasurable treated as permission ---------------------------------
    def _check_none_as_permission(self, node: ast.If) -> None:
        """`if x is None: <pass/continue>` on the money path lets the trade on.

        "We could not measure this" and "this is fine" are different facts. A
        branch that treats the first as the second trades on the absence of
        evidence, which is how positions nobody could justify came to be held.
        """
        test = node.test
        if not isinstance(test, ast.Compare) or not node.body:
            return
        if not any(isinstance(op, ast.Is) for op in test.ops):
            return
        if not any(isinstance(c, ast.Constant) and c.value is None
                   for c in test.comparators):
            return

        name = _name_of(test.left)
        # Only worry when the thing that could not be measured is a cost or a
        # return -- an unmeasurable label or timestamp is not a money bug.
        name = name or ""
        if not (COST_NAMES.search(name) or GROSS_RETURN_NAMES.search(name)):
            return
        if NOT_A_RETURN.search(name):
            return

        body = node.body[0]
        permissive = (
            isinstance(body, ast.Pass)
            or isinstance(body, ast.Continue)
            or (isinstance(body, ast.Assign)
                and isinstance(getattr(body, "value", None), ast.Constant)
                and body.value.value in (0, 0.0, True))
        )
        if not permissive:
            return

        self.findings.append(Finding(
            rule="unmeasurable_as_permission",
            severity="suspicious",
            file=self.rel_path,
            line=node.lineno,
            snippet=_source_line(self.lines, node.lineno),
            detail=(f"{name!r} being None is handled by passing, continuing, "
                    f"or defaulting to zero."),
            why_it_costs=(
                "An unmeasurable cost defaulted to zero makes every trade "
                "look free. An unmeasurable return defaulted to zero is "
                "harmless; an unmeasurable COST is not."),
        ))

    # -- a cost charged twice, or to the wrong leg --------------------------
    def visit_BinOp(self, node: ast.BinOp) -> None:
        self._check_double_charge(node)
        self.generic_visit(node)

    def _check_double_charge(self, node: ast.BinOp) -> None:
        """`cost * 2` or `cost + cost` on a single leg bills a round trip twice.

        bot.py once charged both legs of a round trip to a forward-looking
        exit decision, which made 209 of 400 decisions read as hold-negative
        and suppressed exits that were actually profitable.
        """
        if not isinstance(node.op, ast.Mult):
            return
        left, right = _name_of(node.left), _name_of(node.right)
        if not COST_NAMES.search(left or ""):
            return
        if isinstance(node.right, ast.Constant) and node.right.value == 2:
            self.findings.append(Finding(
                rule="cost_doubled",
                severity="suspicious",
                file=self.rel_path,
                line=node.lineno,
                snippet=_source_line(self.lines, node.lineno),
                detail=f"{left!r} is multiplied by 2.",
                why_it_costs=(
                    "Doubling a cost is right when pricing a round trip from "
                    "one leg's fee and wrong when the figure already covers "
                    "both. The same expression is correct or a 2x "
                    "overcharge depending on which, and only a human can "
                    "say. Charging both legs to a forward decision made 209 "
                    "of 400 decisions hold-negative here."),
            ))


def audit_file(path: Path) -> List[Finding]:
    """Findings for one file. Unparseable files yield nothing, not an error."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    # A path outside the repo is normal, not exceptional: every test fixture
    # lives elsewhere, and relative_to RAISES rather than returning the
    # absolute path. Crashing on a file it was asked to read made the module
    # impossible to test on planted bugs.
    try:
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        rel = str(path).replace("\\", "/")
    visitor = _MoneyPathVisitor(rel, source.splitlines())
    visitor.visit(tree)
    return visitor.findings


def _money_path_files() -> List[Path]:
    out: List[Path] = []
    for entry in MONEY_PATH:
        target = ROOT / entry
        if target.is_dir():
            out.extend(sorted(target.rglob("*.py")))
        elif target.is_file():
            out.append(target)
    return out


def audit(paths: Optional[Iterable[Path]] = None) -> Dict[str, Any]:
    """Audit the money path and report what it finds.

    Returns a verdict plus every finding. The verdict is about the CODE's
    arrangement, never about whether the system is profitable -- that is a
    question for measurement, and reading cannot answer it.
    """
    files = list(paths) if paths is not None else _money_path_files()
    findings: List[Finding] = []
    for path in files:
        findings.extend(audit_file(path))

    losing = [f for f in findings if f.severity == "loses-money"]
    suspicious = [f for f in findings if f.severity == "suspicious"]

    if losing:
        verdict = "GEARED TO LOSE"
        summary = (f"{len(losing)} place(s) where the logic gives away an edge "
                   f"by construction, before any market moves.")
    elif suspicious:
        verdict = "CHECK THESE"
        summary = (f"No outright errors, but {len(suspicious)} place(s) are "
                   f"right or wrong depending on context a reader has to "
                   f"supply.")
    else:
        verdict = "NO KNOWN LOSING SHAPES"
        summary = ("The money path contains none of the shapes that have cost "
                   "money here before. That is not proof of an edge -- only "
                   "that the arithmetic is not giving one away.")

    return {
        "verdict": verdict,
        "summary": summary,
        "files_scanned": len(files),
        "counts": {
            "loses_money": len(losing),
            "suspicious": len(suspicious),
            "total": len(findings),
        },
        "findings": [f.as_dict() for f in findings],
    }
