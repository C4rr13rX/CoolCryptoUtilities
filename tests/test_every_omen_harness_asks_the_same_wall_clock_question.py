"""Nine omen harnesses, one horizon unit -- or nine reports nobody can compare.

THE DEFECT THIS PREVENTS. ``scripts/omen_experiment.py`` was taught in pass 114
to take a horizon in MINUTES and convert it with each corpus's own measured
cadence. It is ONE of nine harnesses that write to ``data/brain_experiments/``.
The other eight still took a bar COUNT and still defaulted to 12, so the fix
covered one report in nine: an ``h12`` from ``omen_agreement_census`` on a 60s
corpus was a 12-MINUTE question, an ``h12`` from ``omen_temporal_census`` on a
86400s corpus was a 12-DAY one, and both files wrote the same string. Measured
pass 113 over all 629 files in ``data/historical_ohlcv``: the cadence runs 60s
to 345600s, a 5760x spread, so the unit is not a detail.

WHAT IS ASSERTED, AND WHY IT GOES RED AGAINST THE OLD DEFAULT. Every check here
runs the harness as a subprocess and reads what it PRINTS or WRITES -- never
its source and never a comment. Against a harness carrying
``add_argument("--horizon", type=int, default=12)`` the flag ``--horizon-minutes``
does not exist, argparse exits with "unrecognized arguments", and every one of
these fails.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.omen_experiment import (  # noqa: E402
    DEFAULT_HORIZON_MINUTES, horizon_request, resolve_horizon, settle_horizon,
    validate_report_horizon,
)

#: The six harnesses this covers. ``omen_experiment`` itself was fixed in pass
#: 114 and has its own test; ``omen_l2_scheme_probe`` and ``omen_layer_probe``
#: are held by another item and are named here so the next reader knows the
#: count is 6 of 8 by decision rather than by oversight.
HARNESSES = (
    "omen_agreement_census",
    "omen_label_audit",
    "omen_self_distinctness",
    "omen_query_path_probe",
    "omen_shape_mutations",
    "omen_temporal_census",
)

#: 180 minutes rather than the 720 default, because it resolves to a DIFFERENT
#: bar count at each test cadence (180 bars at 60s, 3 bars at 3600s) and the
#: whole point is that one number of minutes is two numbers of bars.
TEST_MINUTES = 180

#: The line ``settle_horizon`` prints, with the optional per-corpus label a
#: multi-corpus harness adds. This is the harness SAYING what it resolved.
HORIZON_LINE = re.compile(
    r"horizon(?: \[[^\]]+\])? ([\d.]+) min = (\d+) bars of (\d+)s")


def _bars(count: int, cadence: int) -> list:
    """A deterministic corpus at an exact cadence, with real OHLC and volume.

    Wiggles rather than trends: a monotone ramp labels every bar the same and
    several of these harnesses refuse a corpus with one label.
    """
    out = []
    price = 100.0
    for i in range(count):
        price *= 1.0 + 0.012 * math.sin(i / 3.0) + 0.004 * math.cos(i / 11.0)
        high = price * 1.006
        low = price * 0.994
        out.append({
            "timestamp": 1_700_000_000 + i * cadence,
            "open": round(price * 0.999, 8), "high": round(high, 8),
            "low": round(low, 8), "close": round(price, 8),
            "volume": 1000.0 + 250.0 * math.sin(i / 5.0),
        })
    return out


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    """The same 3000 bars at 60s and at 3600s -- one question, two cadences."""
    base = tmp_path_factory.mktemp("omen_horizon_corpora")
    made = {}
    for cadence in (60, 3600):
        path = base / f"0000_SYNTH{cadence}-USDC.json"
        path.write_text(json.dumps(_bars(3000, cadence)), encoding="utf-8")
        made[cadence] = path
    return made


def _invoke(harness: str, corpus: Path, out: Path | None, *extra: str):
    """Run one harness with a minutes horizon. Returns (rc, combined output).

    Each harness is given arguments that reach ``settle_horizon`` and then stop
    on a clean, node-free exit, because two of the six need a brain node to do
    their real work and this test must never touch one.
    """
    argv = [sys.executable, "-X", "utf8", str(ROOT / "scripts" / f"{harness}.py")]
    if harness == "omen_shape_mutations":
        argv.append("census")
    argv += ["--corpus", str(corpus), "--horizon-minutes", str(TEST_MINUTES)]
    argv += list(extra)
    if out is not None:
        argv += ["--json-out" if harness == "omen_temporal_census" else "--report",
                 str(out)]
    proc = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True,
                          timeout=600)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


#: How each harness is driven to a node-free stop just past the horizon.
#: ``writes`` is True where the harness can produce a report without a node.
DRIVE = {
    # --train larger than the corpus: plan_windows refuses AFTER the horizon is
    # settled, so the node is never contacted.
    "omen_agreement_census": (["--train", "100000", "--test", "50",
                               "--endpoint", "127.0.0.1:8093"], False),
    # an unknown collection name is refused AFTER the horizon is settled.
    "omen_query_path_probe": (["--query-a", "nosuchcollection",
                               "--query-b", "nosuchcollection"], False),
    "omen_label_audit": (["--min-bars", "100", "--symbols", "1"], True),
    "omen_self_distinctness": (["--bars", "400", "--null-trials", "100"], True),
    "omen_shape_mutations": (["--samples", "15"], True),
    "omen_temporal_census": (["--samples", "100", "--live-hours", "0.01"], True),
}


@pytest.mark.parametrize("harness", HARNESSES)
def test_the_same_minutes_resolve_to_different_bars_at_different_cadences(
        harness, corpora, tmp_path):
    """One question in minutes, two cadences, two bar counts -- said out loud.

    This is the whole defect in one assertion. Before the fix all six took
    ``--horizon 12`` and asked 12 minutes of the 60s corpus and 12 hours of the
    3600s one while both reports said "h12".
    """
    extra, writes = DRIVE[harness]
    seen = {}
    for cadence, corpus in corpora.items():
        out = (tmp_path / f"{harness}-{cadence}.json") if writes else None
        rc, text = _invoke(harness, corpus, out, *extra)
        found = HORIZON_LINE.search(text)
        assert found, (f"{harness} never said what horizon it resolved at "
                       f"{cadence}s (rc={rc}):\n{text[-3000:]}")
        minutes, bars, said_cadence = found.groups()
        assert int(said_cadence) == cadence, (
            f"{harness} converted with {said_cadence}s, not the corpus's own "
            f"{cadence}s")
        assert float(minutes) == float(TEST_MINUTES)
        seen[cadence] = int(bars)

    assert seen[60] == TEST_MINUTES, (
        f"{harness}: {TEST_MINUTES} minutes of 60s bars is {TEST_MINUTES} "
        f"bars, not {seen[60]}")
    assert seen[3600] == TEST_MINUTES // 60, (
        f"{harness}: {TEST_MINUTES} minutes of 3600s bars is "
        f"{TEST_MINUTES // 60} bars, not {seen[3600]}")
    assert seen[60] != seen[3600], (
        f"{harness} gave the SAME bar count to two cadences, which is the "
        f"bars-only default this test exists to catch")


@pytest.mark.parametrize("harness",
                         [h for h in HARNESSES if DRIVE[h][1]])
def test_the_report_carries_both_units_and_the_cadence(harness, corpora,
                                                       tmp_path):
    """A report that cannot say what horizon it measured is not comparable."""
    extra, _ = DRIVE[harness]
    for cadence, corpus in corpora.items():
        out = tmp_path / f"{harness}-{cadence}.json"
        rc, text = _invoke(harness, corpus, out, *extra)
        assert out.exists(), (f"{harness} wrote no report at {cadence}s "
                              f"(rc={rc}):\n{text[-3000:]}")
        report = json.loads(out.read_text(encoding="utf-8"))
        for field in ("horizon_minutes", "horizon_bars", "bar_seconds"):
            assert report.get(field) is not None, (
                f"{harness}'s report is missing {field}: a reader cannot tell "
                f"it from a run at another cadence")
        assert report["bar_seconds"] == cadence
        assert float(report["horizon_minutes"]) == float(TEST_MINUTES)
        # The three agree with each other, to within the half bar that
        # rounding a wall clock to whole bars can move.
        implied = report["horizon_bars"] * report["bar_seconds"] / 60.0
        assert abs(implied - report["horizon_minutes"]) <= cadence / 120.0 + 1e-6
        # And the write-time guard is the same one, so it cannot drift.
        validate_report_horizon(report)


@pytest.mark.parametrize("harness", HARNESSES)
def test_bars_remain_an_explicit_override_that_records_its_minutes(
        harness, corpora, tmp_path):
    """An old run reproduces bar-for-bar, and the report says what it asked.

    ``--horizon 12`` must keep working on all six or every recorded command in
    ``data/attempts-revenir.md`` stops running. What changes is that the run
    now WRITES the minutes it worked out to.
    """
    extra, writes = DRIVE[harness]
    corpus = corpora[3600]
    out = (tmp_path / f"{harness}-bars.json") if writes else None
    argv = [sys.executable, "-X", "utf8",
            str(ROOT / "scripts" / f"{harness}.py")]
    if harness == "omen_shape_mutations":
        argv.append("census")
    argv += ["--corpus", str(corpus), "--horizon", "12"] + list(extra)
    if out is not None:
        argv += ["--json-out" if harness == "omen_temporal_census" else "--report",
                 str(out)]
    proc = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True,
                          timeout=600)
    text = (proc.stdout or "") + (proc.stderr or "")
    found = HORIZON_LINE.search(text)
    assert found, f"{harness} refused an explicit bar horizon:\n{text[-3000:]}"
    minutes, bars, cadence = found.groups()
    assert int(bars) == 12
    assert int(cadence) == 3600
    assert float(minutes) == 720.0, (
        f"{harness} recorded {minutes} min for 12 bars of 3600s, not 720")
    if out is not None:
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["horizon_bars"] == 12
        assert float(report["horizon_minutes"]) == 720.0
        assert report.get("horizon_source") == "bars"


@pytest.mark.parametrize("harness", HARNESSES)
def test_asking_in_both_units_at_once_is_refused_rather_than_picked_for_you(
        harness, corpora):
    """Bars and minutes are one quantity: the caller resolves the clash.

    Also the cheapest proof that every one of the six routes through the SHARED
    resolver rather than its own copy -- the message comes from
    ``omen_experiment.horizon_request`` and from nowhere else.
    """
    extra, _ = DRIVE[harness]
    argv = [sys.executable, "-X", "utf8",
            str(ROOT / "scripts" / f"{harness}.py"), "--corpus",
            str(corpora[3600]), "--horizon", "12",
            "--horizon-minutes", str(TEST_MINUTES)]
    if harness == "omen_shape_mutations":
        argv.insert(4, "census")
    argv += list(extra)
    proc = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True,
                          timeout=600)
    text = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode != 0, f"{harness} silently picked one:\n{text[-2000:]}"
    assert "same quantity in two units" in text, (
        f"{harness} did not route through the shared resolver:\n{text[-2000:]}")


def test_a_report_missing_horizon_minutes_is_refused_at_write_time():
    """The guard the six call before writing, checked directly."""
    good = {"horizon_bars": 12, "horizon_minutes": 720.0, "bar_seconds": 3600}
    validate_report_horizon(good)  # the control: a complete report passes

    for dropped in ("horizon_minutes", "horizon_bars", "bar_seconds"):
        partial = {k: v for k, v in good.items() if k != dropped}
        with pytest.raises(ValueError) as caught:
            validate_report_horizon(partial)
        assert dropped in str(caught.value)

    # Present but disagreeing is worse than absent, because it reads as fact.
    with pytest.raises(ValueError, match="disagrees with itself"):
        validate_report_horizon({"horizon_bars": 12, "horizon_minutes": 12.0,
                                 "bar_seconds": 3600})


def test_a_sweep_does_not_apply_the_first_corpus_bar_count_to_the_second():
    """The freeze in ``horizon_request``, which two of the six depend on.

    ``settle_horizon`` rewrites ``args.horizon`` to bars. Without the freeze the
    second corpus in a sweep would read that bar count, conclude the caller
    asked in bars, and apply the FIRST file's horizon to a different cadence --
    the original defect, reintroduced one loop iteration later.
    """
    class Args:
        horizon = None
        horizon_minutes = None

    args = Args()
    first = settle_horizon(args, 60)
    second = settle_horizon(args, 3600)
    assert first["horizon_bars"] == 720 and second["horizon_bars"] == 12
    assert first["horizon_minutes"] == second["horizon_minutes"] == \
        DEFAULT_HORIZON_MINUTES
    assert first["horizon_source"] == second["horizon_source"] == "minutes"

    # And an EXPLICIT bar count stays explicit across the same sweep: 12 bars
    # on both files, with the minutes correctly differing.
    class BarArgs:
        horizon = 12
        horizon_minutes = None

    bar_args = BarArgs()
    a = settle_horizon(bar_args, 60)
    b = settle_horizon(bar_args, 3600)
    assert a["horizon_bars"] == b["horizon_bars"] == 12
    assert a["horizon_minutes"] == 12.0 and b["horizon_minutes"] == 720.0


def test_the_default_reproduces_the_old_twelve_bar_run_on_hourly_bars():
    """720 minutes IS the old default on the 3600s cadence most files carry.

    If this ever fails, every hourly number recorded before pass 114 stopped
    being comparable with every number recorded after it.
    """
    class Args:
        horizon = None
        horizon_minutes = None

    assert horizon_request(Args())["minutes"] == DEFAULT_HORIZON_MINUTES
    assert resolve_horizon(3600, minutes=DEFAULT_HORIZON_MINUTES)[
        "horizon_bars"] == 12


def test_a_zero_bar_horizon_is_never_produced_by_rounding():
    """A zero-bar horizon compares a bar's close with itself.

    That reports a flawless, free, entirely fictional edge, and a coarse enough
    cadence would round to it: 180 minutes of 86400s bars is 0.125 bars.
    """
    assert resolve_horizon(86400, minutes=180)["horizon_bars"] == 1
    with pytest.raises(ValueError):
        resolve_horizon(3600, bars=0)
    with pytest.raises(ValueError):
        resolve_horizon(3600, minutes=0)
