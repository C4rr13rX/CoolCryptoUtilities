"""An L1 negative must name the horizon it measured, in a unit that travels.

THE FAILURE THESE TESTS PREVENT. Pass 114 scored the L1 motif rule over 102
corpora and wrote the result down as ``--horizon 12``. On the 3600s cadence
that sweep was filtered to, twelve bars is TWELVE HOURS -- so the negative was
read for four passes as "L1 does not work", when what was measured was "L1 does
not work at twelve hours". The scope was invisible because the report recorded
the bars and not the minutes.

The second half of the same failure is worse and is the one this file is named
after: asking for a horizon the cadence CANNOT EXPRESS. Thirty minutes on
hourly bars is half a bar. It rounds to one bar, the run happily produces
numbers, and a report that then writes ``horizon_minutes: 30`` has labelled a
sixty-minute measurement as a thirty-minute one -- a stale number that no later
reader can detect, which is exactly how this repo's fake edges have survived.

And the third: ``omen_l1_horizon_sweep.score_window`` is a deliberate copy of
``omen_layer_probe.heldout_edge`` that keeps the per-trade returns the original
aggregates away. A copy that drifts from its original would report a different
edge under the same name, so it is pinned here against the original on a real
corpus rather than trusted.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.omen_experiment import validate_report_horizon  # noqa: E402
from scripts.omen_l1_horizon_sweep import (  # noqa: E402
    PROPERTIES, apply_threshold, auc, best_threshold, pooled_edge,
    score_window, sweep_horizon, window_properties,
)
from scripts.omen_layer_probe import heldout_edge, load_bars  # noqa: E402
from scripts.omen_scorable_windows import median_spacing  # noqa: E402

CORPUS_DIR = ROOT / "data" / "historical_ohlcv" / "base"
REPORT_3600 = ROOT / "data" / "brain_experiments" / \
    "L1-HORIZON-BAND-3600s-pass118-cove.json"
REPORT_300 = ROOT / "data" / "brain_experiments" / \
    "L1-HORIZON-BAND-300s-pass118-cove.json"


def _one_corpus(cadence: float, min_bars: int):
    """The first readable corpus at ``cadence`` with enough bars, or skip."""
    if not CORPUS_DIR.is_dir():
        pytest.skip("no base corpus checked out")
    for path in sorted(CORPUS_DIR.glob("*.json")):
        try:
            bars = load_bars(path)
        except Exception:
            continue
        if len(bars) >= min_bars and median_spacing(bars) == cadence:
            return path.stem.split("_", 1)[-1], bars
    pytest.skip("no corpus at cadence %s with %d bars" % (cadence, min_bars))


class _Args:
    """The subset of the CLI namespace ``sweep_horizon`` reads."""

    train = 350
    test = 60
    min_lift = 1.3
    min_support = 20
    hysteresis = 0.50


# --------------------------------------------------------------------------
# 1. The copied scorer must agree with the original, trade for trade.
# --------------------------------------------------------------------------


def test_score_window_reproduces_heldout_edge_exactly():
    """The per-trade copy is the same measurement as the aggregate original.

    If this ever fails, the sweep's edge numbers are not comparable with any
    number produced by omen_layer_probe, and the two would be quoted side by
    side under the same name.
    """
    symbol, bars = _one_corpus(3600.0, 350 + 60 + 2 + 60)
    window = 350 + 60 + 2 + 60
    slice_ = bars[-window:]
    mine = score_window(slice_, symbol, "base", 2, 350, 60, 1.3, 20, 0.50)
    theirs = heldout_edge(slice_, symbol, "base", 2, 350, 60, 1.3, 20,
                          relative=True, margin=0.50)

    assert "error" not in mine and "error" not in theirs
    assert mine["called_n"] == theirs["called_n"]
    assert mine["train_window"] == theirs["train_window"]
    assert mine["test_window"] == theirs["test_window"]
    assert mine["test_n"] == theirs["test_n"]
    assert mine["buyable_motifs"] == theirs["buyable_motifs"]
    assert mine["called_net"] == pytest.approx(theirs["called_net"], abs=1e-12)
    assert mine["baseline_net"] == pytest.approx(theirs["baseline_net"],
                                                 abs=1e-12)


def test_per_trade_returns_are_the_sample_behind_the_mean():
    """The returns kept for the error bar must BE the mean that is reported.

    A standard error computed over a different sample than the headline mean
    is worse than none: it would size an error bar for a number nobody quoted.
    """
    symbol, bars = _one_corpus(3600.0, 350 + 60 + 2 + 60)
    window = 350 + 60 + 2 + 60
    scored = score_window(bars[-window:], symbol, "base", 2, 350, 60,
                          1.3, 20, 0.50)
    assert len(scored["called_returns"]) == scored["called_n"]
    assert len(scored["test_returns"]) == scored["test_n"]
    if scored["called_n"]:
        mean = sum(scored["called_returns"]) / len(scored["called_returns"])
        assert mean == pytest.approx(scored["called_net"], abs=1e-12)
    mean_base = sum(scored["test_returns"]) / len(scored["test_returns"])
    assert mean_base == pytest.approx(scored["baseline_net"], abs=1e-12)


# --------------------------------------------------------------------------
# 2. A horizon the cadence cannot express must be reported as what it IS.
# --------------------------------------------------------------------------


def test_a_sub_bar_horizon_is_not_reported_as_the_minutes_that_were_asked():
    """Asking 30 minutes of hourly bars must not produce a '30 minute' report.

    THE BUG: ``horizon_bars(30, 3600)`` is 1, so the run measures SIXTY
    minutes. A report writing the asked 30 would be a stale number with no
    way for a later reader to catch it.
    """
    symbol, bars = _one_corpus(3600.0, 350 + 60 + 4 + 60)
    corpora = [{"corpus": "probe.json", "symbol": symbol, "bars": bars}]
    report = sweep_horizon(corpora, "base", 3600.0, 30.0, _Args())

    assert report["asked_minutes"] == 30.0
    assert report["horizon_bars"] == 1
    assert report["horizon_minutes"] == 60.0, \
        "one bar of 3600s is sixty minutes, whatever was asked for"
    assert report["sub_bar"] is True
    validate_report_horizon(report)


def test_a_horizon_the_cadence_can_express_is_not_flagged_sub_bar():
    """The sub-bar flag must discriminate, or it means nothing.

    A flag that is always true would be ignored within a pass, which is how
    a warning becomes noise.
    """
    symbol, bars = _one_corpus(3600.0, 350 + 60 + 4 + 60)
    corpora = [{"corpus": "probe.json", "symbol": symbol, "bars": bars}]
    report = sweep_horizon(corpora, "base", 3600.0, 120.0, _Args())
    assert report["horizon_bars"] == 2
    assert report["horizon_minutes"] == 120.0
    assert report["sub_bar"] is False


def test_every_sweep_report_carries_all_three_horizon_fields():
    """horizon_bars, horizon_minutes and bar_seconds, on every arm.

    Any two determine the third, and a reader with only one cannot tell what
    question was asked. Jet's pass-117 audit found 79 of 91 reports already in
    data/brain_experiments/ missing them.
    """
    symbol, bars = _one_corpus(3600.0, 350 + 60 + 4 + 60)
    corpora = [{"corpus": "probe.json", "symbol": symbol, "bars": bars}]
    for asked in (60.0, 120.0, 240.0):
        report = sweep_horizon(corpora, "base", 3600.0, asked, _Args())
        for field in ("horizon_bars", "horizon_minutes", "bar_seconds"):
            assert report.get(field) is not None, \
                "%s missing from the %g-minute arm" % (field, asked)
        validate_report_horizon(report)


# --------------------------------------------------------------------------
# 3. The statistics must refuse to manufacture an edge out of an abstention.
# --------------------------------------------------------------------------


def test_a_window_that_called_nothing_has_no_per_trade_net():
    """Zero trades is None, never 0.0.

    0.0 against a negative baseline reads as a POSITIVE edge, so an all-
    abstaining sweep would report the rule beating the market by exactly the
    market's own loss. That is the single easiest fake edge to ship here.
    """
    rows = [{"called_n": 0, "called_returns": [], "test_returns": [-0.02, -0.03]}]
    out = pooled_edge(rows)
    assert out["trades"] == 0
    assert out["net"] is None and out["edge_pp"] is None and out["z"] is None


def test_the_baseline_is_taken_only_in_windows_the_rule_actually_traded():
    """Buy-every-bar is scored where the rule fired, not where it declined.

    Pooling the baseline over declined windows scores the rule against a
    market it never traded -- in a sweep that abstains four times in five,
    that is most of the sample.
    """
    rows = [
        {"called_n": 2, "called_returns": [0.01, 0.03],
         "test_returns": [0.0, 0.01, 0.03, 0.04]},
        # A declined window whose market collapsed. If its bars reached the
        # baseline, the rule would look brilliant for not trading it.
        {"called_n": 0, "called_returns": [],
         "test_returns": [-0.5, -0.5, -0.5, -0.5]},
    ]
    out = pooled_edge(rows)
    assert out["fired"] == 1 and out["trades"] == 2
    assert out["baseline"] == pytest.approx(0.02, abs=1e-12)
    assert out["edge_pp"] == pytest.approx(0.0, abs=1e-9)


def test_the_standard_error_is_computed_from_the_trades_not_the_windows():
    """se must shrink with the TRADE count, or it sizes the wrong sample.

    The pass-114 headline was +0.10pp on 194 trades from 19 windows. An error
    bar computed over 19 window means is roughly three times too wide, and
    would have called a real difference noise.
    """
    rows = [{"called_n": 4, "called_returns": [0.01, 0.02, 0.03, 0.04],
             "test_returns": [0.01, 0.02, 0.03, 0.04]}]
    out = pooled_edge(rows)
    expected = statistics.stdev([0.01, 0.02, 0.03, 0.04]) / (4 ** 0.5)
    # se of the difference is se(called) and se(baseline) added in quadrature;
    # here the two samples are identical, so it is sqrt(2) times one of them.
    assert out["se_pp"] == pytest.approx(100.0 * expected * (2 ** 0.5),
                                         rel=1e-9)


# --------------------------------------------------------------------------
# 4. The selector question: properties are pre-scoring, and the check is held out.
# --------------------------------------------------------------------------


def test_window_properties_never_look_past_the_train_stop():
    """A selector fitted on anything the test window knows is a leak.

    Rewriting every bar after ``stop`` must not move a single property. If it
    does, the separation result is reading the answer it is predicting.
    """
    symbol, bars = _one_corpus(3600.0, 350 + 60 + 2 + 60)
    start, stop = 60, 300
    before = window_properties(bars, start, stop)
    poisoned = [dict(b) for b in bars]
    for bar in poisoned[stop:]:
        bar["close"] = 1e9
    after = window_properties(poisoned, start, stop)
    for prop in PROPERTIES:
        if prop == "bar_count":       # length is unchanged by the poisoning
            continue
        assert before[prop] == pytest.approx(after[prop], abs=1e-15), prop


def test_auc_is_a_half_when_the_property_says_nothing():
    assert auc([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.5)
    assert auc([3.0, 4.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert auc([1.0, 2.0], [3.0, 4.0]) == pytest.approx(0.0)
    assert auc([], [1.0]) is None


def test_the_selector_is_fitted_and_checked_on_different_corpora():
    """A threshold read on the sample that chose it always separates it.

    Constructed so the discovery half is separable by ``realised_vol`` and the
    held-out half carries the same relation: the check must therefore report a
    fire rate above the overall rate, and it must be computed on rows the fit
    never saw.
    """
    def row(vol, fired):
        return {"called_n": 1 if fired else 0,
                "properties": {"realised_vol": vol, "up_rate": 0.5,
                               "bar_count": 600.0, "mean_abs_return": 0.01,
                               "train_drift": 0.0}}

    discovery = [row(0.05, True), row(0.06, True), row(0.01, False),
                 row(0.02, False)]
    rule = best_threshold(discovery, "realised_vol")
    assert rule is not None and rule["direction"] == 1
    assert 0.02 < rule["cut"] <= 0.05

    heldout = [row(0.07, True), row(0.08, True), row(0.005, False),
               row(0.015, False)]
    check = apply_threshold(heldout, rule)
    assert check["selected_n"] == 2
    assert check["fire_rate_selected"] == pytest.approx(1.0)
    assert check["fire_rate_rejected"] == pytest.approx(0.0)
    assert check["lift"] == pytest.approx(2.0)


def test_best_threshold_finds_a_rule_that_fires_when_the_property_is_LOW():
    """Both directions, or half the selectors are invisible.

    A rule that fires in QUIET windows is as useful as one that fires in
    volatile ones, and searching one direction would report "no separator"
    for it.
    """
    def row(vol, fired):
        return {"called_n": 1 if fired else 0,
                "properties": {"realised_vol": vol}}

    rows = [row(0.01, True), row(0.02, True), row(0.09, False), row(0.08, False)]
    rule = best_threshold(rows, "realised_vol")
    assert rule["direction"] == -1
    assert rule["youden_j"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 5. The shipped reports say what they measured.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", [REPORT_3600, REPORT_300])
def test_the_shipped_reports_name_the_horizon_of_every_arm(path):
    if not path.exists():
        pytest.skip("%s has not been produced" % path.name)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["horizons"], "a report with no arms measured nothing"
    for arm in payload["horizons"]:
        validate_report_horizon(arm)
        assert arm["bar_seconds"] == int(payload["cadence"])
        assert "asked_minutes" in arm and "sub_bar" in arm
        # The abstention denominator must be the corpora actually scored, so
        # a rate cannot be quoted against a population that includes corpora
        # the sweep silently lost.
        assert arm["abstained"] <= arm["corpora_scored"]


def test_the_thirty_minute_question_is_answered_on_a_cadence_that_can_ask_it():
    """30 minutes is 6 bars of 300s, and that arm must not be flagged sub-bar.

    The 3600s set cannot ask it at all; this is why a second cadence was run.
    """
    if not REPORT_300.exists():
        pytest.skip("the 300s arm has not been produced")
    payload = json.loads(REPORT_300.read_text(encoding="utf-8"))
    assert payload["cadence"] == 300.0
    arms = {arm["asked_minutes"]: arm for arm in payload["horizons"]}
    assert 30.0 in arms, "the 30-minute arm is the reason this set was run"
    assert arms[30.0]["horizon_bars"] == 6
    assert arms[30.0]["horizon_minutes"] == 30.0
    assert arms[30.0]["sub_bar"] is False
