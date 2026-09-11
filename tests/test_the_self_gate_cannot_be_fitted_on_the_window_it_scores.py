"""The self-frame abstention gate must be a TRAIN-window map, frozen.

THE FAILURE THIS PREVENTS, and it has already been paid for once. Pass 110's
L1 motif-to-label map was fitted on the same bars it was then scored on, so
its held-out number was a re-read of its own fit and meant nothing. [1f8c2461]
adds a second map of exactly that shape -- self frame bucket to trough rate --
and the only thing keeping it honest is that it is fitted on the train window
and frozen before a held-out bar is scored.

So these tests assert the two properties that ordering buys:

  1. a bucket that has troughs on TRAIN is never refused, however trough-free
     it looks in the scoring window -- i.e. the fit cannot see the test set;
  2. a bucket with too few train samples is never refused, because "0 troughs
     in 9 samples" at a 13-15% base rate happens by chance about a quarter of
     the time and refusing on it is fitting noise.

Both fail against a gate that counts the scored window, or against one with
no support floor.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.omen_experiment import (  # noqa: E402
    SELF_GATE_KEYS, fit_self_gate, self_gate_refuses,
)
from trading.omen_brain import OMEN_CREST, OMEN_MURK, OMEN_TROUGH  # noqa: E402

QUIET = "slf hit=q3 n=c2 last=hit"
BUSY = "slf hit=q1 n=c2 last=miss"
ERR_OK = "err run=h4 dir=na"


def _sample(label, outcome_frame, error_frame=ERR_OK):
    return {"label": label,
            "self": {"self_outcome": outcome_frame,
                     "self_error_run": error_frame,
                     "self_agreement": "agr unan=q0 rate=na"}}


def test_a_bucket_with_zero_troughs_over_enough_train_samples_refuses():
    train = [_sample(OMEN_MURK, QUIET) for _ in range(40)]
    train += [_sample(OMEN_TROUGH, BUSY) for _ in range(10)]
    train += [_sample(OMEN_CREST, BUSY) for _ in range(30)]

    gate = fit_self_gate(train, min_support=30)

    assert gate["refuse"]["self_outcome"] == [QUIET], (
        "QUIET has 0 troughs over 40 train samples and must be refused"
    )
    assert self_gate_refuses(gate, _sample(OMEN_TROUGH, QUIET)) == "self_outcome"
    assert self_gate_refuses(gate, _sample(OMEN_TROUGH, BUSY)) is None


def test_a_bucket_that_ever_troughed_on_train_is_kept_however_the_test_window_looks():
    """The fit sees TRAIN only. A bucket that troughed once on train stays
    allowed even though the scoring window it is applied to never troughs in
    it -- which is precisely the information a window-fitted map would use."""
    train = [_sample(OMEN_MURK, QUIET) for _ in range(39)]
    train += [_sample(OMEN_TROUGH, QUIET)]          # the one trough on train
    train += [_sample(OMEN_CREST, BUSY) for _ in range(35)]

    gate = fit_self_gate(train, min_support=30)

    assert QUIET not in gate["refuse"]["self_outcome"], (
        "one trough on train is enough to keep the bucket; refusing it would "
        "mean the fit had consulted something other than the train window"
    )
    # And the scoring window is trough-free in that bucket -- a gate fitted on
    # what it scores would refuse here. This one must not.
    scoring = [_sample(OMEN_MURK, QUIET) for _ in range(50)]
    assert all(self_gate_refuses(gate, s) is None for s in scoring)


def test_a_thin_bucket_never_refuses_because_zero_of_nine_is_not_a_rate():
    train = [_sample(OMEN_MURK, QUIET) for _ in range(9)]
    train += [_sample(OMEN_TROUGH, BUSY) for _ in range(40)]

    gate = fit_self_gate(train, min_support=30)

    assert gate["refuse"]["self_outcome"] == [], (
        "0 troughs in 9 samples is a small-sample accident at a 13-15% base "
        "rate, not a bucket that does not trough"
    )
    assert self_gate_refuses(gate, _sample(OMEN_TROUGH, QUIET)) is None


def test_the_gate_reads_every_key_it_declares():
    """A refusal on self_error_run must bite even when self_outcome allows."""
    train = [_sample(OMEN_MURK, BUSY, "err run=m5 dir=climb") for _ in range(35)]
    train += [_sample(OMEN_TROUGH, BUSY, ERR_OK) for _ in range(35)]

    gate = fit_self_gate(train, min_support=30)

    assert "self_error_run" in SELF_GATE_KEYS
    assert gate["refuse"]["self_error_run"] == ["err run=m5 dir=climb"]
    assert self_gate_refuses(
        gate, _sample(OMEN_TROUGH, BUSY, "err run=m5 dir=climb")
    ) == "self_error_run"


def test_a_sample_with_no_self_frame_is_allowed_not_crashed():
    gate = fit_self_gate(
        [_sample(OMEN_MURK, QUIET) for _ in range(40)], min_support=30)
    assert self_gate_refuses(gate, {"label": OMEN_TROUGH}) is None
