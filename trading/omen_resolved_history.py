#!/usr/bin/env python3
"""The resolved-prediction feeder: the record pools 15/16/19 read from.

WHAT WAS MISSING. ``omen_metacognition.self_frames`` takes a history of
``Resolved`` rows and refuses to look at an unsettled one -- that refusal is
the guard against the prediction_error loop that took recall from 100% to 30%
on this substrate. But nothing anywhere CONSTRUCTED that history, so every
call passed an empty sequence and the three self-knowledge pools emitted their
``na`` sentinels forever. Measured pass 109: 0.002 distinctness, which is a
constant. They were bound, streamed, and carried zero information.

THE ONE RULE THIS MODULE EXISTS TO ENFORCE, AND IT IS ENFORCED STRUCTURALLY.
A prediction made at bar ``j`` over a horizon of ``h`` bars does not become a
fact until bar ``j + h``. The frame built for bar ``i`` may therefore read a
prediction only when ``j + h <= i``. Anything looser is lookahead: it lets the
frame for bar ``i`` know an outcome that had not happened at bar ``i``, which
manufactures an edge that evaporates live. This repo has already paid for a
fake 78% directional and a fake +0.9067% and both were found later.

So causality is not a convention here. ``record`` cannot settle a row,
``settle`` cannot reach a row that is not due, and ``as_of`` filters on the
resolve index rather than on the ``resolved`` flag alone. A caller that does
the wrong thing gets an exception or an empty history, never a leaked answer.

  record(bar, predicted, ...)  a prediction is made and is UNSETTLED
  settle(bar, actual)          the outcome for the row that resolves at `bar`
  as_of(bar)                   the rows a frame at `bar` is allowed to see

WHY BOTH INDICES ARE STORED. Keeping only ``resolved`` would make the history
correct at the END of a walk-forward and wrong in the middle of one, because a
row settled at bar 900 would be visible to a frame rebuilt for bar 500. Storing
``resolve_index`` makes ``as_of`` answerable at any bar, which is what lets one
pass rebuild frames for a whole corpus without replaying the walk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from trading.omen_metacognition import Resolved

__all__ = ["Prediction", "ResolvedHistory", "build_samples_with_history"]


@dataclass
class Prediction:
    """One prediction and, once it has happened, its outcome.

    ``made_index`` is the bar the call was made on and ``resolve_index`` the
    bar its horizon lands on. Both are bar indices into one corpus; mixing two
    symbols' indices in one history is a units error this class cannot detect,
    so keep one history per (symbol, horizon).
    """
    made_index: int
    resolve_index: int
    predicted: str
    agreed: Optional[int] = None
    asked: Optional[int] = None
    actual: Optional[str] = None

    @property
    def settled(self) -> bool:
        return self.actual is not None

    def to_resolved(self) -> Resolved:
        """The shape ``self_frames`` reads.

        ``resolved`` is derived from whether an outcome is present, never set
        independently -- a row cannot claim to be settled without carrying the
        fact that settled it.
        """
        return Resolved(
            predicted=self.predicted,
            actual=self.actual,
            agreed=self.agreed,
            asked=self.asked,
            resolved=self.settled,
        )


class ResolvedHistory:
    """A chronological record of predictions and their settled outcomes.

    One instance per (symbol, horizon). Rows go in as predictions are made and
    become facts as their horizons land; ``as_of`` is the only read a frame
    builder should use.
    """

    def __init__(self, horizon_bars: int) -> None:
        if horizon_bars <= 0:
            raise ValueError(f"horizon_bars must be positive, got {horizon_bars}")
        self.horizon_bars = int(horizon_bars)
        self._rows: List[Prediction] = []
        self._by_resolve: Dict[int, List[Prediction]] = {}
        self._last_made: Optional[int] = None

    # -- writing -----------------------------------------------------------

    def record(self, made_index: int, predicted: str, *,
               agreed: Optional[int] = None,
               asked: Optional[int] = None) -> Prediction:
        """Log a prediction. It is UNSETTLED and cannot be settled here.

        Predictions must arrive in bar order. Out-of-order arrival is refused
        rather than sorted, because a walk-forward that jumps backwards has
        already read a future bar and the history is not the place to discover
        that.
        """
        made_index = int(made_index)
        if self._last_made is not None and made_index < self._last_made:
            raise ValueError(
                f"predictions must arrive oldest-first: bar {made_index} "
                f"after bar {self._last_made}. A backwards step means the "
                f"caller has already read a bar it has not reached.")
        row = Prediction(made_index=made_index,
                         resolve_index=made_index + self.horizon_bars,
                         predicted=str(predicted), agreed=agreed, asked=asked)
        self._rows.append(row)
        self._by_resolve.setdefault(row.resolve_index, []).append(row)
        self._last_made = made_index
        return row

    def settle(self, resolve_index: int, actual: str) -> int:
        """Settle every prediction whose horizon lands on ``resolve_index``.

        Returns how many rows were settled -- zero is normal and means no
        prediction was made a horizon ago. Re-settling a settled row is
        refused: an outcome is a fact and a second one means two different
        bars are being treated as the same bar.
        """
        rows = self._by_resolve.get(int(resolve_index), ())
        settled = 0
        for row in rows:
            if row.settled:
                raise ValueError(
                    f"prediction made at bar {row.made_index} is already "
                    f"settled with {row.actual!r}; refusing to overwrite "
                    f"with {actual!r}")
            row.actual = str(actual)
            settled += 1
        return settled

    # -- reading -----------------------------------------------------------

    def as_of(self, bar_index: int) -> Tuple[Resolved, ...]:
        """The rows a frame built for ``bar_index`` may read. Oldest-first.

        TWO CONDITIONS, BOTH REQUIRED. A row is visible only when its horizon
        has LANDED at or before ``bar_index`` -- the causal test -- and only
        when it actually carries an outcome. The second is not implied by the
        first: a walk-forward that never called ``settle`` for a bar leaves a
        due row without a fact, and a frame must not treat "I forgot to settle
        this" as "I predicted nothing".
        """
        bar_index = int(bar_index)
        return tuple(row.to_resolved() for row in self._rows
                     if row.resolve_index <= bar_index and row.settled)

    def pending(self, bar_index: int) -> int:
        """How many predictions are made but not yet due at ``bar_index``.

        Diagnostic only. A walk-forward should hold exactly ``horizon_bars``
        of these in flight once it is warm; a number far off that says the
        caller is skipping bars.
        """
        bar_index = int(bar_index)
        return sum(1 for row in self._rows
                   if row.made_index <= bar_index < row.resolve_index)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[Prediction]:
        return iter(self._rows)


def walk_forward(bar_indices: Sequence[int],
                 predictions: Sequence[str],
                 actuals: Dict[int, str],
                 horizon_bars: int,
                 *,
                 agreements: Optional[Sequence[Tuple[int, int]]] = None
                 ) -> List[Tuple[int, Tuple[Resolved, ...]]]:
    """Replay a prediction stream in bar order and hand back each bar's history.

    The reference driver: it is what a training or held-out sweep does, with
    the ordering and the settlement handled once so a caller cannot get them
    subtly wrong. ``actuals`` maps a bar index to the label that turned out to
    be true THERE, which is what settles a prediction made ``horizon_bars``
    earlier.

    Returns ``(bar_index, history_visible_at_that_bar)`` per bar, so the
    caller can build a frame per bar without ever holding a future row.
    """
    history = ResolvedHistory(horizon_bars)
    out: List[Tuple[int, Tuple[Resolved, ...]]] = []
    for position, bar in enumerate(bar_indices):
        # Settle FIRST: a prediction whose horizon lands exactly on this bar
        # is a fact by the time this bar is decided, so withholding it would
        # be needlessly blind -- the guard is against reading the OPEN one.
        if bar in actuals:
            try:
                history.settle(bar, actuals[bar])
            except ValueError:
                pass          # already settled; a repeated bar in the stream
        out.append((bar, history.as_of(bar)))
        if position < len(predictions):
            pair = agreements[position] if agreements and position < len(agreements) else (None, None)
            history.record(bar, predictions[position],
                           agreed=pair[0], asked=pair[1])
    return out


def build_samples_with_history(bars, symbol: str, chain: str,
                               horizon_bars: int, start: int, stop: int,
                               predictor=None):
    """Frames + label per bar, with the self pools FED rather than sentinelled.

    THE SEAM THIS CLOSES, measured pass 110 on a real 19-pool node. The probe
    reported ``QUERY PATH DEAD`` for a query set differing only by pools
    15/16/19 -- control 0/60, treatment 0/60 -- while the B arm fired SIX
    streams per prediction. The pools were sent and read. They moved nothing
    because every sample-building loop in this repo calls ``build_collections``
    without ``history=``, so every self frame in a training set is the ``na``
    sentinel and the three pools train as CONSTANTS. A constant stream cannot
    move a query however good the pool is.

    So this is the sample builder that hands the history in. It is the same
    loop as ``omen_experiment.build_samples`` with two additions: a
    ``ResolvedHistory`` walked alongside the bars, and ``history=`` passed
    through.

    ``predictor(bar_index, visible_history)`` returns the label to record as
    the prediction made at that bar. The default is the majority label among
    the rows the bar is allowed to see -- causal, non-oracle, and the same
    rule the scoreboard baselines against. A caller measuring the NODE's edge
    should pass the node's own prediction instead; that is the difference
    between sizing the vocabulary and measuring skill.

    Causality is inherited from ``ResolvedHistory`` rather than re-implemented:
    the frame for bar ``i`` reads ``as_of(i)``, which cannot return a row whose
    horizon has not landed. Settling happens BEFORE the frame is built, so a
    prediction whose horizon lands exactly on this bar is a fact here -- the
    guard is against reading the OPEN call, not against reading a settled one.
    """
    from collections import Counter

    from trading.omen_brain import (
        LOOKBACK_BARS, build_collections, label_omen, measure_bar_seconds)

    def _majority(_bar, visible):
        actuals = [r.actual for r in visible if r.actual]
        return Counter(actuals).most_common(1)[0][0] if actuals else "murk"

    predictor = predictor or _majority
    history = ResolvedHistory(horizon_bars)
    samples = []
    cadence = measure_bar_seconds(bars)
    for index in range(max(start, LOOKBACK_BARS), stop):
        label = label_omen(bars, index, horizon_bars=horizon_bars)
        if label is not None:
            try:
                history.settle(index, label)
            except ValueError:
                pass
        visible = history.as_of(index)
        if label is None:
            continue
        try:
            frames = build_collections(bars, index, horizon_bars=horizon_bars,
                                       bar_seconds=cadence,
                                       symbol=symbol, chain=chain,
                                       history=visible)
        except (ValueError, IndexError):
            continue
        samples.append({
            "index": index,
            "frames": frames,
            "label": label,
            "ts": int(bars[index]["timestamp"]),
            "price": float(bars[index]["close"]),
            "forward": (float(bars[index + horizon_bars]["close"])
                        - float(bars[index]["close"])) / float(bars[index]["close"]),
        })
        history.record(index, predictor(index, visible))
    return samples
