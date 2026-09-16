"""Was the champion better than the best of N coin-flippers?

THE ERROR THIS EXISTS TO STOP. ga.py already scores every genome out-of-
sample and penalises it against the majority class, and that is a real guard:
it stops any SINGLE genome being credited for fitting its own training
window. It is not the guard people think it is. A genetic search keeps the
BEST of everything it evaluated, and taking a maximum over many noisy
estimates is itself a selection procedure. With population 8 over 4
generations the search screens roughly 32 genomes, so the right question is
never "could one coin-flipper score +8.67pp on 320 trades" -- it is "could
the BEST OF 32 coin-flippers", and the answers differ enormously.

Measured on this repo, 2026-09-16, on the first four real runs:

    seed  n     observed   null median   null 95th
    909   132   +20.41pp   +9.05pp       +12.83pp
    77    320    +8.67pp   +5.63pp        +8.13pp
    11    276    +5.46pp   ~+6pp          ~+8.5pp
    404   645    +1.27pp   +3.94pp        +5.80pp

Every seed was POSITIVE and a naive binomial z-test called two of them
significant (p=0.002 and p<0.001). Against the correct null the two
largest-sample runs score BELOW the median of pure chance, and the "p=0.002"
one sits at the 95th percentile -- one-in-twenty luck, found by a search that
draws far more than twenty times. The edge also correlated -0.842 with sample
size, which is the shape of an artifact: a real effect holds as n grows, a
selection artifact decays toward zero because rare-trading genomes get to
cherry-pick.

WHY A SIMULATION RATHER THAN A FORMULA. The distribution of the maximum of N
binomial proportions has no convenient closed form, and the approximations
that do exist assume independence the genomes do not have -- they share
ancestry through crossover and elitism, so they are correlated in a way that
depends on the run. A simulation states its assumption plainly (independent
draws at the baseline rate) and that assumption is CONSERVATIVE: correlated
genomes explore fewer effectively-distinct hypotheses, so the true null is
somewhat tighter than this one. Passing this test is therefore meaningful;
failing it is decisive.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence

#: Simulations per verdict. 400 resolves a 95th percentile well enough to act
#: on while staying fast enough to run at the end of every search.
DEFAULT_TRIALS = 400

#: The percentile a champion must clear. 95 rather than 99 because this null
#: is already conservative (see the module docstring) and because the cost of
#: a false negative here is a discarded genome, not a lost trade -- nothing
#: downstream trades on this verdict, it only decides what gets called a
#: finding.
DEFAULT_PERCENTILE = 95.0


@dataclass
class NullVerdict:
    """What the selection null says about one champion."""

    observed_edge: float
    null_median: float
    null_threshold: float
    percentile_of_null: float
    genomes_screened: int
    trades: int
    passed: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "observed_edge": self.observed_edge,
            "null_median": self.null_median,
            "null_threshold": self.null_threshold,
            "percentile_of_null": self.percentile_of_null,
            "genomes_screened": self.genomes_screened,
            "trades": self.trades,
            "passed": self.passed,
            "reason": self.reason,
        }


def _percentile(sorted_vals: Sequence[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * (pct / 100.0))
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def best_of_n_null(
    *,
    genomes_screened: int,
    trades: int,
    baseline: float,
    trials: int = DEFAULT_TRIALS,
    percentile: float = DEFAULT_PERCENTILE,
    seed: int = 0,
) -> tuple:
    """(median, threshold, sorted samples) of the best-of-N edge under chance.

    Each trial draws `genomes_screened` coin-flippers, each judged on `trades`
    independent bets at the `baseline` rate, and keeps the best edge. That
    maximum is what the search would report having found nothing at all.
    """
    if genomes_screened < 1 or trades < 1:
        return 0.0, 0.0, []
    baseline = min(max(float(baseline), 1e-6), 1 - 1e-6)
    rng = random.Random(seed or 0xC0FFEE)
    bests = []
    for _ in range(max(1, trials)):
        best = None
        for _ in range(genomes_screened):
            wins = 0
            for _ in range(trades):
                if rng.random() < baseline:
                    wins += 1
            edge = wins / float(trades) - baseline
            if best is None or edge > best:
                best = edge
        bests.append(best if best is not None else 0.0)
    bests.sort()
    return (_percentile(bests, 50.0), _percentile(bests, percentile), bests)


def judge(
    *,
    observed_accuracy: float,
    baseline: float,
    trades: int,
    population: int,
    generations: int,
    trials: int = DEFAULT_TRIALS,
    percentile: float = DEFAULT_PERCENTILE,
    seed: int = 0,
) -> NullVerdict:
    """Did this champion beat the best of the genomes the search screened?

    `population * generations` is the screening count: an upper bound, since
    elites are re-used rather than re-drawn, and an upper bound is the
    conservative direction -- it makes the null HARDER to beat, so a pass is
    not an artefact of over-counting.
    """
    screened = max(1, int(population) * max(1, int(generations)))
    observed = float(observed_accuracy) - float(baseline)

    if trades < 1:
        return NullVerdict(observed, 0.0, 0.0, 0.0, screened, 0, False,
                           "no held-out trades: nothing was actually tested")

    median, threshold, samples = best_of_n_null(
        genomes_screened=screened, trades=int(trades), baseline=float(baseline),
        trials=trials, percentile=percentile, seed=seed)

    beaten = sum(1 for s in samples if s < observed)
    pct_of_null = 100.0 * beaten / float(len(samples)) if samples else 0.0
    passed = observed > threshold

    if passed:
        reason = ("edge %+.4f clears the %.0fth percentile of best-of-%d "
                  "chance (%+.4f) on %d held-out trades"
                  % (observed, percentile, screened, threshold, trades))
    elif observed <= median:
        reason = ("edge %+.4f is at or BELOW the median of best-of-%d chance "
                  "(%+.4f) on %d trades: the search would have reported this "
                  "having found nothing"
                  % (observed, screened, median, trades))
    else:
        reason = ("edge %+.4f sits at the %.0fth percentile of best-of-%d "
                  "chance and does not clear the %.0fth (%+.4f) on %d trades: "
                  "consistent with selection luck"
                  % (observed, pct_of_null, screened, percentile, threshold,
                     trades))
    return NullVerdict(observed, median, threshold, pct_of_null, screened,
                       int(trades), passed, reason)


def edge_sample_correlation(runs: Sequence[dict]) -> Optional[float]:
    """corr(held-out trades, edge) across runs, or None if under-determined.

    A real effect holds roughly constant as the held-out sample grows; a
    selection artifact decays toward zero, because the genomes that trade
    rarely are the ones with room to get lucky. Measured -0.842 across the
    first four real runs here, which is what prompted this module.

    Needs at least three runs with distinct sample sizes -- two points always
    correlate perfectly and would manufacture a finding from nothing.
    """
    pairs = [(float(r.get("trades") or 0), float(r.get("edge") or 0.0))
             for r in runs if (r.get("trades") or 0) > 0]
    if len(pairs) < 3:
        return None
    ns = [p[0] for p in pairs]
    es = [p[1] for p in pairs]
    if len(set(ns)) < 2 or len(set(es)) < 2:
        return None
    mn = sum(ns) / len(ns)
    me = sum(es) / len(es)
    num = sum((n - mn) * (e - me) for n, e in zip(ns, es))
    den = (sum((n - mn) ** 2 for n in ns) ** 0.5
           * sum((e - me) ** 2 for e in es) ** 0.5)
    return (num / den) if den else None
