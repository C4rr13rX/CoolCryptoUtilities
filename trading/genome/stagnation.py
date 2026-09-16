"""Get the search out of a loop that will never find anything new.

THE FAILURE. `evolve()` ran a fixed generation count with elites copied
forward unchanged and no measure of whether the population was still
exploring. That has one outcome and it is not a slow one: elites dominate the
survivor pool, crossover between near-identical parents produces near-
identical children, mutation is the only remaining source of novelty, and the
search spends its whole budget resampling one point. Nothing in the run says
so. The generation counter still climbs, `on_generation` still fires, the log
still scrolls, and the best fitness has not moved in forty generations.

It is worse here than in a textbook GA because fitness is out-of-sample and
baseline-penalised (see ga.py): most genomes legitimately score zero, so the
selection signal is sparse, and a population that converges early converges
onto a point that may have no edge at all. A GA that cannot tell "we are
still looking" from "we stopped looking" will report the latter as the
former forever.

THREE ESCAPES, IN ORDER OF COST. They are tried in order because they cost
different amounts of the run's remaining budget, and spending the expensive
one first throws away work that was still productive:

  1. DIVERSITY FLOOR (cheapest, most often right). Measure how much of the
     population is genuinely distinct. When that collapses, replace the
     duplicates with fresh random genomes -- "random immigrants". The elites
     survive untouched, so nothing found so far is lost; only the redundant
     copies of them are spent. This fixes the common case, where the search
     is stuck because it has nothing left to recombine.

  2. WIDEN THE SPACE (middle). If diversity is healthy and fitness still is
     not moving, the population is spread out and none of it is any good --
     the answer is not in the box being searched. Widening the numeric gene
     ranges lets the search leave the box it was given, which is strictly
     more than resampling inside it.

  3. RESTART (most expensive, last). Abandon the population and re-seed from
     a new random draw, keeping a record of the best genome found so the
     restart cannot lose ground. Used only when the first two have already
     failed, because it discards every evaluation the run has paid for.

WHAT COUNTS AS IMPROVEMENT is deliberately strict: held-out fitness must beat
the best seen so far by `min_delta`, not merely equal it. A GA whose best
genome is unchanged but whose mean drifts up is not discovering anything, and
treating that as progress is how a stale run justifies itself indefinitely.
The bookkeeping is on the BEST score only, for the same reason ga.py scores
out-of-sample only: the mean is flattered by the elites it already holds.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

#: Generations without a real improvement before the first escape fires.
#: Low, because the cheap escape costs almost nothing and a search that was
#: not stuck loses only a few duplicate genomes by being nudged.
DEFAULT_PATIENCE = 6

#: Fitness must improve by at least this to count. Anything smaller is noise
#: in an out-of-sample score computed over a finite number of trades.
DEFAULT_MIN_DELTA = 1e-4

#: Below this fraction of distinct genomes the population has collapsed.
#: 0.5 rather than something tight: by the time half the population is
#: duplicates, crossover has nothing left to do.
DEFAULT_DIVERSITY_FLOOR = 0.5

#: How much of the population random immigrants may replace at once. Never
#: all of it -- that is a restart, and a restart is a separate, later, and
#: deliberately more expensive decision.
DEFAULT_IMMIGRANT_FRACTION = 0.3

#: Multiplier applied to numeric gene ranges when the space is widened.
DEFAULT_WIDEN_FACTOR = 1.5

#: How many times a single run may widen. Unbounded widening walks the search
#: into absurd parameter values that cannot be traded (a 900-bar lookback on
#: a 400-bar window), which wastes the budget more quietly than being stuck.
DEFAULT_MAX_WIDENINGS = 3


def genome_signature(genome: Any) -> tuple:
    """What makes two genomes the same for diversity purposes.

    The genes, not the id and not the score: two genomes with identical genes
    are the same point in the search space however they were produced, and
    counting them as distinct is exactly the error that lets a converged
    population look healthy.
    """
    genes = getattr(genome, "genes", None)
    if not isinstance(genes, dict):
        return ()
    out = []
    for key in sorted(genes):
        val = genes[key]
        if isinstance(val, float):
            # Bucket floats. Two lookbacks of 14.0000001 and 14.0 are not a
            # meaningfully different search point, and treating them as one
            # would report diversity that does not exist.
            val = round(val, 6)
        elif isinstance(val, (list, tuple)):
            val = tuple(val)
        out.append((key, val))
    return tuple(out)


def diversity(population: Sequence[Any]) -> float:
    """Fraction of the population that is a distinct search point, 0..1."""
    if not population:
        return 0.0
    sigs = {genome_signature(g) for g in population}
    return len(sigs) / float(len(population))


@dataclass
class StagnationMonitor:
    """Tracks whether a run is still discovering, and what to do when it is not.

    Deliberately holds no reference to the population, the bars or the
    evaluator: it is handed the numbers each generation and returns a verdict.
    That makes every branch testable without running a search, which matters
    because the branches that fire are the rare ones -- a bug in the restart
    path would otherwise surface for the first time forty generations into a
    long run.
    """

    patience: int = DEFAULT_PATIENCE
    min_delta: float = DEFAULT_MIN_DELTA
    diversity_floor: float = DEFAULT_DIVERSITY_FLOOR
    max_widenings: int = DEFAULT_MAX_WIDENINGS

    best_fitness: float = float("-inf")
    best_genome: Optional[Any] = None
    stale_generations: int = 0
    widenings: int = 0
    restarts: int = 0
    #: Every escape taken, in order, for the run record. A run that ends with
    #: no champion must be able to say what it tried.
    history: List[Dict[str, Any]] = field(default_factory=list)

    def observe(self, generation: int, population: Sequence[Any]) -> Dict[str, Any]:
        """Fold in one generation. Returns the verdict for that generation.

            {"action": "continue"|"immigrate"|"widen"|"restart",
             "stale": 7, "diversity": 0.29, "best": 0.0413, "reason": "..."}

        The caller applies the action. This never mutates the population,
        because a monitor that silently edits what it is measuring cannot be
        reasoned about.
        """
        div = diversity(population)
        best = None
        for genome in population:
            fit = float(getattr(genome, "fitness", 0.0) or 0.0)
            if best is None or fit > best:
                best = fit
        best = 0.0 if best is None else best

        improved = best > self.best_fitness + self.min_delta
        if improved:
            self.best_fitness = best
            for genome in population:
                if float(getattr(genome, "fitness", 0.0) or 0.0) == best:
                    self.best_genome = genome
                    break
            self.stale_generations = 0
        else:
            self.stale_generations += 1

        verdict = {"generation": generation, "action": "continue",
                   "stale": self.stale_generations, "diversity": div,
                   "best": self.best_fitness, "reason": ""}

        if improved or self.stale_generations < self.patience:
            verdict["reason"] = ("improved to %.6f" % best if improved
                                 else "stale %d/%d, still within patience"
                                      % (self.stale_generations, self.patience))
            return verdict

        # Stuck. Cheapest escape that fits the shape of the stuckness.
        if div < self.diversity_floor:
            verdict["action"] = "immigrate"
            verdict["reason"] = (
                "%d generations without improvement and diversity %.2f is "
                "below the %.2f floor: the population has collapsed onto one "
                "point, so the duplicates are replaced with random immigrants"
                % (self.stale_generations, div, self.diversity_floor))
        elif self.widenings < self.max_widenings:
            verdict["action"] = "widen"
            verdict["reason"] = (
                "%d generations without improvement while diversity is %.2f: "
                "the population is spread out and none of it is any good, so "
                "the answer is not in this box -- widening the gene ranges "
                "(widening %d of %d)"
                % (self.stale_generations, div, self.widenings + 1,
                   self.max_widenings))
        else:
            verdict["action"] = "restart"
            verdict["reason"] = (
                "%d generations without improvement, diversity %.2f, and the "
                "space has already been widened %d times: re-seeding from a "
                "fresh population, keeping the best genome found"
                % (self.stale_generations, div, self.widenings))

        # An escape resets the clock. Without this the monitor would fire an
        # escape every single generation after the first one, and a run that
        # immigrated once would immigrate forever instead of being given the
        # patience to show whether the immigrants helped.
        self.stale_generations = 0
        if verdict["action"] == "widen":
            self.widenings += 1
        elif verdict["action"] == "restart":
            self.restarts += 1
        self.history.append(dict(verdict))
        return verdict


# ----------------------------------------------------------- the escapes --

def inject_immigrants(population: List[Any], rng: random.Random,
                      make_genome: Callable[[], Any], *,
                      fraction: float = DEFAULT_IMMIGRANT_FRACTION,
                      keep_elite: int = 4) -> int:
    """Replace redundant genomes with fresh random ones. Returns how many.

    Replaces from the BACK of a fitness-sorted population and never touches
    the first `keep_elite`, so the best work found so far is never the thing
    thrown away -- the whole point is to spend the copies, not the original.
    """
    if not population:
        return 0
    count = int(len(population) * max(0.0, min(1.0, fraction)))
    count = min(count, max(0, len(population) - max(0, keep_elite)))
    if count <= 0:
        return 0
    for i in range(len(population) - count, len(population)):
        population[i] = make_genome()
    return count


def widen_space(space: Dict[str, Any], *,
                factor: float = DEFAULT_WIDEN_FACTOR) -> Dict[str, Any]:
    """A copy of the gene space with numeric ranges stretched.

    Only ranges expressed as (low, high) numeric pairs are touched; choice
    lists are left exactly as they are, because inventing a categorical value
    the evaluator has never seen produces a crash deep in a worker thread
    rather than a wider search. Integer bounds stay integers -- a lookback of
    14.7 bars is not a thing, and rounding it later would silently collapse
    several widened values back onto the same point.
    """
    out: Dict[str, Any] = {}
    for key, val in (space or {}).items():
        if (isinstance(val, (list, tuple)) and len(val) == 2
                and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        for v in val)):
            low, high = val
            mid = (low + high) / 2.0
            half = (high - low) / 2.0 * factor
            new_low, new_high = mid - half, mid + half
            if isinstance(low, int) and isinstance(high, int):
                new_low, new_high = int(round(new_low)), int(round(new_high))
                # Never let a widened integer range invert or collapse.
                if new_high <= new_low:
                    new_high = new_low + 1
                # Lookbacks, windows and counts are meaningless at zero or
                # below, and a negative one reaches the evaluator as a slice
                # that silently returns nothing.
                new_low = max(1, new_low) if low >= 1 else new_low
            out[key] = type(val)((new_low, new_high))
        else:
            out[key] = val
    return out
