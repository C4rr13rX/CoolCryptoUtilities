"""Configurable genetic algorithm over brain I/O, shape, and sentiment genes.

The search space Adam asked for: brain input/output regions, new regions
derived from consistent subsets of the source data, chart-shape structure under
a dynamic margin, and news-sentiment margins -- cycled back as a prediction
metric that decides when to buy low and when it expects to sell high, within a
margin, before volume makes the profit unreachable.

WHY FITNESS IS OUT-OF-SAMPLE ONLY
---------------------------------
Measured 2026-08-27 on 26,270 real ARB-WETH bars: shapes selected for
consistency on a training split scored hit rates of 62-81% in-sample and
**50.0% out-of-sample against a 53.5% majority-class baseline**. Zero skill.
A GA maximising in-sample fitness would breed exactly those, generation over
generation, and hand back a champion with an impressive scorecard and no edge.

So a genome is scored ONLY on data withheld from every step that shaped it, its
score is penalised against the majority-class baseline rather than raw accuracy,
and it must clear that baseline on a walk-forward split before it can be a
champion at all. A genome that cannot beat always-guessing-the-common-class is
worth exactly zero, no matter how good its training numbers look.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from trading.genome.shapes import (
    ShapeCluster,
    best_match,
    cluster_shapes,
    extract_windows,
    select_predictive,
)

# --------------------------------------------------------------------------
# Gene space. Every entry is (low, high) for floats or a list of choices.
# The GA searches ANY subset of this; the UI is expected to expose it so an
# operator can widen, narrow, or pin any gene on a running search.
# --------------------------------------------------------------------------

GENE_SPACE: Dict[str, Any] = {
    # -- chart shape --------------------------------------------------------
    "window": [12, 16, 24, 32, 48, 64],          # bars in the pattern
    "horizon": [3, 6, 12, 24],                    # bars ahead the trade targets
    "shape_points": [8, 12, 16, 24],              # resampling resolution
    "shape_margin": (0.15, 0.75),                 # THE dynamic margin
    "min_occurrences": [6, 8, 12, 20, 30],
    "min_consistency": (0.15, 0.90),
    "min_abs_return": (0.001, 0.02),
    # -- news sentiment -----------------------------------------------------
    "sentiment_weight": (0.0, 1.0),               # 0 = ignore news entirely
    "sentiment_margin": (0.05, 0.60),             # dynamic margin on sentiment
    "sentiment_lookback_h": [6, 12, 24, 48, 72],
    # -- brain regions ------------------------------------------------------
    # Which W1z4rD pools carry features and outcomes. New "regions" are
    # discovered by binding a consistent subset of the corpus into an unused
    # pool, so these are genuinely searchable rather than fixed at 1/3.
    "brain_input_pool": [1, 2, 5, 6],
    "brain_outcome_pool": [3, 4, 7, 8],
    "brain_weight": (0.0, 1.0),                   # 0 = shapes only, no brain
    "brain_min_confidence": (0.0, 0.60),
    # -- execution: buy low, sell high, before volume kills the edge --------
    "entry_percentile": (0.05, 0.45),             # how "low" a buy-low must be
    "target_margin": (0.005, 0.08),               # expected sell-high margin
    "stop_margin": (0.005, 0.10),
    "max_hold_bars": [3, 6, 12, 24, 48],
    "min_volume_ratio": (0.0, 2.0),               # skip when volume can't fill
}


def _sample_gene(rng: random.Random, spec: Any) -> Any:
    if isinstance(spec, tuple) and len(spec) == 2:
        return rng.uniform(float(spec[0]), float(spec[1]))
    if isinstance(spec, list):
        return rng.choice(spec)
    return spec


@dataclass
class Genome:
    genes: Dict[str, Any] = field(default_factory=dict)
    genome_id: str = ""
    # Scores are populated by evaluate(); train_* is diagnostic only and must
    # never feed selection.
    fitness: float = 0.0
    oos_accuracy: float = 0.0
    oos_baseline: float = 0.0
    oos_edge: float = 0.0
    oos_expectancy: float = 0.0
    oos_trades: int = 0
    train_accuracy: float = 0.0
    generation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["genes"] = dict(self.genes)
        return d


def random_genome(rng: random.Random, space: Optional[Dict[str, Any]] = None) -> Genome:
    space = space or GENE_SPACE
    genes = {k: _sample_gene(rng, v) for k, v in space.items()}
    return Genome(genes=genes, genome_id="%012x" % rng.getrandbits(48))


def mutate(
    g: Genome,
    rng: random.Random,
    *,
    rate: float = 0.25,
    space: Optional[Dict[str, Any]] = None,
) -> Genome:
    space = space or GENE_SPACE
    genes = dict(g.genes)
    for key, spec in space.items():
        if rng.random() >= rate:
            continue
        if isinstance(spec, tuple):
            lo, hi = float(spec[0]), float(spec[1])
            cur = float(genes.get(key, (lo + hi) / 2))
            # Gaussian step scaled to the range: local search, not a re-roll.
            step = (hi - lo) * 0.15
            genes[key] = float(min(hi, max(lo, rng.gauss(cur, step))))
        else:
            genes[key] = _sample_gene(rng, spec)
    return Genome(genes=genes, genome_id="%012x" % rng.getrandbits(48), generation=g.generation + 1)


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    genes = {k: (a.genes[k] if rng.random() < 0.5 else b.genes.get(k, a.genes[k])) for k in a.genes}
    return Genome(
        genes=genes,
        genome_id="%012x" % rng.getrandbits(48),
        generation=max(a.generation, b.generation) + 1,
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def _sentiment_at(sentiment_series: Sequence[Tuple[float, float]], ts: float, lookback_h: float) -> float:
    """Mean sentiment in the lookback window ending at ts. 0.0 when silent."""
    if not sentiment_series:
        return 0.0
    lo = ts - lookback_h * 3600.0
    vals = [s for t, s in sentiment_series if lo <= t <= ts]
    return float(np.mean(vals)) if vals else 0.0


def evaluate(
    genome: Genome,
    bars_by_symbol: Dict[str, Sequence[Dict[str, Any]]],
    *,
    split: float = 0.7,
    sentiment: Optional[Dict[str, Sequence[Tuple[float, float]]]] = None,
    brain_predict: Optional[Callable[[str, float, float], Tuple[Optional[str], float]]] = None,
    min_oos_trades: int = 25,
) -> Genome:
    """Score a genome on data withheld from everything that shaped it.

    Shapes are discovered on the training slice only, then applied unchanged
    to the held-out slice. ``fitness`` is edge over the majority-class
    baseline, so a genome that merely learns "this market mostly fell" scores
    zero.
    """
    g = genome.genes
    window = int(g.get("window", 24))
    horizon = int(g.get("horizon", 6))
    margin = float(g.get("shape_margin", 0.35))
    sent_w = float(g.get("sentiment_weight", 0.0))
    sent_margin = float(g.get("sentiment_margin", 0.2))
    sent_lb = float(g.get("sentiment_lookback_h", 24))
    brain_w = float(g.get("brain_weight", 0.0))
    brain_min_conf = float(g.get("brain_min_confidence", 0.0))
    target = float(g.get("target_margin", 0.02))
    stop = float(g.get("stop_margin", 0.02))

    clusters: List[ShapeCluster] = []
    test_sets: List[Tuple[str, List[Tuple[np.ndarray, float]], List[float]]] = []

    for symbol, bars in bars_by_symbol.items():
        if len(bars) < (window + horizon) * 4:
            continue
        cut = int(len(bars) * split)
        train, test = bars[:cut], bars[cut:]
        tr = extract_windows(train, window=window, horizon=horizon, stride=3)
        if tr:
            clusters.extend(
                cluster_shapes(tr, margin=margin, symbol=symbol, max_clusters=300)
            )
        te = extract_windows(test, window=window, horizon=horizon, stride=1)
        ts_list = [float(b.get("timestamp", 0.0) or 0.0) for b in test]
        test_sets.append((symbol, te, ts_list))

    keep = select_predictive(
        clusters,
        min_occurrences=int(g.get("min_occurrences", 8)),
        min_consistency=float(g.get("min_consistency", 0.35)),
        min_abs_return=float(g.get("min_abs_return", 0.002)),
    )
    if not keep:
        genome.fitness = 0.0
        return genome

    hits = total = 0
    returns: List[float] = []
    all_forward: List[float] = []

    for symbol, te, ts_list in test_sets:
        series = (sentiment or {}).get(symbol, [])
        for idx, (sig, fwd) in enumerate(te):
            all_forward.append(fwd)
            m = best_match(sig, keep, margin=margin)
            if m is None:
                continue
            cluster, _dist = m
            score = cluster.mean_return

            # News sentiment, as a dynamic-margin gate on the shape signal.
            if sent_w > 0.0 and series:
                ts = ts_list[idx] if idx < len(ts_list) else 0.0
                s = _sentiment_at(series, ts, sent_lb)
                if abs(s) >= sent_margin and (s > 0) != (score > 0):
                    continue          # news contradicts the shape -> stand down
                score = score * (1.0 - sent_w) + s * sent_w * abs(score)

            # Brain read, cycled back in as a prediction metric.
            if brain_w > 0.0 and brain_predict is not None:
                try:
                    ans, conf = brain_predict(symbol, float(idx), float(score))
                except Exception:
                    ans, conf = None, 0.0
                if conf >= brain_min_conf and ans:
                    up = ans in ("win", "win_big")
                    score = score * (1.0 - brain_w) + (abs(score) if up else -abs(score)) * brain_w

            if abs(score) < 1e-9:
                continue
            pred_up = score > 0
            total += 1
            if pred_up == (fwd > 0):
                hits += 1
            # Realised return, bounded by the genome's own target/stop.
            realised = fwd if pred_up else -fwd
            returns.append(max(-stop, min(target, realised)))

    genome.oos_trades = total
    if total < min_oos_trades or not all_forward:
        genome.fitness = 0.0
        return genome

    acc = hits / total
    up_rate = sum(1 for f in all_forward if f > 0) / len(all_forward)
    baseline = max(up_rate, 1.0 - up_rate)
    expectancy = float(np.mean(returns)) if returns else 0.0

    genome.oos_accuracy = acc
    genome.oos_baseline = baseline
    genome.oos_edge = acc - baseline
    genome.oos_expectancy = expectancy

    # Fitness: edge over baseline, only when expectancy is positive, with a
    # sample-size discount so a lucky handful of trades cannot win the search.
    if genome.oos_edge <= 0.0 or expectancy <= 0.0:
        genome.fitness = 0.0
    else:
        confidence = math.sqrt(min(1.0, total / 200.0))
        genome.fitness = genome.oos_edge * expectancy * 1000.0 * confidence
    return genome


def evolve(
    bars_by_symbol: Dict[str, Sequence[Dict[str, Any]]],
    *,
    population: int = 24,
    generations: int = 8,
    elite: int = 4,
    seed: int = 0,
    space: Optional[Dict[str, Any]] = None,
    sentiment: Optional[Dict[str, Sequence[Tuple[float, float]]]] = None,
    brain_predict: Optional[Callable[..., Any]] = None,
    on_generation: Optional[Callable[[int, List[Genome]], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> List[Genome]:
    """Run the search. ``on_generation`` streams progress to a UI/service;
    ``should_stop`` lets a running search be cancelled or reconfigured live."""
    rng = random.Random(seed or int(time.time()))
    space = space or GENE_SPACE
    pop = [random_genome(rng, space) for _ in range(population)]

    for gen in range(generations):
        if should_stop and should_stop():
            break
        for genome in pop:
            if genome.fitness == 0.0 and genome.oos_trades == 0:
                evaluate(
                    genome, bars_by_symbol,
                    sentiment=sentiment, brain_predict=brain_predict,
                )
        pop.sort(key=lambda x: x.fitness, reverse=True)
        if on_generation:
            on_generation(gen, list(pop))
        survivors = pop[:max(1, elite)]
        children: List[Genome] = list(survivors)
        while len(children) < population:
            a = rng.choice(survivors)
            b = rng.choice(pop[: max(2, population // 2)])
            child = crossover(a, b, rng) if rng.random() < 0.6 else Genome(
                genes=dict(a.genes), genome_id="%012x" % rng.getrandbits(48),
                generation=a.generation + 1,
            )
            children.append(mutate(child, rng, space=space))
        pop = children

    for genome in pop:
        if genome.oos_trades == 0:
            evaluate(genome, bars_by_symbol, sentiment=sentiment, brain_predict=brain_predict)
    pop.sort(key=lambda x: x.fitness, reverse=True)
    return pop
