"""A genetic search for trading rules, with attention that abandons dead ends.

WHAT IS BEING EVOLVED
---------------------
A genome is a conjunction of conditions over the features in
``features.py`` -- "notional above 1.13 AND the chaos character is
mean-reverting AND fewer than 9 ticks per hour". That is exactly the shape of
a ``Theorem``, so anything this finds can be handed straight to the existing
falsification machinery rather than living in a parallel universe of its own.

Conjunctions only, deliberately. A rule with ORs in it can always be split
into two simpler rules, and two simple rules that each survive a holdout are
worth more than one compound rule that survives once -- they can be traded,
sized and refuted independently.

WHY FITNESS IS NOT PROFIT
-------------------------
Selecting on realised profit finds the luckiest genome, not the best one. On
156 round trips the top few outcomes dominate any sum, so a search that
maximises total return converges on whichever rule happened to contain them --
which is precisely the failure this project already measured, where dropping
three trades out of 85 flipped the ghost book from +6.05% to -1.13%.

Fitness here is the HOLDOUT t-statistic of excess return over cost, penalised
for thinness. That asks the only question worth asking -- is this
distinguishable from noise on data it was not fitted to -- and a rule that
scores well on it has already survived what killed the others.

SELF-ATTENTION
--------------
Each lineage tracks whether its recent generations improved. A lineage whose
best fitness has not moved in ``STALE_GENERATIONS`` is exhausted: continuing
to mutate it spends the whole budget polishing a local optimum. Attention
then shifts -- the stale lineage is retired and its slots are refilled by
branching from a DIFFERENT region of feature space, biased toward features
the retired lineage never used.

That is what keeps the search from magnetising to one shape of answer. The
attention is over FEATURES rather than over genomes, because a lineage that
is stuck is usually stuck on a feature, not on a threshold.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Conditions in one genome. More than this and a rule stops being a claim
#: and becomes a description of the sample it was fitted to.
MAX_CONDITIONS = 3

#: Trades a rule must select on BOTH sides of the split to be judged at all.
MIN_SELECTED = 8

#: Generations without improvement before a lineage is declared exhausted.
STALE_GENERATIONS = 4

#: Fraction of the data fitted on. The rest is never touched until judgment.
FIT_FRACTION = 0.6


class Condition:
    """One clause: a feature, a comparison, a threshold."""

    def __init__(self, feature: str, op: str, value: Any) -> None:
        self.feature = feature
        self.op = op
        self.value = value

    def holds(self, row: Dict[str, Any]) -> Optional[bool]:
        """True, False, or None when this row cannot answer.

        None matters: a row missing the feature is not evidence against the
        rule, and counting it as False would make every rule look better on
        symbols with sparse data.
        """
        actual = row.get(self.feature)
        if actual is None:
            return None
        try:
            if self.op == ">":
                return float(actual) > float(self.value)
            if self.op == "<":
                return float(actual) < float(self.value)
            if self.op == "==":
                return str(actual) == str(self.value)
            if self.op == "!=":
                return str(actual) != str(self.value)
        except (TypeError, ValueError):
            return None
        return None

    def describe(self) -> str:
        if isinstance(self.value, float):
            return f"{self.feature} {self.op} {self.value:.6g}"
        return f"{self.feature} {self.op} {self.value}"


class Genome:
    """A conjunction of conditions, plus its lineage and its scores."""

    def __init__(self, conditions: Sequence[Condition], lineage: str) -> None:
        self.conditions = list(conditions)
        self.lineage = lineage
        self.fit_score: Optional[float] = None
        self.holdout_score: Optional[float] = None
        self.n_selected = 0
        self.detail = ""

    def selects(self, row: Dict[str, Any]) -> Optional[bool]:
        """Whether every condition holds. None if any cannot be answered."""
        for condition in self.conditions:
            verdict = condition.holds(row)
            if verdict is None:
                return None
            if not verdict:
                return False
        return True

    def features_used(self) -> set:
        return {c.feature for c in self.conditions}

    def describe(self) -> str:
        return " AND ".join(c.describe() for c in self.conditions)


def _excess_returns(rows: Sequence[Dict[str, Any]], cost: float
                    ) -> Tuple[List[float], List[float]]:
    """Excess return over cost, split into selected and not-selected."""
    return ([], [])


def _score(genome: Genome, rows: Sequence[Dict[str, Any]], cost: float
           ) -> Tuple[Optional[float], int, str]:
    """Welch t of the selected group's excess return against the rest.

    Returns (t, n_selected, detail). None when the rule cannot be judged --
    too few rows on either side, or no dispersion to measure against.
    """
    inside: List[float] = []
    outside: List[float] = []
    for row in rows:
        verdict = genome.selects(row)
        if verdict is None:
            continue                     # unanswerable, not evidence
        value = row.get("return")
        if not isinstance(value, (int, float)):
            continue
        excess = float(value) - float(cost)
        (inside if verdict else outside).append(excess)

    if len(inside) < MIN_SELECTED or len(outside) < MIN_SELECTED:
        return None, len(inside), (
            f"selects {len(inside)} of {len(inside) + len(outside)}; "
            f"needs {MIN_SELECTED} on each side to be judged")

    mean_in, mean_out = statistics.mean(inside), statistics.mean(outside)
    try:
        var_in = statistics.variance(inside)
        var_out = statistics.variance(outside)
    except statistics.StatisticsError:
        return None, len(inside), "no dispersion to measure against"

    denom = math.sqrt(var_in / len(inside) + var_out / len(outside))
    if denom <= 0:
        return None, len(inside), "zero standard error"

    t = (mean_in - mean_out) / denom
    return t, len(inside), (
        f"selected {len(inside)} at mean excess {mean_in:+.5f} vs "
        f"{mean_out:+.5f} for the other {len(outside)} (t={t:+.2f})")


def _random_condition(vocab: Dict[str, Dict[str, Any]],
                      rng: random.Random,
                      prefer: Optional[set] = None) -> Optional[Condition]:
    """Draw one condition, optionally biased toward unexplored features."""
    names = list(vocab)
    if not names:
        return None
    if prefer:
        weighted = [n for n in names if n in prefer] * 3 + names
        names = weighted
    name = rng.choice(names)
    info = vocab[name]

    if info.get("kind") == "numeric":
        quartiles = info.get("quartiles") or []
        if not quartiles:
            return None
        # Thresholds drawn from the data's own quartiles: a threshold outside
        # the observed range produces a rule that is always or never true.
        return Condition(name, rng.choice([">", "<"]), float(rng.choice(quartiles)))

    values = info.get("values") or []
    if not values:
        return None
    return Condition(name, rng.choice(["==", "!="]), rng.choice(values))


def _mutate(genome: Genome, vocab: Dict[str, Dict[str, Any]],
            rng: random.Random, prefer: Optional[set] = None) -> Genome:
    conditions = list(genome.conditions)
    roll = rng.random()

    if roll < 0.4 and len(conditions) < MAX_CONDITIONS:
        new = _random_condition(vocab, rng, prefer)
        if new:
            conditions.append(new)
    elif roll < 0.7 and len(conditions) > 1:
        conditions.pop(rng.randrange(len(conditions)))
    elif conditions:
        index = rng.randrange(len(conditions))
        replacement = _random_condition(vocab, rng, prefer)
        if replacement:
            conditions[index] = replacement

    return Genome(conditions or genome.conditions, genome.lineage)


class Lineage:
    """A population branch, and whether it is still going anywhere."""

    def __init__(self, name: str, focus: Optional[set] = None) -> None:
        self.name = name
        self.focus = set(focus or ())
        self.best: Optional[float] = None
        self.generations_without_gain = 0
        self.retired = False

    def observe(self, best_this_generation: Optional[float]) -> None:
        if best_this_generation is None:
            self.generations_without_gain += 1
            return
        if self.best is None or best_this_generation > self.best + 1e-6:
            self.best = best_this_generation
            self.generations_without_gain = 0
        else:
            self.generations_without_gain += 1

    @property
    def stale(self) -> bool:
        return self.generations_without_gain >= STALE_GENERATIONS


def publish(result: Dict[str, Any], *, min_t: float = 2.0) -> Dict[str, Any]:
    """Write the survivors where the strategy registry can pick them up.

    A rule only becomes tradeable by passing through here, and only if it
    cleared ``min_t`` on data it was never fitted to. Publishing is
    deliberately a separate step from searching: a search that automatically
    armed whatever it found would put every random fluctuation in front of
    real money.

    Rules already published are replaced by name, so a rule that stops
    clearing the bar disappears rather than lingering on an old statistic.
    """
    import json
    from pathlib import Path

    if not result.get("ok"):
        return {"published": 0, "reason": result.get("reason", "search failed")}

    keep = [s for s in (result.get("survivors") or [])
            if float(s.get("holdout_t") or 0.0) >= min_t]

    path = Path(__file__).resolve().parents[2] / "data" / "discovered_rules.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(keep, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return {"published": 0, "reason": f"{type(exc).__name__}: {exc}"}

    return {
        "published": len(keep),
        "path": str(path),
        "rules": [{"rule": r["rule"], "holdout_t": r["holdout_t"],
                   "mean_excess": r.get("holdout_mean_excess")} for r in keep],
    }


def evolve(rows: Sequence[Dict[str, Any]], *, cost: float = 0.0065,
           generations: int = 12, population: int = 24,
           seed: int = 20260905,
           advisor: Optional[Callable[[Dict[str, Any]], Optional[Sequence[str]]]] = None,
           ) -> Dict[str, Any]:
    """Search for rules that separate returns, judged only on held-out data.

    ``advisor`` is the trading agent, asked ONLY when the search has exhausted
    its own remedy -- every feature explored and a lineage still stalled. It
    receives what stalled and what has been tried, and may return feature
    names to focus on. Anything it returns that is not a real feature is
    ignored: advice is a hint, not an instruction, and a search that could be
    steered into nonsense by a bad suggestion would be worse than one with no
    advisor at all.

    Returns the survivors plus a full account of which lineages went stale and
    what attention moved to -- a search that reports only its winner cannot be
    audited, and an unauditable search is indistinguishable from overfitting.
    """
    from .features import feature_vocabulary

    usable = [r for r in rows if isinstance(r.get("return"), (int, float))]
    if len(usable) < MIN_SELECTED * 4:
        return {
            "ok": False,
            "reason": f"{len(usable)} closed round trips; need "
                      f"{MIN_SELECTED * 4} to fit and hold out",
            "survivors": [],
        }

    rng = random.Random(seed)
    split = int(len(usable) * FIT_FRACTION)
    fit_rows, holdout_rows = usable[:split], usable[split:]
    vocab = feature_vocabulary(usable)
    if not vocab:
        return {"ok": False, "reason": "no usable features", "survivors": []}

    all_features = set(vocab)
    lineages = [Lineage("alpha"), Lineage("beta"), Lineage("gamma")]
    events: List[str] = []

    populations: Dict[str, List[Genome]] = {}
    for lineage in lineages:
        populations[lineage.name] = [
            Genome([c for c in [_random_condition(vocab, rng)] if c], lineage.name)
            for _ in range(population // len(lineages))
        ]

    for generation in range(generations):
        for lineage in lineages:
            if lineage.retired:
                continue
            members = populations[lineage.name]

            scored: List[Tuple[float, Genome]] = []
            for genome in members:
                if not genome.conditions:
                    continue
                t, n, detail = _score(genome, fit_rows, cost)
                genome.fit_score = t
                genome.n_selected = n
                genome.detail = detail
                if t is not None:
                    scored.append((t, genome))

            scored.sort(key=lambda pair: pair[0], reverse=True)
            best = scored[0][0] if scored else None
            lineage.observe(best)

            # SELF-ATTENTION. A lineage that has stopped improving is spending
            # the budget polishing a local optimum; retire it and put its
            # slots behind features nothing has explored.
            if lineage.stale and not lineage.retired:
                lineage.retired = True
                explored = set()
                for genome in members:
                    explored |= genome.features_used()
                unexplored = all_features - explored
                focus = unexplored or all_features

                # THE SEARCH ASKS FOR HELP ONLY AFTER ITS OWN MOVE FAILS.
                #
                # Shifting attention to unexplored features is the search's
                # own remedy and it usually works. But when there is nothing
                # left unexplored, the shift is a no-op dressed as progress:
                # the replacement lineage draws from the same features that
                # just stalled, and stalls the same way.
                #
                # That -- and only that -- is when the agent is worth asking.
                # It can see things the search cannot: which symbols are
                # newly listed, what the news says, which strategies the
                # ledger is starving, whether the whole feature set is stale
                # because the market changed. Escalating on every stall would
                # spend the agent's attention on problems the search can
                # solve itself.
                if not unexplored and advisor is not None:
                    try:
                        hint = advisor({
                            "stalled_lineage": lineage.name,
                            "best_fit_t": lineage.best,
                            "generations_without_gain": lineage.generations_without_gain,
                            "features_exhausted": sorted(all_features),
                            "generation": generation,
                        })
                    except Exception as exc:  # noqa: BLE001 - advice is optional
                        hint = None
                        events.append(f"gen {generation}: advisor raised {exc!r}")
                    if hint:
                        suggested = {f for f in hint if f in all_features}
                        if suggested:
                            focus = suggested
                            events.append(
                                f"gen {generation}: every feature explored and "
                                f"{lineage.name} still stalled; the agent "
                                f"redirected attention to {sorted(suggested)}")

                replacement = Lineage(f"{lineage.name}'", focus)
                lineages.append(replacement)
                populations[replacement.name] = [
                    Genome([c for c in [_random_condition(vocab, rng, focus)] if c],
                           replacement.name)
                    for _ in range(max(4, population // 4))
                ]
                events.append(
                    f"gen {generation}: lineage {lineage.name} stalled at "
                    f"t={lineage.best if lineage.best is None else round(lineage.best, 3)} "
                    f"after {STALE_GENERATIONS} generations; attention moved to "
                    f"{sorted(focus)[:4]}")
                continue

            # Breed from the top half, mutating toward this lineage's focus.
            survivors = [g for _, g in scored[:max(2, len(scored) // 2)]]
            if not survivors:
                survivors = [g for g in members if g.conditions][:2]
            children: List[Genome] = list(survivors)
            while len(children) < len(members):
                parent = rng.choice(survivors) if survivors else None
                if parent is None:
                    break
                children.append(_mutate(parent, vocab, rng, lineage.focus or None))
            populations[lineage.name] = children

    # JUDGMENT ON HELD-OUT DATA ONLY. Everything above fitted; nothing above
    # has seen these rows.
    survivors: List[Dict[str, Any]] = []
    seen: set = set()
    for lineage in lineages:
        for genome in populations.get(lineage.name, []):
            if not genome.conditions:
                continue
            description = genome.describe()
            if description in seen:
                continue
            seen.add(description)
            t, n, detail = _score(genome, holdout_rows, cost)
            if t is None or t <= 1.7:
                continue
            # The mean excess the selected group actually earned, which is
            # what a strategy sizes against. A t-statistic says a difference
            # exists; it does not say how big.
            inside = [float(r["return"]) - cost for r in holdout_rows
                      if genome.selects(r) is True
                      and isinstance(r.get("return"), (int, float))]
            mean_excess = statistics.mean(inside) if inside else 0.0

            survivors.append({
                "id": description,
                "rule": description,
                # Machine-readable, so the strategy layer never has to parse
                # the description back into predicates.
                "conditions": [
                    {"feature": c.feature, "op": c.op, "value": c.value}
                    for c in genome.conditions
                ],
                "lineage": genome.lineage,
                "holdout_t": round(t, 3),
                "holdout_selected": n,
                "holdout_mean_excess": round(mean_excess, 6),
                "fit_t": None if genome.fit_score is None else round(genome.fit_score, 3),
                "detail": detail,
                "found_at": time.time(),
            })

    survivors.sort(key=lambda s: s["holdout_t"], reverse=True)
    return {
        "ok": True,
        "trades": len(usable),
        "fit_rows": len(fit_rows),
        "holdout_rows": len(holdout_rows),
        "features": sorted(vocab),
        "lineages": [
            {"name": l.name, "retired": l.retired,
             "best_fit_t": None if l.best is None else round(l.best, 3),
             "focus": sorted(l.focus)[:6]}
            for l in lineages
        ],
        "attention_events": events,
        "survivors": survivors[:10],
    }
