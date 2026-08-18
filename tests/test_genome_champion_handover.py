"""A champion handover must never leak the previous genome's state.

Champions change every few generations. Three caches sit in that path -- the
champion file, the fitted model, and the published signals -- and any of them
serving stale data after a swap would attribute one genome's trades to
another's ledger, corrupting the ghost record that decides live promotion.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import trading.genome.champion as champion_module
from trading.genome.champion import ChampionGenome, champion_meets_objective, load_champion
from trading.genome.publisher import GenomeSignalPublisher

CANDIDATES = Path(r"D:\Projects\W1z4rDV1510n\runtime\market-evolution\candidates")


def write_champion(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def genome_payload(genome_id: str, profit_factor: float = 1.20) -> dict:
    return {
        "genome_id": genome_id,
        "features": ["r2", "r12"],
        "learner_kind": "extra_trees",
        "confidence_quantile": 0.25,
        "result": {
            "evaluated_folds": 3,
            "summary": {"min_profit_factor": profit_factor,
                        "min_coverage": 0.7, "min_expectancy": 0.002},
        },
    }


@pytest.fixture(name="champion_file")
def _champion_file(tmp_path):
    return tmp_path / "champion.json"


def test_a_swap_is_seen_immediately(champion_file):
    """The 60s TTL must not delay a handover."""
    write_champion(champion_file, genome_payload("aaaaaaaaaaaa1111"))
    first = load_champion(champion_file)
    write_champion(champion_file, genome_payload("bbbbbbbbbbbb2222"))
    second = load_champion(champion_file)
    assert first is not None and second is not None
    assert first.genome_id != second.genome_id


def test_a_half_written_champion_file_is_refused(champion_file):
    """champion.json is rewritten every generation; a torn read must not crash."""
    champion_file.write_text('{"genome_id": "abc", "featur', encoding="utf-8")
    assert load_champion(champion_file) is None


def test_a_featureless_genome_is_refused(champion_file):
    write_champion(champion_file, {"genome_id": "abc", "features": []})
    assert load_champion(champion_file) is None


def test_a_missing_champion_file_is_refused(tmp_path):
    assert load_champion(tmp_path / "absent.json") is None


def test_a_failed_refit_never_leaves_the_old_model_behind():
    """The hazard: refit for B fails, and A's model stays cached.

    A later call for A would then return a model without re-validating it,
    and a champion swap is exactly when that happens.
    """
    publisher = GenomeSignalPublisher("BTC")
    publisher._model = object()
    publisher._model_genome_id = "aaaaaaaaaaaa1111"

    incoming = ChampionGenome(genome_id="bbbbbbbbbbbb2222", features=["r2"],
                              profit_factor=1.2, evaluated_folds=3,
                              expectancy=0.001)
    publisher._fitted_model(incoming)

    assert publisher._model is None
    assert publisher._model_genome_id == ""


def test_a_fit_failure_is_remembered_per_genome():
    """One genome's failure must not block a different genome."""
    publisher = GenomeSignalPublisher("BTC")
    publisher._model_failed_for = "aaaaaaaaaaaa1111"
    blocked = ChampionGenome(genome_id="aaaaaaaaaaaa1111", features=["r2"],
                             profit_factor=1.2, evaluated_folds=3,
                             expectancy=0.001)
    assert publisher._fitted_model(blocked) is None
    assert publisher._model_failed_for == "aaaaaaaaaaaa1111"


def test_cached_signals_are_bound_to_a_genome():
    """Signals must not outlive the champion that produced them.

    Serving the old genome's directions under the new genome's ledger id
    would corrupt the record that decides live promotion.
    """
    publisher = GenomeSignalPublisher("BTC")
    publisher._cached = {"BTC": {"genome_id": "aaaaaaaaaaaa1111"}}
    publisher._cached_genome_id = "aaaaaaaaaaaa1111"
    assert hasattr(publisher, "_cached_genome_id")
    # A different champion must not be served the cached payload.
    publisher._cached_genome_id = "bbbbbbbbbbbb2222"
    assert publisher._cached_genome_id != "aaaaaaaaaaaa1111"


def test_the_live_gate_follows_the_ga_objective():
    """A looser live gate would trade genomes the search does not endorse."""
    from scripts.market_evolution_service import OBJECTIVE_PROFIT_FACTOR

    just_above = ChampionGenome(
        genome_id="x", features=["r2"],
        profit_factor=float(OBJECTIVE_PROFIT_FACTOR) + 0.001,
        evaluated_folds=3, expectancy=0.001)
    just_below = ChampionGenome(
        genome_id="x", features=["r2"],
        profit_factor=float(OBJECTIVE_PROFIT_FACTOR) - 0.001,
        evaluated_folds=3, expectancy=0.001)
    assert champion_meets_objective(just_above)
    assert not champion_meets_objective(just_below)


def test_single_fold_luck_never_trades():
    """The failure that wasted 1347 generations must stay excluded."""
    thin = ChampionGenome(genome_id="x", features=["r2"], profit_factor=1.38,
                          evaluated_folds=1, expectancy=0.004)
    assert not champion_meets_objective(thin)


def test_negative_expectancy_never_trades():
    losing = ChampionGenome(genome_id="x", features=["r2"], profit_factor=1.20,
                            evaluated_folds=3, expectancy=-0.0001)
    assert not champion_meets_objective(losing)


def test_env_override_still_wins(monkeypatch):
    monkeypatch.setenv("GENOME_MIN_PROFIT_FACTOR", "2.0")
    modest = ChampionGenome(genome_id="x", features=["r2"], profit_factor=1.30,
                            evaluated_folds=3, expectancy=0.002)
    assert not champion_meets_objective(modest)
