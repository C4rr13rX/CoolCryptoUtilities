"""A buy-omen lift over every-bar-buy is selection, until it beats selection.

Pass 114 measured AERO-USDC in an UP window: 52 buy omens paid +1.4887% per
trade against +0.5068% for buying every bar. Read as two independent means
that is a 0.98pp edge. It is not -- the 52 are a SUBSET of the same 400 bars,
the per-bar sd is 5.26%, and a random 52 of those bars clears +1.4887% about
7% of the time. Before this test the harness printed the two means side by
side with nothing to tell a reader which it was, and "omens beat buy-and-hold"
is exactly the shape of every fake edge this repo has paid for.

The failure this prevents: quoting a selection artifact as an edge.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.omen_experiment import subset_lift_pvalue


def test_a_random_subset_is_reported_as_noise_not_edge():
    """Picking k bars at random must NOT look like an edge."""
    # Alternating returns: no structure to find, mean 0.0.
    net = [0.05 if i % 2 else -0.05 for i in range(400)]
    # A subset that happens to be mean-zero, i.e. exactly the null.
    out = subset_lift_pvalue(net, 52, observed=0.0)
    assert out["p_value"] > 0.05, (
        f"a mean-zero subset of a mean-zero population was called an edge: "
        f"p={out['p_value']}")
    assert abs(out["z"]) < 2.0


def test_a_genuinely_selective_subset_beats_the_null():
    """A subset that really did pick the winners must clear the null.

    Without this the test above could be satisfied by always returning p=1.0.
    """
    net = [0.05 if i % 2 else -0.05 for i in range(400)]
    # Observed = picked every winner. No random 52-subset can match that.
    out = subset_lift_pvalue(net, 52, observed=0.05)
    assert out["p_value"] < 0.01, (
        f"perfect selection was not distinguished from random: "
        f"p={out['p_value']}")
    assert out["z"] > 2.0


def test_the_null_se_shrinks_as_the_trade_count_grows():
    """The whole point: more trades, tighter null. A 1-trade cell proves nothing.

    This is the arithmetic that makes an n=1 per-trade mean unquotable.
    """
    net = [((i * 37) % 101 - 50) / 1000.0 for i in range(400)]
    few = subset_lift_pvalue(net, 4, observed=0.0)
    many = subset_lift_pvalue(net, 200, observed=0.0)
    assert few["null_se"] > many["null_se"] * 3, (
        f"null SE did not shrink with sample size: "
        f"k=4 {few['null_se']} vs k=200 {many['null_se']}")


def test_it_is_seeded_so_the_same_run_gives_the_same_p_value():
    """An unreproducible p-value is not evidence."""
    net = [((i * 13) % 97 - 48) / 1000.0 for i in range(300)]
    a = subset_lift_pvalue(net, 40, observed=0.004)
    b = subset_lift_pvalue(net, 40, observed=0.004)
    assert a["p_value"] == b["p_value"]
    assert a["null_se"] == b["null_se"]


def test_degenerate_inputs_do_not_manufacture_significance():
    """k larger than the population, or zero trades, must return p=1.0."""
    net = [0.01, -0.02, 0.03]
    assert subset_lift_pvalue(net, 0, observed=1.0)["p_value"] == 1.0
    assert subset_lift_pvalue(net, 99, observed=1.0)["p_value"] == 1.0
    assert subset_lift_pvalue([], 5, observed=1.0)["p_value"] == 1.0


def test_the_pass_114_aero_up_window_lift_is_inside_the_noise():
    """The measurement that motivated this, reproduced as a regression.

    A synthetic population matched to the UP window's dispersion: a 0.98pp
    lift on 52 trades against a ~5% per-bar sd cannot be significant.
    """
    # 400 bars, mean +0.5%, sd of roughly 5% -- the UP window's shape.
    net = []
    for i in range(400):
        net.append(0.005 + (0.05 if i % 2 else -0.05) * (1 + (i % 7) / 7.0))
    out = subset_lift_pvalue(net, 52, observed=0.005 + 0.0098)
    assert out["p_value"] > 0.05, (
        "a 0.98pp lift on 52 trades at this dispersion must not be called an "
        f"edge, got p={out['p_value']}")
