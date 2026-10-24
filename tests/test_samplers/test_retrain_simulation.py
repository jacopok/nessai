# -*- coding: utf-8 -*-
"""Tests of the retrain decision on simulated runs.

The problems are shortened versions of those in
:py:data:`nessai.samplers.retrain_simulation.PROBLEMS`; the full study with
more seeds is in the study's ``retrain_sim_study.py``.
"""

import numpy as np
import pytest

from nessai.samplers.retrain_simulation import (
    PROBLEMS,
    SimulatedProblem,
    _piecewise,
    simulate,
)

OLD = dict(lineage_exponent=None, reset_on_drop=None, reset_rollback=False)


def mean_cost(problem, policy="decision", seeds=(0, 1), **kwargs):
    runs = [simulate(problem, policy, seed=s, **kwargs) for s in seeds]
    return float(np.mean([r.total for r in runs])), runs


def short(name, **kwargs):
    p = PROBLEMS[name]
    scale = kwargs.pop("scale", 0.5)
    fields = dict(iterations=int(p.iterations * scale))
    if p.breaks:
        fields["breaks"] = tuple((int(i * scale), d) for i, d in p.breaks)
    if p.shocks:
        fields["shocks"] = tuple((int(i * scale), d) for i, d in p.shocks)
    return p.with_(**fields, **kwargs)


def test_piecewise_rate():
    assert _piecewise(-0.5, 2000, 1000) == pytest.approx(-1.0)
    spec = ((1000, -1.0), (3000, -0.1))
    assert _piecewise(spec, 500, 1000) == pytest.approx(-0.5)
    assert _piecewise(spec, 2000, 1000) == pytest.approx(-1.1)
    assert _piecewise(spec, 4000, 1000) == pytest.approx(-1.3)


def test_simulation_is_deterministic():
    p = short("lvk-bbh", scale=0.2)
    a = simulate(p, seed=3)
    b = simulate(p, seed=3)
    assert a.total == b.total
    assert a.trainings == b.trainings


def test_every_pool_trains_at_every_pool():
    r = simulate(short("gaussian", scale=0.2), "every-pool", seed=0)
    assert r.trainings == r.pools
    assert r.iterations == short("gaussian", scale=0.2).iterations


def test_interval_policy():
    p = short("lvk-bbh", scale=0.2)
    few = simulate(p, "interval", seed=0, interval=1000)
    many = simulate(p, "interval", seed=0, interval=20)
    assert many.trainings > 2 * few.trainings


def test_unknown_policy():
    with pytest.raises(ValueError, match="Unknown policy"):
        simulate(SimulatedProblem("x"), "sometimes")


def test_measured_costs_match_exact():
    """The decision measures the simulated costs exactly, so measuring or
    providing them gives the same schedule."""
    p = short("lvk-bbh", scale=0.3)
    exact = simulate(p, seed=1)
    measured = simulate(p, seed=1, costs="measured")
    assert measured.trainings == pytest.approx(exact.trainings, rel=0.2)
    assert measured.total == pytest.approx(exact.total, rel=0.05)


@pytest.mark.parametrize(
    "name", ["gaussian", "expensive-training", "expensive-likelihood"]
)
def test_not_worse_than_before(name):
    """Where the lineage term barely matters, the decision costs what it
    did, and less than retraining at every pool."""
    p = short(name, scale=1.0)
    seeds = (0, 1, 2, 3)
    new, _ = mean_cost(p, seeds=seeds)
    old, _ = mean_cost(p, seeds=seeds, decision_kwargs=OLD)
    every, _ = mean_cost(p, "every-pool", seeds=seeds)
    assert new <= 1.05 * old
    assert new < every


def test_expensive_training_trains_rarely():
    p = short("expensive-training")
    _, runs = mean_cost(p)
    _, every = mean_cost(p, "every-pool")
    assert np.mean([r.trainings for r in runs]) < 0.5 * np.mean(
        [r.trainings for r in every]
    )


def test_lineage_term_helps_with_a_lineage_effect():
    """The gain comes late in the run, when the flows have many trainings
    and each iteration costs more, so the full run is simulated."""
    p = PROBLEMS["lvk-bbh"]
    new, runs = mean_cost(p, seeds=(0, 1, 2))
    old, old_runs = mean_cost(p, seeds=(0, 1, 2), decision_kwargs=OLD)
    assert new < old
    assert np.mean([r.trainings for r in runs]) > np.mean(
        [r.trainings for r in old_runs]
    )


def test_true_exponents_approach_the_best_interval():
    """Given the problem's lineage exponents, the decision comes within 15%
    of the best fixed interval in hindsight."""
    p = short("lvk-strong-lineage")
    b = dict(likelihood=0.5, population=0.8)
    new, _ = mean_cost(p, decision_kwargs=dict(lineage_exponent=b))
    best = min(
        mean_cost(p, "interval", interval=i)[0] for i in (50, 100, 200, 500)
    )
    assert new < 1.15 * best


def test_lineage_break_is_reset():
    p = short("lineage-break")
    new, runs = mean_cost(p)
    no_drop, _ = mean_cost(p, decision_kwargs=dict(reset_on_drop=None))
    assert new < 0.5 * no_drop
    assert all(r.resets >= 1 and r.discarded == 0 for r in runs)


def test_shock_reset_is_discarded():
    """When every flow gets worse a reset does not help: it is discarded
    and costs little."""
    p = short("shock")
    seeds = (0, 1, 2)
    new, runs = mean_cost(p, seeds=seeds)
    no_drop, _ = mean_cost(
        p, seeds=seeds, decision_kwargs=dict(reset_on_drop=None)
    )
    assert sum(r.discarded for r in runs) >= 2
    assert new < 1.1 * no_drop


def test_no_resets_on_noisy_trainings():
    _, runs = mean_cost(short("noisy-training"), seeds=(0, 1, 2))
    assert all(r.resets == 0 for r in runs)


def test_no_resets_on_a_fast_transition():
    _, runs = mean_cost(short("early-transition"), seeds=(0, 1, 2))
    assert all(r.resets == 0 for r in runs)


@pytest.mark.parametrize("scale", [0.3, 3.0])
def test_robust_to_the_remaining_estimate(scale):
    p = short("lvk-bbh")
    exact, _ = mean_cost(p)
    off, _ = mean_cost(p, remaining_scale=scale)
    assert off < 1.1 * exact
