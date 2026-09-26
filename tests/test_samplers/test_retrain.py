# -*- coding: utf-8 -*-
"""Tests for the cost-based retrain decision."""

import json
import logging
import pickle

import numpy as np
import pytest

from nessai.samplers.retrain import (
    RetrainCostModel,
    RetrainDecision,
    fit_log_acceptance,
    solve_renewal,
)

COSTS = dict(likelihood=1e-3, population=1e-4, training=1e-5)


def geometric_counts(rng, log_a0, slope, n):
    a = np.exp(log_a0 + slope * np.arange(n))
    return rng.geometric(np.clip(a, 1e-6, 1.0))


@pytest.mark.parametrize("y", [1e-6, 1e-2, 1.0, 10.0, 1e4])
def test_solve_renewal(y):
    x = solve_renewal(y)
    assert x >= 1
    assert np.isclose(x * np.log(x) - x + 1, y, rtol=1e-6)


def test_solve_renewal_small():
    """For small y the solution is 1 + sqrt(2 y)."""
    y = 1e-6
    assert np.isclose(solve_renewal(y), 1 + np.sqrt(2 * y), rtol=1e-3)


def test_solve_renewal_non_positive():
    assert solve_renewal(0.0) == 1.0


def test_fit_log_acceptance(rng):
    """The fit should recover the true intercept and slope."""
    nlive = 500
    log_a0, slope = np.log(0.3), -1.0 / nlive
    counts = geometric_counts(rng, log_a0, slope, 2000)
    mean, cov = fit_log_acceptance(
        counts, 25, prior_mean=[0.0, 0.0], prior_cov=np.diag([np.inf] * 2)
    )
    sd = np.sqrt(np.diag(cov))
    assert abs(mean[0] - log_a0) < 4 * sd[0]
    assert abs(mean[1] - slope) < 4 * sd[1]
    assert sd[1] < 0.2 / nlive


def test_fit_log_acceptance_prior_only():
    mean, cov = fit_log_acceptance(
        [], 10, prior_mean=[-1.0, -0.01], prior_cov=np.diag([0.1, 1e-4])
    )
    np.testing.assert_allclose(mean, [-1.0, -0.01])
    np.testing.assert_allclose(cov, np.diag([0.1, 1e-4]))


def test_cost_model_provided_and_measured():
    cost = RetrainCostModel(dict(likelihood=1e-3))
    assert not cost.deterministic
    assert cost["likelihood"] == 1e-3
    assert cost["population"] is None
    cost.record("population", 2.0, 1000)
    assert cost["population"] == pytest.approx(2e-3)
    cost.record("likelihood", 1.0, 100)
    # Provided value takes precedence
    assert cost["likelihood"] == 1e-3
    assert cost.measured("likelihood") == pytest.approx(1e-2)


def test_cost_model_warning(caplog):
    caplog.set_level(logging.WARNING)
    cost = RetrainCostModel(COSTS)
    assert cost.deterministic
    for _ in range(3):
        cost.record("likelihood", 1.0, 100)
    assert "likelihood cost" in caplog.text
    n = len(caplog.records)
    cost.record("likelihood", 1.0, 100)
    # Only warn once
    assert len(caplog.records) == n


def test_cost_model_invalid_key():
    with pytest.raises(ValueError, match="Unknown retrain cost keys"):
        RetrainCostModel(dict(time=1.0))


def test_cost_model_save_load(tmp_path):
    cost = RetrainCostModel(dict(likelihood=1e-3))
    cost.record("likelihood", 1.0, 1000)
    cost.record("population", 1.0, 100)
    cost.record("training", 1.0, 10)
    filename = tmp_path / "costs.json"
    cost.save(filename)
    with open(filename) as f:
        data = json.load(f)
    assert data["unit_costs"]["likelihood"] == pytest.approx(1e-3)
    new = RetrainCostModel(str(filename))
    assert new.deterministic
    assert new["population"] == pytest.approx(1e-2)
    assert new["training"] == pytest.approx(0.1)


def run_episodes(decision, rng, levels, length, epochs=50, resets=None):
    """Feed synthetic episodes into a decision object."""
    nlive = decision.nlive
    it = 0
    resets = resets or [False] * len(levels)
    for level, reset in zip(levels, resets):
        decision.start_episode(it, reset or it == 0, epochs, nlive)
        for c in geometric_counts(rng, level, -1.0 / nlive, length):
            decision.record_iteration(int(c))
        it += length
    return it


def make_decision(nlive=500, **kwargs):
    d = RetrainDecision(nlive, costs=COSTS, **kwargs)
    d._like_per_point = 1.0
    return d


def test_decide_without_history():
    d = make_decision()
    assert d.decide(0, 1000) is True


@pytest.mark.parametrize("training", [1e-9, 1.0])
def test_decide_depends_on_training_cost(rng, training):
    """Cheap training should retrain, very expensive training should not."""
    d = make_decision(horizon=False, plan_pool=False)
    d.cost.provided["training"] = training
    it = run_episodes(d, rng, [np.log(0.3)] * 4, 300)
    retrain = d.decide(it, 2000)
    assert retrain is (training < 1e-3)


def test_long_run_rate(rng):
    d = make_decision()
    run_episodes(d, rng, [np.log(0.3)] * 5, 500)
    out = d.long_run_rate()
    # tau is the solution of the renewal equation
    x = np.exp(out["k"] * out["tau"])
    assert np.isclose(x * np.log(x) - x + 1, out["y"])
    assert out["a"] == pytest.approx(0.3, rel=0.2)
    assert out["k"] == pytest.approx(1 / d.nlive, rel=0.2)


def test_planned_pool(rng):
    """With pool planning the next pool should not exceed the default."""
    d = make_decision(plan_pool=True)
    it = run_episodes(d, rng, [np.log(0.3)] * 4, 300)
    d.decide(it, 10_000)
    if d.next_poolsize is not None:
        assert d.next_poolsize <= 10_000 or d.log[-1]["retrain"]
        assert d.next_poolsize > 0


def test_reset_gain_degraded_lineage(rng):
    """A lineage that has degraded since its first training favours reset."""
    d = make_decision(allow_reset=True)
    run_episodes(d, rng, [-0.5, -1.0, -1.5, -2.0, -2.5, -2.5], 500)
    delta, var = d.reset_gain()
    assert delta > 0.5
    assert var < d.reset_prior_sd**2


def test_reset_gain_improving_lineage(rng):
    d = make_decision(allow_reset=True)
    run_episodes(d, rng, [-2.0, -1.5, -1.0, -1.0, -1.0], 500)
    delta, _ = d.reset_gain()
    assert delta < 0


def test_reset_gain_failed_reset(rng):
    """A reset that did not help counts against resetting again."""
    d = make_decision(allow_reset=True)
    levels = [-1.0, -1.0, -1.0, -1.5, -1.5, -1.5]
    resets = [True, False, False, True, False, False]
    run_episodes(d, rng, levels, 500, resets=resets)
    delta, _ = d.reset_gain()
    assert delta < 0


def test_pickle(rng):
    d = make_decision()
    run_episodes(d, rng, [np.log(0.3)] * 3, 200)
    d2 = pickle.loads(pickle.dumps(d))
    assert d2.decide(600, 1000) == d.decide(600, 1000)


@pytest.mark.slow_integration_test
def test_sampling_with_retrain_decision(
    integration_model, flow_config, tmp_path
):
    """Run the sampler with the retrain decision and deterministic costs."""
    from nessai.flowsampler import FlowSampler

    fs = FlowSampler(
        integration_model,
        output=str(tmp_path),
        nlive=100,
        plot=False,
        flow_config=flow_config,
        retrain_decision=dict(allow_reset=True),
        retrain_costs=COSTS,
        seed=1234,
        max_iteration=1000,
        checkpointing=False,
    )
    fs.run(plot=False)
    decision = fs.ns.retrain_decision
    assert decision.cost.deterministic
    assert len(decision.log) > 0
    assert (tmp_path / "retrain_costs.json").exists()


def test_cost_model_preset():
    from nessai.samplers.retrain import RETRAIN_COST_PRESETS

    cost = RetrainCostModel("gw")
    assert cost.deterministic
    assert cost["likelihood"] == RETRAIN_COST_PRESETS["gw"]["likelihood"]


def test_default_is_enabled_and_deterministic(integration_model, tmp_path):
    from nessai.samplers.nestedsampler import NestedSampler

    ns = NestedSampler(integration_model, output=str(tmp_path), nlive=50)
    assert ns.retrain_decision is not None
    assert ns.retrain_decision.cost.deterministic


@pytest.mark.slow_integration_test
def test_measure_retrain_costs(integration_model, flow_config, tmp_path):
    from nessai.samplers.retrain_benchmark import measure_retrain_costs

    filename = tmp_path / "costs.json"
    costs = measure_retrain_costs(
        integration_model,
        nlive=100,
        flow_config=flow_config,
        epochs=2,
        filename=str(filename),
        not_a_proposal_kwarg=True,
    )
    assert all(costs[k] > 0 for k in ("likelihood", "training"))
    assert costs["population"] >= 0
    assert integration_model.likelihood_evaluations == 0
    assert RetrainCostModel(str(filename)).deterministic


@pytest.mark.slow_integration_test
def test_retrain_benchmark_cli(tmp_path):
    from nessai.samplers.retrain_benchmark import main

    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(nlive=50, poolsize=100)))
    output = tmp_path / "costs.json"
    costs = main(
        [
            "nessai.utils.testing:IntegrationTestModel",
            "--config",
            str(config),
            "--epochs",
            "2",
            "--output",
            str(output),
        ]
    )
    assert output.exists()
    assert set(costs) == {"likelihood", "population", "training"}


@pytest.mark.slow_integration_test
def test_sampling_with_benchmark_costs(
    integration_model, flow_config, tmp_path
):
    from nessai.flowsampler import FlowSampler

    fs = FlowSampler(
        integration_model,
        output=str(tmp_path),
        nlive=100,
        plot=False,
        flow_config=flow_config,
        retrain_costs="benchmark",
        seed=1234,
        max_iteration=300,
        checkpointing=False,
    )
    fs.run(plot=False)
    assert fs.ns.retrain_decision.cost.deterministic
    assert (tmp_path / "retrain_costs_benchmark.json").exists()
