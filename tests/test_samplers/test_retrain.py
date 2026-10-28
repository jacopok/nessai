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
    lineage_sum,
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


def test_fresh_acceptance_tracks_short_episodes(rng):
    """Episodes shorter than a bin must still update the prediction.

    A run whose acceptance collapses gets pools that last only a few tens of
    iterations; if those episodes were ignored the prediction for a fresh
    flow would stay at the old, much higher level and the flow would be
    retrained every time the pool empties.
    """
    d = make_decision(nlive=4000)
    high, low = np.log(0.026), np.log(0.0045)
    it = run_episodes(d, rng, [high] * 5, 400)
    assert d.fresh_acceptance()[0] == pytest.approx(high, abs=0.2)
    for _ in range(10):
        d.start_episode(it, False, 50, d.nlive)
        for c in geometric_counts(rng, low, -1.0 / d.nlive, 40):
            d.record_iteration(int(c))
        it += 40
    d.start_episode(it, False, 50, d.nlive)
    mu, var = d.fresh_acceptance()
    assert mu == pytest.approx(low, abs=0.3)
    # the prediction errors of the short episodes widen the scatter
    assert np.sqrt(var) > d.fresh_prior_sd


def test_fresh_acceptance_weights_recent_episodes(rng):
    """The latest episode counts more than older ones."""
    d = make_decision()
    it = run_episodes(d, rng, [np.log(0.3)] * 4 + [np.log(0.05)], 500)
    d.start_episode(it, False, 50, d.nlive)
    mu, _ = d.fresh_acceptance()
    # the unweighted mean of the five levels would be log(0.21) ~ -1.6
    assert mu < np.log(0.15)


def test_slope_ignores_short_episodes(rng):
    """Short episodes carry no information on the decay slope."""
    d = make_decision()
    it = run_episodes(d, rng, [np.log(0.3)] * 3, 500)
    d.start_episode(it, False, 50, d.nlive)
    before = d.slope()
    for _ in range(5):
        for c in geometric_counts(rng, np.log(0.3), -1.0 / d.nlive, 10):
            d.record_iteration(int(c))
        it += 10
        d.start_episode(it, False, 50, d.nlive)
    assert d.slope() == before


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


def test_measure_retrain_costs_without_population(
    integration_model, flow_config, tmp_path
):
    """The run then measures the population cost from its own pools."""
    from nessai.samplers.retrain_benchmark import measure_retrain_costs

    filename = tmp_path / "costs.json"
    costs = measure_retrain_costs(
        integration_model,
        nlive=100,
        flow_config=flow_config,
        epochs=2,
        filename=str(filename),
        measure_population=False,
    )
    assert set(costs) == {"likelihood", "training"}
    model = RetrainCostModel(str(filename))
    assert not model.deterministic
    assert model["population"] is None
    model.record("population", 2.0, 100)
    assert model["population"] == pytest.approx(0.02)


def test_measure_retrain_costs_with_pool(
    integration_model, flow_config, tmp_path
):
    """The likelihood is timed on a pool of n_pool processes, as in the
    run, and the pool is closed afterwards."""
    from unittest.mock import patch

    from nessai.samplers.retrain_benchmark import measure_retrain_costs

    configure = integration_model.configure_pool
    with patch.object(
        integration_model, "configure_pool", side_effect=configure
    ) as mock_configure:
        costs = measure_retrain_costs(
            integration_model,
            nlive=100,
            flow_config=flow_config,
            epochs=2,
            measure_population=False,
            n_pool=2,
        )
    mock_configure.assert_called_once_with(n_pool=2)
    assert costs["likelihood"] > 0
    assert integration_model.pool is None
    assert not integration_model._pool_configured
    assert integration_model.likelihood_evaluations == 0


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


def test_retrain_logs_calculation(rng, caplog):
    """Every retrain is logged in brief at info level, with the inputs at
    debug level."""
    caplog.set_level(logging.DEBUG, logger="nessai.samplers.retrain")
    d = make_decision(horizon=False, plan_pool=False, allow_reset=True)
    d.cost.provided["training"] = 1e-9
    it = run_episodes(d, rng, [np.log(0.3)] * 4, 300)
    caplog.clear()
    assert d.decide(it, 2000) is True
    full, short = caplog.records[-2:]
    assert full.levelno == logging.DEBUG
    for key in ("Retraining", "c*P=", "g**n=", "acc=", "decay=", "T=", "tau*="):
        assert key in full.getMessage()
    assert "reset gain" in full.getMessage()
    assert short.levelno == logging.INFO
    assert f"etraining at it {it} (" in short.getMessage()
    assert "rule): acc " in short.getMessage()
    assert "fresh" in short.getMessage()


def test_continue_not_logged_at_info(rng, caplog):
    caplog.set_level(logging.INFO, logger="nessai.samplers.retrain")
    d = make_decision(horizon=False, plan_pool=False)
    d.cost.provided["training"] = 1.0
    it = run_episodes(d, rng, [np.log(0.3)] * 4, 300)
    caplog.clear()
    assert d.decide(it, 2000) is False
    assert not caplog.records


def test_fresh_acceptance_uses_the_running_episode(rng):
    """The flow that is ending informs the prediction for the next one."""
    d = make_decision()
    levels = [np.log(0.2)] * 4 + [np.log(0.02)]
    run_episodes(d, rng, levels, 400)
    # The running (last) episode is the only one at the low level
    mu_last = make_decision(fresh_predictor="last")
    run_episodes(mu_last, rng, levels, 400)
    assert abs(mu_last.fresh_acceptance()[0] - np.log(0.02)) < 0.3
    # Weighted average: the running episode carries the largest weight
    mu = d.fresh_acceptance()[0]
    assert np.log(0.02) < mu < 0.5 * (np.log(0.02) + np.log(0.2))
    # The running flow's own prior excludes its data
    assert d._warm_level(include_current=False)[0] > np.log(0.1)


def test_fresh_acceptance_trend_extrapolates(rng):
    """A steady decline is extrapolated by the trend, not by the average."""
    nlive = 500
    length = 250
    slope = -0.6  # log-acceptance per nlive iterations
    levels = [np.log(0.3) + slope * i * length / nlive for i in range(8)]
    out = {}
    for predictor in ("average", "trend"):
        d = make_decision(nlive=nlive, fresh_predictor=predictor)
        it = run_episodes(d, rng, levels, length)
        out[predictor] = d._warm_level(at=it)[0]
    truth = np.log(0.3) + slope * 8 * length / nlive
    assert abs(out["trend"] - truth) < abs(out["average"] - truth)
    assert abs(out["trend"] - truth) < 0.25


def test_fresh_acceptance_spread_from_recent_errors(rng):
    """The predictive scatter follows the recent prediction errors."""
    d = make_decision(fresh_predictor="last")
    noisy = np.log(0.1) + rng.choice([-1.0, 1.0], size=30)
    run_episodes(d, rng, list(noisy), 200)
    sd_noisy = np.sqrt(d.fresh_acceptance()[1])
    run_episodes(d, rng, [np.log(0.1)] * d.error_window, 200)
    sd_calm = np.sqrt(d.fresh_acceptance()[1])
    assert sd_noisy > 1.0
    assert sd_calm < 0.5


def test_fresh_predictor_invalid():
    with pytest.raises(ValueError, match="fresh_predictor"):
        make_decision(fresh_predictor="median")


def test_fresh_predictor_missing_after_unpickling(rng):
    """Checkpoints from before the option fall back to the default."""
    d = make_decision()
    run_episodes(d, rng, [np.log(0.1)] * 3, 200)
    del d.__dict__["fresh_predictor"]
    d = pickle.loads(pickle.dumps(d))
    assert d.fresh_acceptance() is not None


# ------------------------------------------------------------------ lineage
def test_lineage_sum_matches_the_sum():
    n, tau, h = 5, 50.0, 5000
    exact = np.sum(1.0 / (n + np.arange(h) / tau))
    assert lineage_sum(n, tau, h) == pytest.approx(exact, rel=0.02)
    growth = 2e-4
    exact = np.sum(np.exp(growth * np.arange(h)) / (n + np.arange(h) / tau))
    assert lineage_sum(n, tau, h, growth=growth) == pytest.approx(
        exact, rel=0.02
    )


def test_lineage_sum_edge_cases():
    assert lineage_sum(5, 50.0, 0) == 0.0
    with pytest.raises(ValueError, match="tau"):
        lineage_sum(5, 0.0, 100)


def lineage_decision(rng, levels=None, length=300, **kwargs):
    d = make_decision(**kwargs)
    run_episodes(d, rng, levels or [np.log(0.3)] * 6, length)
    d._horizon = 5000.0
    return d


@pytest.mark.parametrize(
    "exponent", [None, 0, 0.0, dict(likelihood=0.0, population=0.0)]
)
def test_lineage_gain_off(rng, exponent):
    d = lineage_decision(rng, lineage_exponent=exponent)
    rate = d.long_run_rate()
    assert rate["G"] == 0.0
    assert rate["T"] == rate["T_train"]


def test_lineage_gain_without_horizon(rng):
    d = lineage_decision(rng)
    d._horizon = None
    assert d.lineage_gain(100.0, 1e-3, 0.3, 1 / d.nlive) == 0.0


def test_lineage_gain_scales(rng):
    """The gain grows with the exponent and the horizon and falls with the
    length of the lineage."""
    args = (100.0, 1e-3, 0.3, 1 / 500)
    d1 = lineage_decision(rng, lineage_exponent=0.1)
    g1 = d1.lineage_gain(*args)
    assert g1 > 0
    d1.lineage_exponent = 0.2
    assert d1.lineage_gain(*args) == pytest.approx(2 * g1, rel=1e-6)
    d1.lineage_exponent = 0.1
    assert d1.lineage_gain(*args, horizon=1000.0) < g1
    longer = lineage_decision(
        rng, levels=[np.log(0.3)] * 20, lineage_exponent=0.1
    )
    assert longer.lineage_length() == 20
    longer._horizon = 5000.0
    assert longer.lineage_gain(*args) < g1


def test_lineage_gain_weights_cost_components(rng):
    """Each exponent applies to its share of the cost of a pool point."""
    d = lineage_decision(rng, lineage_exponent=0.2)
    like, pop = d.cost_components()
    args = (100.0, 1e-3, 0.3, 1 / 500)
    full = d.lineage_gain(*args)
    d.lineage_exponent = dict(likelihood=0.2)
    assert d.lineage_gain(*args) == pytest.approx(
        full * like / (like + pop), rel=1e-6
    )


def test_lineage_exponent_invalid():
    with pytest.raises(ValueError, match="lineage exponent keys"):
        RetrainDecision(100, lineage_exponent=dict(training=0.1))
    with pytest.raises(ValueError, match="non-negative"):
        RetrainDecision(100, lineage_exponent=-0.1)


def test_long_run_rate_with_lineage_is_self_consistent(rng):
    d = lineage_decision(rng, lineage_exponent=0.3)
    off = lineage_decision(rng, lineage_exponent=None).long_run_rate()
    rate = d.long_run_rate()
    assert 0 < rate["G"]
    assert rate["T"] == pytest.approx(max(rate["T_train"] - rate["G"], 0.0))
    assert rate["tau"] <= off["tau"]
    assert rate["g"] <= off["g"]
    # the gain is the one at the cycle the lower cost implies
    tau = max(rate["tau"], d._min_block())
    gain = d.lineage_gain(tau, rate["c"], rate["a"], rate["k"])
    assert gain == pytest.approx(rate["G"], rel=0.05)


def test_lineage_retrains_more_often(rng):
    """With the lineage term the decision retrains where it otherwise would
    continue."""
    kwargs = dict(horizon=False, plan_pool=False, reset_on_drop=None)
    old = make_decision(lineage_exponent=None, **kwargs)
    new = make_decision(lineage_exponent=0.3, **kwargs)
    for d in (old, new):
        d.cost.provided["training"] = 1.6e-5
        it = run_episodes(d, np.random.default_rng(3), [np.log(0.3)] * 4, 150)
    assert old.decide(it, 300, remaining=20_000) is False
    assert new.decide(it, 300, remaining=20_000) is True
    assert new.log[-1]["G"] > 0


def test_cost_trend(rng):
    d = make_decision()
    levels = [np.log(0.3) - 0.5 * i for i in range(8)]
    run_episodes(d, rng, levels, 250)
    slope, var = d.cost_trend()
    # the log cost per iteration rises by 0.5 per episode of 250 iterations
    assert slope * 250 == pytest.approx(0.5, abs=0.15)
    assert var > 0
    flat = make_decision()
    run_episodes(flat, rng, [np.log(0.3)] * 8, 250)
    assert abs(flat.cost_trend()[0] * 250) < 0.1


# ------------------------------------------------------- population costs
def test_cost_model_per_draw():
    assert RetrainCostModel(COSTS).per_draw is False
    assert RetrainCostModel(COSTS | dict(draw=1e-6)).per_draw is True
    measured = RetrainCostModel(None)
    assert measured.per_draw is False
    measured.record("draw", 1.0, 1e5)
    assert measured.per_draw is True
    assert measured["draw"] == pytest.approx(1e-5)


def test_pool_point_cost_counts_draws():
    d = RetrainDecision(500, costs=COSTS | dict(draw=1e-6))
    d.start_episode(0, True, 10, 500)
    d.record_populations(1.0, 1000, 0.5, 1000, n_draws=50_000)
    like, pop = d.cost_components()
    assert like == pytest.approx(1e-3)
    assert pop == pytest.approx(1e-6 * 50)
    # a later flow whose pool takes fewer draws costs less
    d.start_episode(100, False, 10, 500)
    d.record_populations(1.0, 1000, 0.5, 1000, n_draws=10_000)
    assert d.pool_point_cost() == pytest.approx(1e-3 + 1e-5)
    assert d.pool_point_cost(d.episodes[0]) == pytest.approx(1e-3 + 5e-5)


def test_pool_point_cost_without_draws():
    """Without draws the provided cost per pool point is used."""
    d = RetrainDecision(500, costs=COSTS | dict(draw=1e-6))
    d.start_episode(0, True, 10, 500)
    d.record_populations(1.0, 1000, 0.5, 1000)
    d._like_per_point = 1.0
    assert d.pool_point_cost() == pytest.approx(1e-3 + 1e-4)


def test_cost_model_unpickles_without_draw():
    cost = RetrainCostModel(COSTS)
    for attr in ("provided", "totals", "units", "samples"):
        del getattr(cost, attr)["draw"]
    new = pickle.loads(pickle.dumps(cost))
    assert new.provided["draw"] is None
    new.record("draw", 1.0, 10)
    assert new.measured("draw") == pytest.approx(0.1)


def test_cost_model_saves_draw(tmp_path):
    cost = RetrainCostModel(COSTS)
    cost.record("draw", 1.0, 1e5)
    cost.save(tmp_path / "costs.json")
    new = RetrainCostModel(str(tmp_path / "costs.json"))
    assert new.provided["draw"] == pytest.approx(1e-5)
    assert new.per_draw


# ---------------------------------------------------------- reset on drop
def drop_decision(rng, drop=3.0, n_before=6, length=300, **kwargs):
    """A lineage at a steady level whose latest warm training is ``drop``
    lower in log acceptance."""
    d = make_decision(**kwargs)
    levels = [np.log(0.3)] * n_before + [np.log(0.3) - drop]
    it = run_episodes(d, rng, levels, length)
    return d, it


def test_drop_triggers_reset(rng, caplog):
    caplog.set_level(logging.INFO)
    d, it = drop_decision(rng)
    assert d.decide(it, 1000) is True
    assert d.reset_next is True
    assert d.log[-1]["reason"] == "drop"
    assert d.log[-1]["drop"] > np.log(4)
    assert "Resetting the flow" in caplog.text
    # the flow it replaces is the baseline of the reset
    d.start_episode(it, True, 100, d.nlive)
    assert d.current.attempt
    assert np.isfinite(d.current.baseline[0])


@pytest.mark.parametrize("drop", [0.0, 0.5, 1.0])
def test_no_reset_on_small_drop(rng, drop):
    d, it = drop_decision(rng, drop=drop)
    d.decide(it, 1000)
    assert d.reset_next is False


def test_no_reset_on_gradual_drift(rng):
    """A cost that rises steadily over the run is not a drop."""
    d = make_decision()
    levels = [np.log(0.3) - 0.15 * i for i in range(30)]
    it = 0
    for level in levels:
        d.start_episode(it, it == 0, 50, d.nlive)
        for c in geometric_counts(rng, level, -1.0 / d.nlive, 100):
            d.record_iteration(int(c))
        it += 100
        d.decide(it, 1000)
        assert d.log[-1]["reason"] != "drop"


def test_drop_disabled(rng):
    d, it = drop_decision(rng, reset_on_drop=None)
    d.decide(it, 1000)
    assert not d.reset_next
    assert d.log[-1]["reason"] != "drop"


def test_no_reset_on_drop_of_noisy_trainings(rng):
    """Trainings that scatter widely do not trigger resets."""
    d = make_decision()
    noisy = np.random.default_rng(1)
    it = 0
    resets = 0
    for i in range(40):
        d.start_episode(it, it == 0, 50, d.nlive)
        level = np.log(0.2) + noisy.normal(0, 0.5)
        for c in geometric_counts(rng, level, -1.0 / d.nlive, 100):
            d.record_iteration(int(c))
        it += 100
        d.decide(it, 1000)
        resets += d.reset_next
    assert resets == 0


def attempt(d, rng, it, level, length=100):
    d.start_episode(it, True, 100, d.nlive)
    for c in geometric_counts(rng, level, -1.0 / d.nlive, length):
        d.record_iteration(int(c))
    return it + length


def test_reset_discarded_when_worse(rng, caplog):
    caplog.set_level(logging.INFO)
    d, it = drop_decision(rng)
    n = d.lineage_length()
    d.decide(it, 1000)
    it = attempt(d, rng, it, np.log(0.3) - 4.0)
    assert d.lineage_length() == 1
    assert d.decide(it, 1000) is True
    assert d.rollback_next is True
    assert d.reset_next is False
    assert d.current.rejected
    assert d.log[-1]["reason"] == "rollback"
    assert "Discarding the reset flow" in caplog.text
    # the restored flow continues its lineage
    d.start_episode(it, False, 50, d.nlive)
    assert d.rollback_next is False
    assert d.lineage_length() == n + 1
    assert d.summary()["n_resets_discarded"] == 1
    # and is not reset again straight away
    for c in geometric_counts(rng, np.log(0.3) - 3.0, -1.0 / d.nlive, 100):
        d.record_iteration(int(c))
    d.decide(it + 100, 1000)
    assert not d.reset_next


def test_reset_kept_when_better(rng, caplog):
    caplog.set_level(logging.INFO)
    d, it = drop_decision(rng)
    d.decide(it, 1000)
    it = attempt(d, rng, it, np.log(0.3))
    d.decide(it, 1000)
    assert d.rollback_next is False
    assert d.current.resolved and not d.current.rejected
    assert "Keeping the reset flow" in caplog.text
    assert d.lineage_length() == 1


def test_reset_kept_without_rollback(rng):
    d, it = drop_decision(rng, reset_rollback=False)
    d.decide(it, 1000)
    assert d.reset_next
    it = attempt(d, rng, it, np.log(0.3) - 4.0)
    assert not d.current.attempt
    d.decide(it, 1000)
    assert d.rollback_next is False


def test_cost_based_reset_is_checked(rng):
    """Resets chosen by ``allow_reset`` are also compared to the flow they
    replace."""
    d = make_decision(allow_reset=True, reset_on_drop=None)
    run_episodes(d, rng, [-0.5, -1.0, -1.5, -2.0, -2.5, -2.5], 500)
    it = 3000
    for _ in range(20):
        retrain = d.decide(it, 2000)
        if retrain and d.reset_next:
            break
        it += 1
    else:
        pytest.skip("no reset chosen")
    assert d._attempt_next is not None


def test_new_options_missing_after_unpickling(rng):
    """Decisions pickled before the lineage and drop options keep the old
    behaviour."""
    d = make_decision()
    run_episodes(d, rng, [np.log(0.3)] * 3, 200)
    for attr in (
        "lineage_exponent",
        "lineage_horizon",
        "reset_on_drop",
        "reset_window",
        "reset_rollback",
        "reset_tolerance",
        "rollback_next",
        "_attempt_next",
        "_drop_anchor",
        "_horizon",
    ):
        del d.__dict__[attr]
    d2 = pickle.loads(pickle.dumps(d))
    assert d2.lineage_exponent is None
    assert d2.reset_on_drop is None
    assert d2.can_reset is False
    d2.decide(600, 1000, remaining=1000)
    assert d2.log[-1].get("G", 0.0) == 0.0


# -------------------------------------------------------- sampler rollback
class _Flow:
    def __init__(self):
        import torch

        self.model = torch.nn.Linear(2, 2)
        self._optimiser = torch.optim.Adam(self.model.parameters())

    def get_optimiser(self):
        import torch

        return torch.optim.Adam(self.model.parameters())


def test_sampler_saves_and_restores_flow():
    from types import SimpleNamespace

    import torch

    from nessai.samplers.nestedsampler import NestedSampler

    ns = object.__new__(NestedSampler)
    ns.proposal = SimpleNamespace(flow=_Flow())
    before = ns.proposal.flow.model.weight.detach().clone()
    ns._save_rollback_flow()
    with torch.no_grad():
        ns.proposal.flow.model.weight += 1.0
    ns._restore_rollback_flow()
    assert torch.equal(ns.proposal.flow.model.weight, before)
    opt = ns.proposal.flow._optimiser
    assert opt.param_groups[0]["params"][0] is ns.proposal.flow.model.weight
    assert ns._rollback_flow is None


def test_sampler_restore_without_saved_flow(caplog):
    from types import SimpleNamespace

    from nessai.samplers.nestedsampler import NestedSampler

    ns = object.__new__(NestedSampler)
    ns.proposal = SimpleNamespace(flow=_Flow())
    ns._rollback_flow = None
    model = ns.proposal.flow.model
    ns._restore_rollback_flow()
    assert ns.proposal.flow.model is model
    assert "not available" in caplog.text


@pytest.mark.parametrize("rollback", [False, True])
def test_check_flow_model_reset_with_decision(rollback):
    from unittest.mock import MagicMock

    from nessai.samplers.nestedsampler import NestedSampler

    ns = object.__new__(NestedSampler)
    ns.proposal = MagicMock(training_count=3)
    ns.retrain_decision = MagicMock(
        can_reset=True,
        rollback_next=rollback,
        reset_next=not rollback,
        reset_rollback=True,
    )
    ns._save_rollback_flow = MagicMock()
    ns._restore_rollback_flow = MagicMock()
    ns.check_flow_model_reset()
    if rollback:
        ns._restore_rollback_flow.assert_called_once()
        ns.proposal.reset_model_weights.assert_not_called()
    else:
        ns._save_rollback_flow.assert_called_once()
        ns.proposal.reset_model_weights.assert_called_once_with(
            weights=True, permutations=True
        )


def test_sampler_checkpoint_excludes_saved_flow():
    from nessai.samplers.nestedsampler import NestedSampler

    ns = object.__new__(NestedSampler)
    ns.__dict__.update(
        model=MagicModel(),
        proposal=None,
        checkpoint_callback=None,
        _rollback_flow=("big", "flow"),
        other=1,
    )
    state = ns.__getstate__()
    assert "_rollback_flow" not in state
    assert state["other"] == 1


class MagicModel:
    likelihood_evaluations = 0

    class _T:
        @staticmethod
        def total_seconds():
            return 0.0

    likelihood_evaluation_time = _T()


@pytest.mark.slow_integration_test
def test_sampling_with_resets_and_rollbacks(
    integration_model, flow_config, tmp_path
):
    """A run in which every warm training counts as a drop, so the flow is
    reset (and compared, and maybe restored) many times."""
    from nessai.flowsampler import FlowSampler

    fs = FlowSampler(
        integration_model,
        output=str(tmp_path),
        nlive=100,
        plot=False,
        flow_config=flow_config,
        retrain_decision=dict(reset_on_drop=-50.0, reset_window=10.0),
        retrain_costs=COSTS,
        seed=1234,
        max_iteration=1500,
        checkpointing=False,
    )
    fs.run(plot=False)
    summary = fs.ns.retrain_decision.summary()
    assert summary["n_resets"] > 0
    episodes = fs.ns.retrain_decision.episodes
    assert any(e.attempt and e.resolved for e in episodes)


def test_region_saves_and_restores_flow_and_secondary():
    """A region's reset is discarded with its own flows, the mixture's
    secondary included, and the prime-space action is realigned."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import torch

    from nessai.samplers.nestedsampler import NestedSampler

    ns = object.__new__(NestedSampler)
    ns.proposal = SimpleNamespace(flow=_Flow())
    proposal = SimpleNamespace(
        flow=_Flow(),
        secondary=SimpleNamespace(flow=_Flow()),
        _realign_prime_space_action=MagicMock(),
    )
    region = SimpleNamespace(label="1", proposal=proposal)
    flows = (proposal.flow, proposal.secondary.flow)
    before = [f.model.weight.detach().clone() for f in flows]
    run_flow = ns.proposal.flow.model
    ns._save_rollback_flow(region)
    with torch.no_grad():
        for f in flows:
            f.model.weight += 1.0
    ns._restore_rollback_flow(region)
    for f, b in zip(flows, before):
        assert torch.equal(f.model.weight, b)
    proposal._realign_prime_space_action.assert_called_once()
    assert ns.proposal.flow.model is run_flow
    assert ns._region_rollback == {}


def test_region_restore_after_resume(caplog):
    """After a resume the saved flows are gone: the reset flow is kept."""
    from types import SimpleNamespace

    from nessai.samplers.nestedsampler import NestedSampler

    ns = object.__new__(NestedSampler)
    region = SimpleNamespace(label="0", proposal=SimpleNamespace(flow=_Flow()))
    model = region.proposal.flow.model
    ns._restore_rollback_flow(region)
    assert region.proposal.flow.model is model
    assert "not available" in caplog.text


@pytest.mark.parametrize("rollback", [False, True])
def test_region_flow_reset_with_decision(rollback):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from nessai.samplers.nestedsampler import NestedSampler

    ns = object.__new__(NestedSampler)
    region = SimpleNamespace(
        proposal=MagicMock(training_count=3),
        retrain_decision=MagicMock(
            can_reset=True,
            rollback_next=rollback,
            reset_next=not rollback,
            reset_rollback=True,
        ),
    )
    ns._save_rollback_flow = MagicMock()
    ns._restore_rollback_flow = MagicMock()
    ns._region_flow_reset(region)
    if rollback:
        ns._restore_rollback_flow.assert_called_once_with(region)
        region.proposal.reset_model_weights.assert_not_called()
    else:
        ns._save_rollback_flow.assert_called_once_with(region)
        region.proposal.reset_model_weights.assert_called_once_with(
            weights=True, permutations=True
        )
