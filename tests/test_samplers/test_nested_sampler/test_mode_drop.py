"""Tests for the drop of under-covered dying modes."""

from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest

from nessai.evidence import _NSIntegralState
from nessai.livepoint import empty_structured_array
from nessai.samplers.nestedsampler import NestedSampler


def _live(logl):
    live = empty_structured_array(len(logl), names=["x"])
    live["x"] = np.arange(len(logl))
    live["logL"] = np.sort(logl)
    return live


@pytest.fixture
def ds(rng):
    """Mock sampler with 200 live points, 10 of them in a mode the flow
    under-covers by 10 nats."""
    s = create_autospec(NestedSampler)
    s.nlive = 200
    s.live_points = _live(rng.normal(size=200))
    s.state = _NSIntegralState(200)
    s.state.logw = -5.0
    s.state.logZ = 1.0
    s.iteration = 1000
    s.completed_training = True
    s._refilling = False
    s.retrain_decision = None
    s._flow_proposal = MagicMock()
    s._flow_proposal.populated = False
    s._flow_proposal.populating = False
    s._flow_proposal.training_count = 3
    s.proposal = s._flow_proposal
    level = np.zeros(200)
    s.in_mode = np.zeros(200, dtype=bool)
    s.in_mode[::20] = True
    level[s.in_mode] = -10.0
    rule = MagicMock(quantile=0.005, prior_relative=True)
    rule.point_levels.return_value = level
    s._flow_proposal.get_truncation_rule.return_value = rule
    s._mode_drop_rule = lambda: NestedSampler._mode_drop_rule(s)
    NestedSampler.configure_mode_drop(s, True)
    return s


def test_configure_mode_drop_defaults(ds):
    assert ds.mode_drop == dict(
        delta=6.0,
        min_share=None,
        max_share=0.1,
        max_posterior=1e-3,
        retrain=True,
        check_every=10,
    )
    assert ds.mode_drops == []
    assert ds._log_volume_dropped == 0.0


@pytest.mark.parametrize("value", [None, False])
def test_configure_mode_drop_disabled(ds, value):
    NestedSampler.configure_mode_drop(ds, value)
    assert ds.mode_drop is None
    assert NestedSampler._check_mode_drop(ds) is False


@pytest.mark.parametrize(
    "config, match",
    [
        (dict(gap=6), "Unknown"),
        (dict(delta=0), "delta"),
        (dict(max_share=1.5), "max_share"),
    ],
)
def test_configure_mode_drop_invalid(ds, config, match):
    with pytest.raises(ValueError, match=match):
        NestedSampler.configure_mode_drop(ds, config)


def test_check_mode_drop_fires(ds):
    assert NestedSampler._check_mode_drop(ds) is True
    ds._drop_mode.assert_called_once()
    in_mode, level, median, bound = ds._drop_mode.call_args.args
    np.testing.assert_array_equal(in_mode, ds.in_mode)
    assert median == 0.0
    # bound: share x X x max L of the mode / (Z + X mean L)
    logl = ds.live_points["logL"]
    log_z = np.logaddexp(1.0, -5.0 + np.log(np.mean(np.exp(logl))))
    expected = np.exp(
        np.log(0.05) - 5.0 + logl[ds.in_mode].max() - log_z
    )
    assert bound == pytest.approx(expected)
    assert ds._mode_drop_checked == 1000


def test_check_mode_drop_only_once_per_interval(ds):
    ds._mode_drop_checked = 995
    assert NestedSampler._check_mode_drop(ds) is False
    ds._flow_proposal.get_truncation_rule.assert_not_called()


@pytest.mark.parametrize(
    "attr, value",
    [
        ("populated", True),
        ("populating", True),
        ("training_count", 0),
    ],
)
def test_check_mode_drop_needs_empty_pool_of_trained_flow(ds, attr, value):
    setattr(ds._flow_proposal, attr, value)
    assert NestedSampler._check_mode_drop(ds) is False
    ds._drop_mode.assert_not_called()


def test_check_mode_drop_not_while_refilling(ds):
    ds._refilling = True
    assert NestedSampler._check_mode_drop(ds) is False


@pytest.mark.parametrize(
    "config",
    [
        dict(max_share=0.04),  # the mode holds 5%
        dict(min_share=0.05),
        dict(max_posterior=1e-12),
    ],
)
def test_check_mode_drop_gates(ds, config):
    NestedSampler.configure_mode_drop(ds, config)
    assert NestedSampler._check_mode_drop(ds) is False
    ds._drop_mode.assert_not_called()


def test_check_mode_drop_budget_is_cumulative(ds):
    """The bounds of the drops so far count against the budget."""
    assert NestedSampler._check_mode_drop(ds) is True
    bound = ds._drop_mode.call_args.args[3]
    assert 1e-5 < bound < 1e-3
    ds._mode_drop_checked = -np.inf
    ds._mode_drop_posterior = 1e-3 - 0.5 * bound
    assert NestedSampler._check_mode_drop(ds) is False
    ds._mode_drop_checked = -np.inf
    ds._mode_drop_posterior = 1e-3 - 2 * bound
    assert NestedSampler._check_mode_drop(ds) is True


def test_check_mode_drop_without_prior_relative_rule(ds, caplog):
    ds._flow_proposal.get_truncation_rule.return_value = None
    assert NestedSampler._check_mode_drop(ds) is False
    assert "disabled" in caplog.text


def test_drop_mode(ds):
    live = ds.live_points.copy()
    level = np.where(ds.in_mode, -10.0, 0.0)
    NestedSampler._drop_mode(ds, ds.in_mode, level, 0.0, 1e-5)
    assert len(ds.live_points) == 190
    np.testing.assert_array_equal(ds.live_points["x"], live["x"][~ds.in_mode])
    np.testing.assert_array_equal(
        ds.dropped_live_points["x"], live["x"][ds.in_mode]
    )
    assert ds.state.logw == pytest.approx(-5.0 + np.log(0.95))
    assert ds._log_volume_dropped == pytest.approx(np.log(0.95))
    assert ds._mode_drop_posterior == 1e-5
    record = ds.mode_drops[-1]
    assert record["n"] == 10
    assert record["share"] == 0.05
    assert record["mode_level_median"] == -10.0
    ds.train_proposal.assert_called_once_with(force=True)
    ds._refill_live_points.assert_called_once()


def test_drop_mode_restarts_retrain_history(ds):
    ds.retrain_decision = MagicMock()
    NestedSampler._drop_mode(
        ds, ds.in_mode, np.zeros(200), 0.0, 1e-5
    )
    ds.retrain_decision.restart_history.assert_called_once()


def test_drop_mode_without_retrain(ds):
    NestedSampler.configure_mode_drop(ds, dict(retrain=False))
    NestedSampler._drop_mode(
        ds, ds.in_mode, np.zeros(200), 0.0, 1e-5
    )
    ds.train_proposal.assert_not_called()
    ds._refill_live_points.assert_called_once()


def test_refill_live_points(ds):
    """Draws above the contour are inserted in order; a draw returned when
    the pool ran out (at the contour) triggers a state check instead."""
    ds.live_points = ds.live_points[:-3]
    ds.logLmin = -10.0
    new = [5.0, -20.0, -10.0, 0.25, 7.0]
    stale = []

    def yield_sample(old):
        p = old.copy()
        p["logL"] = new.pop(0)
        stale.append(p["logL"] <= ds.logLmin)
        yield 4, p

    ds.yield_sample = yield_sample
    ds.retrain_decision = MagicMock()
    NestedSampler._refill_live_points(ds)
    assert len(ds.live_points) == 200
    assert np.all(np.diff(ds.live_points["logL"]) >= 0)
    assert {5.0, 0.25, 7.0} <= set(ds.live_points["logL"])
    assert ds.check_state.call_count == 2
    assert ds.retrain_decision.record_iteration.call_count == 3
    assert ds._refilling is False


@pytest.mark.slow_integration_test
def test_sampling_with_mode_drop(integration_model, flow_config, tmp_path):
    """A run where the drop fires (a low threshold on the level gap, any
    share and posterior) completes with nlive live points, and records
    the drops and the removed volume."""
    from nessai.flowsampler import FlowSampler

    fs = FlowSampler(
        integration_model,
        output=str(tmp_path / "mode_drop"),
        nlive=200,
        plot=False,
        checkpointing=False,
        resume=False,
        seed=1234,
        flow_config=flow_config,
        poolsize=500,
        truncation_methods=["log_proposal_threshold"],
        truncation_kwargs={
            "log_proposal_threshold": dict(
                quantile=0.005, prior_relative=True, mode="clip"
            )
        },
        threshold_from_live_points=True,
        retrain_decision=True,
        retrain_costs="gw",
        mode_drop=dict(
            delta=0.1, min_share=0.0, max_share=0.5, max_posterior=1.0
        ),
        max_iteration=1500,
    )
    fs.run(plot=False, save=False)
    drops = fs.ns.mode_drops
    assert drops
    removed = fs.ns.state.removed_volumes
    assert [f for _, f in removed] == [d["share"] for d in drops]
    assert fs.ns._log_volume_dropped == pytest.approx(
        np.sum(np.log1p(-np.array([d["share"] for d in drops])))
    )
