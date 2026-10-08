# -*- coding: utf-8 -*-
"""Tests for staged truncation during FlowProposal population."""

import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from nessai.livepoint import empty_structured_array
from nessai.proposal import FlowProposal
from nessai.proposal.flowproposal.truncation import (
    LatentRadiusTruncation,
    LikelihoodThresholdTruncation,
    MinLogQTruncation,
    TruncationScheme,
)


def configure_population_test_proposal(proposal, rng, samples):
    proposal.population_time = datetime.timedelta()
    proposal.initialised = True
    proposal.indices = []
    proposal.acceptance = []
    proposal.keep_samples = False
    proposal.check_acceptance = False
    proposal._plot_pool = False
    proposal.populated_count = 0
    proposal.map_to_unit_hypercube = False
    proposal.population_dtype = empty_structured_array(
        0, names=["x", "y"]
    ).dtype
    proposal.convert_to_samples = MagicMock(
        side_effect=lambda x, plot: x.copy()
    )
    proposal.compute_weights = MagicMock(
        side_effect=lambda x, log_q: np.zeros(x.size)
    )
    proposal.rng = MagicMock(wraps=rng)
    proposal.rng.random = MagicMock(side_effect=lambda n: np.full(n, 0.5))
    proposal.rng.permutation = MagicMock(side_effect=lambda n: np.arange(n))
    proposal.model = MagicMock()
    proposal.training_data = samples([(0.0, 0.0), (1.0, 1.0)])
    proposal._truncation_scheme = TruncationScheme()
    proposal.adapt_latent_temperature = False
    proposal.clip_population_weights = False
    proposal._get_population_log_weights = (
        lambda log_w, **kw: FlowProposal._get_population_log_weights(
            proposal, log_w, **kw)
    )
    proposal.drawsize = 3
    proposal.flow = MagicMock()
    proposal.sample_latent_distribution = MagicMock(
        side_effect=proposal.flow.sample_latent_distribution
    )


def test_populate_applies_truncation_in_stages(proposal, rng, point, samples):
    configure_population_test_proposal(proposal, rng, samples)
    proposal._truncation_scheme = TruncationScheme(
        [
            LatentRadiusTruncation(fixed_radius=1.0, radius_mode="fixed"),
            MinLogQTruncation(),
            LikelihoodThresholdTruncation(),
        ]
    )
    proposal.flow.sample_latent_distribution.return_value = np.array(
        [[0.0, 0.0], [2.0, 0.0], [0.5, 0.0]]
    )
    proposal.forward_pass = MagicMock(
        return_value=(np.zeros((2, 2)), np.array([0.0, -1.0]))
    )
    proposal.backward_pass = MagicMock(
        return_value=(
            samples([(1.0, 1.0), (2.0, 2.0)]),
            np.array([0.5, -2.0]),
            np.array([[0.0, 0.0], [0.5, 0.0]]),
        )
    )
    proposal.model.batch_evaluate_log_likelihood.return_value = np.array([1.0])

    FlowProposal.populate(
        proposal, point(0.0, 0.0, logl=0.5), n_samples=1, plot=False
    )

    proposal.sample_latent_distribution.assert_called_once_with(3)
    proposal.backward_pass.assert_called_once()
    proposal.model.batch_evaluate_log_likelihood.assert_called_once()
    proposal.compute_weights.assert_called_once()
    assert proposal.x.size == 1
    assert proposal.samples.size == 1
    assert proposal._truncation_scheme.get_rule("latent_radius").radius == 1.0
    assert proposal.populated is True


def test_populate_continues_after_empty_latent_batch(
    proposal, rng, point, samples
):
    configure_population_test_proposal(proposal, rng, samples)
    proposal._truncation_scheme = TruncationScheme(
        [LatentRadiusTruncation(fixed_radius=1.0, radius_mode="fixed")]
    )
    proposal.flow.sample_latent_distribution.side_effect = [
        np.array([[2.0, 0.0], [3.0, 0.0]]),
        np.array([[0.0, 0.0], [2.0, 0.0]]),
    ]
    proposal.backward_pass = MagicMock(
        return_value=(
            samples([(1.0, 1.0)]),
            np.array([0.5]),
            np.array([[0.0, 0.0]]),
        )
    )
    proposal.model.batch_evaluate_log_likelihood.return_value = np.array([1.0])

    FlowProposal.populate(
        proposal, point(0.0, 0.0, logl=0.5), n_samples=1, plot=False
    )

    assert proposal.sample_latent_distribution.call_count == 2
    proposal.backward_pass.assert_called_once()
    proposal.model.batch_evaluate_log_likelihood.assert_called_once()
    assert proposal.x.size == 1


def test_populate_accumulate_weights_recomputes_accept_on_max_samples(
    proposal, rng, point, samples
):
    configure_population_test_proposal(proposal, rng, samples)
    proposal.accumulate_weights = True
    proposal.flow.sample_latent_distribution.return_value = np.zeros((3, 2))
    proposal.backward_pass = MagicMock(
        return_value=(
            samples([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
            np.zeros(3),
            np.zeros((3, 2)),
        )
    )
    proposal.compute_weights.return_value = np.zeros(3)
    proposal.model.batch_evaluate_log_likelihood.return_value = np.array(
        [1.0, 2.0]
    )

    FlowProposal.populate(
        proposal,
        point(0.0, 0.0, logl=0.5),
        n_samples=2,
        plot=False,
        max_samples=2,
    )

    assert proposal.x.size == 2
    assert proposal.samples.size == 2
    assert proposal.sample_latent_distribution.call_count == 1
    assert proposal.rng.random.call_count == 1


def test_populate_stops_at_max_samples_after_empty_latent_batches(
    proposal, rng, point, samples
):
    configure_population_test_proposal(proposal, rng, samples)
    proposal._truncation_scheme = TruncationScheme(
        [LatentRadiusTruncation(fixed_radius=0.1, radius_mode="fixed")]
    )
    proposal.flow.sample_latent_distribution.return_value = np.array(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    )
    proposal.model.batch_evaluate_log_likelihood.return_value = np.array([])

    FlowProposal.populate(
        proposal,
        point(0.0, 0.0, logl=0.5),
        n_samples=1,
        plot=False,
        max_samples=2,
    )

    proposal.sample_latent_distribution.assert_called_once_with(3)
    assert proposal.backward_pass.call_count == 0
    assert proposal.x.size == 0
    assert proposal.population_acceptance == 0.0


def test_populate_stops_at_max_samples_after_all_likelihood_rejected(
    proposal, rng, point, samples
):
    configure_population_test_proposal(proposal, rng, samples)
    proposal._truncation_scheme = TruncationScheme(
        [LikelihoodThresholdTruncation()]
    )
    proposal.flow.sample_latent_distribution.return_value = np.zeros((3, 2))
    proposal.backward_pass = MagicMock(
        return_value=(
            samples([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
            np.zeros(3),
            np.zeros((3, 2)),
        )
    )
    proposal.model.batch_evaluate_log_likelihood.return_value = np.zeros(3)

    FlowProposal.populate(
        proposal,
        point(0.0, 0.0, logl=0.5),
        n_samples=1,
        plot=False,
        max_samples=2,
    )

    proposal.sample_latent_distribution.assert_called_once_with(3)
    proposal.backward_pass.assert_called_once()
    proposal.compute_weights.assert_not_called()
    assert proposal.x.size == 0
    assert proposal.population_acceptance == 0.0


def test_weight_cap_for_mass_bounds_the_excess():
    from nessai.proposal.flowproposal.flowproposal import _weight_cap_for_mass

    rng = np.random.default_rng(0)
    w = rng.pareto(3.0, size=20000) + 1.0
    for mass in (1e-3, 1e-2, 0.1):
        cap = _weight_cap_for_mass(w, mass)
        excess = np.maximum(w - cap, 0.0).sum() / w.sum()
        assert excess == pytest.approx(mass, rel=1e-6)
    # a cap below the smallest weight is never returned
    assert _weight_cap_for_mass(np.ones(10), 0.5) == pytest.approx(1.0)


def test_population_log_weights_capped_by_mass(proposal):
    proposal.clip_population_weights = False
    proposal.population_weight_cap_mass = 0.01
    rng = np.random.default_rng(1)
    log_w = np.log(rng.pareto(2.0, size=5000) + 1.0)
    out = FlowProposal._get_population_log_weights(proposal, log_w)
    w = np.exp(log_w)
    assert out.max() == 0.0
    # rejection accepts min(w, c) / c: the prior mass lost is the excess
    from nessai.proposal.flowproposal.flowproposal import _weight_cap_for_mass

    cap = _weight_cap_for_mass(w / w.max(), 0.01) * w.max()
    np.testing.assert_allclose(np.exp(out), np.minimum(w, cap) / cap)
    lost = 1.0 - (np.exp(out) * cap).sum() / w.sum()
    assert lost == pytest.approx(0.01, rel=1e-6)


def test_population_weight_cap_keeps_a_minimum_yield(proposal):
    """A heavy tail (a few draws carry most of the mass) must not stall the
    population: the cap is lowered until the draws yield ``min_yield``
    points, and the prior mass then under-sampled is recorded."""
    from nessai.proposal.flowproposal.flowproposal import _weight_cap_for_yield

    proposal.clip_population_weights = False
    proposal.population_weight_cap_mass = 0.005
    n = 10_000
    w = np.ones(n)
    w[:3] = 1e6   # three draws hold ~99.7 % of the mass
    out = FlowProposal._get_population_log_weights(
        proposal, np.log(w), min_yield=50.0)
    assert np.exp(out).sum() == pytest.approx(50.0)
    assert proposal._population_cap_lost_mass > 0.99
    # without the floor the mass cap yields ~3 points
    out = FlowProposal._get_population_log_weights(proposal, np.log(w))
    assert np.exp(out).sum() < 4
    assert proposal._population_cap_lost_mass is None
    # a light tail keeps the mass cap when it already yields enough
    rng = np.random.default_rng(2)
    log_w = np.log(rng.pareto(3.0, size=n) + 1.0)
    out = FlowProposal._get_population_log_weights(
        proposal, log_w, min_yield=50.0)
    assert np.exp(out).sum() > 50.0
    assert proposal._population_cap_lost_mass is None
    # the yield cap solves sum(min(w, c)) / c = Y
    v = rng.pareto(1.5, size=1000) + 1.0
    for y in (1.5, 10.0, 400.0):
        c = _weight_cap_for_yield(v, y)
        assert np.minimum(v, c).sum() / c == pytest.approx(y)


def test_population_weight_cap_mass_validation(proposal):
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError, match="population_weight_cap_mass"):
            FlowProposal.configure_population(
                proposal, 10, population_weight_cap_mass=bad
            )
    with pytest.raises(ValueError, match="mutually exclusive"):
        FlowProposal.configure_population(
            proposal, 10, clip_population_weights=True,
            population_weight_cap_mass=0.01,
        )


def test_max_samples_warning_once_per_training_round(caplog):
    proposal = MagicMock(spec=FlowProposal)
    proposal.training_count = 3
    proposal._max_samples_warned = None
    caplog.set_level("DEBUG", logger="nessai.proposal.flowproposal")
    for _ in range(3):
        FlowProposal._warn_max_samples(proposal, 10)
    proposal.training_count = 4
    FlowProposal._warn_max_samples(proposal, 10)
    levels = [r.levelname for r in caplog.records
              if "Reached max samples" in r.getMessage()]
    assert levels == ["WARNING", "DEBUG", "DEBUG", "WARNING"]
