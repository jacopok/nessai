# -*- coding: utf-8 -*-
"""Clip mode and live-point threshold of the log-proposal truncation."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from nessai.livepoint import empty_structured_array
from nessai.proposal.flowproposal.truncation import (
    LogProposalThresholdTruncation,
    TruncationScheme,
)


def _arr(v, logl=None):
    a = empty_structured_array(len(v), names=["x"])
    a["x"] = v
    if logl is not None:
        a["logL"] = logl
    return a


def _proposal(training, current=None, from_live=False):
    proposal = MagicMock()
    proposal.training_data = training
    proposal.current_live_points = current
    proposal.threshold_from_live_points = from_live
    proposal.sum_mirror_density = False
    proposal.latent_temperature = None
    proposal.forward_pass = MagicMock(
        side_effect=lambda x: (np.zeros((len(x), 1)), -(x["x"] ** 2))
    )
    proposal.compute_weights = MagicMock(
        side_effect=lambda x, log_q: 2.0 * x["x"] - log_q
    )
    return proposal


@pytest.mark.parametrize("from_live", [False, True])
def test_threshold_source(from_live):
    """The threshold comes from the current live points only with
    ``threshold_from_live_points``."""
    rng = np.random.default_rng(0)
    training = _arr(rng.uniform(-1, 1, 500))
    current = _arr(rng.uniform(-0.5, 0.5, 500))
    proposal = _proposal(training, current, from_live)
    rule = LogProposalThresholdTruncation(quantile=0.1, prior_relative=True)
    rule.prepare(proposal, worst_point=None)
    src = current if from_live else training
    level = -(src["x"] ** 2) - 2.0 * src["x"]
    assert rule.threshold == pytest.approx(np.quantile(level, 0.1))


def test_clip_requires_prior_relative():
    with pytest.raises(ValueError, match="prior_relative"):
        LogProposalThresholdTruncation(mode="clip")
    with pytest.raises(ValueError, match="mode"):
        LogProposalThresholdTruncation(prior_relative=True, mode="cut")


@pytest.mark.parametrize("mode", ["drop", "clip"])
def test_clip_keeps_draws_and_sets_cap(mode):
    """Drop discards the draws above the weight cap; clip keeps them (minus
    non-finite weights) and exposes the cap to the scheme."""
    rng = np.random.default_rng(1)
    live = _arr(rng.uniform(-1, 1, 1000), logl=rng.normal(size=1000))
    proposal = _proposal(live)
    rule = LogProposalThresholdTruncation(
        quantile=0.05,
        prior_relative=True,
        mode=mode,
        roundtrip_tolerance=None,
    )
    scheme = TruncationScheme([rule])
    scheme.prepare(proposal, worst_point=None)
    x = _arr(np.linspace(-1.5, 1.5, 301))
    log_q = -(x["x"] ** 2)
    log_q[0] = -np.inf  # infinite weight: dropped in both modes
    out, _, _ = scheme.apply_after_backward(
        proposal, x, log_q, np.zeros((len(x), 1))
    )
    log_w = 2.0 * x["x"] - log_q
    above = log_w > -rule.threshold
    if mode == "drop":
        assert len(out) == (~above).sum()
        assert scheme.log_weight_cap == np.inf
    else:
        assert len(out) == len(x) - 1
        assert scheme.log_weight_cap == pytest.approx(-rule.threshold)
    assert rule.excluded_live_fraction == pytest.approx(0.05, abs=0.002)
    assert 0.0 <= rule.excluded_live_mean_rank <= 1.0


def test_excluded_rank_low_edge():
    """Live points outside the cut at the lowest likelihoods give a low
    mean rank."""
    x = np.linspace(-1, 1, 1001)
    # the lowest levels (largest x) have the lowest likelihoods
    live = _arr(x, logl=-x)
    proposal = _proposal(live)
    rule = LogProposalThresholdTruncation(quantile=0.05, prior_relative=True)
    rule.prepare(proposal, worst_point=None)
    assert rule.excluded_live_mean_rank < 0.05
