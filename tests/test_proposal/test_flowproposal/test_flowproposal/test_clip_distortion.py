# -*- coding: utf-8 -*-
"""Clip mode and live-point threshold of the log-proposal truncation, and
the latent-distortion algebra."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from nessai.livepoint import empty_structured_array
from nessai.proposal.flowproposal import latent_distortion as LD
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
        side_effect=lambda x: (np.zeros((len(x), 1)), -x["x"] ** 2)
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
        quantile=0.05, prior_relative=True, mode=mode,
        roundtrip_tolerance=None,
    )
    scheme = TruncationScheme([rule])
    scheme.prepare(proposal, worst_point=None)
    x = _arr(np.linspace(-1.5, 1.5, 301))
    log_q = -x["x"] ** 2
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


def test_roundtrip_discard(monkeypatch):
    """In clip mode the draws above the cap that fail the round-trip check
    are discarded; the others are kept."""
    from nessai.proposal.flowproposal import latent_distortion

    rng = np.random.default_rng(2)
    live = _arr(rng.uniform(-1, 1, 1000))
    proposal = _proposal(live)
    rule = LogProposalThresholdTruncation(
        quantile=0.05, prior_relative=True, mode="clip",
        roundtrip_tolerance=2.0,
    )
    rule.prepare(proposal, worst_point=None)
    x = _arr(np.linspace(-1.5, 1.5, 31))
    log_q = -x["x"] ** 2
    above = 2.0 * x["x"] - log_q > -rule.threshold
    assert above.sum() >= 2
    z = np.zeros((len(x), 1))
    gap = np.zeros(above.sum())
    gap[0] = 30.0
    monkeypatch.setattr(
        latent_distortion, "roundtrip_gap", lambda p, xx, zz: gap
    )
    out, _, _ = rule.apply_after_backward(proposal, x, log_q, z)
    assert len(out) == len(x) - 1
    assert rule.roundtrip_discarded == 1


def test_delta_and_log_q_distorted():
    """``Delta`` is the log-ratio of the distorted to the standard latent
    density on the real dims, and ``log_q_distorted`` the log-sum-exp over
    sides and experts."""
    rng = np.random.default_rng(3)
    z = rng.normal(size=(2, 5, 3, 4))  # sides, n, experts, dims
    lp = rng.normal(size=(2, 5, 3))
    mu = rng.normal(scale=0.3, size=(3, 4))
    ls = rng.normal(scale=0.2, size=(3, 4))
    real = np.array([True, True, False, True])
    from scipy.stats import norm

    expect = (
        norm.logpdf(z, mu, np.exp(ls)) - norm.logpdf(z)
    )[..., real].sum(-1)
    np.testing.assert_allclose(LD.delta(z, mu, ls, real), expect)
    t = (lp + expect).transpose(1, 0, 2).reshape(5, -1)
    from scipy.special import logsumexp

    np.testing.assert_allclose(
        LD.log_q_distorted(lp, z, mu, ls, real), logsumexp(t, axis=1)
    )
    # numpy and torch agree
    tt = LD.log_q_distorted(
        torch.as_tensor(lp), torch.as_tensor(z), torch.as_tensor(mu),
        torch.as_tensor(ls), torch.as_tensor(real),
    )
    np.testing.assert_allclose(tt.numpy(), LD.log_q_distorted(
        lp, z, mu, ls, real))


@pytest.mark.parametrize("mode", LD.MODES)
def test_fit_broadens_for_wide_target(mode):
    """A single Gaussian expert whose target is twice as wide: the fit
    raises the low tail of ``log q' - log p`` (sigma > 1) and stays within
    the bounds; the identity parameters are a fixed point of the bounds."""
    rng = np.random.default_rng(4)
    n, d = 2000, 3
    x = rng.normal(scale=2.0, size=(n, d))
    z = x[None, :, None, :]                       # identity flow
    lp = (-0.5 * x**2 - 0.5 * np.log(2 * np.pi)).sum(-1)[None, :, None]
    log_p = (-0.5 * (x / 2) ** 2 - np.log(2) - 0.5 * np.log(2 * np.pi)).sum(-1)
    real = np.ones(d, dtype=bool)
    mu0, ls0 = LD._unpack(LD._theta0((1, d), real, mode), (1, d), real, mode)
    np.testing.assert_allclose(mu0.detach().numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(ls0.detach().numpy(), 0.0, atol=1e-12)
    mu, ls = LD.fit(lp, z, log_p, real, mode=mode, tail=0.02, reg=1e-3,
                    steps=300)
    assert np.all(np.exp(ls) > 1.3)
    assert np.all(np.exp(ls) <= LD.SIGMA_MAX + 1e-9)
    assert np.all(np.abs(mu) <= LD.MU_MAX + 1e-9)
    base = np.quantile(LD.log_q_distorted(lp, z, 0 * mu, 0 * ls, real)
                       - log_p, 0.01)
    new = np.quantile(LD.log_q_distorted(lp, z, mu, ls, real) - log_p, 0.01)
    assert new > base + 1.0


def test_roundtrip_failure_keeps_draws(monkeypatch, caplog):
    """A failing round-trip check does not stop the population: the draws
    are kept and a warning is logged once."""
    from nessai.proposal.flowproposal import latent_distortion

    rng = np.random.default_rng(5)
    proposal = _proposal(_arr(rng.uniform(-1, 1, 1000)))
    rule = LogProposalThresholdTruncation(
        quantile=0.05, prior_relative=True, mode="clip"
    )
    rule.prepare(proposal, worst_point=None)

    def boom(*args):
        raise IndexError("table lookup")

    monkeypatch.setattr(latent_distortion, "roundtrip_gap", boom)
    x = _arr(np.linspace(-1.5, 1.5, 31))
    log_q = -x["x"] ** 2
    with caplog.at_level("WARNING"):
        for _ in range(2):
            out, _, _ = rule.apply_after_backward(
                proposal, x, log_q, np.zeros((len(x), 1))
            )
            assert len(out) == len(x)
    assert sum("Round-trip check failed" in r.message for r in caplog.records) == 1
