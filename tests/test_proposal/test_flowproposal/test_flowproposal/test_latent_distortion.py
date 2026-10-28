# -*- coding: utf-8 -*-
"""Latent distortion of the flow proposal: the algebra, the fit, the checks
and the wiring into training, resume and the clip-mode round-trip guard."""

import pickle
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from scipy.special import logsumexp
from scipy.stats import norm

from nessai.livepoint import empty_structured_array
from nessai.proposal.flowproposal import FlowProposal
from nessai.proposal.flowproposal import latent_distortion as LD
from nessai.proposal.flowproposal.truncation import (
    LogProposalThresholdTruncation,
)


def _arr(v):
    a = empty_structured_array(len(v), names=["x"])
    a["x"] = v
    return a


def test_delta_and_log_q_distorted():
    """``Delta`` is the log-ratio of the distorted to the standard latent
    density on the real dims, and ``log_q_distorted`` the log-sum-exp over
    sides and experts; numpy and torch agree."""
    rng = np.random.default_rng(3)
    z = rng.normal(size=(2, 5, 3, 4))  # sides, n, experts, dims
    lp = rng.normal(size=(2, 5, 3))
    mu = rng.normal(scale=0.3, size=(3, 4))
    ls = rng.normal(scale=0.2, size=(3, 4))
    real = np.array([True, True, False, True])
    expect = (norm.logpdf(z, mu, np.exp(ls)) - norm.logpdf(z))[..., real].sum(
        -1
    )
    np.testing.assert_allclose(LD.delta(z, mu, ls, real), expect)
    t = (lp + expect).transpose(1, 0, 2).reshape(5, -1)
    np.testing.assert_allclose(
        LD.log_q_distorted(lp, z, mu, ls, real), logsumexp(t, axis=1)
    )
    tt = LD.log_q_distorted(
        torch.as_tensor(lp),
        torch.as_tensor(z),
        torch.as_tensor(mu),
        torch.as_tensor(ls),
        torch.as_tensor(real),
    )
    np.testing.assert_allclose(
        tt.numpy(), LD.log_q_distorted(lp, z, mu, ls, real)
    )


@pytest.mark.parametrize("mode", LD.MODES)
def test_fit_broadens_for_wide_target(mode):
    """A standard Gaussian flow whose target is twice as wide: the fit
    raises the low tail of ``log q' - log p`` (sigma > 1) within the
    bounds; the identity parameters are a fixed point of the bounds."""
    rng = np.random.default_rng(4)
    n, d = 2000, 3
    x = rng.normal(scale=2.0, size=(n, d))
    z = x[None, :, None, :]
    lp = (-0.5 * x**2 - 0.5 * np.log(2 * np.pi)).sum(-1)[None, :, None]
    log_p = (-0.5 * (x / 2) ** 2 - np.log(2) - 0.5 * np.log(2 * np.pi)).sum(-1)
    real = np.ones(d, dtype=bool)
    mu0, ls0 = LD._unpack(LD._theta0((1, d), real, mode), (1, d), real, mode)
    np.testing.assert_allclose(mu0.detach().numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(ls0.detach().numpy(), 0.0, atol=1e-12)
    mu, ls = LD.fit(
        lp, z, log_p, real, mode=mode, tail=0.02, reg=1e-3, steps=300
    )
    assert np.all(np.exp(ls) > 1.3)
    assert np.all(np.exp(ls) <= LD.SIGMA_MAX + 1e-9)
    assert np.all(np.abs(mu) <= LD.MU_MAX + 1e-9)
    zero = 0 * mu
    base = np.quantile(
        LD.log_q_distorted(lp, z, zero, zero, real) - log_p, 0.01
    )
    new = np.quantile(LD.log_q_distorted(lp, z, mu, ls, real) - log_p, 0.01)
    assert new > base + 1.0


@pytest.fixture()
def proposal(model, flow_config, training_config, tmp_path):
    fp = FlowProposal(
        model,
        output=str(tmp_path / "fp"),
        flow_config=flow_config,
        training_config=training_config,
        plot=False,
        poolsize=100,
        truncation_method="log_proposal_threshold",
        truncation_kwargs=dict(quantile=0.05, mode="clip"),
        latent_distortion="expert",
        latent_distortion_kwargs=dict(steps=100),
    )
    fp.initialise()
    return fp


def _live(model, n, scale, rng):
    x = empty_structured_array(n, names=model.names)
    for name in model.names:
        x[name] = np.clip(rng.standard_t(3, size=n) * scale, -4.9, 4.9)
    x["logP"] = model.log_prior(x)
    x["logL"] = model.log_likelihood(x)
    return x


def test_fit_and_check_plain_flow(proposal, model, rng):
    """On a plain flow the fit runs through the decomposition: a fit that
    passes reproduces the flow's distorted density (the exactness check),
    and either way the info names the outcome."""
    x = _live(model, 400, 1.5, rng)
    assert LD.supports_distortion(proposal.flow.model)
    mu, ls, info = LD.fit_and_check(proposal, x, 0.05, steps=100)
    assert "heldout_gain" in info
    if mu is None:
        assert info["reason"]
    else:
        assert info["accepted"]
        assert mu.shape == ls.shape == (1, proposal.prime_dims)
        assert info["exactness_error"] < 1e-3
    assert proposal.flow.model.latent_distortion is None


def test_distortion_conflicts_with_temperature(model):
    with pytest.raises(ValueError, match="latent temperature"):
        FlowProposal(
            model,
            plot=False,
            poolsize=10,
            latent_distortion="expert",
            latent_temperature=2,
        )


def test_invalid_distortion_mode(model):
    with pytest.raises(ValueError, match="latent_distortion"):
        FlowProposal(model, plot=False, poolsize=10, latent_distortion="bad")


def test_train_installs_and_clears(proposal, model, rng, monkeypatch):
    """Training clears the distortion, then installs an accepted fit and
    records it; resume re-applies it after reloading the weights."""
    d = proposal.prime_dims
    mu, ls = np.full((1, d), 0.2), np.full((1, d), 0.3)
    calls = []

    def fake_fit(p, x, quantile, **kwargs):
        # trained undistorted
        calls.append(p.flow.model.latent_distortion)
        return (
            mu,
            ls,
            dict(accepted=True, heldout_gain=1.0, heldout_excluded=0.05),
        )

    monkeypatch.setattr(LD, "fit_and_check", fake_fit)
    x = _live(model, 200, 1.0, rng)
    proposal.train(x, plot=False)
    proposal.train(x, plot=False)
    assert calls == [None, None]
    np.testing.assert_allclose(proposal.flow.model.latent_distortion[0], mu[0])
    assert len(proposal.latent_distortion_history) == 2
    assert proposal.latent_distortion_history[-1]["accepted"]

    weights = proposal.flow.weights_file
    flow_config = proposal.flow_config
    resumed = pickle.loads(pickle.dumps(proposal))
    resumed.resume(model, flow_config, weights_file=weights)
    np.testing.assert_allclose(resumed.flow.model.latent_distortion[1], ls[0])


def test_roundtrip_discard(monkeypatch):
    """In clip mode the draws above the cap that fail the round-trip check
    are discarded; the others are kept."""
    rng = np.random.default_rng(2)
    live = _arr(rng.uniform(-1, 1, 1000))
    proposal = MagicMock()
    proposal.training_data = live
    proposal.current_live_points = None
    proposal.threshold_from_live_points = False
    proposal.sum_mirror_density = False
    proposal.latent_temperature = None
    proposal.forward_pass = MagicMock(
        side_effect=lambda x: (np.zeros((len(x), 1)), -(x["x"] ** 2))
    )
    proposal.compute_weights = MagicMock(
        side_effect=lambda x, log_q: 2.0 * x["x"] - log_q
    )
    rule = LogProposalThresholdTruncation(
        quantile=0.05, prior_relative=True, mode="clip"
    )
    rule.prepare(proposal, worst_point=None)
    x = _arr(np.linspace(-1.5, 1.5, 31))
    log_q = -(x["x"] ** 2)
    above = -(2.0 * x["x"] - log_q) <= rule.threshold
    assert above.sum() >= 2
    gap = np.zeros(above.sum())
    gap[0] = 30.0
    monkeypatch.setattr(LD, "roundtrip_gap", lambda p, xx, zz: gap)
    out, _, _ = rule.apply_after_backward(
        proposal, x, log_q, np.zeros((len(x), 1))
    )
    assert len(out) == len(x) - 1
    assert rule.roundtrip_discarded == 1
