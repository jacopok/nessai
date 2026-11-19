# -*- coding: utf-8 -*-
"""Per-element latent Gaussians of the group-mixture wrapper
(``set_element_latent``) and their fit
(:mod:`nessai.proposal.flowproposal.element_latent`)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nessai.flowmodel.group_mixture import DiscreteGroupMixtureFlowWrapper
from nessai.flows.utils import configure_model
from nessai.proposal.flowproposal import element_latent as EL

GROUP_SIZE = 4


@pytest.fixture(autouse=True)
def float64():
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)


def shift_group_action(point_dict, modes, inverse=False):
    shift = modes.to(point_dict["x"].dtype)
    x = point_dict["x"] - shift if inverse else point_dict["x"] + shift
    return {"x": x, "y": point_dict["y"]}


def in_fundamental_domain(point_dict):
    return (point_dict["x"] >= 0.0) & (point_dict["x"] < 1.0)


def wide_group_action(point_dict, modes, inverse=False):
    shift = 10.0 * modes.to(point_dict["x"].dtype)
    x = point_dict["x"] - shift if inverse else point_dict["x"] + shift
    return {"x": x, "y": point_dict["y"]}


def wide_domain(point_dict):
    return point_dict["x"].abs() < 5.0


def make_wrapper(truncate=True, wide=False):
    torch.manual_seed(0)
    base_flow = configure_model(
        {
            "n_inputs": 2,
            "ftype": "realnvp",
            "n_blocks": 2,
            "n_neurons": 4,
            "n_layers": 1,
            "batch_norm_between_layers": False,
        }
    ).double()
    flow = DiscreteGroupMixtureFlowWrapper(
        base_flow=base_flow,
        num_features=2,
        group_action_fn=wide_group_action if wide else shift_group_action,
        group_size=GROUP_SIZE,
        param_names=["x", "y"],
        in_fundamental_domain=wide_domain if wide else in_fundamental_domain,
        truncate_base_to_domain=truncate,
    )
    flow.eval()
    return flow


def element_latent():
    mu = np.zeros((GROUP_SIZE, 2))
    chol = np.tile(np.eye(2), (GROUP_SIZE, 1, 1))
    mu[1] = [0.4, -0.3]
    chol[1] = [[0.5, 0.0], [0.2, 0.3]]
    mu[3] = [-0.2, 0.1]
    chol[3] = [[0.7, 0.0], [0.0, 1.4]]
    return mu, chol


def test_identity_by_default():
    w = make_wrapper()
    assert w.element_latent is None
    x = torch.tensor([[0.3, 0.1], [1.4, -0.5], [3.7, 0.2]])
    with torch.no_grad():
        lp = w.log_prob(x)
        w.set_element_latent(np.zeros((GROUP_SIZE, 2)), np.tile(np.eye(2), (GROUP_SIZE, 1, 1)))
        np.testing.assert_allclose(w.log_prob(x), lp, atol=1e-12)


def test_density_is_exact_on_every_path():
    """Draws (sample_and_log_prob and inverse) are scored with log_prob,
    and forward inverts inverse (the domain-truncated base, the default:
    without truncation an out-of-domain draw is scored approximately with
    or without element latents)."""
    w = make_wrapper()
    w.set_element_latent(*element_latent())
    torch.manual_seed(1)
    with torch.no_grad():
        x, lq = w.sample_and_log_prob(3000)
        np.testing.assert_allclose(lq, w.log_prob(x), atol=1e-8)
        z = torch.randn(500, 2)
        x2, log_j = w.inverse(z)
        ok = torch.isfinite(log_j)
        np.testing.assert_allclose(
            (w.base_distribution_log_prob(z) - log_j)[ok], w.log_prob(x2)[ok], atol=1e-8
        )
        z2, _ = w.forward(x2[ok])
        np.testing.assert_allclose(z2, z[ok], atol=1e-6)


def test_density_is_normalised():
    """The truncated density integrates to one with element latents (the
    redraw loop makes the in-domain mass a single constant, sum_g w_g Z_g):
    importance sampling with a broad uniform proposal over the tiles."""
    w = make_wrapper()
    w.set_element_latent(*element_latent())
    torch.manual_seed(8)
    w.measure_domain_mass(400_000)
    rng = np.random.default_rng(3)
    n = 400_000
    x = np.stack([rng.uniform(0, GROUP_SIZE, n), rng.uniform(-12, 12, n)], 1)
    with torch.no_grad():
        q = torch.exp(w.log_prob(torch.as_tensor(x))).numpy()
    vol = GROUP_SIZE * 24.0
    assert q.mean() * vol == pytest.approx(1.0, abs=0.03)


def test_draws_follow_the_element_gaussian():
    """Element 1's draws have base noise N(m_1, L_1 L_1^T)."""
    w = make_wrapper(wide=True)
    mu, chol = element_latent()
    w.set_element_latent(mu, chol)
    w.set_branch_weights(np.array([0.0, 1.0, 0.0, 0.0]))
    torch.manual_seed(2)
    with torch.no_grad():
        x, _ = w.sample_and_log_prob(20000)
        n = noise_of(w, x, 1)
    np.testing.assert_allclose(n.mean(0), mu[1], atol=0.03)
    np.testing.assert_allclose(np.cov(n.numpy().T), chol[1] @ chol[1].T, atol=0.03)


def test_distortion_decomposition_stays_exact():
    """With the element latents in place, log q = log q_undistorted +
    Delta(z) with z from latent_components (the fits' decomposition)."""
    w = make_wrapper()
    w.set_element_latent(*element_latent())
    mu_d, ls_d = np.array([0.2, -0.1]), np.array([0.1, -0.3])
    torch.manual_seed(4)
    with torch.no_grad():
        w.set_latent_distortion(mu_d, ls_d)
        x, lq = w.sample_and_log_prob(1000)
        lp, z = w.latent_components(x)
    z = z.numpy()
    delta = (-0.5 * ((z - mu_d) / np.exp(ls_d)) ** 2 - ls_d + 0.5 * z**2).sum(-1)
    np.testing.assert_allclose(lp.numpy() + delta, lq.numpy(), atol=1e-8)


def noise_of(w, x, g):
    """Base noise of draws ``x`` made in element ``g`` (inside the domain or
    not)."""
    canon = x.clone()
    canon[:, 0] = x[:, 0] - (10.0 * g if w.group_action_fn is wide_group_action else g)
    t, _ = w._canon_to_base(canon)
    return w.base_flow.forward(w._standardise(t, torch.full((len(x),), g)))[0]


def _proposal(w, x):
    """The bits of a FlowProposal the fit reads."""
    return SimpleNamespace(flow=SimpleNamespace(model=w), latent_real_mask=None, _x=x)


def test_fit_recovers_the_element_gaussians(monkeypatch):
    """Live points drawn with known element latents (a large primary in
    element 0, small secondaries) give those latents back, the primary and
    elements under min_count untouched."""
    w = make_wrapper(wide=True)
    mu, chol = element_latent()
    w.set_element_latent(mu, chol)
    torch.manual_seed(5)
    with torch.no_grad():
        parts = []
        for g, m in ((0, 6000), (1, 3000), (2, 5), (3, 3000)):
            wts = np.zeros(GROUP_SIZE)
            wts[g] = 1.0
            w.set_branch_weights(wts)
            parts.append(w.sample_and_log_prob(m)[0])
        x = torch.cat(parts)
    w.set_branch_weights(None)
    w.set_element_latent(None)
    monkeypatch.setattr(EL, "_prime_tensor", lambda p, x: (x, None, 1))
    params, info = EL.fit(_proposal(w, x), x, inflation=1.0, max_share=0.45)
    assert info["elements"] == [[1, 3]]
    m_fit, c_fit = params[0]
    for g in (1, 3):
        np.testing.assert_allclose(m_fit[g], mu[g], atol=0.05)
        np.testing.assert_allclose(c_fit[g] @ c_fit[g].T, chol[g] @ chol[g].T, atol=0.06)
    for g in (0, 2):
        np.testing.assert_allclose(m_fit[g], 0.0)
        np.testing.assert_allclose(c_fit[g], np.eye(2))


def test_fit_accounts_for_the_distortion(monkeypatch):
    """With a latent distortion in place the element's noise still follows
    the fitted Gaussian: n = m + L (mu + sigma z) ~ N(c, S)."""
    w = make_wrapper(wide=True)
    mu, chol = element_latent()
    w.set_element_latent(mu, chol)
    torch.manual_seed(6)
    with torch.no_grad():
        wts = np.array([0.0, 1.0, 0.0, 0.0])
        w.set_branch_weights(wts)
        x1 = w.sample_and_log_prob(4000)[0]
        w.set_branch_weights(np.array([1.0, 0.0, 0.0, 0.0]))
        x0 = w.sample_and_log_prob(6000)[0]
    x = torch.cat([x0, x1])
    w.set_branch_weights(None)
    w.set_element_latent(None)
    w.set_latent_distortion(np.array([0.5, -0.4]), np.array([0.3, -0.2]))
    monkeypatch.setattr(EL, "_prime_tensor", lambda p, x: (x, None, 1))
    params, _ = EL.fit(_proposal(w, x), x, inflation=1.0)
    w.set_element_latent(*params[0])
    w.set_branch_weights(wts)
    torch.manual_seed(7)
    with torch.no_grad():
        y = w.sample_and_log_prob(20000)[0]
        n = noise_of(w, y, 1)
    np.testing.assert_allclose(n.mean(0), mu[1], atol=0.04)
    np.testing.assert_allclose(np.cov(n.numpy().T), chol[1] @ chol[1].T, atol=0.04)
