"""Latent distortion of a plain flow (:class:`nessai.flows.base.NFlow`)."""

import numpy as np
import pytest
import torch

from nessai.flows import RealNVP


@pytest.fixture
def flow():
    torch.manual_seed(0)
    f = RealNVP(3, 8, 2, 2).double()
    f.eval()
    return f


@pytest.fixture
def distortion():
    return np.array([0.3, -0.5, 0.1]), np.array([0.2, -0.4, 0.5])


def test_identity_by_default(flow):
    assert flow.latent_distortion is None
    x = torch.randn(10, 3, dtype=torch.float64)
    with torch.no_grad():
        lp, z = flow.latent_components(x)
        np.testing.assert_allclose(lp, flow.log_prob(x), rtol=0, atol=1e-6)
        np.testing.assert_allclose(z, flow.forward(x)[0], rtol=0, atol=1e-6)


def test_distorted_density_is_exact(flow, distortion):
    """``log_prob``, ``forward_and_log_prob``, ``sample_and_log_prob`` and
    ``inverse`` agree with ``log q + Delta(z)`` of the undistorted terms."""
    mu, ls = distortion
    torch.manual_seed(1)
    with torch.no_grad():
        flow.set_latent_distortion(mu, ls)
        x, lq = flow.sample_and_log_prob(200)
        np.testing.assert_allclose(flow.log_prob(x), lq, atol=1e-8)
        z, lq2 = flow.forward_and_log_prob(x)
        np.testing.assert_allclose(lq2, lq, atol=1e-8)
        x2, log_j = flow.inverse(z)
        np.testing.assert_allclose(x2, x, atol=1e-8)
        np.testing.assert_allclose(
            flow.base_distribution_log_prob(z) - log_j, lq, atol=1e-5
        )
        flow.set_latent_distortion(None)
        lp, zu = flow.latent_components(x)
    zu, lp = zu.numpy(), lp.numpy()
    delta = (
        -0.5 * ((zu - mu) / np.exp(ls)) ** 2 - ls + 0.5 * zu**2
    ).sum(-1)
    np.testing.assert_allclose(lp + delta, lq.numpy(), atol=1e-8)


def test_distorted_draws_follow_the_shifted_latent(flow, distortion):
    mu, ls = distortion
    torch.manual_seed(2)
    with torch.no_grad():
        flow.set_latent_distortion(mu, ls)
        x = flow.sample(20000)
        flow.set_latent_distortion(None)
        z = flow.forward(x)[0].numpy()
    np.testing.assert_allclose(z.mean(0), mu, atol=0.05)
    np.testing.assert_allclose(z.std(0), np.exp(ls), rtol=0.05)


def test_state_dict_keys_unchanged(flow, distortion):
    keys = set(flow.state_dict())
    flow.set_latent_distortion(*distortion)
    assert set(flow.state_dict()) == keys


@pytest.mark.integration_test
def test_fit_with_auxiliary_radius(tmp_path, rng):
    """The fit evaluates the prior of the live points, which do not carry
    the auxiliary radius of an angle (e.g. bilby's ``time_jitter`` under
    time marginalisation, sampled with GWFlowProposal's periodic angle):
    it is drawn once for them."""
    from scipy.stats import norm

    from nessai.model import Model
    from nessai.proposal import FlowProposal

    class AngleModel(Model):
        def __init__(self):
            self.names = ["x", "phi"]
            self.bounds = {"x": [-5, 5], "phi": [0, 2 * np.pi]}

        def log_prior(self, x):
            log_p = np.log(self.in_bounds(x), dtype="float")
            return log_p - np.log(10 * 2 * np.pi)

        def log_likelihood(self, x):
            return norm.logpdf(x["x"]) + 2 * np.cos(x["phi"])

    model = AngleModel()
    model.set_rng(rng)
    output = tmp_path / "flowproposal"
    output.mkdir()
    fp = FlowProposal(
        model,
        output=output,
        flow_config=dict(n_neurons=4, n_blocks=2, n_layers=1),
        plot=False,
        poolsize=100,
        reparameterisations={"phi": "angle-2pi"},
        truncation_methods=["log_proposal_threshold"],
        truncation_kwargs={
            "log_proposal_threshold": {"quantile": 0.1, "prior_relative": True}
        },
        latent_distortion="shared",
    )
    fp.initialise()
    live = model.new_point(300)
    live["logL"] = model.batch_evaluate_log_likelihood(live)
    fp.train(live, plot=False)
    info = fp.latent_distortion_history[-1]
    assert info.get("reason") != "flow model not supported"
    assert "heldout_gain" in info or "reason" in info
