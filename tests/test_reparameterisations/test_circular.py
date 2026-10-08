# -*- coding: utf-8 -*-
"""
Test the Circular reparameterisation and its use in FlowProposal.
"""

import numpy as np
import pytest
from scipy.stats import norm

from nessai.livepoint import empty_structured_array, numpy_array_to_live_points
from nessai.model import Model
from nessai.proposal.flowproposal import FlowProposal
from nessai.reparameterisations import Circular, get_reparameterisation


@pytest.fixture
def circular():
    return Circular(
        parameters=["phi", "psi"],
        prior_bounds={"phi": [0.0, 2 * np.pi], "psi": [1.0, 1.0 + np.pi]},
    )


def test_registered():
    assert get_reparameterisation("circular")[0] is Circular


def test_output_names(circular):
    assert circular.output_parameters == ["phi_circ", "psi_circ"]
    assert circular.circular_parameters == ["phi_circ", "psi_circ"]


def test_round_trip(circular, rng):
    n = 1000
    x = numpy_array_to_live_points(
        np.stack(
            [rng.uniform(0, 2 * np.pi, n), rng.uniform(1, 1 + np.pi, n)],
            axis=1,
        ),
        ["phi", "psi"],
    )
    x_prime = empty_structured_array(n, names=circular.output_parameters)
    log_j = np.zeros(n)
    x, x_prime, log_j = circular.reparameterise(x, x_prime, log_j)
    for p in circular.output_parameters:
        assert (x_prime[p] >= -np.pi).all() and (x_prime[p] < np.pi).all()
    np.testing.assert_allclose(log_j, np.log(2.0))  # 1 (phi) * 2 (psi)

    x_out = empty_structured_array(n, names=["phi", "psi"])
    log_j_inv = np.zeros(n)
    x_out, _, log_j_inv = circular.inverse_reparameterise(
        x_out, x_prime, log_j_inv
    )
    np.testing.assert_allclose(x_out["phi"], x["phi"], atol=1e-12)
    np.testing.assert_allclose(x_out["psi"], x["psi"], atol=1e-12)
    np.testing.assert_allclose(log_j_inv, -log_j)


def test_wraps(circular):
    """Upper prior bound and out-of-range angles map onto the period."""
    x = numpy_array_to_live_points(
        np.array([[2 * np.pi, 1.0 + np.pi]]), ["phi", "psi"]
    )
    x_prime = empty_structured_array(1, names=circular.output_parameters)
    _, x_prime, _ = circular.reparameterise(x, x_prime, np.zeros(1))
    assert x_prime["phi_circ"][0] == -np.pi
    x_prime["phi_circ"] = 3 * np.pi + 0.5
    x_prime["psi_circ"] = -np.pi - 0.5
    x_out = empty_structured_array(1, names=["phi", "psi"])
    x_out, _, _ = circular.inverse_reparameterise(
        x_out, x_prime, np.zeros(1)
    )
    np.testing.assert_allclose(x_out["phi"], 0.5)
    np.testing.assert_allclose(x_out["psi"], 1.0 + (2 * np.pi - 0.5) / 2)


class PeriodicModel(Model):
    """A peak that crosses the period boundary of phi."""

    def __init__(self):
        self.names = ["phi", "x"]
        self.bounds = {"phi": [0, 2 * np.pi], "x": [-5, 5]}

    def log_prior(self, x):
        return np.log(self.in_bounds(x), dtype=float) - np.log(
            20 * np.pi
        )

    def log_likelihood(self, x):
        return 5 * np.cos(x["phi"]) + norm.logpdf(x["x"])


@pytest.mark.integration_test
@pytest.mark.parametrize("temperature", [None, 1.5])
def test_flowproposal_with_circular_flow(tmp_path, temperature):
    """The circular flow is built, trained and sampled through FlowProposal"""
    from nessai.flows.circular import CircularNeuralSplineFlow

    model = PeriodicModel()
    model.set_rng(np.random.default_rng(0))
    proposal = FlowProposal(
        model,
        output=str(tmp_path),
        poolsize=200,
        reparameterisations={"phi": "circular"},
        flow_config={"ftype": "circular", "n_blocks": 2, "n_neurons": 8},
        training_config={"max_epochs": 3},
        latent_temperature=temperature,
    )
    proposal.initialise()
    assert isinstance(proposal.flow.model, CircularNeuralSplineFlow)
    phi_index = proposal.prime_parameters.index("phi_circ")
    assert proposal.flow.model.circular_features == [phi_index]
    assert proposal.latent_real_mask.tolist() == [
        p != "phi_circ" for p in proposal.prime_parameters
    ]

    z = proposal.sample_latent_distribution(1000)
    assert (np.abs(z[:, phi_index]) <= np.pi).all()

    x = model.new_point(500)
    x["logL"] = model.log_likelihood(x)
    proposal.train(x, plot=False)
    worst = x[np.argmin(x["logL"])]
    proposal.populate(worst, n_samples=50)
    assert proposal.populated
    samples = proposal.samples
    assert (samples["phi"] >= 0).all() and (samples["phi"] <= 2 * np.pi).all()
    assert np.isfinite(model.log_prior(samples)).all()
