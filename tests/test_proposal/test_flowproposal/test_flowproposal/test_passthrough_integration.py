# -*- coding: utf-8 -*-
"""A FlowProposal with a Gaussian block of nuisance parameters that pass
through its flow (passthrough_parameters + the gaussian-block
reparameterisation)."""

import numpy as np
import pytest
import torch
from scipy.stats import multivariate_normal, norm

from nessai.flows.passthrough import PassThroughFlow
from nessai.livepoint import dict_to_live_points
from nessai.model import Model
from nessai.proposal import FlowProposal

torch.set_num_threads(1)

NUISANCE = ["n0", "n1", "n2", "n3"]


class BlockModel(Model):
    """x, y uniform on [-5, 5]; four standard normal nuisance parameters."""

    def __init__(self):
        self.names = ["x", "y"] + NUISANCE
        self.bounds = {"x": [-5, 5], "y": [-5, 5]} | {
            n: [-np.inf, np.inf] for n in NUISANCE
        }

    def new_point(self, N=1):
        return dict_to_live_points(
            {"x": self.rng.uniform(-5, 5, N), "y": self.rng.uniform(-5, 5, N)}
            | {n: self.rng.standard_normal(N) for n in NUISANCE}
        )

    def new_point_log_prob(self, x):
        return self.log_prior(x)

    def log_prior(self, x):
        log_p = np.log(self.in_bounds(x), dtype=float) - 2 * np.log(10)
        for n in NUISANCE:
            log_p += norm.logpdf(x[n])
        return log_p

    def log_likelihood(self, x):
        return norm.logpdf(x["x"]) + norm.logpdf(x["y"]) + norm.logpdf(x["n0"], scale=0.5)


@pytest.mark.integration_test
@pytest.mark.timeout(120)
def test_passthrough_block(tmp_path):
    model = BlockModel()
    model.rng = np.random.default_rng(0)
    fp = FlowProposal(
        model,
        output=str(tmp_path),
        flow_config=dict(n_neurons=8, n_blocks=2, n_layers=1),
        training_config=dict(max_epochs=20, patience=5),
        plot=False,
        poolsize=200,
        reparameterisations={
            "n0": {"reparameterisation": "gaussian-block", "parameters": NUISANCE}
        },
        passthrough_parameters=NUISANCE,
    )
    fp.initialise()
    assert fp.passthrough_prime_parameters == [f"{n}_prime" for n in NUISANCE]
    assert isinstance(fp.flow.model, PassThroughFlow)
    passthrough = [fp.prime_parameters.index(f"{n}_prime") for n in NUISANCE]
    assert sorted(fp.flow.model.passthrough_features) == sorted(passthrough)
    assert not fp.latent_real_mask[passthrough].any()

    # training points: the nuisance block correlated and shrunk (n0 by the
    # likelihood), as live points would be
    rng = np.random.default_rng(1)
    n = 2000
    cov = np.eye(4)
    cov[0, 0] = 0.2
    cov[1, 2] = cov[2, 1] = 0.5
    v = rng.multivariate_normal(np.zeros(4), cov, size=n)
    train = dict_to_live_points(
        {"x": rng.standard_normal(n), "y": rng.standard_normal(n)}
        | {k: v[:, i] for i, k in enumerate(NUISANCE)}
    )
    train["logL"] = model.log_likelihood(train)
    train["logP"] = model.log_prior(train)
    fp.train(train, plot=False)
    block = next(
        r for r in fp._reparameterisation.values() if type(r).__name__ == "GaussianBlock"
    )
    assert block.n_updates >= 1
    np.testing.assert_allclose(np.sqrt(np.diag(block.covariance))[0], np.sqrt(0.2), rtol=0.1)

    worst = train[np.argmin(train["logL"])]
    fp.populate(worst, n_samples=200)
    assert fp.x.size == 200
    # the pool's density: the flow of (x, y) times the block's Gaussian
    x_prime, log_j = fp.rescale(fp.x.copy())
    from nessai.livepoint import live_points_to_array

    arr = np.ascontiguousarray(live_points_to_array(x_prime, fp.prime_parameters), dtype=float)
    with torch.no_grad():
        log_q = fp.flow.log_prob(arr) + log_j
    keep = [i for i in range(len(fp.prime_parameters)) if i not in passthrough]
    inner_model = fp.flow.model
    with torch.no_grad():
        t = torch.as_tensor(arr[:, keep], dtype=torch.float32)
        z, lj = inner_model._transform.transform.forward(t)
        log_q_inner = (inner_model._distribution.inner.log_prob(z) + lj).numpy()
    nuis = np.column_stack([fp.x[k] for k in NUISANCE])
    log_q_block = multivariate_normal(block.mean, block.covariance).logpdf(nuis)
    # the (x, y) part's rescaling Jacobian
    log_j_xy = log_j - (-block.log_det)
    np.testing.assert_allclose(
        log_q, log_q_inner + log_j_xy + log_q_block, rtol=1e-4, atol=1e-3
    )
