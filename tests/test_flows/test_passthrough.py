# -*- coding: utf-8 -*-
"""Pass-through features of a flow (flows.passthrough, configure_model)."""

import math

import numpy as np
import pytest
import torch

from nessai.flows.passthrough import PassThroughFlow
from nessai.flows.utils import configure_model

torch.set_num_threads(1)


def _config(n, **kwargs):
    return dict(n_inputs=n, n_neurons=4, n_blocks=2, n_layers=1, ftype="realnvp", **kwargs)


def _normal(x):
    return -0.5 * (x**2).sum(1) - 0.5 * x.shape[1] * math.log(2 * math.pi)


@pytest.mark.parametrize("passthrough", [[0], [1, 3], [4, 5, 6]])
def test_log_prob_factorises(passthrough):
    """log q = the inner flow's log q of the other features + a standard
    normal in the pass-through ones, and the inverse leaves them alone."""
    torch.manual_seed(0)
    n = 7
    flow = configure_model(_config(n, passthrough_features=passthrough))
    assert isinstance(flow, PassThroughFlow)
    assert flow.passthrough_features == sorted(passthrough)
    keep = [i for i in range(n) if i not in passthrough]
    # perturb the inner flow away from the identity
    with torch.no_grad():
        for p in flow.parameters():
            p.add_(0.1 * torch.randn_like(p))
    flow.eval()
    x = torch.randn(50, n, dtype=torch.float64).float()
    with torch.no_grad():
        log_q = flow.log_prob(x)
        z, log_j = flow._transform.transform.forward(x[:, keep])
        inner = flow._distribution.inner.log_prob(z) + log_j
        z_full, _ = flow.forward(x)
        x_back, _ = flow.inverse(z_full)
    np.testing.assert_allclose(
        log_q.numpy(), (inner + _normal(x[:, passthrough])).numpy(), rtol=1e-5
    )
    np.testing.assert_array_equal(z_full[:, passthrough].numpy(), x[:, passthrough].numpy())
    np.testing.assert_allclose(x_back.numpy(), x.numpy(), atol=1e-3)  # float32 LU


def test_sample_and_log_prob():
    torch.manual_seed(1)
    flow = configure_model(_config(5, passthrough_features=[2, 4]))
    flow.eval()
    with torch.no_grad():
        x, log_q = flow.sample_and_log_prob(1000)
        again = flow.log_prob(x)
    assert x.shape == (1000, 5)
    np.testing.assert_allclose(log_q.numpy(), again.numpy(), rtol=1e-4, atol=1e-4)
    # the pass-through features are standard normal
    free = x[:, [2, 4]].numpy()
    assert abs(free.mean()) < 0.1 and abs(free.std() - 1) < 0.1


def test_circular_features_remapped():
    torch.manual_seed(2)
    flow = configure_model(
        dict(n_inputs=6, n_neurons=4, n_blocks=2, n_layers=1, ftype="circular",
             circular_features=[1, 5], passthrough_features=[2, 3])
    )
    assert flow.circular_features == [1, 5]
    assert flow.passthrough_features == [2, 3]


@pytest.mark.parametrize(
    "passthrough, error",
    [([7], "distinct indices"), ([1, 1], "distinct indices"), ([0, 1, 2], "no inputs")],
)
def test_invalid(passthrough, error):
    n = 7 if error == "distinct indices" else 3
    with pytest.raises(ValueError, match=error):
        configure_model(_config(n, passthrough_features=passthrough))


def test_circular_and_passthrough_refused():
    with pytest.raises(ValueError, match="both circular and pass-through"):
        configure_model(
            dict(n_inputs=4, n_neurons=4, n_blocks=2, n_layers=1, ftype="circular",
                 circular_features=[1], passthrough_features=[1])
        )


def test_no_passthrough_unchanged():
    flow = configure_model(_config(4))
    assert not isinstance(flow, PassThroughFlow)
