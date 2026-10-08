"""Tests for the circular flows."""

import math

import numpy as np
import pytest
import torch

from nessai.flows.circular import (
    CircularNeuralSplineFlow,
    CircularUniformNormal,
    MixedCouplingTransform,
    mixed_coupling_masks,
    wrap,
)


@pytest.fixture(autouse=True)
def float64():
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(dtype)


def perturb(module, scale=0.1):
    """Move the weights away from their initial values."""
    torch.manual_seed(1)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn_like(p))


def mixed_inputs(n, circular):
    x = torch.randn(n, len(circular))
    c = torch.as_tensor(circular)
    x[:, c] = math.pi * (2 * torch.rand(n, int(c.sum())) - 1)
    return x


def test_wrap():
    x = torch.tensor([-math.pi, 0.0, math.pi, 3 * math.pi + 0.1, -4.0])
    y = wrap(x)
    assert (y >= -math.pi).all() and (y < math.pi).all()
    np.testing.assert_allclose(torch.cos(y), torch.cos(x), atol=1e-12)
    np.testing.assert_allclose(torch.sin(y), torch.sin(x), atol=1e-12)


def test_base_distribution():
    dist = CircularUniformNormal([True, False])
    x = dist.sample(1000)
    assert (x[:, 0].abs() <= math.pi).all()
    expected = -math.log(2 * math.pi) - 0.5 * x[:, 1] ** 2 - 0.5 * math.log(
        2 * math.pi
    )
    np.testing.assert_allclose(dist.log_prob(x), expected)
    assert dist.log_prob(torch.tensor([[3.5, 0.0]])).item() == -math.inf


@pytest.mark.parametrize("n_features", [1, 2, 5])
def test_masks_cover_every_feature(n_features):
    masks = mixed_coupling_masks(n_features, 6, rng=0)
    assert all(m.any() for m in masks)
    for a, b in zip(masks[::2], masks[1::2]):
        assert (a | b).all()


@pytest.mark.parametrize("real_transform", ["spline", "affine"])
@pytest.mark.parametrize("circular_shift", ["angle", "real", None])
@pytest.mark.parametrize("mask", [[1, 0, 0, 0], [0, 1, 0, 1], [1, 1, 0, 1]])
def test_coupling_invertible(real_transform, circular_shift, mask):
    circular = [True, False, True, False]
    t = MixedCouplingTransform(
        np.array(mask, dtype=bool),
        circular,
        hidden_features=16,
        num_blocks=1,
        real_transform=real_transform,
        circular_shift=circular_shift,
    )
    perturb(t, 0.5)
    x = mixed_inputs(256, circular)
    with torch.no_grad():
        z, ld = t(x)
        x_rec, ld_inv = t.inverse(z)
    np.testing.assert_allclose(wrap(x_rec - x), 0, atol=1e-9)
    np.testing.assert_allclose(ld + ld_inv, 0, atol=1e-9)
    assert (z[:, [0, 2]].abs() <= math.pi).all()


def test_unconditional_circle():
    """A single circular feature has nothing to condition on."""
    flow = CircularNeuralSplineFlow(1, 8, 2, 1, circular_features=[0])
    perturb(flow)
    g = torch.linspace(-math.pi, math.pi, 2001)[:-1] + math.pi / 2000
    with torch.no_grad():
        p = flow.log_prob(g[:, None]).exp()
    assert p.sum().item() * 2 * math.pi / 2000 == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize("real_transform", ["spline", "affine"])
def test_flow_log_det_matches_autograd(real_transform):
    flow = CircularNeuralSplineFlow(
        4,
        16,
        4,
        1,
        circular_features=[0, 2],
        real_transform=real_transform,
        linear_transform="lu",
        batch_norm_between_layers=True,
        mask_seed=1,
    )
    perturb(flow)
    x = mixed_inputs(64, [True, False, True, False])
    with torch.no_grad():
        for _ in range(20):  # batch-norm running statistics
            flow.forward(x)
    flow.eval()
    z, ld = flow.forward(x[:8])
    jac = torch.autograd.functional.jacobian(
        lambda v: flow.forward(v)[0].sum(0), x[:8]
    )
    expected = torch.stack(
        [torch.linalg.slogdet(jac[:, i, :])[1] for i in range(8)]
    )
    np.testing.assert_allclose(ld.detach(), expected, atol=1e-8)
    x_rec, _ = flow.inverse(z)
    np.testing.assert_allclose(wrap(x_rec - x[:8]).detach(), 0, atol=1e-9)


def test_torus_density_normalised_and_continuous():
    flow = CircularNeuralSplineFlow(2, 16, 4, 1, circular_features=[0, 1])
    perturb(flow, 0.3)
    flow.eval()
    n = 800
    g = torch.linspace(-math.pi, math.pi, n + 1)[:-1] + math.pi / n
    grid = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    with torch.no_grad():
        integral = flow.log_prob(grid).exp().sum() * (2 * math.pi / n) ** 2
        e = 1e-7
        seam = flow.log_prob(
            torch.tensor(
                [
                    [math.pi - e, 0.3],
                    [-math.pi + e, 0.3],
                    [0.3, math.pi - e],
                    [0.3, -math.pi + e],
                ]
            )
        )
    assert integral.item() == pytest.approx(1, abs=1e-4)
    np.testing.assert_allclose(seam[0], seam[1], atol=1e-4)
    np.testing.assert_allclose(seam[2], seam[3], atol=1e-4)


def test_sample_and_log_prob():
    flow = CircularNeuralSplineFlow(3, 16, 4, 1, circular_features=[1])
    perturb(flow)
    flow.eval()
    with torch.no_grad():
        x, log_q = flow.sample_and_log_prob(500)
        np.testing.assert_allclose(log_q, flow.log_prob(x), atol=1e-8)
    assert (x[:, 1].abs() <= math.pi).all()


def test_rebuilt_flow_loads_weights():
    """A flow rebuilt from the same configuration (resume, reset) has the
    same architecture, so the saved weights load into it."""
    kwargs = dict(
        circular_features=[1, 4],
        linear_transform="lu",
        batch_norm_between_layers=True,
        real_transform="affine",
    )
    torch.manual_seed(0)
    a = CircularNeuralSplineFlow(6, 16, 6, 2, **kwargs)
    perturb(a)
    torch.manual_seed(1)
    b = CircularNeuralSplineFlow(6, 16, 6, 2, **kwargs)
    b.load_state_dict(a.state_dict())
    a.eval()
    b.eval()
    x = mixed_inputs(100, [False, True, False, False, True, False])
    with torch.no_grad():
        np.testing.assert_allclose(a.log_prob(x), b.log_prob(x))


def test_non_finite_inputs_give_nan_not_errors():
    """Overflowing draws propagate NaN (as in the other flows), leaving the
    finite rows untouched."""
    flow = CircularNeuralSplineFlow(
        4, 16, 4, 1, circular_features=[0, 2], real_transform="affine"
    )
    perturb(flow)
    flow.eval()
    x = mixed_inputs(6, [True, False, True, False])
    ref = flow.log_prob(x).detach()
    x[1, 1] = float("inf")
    x[3, 3] = float("nan")
    x[4, 0] = float("nan")
    with torch.no_grad():
        lp = flow.log_prob(x)
        z = torch.randn(5, 4) * 1e30
        flow.inverse(z)
    assert torch.isnan(lp[[1, 3, 4]]).all()
    np.testing.assert_allclose(lp[[0, 2, 5]], ref[[0, 2, 5]])
