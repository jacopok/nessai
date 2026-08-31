# -*- coding: utf-8 -*-
"""Specific tests for different included flows."""

import numpy as np
import pytest
import torch

from nessai.flows import (
    MaskedAutoregressiveFlow,
    NeuralSplineFlow,
    RealNVP,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(
            net="mlp", batch_norm_within_layers=True, dropout_probability=0.5
        ),
        dict(use_volume_preserving=True),
        dict(actnorm=True, batch_norm_between_layers=False),
        dict(linear_transform="permutation"),
        dict(linear_transform="svd"),
        dict(linear_transform="lu"),
        dict(linear_transform=None),
        dict(linear_transform="None"),
        dict(mask=np.array([[1, -1], [-1, 1]])),
        dict(mask=[1, -1]),
        dict(pre_transform="batch_norm"),
        dict(pre_transform="batch_norm", pre_transform_kwargs=dict(eps=1e-8)),
        dict(scale_activation=lambda x: torch.sigmoid(x + 2) + 1e-3),
        dict(shift_bound=5.0),
    ],
)
def test_with_realnvp_kwargs(kwargs):
    """Test RealNVP with specific kwargs"""
    flow = RealNVP(2, 2, 2, 2, **kwargs)
    x = torch.randn(10, 2)
    z, _ = flow.forward(x)
    assert z.shape == (10, 2)


def test_realnvp_shift_bound_clamps_translation():
    """``shift_bound`` should bound every coupling layer's translation term.

    Checked on a single layer (matching identical weights and inputs on
    both branches): a coupling layer's own log-Jacobian only depends on
    ``scale``, so clamping only the shift must not change it there. Composed
    across many layers, later layers see the (now different) output of
    earlier ones, so their own scale -- and hence the *composite*
    log-Jacobian -- legitimately differs; that is not tested here.
    """
    from glasflow.nflows.transforms import AffineCouplingTransform

    # A tight bound relative to the (randomly initialised, ~O(1)) shift
    # network output guarantees the clamp actually engages below.
    bound = 0.01
    bounded = RealNVP(
        4, 8, 6, 2, batch_norm_between_layers=False, shift_bound=bound
    )
    unbounded = RealNVP(4, 8, 6, 2, batch_norm_between_layers=False)
    unbounded.load_state_dict(bounded.state_dict())

    coupling_layers = [
        t
        for t in bounded._transform._transforms
        if hasattr(t, "shift_bound")
    ]
    assert len(coupling_layers) == 6
    for layer in coupling_layers:
        assert layer.shift_bound == bound

    unbounded_coupling_layers = [
        t
        for t in unbounded._transform._transforms
        if isinstance(t, AffineCouplingTransform)
    ]
    x = 100.0 * torch.randn(64, 4)
    layer_b = coupling_layers[0]
    layer_u = unbounded_coupling_layers[0]
    z_b, logabsdet_b = layer_b(x)
    z_u, logabsdet_u = layer_u(x)
    # Same log-Jacobian (depends only on scale, identical weights and input).
    assert torch.allclose(logabsdet_b, logabsdet_u, atol=1e-4)
    # Different output (the shift is actually clamped) but still a valid,
    # invertible transform.
    assert not torch.allclose(z_b, z_u)
    x_rt, _ = layer_b.inverse(z_b)
    assert torch.allclose(x_rt, x, atol=1e-3)


def test_realnvp_shift_bound_default_is_unbounded():
    """Without ``shift_bound`` the coupling transform is the plain glasflow one."""
    from glasflow.nflows.transforms import AffineCouplingTransform

    flow = RealNVP(4, 8, 6, 2, batch_norm_between_layers=False)
    coupling_layers = [
        t
        for t in flow._transform._transforms
        if isinstance(t, AffineCouplingTransform)
    ]
    assert len(coupling_layers) == 6
    assert all(
        not hasattr(t, "shift_bound") for t in coupling_layers
    )


@pytest.mark.parametrize(
    "kwargs, string",
    [
        (dict(net="res"), "Unknown nn type: res"),
        (dict(linear_transform="test"), "Unknown linear transform: test"),
        (dict(mask=[1, 1, -1]), "Mask does not match number of features"),
        (
            dict(mask=[[-1, 1], [1, -1], [1, -1]]),
            "Mask does not match number of layers",
        ),
    ],
)
def test_realnvp_value_errors(kwargs, string):
    """Assert incorrect values for some inputs raise an error"""
    with pytest.raises(ValueError) as excinfo:
        RealNVP(2, 2, 2, 2, **kwargs)
    assert string in str(excinfo.value)


def test_realnvp_actnorm_batchnorm():
    """
    Assert an error is raised if actnorm and batchnorm are enabled at once.
    """
    with pytest.raises(
        RuntimeError, match=r"Cannot enable actnorm and batchnorm .*"
    ):
        RealNVP(2, 2, 2, 2, actnorm=True, batch_norm_between_layers=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(batch_norm_between_layers=True),
        dict(linear_transform="permutation"),
        dict(linear_transform="svd"),
        dict(linear_transform="lu"),
        dict(linear_transform=None),
        dict(linear_transform="None"),
        dict(num_bins=10),
    ],
)
def test_with_nsf_kwargs(kwargs):
    """Test NSF with specific kwargs"""
    flow = NeuralSplineFlow(2, 2, 2, 2, **kwargs)
    x = torch.randn(10, 2)
    z, _ = flow.forward(x)
    assert z.shape == (10, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(batch_norm_between_layers=True),
        dict(batch_norm_within_layers=True),
        dict(use_random_permutations=True),
        dict(use_residual_blocks=True),
        dict(use_random_masks=False),
    ],
)
def test_with_maf_kwargs(kwargs, caplog):
    """Test MAF with specific kwargs"""
    flow = MaskedAutoregressiveFlow(2, 2, 2, 2, **kwargs)
    x = torch.randn(10, 2)
    z, _ = flow.forward(x)
    assert z.shape == (10, 2)


@pytest.mark.parametrize("FlowClass", [RealNVP, NeuralSplineFlow])
def test_1d_inputs(FlowClass):
    """Assert an error is raised if 1-d inputs are specified."""
    with pytest.raises(ValueError) as excinfo:
        FlowClass(1, 2, 2, 2)
    assert "requires at least 2 dimensions" in str(excinfo.value)
