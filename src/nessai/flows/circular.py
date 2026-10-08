# -*- coding: utf-8 -*-
"""
Normalising flows on a product of circles and the real line.

Circular features are angles on ``[-pi, pi)``. They start uniform on the
circle (the angle of an isotropic Gaussian pair, without its radius) and are
moved by circle diffeomorphisms: a rotation by an angle the conditioner
predicts, then a rational-quadratic spline on ``[-pi, pi]`` whose end
derivatives are equal, so it is smooth across the seam (Rezende et al. 2020,
arXiv:2002.02428). Real features use the usual linear-tailed spline or
affine transform. Conditioners see each circular feature as its cosine and
sine, so they are continuous across the seam too.

The rotation angle is one real conditioner output by default. A real-valued
function of an angle cannot wind, so one coupling layer cannot follow a
density that winds around the torus (such as the ``phase + 2 psi``
degeneracy of gravitational-wave signals) by rotation alone;
``circular_shift='angle'`` takes the rotation as ``atan2(b, a)`` of two
outputs, which can. On toy problems, including a winding stripe, a stack of
coupling layers models the winding either way, and the real shift trains
more reliably (the ``atan2`` gradient diverges as ``(a, b) -> 0``).
"""

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from glasflow.nflows import transforms
from glasflow.nflows.distributions import Distribution
from glasflow.nflows.nn.nets import ResidualNet
from glasflow.nflows.transforms.splines import rational_quadratic as rqs

from .base import NFlow

logger = logging.getLogger(__name__)

TWO_PI = 2.0 * math.pi
#: Unnormalised derivative for which the spline's derivative is one.
_UNIT_DERIVATIVE = float(np.log(np.exp(1 - rqs.DEFAULT_MIN_DERIVATIVE) - 1))


def wrap(x):
    """Wrap angles to ``[-pi, pi)``."""
    return torch.remainder(x + math.pi, TWO_PI) - math.pi


def _index(mask):
    """Indices of the true entries of a boolean mask."""
    return torch.as_tensor(np.flatnonzero(np.asarray(mask)), dtype=torch.long)


def embed(x, circular):
    """Replace each circular feature by its cosine and sine."""
    if not circular.any():
        return x
    return torch.cat(
        [x[:, ~circular], torch.cos(x[:, circular]), torch.sin(x[:, circular])],
        dim=1,
    )


class CircularUniformNormal(Distribution):
    """Uniform on ``[-pi, pi)`` in the circular features, standard normal in
    the others.

    Parameters
    ----------
    circular : array_like of bool
        Which features are circular.
    """

    def __init__(self, circular):
        super().__init__()
        circular = np.asarray(circular, dtype=bool)
        self.register_buffer(
            "circular", torch.as_tensor(circular), persistent=False
        )
        self.register_buffer("_circ_idx", _index(circular), persistent=False)
        self.register_buffer("_real_idx", _index(~circular), persistent=False)
        self._shape = torch.Size([len(circular)])
        n_circ = int(circular.sum())
        n_real = len(circular) - n_circ
        self._log_z = n_circ * math.log(TWO_PI) + 0.5 * n_real * math.log(
            TWO_PI
        )

    def _log_prob(self, inputs, context):
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                f"Expected input of shape {self._shape}, got "
                f"{inputs.shape[1:]}"
            )
        real = inputs.index_select(1, self._real_idx)
        circ = inputs.index_select(1, self._circ_idx)
        log_prob = -0.5 * (real**2).sum(dim=1) - self._log_z
        outside = ((circ < -math.pi) | (circ > math.pi)).any(dim=1)
        return log_prob.masked_fill(outside, -math.inf)

    def _sample(self, num_samples, context):
        if context is not None:
            num_samples = num_samples * context.shape[0]
        x = torch.randn(
            num_samples, *self._shape, device=self.circular.device
        )
        n_circ = int(self.circular.sum())
        x[:, self.circular] = math.pi * (
            2 * torch.rand(num_samples, n_circ, device=x.device) - 1
        )
        if context is not None:
            x = x.reshape(context.shape[0], -1, *self._shape)
        return x

    def _mean(self, context):
        return torch.zeros(self._shape, device=self.circular.device)


class MixedCouplingTransform(transforms.Transform):
    """Coupling transform on circular and real features.

    Transformed circular features are rotated and then passed through a
    circular rational-quadratic spline; transformed real features through a
    linear-tailed spline or an affine transform. Both are conditioned on the
    other features (circular ones embedded as cosine and sine) and on the
    context.

    Parameters
    ----------
    mask : array_like of bool
        Features transformed by this layer; the rest condition it.
    circular : array_like of bool
        Which features are circular.
    hidden_features, num_blocks, activation, dropout_probability,
    use_batch_norm :
        Conditioner (``ResidualNet``) settings.
    context_features : int, optional
        Number of context features.
    num_bins : int
        Number of spline bins.
    tail_bound : float
        Interval of the real splines.
    real_transform : {'spline', 'affine'}
        Transform for the real features.
    circular_shift : {'real', 'angle', None}
        How the rotation of the circular features is predicted: one real
        output (default), the angle of two outputs (can wind around the
        circle as the conditioning angles go round theirs), or no rotation.
    """

    def __init__(
        self,
        mask,
        circular,
        hidden_features,
        num_blocks=2,
        context_features=None,
        num_bins=8,
        tail_bound=5.0,
        real_transform="spline",
        circular_shift="real",
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        min_bin_width=rqs.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=rqs.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=rqs.DEFAULT_MIN_DERIVATIVE,
    ):
        super().__init__()
        mask = torch.as_tensor(np.asarray(mask, dtype=bool))
        circular = torch.as_tensor(np.asarray(circular, dtype=bool))
        if mask.shape != circular.shape:
            raise ValueError("mask and circular must have the same length")
        if not mask.any():
            raise ValueError("The mask must transform at least one feature")
        if real_transform not in {"spline", "affine"}:
            raise ValueError(f"Unknown real transform: {real_transform}")
        if circular_shift not in {"angle", "real", None}:
            raise ValueError(f"Unknown circular shift: {circular_shift}")

        self.register_buffer("mask", mask, persistent=False)
        self.register_buffer("circular", circular, persistent=False)
        self.register_buffer(
            "circ_out", circular[mask], persistent=False
        )
        self.register_buffer(
            "circ_in", circular[~mask], persistent=False
        )
        # Gather and scatter with index tensors: boolean-mask indexing and
        # in-place scatters dominate the cost of small batches otherwise
        m, c = mask.numpy(), circular.numpy()
        order = np.concatenate(
            [np.flatnonzero(~m), np.flatnonzero(m & c), np.flatnonzero(m & ~c)]
        )
        for name, idx in [
            ("_identity_idx", np.flatnonzero(~m)),
            ("_identity_real_idx", np.flatnonzero(~m & ~c)),
            ("_identity_circ_idx", np.flatnonzero(~m & c)),
            ("_target_circ_idx", np.flatnonzero(m & c)),
            ("_target_real_idx", np.flatnonzero(m & ~c)),
            ("_inverse_order", np.argsort(order)),
        ]:
            self.register_buffer(
                name, torch.as_tensor(idx, dtype=torch.long), persistent=False
            )
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.real_transform = real_transform
        self.circular_shift = circular_shift
        self.hidden_features = hidden_features
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        self.n_circ = int(self.circ_out.sum())
        self.n_real = int((~self.circ_out).sum())
        self._n_shift = {"angle": 2, "real": 1, None: 0}[circular_shift]
        self._n_circ_params = 3 * num_bins + self._n_shift
        self._n_real_params = (
            3 * num_bins - 1 if real_transform == "spline" else 2
        )
        n_out = (
            self.n_circ * self._n_circ_params
            + self.n_real * self._n_real_params
        )
        n_in = int((~mask).sum()) + int(self.circ_in.sum())

        if n_in == 0 and context_features is None:
            # Nothing to condition on: learn the parameters directly
            self.transform_net = None
            self.unconditional_params = torch.nn.Parameter(torch.zeros(n_out))
        else:
            self.transform_net = ResidualNet(
                n_in,
                n_out,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=num_blocks,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
            )

    def reset_parameters(self):
        """Reset the unconditional parameters (the conditioner's layers are
        reset by ``reset_weights`` on their own)."""
        if self.transform_net is None:
            torch.nn.init.zeros_(self.unconditional_params)

    def _params(self, inputs, context):
        if self.transform_net is None:
            return self.unconditional_params.expand(inputs.shape[0], -1)
        circ = inputs.index_select(1, self._identity_circ_idx)
        embedded = torch.cat(
            [
                inputs.index_select(1, self._identity_real_idx),
                torch.cos(circ),
                torch.sin(circ),
            ],
            dim=1,
        )
        return self.transform_net(embedded, context)

    def _split_params(self, params):
        n = params.shape[0]
        circ = params[:, : self.n_circ * self._n_circ_params].reshape(
            n, self.n_circ, self._n_circ_params
        )
        real = params[:, self.n_circ * self._n_circ_params :].reshape(
            n, self.n_real, self._n_real_params
        )
        return circ, real

    def _spline_inputs(self, p):
        k = self.num_bins
        scale = math.sqrt(self.hidden_features)
        return p[..., :k] / scale, p[..., k : 2 * k] / scale, p[..., 2 * k :]

    def _shift(self, circ_params):
        if self.circular_shift == "angle":
            a = 1.0 + circ_params[..., -2]
            b = circ_params[..., -1]
            return torch.atan2(b, a)
        elif self.circular_shift == "real":
            return circ_params[..., -1]
        return torch.zeros_like(circ_params[..., 0])

    def _circular_spline(self, x, circ_params, inverse):
        k = self.num_bins
        params = circ_params[..., : 3 * k]
        # A non-finite input or parameter (an overflowing extreme draw) would
        # send the bin search out of range: evaluate a placeholder and return
        # NaN for it, as the other transforms do.
        bad = ~(torch.isfinite(x) & torch.isfinite(params).all(dim=-1))
        if bool(bad.any()):
            x = torch.where(bad, torch.zeros_like(x), x)
            params = torch.where(bad[..., None], torch.zeros_like(params), params)
        widths, heights, derivs = self._spline_inputs(params)
        # Equal end derivatives: smooth across the seam
        derivs = derivs + _UNIT_DERIVATIVE
        derivs = torch.cat([derivs, derivs[..., :1]], dim=-1)
        y, logabsdet = rqs.rational_quadratic_spline(
            x.clamp(-math.pi, math.pi),
            widths,
            heights,
            derivs,
            inverse=inverse,
            left=-math.pi,
            right=math.pi,
            bottom=-math.pi,
            top=math.pi,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        if bool(bad.any()):
            y = y.masked_fill(bad, math.nan)
            logabsdet = logabsdet.masked_fill(bad, math.nan)
        return y, logabsdet

    def _real_transform(self, x, real_params, inverse):
        if self.real_transform == "spline":
            widths, heights, derivs = self._spline_inputs(real_params)
            return rqs.unconstrained_rational_quadratic_spline(
                x,
                widths,
                heights,
                derivs,
                inverse=inverse,
                tails="linear",
                tail_bound=self.tail_bound,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative,
            )
        scale = torch.sigmoid(real_params[..., 0] + 2.0) + 1e-3
        shift = real_params[..., 1]
        if inverse:
            return (x - shift) / scale, -torch.log(scale)
        return x * scale + shift, torch.log(scale)

    def _transform(self, inputs, context, inverse):
        circ_params, real_params = self._split_params(
            self._params(inputs, context)
        )
        parts = [inputs.index_select(1, self._identity_idx)]
        logabsdet = inputs.new_zeros(inputs.shape[0])

        if self.n_circ:
            theta = inputs.index_select(1, self._target_circ_idx)
            shift = self._shift(circ_params)
            if inverse:
                theta, ld = self._circular_spline(theta, circ_params, True)
                theta = wrap(theta + shift)
            else:
                theta, ld = self._circular_spline(
                    wrap(theta - shift), circ_params, False
                )
            parts.append(theta)
            logabsdet = logabsdet + ld.sum(dim=1)

        if self.n_real:
            y, ld = self._real_transform(
                inputs.index_select(1, self._target_real_idx),
                real_params,
                inverse,
            )
            parts.append(y)
            logabsdet = logabsdet + ld.sum(dim=1)

        outputs = torch.cat(parts, dim=1).index_select(1, self._inverse_order)
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        return self._transform(inputs, context, inverse=False)

    def inverse(self, inputs, context=None):
        return self._transform(inputs, context, inverse=True)


class WrapCircular(transforms.Transform):
    """Wrap the circular features to ``[-pi, pi)``, e.g. after training
    noise has been added. Unit Jacobian."""

    def __init__(self, circular):
        super().__init__()
        circular = torch.as_tensor(np.asarray(circular, dtype=bool))
        self.register_buffer("circular", circular, persistent=False)

    def forward(self, inputs, context=None):
        outputs = torch.where(self.circular, wrap(inputs), inputs)
        return outputs, inputs.new_zeros(inputs.shape[0])

    def inverse(self, inputs, context=None):
        return self.forward(inputs, context)


class SubsetTransform(transforms.Transform):
    """Apply a transform to a subset of the features only."""

    def __init__(self, transform, indices, features):
        super().__init__()
        self.transform = transform
        indices = np.asarray(indices, dtype=int)
        others = np.setdiff1d(np.arange(features), indices)
        self.register_buffer(
            "indices", torch.as_tensor(indices, dtype=torch.long)
        )
        self.register_buffer(
            "_others", torch.as_tensor(others, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_inverse_order",
            torch.as_tensor(
                np.argsort(np.concatenate([others, indices])),
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.features = features

    def _subset(self, fn, inputs, context):
        y, logabsdet = fn(inputs.index_select(1, self.indices), context)
        outputs = torch.cat(
            [inputs.index_select(1, self._others), y], dim=1
        ).index_select(1, self._inverse_order)
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        return self._subset(self.transform.forward, inputs, context)

    def inverse(self, inputs, context=None):
        return self._subset(self.transform.inverse, inputs, context)


def mixed_coupling_masks(features, num_layers, rng=None):
    """Coupling masks that transform every feature once per pair of layers.

    Each pair of layers uses an alternating mask on a fresh random order of
    the features and then its complement, which plays the role of the
    permutations between the layers of a plain coupling flow (permutations
    cannot be used here, since they would move circular features into real
    slots).
    """
    rng = np.random.default_rng(rng)
    masks = []
    for i in range(num_layers):
        if features == 1:
            masks.append(np.ones(1, dtype=bool))
            continue
        if i % 2 == 0:
            order = np.arange(features) if i == 0 else rng.permutation(features)
            mask = np.zeros(features, dtype=bool)
            mask[order[::2]] = True
            masks.append(mask)
        else:
            masks.append(~masks[-1])
    return masks


class CircularNeuralSplineFlow(NFlow):
    """Coupling flow on a product of circles and the real line.

    Circular features must lie on ``[-pi, pi)``; the base distribution is
    uniform there and standard normal in the real features.

    Parameters
    ----------
    features : int
        Number of features.
    hidden_features : int
        Neurons per layer of each conditioner.
    num_layers : int
        Number of coupling transforms.
    num_blocks_per_layer : int
        Residual blocks per conditioner.
    circular_features : list of int
        Indices of the circular features.
    num_bins : int
        Number of spline bins.
    tail_bound : float
        Interval of the real splines.
    real_transform : {'spline', 'affine'}
        Coupling transform of the real features.
    circular_shift : {'real', 'angle', None}
        See :class:`MixedCouplingTransform`.
    linear_transform : {'lu', 'permutation', None}
        Linear transform of the real features before each coupling layer.
    batch_norm_between_layers : bool
        Batch norm of the real features after each coupling layer.
    mask_seed : int, optional
        Seed for the order of the features in the coupling masks. The masks
        fix the conditioners' shapes and are not saved with the weights, so
        a flow rebuilt from its configuration (resume, reset) must draw the
        same ones: the default is a fixed seed.
    """

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        circular_features=None,
        context_features=None,
        num_bins=8,
        tail_bound=5.0,
        real_transform="spline",
        circular_shift="real",
        linear_transform=None,
        batch_norm_between_layers=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        mask_seed=0,
        distribution=None,
        **kwargs,
    ):
        if kwargs:
            logger.warning(f"Ignoring unused keyword arguments: {kwargs}")
        circular = np.zeros(features, dtype=bool)
        if circular_features is not None:
            circular[list(circular_features)] = True
        real_idx = np.flatnonzero(~circular)
        n_real = len(real_idx)

        layers = [WrapCircular(circular)] if circular.any() else []
        for mask in mixed_coupling_masks(features, num_layers, rng=mask_seed):
            if linear_transform is not None and n_real > 1:
                if linear_transform.lower() == "lu":
                    lin = transforms.CompositeTransform(
                        [
                            transforms.RandomPermutation(n_real),
                            transforms.LULinear(
                                n_real, identity_init=True, using_cache=True
                            ),
                        ]
                    )
                elif linear_transform.lower() == "permutation":
                    lin = transforms.RandomPermutation(n_real)
                else:
                    raise ValueError(
                        f"Unknown linear transform: {linear_transform}"
                    )
                layers.append(SubsetTransform(lin, real_idx, features))
            layers.append(
                MixedCouplingTransform(
                    mask,
                    circular,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks_per_layer,
                    context_features=context_features,
                    num_bins=num_bins,
                    tail_bound=tail_bound,
                    real_transform=real_transform,
                    circular_shift=circular_shift,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers,
                )
            )
            if batch_norm_between_layers and n_real:
                layers.append(
                    SubsetTransform(
                        transforms.BatchNorm(features=n_real),
                        real_idx,
                        features,
                    )
                )

        if distribution is None:
            distribution = CircularUniformNormal(circular)
        self.circular_features = np.flatnonzero(circular).tolist()
        super().__init__(
            transform=transforms.CompositeTransform(layers),
            distribution=distribution,
        )
