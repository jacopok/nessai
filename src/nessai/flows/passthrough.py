# -*- coding: utf-8 -*-
"""
Flows that leave some features untransformed (pass-through features).

A pass-through feature is not seen by the flow's transform, neither
transformed nor conditioned on, and has a standard normal base: the flow's
density factorises into the inner flow's density of the other features times
a standard normal in the pass-through ones. This is the right model for
coordinates that are already standard normal under the proposal by
construction, e.g. a block of nuisance parameters whitened by their own
Gaussian fit (:class:`~nessai.reparameterisations.GaussianBlock`): a flow
fitted to them in many dimensions only adds noise to the rejection weights.
"""

import math

import numpy as np
import torch
from glasflow.nflows.distributions import Distribution

from .base import NFlow
from .circular import SubsetTransform


def _index(indices):
    return torch.as_tensor(np.asarray(indices, dtype=int), dtype=torch.long)


class ProductDistribution(Distribution):
    """``inner`` on the ``keep`` features, standard normal on the others.

    Parameters
    ----------
    inner : :obj:`glasflow.nflows.distributions.Distribution`
        Base distribution of the inner flow (``len(keep)`` features).
    keep : array_like of int
        The features the inner distribution describes, in its order.
    features : int
        The total number of features.
    """

    def __init__(self, inner, keep, features):
        super().__init__()
        keep = np.asarray(keep, dtype=int)
        others = np.setdiff1d(np.arange(features), keep)
        self.inner = inner
        self.register_buffer("_keep", _index(keep), persistent=False)
        self.register_buffer("_others", _index(others), persistent=False)
        self._shape = torch.Size([int(features)])
        self._log_z = 0.5 * len(others) * math.log(2 * math.pi)

    def _log_prob(self, inputs, context):
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                f"Expected input of shape {self._shape}, got "
                f"{inputs.shape[1:]}"
            )
        free = inputs.index_select(1, self._others)
        return (
            self.inner.log_prob(inputs.index_select(1, self._keep), context)
            - 0.5 * (free**2).sum(dim=1)
            - self._log_z
        )

    def _sample(self, num_samples, context):
        if context is not None:
            raise NotImplementedError("pass-through features with a context")
        inner = self.inner.sample(num_samples)
        x = torch.randn(
            num_samples, *self._shape, dtype=inner.dtype, device=inner.device
        )
        x[:, self._keep] = inner
        return x

    def _mean(self, context):
        mean = torch.zeros(self._shape, device=self._keep.device)
        mean[self._keep] = self.inner.mean(context)
        return mean


class PassThroughFlow(NFlow):
    """An inner flow on the ``keep`` features, the identity with a standard
    normal base on the others (``passthrough_features``).

    Built by :func:`~nessai.flows.utils.configure_model` from a flow config
    with ``passthrough_features``. The inner flow's ``circular_features``
    are reported in the full feature order.
    """

    def __init__(self, inner, keep, features):
        keep = np.asarray(keep, dtype=int)
        super().__init__(
            SubsetTransform(inner._transform, keep, features),
            ProductDistribution(inner._distribution, keep, features),
        )
        # not ``self.inner``: its transform and base are the submodules above
        self.inner_class = type(inner).__name__
        self.features = int(features)
        self.keep_features = keep.tolist()
        self.passthrough_features = np.setdiff1d(
            np.arange(features), keep
        ).tolist()
        circular = getattr(inner, "circular_features", None)
        if circular is not None:
            self.circular_features = [int(keep[i]) for i in circular]

    def _inner_parts(self):
        return self._transform.transform, self._distribution.inner

    def finalise(self):
        """Finalise the inner flow's base distribution and transform."""
        for part in self._inner_parts():
            if hasattr(part, "finalise"):
                part.finalise()

    def end_iteration(self):
        """End an iteration of the inner flow's base and transform."""
        for part in self._inner_parts():
            if hasattr(part, "end_iteration"):
                part.end_iteration()
