# -*- coding: utf-8 -*-
"""
A block of parameters whitened by a Gaussian fitted to the training points.
"""

import logging

import numpy as np

from .base import Reparameterisation

logger = logging.getLogger(__name__)


def ledoit_wolf_correlation(x):
    """Ledoit-Wolf shrinkage of the correlation matrix of the rows of ``x``
    towards the identity (Ledoit & Wolf 2004, the scaled-identity target of
    the standardised data): the variances are kept, the correlations shrunk
    by the optimal intensity. Returns ``(correlation, intensity)``."""
    n, d = x.shape
    z = (x - x.mean(0)) / x.std(0)
    c = z.T @ z / n
    off = c - np.eye(d)
    gamma = float((off**2).sum())
    if gamma <= 0:
        return np.eye(d), 1.0
    # the variance of the sample correlations, sum over entries
    pi_hat = 0.0
    for start in range(0, n, 2048):
        zz = z[start : start + 2048]
        pi_hat += float(
            ((zz[:, :, None] * zz[:, None, :] - c) ** 2).sum()
        )
    pi_hat /= n
    intensity = min(1.0, max(0.0, pi_hat / gamma / n))
    return (1 - intensity) * c + intensity * np.eye(d), intensity


class GaussianBlock(Reparameterisation):
    """Whiten a block of parameters by a Gaussian fitted to the points the
    reparameterisation is updated with (the proposal's training points):

    .. math::
        x' = L^{-1} (x - m),

    ``m`` their mean and ``L L^T`` their covariance, with its correlations
    shrunk (``shrinkage``) and its scale times ``inflation``. Before the
    first update ``m = 0`` and ``L = I``.

    Meant for nuisance parameters close to Gaussian under the proposal's
    target (e.g. detector calibration nodes) whose prime parameters pass
    through the flow (the proposal's ``passthrough_parameters``): their
    proposal density is then exactly ``N(m, L L^T)``.

    Parameters
    ----------
    parameters : list of str
        The block's parameters.
    prior_bounds : dict, optional
        Not used (the parameters need not be bounded).
    shrinkage : {"ledoit-wolf", None} or float, optional
        Shrinkage of the fitted correlations towards zero: Ledoit-Wolf's
        optimal intensity (the default), none, or a fixed intensity in
        ``[0, 1]`` (1: diagonal).
    inflation : float, optional
        Factor on the fitted standard deviations. Default 1.
    min_points : int, optional
        Fewer finite training points than this (default ``2 * len(parameters)
        + 2``) and the update keeps the previous fit.
    """

    requires_bounded_prior = False
    _update = True

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        shrinkage="ledoit-wolf",
        inflation=1.0,
        min_points=None,
        rng=None,
        **kwargs,
    ):
        super().__init__(
            parameters=parameters, prior_bounds=prior_bounds, rng=rng, **kwargs
        )
        if not (
            shrinkage is None
            or shrinkage == "ledoit-wolf"
            or (isinstance(shrinkage, (int, float)) and 0 <= shrinkage <= 1)
        ):
            raise ValueError(
                "shrinkage must be 'ledoit-wolf', None or a number in [0, 1]; "
                f"got {shrinkage!r}"
            )
        if not inflation > 0:
            raise ValueError(f"inflation must be positive; got {inflation}")
        self.shrinkage = shrinkage
        self.inflation = float(inflation)
        d = len(self.input_parameters)
        self.min_points = int(min_points) if min_points else 2 * d + 2
        self.shrinkage_intensity = None
        self.n_updates = 0
        self.reset()

    def reset(self):
        d = len(self.input_parameters)
        self._set(np.zeros(d), np.eye(d))

    def _set(self, mean, cov):
        self.mean = np.asarray(mean, dtype=float)
        self.chol = np.linalg.cholesky(np.asarray(cov, dtype=float))
        self.chol_inv = np.linalg.inv(self.chol)
        self.log_det = float(np.log(np.diag(self.chol)).sum())

    @property
    def covariance(self):
        return self.chol @ self.chol.T

    def _values(self, x, x_prime=None):
        return np.column_stack(
            [
                np.asarray(self.get_parameter_value(p, x, x_prime), dtype=float)
                for p in self.input_parameters
            ]
        )

    def update(self, x, x_prime=None):
        """Fit the Gaussian to ``x`` (the training points)."""
        v = self._values(x, x_prime)
        v = v[np.isfinite(v).all(axis=1)]
        if len(v) < self.min_points:
            logger.debug(
                "GaussianBlock: %d points, fewer than %d: fit kept",
                len(v),
                self.min_points,
            )
            return
        std = v.std(axis=0)
        if not (std > 0).all():
            logger.debug("GaussianBlock: a parameter has no spread: fit kept")
            return
        if self.shrinkage == "ledoit-wolf":
            corr, intensity = ledoit_wolf_correlation(v)
        else:
            z = (v - v.mean(0)) / std
            corr = z.T @ z / len(v)
            intensity = float(self.shrinkage or 0.0)
            corr = (1 - intensity) * corr + intensity * np.eye(len(std))
        self.shrinkage_intensity = intensity
        scale = self.inflation * std
        self._set(v.mean(axis=0), corr * np.outer(scale, scale))
        self.n_updates += 1
        logger.debug(
            "GaussianBlock: %d points, shrinkage intensity %.3f",
            len(v),
            intensity,
        )

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        u = (self._values(x, x_prime) - self.mean) @ self.chol_inv.T
        for j, pp in enumerate(self.output_parameters):
            x_prime[pp] = u[:, j]
        log_j -= self.log_det
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        u = np.column_stack(
            [np.asarray(x_prime[pp], dtype=float) for pp in self.output_parameters]
        )
        v = self.mean + u @ self.chol.T
        for j, p in enumerate(self.input_parameters):
            x, x_prime = self.set_parameter_value(p, v[:, j], x, x_prime)
        log_j += self.log_det
        return x, x_prime, log_j
