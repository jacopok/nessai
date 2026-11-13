# -*- coding: utf-8 -*-
"""GaussianBlock: a block whitened by a Gaussian fitted to the training points."""

import numpy as np
import pytest

from nessai.livepoint import dict_to_live_points, empty_structured_array
from nessai.reparameterisations import GaussianBlock, get_reparameterisation
from nessai.reparameterisations.gaussian_block import ledoit_wolf_correlation

NAMES = ["a", "b", "c"]


def _points(n, rng, cov=None, mean=(1.0, -2.0, 0.5)):
    cov = np.diag([0.5, 2.0, 1.0]) if cov is None else cov
    v = rng.multivariate_normal(mean, cov, size=n)
    return dict_to_live_points({k: v[:, i] for i, k in enumerate(NAMES)}), v


def _forward(block, x):
    x_prime = empty_structured_array(len(x), names=block.output_parameters)
    return block.reparameterise(x, x_prime, np.zeros(len(x)))


def test_registered():
    cls, kwargs = get_reparameterisation("gaussian-block")
    assert cls is GaussianBlock


def test_identity_before_update():
    block = GaussianBlock(parameters=NAMES)
    x, v = _points(5, np.random.default_rng(0))
    _, x_prime, log_j = _forward(block, x)
    for i, p in enumerate(block.output_parameters):
        np.testing.assert_array_equal(x_prime[p], v[:, i])
    np.testing.assert_array_equal(log_j, 0)


@pytest.mark.parametrize("shrinkage", [None, 0.3, "ledoit-wolf"])
def test_whitens_and_round_trips(shrinkage):
    rng = np.random.default_rng(1)
    cov = np.array([[1.0, 0.6, 0.0], [0.6, 2.0, -0.4], [0.0, -0.4, 0.5]])
    x, v = _points(20000, rng, cov=cov)
    block = GaussianBlock(parameters=NAMES, shrinkage=shrinkage)
    block.update(x)
    _, x_prime, log_j = _forward(block, x)
    u = np.column_stack([x_prime[p] for p in block.output_parameters])
    if shrinkage is None:
        np.testing.assert_allclose(np.cov(u.T), np.eye(3), atol=0.03)
        np.testing.assert_allclose(log_j, -0.5 * np.log(np.linalg.det(np.cov(v.T))), atol=0.02)
    np.testing.assert_allclose(u.mean(0), 0, atol=0.03)
    x_back = empty_structured_array(len(x), names=NAMES)
    x_back, _, log_j_inv = block.inverse_reparameterise(x_back, x_prime, np.zeros(len(x)))
    for i, p in enumerate(NAMES):
        np.testing.assert_allclose(x_back[p], v[:, i], atol=1e-10)
    np.testing.assert_allclose(log_j_inv, -log_j)


def test_log_density_matches_the_fitted_gaussian():
    """ln N(u; 0, I) + log_j is ln N(x; m, L L^T)."""
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(2)
    x, v = _points(3000, rng)
    block = GaussianBlock(parameters=NAMES, shrinkage=None)
    block.update(x)
    _, x_prime, log_j = _forward(block, x[:10])
    u = np.column_stack([x_prime[p] for p in block.output_parameters])
    log_q = -0.5 * (u**2).sum(1) - 1.5 * np.log(2 * np.pi) + log_j
    want = multivariate_normal(block.mean, block.covariance).logpdf(v[:10])
    np.testing.assert_allclose(log_q, want, rtol=1e-10)


def test_ledoit_wolf_shrinks_noise_only():
    rng = np.random.default_rng(3)
    d = 40
    white = rng.standard_normal((2000, d))
    _, intensity = ledoit_wolf_correlation(white)
    assert intensity > 0.9  # no correlation: the identity
    mixed = white.copy()
    mixed[:, 1] = 0.9 * mixed[:, 0] + np.sqrt(1 - 0.81) * mixed[:, 1]
    corr, intensity = ledoit_wolf_correlation(mixed)
    assert intensity < 0.5
    assert corr[0, 1] > 0.5


def test_inflation_and_too_few_points():
    rng = np.random.default_rng(4)
    x, v = _points(5000, rng)
    block = GaussianBlock(parameters=NAMES, shrinkage=1.0, inflation=1.5)
    block.update(x)
    np.testing.assert_allclose(np.sqrt(np.diag(block.covariance)), 1.5 * v.std(0), rtol=1e-10)
    before = block.mean.copy()
    block.update(x[:3])  # fewer than min_points: the fit is kept
    np.testing.assert_array_equal(block.mean, before)
    block.reset()
    np.testing.assert_array_equal(block.mean, 0)


def test_invalid():
    with pytest.raises(ValueError, match="shrinkage"):
        GaussianBlock(parameters=NAMES, shrinkage="other")
    with pytest.raises(ValueError, match="inflation"):
        GaussianBlock(parameters=NAMES, inflation=0)
