"""Tests for retroactive latent-temperature optimisation."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from nessai.utils.temperature import (
    log_reweight_factor,
    log_truncation_fraction,
    optimise_latent_temperature,
    reweight_log_weights,
    secondary_effective_sample_size,
)


def _identity_flow_batch(
    dims, temperature, target_var, n, radius=np.inf, seed=1
):
    """Draw a batch from a truncated N(0, T I) with an identity flow.

    ``f = identity`` so ``x = z`` and ``p_T(x)`` is exactly the truncated
    Gaussian. The target is an isotropic Gaussian with variance ``target_var``.
    Returns ``r`` and the importance log-weights ``log w_i(T)``.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=np.sqrt(temperature), size=(4 * n, dims))
    r = np.linalg.norm(z, axis=1)
    if np.isfinite(radius):
        z = z[r <= radius]
        r = r[r <= radius]
    z, r = z[:n], r[:n]

    log_fT = log_truncation_fraction(dims, radius, temperature)
    log_p = (
        multivariate_normal.logpdf(z, mean=np.zeros(dims), cov=temperature)
        - log_fT
    )
    log_rho = multivariate_normal.logpdf(
        z, mean=np.zeros(dims), cov=target_var
    )
    return r, log_rho - log_p


def test_reweight_factor_is_one_at_same_temperature():
    r = np.linspace(0, 3, 50)
    log_s = log_reweight_factor(
        r, dims=2, temperature=1.3, new_temperature=1.3
    )
    np.testing.assert_allclose(log_s, 0.0, atol=1e-12)


def test_secondary_ess_peaks_at_draw_temperature():
    r, _ = _identity_flow_batch(
        dims=2, temperature=1.0, target_var=1.0, n=5000
    )
    at_t = secondary_effective_sample_size(log_reweight_factor(r, 2, 1.0, 1.0))
    away = secondary_effective_sample_size(log_reweight_factor(r, 2, 1.0, 2.0))
    assert at_t == pytest.approx(5000, rel=1e-9)
    assert away < at_t


def test_reweight_matches_direct_draw():
    """Reweighted weights at T' match weights from a fresh draw at T'."""
    dims, target_var = 2, 1.6
    r, log_w = _identity_flow_batch(dims, 1.0, target_var, n=40000, seed=3)
    log_w_rw = reweight_log_weights(log_w, r, dims, 1.0, target_var)
    # ESS of a batch drawn directly at T' = target_var (perfect proposal).
    _, log_w2 = _identity_flow_batch(
        dims, target_var, target_var, n=40000, seed=4
    )

    def ess(lw):
        return np.exp(
            2 * np.log(np.sum(np.exp(lw - lw.max())))
            - np.log(np.sum(np.exp(2 * (lw - lw.max()))))
        )

    # Reweighting to the perfect temperature should give near-perfect ESS.
    assert ess(log_w_rw) / log_w_rw.size > 0.9
    assert ess(log_w2) / log_w2.size > 0.99


@pytest.mark.parametrize("criterion", ["efficiency", "ess"])
@pytest.mark.parametrize("target_var", [0.5, 1.5])
def test_optimiser_recovers_target_variance(target_var, criterion):
    dims = 2
    r, log_w = _identity_flow_batch(
        dims, temperature=1.0, target_var=target_var, n=50000, seed=7
    )
    result = optimise_latent_temperature(
        log_w, r, dims, temperature=1.0, tau=0.5, criterion=criterion
    )
    assert result["temperature"] == pytest.approx(target_var, rel=0.15)
    assert not result["at_trust_boundary"]


def test_efficiency_is_the_default_criterion():
    dims, target_var = 2, 1.3
    r, log_w = _identity_flow_batch(dims, 1.0, target_var, n=50000, seed=9)
    default = optimise_latent_temperature(log_w, r, dims, temperature=1.0)
    explicit = optimise_latent_temperature(
        log_w, r, dims, temperature=1.0, criterion="efficiency"
    )
    assert default["temperature"] == explicit["temperature"]


def test_optimiser_flags_trust_boundary_and_iterates():
    """A target far from T lands on the boundary; iterating fixes it."""
    dims, target_var = 2, 4.0
    r, log_w = _identity_flow_batch(dims, 1.0, target_var, n=50000, seed=11)
    first = optimise_latent_temperature(log_w, r, dims, temperature=1.0)
    assert first["at_trust_boundary"]

    # Step 6: redraw at the estimate, treat as new T, repeat until inside.
    t = first["temperature"]
    result = first
    for seed in range(12, 25):
        r_i, log_w_i = _identity_flow_batch(
            dims, t, target_var, n=50000, seed=seed
        )
        result = optimise_latent_temperature(log_w_i, r_i, dims, temperature=t)
        t = result["temperature"]
        if not result["at_trust_boundary"]:
            break
    assert not result["at_trust_boundary"]
    assert t == pytest.approx(target_var, rel=0.2)
