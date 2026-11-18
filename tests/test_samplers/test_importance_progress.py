"""The importance nested sampler's end-of-run and ETA estimate."""

from types import SimpleNamespace

import numpy as np
from scipy.special import gammainc

from nessai.samplers.progress import (
    estimate_importance_progress,
    format_importance_progress,
    remaining_log_volume_ratio,
)


def _gaussian_run(dimension=8.0, l_max=100.0, n_live=500, per_x=5000):
    """An exact Gaussian peak: ln Lmax - ln L = 0.5 chi2 quantile, the
    share of the evidence above the threshold P(D/2, Delta), N growing
    linearly in -logX."""
    a = dimension / 2
    x = np.linspace(0.5, 60, 400)
    c = 1e4  # Delta at x = 0
    delta = c * np.exp(-x / a)
    n = n_live + per_x * x
    log_ratio = np.log(gammainc(a, delta)) + np.log(n / n_live)
    stop = x[np.argmax(log_ratio <= 0)]
    return x, l_max - delta, np.full_like(x, l_max), log_ratio, n, stop


def test_remaining_log_volume_ratio_exact_for_gaussian():
    x, log_l, log_l_max, log_ratio, n, stop = _gaussian_run()
    for k in (120, 200, 260):
        rem, dim = remaining_log_volume_ratio(
            x[: k + 1], log_l[: k + 1], log_l_max[: k + 1],
            log_ratio[: k + 1], n[: k + 1],
        )
        np.testing.assert_allclose(dim, 8.0, rtol=1e-6)
        np.testing.assert_allclose(x[k] + rem, stop, atol=0.2)


def test_remaining_log_volume_ratio_before_fit_span():
    rem, dim = remaining_log_volume_ratio(
        [0.5, 1.0], [0.0, 1.0], [10.0, 10.0], [3.0, 2.0], [10, 20]
    )
    assert np.isnan(rem) and np.isnan(dim)


def test_estimate_importance_progress_line():
    x, log_l, log_l_max, log_ratio, n, stop = _gaussian_run()
    k = 200
    sampler = SimpleNamespace(
        history=dict(
            log_prior_volume=list(-x[: k + 1]),
            logL_threshold=list(log_l[: k + 1]),
            max_log_likelihood=list(log_l_max[: k + 1]),
            sampling_time=list(60.0 * np.arange(k + 1)),
            n_samples=list(n[: k + 1]),
            stopping_criteria={"log_evidence_ratio": list(log_ratio[: k + 1])},
        ),
        combined_criterion=SimpleNamespace(
            names=["log_evidence_ratio"],
            tolerances={"log_evidence_ratio": 0.0},
        ),
        model=SimpleNamespace(names=[f"p{i}" for i in range(12)]),
    )
    p = estimate_importance_progress(sampler)
    np.testing.assert_allclose(p["log_volume"] + p["remaining"], stop, atol=0.2)
    np.testing.assert_allclose(p["seconds_per_level"], 60.0)
    levels_per_x = 1 / (x[1] - x[0])
    np.testing.assert_allclose(p["levels"], p["remaining"] * levels_per_x)
    line = format_importance_progress(p)
    assert line.startswith("Progress: -logX") and "ETA >=" in line
    sampler.combined_criterion.names = ["ess"]
    assert "no ETA" in format_importance_progress(
        estimate_importance_progress(sampler)
    )
