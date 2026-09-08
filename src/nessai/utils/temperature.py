"""
Retroactive latent-temperature optimisation for flow-based proposals.

Given a single batch of latent samples drawn at temperature ``T`` from a
truncated isotropic Gaussian base density, this module estimates the optimal
latent temperature ``T*`` without redrawing from the flow.

The key identity is that, for a *fixed* flow ``f``, the ratio of pushforward
densities at two temperatures depends only on the latent radius ``r = |z|``::

    s_i(T, T') = q_{T'}(z_i) / q_T(z_i)
               = (T / T') ** (d / 2)
                 * (F_T / F_{T'})
                 * exp[ r_i ** 2 / 2 * (1 / T - 1 / T') ]

where ``F_T = P(d/2, R**2 / (2 T))`` is the regularised lower incomplete gamma
function (the fraction of an isotropic Gaussian of variance ``T`` inside the
ball of radius ``R``). The flow Jacobian cancels exactly.

See ``latent_temperature_optimization.md`` for the full derivation.
"""

import numpy as np
from scipy.special import gammainc, logsumexp

__all__ = [
    "log_effective_sample_size",
    "log_rejection_efficiency",
    "log_reweight_factor",
    "log_truncation_fraction",
    "optimise_latent_temperature",
    "reweight_log_weights",
    "secondary_effective_sample_size",
]


def log_truncation_fraction(dims, radius, temperature):
    """Log of the fraction of a truncated isotropic Gaussian inside the ball.

    ``F_T = P(d / 2, R ** 2 / (2 T))`` with ``P`` the regularised lower
    incomplete gamma function. Returns ``0`` when ``radius`` is infinite.

    Parameters
    ----------
    dims : int
        Dimensionality of the latent space.
    radius : float
        Truncation radius ``R``. Use ``np.inf`` for an untruncated Gaussian.
    temperature : float or array_like
        Latent temperature(s) ``T``.

    Returns
    -------
    float or numpy.ndarray
        ``log F_T``.
    """
    temperature = np.asarray(temperature, dtype=float)
    if not np.isfinite(radius):
        return np.zeros_like(temperature)[()]
    return np.log(gammainc(0.5 * dims, radius**2 / (2.0 * temperature)))


def log_reweight_factor(r, dims, temperature, new_temperature, radius=np.inf):
    """Log of the importance-sampling correction ``s_i(T, T')``.

    Parameters
    ----------
    r : array_like
        Per-sample latent radii ``r_i = |z_i|`` recorded at draw time.
    dims : int
        Dimensionality of the latent space.
    temperature : float
        Temperature ``T`` the samples were drawn at.
    new_temperature : float
        Target temperature ``T'``.
    radius : float, optional
        Truncation radius ``R``.

    Returns
    -------
    numpy.ndarray
        ``log s_i(T, T')``, one value per sample.
    """
    r = np.asarray(r, dtype=float)
    log_norm = (
        0.5 * dims * np.log(temperature / new_temperature)
        + log_truncation_fraction(dims, radius, temperature)
        - log_truncation_fraction(dims, radius, new_temperature)
    )
    return log_norm + 0.5 * r**2 * (1.0 / temperature - 1.0 / new_temperature)


def reweight_log_weights(
    log_w, r, dims, temperature, new_temperature, radius=np.inf
):
    """Reweight existing importance log-weights to a new temperature.

    ``w_i(T') = w_i(T) / s_i(T, T')``.
    """
    return np.asarray(log_w, dtype=float) - log_reweight_factor(
        r, dims, temperature, new_temperature, radius=radius
    )


def _log_ess(log_w):
    log_w = np.asarray(log_w, dtype=float)
    return 2.0 * logsumexp(log_w) - logsumexp(2.0 * log_w)


def log_effective_sample_size(log_w):
    """Log of Kish's effective sample size for the given log-weights."""
    return _log_ess(log_w)


def log_rejection_efficiency(log_w):
    """Log of the rejection-sampling efficiency ``mean(w) / max(w)``."""
    log_w = np.asarray(log_w, dtype=float)
    return logsumexp(log_w) - np.log(log_w.size) - np.max(log_w)


def secondary_effective_sample_size(log_s):
    """Effective sample size of the reweighting factor itself (Step 4).

    ``ESS_sec(T, T') = (sum_i s_i) ** 2 / sum_i s_i ** 2``. Equals ``N`` at
    ``T' = T`` and decays as ``T'`` moves away, measuring how trustworthy any
    quantity extrapolated from ``T`` to ``T'`` is.
    """
    return float(np.exp(_log_ess(log_s)))


def optimise_latent_temperature(
    log_w,
    r,
    dims,
    temperature,
    radius=np.inf,
    criterion="efficiency",
    tau=0.5,
    grid=None,
    n_grid=201,
    grid_range=(0.25, 4.0),
):
    """Estimate the optimal latent temperature from a single batch.

    Maximises the target-quality criterion over ``T'`` subject to the
    trust-region constraint ``ESS_sec(T, T') >= tau * N`` (Step 5).

    Parameters
    ----------
    log_w : array_like
        Existing importance log-weights ``log w_i(T)``.
    r : array_like
        Per-sample latent radii ``r_i = |z_i|``.
    dims : int
        Dimensionality of the latent space.
    temperature : float
        Temperature ``T`` the batch was drawn at.
    radius : float, optional
        Truncation radius ``R``.
    criterion : {"efficiency", "ess"}, optional
        Target-quality criterion to maximise (Step 3). Defaults to the
        rejection-sampling efficiency ``mean(w) / max(w)``.
    tau : float, optional
        Trust-region threshold as a fraction of ``N``.
    grid : array_like, optional
        Explicit grid of ``T'`` values. If ``None`` a log-spaced grid is built
        from ``grid_range`` (multiples of ``temperature``) with ``n_grid``
        points.
    n_grid, grid_range : int, tuple
        Used to build the default grid.

    Returns
    -------
    dict
        Keys: ``temperature`` (``T_hat*``), ``at_trust_boundary`` (bool, True if
        the optimum sits at the edge of the trust region -- iterate per Step 6),
        ``grid``, ``criterion`` (values on the grid), ``secondary_ess`` (values
        on the grid), ``trust_region`` (bool mask), ``n_samples``.
    """
    log_w = np.asarray(log_w, dtype=float)
    r = np.asarray(r, dtype=float)
    n = log_w.size

    if grid is None:
        grid = temperature * np.geomspace(grid_range[0], grid_range[1], n_grid)
    grid = np.asarray(grid, dtype=float)

    if criterion == "ess":
        score_fn = log_effective_sample_size
    elif criterion == "efficiency":
        score_fn = log_rejection_efficiency
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    scores = np.full(grid.size, -np.inf)
    sec_ess = np.zeros(grid.size)
    for i, tp in enumerate(grid):
        log_s = log_reweight_factor(r, dims, temperature, tp, radius=radius)
        sec_ess[i] = float(np.exp(_log_ess(log_s)))
        scores[i] = score_fn(log_w - log_s)

    trust = sec_ess >= tau * n
    if not np.any(trust):
        raise RuntimeError(
            "Trust region is empty; draw a smaller grid_range or lower tau."
        )

    masked = np.where(trust, scores, -np.inf)
    best = int(np.argmax(masked))
    trust_idx = np.flatnonzero(trust)
    at_boundary = best in (trust_idx[0], trust_idx[-1])

    return {
        "temperature": float(grid[best]),
        "at_trust_boundary": bool(at_boundary),
        "grid": grid,
        "criterion": scores,
        "secondary_ess": sec_ess,
        "trust_region": trust,
        "n_samples": n,
    }
