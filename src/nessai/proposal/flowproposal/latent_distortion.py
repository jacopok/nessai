"""Affine latent distortion of a flow proposal.

The flow draws its latent as ``z' = mu + sigma * z`` on the Gaussian
(non-circular) latent dimensions, so its latent density is
``N(z'; mu, sigma^2)``: an independent quadratic in each dimension's
log-density. The latent temperature is the special case ``mu = 0`` with one
shared ``sigma``. The flow density stays exact:

    log q'(u) = log q(u) + Delta(f(u)),
    Delta(z) = sum_i [log N(z_i; mu_i, s_i^2) - log N(z_i; 0, 1)].

The parameters have shape ``[E, D]``: one row per expert of a mixture flow,
``E = 1`` for a plain :class:`~nessai.flows.base.NFlow`, on which the
``"shared"`` and ``"expert"`` modes are then the same fit (per-dimension
``mu`` and ``sigma``).

With the prior-relative log-proposal cut, a populate yields ``X* / c`` new
points per flow draw, ``c = exp(-t)`` the weight cap set by the ``quantile``
of the live points' ``log q - log p``. The distortion is fitted after each
training to raise the low tail of ``log q' - log p`` over the live points
(the mean of the lowest ``tail`` fraction, a smooth stand-in for the
quantile), bounded (``|mu| <= MU_MAX``, ``sigma`` in ``[SIGMA_MIN,
SIGMA_MAX]``) and L2-regularised towards the identity. It matters most
where the flow undercovers the live points (heavy tails), which a ``sigma >
1`` broadens.

Checks before a fit is used (otherwise the identity is kept):

* a fit on one random half of the live points must raise the threshold on
  the other half (``min_gain``), and the fraction of that half its cut
  excludes may exceed the identity's (threshold from the first half too) by
  at most ``max_excess_factor * quantile``: the paired difference, since
  the excluded fraction of either alone is a count of a few points, too
  noisy to gate on;
* the decomposition must reproduce the flow's own distorted density on a
  subset of the live points (``exactness_tolerance``).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.special import logsumexp

from ...livepoint import live_points_to_array

logger = logging.getLogger(__name__)

MU_MAX = 1.5
SIGMA_MIN, SIGMA_MAX = 0.5, 3.0
MODES = ("temperature", "shared", "expert")

DEFAULT_KWARGS = dict(
    mode="expert",
    tail=0.14,
    reg=0.0035,
    steps=400,
    lr=0.05,
    min_gain=0.0,
    max_excess_factor=1.0,
    exactness_tolerance=1e-3,
    n_exactness=500,
    seed=1234,
)
# tail / regularisation per mode, from replays of gravitational-wave runs
# (expert: tuned on the held-out, gated threshold gain; the optimal tail is
# 0.1-0.2, a fit to the lowest 2% overfits).
MODE_DEFAULTS = {
    "temperature": dict(tail=0.01, reg=1e-3),
    "shared": dict(tail=0.05, reg=0.03),
    "expert": dict(tail=0.14, reg=0.0035),
}


def supports_distortion(model) -> bool:
    """Whether the flow model implements the distortion hooks."""
    return (
        model is not None
        and hasattr(model, "latent_components")
        and hasattr(model, "set_latent_distortion")
        and bool(getattr(model, "supports_latent_components", False))
    )


def _prime_tensor(proposal, x):
    n = len(x)
    x_prime, log_j = proposal.rescale(x.copy())
    arr = live_points_to_array(x_prime, proposal.prime_parameters, copy=True)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if len(arr) % n:
        raise RuntimeError(
            f"{len(arr)} rows for {n} points: not whole blocks of mirror "
            "copies"
        )
    return (
        proposal.flow.numpy_array_to_tensor(arr),
        np.asarray(log_j, dtype=float),
        len(arr) // n,
    )


def components(proposal, x):
    """Undistorted terms of the pool density of physical points ``x``.

    Returns ``lp`` ``(sides, n, experts)`` -- ``log pi_j + log q_j(x_s) +
    log|J_rescale|`` -- and ``z`` ``(sides, n, experts, dims)``, each
    expert's base-flow latent of each mirror copy. ``logsumexp`` of ``lp``
    over sides and experts is the pool's ``log q`` (with
    ``sum_mirror_density``, or without boundary-inversion copies).
    """
    n = len(x)
    xt, log_j, k = _prime_tensor(proposal, x)
    model = proposal.flow.model
    model.eval()
    lp, z = model.latent_components(xt)
    if lp.dim() == 1:  # a single group-mixture wrapper: one expert
        lp, z = lp[:, None], z[:, None, :]
    lp = lp.cpu().numpy().astype(float) + log_j[:, None]
    z = z.cpu().numpy().astype(float)
    return lp.reshape(k, n, -1), z.reshape(k, n, z.shape[-2], z.shape[-1])


def delta(z, mu, log_sigma, real):
    """``Delta(z)`` summed over the real dims (numpy or torch):
    ``z [..., E, D]``, ``mu`` / ``log_sigma`` ``[E, D]``, ``real [D]``."""
    lib = torch if isinstance(z, torch.Tensor) else np
    zz = z[..., real]
    m = mu[..., real]
    ls = log_sigma[..., real]
    return (-0.5 * ((zz - m) / lib.exp(ls)) ** 2 - ls + 0.5 * zz**2).sum(-1)


def log_q_distorted(lp, z, mu, log_sigma, real):
    """``log q'`` of each point from :func:`components`' terms."""
    t = lp + delta(z, mu, log_sigma, real)
    if isinstance(t, torch.Tensor):
        return torch.logsumexp(
            t.permute(1, 0, 2).reshape(t.shape[1], -1), dim=-1
        )
    return logsumexp(np.moveaxis(t, 0, 1).reshape(t.shape[1], -1), axis=-1)


def _bounds():
    lo, hi = np.log(SIGMA_MIN), np.log(SIGMA_MAX)
    return (lo + hi) / 2, (hi - lo) / 2


def _unpack(theta, shape, real, mode):
    """Bounded ``(mu, log_sigma)`` ``[E, D]`` from the free parameters."""
    n_exp, dims = shape
    nr = int(real.sum())
    mid, half = _bounds()
    mu = torch.zeros(n_exp, dims, dtype=theta.dtype)
    ls = torch.zeros(n_exp, dims, dtype=theta.dtype)
    idx = torch.as_tensor(np.flatnonzero(real))
    if mode == "temperature":
        ls[:, idx] = mid + half * torch.tanh(theta[0])
        return mu, ls
    if mode == "shared":
        th = theta.reshape(2, nr)
        mu[:, idx] = MU_MAX * torch.tanh(th[0])
        ls[:, idx] = mid + half * torch.tanh(th[1])
        return mu, ls
    th = theta.reshape(n_exp, 2, nr)
    mu[:, idx] = MU_MAX * torch.tanh(th[:, 0])
    ls[:, idx] = mid + half * torch.tanh(th[:, 1])
    return mu, ls


def _theta0(shape, real, mode):
    """Free parameters of the identity (``mu = 0``, ``sigma = 1``)."""
    mid, half = _bounds()
    a = np.arctanh(-mid / half)
    nr = int(real.sum())
    n = {"temperature": 1, "shared": 2 * nr, "expert": 2 * shape[0] * nr}[mode]
    th = np.zeros(n)
    if mode == "temperature":
        th[:] = a
    elif mode == "shared":
        th.reshape(2, -1)[1] = a
    else:
        th.reshape(shape[0], 2, -1)[:, 1] = a
    return torch.tensor(th, dtype=torch.float64, requires_grad=True)


def fit(
    lp, z, log_p, real, mode="expert", tail=0.02, reg=0.3, steps=400, lr=0.05
):
    """Maximise the mean of the lowest ``tail`` fraction of the levels
    ``log q' - log p``. Returns ``(mu, log_sigma)`` ``[E, D]`` arrays."""
    lp_t, z_t = torch.as_tensor(lp), torch.as_tensor(z)
    lpr, real_t = torch.as_tensor(log_p), torch.as_tensor(real)
    shape = (z.shape[-2], z.shape[-1])
    m = max(10, int(round(tail * lp.shape[1])))
    m = min(m, lp.shape[1])
    theta = _theta0(shape, real, mode)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        mu, ls = _unpack(theta, shape, real, mode)
        lev = log_q_distorted(lp_t, z_t, mu, ls, real_t) - lpr
        low = torch.topk(lev, m, largest=False).values
        loss = -low.mean() + reg * (mu**2 + ls**2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        mu, ls = _unpack(theta, shape, real, mode)
    return mu.numpy(), ls.numpy()


def fit_and_check(proposal, x, quantile, **kwargs):
    """Fit the distortion to the live points ``x`` and run the checks.

    Returns ``(mu, log_sigma, info)``; ``mu`` is ``None`` when a check
    fails (the identity is then kept). ``info`` holds the held-out gain and
    excluded fraction, the in-sample gain, the exactness error and the
    reason for a rejection.
    """
    opts = dict(DEFAULT_KWARGS)
    mode = kwargs.get("mode", opts["mode"])
    if mode not in MODES:
        raise ValueError(f"latent distortion mode must be one of {MODES}")
    opts.update(MODE_DEFAULTS[mode])
    opts.update(kwargs)
    info = dict(mode=mode, accepted=False)
    real = np.asarray(proposal.latent_real_mask, dtype=bool)
    fit_kw = {k: opts[k] for k in ("mode", "tail", "reg", "steps", "lr")}

    from .truncation import _with_auxiliary_parameters

    # the live points' auxiliary parameters (an angle's radius) drawn once,
    # for their prior and their flow density alike
    x = _with_auxiliary_parameters(proposal, x)
    lp, z = components(proposal, x)
    if lp.shape[0] > 1 and not getattr(proposal, "sum_mirror_density", False):
        # each draw is then scored with its own side's density only, which
        # the decomposition does not reproduce
        info["reason"] = "boundary-inversion copies need sum_mirror_density"
        return None, None, info
    log_p = np.asarray(proposal.log_prior(x.copy()), dtype=float)
    # an expert term without a finite latent does not contribute
    bad = ~np.isfinite(z).all(axis=-1)
    z[bad] = 0.0
    lp[bad] = -np.inf
    ok = np.isfinite(log_p) & np.isfinite(
        log_q_distorted(
            lp, z, np.zeros(z.shape[-2:]), np.zeros(z.shape[-2:]), real
        )
    )
    lp, z, log_p = lp[:, ok], z[:, ok], log_p[ok]
    n = lp.shape[1]
    zero = np.zeros(z.shape[-2:])

    def levels(mu, ls, idx=slice(None)):
        return (
            log_q_distorted(lp[:, idx], z[:, idx], mu, ls, real) - log_p[idx]
        )

    # held-out gate: fit on one half, test on the other
    rng = np.random.default_rng(opts["seed"])
    perm = rng.permutation(n)
    a, b = perm[: n // 2], perm[n // 2 :]
    mu_a, ls_a = fit(lp[:, a], z[:, a], log_p[a], real, **fit_kw)
    lev_b = levels(mu_a, ls_a, b)
    gain = float(
        np.quantile(lev_b, quantile)
        - np.quantile(levels(zero, zero, b), quantile)
    )
    t_a = np.quantile(levels(mu_a, ls_a, a), quantile)
    outside = float(np.mean(lev_b < t_a))
    outside_id = float(
        np.mean(
            levels(zero, zero, b)
            < np.quantile(levels(zero, zero, a), quantile)
        )
    )
    excess = outside - outside_id
    info.update(
        heldout_gain=gain, heldout_excluded=outside, heldout_excess=excess
    )
    if not gain > opts["min_gain"]:
        info["reason"] = f"held-out gain {gain:.3g} <= {opts['min_gain']}"
        return None, None, info
    if excess > opts["max_excess_factor"] * quantile:
        info["reason"] = (
            f"held-out excluded fraction {outside:.3g} exceeds the "
            f"identity's ({outside_id:.3g}) by more than "
            f"{opts['max_excess_factor']} x {quantile}"
        )
        return None, None, info

    mu, ls = fit(lp, z, log_p, real, **fit_kw)
    info["gain"] = float(
        np.quantile(levels(mu, ls), quantile)
        - np.quantile(levels(zero, zero), quantile)
    )

    # exactness: the flow's own distorted density against the formula
    model = proposal.flow.model
    sub = rng.choice(n, size=min(n, opts["n_exactness"]), replace=False)
    xs = x[ok][sub]
    model.set_latent_distortion(mu, ls)
    try:
        actual = logsumexp(proposal.mirror_log_q(xs), axis=0)
    finally:
        model.set_latent_distortion(None)
    err = float(np.max(np.abs(actual - levels(mu, ls, sub) - log_p[sub])))
    info["exactness_error"] = err
    if not err < opts["exactness_tolerance"]:
        info["reason"] = f"decomposition error {err:.3g}"
        return None, None, info
    info["accepted"] = True
    return mu, ls, info


def roundtrip_gap(proposal, x, z):
    """For draws ``x`` generated from the standard latents ``z``: the
    smallest difference, over the experts and mirror copies, between
    ``|z|`` and the norm of the (standardised) latent the flow's forward map
    gives for the draw, on the real dims. A draw that the inverse and
    forward maps agree on has a gap ~0; one they disagree on (a numerical
    failure of the inverse) a large gap, and a density the flow does not
    describe. ``None`` if the model has no latent decomposition."""
    model = getattr(proposal.flow, "model", None)
    if not supports_distortion(model) or not len(x):
        return None
    real = np.asarray(proposal.latent_real_mask, dtype=bool)
    n = len(x)
    if n == 1:
        # some reparameterisations' table lookups fail on a single point
        x, z = np.concatenate([x, x]), np.concatenate([z, z])
    lpf, zf = components(proposal, x)  # (sides, n, E, D), undistorted noise
    lpf, zf = lpf[:, :n], zf[:, :n]
    z = np.asarray(z)[:n]
    zf = np.where(np.isfinite(lpf)[..., None], zf, np.inf)
    dist = model.latent_distortion
    if dist is not None:
        mu, ls = dist
        zf = (zf - mu) / np.exp(ls)
    nf = np.linalg.norm(zf[..., real], axis=-1)  # (sides, n, E)
    ng = np.linalg.norm(np.asarray(z, dtype=float)[:, real], axis=-1)
    return np.min(np.abs(nf - ng[None, :, None]), axis=(0, 2))
