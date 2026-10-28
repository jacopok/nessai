"""Fitted group-element (branch) weights of a group-mixture flow proposal.

Each expert's density is a mixture over the group elements ``g``,
``q_j(x) = sum_g w_jg q_jg(x)``, and with a fundamental domain that tiles the
space exactly one element claims each point: ``log q_j(x) = log w_jg(x) +
(the base term)``. A change of the weights to ``w'`` therefore shifts each
point's log-density by ``log w'_jg(x) - log w_jg(x)``, without evaluating the
flow again.

The element weights do not cancel in the pool: with the prior-relative
log-proposal cut the pool follows ``min(prior, c q)``, ``c`` the weight cap
set by the ``quantile`` of the live points' ``log q - log p``, and an element
the live points rarely occupy, at the weight its prior mass gives it (the
importance weights), lowers the levels of none of them while its draws fall
outside the contour. The weights are fitted after each training as the latent
distortion is (:mod:`.latent_distortion`): to raise the low tail of ``log q'
- log p`` over the live points (the mean of the lowest ``tail`` fraction),
bounded (``|log w' / w| <= LOG_RATIO_MAX`` before normalisation) and
L2-regularised towards the weights in place, with the latent distortion
already installed held fixed.

Checks before a fit is used (otherwise the weights in place are kept), as
for the distortion:

* a fit on one random half of the live points must raise the threshold on
  the other half (``min_gain``), and the fraction of that half its cut
  excludes may exceed that of the weights in place by at most
  ``max_excess_factor * quantile``;
* the decomposition must reproduce the flow's own density under the fitted
  weights on a subset of the live points (``exactness_tolerance``): it is
  exact only where one element claims each point.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.special import logsumexp

from .latent_distortion import _prime_tensor, components, delta

logger = logging.getLogger(__name__)

LOG_RATIO_MAX = 3.0

DEFAULT_KWARGS = dict(
    tail=0.14,
    reg=0.0035,
    steps=400,
    lr=0.05,
    min_gain=0.0,
    max_excess_factor=1.0,
    exactness_tolerance=1e-3,
    n_exactness=500,
    seed=4321,
)


def _experts(model):
    """The active group-mixture wrappers of ``model``."""
    experts = getattr(model, "experts", None)
    if experts is None:
        return [model]
    return [experts[j] for j in range(model._active())]


def supports_branch_weights(model) -> bool:
    """Whether the flow model implements the branch-weight hooks."""
    return (
        model is not None
        and hasattr(model, "set_branch_weights")
        and hasattr(model, "latent_components")
        and bool(getattr(model, "supports_latent_components", False))
    )


def branch_terms(proposal, x):
    """Terms of the pool density of physical points ``x`` under the latent
    distortion in place: ``lp`` ``(sides, n, experts)`` (as
    :func:`.latent_distortion.components`, distortion included), the element
    ``a`` ``(sides, n, experts)`` claiming each mirror copy for each expert
    (``-1`` where none does) and the experts' element weights ``w0``
    ``(experts, group_size)`` in place."""
    model = proposal.flow.model
    lp, z = components(proposal, x)
    dist = getattr(model, "latent_distortion", None)
    if dist is not None:
        real = np.asarray(proposal.latent_real_mask, dtype=bool)
        mu, ls = dist
        zz = np.where(np.isfinite(z), z, 0.0)
        lp = lp + delta(zz, mu, ls, real)
    xt, _, k = _prime_tensor(proposal, x)
    n = len(x)
    experts = _experts(model)
    a = np.full((k * n, len(experts)), -1, dtype=int)
    with torch.no_grad():
        for j, e in enumerate(experts):
            assigned, _, claimed = e._assign_branch(xt)
            a[:, j] = np.where(
                claimed.cpu().numpy(), assigned.cpu().numpy(), -1
            )
    w0 = np.stack(
        [e.proposal_weights.detach().cpu().numpy() for e in experts]
    ).astype(float)
    return lp, a.reshape(k, n, len(experts)), w0


def _log_weights(theta, log_w0, active):
    """Normalised fitted log-weights ``[E, K]`` from the free parameters."""
    lw = log_w0 + LOG_RATIO_MAX * torch.tanh(theta)
    lw = torch.where(active, lw, torch.full_like(lw, -torch.inf))
    return lw - torch.logsumexp(lw, dim=-1, keepdim=True)


def levels_of(lp, a, log_w0, log_w, log_p):
    """``log q' - log p`` of each point under the log-weights ``log_w``
    (numpy or torch, ``[E, K]``)."""
    lib = torch if isinstance(lp, torch.Tensor) else np
    e_idx = lib.arange(a.shape[-1])
    safe = lib.where(a >= 0, a, 0)
    shift = log_w[e_idx, safe] - log_w0[e_idx, safe]
    zero = torch.zeros_like(shift) if lib is torch else 0.0
    t = lp + lib.where(a >= 0, shift, zero)
    if lib is torch:
        flat = t.permute(1, 0, 2).reshape(t.shape[1], -1)
        return torch.logsumexp(flat, dim=-1) - log_p
    flat = np.moveaxis(t, 0, 1).reshape(t.shape[1], -1)
    return logsumexp(flat, axis=-1) - log_p


def fit(lp, a, w0, log_p, tail=0.14, reg=0.0035, steps=400, lr=0.05):
    """Maximise the mean of the lowest ``tail`` fraction of the levels
    ``log q' - log p``. Returns the fitted weights ``[E, K]``."""
    lp_t, a_t = torch.as_tensor(lp), torch.as_tensor(a)
    lpr = torch.as_tensor(log_p)
    active = torch.as_tensor(w0 > 0)
    log_w0 = torch.log(torch.as_tensor(w0).clamp_min(1e-300))
    log_w0 = torch.where(active, log_w0, torch.zeros_like(log_w0))
    m = min(max(10, int(round(tail * lp.shape[1]))), lp.shape[1])
    theta = torch.zeros(w0.shape, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        lw = _log_weights(theta, log_w0, active)
        lev = levels_of(lp_t, a_t, log_w0, lw, lpr)
        low = torch.topk(lev, m, largest=False).values
        loss = -low.mean() + reg * (theta**2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        lw = _log_weights(theta, log_w0, active)
    return torch.exp(lw).numpy()


def fit_and_check(proposal, x, quantile, **kwargs):
    """Fit the element weights to the live points ``x`` and run the checks.

    Returns ``(weights, info)``; ``weights`` (``[experts, group_size]``) is
    ``None`` when a check fails (the weights in place are then kept).
    ``info`` holds the held-out gain and excluded fraction, the in-sample
    gain, the exactness error and the reason for a rejection.
    """
    opts = dict(DEFAULT_KWARGS)
    opts.update(kwargs)
    info = dict(accepted=False)
    fit_kw = {k: opts[k] for k in ("tail", "reg", "steps", "lr")}

    lp, a, w0 = branch_terms(proposal, x)
    if lp.shape[0] > 1 and not getattr(proposal, "sum_mirror_density", False):
        info["reason"] = "boundary-inversion copies need sum_mirror_density"
        return None, info
    if not (w0 > 0).sum(axis=-1).max() > 1:
        info["reason"] = "one active element"
        return None, info
    log_p = np.asarray(proposal.log_prior(x.copy()), dtype=float)
    log_w0 = np.where(w0 > 0, np.log(np.where(w0 > 0, w0, 1.0)), 0.0)
    ok = np.isfinite(log_p) & np.isfinite(
        levels_of(lp, a, log_w0, log_w0, np.zeros_like(log_p))
    )
    lp, a, log_p = lp[:, ok], a[:, ok], log_p[ok]
    n = lp.shape[1]

    def levels(w, idx=slice(None)):
        lw = np.where(w > 0, np.log(np.where(w > 0, w, 1.0)), -np.inf)
        return levels_of(lp[:, idx], a[:, idx], log_w0, lw, log_p[idx])

    rng = np.random.default_rng(opts["seed"])
    perm = rng.permutation(n)
    ia, ib = perm[: n // 2], perm[n // 2:]
    w_a = fit(lp[:, ia], a[:, ia], w0, log_p[ia], **fit_kw)
    lev_b, lev0_b = levels(w_a, ib), levels(w0, ib)
    gain = float(
        np.quantile(lev_b, quantile) - np.quantile(lev0_b, quantile)
    )
    outside = float(np.mean(lev_b < np.quantile(levels(w_a, ia), quantile)))
    outside0 = float(np.mean(lev0_b < np.quantile(levels(w0, ia), quantile)))
    excess = outside - outside0
    info.update(heldout_gain=gain, heldout_excluded=outside,
                heldout_excess=excess)
    if not gain > opts["min_gain"]:
        info["reason"] = f"held-out gain {gain:.3g} <= {opts['min_gain']}"
        return None, info
    if excess > opts["max_excess_factor"] * quantile:
        info["reason"] = (
            f"held-out excluded fraction {outside:.3g} exceeds that of the "
            f"weights in place ({outside0:.3g}) by more than "
            f"{opts['max_excess_factor']} x {quantile}"
        )
        return None, info

    w = fit(lp, a, w0, log_p, **fit_kw)
    info["gain"] = float(
        np.quantile(levels(w), quantile) - np.quantile(levels(w0), quantile)
    )

    # exactness: the flow's own density under the fitted weights
    model = proposal.flow.model
    sub = rng.choice(n, size=min(n, opts["n_exactness"]), replace=False)
    xs = x[ok][sub]
    previous = getattr(model, "fitted_branch_weights", None)
    model.set_branch_weights(list(w) if hasattr(model, "experts") else w[0])
    try:
        actual = logsumexp(proposal.mirror_log_q(xs), axis=0)
    finally:
        model.set_branch_weights(previous)
    err = float(np.max(np.abs(actual - levels(w, sub) - log_p[sub])))
    info["exactness_error"] = err
    if not err < opts["exactness_tolerance"]:
        info["reason"] = f"decomposition error {err:.3g}"
        return None, info
    info["accepted"] = True
    info["weights_in_place"] = w0
    return w, info
