"""Per-element latent Gaussians of a group-mixture flow proposal.

Every group element of a group-mixture expert draws from the same base flow
in the fundamental domain, so an element whose mode differs from the one the
flow was trained on (the primary, holding most live points) is proposed
with the primary's shape: a secondary mode that sits off the primary's
centre, is narrower, or shrinks as it dies (its restricted region a small
part of the primary's) has most of its draws fall outside its contour. The
weights cannot help: an element's live points must stay covered by the clip
cap, so its weight rises until they are, and the draws are spent where the
shared shape puts them.

Here each such element gets its own latent Gaussian, fitted to its live
points: with ``n`` the base-flow noise of a point (no distortion), the
element's live points have mean ``c_g`` and covariance ``S_g`` (shrunk
towards ``s_g^2 I`` with ``kappa`` pseudo-points, ``kappa = dims + 2`` by
default, and inflated by ``inflation^2``), and the element draws
``n = m_g + L_g y`` with ``y`` the (distorted) latent ``mu + sigma z``:
``L_g = chol(S_g) diag(1 / sigma)`` and ``m_g = c_g - L_g mu`` give ``n ~
N(c_g, S_g)``. The density is exact (an affine layer per element, see
:meth:`~nessai.flowmodel.group_mixture.DiscreteGroupMixtureFlowWrapper.set_element_latent`).

Only elements with at least ``min_count`` live points and under
``max_share`` of the expert's are fitted (the primary keeps the trained
base flow); the augmented (not real) latent dims keep the identity. Fitted
after each training (after the latent distortion, before the element
weights) and before each populate.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from .branch_weights import _experts
from .latent_distortion import _prime_tensor, components

logger = logging.getLogger(__name__)

DEFAULT_KWARGS = dict(min_count=10, max_share=0.5, inflation=1.25, kappa=None, shape="full")


def supports_element_latent(model) -> bool:
    """Whether the flow model implements the element-latent hooks."""
    experts = _experts(model) if model is not None else []
    return bool(experts) and all(
        hasattr(e, "set_element_latent") and hasattr(e, "latent_noise")
        for e in experts
    )


def fit(proposal, x, min_count=10, max_share=0.5, inflation=1.25, kappa=None,
        shape="full"):
    """Element latents of every active expert, fitted to the live points
    ``x``. Returns ``(params, info)``: ``params`` a list of ``(mu, chol)``
    per expert (``None`` for an expert with no element fitted).

    ``shape="shift"`` keeps the shared shape and moves each fitted element's
    latent by the offset of its live points' mean noise from that of all the
    expert's (``L_g = I``, ``m_g = c_g - c``): one mean per element, so far
    less prone to over-fitting a few points than ``"full"`` (mean and
    covariance), which the cap, set by the same points, cannot detect."""
    model = proposal.flow.model
    experts = _experts(model)
    n = len(x)
    xt, _, _ = _prime_tensor(proposal, x)
    xt = xt[:n]
    if len(experts) > 1:
        lp, _ = components(proposal, x)
        lp0 = np.where(np.isfinite(lp[0]), lp[0], -np.inf)
        route = np.argmax(lp0, axis=1)
    else:
        route = np.zeros(n, dtype=int)
    real = np.asarray(
        getattr(proposal, "latent_real_mask", None)
        if getattr(proposal, "latent_real_mask", None) is not None
        else np.ones(xt.shape[1], dtype=bool),
        dtype=bool,
    )
    params, fitted = [], []
    for j, e in enumerate(experts):
        sel = route == j
        k = int(e.group_size)
        if sel.sum() < min_count:
            params.append(None)
            fitted.append([])
            continue
        with torch.no_grad():
            noise, g = e.latent_noise(xt[torch.as_tensor(sel)])
        noise = noise.detach().cpu().numpy().astype(float)
        g = g.detach().cpu().numpy()
        d = noise.shape[1]
        r = real[:d] if len(real) >= d else np.ones(d, dtype=bool)
        dr = int(r.sum())
        dist = e.latent_distortion
        mu_glob = np.zeros(d) if dist is None else np.asarray(dist[0], float)
        sig_glob = np.ones(d) if dist is None else np.exp(np.asarray(dist[1], float))
        mu = np.zeros((k, d))
        chol = np.tile(np.eye(d), (k, 1, 1))
        done = []
        counts = np.bincount(g, minlength=k)
        kap = dr + 2 if kappa is None else kappa
        c_all = noise[:, r].mean(0)
        for el in np.nonzero(counts >= min_count)[0]:
            if counts[el] / sel.sum() >= max_share:
                continue
            v = noise[g == el][:, r]
            c = v.mean(0)
            if shape == "shift":
                # James-Stein (positive part): a shift at the noise level of
                # the mean of a few points shrinks to nothing, a real offset
                # stays (an unneeded shift costs coverage the cap cannot see)
                off = c - c_all
                s2g = float(((v - c) ** 2).mean())
                norm2 = float(off @ off)
                f = max(0.0, 1.0 - (dr - 2) * s2g / (len(v) * norm2)) if norm2 > 0 else 0.0
                if f > 0:
                    mu[el, r] = f * off
                    done.append(int(el))
                continue
            dev = v - c
            s2 = float((dev**2).mean())
            if not s2 > 0:
                continue
            c_hat = dev.T @ dev / max(len(v) - 1, 1)
            cov = (len(v) * c_hat + kap * s2 * np.eye(dr)) / (len(v) + kap)
            try:
                lc = np.linalg.cholesky(cov * inflation**2)
            except np.linalg.LinAlgError:
                continue
            lg = lc / sig_glob[r][None, :]
            block = np.eye(d)
            block[np.ix_(r, r)] = lg
            chol[el] = block
            mu[el, r] = c - lg @ mu_glob[r]
            done.append(int(el))
        params.append((mu, chol) if done else None)
        fitted.append(done)
    info = dict(
        accepted=any(fitted),
        elements=fitted,
        n_elements=int(sum(len(f) for f in fitted)),
    )
    return params, info
