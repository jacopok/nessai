"""Toy benchmark: flows on targets with periodic parameters.

Targets live on (phi, psi, x, y): phi with period 2 pi, psi with period pi
(as the GW phase and polarisation), x and y real. Each method is trained on
``n_train`` target draws (10% held out for early stopping) with the xg_pe
``v18`` settings (6 coupling layers, 2 residual blocks of 64 neurons, LU
and batch norm between layers; AdamW, lr 1e-3, batch 1000, patience 20) and
scored on

* ``kl``: forward KL, the mean of ``log p - log q`` over fresh target draws,
  with ``q`` normalised on the target's domain (the flow's mass that falls
  outside is discarded, as nessai rejects points outside the prior);
* ``ess``: the effective sample size of ``p / q`` over draws from the flow,
  as a fraction of the draws, with points outside the domain counted as
  zero weight -- the efficiency of the proposal for rejection or importance
  sampling;
* ``inside``: the fraction of flow draws that land on the domain.

For the ``chi`` methods (nessai's ``Angle``: an angle and an auxiliary
chi(2) radius as two Cartesian coordinates) both are computed in the
augmented space, which is what nessai samples.

Usage: python toy_benchmark.py OUTDIR CASE METHOD SEED
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import special, stats

from nessai.flows import NeuralSplineFlow, RealNVP
from nessai.flows.circular import CircularNeuralSplineFlow

TWO_PI = 2 * math.pi
PERIODS = np.array([TWO_PI, math.pi])  # phi, psi
NAMES = ["phi", "psi", "x", "y"]


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------


def vm_logpdf(theta, mu, kappa):
    """von Mises log density of an angle (any branch)."""
    return kappa * (np.cos(theta - mu) - 1.0) - np.log(
        TWO_PI * special.i0e(kappa)
    )


def wrap_angle(a, period):
    """Angle on [0, period); ``np.mod`` can round up to ``period`` itself."""
    a = np.mod(a, period)
    return np.where(a >= period, 0.0, a)


def norm_logpdf(x):
    return -0.5 * x**2 - 0.5 * np.log(TWO_PI)


class Target:
    """Base class: ``sample(n, rng)`` and ``log_prob(X)``."""

    def in_domain(self, X):
        return (
            (X[:, 0] >= 0)
            & (X[:, 0] < TWO_PI)
            & (X[:, 1] >= 0)
            & (X[:, 1] < math.pi)
        )


class Uniform(Target):
    """Uniform angles, correlated Gaussian (x, y) independent of them."""

    rho = 0.8

    def sample(self, n, rng):
        phi = rng.uniform(0, TWO_PI, n)
        psi = rng.uniform(0, math.pi, n)
        x = rng.standard_normal(n)
        y = self.rho * x + math.sqrt(1 - self.rho**2) * rng.standard_normal(n)
        return np.stack([phi, psi, x, y], axis=1)

    def log_prob(self, X):
        x, y = X[:, 2], X[:, 3]
        s = math.sqrt(1 - self.rho**2)
        lp = (
            -np.log(TWO_PI)
            - np.log(math.pi)
            + norm_logpdf(x)
            + norm_logpdf((y - self.rho * x) / s)
            - np.log(s)
        )
        return np.where(self.in_domain(X), lp, -np.inf)


class Peak(Target):
    """Sharp phase peak (kappa 400, ~3 deg) whose location sweeps the whole
    circle with x; a psi peak (kappa 100 on 2 psi) that moves with y."""

    k_phi = 400.0
    k_psi = 100.0

    def mu_phi(self, x):
        return 1.0 * x

    def mu_2psi(self, y):
        return math.pi + 0.8 * y

    def sample(self, n, rng):
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        phi = wrap_angle(rng.vonmises(self.mu_phi(x), self.k_phi), TWO_PI)
        psi = wrap_angle(rng.vonmises(self.mu_2psi(y), self.k_psi), TWO_PI) / 2
        return np.stack([phi, psi, x, y], axis=1)

    def log_prob(self, X):
        phi, psi, x, y = X.T
        lp = (
            norm_logpdf(x)
            + norm_logpdf(y)
            + vm_logpdf(phi, self.mu_phi(x), self.k_phi)
            + vm_logpdf(2 * psi, self.mu_2psi(y), self.k_psi)
            + np.log(2)
        )
        return np.where(self.in_domain(X), lp, -np.inf)


class Multimodal(Target):
    """Three phase modes (kappa 8) that shift with x, with weights set by y;
    a broad psi mode that moves with x."""

    mus = np.array([0.3, 2.4, 4.4])
    kappa = 8.0

    def logits(self, y):
        return np.stack([0.8 * y, 0 * y, -0.8 * y], axis=1)

    def sample(self, n, rng):
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        w = special.softmax(self.logits(y), axis=1)
        k = (rng.uniform(size=(n, 1)) > np.cumsum(w, axis=1)).sum(axis=1)
        k = np.minimum(k, len(self.mus) - 1)
        phi = wrap_angle(
            rng.vonmises(self.mus[k] + 0.6 * x, self.kappa), TWO_PI
        )
        psi = wrap_angle(rng.vonmises(1.0 + 0.5 * x, 3.0), TWO_PI) / 2
        return np.stack([phi, psi, x, y], axis=1)

    def log_prob(self, X):
        phi, psi, x, y = X.T
        logw = special.log_softmax(self.logits(y), axis=1)
        comp = vm_logpdf(
            phi[:, None], self.mus[None, :] + 0.6 * x[:, None], self.kappa
        )
        lp = (
            norm_logpdf(x)
            + norm_logpdf(y)
            + special.logsumexp(logw + comp, axis=1)
            + vm_logpdf(2 * psi, 1.0 + 0.5 * x, 3.0)
            + np.log(2)
        )
        return np.where(self.in_domain(X), lp, -np.inf)


class Stripe(Target):
    """GW-like polarisation-phase degeneracy: phi + 2 psi (y > 0) or
    phi - 2 psi (y < 0) is measured (kappa 20) and its value moves with x,
    the other combination is free. The stripe winds around the torus."""

    kappa = 20.0

    def c(self, x):
        return 1.0 + 0.8 * x

    def sample(self, n, rng):
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        s = np.where(rng.uniform(size=n) < special.expit(3 * y), 1.0, -1.0)
        psi = rng.uniform(0, math.pi, n)
        u = rng.vonmises(self.c(x), self.kappa)
        phi = wrap_angle(u - 2 * s * psi, TWO_PI)
        return np.stack([phi, psi, x, y], axis=1)

    def log_prob(self, X):
        phi, psi, x, y = X.T
        lplus = vm_logpdf(phi + 2 * psi, self.c(x), self.kappa)
        lminus = vm_logpdf(phi - 2 * psi, self.c(x), self.kappa)
        mix = np.logaddexp(
            special.log_expit(3 * y) + lplus,
            special.log_expit(-3 * y) + lminus,
        )
        lp = norm_logpdf(x) + norm_logpdf(y) + mix - np.log(math.pi)
        return np.where(self.in_domain(X), lp, -np.inf)


TARGETS = {
    "uniform": Uniform,
    "peak": Peak,
    "multimodal": Multimodal,
    "stripe": Stripe,
}


# --------------------------------------------------------------------------
# methods: map to the flow's space, train, evaluate in the target's space
# --------------------------------------------------------------------------

FLOW = dict(hidden_features=64, num_layers=6, num_blocks_per_layer=2)
TRAIN = dict(
    lr=1e-3, batch_size=1000, max_epochs=500, patience=20, clip=5.0, val=0.1
)
if "TOY_MAX_EPOCHS" in __import__("os").environ:
    TRAIN["max_epochs"] = int(__import__("os").environ["TOY_MAX_EPOCHS"])


def cosine_taper(d, width):
    """Keep probability of a ghost at distance ``d`` outside the domain."""
    return np.where(d < width, 0.5 * (1 + np.cos(np.pi * d / width)), 0.0)


def pou_weight(t, width):
    """Partition-of-unity weight on box coordinates (period 2): one on
    ``|t| < 1 - width``, falling smoothly to zero at ``|t| = 1 + width``, with
    ``v(t) + v(t - 2) = 1`` across each seam, so folding ``p v`` gives ``p``.
    """
    a = np.abs(t)
    v = 0.5 * (1 - np.sin(np.pi * np.clip(a - 1, -width, width) / (2 * width)))
    return np.where(a >= 1 + width, 0.0, v)


class Method:
    """Flow on a transformed space. Subclasses define the map."""

    #: extra latent dimensions (auxiliary radii)
    n_aux = 0

    def __init__(self, flow_type, rng, **flow_kwargs):
        self.flow_type = flow_type
        self.rng = rng
        self.flow_kwargs = flow_kwargs

    # standardise x, y with the training statistics
    def fit_real(self, X):
        self.mean = X[:, 2:].mean(axis=0)
        self.std = X[:, 2:].std(axis=0)

    def real_fwd(self, X):
        return (X[:, 2:] - self.mean) / self.std

    def real_inv(self, Y):
        return Y * self.std + self.mean

    def real_logj(self):
        return -np.log(self.std).sum()

    def make_flow(self, dims):
        if self.flow_type == "realnvp":
            return RealNVP(
                dims, **FLOW, batch_norm_between_layers=True, linear_transform="lu"
            )
        if self.flow_type == "nsf":
            return NeuralSplineFlow(
                dims,
                **FLOW,
                num_bins=8,
                tail_bound=5.0,
                batch_norm_between_layers=True,
                linear_transform="lu",
            )
        raise ValueError(self.flow_type)

    def augment(self, Y):
        """Training-set augmentation in the flow's space and the training
        weights (default: none, unit weights)."""
        return Y, np.ones(len(Y))

    def train(self, X):
        self.fit_real(X)
        n_val = int(TRAIN["val"] * len(X))
        perm = self.rng.permutation(len(X))
        # augment the two splits separately: ghosts stay with their source
        Yt, wt = self.augment(self.to_flow(X[perm[n_val:]])[0])
        Yv, wv = self.augment(self.to_flow(X[perm[:n_val]])[0])
        self.n_train_aug = len(Yt)
        self.flow = self.make_flow(Yt.shape[1])
        self.history = train_flow(self.flow, Yt, Yv, wt, wv)

    @torch.no_grad()
    def flow_log_prob(self, Y):
        out = np.full(len(Y), -np.inf)
        ok = np.isfinite(Y).all(axis=1)
        t = torch.as_tensor(Y[ok], dtype=torch.get_default_dtype())
        lp = []
        for chunk in torch.split(t, 20000):
            lp.append(self.flow.log_prob(chunk).numpy())
        out[ok] = np.concatenate(lp) if lp else []
        return out

    @torch.no_grad()
    def flow_sample(self, n):
        Y, lp = [], []
        for m in [20000] * (n // 20000) + [n % 20000]:
            if m:
                y, l = self.flow.sample_and_log_prob(m)
                Y.append(y.numpy())
                lp.append(l.numpy())
        return np.concatenate(Y).astype(float), np.concatenate(lp)

    # ---- evaluation interface -------------------------------------------
    def log_q(self, X):
        """log q(X) on the domain, unnormalised (see ``log_norm``)."""
        Y, logj = self.to_flow(X)
        return self.flow_log_prob(Y) + logj

    def sample(self, n):
        """Draws in the target space (NaN rows: off the domain) with log q."""
        Y, lp = self.flow_sample(n)
        X, logj = self.from_flow(Y)
        return X, lp + logj


class Box(Method):
    """Angles rescaled linearly to [-1, 1): nessai's default for a bounded
    parameter. No periodicity."""

    def to_flow(self, X):
        a = 2 * X[:, :2] / PERIODS - 1
        logj = np.log(2 / PERIODS).sum() + self.real_logj()
        return np.concatenate([a, self.real_fwd(X)], axis=1), logj

    def from_flow(self, Y):
        X = np.concatenate(
            [(Y[:, :2] + 1) * PERIODS / 2, self.real_inv(Y[:, 2:])], axis=1
        )
        logj = np.log(2 / PERIODS).sum() + self.real_logj()
        out = (np.abs(Y[:, :2]) > 1).any(axis=1)
        X[out] = np.nan
        return X, np.full(len(Y), logj)


class Chi(Method):
    """nessai's ``Angle``: each angle with a chi(2) auxiliary radius as two
    Cartesian coordinates. Evaluated in the augmented space."""

    n_aux = 2

    def to_flow(self, X):
        r = stats.chi(2).rvs(size=(len(X), 2), random_state=self.rng)
        ang = TWO_PI * X[:, :2] / PERIODS
        Y = np.concatenate(
            [r * np.cos(ang), r * np.sin(ang), self.real_fwd(X)], axis=1
        )
        # q_eff(X) = q_aug(X, r) / chi(r): its ratio to p equals the
        # augmented-space weight
        logj = (
            np.log(r).sum(axis=1)
            + np.log(TWO_PI / PERIODS).sum()
            + self.real_logj()
            - stats.chi(2).logpdf(r).sum(axis=1)
        )
        return Y, logj

    def from_flow(self, Y):
        r = np.hypot(Y[:, :2], Y[:, 2:4])
        ang = wrap_angle(np.arctan2(Y[:, 2:4], Y[:, :2]), TWO_PI)
        X = np.concatenate(
            [ang * PERIODS / TWO_PI, self.real_inv(Y[:, 4:])], axis=1
        )
        logj = (
            np.log(r).sum(axis=1)
            + np.log(TWO_PI / PERIODS).sum()
            + self.real_logj()
            - stats.chi(2).logpdf(r).sum(axis=1)
        )
        return X, logj


class Ghost(Box):
    """Box coordinates, trained with damped periodic ghosts: copies of the
    points within ``pad`` (a fraction of the period) of either edge, shifted
    by one period across it and kept with a cosine-tapered probability.

    ``mode='discard'`` drops flow draws outside the domain; ``mode='fold'``
    keeps draws within the padded domain, wraps them back and uses the
    density summed over their images.
    """

    def __init__(
        self, flow_type, rng, pad=0.25, mode="discard", pou=False, **kw
    ):
        super().__init__(flow_type, rng, **kw)
        self.pad = pad
        self.mode = mode
        self.pou = pou

    def augment(self, Y):
        w = 2 * self.pad  # pad width in [-1, 1] units (period 2)
        if self.pou:
            # every image, weighted by the partition of unity: the weights
            # of a point and its ghosts sum to one
            Ys, ws = [], []
            for s0 in (0.0, 2.0, -2.0):
                for s1 in (0.0, 2.0, -2.0):
                    Z = Y.copy()
                    Z[:, 0] += s0
                    Z[:, 1] += s1
                    v = pou_weight(Z[:, 0], w) * pou_weight(Z[:, 1], w)
                    Ys.append(Z[v > 0])
                    ws.append(v[v > 0])
            return np.concatenate(Ys), np.concatenate(ws)
        for d in range(2):
            ghosts = []
            for shift in (2.0, -2.0):
                G = Y.copy()
                G[:, d] += shift
                dist = np.abs(G[:, d]) - 1  # distance outside the box
                keep = self.rng.uniform(size=len(G)) < cosine_taper(dist, w)
                ghosts.append(G[keep & (dist > 0)])
            Y = np.concatenate([Y] + ghosts)
        return Y, np.ones(len(Y))

    def _images(self, Y):
        """All images of Y (in the box) within the padded box: list of
        arrays, NaN where an image is outside the padded box."""
        w = 2 * self.pad
        images = []
        for s0 in (0.0, 2.0, -2.0):
            for s1 in (0.0, 2.0, -2.0):
                Z = Y.copy()
                Z[:, 0] += s0
                Z[:, 1] += s1
                bad = (np.abs(Z[:, :2]) > 1 + w).any(axis=1)
                Z[bad] = np.nan
                images.append(Z)
        return images

    def log_q(self, X):
        Y, logj = self.to_flow(X)
        if self.mode == "discard":
            return self.flow_log_prob(Y) + logj
        lps = np.stack([self.flow_log_prob(Z) for Z in self._images(Y)])
        return special.logsumexp(lps, axis=0) + logj

    def sample(self, n):
        Y, lp = self.flow_sample(n)
        if self.mode == "discard":
            X, logj = self.from_flow(Y)
            return X, lp + logj
        w = 2 * self.pad
        out = (np.abs(Y[:, :2]) > 1 + w).any(axis=1)
        Yf = Y.copy()
        Yf[:, :2] = np.mod(Yf[:, :2] + 1, 2) - 1
        X, logj = self.from_flow(Yf)
        X[out] = np.nan
        return X, self.log_q(np.where(out[:, None], 0.0, X))


class Circular(Method):
    """Angles as points on the circle: ``CircularNeuralSplineFlow``."""

    def make_flow(self, dims):
        return CircularNeuralSplineFlow(
            dims,
            **FLOW,
            circular_features=[0, 1],
            num_bins=8,
            tail_bound=5.0,
            batch_norm_between_layers=True,
            linear_transform="lu",
            mask_seed=int(self.rng.integers(2**31)),
            **self.flow_kwargs,
        )

    def to_flow(self, X):
        a = TWO_PI * X[:, :2] / PERIODS - math.pi
        logj = np.log(TWO_PI / PERIODS).sum() + self.real_logj()
        return np.concatenate([a, self.real_fwd(X)], axis=1), logj

    def from_flow(self, Y):
        a = wrap_angle(Y[:, :2] + math.pi, TWO_PI)
        X = np.concatenate(
            [a * PERIODS / TWO_PI, self.real_inv(Y[:, 2:])], axis=1
        )
        logj = np.log(TWO_PI / PERIODS).sum() + self.real_logj()
        return X, np.full(len(Y), logj)


METHODS = {
    "realnvp_box": lambda rng: Box("realnvp", rng),
    "realnvp_chi": lambda rng: Chi("realnvp", rng),
    "realnvp_ghost": lambda rng: Ghost("realnvp", rng),
    "nsf_box": lambda rng: Box("nsf", rng),
    "nsf_chi": lambda rng: Chi("nsf", rng),
    "nsf_ghost": lambda rng: Ghost("nsf", rng),
    "realnvp_ghostpou": lambda rng: Ghost("realnvp", rng, mode="fold", pou=True),
    "nsf_ghostpou": lambda rng: Ghost("nsf", rng, mode="fold", pou=True),
    "circ_spline": lambda rng: Circular(
        None, rng, real_transform="spline", circular_shift="angle"
    ),
    "circ_affine": lambda rng: Circular(
        None, rng, real_transform="affine", circular_shift="angle"
    ),
    "circ_affine_realshift": lambda rng: Circular(
        None, rng, real_transform="affine", circular_shift="real"
    ),
    "circ_spline_realshift": lambda rng: Circular(
        None, rng, real_transform="spline", circular_shift="real"
    ),
    "circ_spline_noshift": lambda rng: Circular(
        None, rng, real_transform="spline", circular_shift=None
    ),
}


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def train_flow(flow, Yt, Yv, wt=None, wv=None):
    dtype = torch.get_default_dtype()
    Yt = torch.as_tensor(Yt, dtype=dtype)
    Yv = torch.as_tensor(Yv, dtype=dtype)
    wt = torch.ones(len(Yt)) if wt is None else torch.as_tensor(wt, dtype=dtype)
    wv = torch.ones(len(Yv)) if wv is None else torch.as_tensor(wv, dtype=dtype)
    opt = torch.optim.AdamW(flow.parameters(), lr=TRAIN["lr"])
    best, best_state, wait = np.inf, None, 0
    history = []
    for epoch in range(TRAIN["max_epochs"]):
        flow.train()
        perm = torch.randperm(len(Yt))
        for idx in torch.split(perm, TRAIN["batch_size"]):
            opt.zero_grad()
            w = wt[idx]
            loss = -(w * flow.log_prob(Yt[idx])).sum() / w.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), TRAIN["clip"])
            opt.step()
        flow.eval()
        with torch.no_grad():
            val = (-(wv * flow.log_prob(Yv)).sum() / wv.sum()).item()
        history.append(val)
        if val < best:
            best, wait = val, 0
            best_state = {k: v.clone() for k, v in flow.state_dict().items()}
        else:
            wait += 1
            if wait > TRAIN["patience"]:
                break
    flow.load_state_dict(best_state)
    flow.eval()
    return history


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def evaluate(method, target, rng, n_test=20000, n_draw=50000):
    # normalisation of q on the domain, from flow draws
    Xs, lq_s = method.sample(n_draw)
    inside = np.isfinite(Xs).all(axis=1)
    frac_in = inside.mean()

    # forward KL with q normalised on the domain
    Xt = target.sample(n_test, rng)
    lq_t = method.log_q(Xt) - np.log(frac_in)
    lp_t = target.log_prob(Xt)
    kl = np.mean(lp_t - lq_t)
    kl_se = np.std(lp_t - lq_t) / math.sqrt(n_test)

    # importance weights over flow draws, zero off the domain
    logw = np.full(n_draw, -np.inf)
    logw[inside] = target.log_prob(Xs[inside]) - lq_s[inside]
    w = np.exp(logw - logw[inside].max())
    ess = w.sum() ** 2 / (w**2).sum() / n_draw
    # E_q[p / q] = 1 when the reported density is the sampling density
    zhat = np.exp(special.logsumexp(logw) - np.log(n_draw))
    # rejection-sampling efficiency against the batch maximum
    rej = w.mean()
    return dict(
        kl=float(kl),
        kl_se=float(kl_se),
        ess=float(ess),
        inside=float(frac_in),
        rejection=float(rej),
        zhat=float(zhat),
    )


def main(outdir, case, method_name, seed, n_train=5000):
    torch.set_num_threads(1)
    seed = int(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    target = TARGETS[case]()
    X = target.sample(int(n_train), rng)
    t0 = time.time()
    method = METHODS[method_name](rng)
    method.train(X)
    t_train = time.time() - t0
    rows = []
    if isinstance(method, Ghost):
        modes = ["fold"] if method.pou else ["discard", "fold"]
    else:
        modes = [None]
    for mode in modes:
        if mode is not None:
            method.mode = mode
        res = evaluate(method, target, np.random.default_rng(seed + 10_000))
        name = method_name + (f"_{mode}" if mode else "")
        rows.append(
            dict(
                case=case,
                method=name,
                seed=seed,
                n_train=int(n_train),
                n_train_aug=int(method.n_train_aug),
                epochs=len(method.history),
                train_time=t_train,
                **res,
            )
        )
        print(json.dumps(rows[-1]))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{case}_{method_name}_{seed}.json", "w") as f:
        json.dump(rows, f)
    torch.save(method.flow.state_dict(), out / f"{case}_{method_name}_{seed}.pt")


if __name__ == "__main__":
    main(*sys.argv[1:])
