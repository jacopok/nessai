# -*- coding: utf-8 -*-
"""
Cost-based decision model for when to retrain (or reset) the flow.

Summary
-------
The proposal pool is paid for up front (every pool point is drawn from the
flow and has its likelihood evaluated when the pool is populated), so the
only sensible moments to retrain are when the pool is empty: retraining
earlier throws away points that have already been paid for. At each such
moment we choose between

* **continue**: populate another pool with the current flow, or
* **retrain**: pay the training cost and populate from a fresh flow
  (optionally resetting the flow first).

Between trainings the nested-sampling acceptance decays exponentially,
``a(s) = A exp(-k s)``, where ``s`` is the number of iterations since the
flow was trained. Empirically ``k ~= 1 / nlive``: the flow does not change so
the acceptance simply tracks the shrinking prior volume. The cost of one
iteration is ``c / a(s)``, where ``c`` is the cost of one pool point
(population + likelihood evaluation).

This is the classic renewal (machine-replacement) problem. The minimal
long-run cost per iteration ``g*`` over a training cycle satisfies

.. math::

    T = \\int_0^\\infty \\left(g^* - \\frac{c}{A'} e^{k s}\\right)^+ ds,

i.e. the training cost is recovered by the savings accumulated while the
new flow runs below the break-even rate. With ``x = g* A' / c`` this becomes
``x ln x - x + 1 = T k A' / c`` which is solved by bisection. The optimal
policy is then *retrain as soon as continuing costs more per iteration than
g\\**: continuing for one more pool of ``P`` points yielding ``n`` iterations
costs ``c P`` while the same ``n`` iterations cost ``g* n`` in the optimal
long-run regime (the remaining future is merely shifted by ``n``
iterations, so everything else cancels). This removes the need to pick a
look-ahead horizon.

Uncertainties
-------------
* The acceptance of the current flow is described by a Gaussian posterior on
  ``(log A, k)`` from a weighted least-squares fit to binned log-acceptance
  with a conjugate Gaussian prior (from previous episodes).
* The fresh acceptance ``A'`` after retraining is predicted from the last
  few episodes with a Gaussian in ``log A'``.
* Costs are convex in ``1/a`` so these enter through log-normal moments,
  e.g. ``E[1 / A'] = exp(-mu + sigma^2 / 2)``.
* Costs ``c`` and ``T`` enter linearly, so only their means matter for the
  expected cost; their scatter is tracked for diagnostics.

Determinism
-----------
Costs are computed as deterministic counts (likelihood evaluations, pool
points, training epochs x training samples) multiplied by unit costs. The
unit costs can be provided by the user (deterministic run), or measured
during the run (non-deterministic). The measured values are always tracked,
compared to the provided ones and can be saved for future runs.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

COST_KEYS = ("likelihood", "population", "training")

#: Preset unit costs (seconds per likelihood evaluation, per pool point and
#: per training sample-epoch). ``"gw"`` is an order-of-magnitude estimate for
#: compact-binary analyses with bilby on a CPU (relative-binning or ROQ
#: likelihood, ~15 parameters with reparameterisations). The decision is only
#: sensitive to the ratios of the costs to within a factor of a few.
RETRAIN_COST_PRESETS = {
    "gw": dict(likelihood=2e-3, population=5e-4, training=5e-5),
}
"""Unit costs: seconds per likelihood evaluation, seconds per pool point
(excluding the likelihood) and seconds per training sample per epoch."""


def solve_renewal(y: float) -> float:
    """Solve ``x ln x - x + 1 = y`` for ``x >= 1``.

    ``y = T k A / c`` is the dimensionless training cost. The solution
    ``x = g* A / c`` is the ratio between the optimal long-run cost per
    iteration and the cost per iteration of a freshly trained flow and
    ``ln(x) / k`` is the optimal number of iterations between trainings.
    """
    if not np.isfinite(y):
        return np.inf
    if y <= 0:
        return 1.0

    def f(x):
        return x * np.log(x) - x + 1 - y

    lo, hi = 1.0, 2.0
    while f(hi) < 0:
        lo, hi = hi, 2 * hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10 * hi:
            break
    return 0.5 * (lo + hi)


@dataclass
class Episode:
    """Data for the period between two trainings of the flow."""

    start: int
    reset: bool
    train_cost_units: float = np.nan
    """Epochs x training samples of the training that started the episode"""
    counts: list = field(default_factory=list)
    """Number of pool draws for each iteration"""
    log_a0: float = np.nan
    """Data-only estimate of log-acceptance at the start of the episode"""
    log_a0_var: float = np.nan
    slope: float = np.nan
    """Data-only estimate of d log(acceptance) / d iteration"""
    slope_var: float = np.nan
    predicted: float = np.nan
    """Log-acceptance predicted for this episode before it started"""


def _bin_counts(counts, bin_size):
    """Bin draws per iteration into (mid iteration, log acc, variance)."""
    counts = np.asarray(counts, dtype=float)
    n = len(counts)
    if n == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    edges = list(range(0, n, bin_size))
    # Merge a short final bin into the previous one
    if len(edges) > 1 and n - edges[-1] < bin_size // 2:
        edges = edges[:-1]
    edges.append(n)
    x, y, v = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = b - a
        total = counts[a:b].sum()
        acc = m / total
        x.append(0.5 * (a + b - 1))
        y.append(np.log(acc))
        # Draws are geometric, var[log(m / sum)] ~= (1 - a) / m
        v.append(max(1.0 - acc, 0.05) / m)
    return np.array(x), np.array(y), np.array(v)


def fit_log_acceptance(counts, bin_size, prior_mean, prior_cov):
    """Bayesian weighted least squares for ``log a(s) = alpha + beta s``.

    Parameters
    ----------
    counts : array_like
        Draws needed for each iteration since the last training.
    bin_size : int
        Number of iterations per bin.
    prior_mean : array_like
        Prior mean for (alpha, beta).
    prior_cov : array_like
        Prior covariance for (alpha, beta). Use ``np.inf`` on the diagonal
        for a flat prior.

    Returns
    -------
    mean : numpy.ndarray
        Posterior mean of (alpha, beta).
    cov : numpy.ndarray
        Posterior covariance of (alpha, beta).
    """
    prior_mean = np.asarray(prior_mean, dtype=float)
    prior_cov = np.asarray(prior_cov, dtype=float)
    prec = np.zeros((2, 2))
    h = np.zeros(2)
    finite = np.isfinite(np.diag(prior_cov))
    if finite.all():
        p0 = np.linalg.inv(prior_cov)
    else:
        # Independent priors when some components are flat
        p0 = np.diag(
            np.where(finite, 1.0 / np.where(finite, np.diag(prior_cov), 1), 0)
        )
    prec += p0
    h += p0 @ prior_mean
    x, y, v = _bin_counts(counts, bin_size)
    if len(x):
        X = np.stack([np.ones_like(x), x], axis=1)
        w = 1.0 / v
        prec += X.T @ (w[:, None] * X)
        h += X.T @ (w * y)
    cov = np.linalg.inv(prec)
    return cov @ h, cov


class RetrainCostModel:
    """Unit costs used by the retraining decision.

    Parameters
    ----------
    costs : dict, str or None
        Unit costs (seconds per likelihood evaluation, per pool point and per
        training sample-epoch) with keys :code:`likelihood`,
        :code:`population` and :code:`training`. Can also be the name of a
        preset in :py:data:`RETRAIN_COST_PRESETS` or the path to a JSON file
        written by a previous run or by
        :py:func:`nessai.samplers.retrain_benchmark.measure_retrain_costs`.
        Missing values are measured
        during the run, which makes the run non-deterministic.
    warn_ratio : float
        Warn if a measured unit cost differs from the provided one by more
        than this factor.
    """

    def __init__(self, costs=None, warn_ratio=2.0):
        if isinstance(costs, str) and costs in RETRAIN_COST_PRESETS:
            logger.info("Retrain decision using '%s' cost preset", costs)
            costs = RETRAIN_COST_PRESETS[costs]
        elif isinstance(costs, (str, os.PathLike)):
            with open(costs, "r") as f:
                costs = json.load(f)
            costs = costs.get("unit_costs", costs)
        costs = dict(costs or {})
        unknown = set(costs) - set(COST_KEYS)
        if unknown:
            raise ValueError(f"Unknown retrain cost keys: {unknown}")
        self.provided = {k: costs.get(k) for k in COST_KEYS}
        self.warn_ratio = warn_ratio
        self.totals = {k: 0.0 for k in COST_KEYS}
        self.units = {k: 0.0 for k in COST_KEYS}
        self.samples = {k: [] for k in COST_KEYS}
        self._warned = set()
        missing = [k for k in COST_KEYS if self.provided[k] is None]
        if missing:
            logger.info(
                "Retrain decision will use measured costs for %s. The "
                "training schedule will depend on the hardware and is not "
                "deterministic.",
                missing,
            )

    @property
    def deterministic(self):
        return all(v is not None for v in self.provided.values())

    def record(self, key, seconds, units):
        """Record a measurement of ``seconds`` spent on ``units`` units."""
        if units <= 0 or not np.isfinite(seconds) or seconds < 0:
            return
        self.totals[key] += seconds
        self.units[key] += units
        self.samples[key].append(seconds / units)
        self._check(key)

    def measured(self, key):
        if self.units[key] > 0:
            return self.totals[key] / self.units[key]
        return None

    def __getitem__(self, key):
        """Unit cost used in the decision."""
        if self.provided[key] is not None:
            return self.provided[key]
        return self.measured(key)

    def _check(self, key):
        p = self.provided[key]
        m = self.measured(key)
        if p is None or m is None or key in self._warned:
            return
        if len(self.samples[key]) < 3:
            return
        ratio = m / p if p > 0 else np.inf
        if ratio > self.warn_ratio or ratio < 1 / self.warn_ratio:
            self._warned.add(key)
            logger.warning(
                "Measured %s cost (%.3g s) differs from the provided value "
                "(%.3g s) by a factor of %.2f. The retraining schedule may be "
                "suboptimal; consider using the updated estimates saved at "
                "the end of the run.",
                key,
                m,
                p,
                ratio,
            )

    def summary(self):
        """Measured unit costs, suitable for a future run."""
        out = {}
        for k in COST_KEYS:
            s = np.array(self.samples[k])
            out[k] = dict(
                provided=self.provided[k],
                measured=self.measured(k),
                std=float(np.std(s)) if len(s) > 1 else None,
                n=len(s),
            )
        return out

    def save(self, filename):
        """Save the measured unit costs as JSON."""
        unit_costs = {
            k: self.measured(k) if self.measured(k) is not None else v
            for k, v in self.provided.items()
        }
        with open(filename, "w") as f:
            json.dump(
                dict(unit_costs=unit_costs, details=self.summary()),
                f,
                indent=2,
            )


class RetrainDecision:
    """Decide whether to retrain the flow when the proposal pool is empty.

    Parameters
    ----------
    nlive : int
        Number of live points.
    costs : dict, str or None
        See :py:class:`RetrainCostModel`.
    history_length : int
        Number of previous episodes used to predict the acceptance of a
        freshly trained flow and the training cost.
    bin_size : int, optional
        Iterations per bin for the acceptance fit. Defaults to
        ``max(10, nlive // 20)``.
    allow_reset : bool
        If true, also decide whether to reset the flow before training by
        comparing the long-run cost of warm-started and reset training.
    horizon : bool
        If true, account for the finite number of remaining iterations.
    plan_pool : bool
        If true, limit the size of the next pool so that it is used up
        around the predicted optimal retraining time. Without this, the flow
        cannot be retrained more often than once per pool.
    """

    # Prior on the slope: acceptance tracks the prior volume, k = 1 / nlive
    slope_prior_width = 0.3
    # Prior scatter of log-acceptance of a fresh flow when there is no history
    log_a0_prior_sd = 1.0
    # Floor on the predicted scatter of a fresh flow
    min_log_a0_sd = 0.05
    # Prior on the scatter of a fresh flow and its weight in pseudo-episodes.
    # The scatter is calibrated using the errors of previous predictions.
    fresh_prior_sd = 0.3
    fresh_prior_weight = 2.0
    # Reset model: log A_reset = log A_warm + delta
    reset_prior_sd = 1.0
    """Prior width on delta"""
    reset_sd = 0.3
    """Scatter of a single training outcome"""
    reset_drift = 0.05
    """Growth of the uncertainty on a reference level per nlive iterations"""

    def __init__(
        self,
        nlive,
        costs=None,
        history_length=5,
        bin_size=None,
        allow_reset=False,
        horizon=True,
        plan_pool=True,
    ):
        self.nlive = nlive
        self.cost = RetrainCostModel(costs)
        self.history_length = history_length
        self.bin_size = bin_size or max(10, nlive // 20)
        self.allow_reset = allow_reset
        self.horizon = horizon
        self.plan_pool = plan_pool
        self.next_poolsize = None
        self.episodes = []
        self.log = []
        self.reset_next = False
        self._last = None

    # ------------------------------------------------------------ recording
    @property
    def current(self) -> Optional[Episode]:
        return self.episodes[-1] if self.episodes else None

    def start_episode(self, iteration, reset, epochs, n_train):
        """Record a training of the flow."""
        if self.current is not None:
            self._finish_episode(self.current)
        units = epochs * n_train if epochs is not None else np.nan
        warm = self._warm_level()
        self.episodes.append(
            Episode(
                start=iteration,
                reset=reset,
                train_cost_units=units,
                predicted=warm[0] if (warm and not reset) else np.nan,
            )
        )

    def record_training_time(self, seconds, epochs, n_train):
        if epochs is not None:
            self.cost.record("training", seconds, epochs * n_train)

    def record_iteration(self, count):
        if self.current is not None:
            self.current.counts.append(count)

    def record_populations(self, seconds, n_points, like_seconds, n_like):
        """Record the time spent between trainings.

        ``seconds`` should include everything except training, it is
        attributed to pool points after subtracting the likelihood time.
        """
        self.cost.record("likelihood", like_seconds, n_like)
        self.cost.record("population", seconds - like_seconds, n_points)
        self._like_per_point = n_like / n_points if n_points else 1.0

    def _finish_episode(self, ep: Episode):
        if len(ep.counts) < self.bin_size:
            return
        mean, cov = fit_log_acceptance(
            ep.counts,
            self.bin_size,
            prior_mean=[0.0, -1.0 / self.nlive],
            prior_cov=np.diag(
                [np.inf, (self.slope_prior_width / self.nlive) ** 2]
            ),
        )
        ep.log_a0, ep.slope = mean
        ep.log_a0_var, ep.slope_var = np.diag(cov)

    # ----------------------------------------------------------- estimates
    def _finished(self, episodes):
        return [e for e in episodes if np.isfinite(e.log_a0)]

    def _lineage(self):
        """Finished episodes since the last reset (or the first training)."""
        start = 0
        for i, e in enumerate(self.episodes):
            if e.reset:
                start = i
        return self._finished(self.episodes[start:-1])

    def _warm_level(self):
        """Mean and predictive variance of log A for a warm retraining.

        The mean is the average over the recent episodes of the current
        lineage. The variance is calibrated on the errors of the previous
        predictions made in the same way, combined with a weak prior.
        """
        eps = self._lineage() or self._finished(self.episodes[:-1])
        eps = eps[-self.history_length :]
        if not eps:
            return None
        mu = float(np.mean([e.log_a0 for e in eps]))
        errors = np.array(
            [
                e.log_a0 - e.predicted
                for e in self._finished(self.episodes[:-1])
                if np.isfinite(e.predicted)
            ]
        )
        w = self.fresh_prior_weight
        var = (w * self.fresh_prior_sd**2 + np.sum(errors**2)) / (
            w + len(errors)
        )
        return mu, max(var, self.min_log_a0_sd**2)

    def reset_gain(self):
        """Gaussian posterior on delta = log A_reset - log A_warm.

        Two sources of evidence are combined:

        * the degradation of the current lineage of warm-started flows
          relative to its first (from scratch) training, whose relevance
          decreases as the run progresses;
        * the outcome of previous resets relative to the preceding flow.
        """
        warm = self._warm_level()
        if warm is None:
            return None
        values, variances = [], []
        lineage = self._lineage()
        if lineage and lineage[0].reset:
            first = lineage[0]
            now = self.current.start + len(self.current.counts)
            dt = (now - first.start) / self.nlive
            values.append(first.log_a0 - warm[0])
            variances.append(self.reset_sd**2 + (self.reset_drift * dt) ** 2)
        finished = self._finished(self.episodes[:-1])
        for prev, e in zip(finished[:-1], finished[1:]):
            if e.reset:
                values.append(e.log_a0 - prev.log_a0)
                variances.append(2 * self.reset_sd**2)
        prec = 1 / self.reset_prior_sd**2
        h = 0.0
        for v, var in zip(values, variances):
            prec += 1 / var
            h += v / var
        return h / prec, 1 / prec

    def fresh_acceptance(self, reset=False):
        """Predictive mean and variance of log-acceptance of a fresh flow."""
        warm = self._warm_level()
        if warm is None or not reset:
            return warm
        delta, delta_var = self.reset_gain()
        return warm[0] + delta, warm[1] + delta_var

    def slope(self):
        """Mean and variance of the decay slope from recent episodes."""
        eps = self._finished(self.episodes[:-1])[-self.history_length :]
        prior_mean = -1.0 / self.nlive
        prior_var = (self.slope_prior_width / self.nlive) ** 2
        if not eps:
            return prior_mean, prior_var
        s = np.array([e.slope for e in eps])
        v = np.array([e.slope_var for e in eps])
        w = 1 / v
        mean = np.sum(w * s) / np.sum(w)
        scatter = s.var(ddof=1) if len(s) > 1 else 0.0
        return mean, max(1 / np.sum(w), scatter / len(s)) + scatter

    def training_units(self, reset=False):
        """Expected training cost units (epochs x samples)."""
        eps = [
            e
            for e in self.episodes
            if e.reset == reset and np.isfinite(e.train_cost_units)
        ]
        if reset and len(eps) == 0:
            eps = self.episodes[:1]
        eps = eps[-self.history_length :]
        if not eps:
            return None
        return float(np.mean([e.train_cost_units for e in eps]))

    def pool_point_cost(self):
        lk = self.cost["likelihood"]
        pop = self.cost["population"]
        if lk is None or pop is None:
            return None
        return pop + lk * getattr(self, "_like_per_point", 1.0)

    def long_run_rate(self, reset=False):
        """Optimal long-run cost per iteration, g*, and the cycle length."""
        c = self.pool_point_cost()
        units = self.training_units(reset=reset)
        fresh = self.fresh_acceptance(reset=reset)
        t_unit = self.cost["training"]
        if None in (c, units, fresh, t_unit):
            return None
        T = units * t_unit
        mu, var = fresh
        # E[1 / A'] for log-normal A'
        a_eff = np.exp(mu - 0.5 * var)
        k_mean, k_var = self.slope()
        k = -k_mean
        if k <= 0:
            return None
        # E[1 / k] to second order
        k_eff = k / (1 + k_var / k**2)
        y = T * k_eff * a_eff / c
        x = solve_renewal(y)
        return dict(
            g=x * c / a_eff,
            tau=np.log(x) / k_eff,
            T=T,
            c=c,
            a=a_eff,
            k=k_eff,
            y=y,
        )

    def current_fit(self):
        """Posterior of (log A, slope) for the current flow."""
        ep = self.current
        fresh = self._warm_level()
        if fresh is None:
            fresh = (np.log(0.1), self.log_a0_prior_sd**2)
        k_mean, k_var = self.slope()
        return fit_log_acceptance(
            ep.counts,
            self.bin_size,
            prior_mean=[fresh[0], k_mean],
            prior_cov=np.diag([fresh[1], k_var]),
        )

    def expected_block(self, mean, cov, s, pool_size, quad=7):
        """Expected number of iterations the next pool of the current flow
        will provide and the mean log-acceptance at its start."""
        nodes, weights = np.polynomial.hermite_e.hermegauss(quad)
        weights = weights / weights.sum()
        x = np.array([1.0, s])
        m = x @ mean
        sd = np.sqrt(max(x @ cov @ x, 0.0))
        k = -mean[1]
        if k <= 0:
            k = 1.0 / self.nlive
        log_a = m + sd * nodes
        a = np.minimum(np.exp(log_a), 1.0)
        n = np.log1p(pool_size * k * a) / k
        return float(np.sum(weights * n)), m, sd

    # ------------------------------------------------------------ decision
    def decide(self, iteration, pool_size, remaining=None):
        """Decide whether to retrain before populating the next pool.

        Parameters
        ----------
        iteration : int
            Current iteration.
        pool_size : int
            Size of the next pool if it is drawn from the current flow.
        remaining : float, optional
            Estimate of the remaining number of iterations.

        Returns
        -------
        bool
            True if the flow should be retrained.
        """
        ep = self.current
        self.reset_next = False
        if ep is None:
            return True
        s = len(ep.counts)
        c = self.pool_point_cost()
        warm = self.long_run_rate(reset=False)
        if c is None or warm is None:
            # Not enough information yet, retrain to gather it
            self._log(iteration, s, True, reason="no information")
            return True
        best = warm
        reset = False
        if self.allow_reset:
            rst = self.long_run_rate(reset=True)
            if rst is not None and rst["g"] < warm["g"]:
                best = rst
                reset = True
        mean, cov = self.current_fit()
        block = pool_size
        if self.plan_pool:
            block = self._planned_pool(mean, cov, s, best, c, pool_size)
        n, log_a, sd = self.expected_block(mean, cov, s, block)
        cost_continue = c * block
        cost_optimal = best["g"] * n
        retrain = bool(cost_continue > cost_optimal)
        reason = "rate"
        if (
            self.horizon
            and remaining is not None
            and np.isfinite(remaining)
            and remaining < best["tau"]
        ):
            # Few iterations left: compare costs to the end with no further
            # training.
            k = best["k"]
            h = max(remaining, 1.0)
            k_cur = -mean[1] if mean[1] < 0 else k
            # E[exp(-log a(s'))] integrated over the remaining iterations
            cont = (
                c
                * np.exp(-log_a + 0.5 * sd**2)
                * np.expm1(k_cur * h)
                / k_cur
            )
            new = best["T"] + c / best["a"] * np.expm1(k * h) / k
            retrain = bool(new < cont)
            reason = "horizon"
        self.reset_next = retrain and reset
        self.next_poolsize = None
        if self.plan_pool:
            if retrain:
                p = (best["g"] / c - 1 / best["a"]) / best["k"]
                p_min = self._min_block() / best["a"]
                self.next_poolsize = int(max(p, p_min))
            elif block < pool_size:
                self.next_poolsize = int(block)
        self._log(
            iteration,
            s,
            retrain,
            reason=reason,
            reset=self.reset_next,
            log_a=log_a,
            log_a_sd=sd,
            n_block=n,
            rate_continue=cost_continue / max(n, 1e-12),
            g=best["g"],
            tau=best["tau"],
            y=best["y"],
            remaining=remaining,
            poolsize=self.next_poolsize,
        )
        return retrain

    def _min_block(self):
        """Minimum number of iterations a pool should provide."""
        return max(self.bin_size, self.nlive // 10)

    def _planned_pool(self, mean, cov, s, best, c, pool_size):
        """Pool size needed to reach the point where continuing costs more
        per iteration than the optimal long-run rate."""
        x = np.array([1.0, s])
        log_a = x @ mean
        var = max(x @ cov @ x, 0.0)
        k = -mean[1] if mean[1] < 0 else 1.0 / self.nlive
        inv_a = np.exp(-log_a + 0.5 * var)
        p = (best["g"] / c - inv_a) / k
        p_min = self._min_block() * inv_a
        return int(min(pool_size, max(p, p_min)))

    def _log(self, iteration, s, retrain, **kwargs):
        d = dict(iteration=iteration, since_training=s, retrain=retrain)
        d.update(kwargs)
        self.log.append(d)
        if "g" in kwargs:
            logger.debug(
                "Retrain decision at it %d (s=%d): acc=%.3g+/-%.2g(log) "
                "rate=%.3g g*=%.3g tau*=%.0f -> %s%s",
                iteration,
                s,
                np.exp(kwargs["log_a"]),
                kwargs["log_a_sd"],
                kwargs["rate_continue"],
                kwargs["g"],
                kwargs["tau"],
                "retrain" if retrain else "continue",
                " (reset)" if kwargs.get("reset") else "",
            )

    def summary(self):
        """Summary of the fitted quantities at the end of a run."""
        out = dict(unit_costs=self.cost.summary())
        warm = self.long_run_rate(reset=False)
        if warm is not None:
            out["long_run"] = {k: float(v) for k, v in warm.items()}
        out["n_trainings"] = len(self.episodes)
        out["n_resets"] = int(sum(e.reset for e in self.episodes[1:]))
        return out
