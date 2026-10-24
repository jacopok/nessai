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

Lineage
-------
Warm-started trainings accumulate: a flow trained ``N`` times since its
last reset is better than one trained once, and its costs per iteration
fall as ``N^-b``, with one exponent for the likelihood part of the cost and
one for the population part (``lineage_exponent``). The renewal model above
only sees what one training buys before the flow goes stale. One more
training also lowers the cost of every later iteration by a fraction
``b / N(t)``, which saves

.. math::

    G = b \\, C \\sum_{j=0}^{H} \\frac{1}{N + j / \\tau}
      \\approx b \\, C \\, \\tau \\ln\\left(1 + \\frac{H}{N \\tau}\\right),

where ``C`` is the sampling cost per iteration, ``H`` the remaining number
of iterations and ``tau`` the interval between trainings (later trainings
make each one worth less). The training cost in the renewal problem is
replaced by ``T - G``, and ``tau`` is solved for self-consistently. When
``G >= T`` the flow is retrained whenever the pool is empty. ``C`` is
extrapolated with the trend of the cost per iteration of recent fresh flows
(``cost_trend``), capped at a growth of ``exp(max_cost_growth)`` over the
horizon. With ``allow_reset``, the reset branch uses the same gain (that of
the current lineage); what a reset costs in lineage is left to the
measured reset gain.

Reset on a drop
---------------
When the cost per iteration of a freshly trained flow (likelihood and
population, at the start of its episode) rises by more than
``reset_on_drop`` (in log), and by more than twice its uncertainty, above
the median of the current lineage in the last ``reset_window * nlive``
iterations, the warm-started lineage is
assumed broken and the next training starts from a reset flow. With
``reset_rollback`` the previous flow is kept, and when the reset flow's
first pool costs more per iteration than the flow it replaced, the reset is
discarded: the previous flow is restored and trained warm, and its lineage
continues. The same check applies to resets chosen by ``allow_reset``.

Uncertainties
-------------
* The acceptance of the current flow is described by a Gaussian posterior on
  ``(log A, k)`` from a weighted least-squares fit to binned log-acceptance
  with a conjugate Gaussian prior (from previous episodes).
* The fresh acceptance ``A'`` after retraining is predicted with a Gaussian
  in ``log A'`` from the acceptance measured at the start of the recent
  episodes, including the one that is ending (see ``fresh_predictor``). Its
  scatter is calibrated on the errors of the recent predictions.
* Costs are convex in ``1/a`` so these enter through log-normal moments,
  e.g. ``E[1 / A'] = exp(-mu + sigma^2 / 2)``.
* Costs ``c`` and ``T`` enter linearly, so only their means matter for the
  expected cost; their scatter is tracked for diagnostics.

Determinism
-----------
Costs are computed as deterministic counts (likelihood evaluations, pool
points or flow draws, training epochs x training samples) multiplied by unit
costs. The population cost of a pool point depends on how many flow draws it
takes, which changes with the flow and the state of the run. It is counted
per draw when a cost per draw (``draw``) is provided, or when the population
cost is measured rather than provided. The
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
#: Unit costs that are used when known: ``draw``, seconds per flow draw when
#: populating (rejection sampling included)
OPTIONAL_COST_KEYS = ("draw",)
ALL_COST_KEYS = COST_KEYS + OPTIONAL_COST_KEYS

#: Lineage exponents of the likelihood and population costs per iteration,
#: measured for a precessing BBH with nessai-gw (cost per new point against
#: the number of warm-started trainings, at a fixed state of the run)
DEFAULT_LINEAGE_EXPONENT = dict(likelihood=0.07, population=0.3)

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


def lineage_sum(n, tau, horizon, growth=0.0, points=256):
    """``sum_{j=0}^{H} exp(growth j) / (n + j / tau)``: the future iterations
    of a lineage of ``n`` trainings that gains one training every ``tau``
    iterations, weighted by a cost per iteration that grows at the rate
    ``growth`` (per iteration), over ``horizon`` iterations (integral
    approximation)."""
    if horizon <= 0 or n <= 0:
        return 0.0
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive and finite")
    if growth == 0.0:
        return float(tau * np.log1p(horizon / (n * tau)))
    # trapezoid in u = log(n + j / tau), where the integrand is smooth
    u = np.linspace(np.log(n), np.log(n + horizon / tau), points)
    j = tau * (np.exp(u) - n)
    f = tau * np.exp(growth * j)
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(u)))


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
    points: float = 0.0
    """Pool points populated during the episode"""
    draws: float = 0.0
    """Flow draws made to populate them (0 if unknown)"""
    attempt: bool = False
    """The episode is a reset that is compared to the flow it replaced"""
    baseline: tuple = (np.nan, np.nan)
    """Mean and variance of the log cost per iteration of that flow"""
    resolved: bool = False
    """The comparison has been made"""
    rejected: bool = False
    """The reset did worse and the previous flow was restored"""

    @property
    def draws_per_point(self):
        if self.points > 0 and self.draws > 0:
            return self.draws / self.points
        return None


def initial_level(counts, bin_size):
    """Log-acceptance over the first ``bin_size`` iterations of an episode
    and its variance (draws are geometric)."""
    m = min(len(counts), bin_size)
    if m == 0:
        return np.nan, np.nan
    a = m / float(np.sum(counts[:m]))
    return float(np.log(a)), max(1.0 - a, 0.05) / m


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
        unknown = set(costs) - set(ALL_COST_KEYS)
        if unknown:
            raise ValueError(f"Unknown retrain cost keys: {unknown}")
        self.provided = {k: costs.get(k) for k in ALL_COST_KEYS}
        self.warn_ratio = warn_ratio
        self.totals = {k: 0.0 for k in ALL_COST_KEYS}
        self.units = {k: 0.0 for k in ALL_COST_KEYS}
        self.samples = {k: [] for k in ALL_COST_KEYS}
        self._warned = set()
        missing = [k for k in COST_KEYS if self.provided[k] is None]
        if missing:
            logger.info(
                "Retrain decision will use measured costs for %s. The "
                "training schedule will depend on the hardware and is not "
                "deterministic.",
                missing,
            )

    def __setstate__(self, state):
        # objects pickled before the optional unit costs existed
        self.__dict__.update(state)
        for d, v in (
            (self.provided, None),
            (self.totals, 0.0),
            (self.units, 0.0),
        ):
            for k in ALL_COST_KEYS:
                d.setdefault(k, v)
        for k in ALL_COST_KEYS:
            self.samples.setdefault(k, [])

    @property
    def deterministic(self):
        return all(self.provided[k] is not None for k in COST_KEYS)

    @property
    def per_draw(self):
        """Whether the population cost is counted per flow draw: when a
        cost per draw is provided, or when the population cost is not
        provided (so it is measured either way)."""
        if self.provided.get("draw") is not None:
            return True
        return self.provided["population"] is None and self["draw"] is not None

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
        if self.provided.get(key) is not None:
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
        for k in ALL_COST_KEYS:
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
    fresh_predictor : {'average', 'last', 'trend'}
        How the acceptance of a freshly trained flow is predicted from the
        acceptance measured over the first ``bin_size`` iterations of the
        recent episodes (the one ending included):

        * ``'average'``: weighted average of the last ``history_length``
          episodes, halving the weight per older episode;
        * ``'last'``: the episode that is ending;
        * ``'trend'``: weighted linear fit in iteration over the last
          ``history_length`` episodes, extrapolated to the current
          iteration, with a Gaussian prior of width ``trend_slope_sd`` (per
          ``nlive`` iterations) on the slope.
    lineage_exponent : float, dict or None
        Exponents ``b`` of the fall of the likelihood and population costs
        per iteration with the number of warm-started trainings since the
        last reset, ``N^-b`` (see the module's "Lineage" section). A float
        applies to both; a dict has keys ``likelihood`` and ``population``.
        None or 0 disables the lineage term.
    lineage_horizon : float
        Cap on the remaining iterations that the lineage term counts, in
        units of ``nlive`` (the estimate of the remaining iterations is poor
        early in a run).
    reset_on_drop : float or None
        Reset the flow when the log cost per iteration of a fresh flow rises
        by more than this (and by more than twice its uncertainty) above the
        median of the current lineage over the last ``reset_window`` x
        ``nlive`` iterations. None disables it.
    reset_window : float
        Window for ``reset_on_drop``, in units of ``nlive``.
    reset_rollback : bool
        Keep the flow that a reset replaces, and restore it when the reset
        flow costs more per iteration (by more than ``reset_tolerance`` in
        log) over its first pool.
    reset_tolerance : float
        See ``reset_rollback``.
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
    # Weight of an episode relative to the one after it when predicting the
    # acceptance of a fresh flow, so the prediction follows the latest trend
    history_decay = 0.5
    fresh_predictor = "average"
    # Prior width on the slope of the 'trend' predictor, in log-acceptance
    # per nlive iterations. Without it, back-to-back short episodes leave the
    # slope unconstrained.
    trend_slope_sd = 0.5
    # Number of recent prediction errors that calibrate the scatter
    error_window = 10
    # Episodes shorter than this are too noisy to enter the estimates;
    # episodes shorter than ``bin_size`` only inform the acceptance level
    min_episode_length = 5
    # Reset model: log A_reset = log A_warm + delta
    reset_prior_sd = 1.0
    """Prior width on delta"""
    reset_sd = 0.3
    """Scatter of a single training outcome"""
    reset_drift = 0.05
    """Growth of the uncertainty on a reference level per nlive iterations"""
    # Defaults of objects pickled before these options existed: off
    lineage_exponent = None
    lineage_horizon = 50.0
    reset_on_drop = None
    reset_window = 1.0
    reset_rollback = False
    reset_tolerance = 0.0
    drop_sigma = 2.0
    """Significance (in standard deviations) a drop must also have"""
    trend_window = 2.0
    """Window (x nlive iterations) for the trend of the cost per iteration"""
    max_cost_growth = 3.0
    """Cap on the log growth of the cost per iteration over the horizon"""
    rollback_next = False
    _attempt_next = None
    _drop_anchor = -np.inf
    _horizon = None

    def __init__(
        self,
        nlive,
        costs=None,
        history_length=5,
        bin_size=None,
        allow_reset=False,
        horizon=True,
        plan_pool=True,
        fresh_predictor="average",
        lineage_exponent=DEFAULT_LINEAGE_EXPONENT,
        lineage_horizon=50.0,
        reset_on_drop=np.log(4.0),
        reset_window=1.0,
        reset_rollback=True,
        reset_tolerance=0.0,
    ):
        self.nlive = nlive
        self.cost = RetrainCostModel(costs)
        self.history_length = history_length
        self.bin_size = bin_size or max(10, nlive // 20)
        self.allow_reset = allow_reset
        self.horizon = horizon
        self.plan_pool = plan_pool
        if fresh_predictor not in ("average", "last", "trend"):
            raise ValueError(f"Unknown fresh_predictor: {fresh_predictor}")
        self.fresh_predictor = fresh_predictor
        if isinstance(lineage_exponent, dict):
            unknown = set(lineage_exponent) - {"likelihood", "population"}
            if unknown:
                raise ValueError(f"Unknown lineage exponent keys: {unknown}")
            lineage_exponent = dict(lineage_exponent)
        elif lineage_exponent is not None and lineage_exponent < 0:
            raise ValueError("lineage_exponent must be non-negative")
        self.lineage_exponent = lineage_exponent
        self.lineage_horizon = lineage_horizon
        self.reset_on_drop = reset_on_drop
        self.reset_window = reset_window
        self.reset_rollback = reset_rollback
        self.reset_tolerance = reset_tolerance
        self.rollback_next = False
        self._attempt_next = None
        self._drop_anchor = -np.inf
        self._horizon = None
        self.next_poolsize = None
        self.episodes = []
        self.log = []
        self.reset_next = False
        self._last = None

    # ------------------------------------------------------------ recording
    @property
    def current(self) -> Optional[Episode]:
        return self.episodes[-1] if self.episodes else None

    @property
    def can_reset(self):
        """Whether the decision may reset the flow."""
        return bool(self.allow_reset) or self.reset_on_drop is not None

    def start_episode(self, iteration, reset, epochs, n_train):
        """Record a training of the flow."""
        if self.current is not None:
            self._finish_episode(self.current)
        units = epochs * n_train if epochs is not None else np.nan
        warm = self._warm_level(at=iteration)
        ep = Episode(
            start=iteration,
            reset=reset,
            train_cost_units=units,
            predicted=warm[0] if (warm and not reset) else np.nan,
        )
        if reset and self._attempt_next is not None:
            ep.attempt = True
            ep.baseline = self._attempt_next
        self._attempt_next = None
        self.rollback_next = False
        self.episodes.append(ep)

    def record_training_time(self, seconds, epochs, n_train):
        if epochs is not None:
            self.cost.record("training", seconds, epochs * n_train)

    def record_iteration(self, count):
        if self.current is not None:
            self.current.counts.append(count)

    def record_populations(
        self, seconds, n_points, like_seconds, n_like, n_draws=None
    ):
        """Record the time spent between trainings.

        ``seconds`` should include everything except training, it is
        attributed to pool points (and to flow draws, if ``n_draws`` is
        given) after subtracting the likelihood time.
        """
        self.cost.record("likelihood", like_seconds, n_like)
        self.cost.record("population", seconds - like_seconds, n_points)
        if n_draws:
            self.cost.record("draw", seconds - like_seconds, n_draws)
        self._like_per_point = n_like / n_points if n_points else 1.0
        ep = self.current
        if ep is not None and n_points:
            ep.points += n_points
            ep.draws += n_draws or 0.0

    def _finish_episode(self, ep: Episode):
        # Short episodes are fitted as a single bin with the slope from its
        # prior. Skipping them would freeze the prediction for a fresh flow
        # at the level of the last long episode exactly when the acceptance
        # has dropped so far that no pool lasts a full bin.
        n = len(ep.counts)
        if n < self.min_episode_length:
            return
        mean, cov = fit_log_acceptance(
            ep.counts,
            min(self.bin_size, n),
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

    def _lineage_start(self):
        """Index of the episode that started the current lineage: the last
        reset that was kept (or the first training)."""
        start = 0
        for i, e in enumerate(self.episodes):
            if e.reset and not e.rejected:
                start = i
        return start

    def _lineage_episodes(self, include_current=True):
        eps = self.episodes if include_current else self.episodes[:-1]
        return [e for e in eps[self._lineage_start() :] if not e.rejected]

    def lineage_length(self):
        """Number of trainings of the current flow since its last reset."""
        return max(len(self._lineage_episodes()), 1)

    def _lineage(self):
        """Finished episodes since the last reset (or the first training)."""
        return self._finished(self._lineage_episodes(include_current=False))

    def _level_history(self, include_current=True):
        """``(iteration, log-acceptance, variance)`` at the start of the
        recent episodes of the current lineage.

        The running episode is included by default: the decision to retrain
        is taken when it ends, so the next flow follows it.
        """
        start = self._lineage_start()
        eps_all = self.episodes if include_current else self.episodes[:-1]
        eps = [e for e in eps_all if not e.rejected]
        lineage_eps = [
            e for i, e in enumerate(eps_all) if i >= start and not e.rejected
        ]
        rows = []
        for lineage in (lineage_eps, eps):
            rows = [
                (e.start, *initial_level(e.counts, self.bin_size))
                for e in lineage
                if len(e.counts) >= self.min_episode_length
            ]
            if rows:
                break
        return rows

    def _predict_level(self, rows, at):
        rows = rows[-self.history_length :]
        x = np.array([r[0] for r in rows], dtype=float)
        y = np.array([r[1] for r in rows])
        v = np.array([r[2] for r in rows])
        if self.fresh_predictor == "last":
            return float(y[-1])
        w = 1.0 / (v + self.fresh_prior_sd**2)
        if self.fresh_predictor == "average":
            w = w * self.history_decay ** np.arange(len(rows))[::-1]
            return float(np.sum(w * y) / np.sum(w))
        # 'trend': centred on the target so the intercept is the prediction
        X = np.stack([np.ones_like(x), (x - at) / self.nlive], axis=1)
        prec = X.T @ (w[:, None] * X)
        prec[1, 1] += 1.0 / self.trend_slope_sd**2
        return float(np.linalg.solve(prec, X.T @ (w * y))[0])

    def _warm_level(self, at=None, include_current=True):
        """Mean and predictive variance of log A for a warm retraining.

        The mean comes from the acceptance at the start of the recent
        episodes of the current lineage (see ``fresh_predictor``), predicted
        at iteration ``at`` (default: now). The variance is calibrated on
        the errors of the last ``error_window`` predictions, stored when
        each episode started, combined with a weak prior.
        """
        rows = self._level_history(include_current=include_current)
        if not rows:
            return None
        if at is None:
            ep = self.current
            at = ep.start + len(ep.counts) if ep is not None else 0
        mu = self._predict_level(rows, at)
        errors = []
        for e in self.episodes if include_current else self.episodes[:-1]:
            if len(e.counts) >= self.min_episode_length and np.isfinite(
                e.predicted
            ):
                errors.append(
                    initial_level(e.counts, self.bin_size)[0] - e.predicted
                )
        errors = np.array(errors[-self.error_window :])
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
        # Only episodes spanning at least a bin measure the slope; shorter
        # ones return its prior
        eps = [
            e
            for e in self._finished(self.episodes[:-1])
            if len(e.counts) >= self.bin_size
        ][-self.history_length :]
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

    def _draws_per_point(self, episode=None):
        """Flow draws per pool point of an episode, or of the most recent one
        that recorded them."""
        if episode is not None and episode.draws_per_point is not None:
            return episode.draws_per_point
        for e in reversed(self.episodes):
            if e.draws_per_point is not None:
                return e.draws_per_point
        return None

    def cost_components(self, episode=None):
        """Likelihood and population costs of one pool point (for the flow of
        ``episode``, default: the latest one)."""
        lk = self.cost["likelihood"]
        if lk is None:
            return None
        like = lk * getattr(self, "_like_per_point", 1.0)
        dpp = self._draws_per_point(episode)
        if self.cost.per_draw and dpp is not None:
            pop = self.cost["draw"] * dpp
        else:
            pop = self.cost["population"]
        if pop is None:
            return None
        return like, pop

    def pool_point_cost(self, episode=None):
        parts = self.cost_components(episode)
        return None if parts is None else float(sum(parts))

    def _lineage_weights(self):
        b = self.lineage_exponent
        if not b:
            return None
        if isinstance(b, dict):
            return b.get("likelihood", 0.0), b.get("population", 0.0)
        return float(b), float(b)

    def lineage_gain(self, tau, c, a, k, horizon=None):
        """Saving over the rest of the run from one more training of the
        current lineage, if the flow is then trained every ``tau``
        iterations (``c``, ``a`` and ``k``: cost per pool point, fresh
        acceptance and decay rate)."""
        weights = self._lineage_weights()
        parts = self.cost_components()
        horizon = self._horizon if horizon is None else horizon
        if weights is None or parts is None or horizon is None:
            return 0.0
        h = self.lineage_horizon * self.nlive
        if np.isfinite(horizon):
            h = min(horizon, h)
        if h <= 0:
            return 0.0
        like, pop = parts
        b = (weights[0] * like + weights[1] * pop) / (like + pop)
        t = max(tau, float(self._min_block()))
        # sampling cost per iteration, averaged over a cycle
        cost = c / a * np.expm1(k * t) / (k * t)
        growth = self.cost_trend()[0]
        if growth * h > self.max_cost_growth:
            growth = self.max_cost_growth / h
        return float(
            b * cost * lineage_sum(self.lineage_length(), t, h, growth=growth)
        )

    def cost_trend(self):
        """Mean and variance of the rate (per iteration) at which the log
        cost per iteration of a fresh flow changes, from the episodes of the
        current lineage in the last ``trend_window`` x ``nlive`` iterations,
        with a Gaussian prior of width ``trend_slope_sd`` per ``nlive``
        centred on zero."""
        ep = self.current
        now = ep.start + len(ep.counts) if ep is not None else 0
        rows = []
        for e in self._lineage_episodes():
            if e.start < now - self.trend_window * self.nlive:
                continue
            if len(e.counts) < self.min_episode_length:
                continue
            level = self._episode_log_cost(e)
            if level is not None:
                rows.append((e.start, *level))
        prior_prec = (self.nlive / self.trend_slope_sd) ** 2
        if len(rows) < 3:
            return 0.0, 1.0 / prior_prec
        x = np.array([r[0] for r in rows], dtype=float)
        y = np.array([r[1] for r in rows])
        w = 1.0 / (np.array([r[2] for r in rows]) + self.fresh_prior_sd**2)
        xc = x - np.sum(w * x) / np.sum(w)
        prec = np.sum(w * xc**2) + prior_prec
        return float(np.sum(w * xc * y) / prec), float(1.0 / prec)

    def _renewal(self, T, c, a, k):
        y = T * k * a / c
        x = solve_renewal(y)
        return dict(g=x * c / a, tau=np.log(x) / k, T=T, c=c, a=a, k=k, y=y)

    def long_run_rate(self, reset=False):
        """Optimal long-run cost per iteration, g*, and the cycle length.

        With the lineage term, ``T`` in the result is the training cost
        less the lineage gain ``G`` (``T_train`` is the cost itself), for
        the cycle length that this lower cost implies.
        """
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
        base = self._renewal(T, c, a_eff, k_eff)
        if self._lineage_weights() is None or self._horizon is None:
            return base | dict(T_train=T, G=0.0)

        def solve(tau):
            gain = self.lineage_gain(tau, c, a_eff, k_eff)
            return self._renewal(max(T - gain, 0.0), c, a_eff, k_eff), gain

        # The gain grows with the cycle length and the cycle length falls
        # with the gain, so the self-consistent cycle is unique
        lo = float(self._min_block())
        best, gain = solve(lo)
        if best["tau"] > lo:
            hi = max(base["tau"], lo)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if solve(mid)[0]["tau"] > mid:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-3 * hi:
                    break
            best, gain = solve(hi)
        return best | dict(T_train=T, G=gain)

    def current_fit(self):
        """Posterior of (log A, slope) for the current flow."""
        ep = self.current
        fresh = self._warm_level(at=ep.start, include_current=False)
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
        self.rollback_next = False
        self._attempt_next = None
        if ep is None:
            return True
        self._horizon = remaining
        s = len(ep.counts)
        if ep.attempt and not ep.resolved and s:
            if self._resolve_attempt(iteration, ep):
                return True
        if self.reset_on_drop is not None and not ep.reset:
            drop = self._drop()
            if drop is not None and (
                drop[0] > self.reset_on_drop + self.drop_sigma * drop[1]
            ):
                self.reset_next = True
                self._attempt_next = drop[2] if self.reset_rollback else None
                self._drop_anchor = iteration
                self.next_poolsize = None
                self._log(
                    iteration,
                    s,
                    True,
                    reason="drop",
                    reset=True,
                    drop=drop[0],
                    drop_sd=drop[1],
                )
                return True
        c = self.pool_point_cost()
        warm = self.long_run_rate(reset=False)
        if c is None or warm is None:
            # Not enough information yet, retrain to gather it
            self._log(iteration, s, True, reason="no information")
            return True
        best = warm
        reset = False
        gain = None
        if self.allow_reset:
            gain = self.reset_gain()
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
            cost_continue = (
                c * np.exp(-log_a + 0.5 * sd**2) * np.expm1(k_cur * h) / k_cur
            )
            cost_optimal = best["T"] + c / best["a"] * np.expm1(k * h) / k
            retrain = bool(cost_optimal < cost_continue)
            reason = "horizon"
        self.reset_next = retrain and reset
        if self.reset_next and self.reset_rollback:
            self._attempt_next = self._episode_log_cost(ep)
        self.next_poolsize = None
        if self.plan_pool:
            if retrain:
                p = (best["g"] / c - 1 / best["a"]) / best["k"]
                p_min = self._min_block() / best["a"]
                self.next_poolsize = int(max(p, p_min))
            elif block < pool_size:
                self.next_poolsize = int(block)
        fresh = self.fresh_acceptance(reset=reset)
        k_mean, k_var = self.slope()
        self._log(
            iteration,
            s,
            retrain,
            reason=reason,
            reset=self.reset_next,
            log_a=log_a,
            log_a_sd=sd,
            slope=-mean[1],
            slope_sd=np.sqrt(max(cov[1, 1], 0.0)),
            block=block,
            n_block=n,
            cost_continue=cost_continue,
            cost_optimal=cost_optimal,
            rate_continue=c * block / max(n, 1e-12),
            fresh_log_a=fresh[0],
            fresh_log_a_sd=np.sqrt(fresh[1]),
            k=-k_mean,
            k_sd=np.sqrt(k_var),
            reset_gain=gain[0] if gain else None,
            reset_gain_sd=np.sqrt(gain[1]) if gain else None,
            c=c,
            T=best["T"],
            T_train=best.get("T_train", best["T"]),
            G=best.get("G", 0.0),
            lineage=self.lineage_length(),
            g=best["g"],
            tau=best["tau"],
            y=best["y"],
            remaining=remaining,
            poolsize=self.next_poolsize,
        )
        return retrain

    def _episode_log_cost(self, ep):
        """Mean and variance of the log cost per iteration of the flow of an
        episode at its start (likelihood and population)."""
        if not ep.counts:
            return None
        c = self.pool_point_cost(ep)
        if c is None:
            return None
        log_a, var = initial_level(ep.counts, self.bin_size)
        return float(np.log(c) - log_a), float(var)

    def _drop(self):
        """Rise of the log cost per iteration of the current flow above the
        median of the current lineage in the recent window: ``(rise, sd,
        (mean, variance) of the current flow)``.

        The reference is the median of the window, and the standard
        deviation combines the measurement of the current flow with a robust
        scatter of fresh flows (from the median error of past predictions,
        so it includes the scatter between trainings), so that the first
        flows after a drop neither drag the reference down nor inflate the
        scatter.
        """
        ep = self.current
        cur = self._episode_log_cost(ep)
        if cur is None:
            return None
        since = max(
            ep.start - self.reset_window * self.nlive, self._drop_anchor
        )
        refs = [
            self._episode_log_cost(e)
            for e in self._lineage_episodes(include_current=False)
            if e.start >= since and len(e.counts) >= self.min_episode_length
        ]
        refs = [r for r in refs if r is not None]
        if not refs:
            return None
        ref = float(np.median([r[0] for r in refs]))
        var = cur[1] + self._robust_scatter() * (1.0 + 1.0 / len(refs))
        return cur[0] - ref, float(np.sqrt(var)), cur

    def _robust_scatter(self):
        """Variance of the log-acceptance of a fresh flow about its
        prediction, from the median absolute error of the last
        ``error_window`` predictions (the running episode excluded),
        combined with the prior as in :meth:`_warm_level`."""
        errors = [
            initial_level(e.counts, self.bin_size)[0] - e.predicted
            for e in self.episodes[:-1]
            if len(e.counts) >= self.min_episode_length
            and np.isfinite(e.predicted)
            and not e.rejected
        ][-self.error_window :]
        w = self.fresh_prior_weight
        var = w * self.fresh_prior_sd**2
        if errors:
            mad = 1.4826 * np.median(np.abs(errors))
            var += len(errors) * mad**2
        var /= w + len(errors)
        return max(var, self.min_log_a0_sd**2)

    def _resolve_attempt(self, iteration, ep):
        """Compare a reset flow with the flow it replaced after its first
        pool. Returns True if the reset is discarded (the previous flow is
        to be restored and retrained)."""
        ep.resolved = True
        cur = self._episode_log_cost(ep)
        if cur is None or not np.isfinite(ep.baseline[0]):
            return False
        excess = cur[0] - ep.baseline[0]
        if self.reset_rollback and excess > self.reset_tolerance:
            ep.rejected = True
            self.rollback_next = True
            self._drop_anchor = iteration
            self.next_poolsize = None
            self._log(
                iteration,
                len(ep.counts),
                True,
                reason="rollback",
                excess=excess,
                excess_sd=float(np.sqrt(cur[1] + ep.baseline[1])),
            )
            return True
        logger.info(
            "Keeping the reset flow at iteration %d: log cost per iteration "
            "%.2f against %.2f for the flow it replaced",
            iteration,
            cur[0],
            ep.baseline[0],
        )
        return False

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
        if kwargs.get("reason") == "drop":
            logger.info(
                "Resetting the flow at iteration %d: the cost per iteration "
                "of a fresh flow rose by %.2f +/- %.2f (log) in the last "
                "%.3g x nlive iterations",
                iteration,
                kwargs["drop"],
                kwargs["drop_sd"],
                self.reset_window,
            )
            return
        if kwargs.get("reason") == "rollback":
            logger.info(
                "Discarding the reset flow at iteration %d: its cost per "
                "iteration is higher by %.2f +/- %.2f (log) than the flow it "
                "replaced, which is restored and retrained",
                iteration,
                kwargs["excess"],
                kwargs["excess_sd"],
            )
            return
        if "g" not in kwargs:
            if retrain:
                logger.info(
                    "Retraining at iteration %d (%d iterations since the "
                    "last training): %s",
                    iteration,
                    s,
                    kwargs.get("reason"),
                )
            return
        logger.debug(self.format_decision(d))
        if retrain:
            logger.info(self.format_decision_short(d))

    def format_decision(self, d):
        """One-line description of a logged decision and its inputs."""
        nlive = self.nlive
        if d["reason"] == "horizon":
            compare = (
                f"cost to the end ({d['remaining']:.0f} it) continuing "
                f"{d['cost_continue']:.3g}s vs retraining "
                f"{d['cost_optimal']:.3g}s"
            )
        else:
            compare = (
                f"next pool P={d['block']:.0f} -> n={d['n_block']:.0f} it: "
                f"c*P={d['cost_continue']:.3g}s vs g**n="
                f"{d['cost_optimal']:.3g}s"
            )
        reset = ""
        if d.get("reset_gain") is not None:
            reset = (
                f"; reset gain dlogA={d['reset_gain']:.2f}"
                f"+/-{d['reset_gain_sd']:.2f}"
            )
        pool = ""
        if d.get("poolsize") is not None:
            pool = f"; next pool {d['poolsize']}"
        action = "Retraining" if d["retrain"] else "Not retraining"
        if d.get("reset"):
            action = "Resetting and retraining"
        return (
            f"{action} at iteration {d['iteration']} "
            f"(s={d['since_training']}, rule={d['reason']}): {compare}. "
            f"Current flow: acc={np.exp(d['log_a']):.3g} "
            f"(log sd {d['log_a_sd']:.2f}), "
            f"decay={d['slope'] * nlive:.2f}+/-{d['slope_sd'] * nlive:.2f}"
            f"/nlive. Fresh flow: acc={np.exp(d['fresh_log_a']):.3g} "
            f"(log sd {d['fresh_log_a_sd']:.2f}), "
            f"decay={d['k'] * nlive:.2f}+/-{d['k_sd'] * nlive:.2f}/nlive. "
            f"Costs: c={d['c']:.3g}s/point, T={d.get('T_train', d['T']):.3g}s"
            f" less lineage gain {d.get('G', 0.0):.3g}s (N={d.get('lineage')})"
            f" -> "
            f"y={d['y']:.3g}, g*={d['g']:.3g}s/it, tau*={d['tau']:.0f} it"
            f"{reset}{pool}"
        )

    def format_decision_short(self, d):
        """The retrain decision in one short line (INFO); the inputs are in
        :meth:`format_decision` (DEBUG)."""
        action = "Retraining"
        if d.get("reset"):
            action = "Resetting and retraining"
        pool = ""
        if d.get("poolsize") is not None:
            pool = f"; next pool {d['poolsize']}"
        return (
            f"{action} at it {d['iteration']} ({d['since_training']} since "
            f"last, {d['reason']} rule): acc {np.exp(d['log_a']):.3g} now vs "
            f"{np.exp(d['fresh_log_a']):.3g} fresh{pool}"
        )

    def summary(self):
        """Summary of the fitted quantities at the end of a run."""
        out = dict(unit_costs=self.cost.summary())
        warm = self.long_run_rate(reset=False)
        if warm is not None:
            out["long_run"] = {k: float(v) for k, v in warm.items()}
        out["n_trainings"] = len(self.episodes)
        out["n_resets"] = int(sum(e.reset for e in self.episodes[1:]))
        out["n_resets_discarded"] = int(sum(e.rejected for e in self.episodes))
        out["lineage_length"] = self.lineage_length()
        return out
