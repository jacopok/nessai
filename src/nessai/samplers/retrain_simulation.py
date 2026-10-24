# -*- coding: utf-8 -*-
"""
Simulated runs for testing and tuning the retrain decision.

A simulated run replaces the likelihood, the flow and the clock with a
model of what :py:class:`~nessai.samplers.retrain.RetrainDecision` sees:

* a flow trained at iteration ``t0`` has likelihood acceptance (new live
  points per pool point) ``a(t) = A exp(-k (t - t0) / nlive)`` and
  population acceptance (pool points per flow draw) ``p``;
* a training sets ``log A`` and ``log p`` to the level of the problem at
  that iteration (``acceptance_drift``, ``population_drift``), plus the
  lineage terms ``b log N`` (``N`` trainings since the last reset), plus
  Gaussian scatter;
* each iteration takes a geometric number of pool points;
* costs are counts times unit costs: likelihood evaluations, flow draws,
  pool points and training sample-epochs;
* ``breaks`` lower the acceptance of every lineage started before them
  until the flow is reset, ``shocks`` lower that of every flow.

The pool is drawn as in :py:class:`~nessai.proposal.FlowProposal` with
``update_poolsize``: ``poolsize`` scaled by the inverse of the last pool's
acceptance, up to ``max_poolsize_scale``. Points left in a pool that
cannot complete the next iteration are discarded.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from .retrain import RetrainDecision


def _piecewise(spec, t, nlive):
    """Integral of a piecewise-constant rate (per nlive iterations) from 0
    to ``t``. ``spec`` is a float or ``((end_iteration, rate), ...)``, the
    last rate holding beyond its end."""
    if np.isscalar(spec):
        return spec * t / nlive
    total, start = 0.0, 0.0
    for end, rate in spec:
        if t <= end:
            return total + rate * (t - start) / nlive
        total += rate * (end - start) / nlive
        start = end
    return total + spec[-1][1] * (t - start) / nlive


@dataclass(frozen=True)
class SimulatedProblem:
    """Parameters of a simulated run."""

    name: str
    nlive: int = 1000
    iterations: int = 10000
    log_acceptance: float = np.log(0.3)
    """Log likelihood acceptance of a flow trained once, at iteration 0"""
    acceptance_drift: object = 0.0
    """Change of that level per nlive iterations (float or segments)"""
    log_population: float = np.log(0.5)
    """Log population acceptance of a flow trained once, at iteration 0"""
    population_drift: object = 0.0
    lineage_likelihood: float = 0.0
    """Exponent of the likelihood acceptance in the lineage length"""
    lineage_population: float = 0.0
    """Exponent of the population acceptance in the lineage length"""
    decay: float = 1.0
    """Decay of the likelihood acceptance of a flow, per nlive iterations"""
    training_scatter: float = 0.1
    reset_scatter: float = 0.3
    likelihood_cost: float = 1e-3
    draw_cost: float = 1e-5
    point_cost: float = 0.0
    training_cost: float = 1e-4
    """Seconds per training sample-epoch"""
    warm_epochs: float = 25.0
    reset_epochs: float = 100.0
    breaks: tuple = ()
    """``((iteration, delta), ...)``: lineages started before ``iteration``
    lose ``delta`` in log likelihood acceptance from then on"""
    shocks: tuple = ()
    """``((iteration, delta), ...)``: every flow loses ``delta`` from then
    on"""
    poolsize: int = 1000
    max_poolsize_scale: float = 10.0

    def unit_costs(self):
        """Unit costs as given to the decision (exact)."""
        return dict(
            likelihood=self.likelihood_cost,
            population=self.point_cost
            + self.draw_cost / np.exp(self.log_population),
            training=self.training_cost,
            draw=self.draw_cost,
        )

    def with_(self, **kwargs):
        return replace(self, **kwargs)


@dataclass
class SimulatedRun:
    """Outcome of a simulated run."""

    problem: str
    policy: str
    seed: int
    likelihood: float = 0.0
    population: float = 0.0
    training: float = 0.0
    trainings: int = 0
    resets: int = 0
    discarded: int = 0
    pools: int = 0
    iterations: int = 0
    log: list = field(default_factory=list, repr=False)

    @property
    def total(self):
        return self.likelihood + self.population + self.training


class _Flow:
    """State of the simulated flow."""

    def __init__(self, problem, rng):
        self.p = problem
        self.rng = rng
        self.n = 0
        self.born = 0
        self.trained = 0
        self.log_a0 = self.log_p0 = 0.0

    def _penalty(self, t, since=None):
        """Log-acceptance lost to shocks, and to breaks for this lineage,
        up to iteration ``t`` (from iteration ``since``, if given)."""
        lo = -np.inf if since is None else since
        out = sum(d for it, d in self.p.shocks if lo < it <= t)
        out += sum(
            d for it, d in self.p.breaks if lo < it <= t and self.born < it
        )
        return out

    def train(self, t, reset):
        p = self.p
        if reset:
            self.n, self.born = 1, t
        else:
            self.n += 1
        sd = p.reset_scatter if reset else p.training_scatter
        self.log_a0 = (
            p.log_acceptance
            + _piecewise(p.acceptance_drift, t, p.nlive)
            + p.lineage_likelihood * np.log(self.n)
            - self._penalty(t)
            + self.rng.normal(0, sd)
        )
        self.log_p0 = (
            p.log_population
            + _piecewise(p.population_drift, t, p.nlive)
            + p.lineage_population * np.log(self.n)
            + self.rng.normal(0, 0.5 * sd)
        )
        self.trained = t
        mean = p.reset_epochs if reset else p.warm_epochs
        epochs = max(1, int(round(self.rng.gamma(10.0, mean / 10.0))))
        return epochs

    def acceptance(self, t):
        p = self.p
        log_a = (
            self.log_a0
            - p.decay * (t - self.trained) / p.nlive
            - self._penalty(t, since=self.trained)
        )
        return float(np.exp(min(log_a, 0.0)))

    def population_acceptance(self):
        return float(np.exp(min(self.log_p0, 0.0)))

    def state(self):
        return (self.n, self.born)

    def restore(self, state):
        self.n, self.born = state


def simulate(
    problem,
    policy="decision",
    seed=0,
    decision_kwargs=None,
    costs="exact",
    interval=None,
    remaining_scale=1.0,
    keep_log=False,
):
    """Simulate one run.

    Parameters
    ----------
    problem : SimulatedProblem
    policy : {'decision', 'every-pool', 'interval'}
        ``'every-pool'`` retrains whenever the pool is empty (nessai's
        default); ``'interval'`` also caps each pool so that it lasts about
        ``interval`` iterations; ``'decision'`` uses a
        :py:class:`~nessai.samplers.retrain.RetrainDecision`.
    seed : int
    decision_kwargs : dict, optional
        Keyword arguments for the decision.
    costs : {'exact', 'measured'} or dict
        Unit costs given to the decision.
    interval : float, optional
        Iterations per training for ``'interval'``.
    remaining_scale : float
        Factor applied to the remaining iterations reported to the decision
        (an estimate that is off by this factor).
    keep_log : bool
        Keep the decision's log in the result.
    """
    p = problem
    rng = np.random.default_rng(seed)
    flow = _Flow(p, rng)
    out = SimulatedRun(problem=p.name, policy=policy, seed=seed)
    decision = None
    if policy == "decision":
        if costs == "exact":
            costs = p.unit_costs()
        elif costs == "measured":
            costs = None
        decision = RetrainDecision(
            p.nlive, costs=costs, **(decision_kwargs or {})
        )
    elif policy not in ("every-pool", "interval"):
        raise ValueError(f"Unknown policy: {policy}")
    saved = None

    def train(t, reset):
        epochs = flow.train(t, reset)
        seconds = epochs * p.nlive * p.training_cost
        out.training += seconds
        out.trainings += 1
        out.resets += int(reset and t > 0)
        if decision is not None:
            decision.record_training_time(seconds, epochs, p.nlive)
            decision.start_episode(t, reset, epochs, p.nlive)

    t = 0
    train(0, True)
    last_acc = None
    while t < p.iterations:
        scale = 1.0 if last_acc is None else 1.0 / max(last_acc, 1e-12)
        size = int(p.poolsize * min(max(scale, 1.0), p.max_poolsize_scale))
        if decision is not None and decision.next_poolsize is not None:
            size = min(size, int(decision.next_poolsize))
        if policy == "interval" and interval is not None:
            a = flow.acceptance(t)
            size = min(size, max(int(interval / max(a, 1e-12)), 1))
        size = max(size, 1)
        # populate
        draws = size / flow.population_acceptance()
        pop_s = draws * p.draw_cost + size * p.point_cost
        like_s = size * p.likelihood_cost
        out.population += pop_s
        out.likelihood += like_s
        out.pools += 1
        # consume
        left, start = size, t
        while t < p.iterations:
            count = int(rng.geometric(flow.acceptance(t)))
            if count > left:
                break
            left -= count
            t += 1
            if decision is not None:
                decision.record_iteration(count)
        used = size - left
        last_acc = (t - start) / used if used > 0 else 1.0 / size
        if t >= p.iterations:
            break
        # pool empty: decide
        if decision is None:
            train(t, False)
            continue
        decision.record_populations(
            pop_s + like_s, size, like_s, size, n_draws=draws
        )
        remaining = (p.iterations - t) * remaining_scale
        next_size = int(
            p.poolsize
            * min(max(1.0 / max(last_acc, 1e-12), 1.0), p.max_poolsize_scale)
        )
        if not decision.decide(t, next_size, remaining=remaining):
            continue
        if decision.rollback_next and saved is not None:
            flow.restore(saved)
            saved = None
            out.discarded += 1
            train(t, False)
        elif decision.reset_next:
            if decision.reset_rollback:
                saved = flow.state()
            train(t, True)
        else:
            train(t, False)
    out.iterations = t
    if keep_log and decision is not None:
        out.log = decision.log
    return out


#: Problems spanning easy and hard regimes. Unit costs in seconds.
PROBLEMS = {
    # precessing BBH with nessai-gw (relative binning, 0.5% weight cap):
    # population-dominated, cheap training, strong lineage effect
    "lvk-bbh": SimulatedProblem(
        "lvk-bbh",
        nlive=1000,
        iterations=12000,
        log_acceptance=np.log(0.2),
        acceptance_drift=-0.45,
        log_population=np.log(0.5),
        population_drift=-0.4,
        lineage_likelihood=0.07,
        lineage_population=0.23,
        likelihood_cost=2e-3,
        draw_cost=2.4e-5,
        training_cost=1.5e-4,
    ),
    # the same with the stronger lineage effect seen late in the LVK runs
    # (385 against 80 trainings: 3x fewer likelihood calls per iteration)
    "lvk-strong-lineage": SimulatedProblem(
        "lvk-strong-lineage",
        nlive=1000,
        iterations=12000,
        log_acceptance=np.log(0.2),
        acceptance_drift=-0.45,
        log_population=np.log(0.5),
        population_drift=-0.4,
        lineage_likelihood=0.5,
        lineage_population=0.3,
        likelihood_cost=2e-3,
        draw_cost=2.4e-5,
        training_cost=1.5e-4,
    ),
    # the same with no lineage effect (the decision's exponents are wrong)
    "lvk-no-lineage": SimulatedProblem(
        "lvk-no-lineage",
        nlive=1000,
        iterations=12000,
        log_acceptance=np.log(0.2),
        acceptance_drift=-0.45,
        log_population=np.log(0.5),
        population_drift=-0.4,
        likelihood_cost=2e-3,
        draw_cost=2.4e-5,
        training_cost=1.5e-4,
    ),
    # an easy analytic problem (unimodal Gaussian): high, stable acceptance
    "gaussian": SimulatedProblem(
        "gaussian",
        nlive=1000,
        iterations=16000,
        log_acceptance=np.log(0.5),
        acceptance_drift=-0.05,
        log_population=np.log(0.9),
        likelihood_cost=1e-4,
        draw_cost=2e-6,
        training_cost=5e-5,
        warm_epochs=30,
    ),
    # expensive likelihood: training is cheap by comparison
    "expensive-likelihood": SimulatedProblem(
        "expensive-likelihood",
        nlive=1000,
        iterations=12000,
        log_acceptance=np.log(0.3),
        acceptance_drift=-0.15,
        log_population=np.log(0.7),
        lineage_likelihood=0.05,
        likelihood_cost=5e-2,
        draw_cost=1e-5,
        training_cost=5e-5,
    ),
    # expensive training (large flow, cheap likelihood): train rarely
    "expensive-training": SimulatedProblem(
        "expensive-training",
        nlive=1000,
        iterations=12000,
        log_acceptance=np.log(0.3),
        acceptance_drift=-0.15,
        log_population=np.log(0.7),
        lineage_likelihood=0.05,
        likelihood_cost=1e-5,
        draw_cost=1e-6,
        training_cost=2e-3,
        warm_epochs=40,
        reset_epochs=150,
    ),
    # a warm-started lineage breaks at 6000 iterations (e.g. it keeps a mode
    # that has died); a reset restores it
    "lineage-break": SimulatedProblem(
        "lineage-break",
        nlive=1000,
        iterations=12000,
        log_acceptance=np.log(0.2),
        acceptance_drift=-0.3,
        log_population=np.log(0.5),
        population_drift=-0.3,
        lineage_likelihood=0.07,
        lineage_population=0.2,
        likelihood_cost=2e-3,
        draw_cost=2.4e-5,
        training_cost=1.5e-4,
        breaks=((6000, 2.5),),
    ),
    # the problem gets harder at 6000 iterations for every flow; a reset
    # does not help
    "shock": SimulatedProblem(
        "shock",
        nlive=1000,
        iterations=12000,
        log_acceptance=np.log(0.2),
        acceptance_drift=-0.3,
        log_population=np.log(0.5),
        population_drift=-0.3,
        lineage_likelihood=0.07,
        lineage_population=0.2,
        likelihood_cost=2e-3,
        draw_cost=2.4e-5,
        training_cost=1.5e-4,
        shocks=((6000, 2.5),),
    ),
    # a fast early transition (prior to bulk), then a slow drift: a gradual
    # rise in cost that should not trigger resets
    "early-transition": SimulatedProblem(
        "early-transition",
        nlive=1000,
        iterations=12000,
        log_acceptance=np.log(0.5),
        acceptance_drift=((2000, -1.5), (12000, -0.2)),
        log_population=np.log(0.7),
        population_drift=((2000, -1.0), (12000, -0.1)),
        lineage_likelihood=0.07,
        lineage_population=0.2,
        likelihood_cost=1e-3,
        draw_cost=1e-5,
        training_cost=1e-4,
    ),
    # trainings scatter widely (noisy fits)
    "noisy-training": SimulatedProblem(
        "noisy-training",
        nlive=1000,
        iterations=12000,
        log_acceptance=np.log(0.2),
        acceptance_drift=-0.3,
        log_population=np.log(0.5),
        population_drift=-0.3,
        lineage_likelihood=0.07,
        lineage_population=0.2,
        training_scatter=0.5,
        reset_scatter=0.8,
        likelihood_cost=2e-3,
        draw_cost=2.4e-5,
        training_cost=1.5e-4,
    ),
    # a warm-started lineage that degrades as it grows (negative exponent),
    # as in some bimodal runs
    "degrading-lineage": SimulatedProblem(
        "degrading-lineage",
        nlive=1000,
        iterations=12000,
        log_acceptance=np.log(0.3),
        acceptance_drift=-0.05,
        log_population=np.log(0.7),
        lineage_likelihood=-0.3,
        likelihood_cost=1e-3,
        draw_cost=1e-5,
        training_cost=1e-4,
    ),
}
