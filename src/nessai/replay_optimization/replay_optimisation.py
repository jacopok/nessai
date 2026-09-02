#!/usr/bin/env python
"""
Optimise nessai's flow and training settings by *replaying* a finished nested
sampling run, without ever calling the likelihood again.

Motivation
----------
A finished run is a complete record of what the sampler had to model: at every
iteration the live points are an exact sample from the prior constrained to
``logL > logL_min``, and that constrained prior is precisely what the flow is
trained on.  A bilby result file stores every dead point together with the
iteration at which it was *born*, so the live set at any iteration can be
reconstructed exactly (see :func:`ArchivedRun.live_indices`).  That turns a
single expensive run into a stack of free, fully realistic training problems.

The likelihood is never evaluated: :class:`ReplayModel.log_likelihood` raises.
Everything below needs only the prior (analytic and cheap) and the archived
log-likelihood *values*.

What is optimised against
-------------------------
Two things went wrong in the run this was written for: the insertion-index KS
test failed badly, and it took a long time.  Both are predictable from a
trained flow without new likelihood calls.

nessai draws ``z`` from the latent distribution, discards ``|z| > r``
(``constant_volume_mode``), maps back to the physical space and rejection
samples with ``log_w = log_prior - log_q``.  The accepted points are therefore
distributed as ``prior(x) * 1[x in S]``, where ``S`` is the region the flow
maps inside the latent ball.  Two consequences drive the metrics:

1. *Fidelity.*  Conditioned on passing the likelihood threshold, a proposed
   point is distributed exactly like a uniformly-drawn live point restricted
   to ``S``.  So the insertion-index distribution can be reconstructed by
   taking held-out live points, masking those the flow maps outside the ball,
   and ranking their (archived) log-likelihoods against the live set.  A flow
   that fails to cover part of the constrained prior starves those ranks --
   which is exactly the insertion-index pathology.  Objective 1 is the KS
   *p-value* of the pooled indices (computed with nessai's own
   :func:`~nessai.utils.indices.compute_indices_ks_test`), *maximised*: a
   p-value near 1 means the reconstructed indices are indistinguishable from
   uniform, which is what a faithful proposal produces.

2. *Cost.*  The likelihood is evaluated for every point put in the pool, so
   the run's dominant cost is likelihood calls per nested sampling iteration,
   i.e. ``1 / P(logL > logL_min | proposed point)``.  With ``m(S)`` the prior
   mass of ``S``, ``X_i = exp(-i / nlive)`` the constrained prior volume and
   ``f`` the fraction of live points inside ``S``:

       P(logL > logL_min) = X_i * f / m(S)

   ``m(S) = v * E[exp(log_w)]`` over latent draws inside the ball, with ``v``
   the latent mass inside the ball -- all computable from the flow alone.
   A perfect flow gives m(S) = v * X_i and f = v, hence exactly one likelihood
   call per iteration.  Objective 2 is the base-10 log of the mean over
   checkpoints: the deepest checkpoints, where the flow struggles most, are
   meant to dominate, because a real run must pass through every iteration.
   The log is taken purely so the objective is easier to read -- likelihood
   calls per iteration span several orders of magnitude across the front, so
   objective 2 is effectively "order of magnitude of likelihood calls".

The two objectives are genuinely complementary: a flow that is too *narrow*
looks cheap but wrecks the insertion indices, while one that is too *broad*
keeps the indices clean and burns likelihood calls.  Optuna is run in
multi-objective mode and the archived configuration is enqueued as the first
trial, so the resulting Pareto front always contains the status quo.

Caveats
-------
* Held-out live points are ranked against the full live set they belong to,
  which shifts the insertion index by O(1 / nlive).  It is the same shift for
  every trial, so comparisons are unaffected.
* With ``inversion_type="duplicate"`` a physical point has two representations
  in the prime space and nessai's ``log_q`` only accounts for one of them.
  ``m(S)`` inherits that bias, so the absolute scale of objective 2 is
  approximate; the ranking between configurations is not.
* Each checkpoint trains a flow from scratch, whereas a real run retrains from
  the previous weights until ``reset_flow``.  Training from scratch is the
  harder problem, so this is conservative.
* ``m(S)`` is an importance sum over ``N_POPULATION_DRAWS`` latent draws whose
  weights are heavy-tailed at the deepest checkpoints.  A trial is pruned if
  the Kish effective sample size falls below ``MIN_EFFECTIVE_SAMPLE_SIZE``
  there; above it the estimate is usable but noisier the deeper the checkpoint.

The proposal settings the run used (``constant_volume_mode``,
``volume_fraction``, ``poolsize``, noise augmentation, the reparameterisation
flags) are read from the result file's ``sampler_kwargs``.  ``--config`` still
overrides them from an explicit ``config.json`` if one is given, and
:data:`FALLBACK_PROPOSAL_KWARGS` is the last resort.  With ``nessai-gw``
installed the replay uses the real ``GWFlowProposal`` (or
``AugmentedGWFlowProposal`` when the run was augmented).

Usage
-----
    python replay_optimisation.py --result run_result.json
    python replay_optimisation.py --result run_result.json --config config.json
    python replay_optimisation.py --result run_result.json --baseline
    python replay_optimisation.py --result run_result.json --n-trials 200
    python replay_optimisation.py --result run_result.json --n-jobs 16

Inspect with ``optuna-dashboard sqlite:///nessai_replay.sqlite3``.

Parallelism
-----------
Each trial trains a small flow, so PyTorch's own intra-op parallelism does not
come close to saturating a modern machine (its matrices are too small to
benefit from many threads).  The CPU-efficient way to use many cores is
instead to run several trials concurrently.  ``--n-jobs`` does this with
*processes*, not threads: a flow-training step is mostly a long sequence of
short, GIL-holding Python calls (the batch loop, the optimiser step, early
-stopping bookkeeping) rather than one big matrix multiply, so threads mostly
take turns holding the GIL instead of running on separate cores.  Each worker
process loads the archived run and opens the shared Optuna storage itself, so
they need no shared memory.  ``--pytorch-threads`` caps how many threads
*each worker* is allowed so that ``n-jobs`` of them don't oversubscribe the
machine between them.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
import warnings
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import optuna
from scipy.special import logsumexp

from nessai.livepoint import numpy_array_to_live_points
from nessai.model import Model
from nessai.proposal.flowproposal import FlowProposal
from nessai.utils.indices import compute_indices_ks_test
from nessai.utils.threading import configure_threads

logger = logging.getLogger("replay_optimisation")

# --------------------------------------------------------------------------
# Configuration -- the run being replayed is passed in via ``--result`` (and,
# optionally, ``--config`` for the sampler's ``config.json``); everything
# else generic to the optimisation lives here.
# --------------------------------------------------------------------------

STUDY_NAME = "nessai-replay"
STORAGE = "sqlite:///nessai_replay.sqlite3"
# Separate study/storage for the ET-triangle group-mixture proposal
# (``--group``): the flow is wrapped in a discrete eight-element symmetry
# mixture and z-scores every parameter (the group action lives in physical
# space and needs an affine reparameterisation), so its trials are not
# comparable with the GW-reparameterised latent-ball runs.
GROUP_STUDY_NAME = "nessai-replay-group"
GROUP_STORAGE = "sqlite:///nessai_replay_group.sqlite3"
N_TRIALS = 2000


def _make_storage(storage: str = STORAGE) -> optuna.storages.RDBStorage:
    """SQLite storage with a longer busy timeout for concurrent workers.

    With several worker *processes* writing at once, SQLite's default
    behaviour of failing immediately on a locked database becomes a real
    concern; a longer busy timeout makes it wait and retry instead.
    """
    return optuna.storages.RDBStorage(
        storage, engine_kwargs={"connect_args": {"timeout": 30}}
    )


def _make_sampler() -> optuna.samplers.BaseSampler:
    """Multi-objective TPE, tuned for a small, parallel, mixed-type budget.

    NSGA-II is population-based (default population size 50) and only starts
    to pay off after thousands of trials; at ``N_TRIALS`` order 100 it barely
    completes a couple of generations.  TPE is Optuna's documented default
    for this regime and handles the conditional search space natively:
    ``group=True`` is exactly for spaces where some parameters (e.g.
    ``num_bins``) only appear for certain values of another (``ftype``).
    ``constant_liar=True`` matters because several worker processes draw
    trials concurrently against the same storage; without it, they cannot see
    each other's in-flight trials and tend to sample redundantly nearby.
    """
    # multivariate/group/constant_liar are all still flagged "experimental" by
    # Optuna but are the right choice here (see the docstring); silence the
    # per-construction warnings rather than carry the noise through every run.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        return optuna.samplers.TPESampler(
            seed=SEED, multivariate=True, group=True, constant_liar=True
        )


# Checkpoints are spaced uniformly in iteration, i.e. uniformly in log prior
# volume.  The first FIRST_CHECKPOINT_FRACTION of the run is skipped: early on
# the sampler is still using the uninformed proposal.
N_CHECKPOINTS = 8
FIRST_CHECKPOINT_FRACTION = 0.1
HELDOUT_FRACTION = 0.2

# Latent draws used to estimate the prior mass of the flow's support and the
# rejection-sampling efficiency.  The support-mass estimate is an importance
# sum whose weights become very heavy-tailed at the deepest checkpoints (the
# constrained prior is tiny and the flow's tails dominate); 20k draws left an
# effective sample size of order 1 there, so the cost objective was decided by
# a handful of latent points.  See MIN_EFFECTIVE_SAMPLE_SIZE.
N_POPULATION_DRAWS = 200_000

# Kish effective sample size below which the support-mass estimate at a
# checkpoint is not a measurement but noise from a few dominant weights.  Kept
# deliberately low: the intent is only to reject configurations that cannot be
# scored at all, not to demand a tight estimate.  Checkpoints are evaluated
# deepest-first (see evaluate_config) so a configuration that will trip this is
# usually pruned on its first checkpoint, before the other seven are trained.
MIN_EFFECTIVE_SAMPLE_SIZE = 50.0

SEED = 1234
MAX_EPOCHS = 1000  # hard cap; `patience` is what actually stops training

# Last-resort proposal settings, used only when the result file records no
# ``sampler_kwargs`` and no ``config.json`` is supplied.  A real bilby result
# from a recent nessai always has ``sampler_kwargs`` (see
# :func:`_proposal_kwargs_from_sampler_kwargs`), so this should rarely be hit.
FALLBACK_NLIVE = 5000
FALLBACK_PROPOSAL_KWARGS = dict(
    poolsize=1000,
    constant_volume_mode=True,
    volume_fraction=0.97,
    expansion_fraction=4.0,
    max_radius=50.0,
    fallback_reparameterisation="zscore",
)
BASELINE_FLOW_CONFIG = dict(n_blocks=10, n_layers=4, n_neurons=48)
BASELINE_TRAINING_CONFIG = dict(patience=25, max_epochs=1000)

# The run used GWFlowProposal, whose default reparameterisations live in the
# separate `nessai-gw` package.  When that is not installed, this reproduces
# the setup the run's log reports, using core nessai reparameterisations, so
# the flow still sees the prime space it will see in production.  Entries for
# parameters the run does not have are dropped.  `luminosity_distance` is the
# one approximation: the real run used DistanceReparameterisation, which
# additionally applies a distance-specific transform before rescaling.
GW_FALLBACK_REPARAMETERISATIONS = {
    "chirp_mass": {
        "reparameterisation": "rescale-to-bounds",
        "update_bounds": True,
    },
    "mass_ratio": {
        "reparameterisation": "rescale-to-bounds",
        "detect_edges": True,
        "boundary_inversion": True,
        "inversion_type": "duplicate",
        "update_bounds": True,
    },
    "luminosity_distance": {
        "reparameterisation": "rescale-to-bounds",
        "detect_edges": True,
        "boundary_inversion": True,
        "inversion_type": "duplicate",
        "update_bounds": True,
    },
    "theta_jn": {"reparameterisation": "rescale-to-bounds"},
    "psi": {"reparameterisation": "angle", "scale": 2.0, "prior": "uniform"},
    "ra": {
        "reparameterisation": "angle-pair",
        "parameters": ["ra", "dec"],
        "convention": "ra-dec",
    },
    "chi_1": {"reparameterisation": "rescale-to-bounds"},
    "chi_2": {"reparameterisation": "rescale-to-bounds"},
}

# Keys of config.json that FlowProposal takes directly, and those that belong
# to the latent-radius truncation rule.
_PROPOSAL_KEYS = (
    "poolsize",
    "drawsize",
    "latent_temperature",
    "truncate_log_q",
    "check_acceptance",
    "max_poolsize_scale",
    "update_poolsize",
    "accumulate_weights",
    "reparameterisations",
    "fallback_reparameterisation",
    "use_default_reparameterisations",
    "reverse_reparameterisations",
    "map_to_unit_hypercube",
)
_LATENT_RADIUS_KEYS = (
    "constant_volume_mode",
    "volume_fraction",
    "fuzz",
    "fixed_radius",
    "min_radius",
    "max_radius",
    "compute_radius_with_all",
    "expansion_fraction",
)

# Augmented-proposal keys.  When the run recorded a positive ``augment_dims``
# the replay uses ``AugmentedGWFlowProposal`` and passes these through.
_AUGMENT_KEYS = (
    "augment_dims",
    "generate_augment",
    "marginalise_augment",
    "n_marg",
)


# --------------------------------------------------------------------------
# Model: prior only, the likelihood is deliberately unavailable
# --------------------------------------------------------------------------


class ReplayModel(Model):
    """A nessai model backed by the archived run's priors.

    The likelihood is never needed -- every metric is built from the archived
    log-likelihood *values* -- so it raises rather than silently returning a
    placeholder that could quietly corrupt a result.
    """

    def __init__(self, priors, names, rng=None):
        self.priors = priors
        self.names = list(names)
        self.bounds = {
            n: [priors[n].minimum, priors[n].maximum] for n in self.names
        }
        self.rng = rng if rng is not None else np.random.default_rng(SEED)

    def log_prior(self, x):
        log_p = np.log(self.in_bounds(x), dtype=float)
        log_p += self.priors.ln_prob(
            {n: np.atleast_1d(x[n]) for n in self.names}, axis=0
        )
        return log_p

    def log_likelihood(self, x):
        raise RuntimeError(
            "The likelihood is not available when replaying an archived run. "
            "Reaching this means a code path needs log-likelihoods that were "
            "meant to come from the archive."
        )

    def to_unit_hypercube(self, x):
        """Map from the prior space to the unit hypercube (per-parameter CDF).

        Needed by ``map_to_unit_hypercube`` proposals -- used by the ``--group``
        replay so the flow's prime space maps to *strictly* in-bounds physical
        values (the ET group action is only well defined on the physical box).
        """
        out = x.copy()
        u = self.priors.cdf({n: np.atleast_1d(x[n]).astype(float) for n in self.names})
        for n in self.names:
            out[n] = u[n]
        return out

    def from_unit_hypercube(self, x):
        """Inverse of :meth:`to_unit_hypercube` (per-parameter inverse CDF).

        Unit values are clipped to ``[0, 1]`` first: the flow's prime space is
        unbounded, so ``map_to_unit_hypercube`` proposals routinely land a hair
        outside, and some bilby priors (interpolated ones) raise rather than
        extrapolate.  Clipping makes the physical output *strictly* in-bounds,
        which is exactly what the ET group action needs.
        """
        out = x.copy()
        rescaled = self.priors.rescale(
            self.names,
            [
                np.clip(np.atleast_1d(x[n]).astype(float), 0.0, 1.0)
                for n in self.names
            ],
        )
        for n, v in zip(self.names, rescaled):
            out[n] = np.asarray(v)
        return out


# --------------------------------------------------------------------------
# The archived run
# --------------------------------------------------------------------------


@dataclass
class ArchivedRun:
    """Nested samples from a finished run, indexed by iteration."""

    names: list
    theta: np.ndarray  # (n_samples, n_dims), sorted by log-likelihood
    log_likelihood: np.ndarray
    birth_iteration: np.ndarray
    death_iteration: np.ndarray  # np.inf for the final live points
    nlive: int
    n_iterations: int
    priors: object
    proposal_kwargs: dict
    # Non-empty only when the run used noise augmentation; then the replay must
    # use ``AugmentedGWFlowProposal`` and pass these through.
    augment_kwargs: dict = field(default_factory=dict)

    def live_indices(self, iteration: int) -> np.ndarray:
        """Rows that were live when the sampler started ``iteration``.

        A point is live if it was born earlier and has not died yet.  Row ``j``
        (0-indexed, sorted by log-likelihood) is the point removed at iteration
        ``j + 1``, so ``death_iteration[j] = j + 1``.  The result is always
        exactly ``nlive`` rows, and its first entry is the worst point.
        """
        alive = (self.birth_iteration < iteration) & (
            self.death_iteration >= iteration
        )
        return np.flatnonzero(alive)

    def live_points(self, indices: np.ndarray, model: Model) -> np.ndarray:
        """Build a nessai structured array for the given rows."""
        x = numpy_array_to_live_points(
            self.theta[indices].copy(), self.names
        )
        x["logL"] = self.log_likelihood[indices]
        x["logP"] = model.batch_evaluate_log_prior(x)
        return x

    def log_volume(self, iteration: int) -> float:
        """log X at ``iteration``, using the standard ``E[log t] = -1/nlive``."""
        return -iteration / self.nlive

    def checkpoints(self, n: int) -> np.ndarray:
        start = int(FIRST_CHECKPOINT_FRACTION * self.n_iterations)
        start = max(start, self.nlive)
        return np.linspace(start, self.n_iterations, n).astype(int)


def _read_run_config(config_path: Path | None) -> dict:
    if config_path is None:
        return {}
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("No config.json at %s, using fallbacks", config_path)
        return {}
    with open(config_path) as f:
        return json.load(f)


def _sampler_kwargs_from_result(result) -> dict:
    """The nessai sampler settings the run was launched with.

    A bilby result from any recent nessai records every keyword passed to the
    sampler under ``result.sampler_kwargs``.  Those keys are exactly the ones
    :func:`_proposal_kwargs_from_config` expects, so the run's real proposal
    configuration -- ``constant_volume_mode``, ``volume_fraction``,
    ``poolsize``, augmentation, the reparameterisation flags -- can be replayed
    without a separate ``config.json``.
    """
    sk = getattr(result, "sampler_kwargs", None)
    if isinstance(sk, dict) and sk:
        return dict(sk)
    return {}


def load_archived_run(
    result_path: Path, config_path: Path | None = None
) -> ArchivedRun:
    """Load the nested samples and the sampler settings from a bilby result.

    Proposal settings are taken, in order of precedence, from an explicit
    ``config.json`` (``config_path``), then the run's own
    ``sampler_kwargs`` recorded in the result file, then
    :data:`FALLBACK_PROPOSAL_KWARGS`.
    """
    import bilby

    logger.info("Loading %s", result_path)
    result = bilby.core.result.read_in_result(str(result_path))

    names = list(result.search_parameter_keys)
    samples = result.nested_samples
    if samples is None:
        raise RuntimeError(
            "The result file has no nested samples; this script needs them."
        )
    if "iteration" not in samples:
        raise RuntimeError(
            "The nested samples have no `iteration` column, so live sets "
            "cannot be reconstructed.  A run made with a nessai version that "
            "records the birth iteration is required."
        )

    log_likelihood = samples["log_likelihood"].to_numpy()
    order = np.argsort(log_likelihood, kind="stable")
    samples = samples.iloc[order]
    log_likelihood = log_likelihood[order]

    birth = samples["iteration"].to_numpy()
    n_samples = len(samples)
    # The final live points never die; they are the highest-likelihood rows.
    n_iterations = int(birth.max())
    nlive = n_samples - n_iterations

    explicit_config = _read_run_config(config_path)
    sampler_kwargs = _sampler_kwargs_from_result(result)
    if explicit_config:
        config, source = explicit_config, "config.json"
    elif sampler_kwargs:
        config, source = sampler_kwargs, "result sampler_kwargs"
    else:
        config, source = dict(FALLBACK_PROPOSAL_KWARGS), "FALLBACK_PROPOSAL_KWARGS"
    logger.info("Proposal settings taken from %s", source)

    if config.get("nlive") is not None and config["nlive"] != nlive:
        logger.warning(
            "nlive from %s (%s) disagrees with the nested samples (%s); "
            "using the latter",
            source,
            config["nlive"],
            nlive,
        )

    death = np.arange(1, n_samples + 1, dtype=float)
    death[n_samples - nlive :] = np.inf

    proposal_kwargs, augment_kwargs = _proposal_kwargs_from_config(config)
    if augment_kwargs:
        logger.warning(
            "The run used noise augmentation (%s); replaying with "
            "AugmentedGWFlowProposal.  Pass --config to override.",
            augment_kwargs,
        )

    run = ArchivedRun(
        names=names,
        theta=samples[names].to_numpy(),
        log_likelihood=log_likelihood,
        birth_iteration=birth,
        death_iteration=death,
        nlive=nlive,
        n_iterations=n_iterations,
        priors=result.priors,
        proposal_kwargs=proposal_kwargs,
        augment_kwargs=augment_kwargs,
    )

    # Cheap consistency check: the live set must have exactly nlive members.
    for iteration in run.checkpoints(3):
        n_live = len(run.live_indices(iteration))
        if n_live != nlive:
            raise RuntimeError(
                f"Reconstructed {n_live} live points at iteration "
                f"{iteration}, expected {nlive}.  The `iteration` column is "
                "probably not the birth iteration."
            )
    logger.info(
        "Loaded %s nested samples, nlive=%s, %s iterations, %s dimensions",
        n_samples,
        nlive,
        n_iterations,
        len(names),
    )
    return run


def _proposal_kwargs_from_config(config: dict) -> tuple[dict, dict]:
    """Split a sampler-config dict into ``(proposal_kwargs, augment_kwargs)``.

    ``config`` is either a ``config.json`` or the run's ``sampler_kwargs``;
    both use the same keys as nessai's sampler.  Keys that are ``None`` or
    ``False`` are dropped, since they are the proposal defaults and only
    ``constant_volume_mode`` (which may be a meaningful ``True``) and the
    numeric radius settings ever need to be passed on.
    """
    if not config:
        config = dict(FALLBACK_PROPOSAL_KWARGS)

    def keep(value) -> bool:
        return value is not None and value is not False

    kwargs = {k: config[k] for k in _PROPOSAL_KEYS if k in config and keep(config[k])}
    latent_radius = {
        k: config[k] for k in _LATENT_RADIUS_KEYS if keep(config.get(k))
    }
    if latent_radius:
        kwargs["truncation_methods"] = ["latent_radius"]
        kwargs["truncation_kwargs"] = {"latent_radius": latent_radius}

    augment_kwargs = {}
    if config.get("augment_dims"):
        augment_kwargs = {
            k: config[k] for k in _AUGMENT_KEYS if keep(config.get(k))
        }

    return kwargs, augment_kwargs


def _apply_proposal_overrides(
    proposal_kwargs: dict,
    overrides: dict,
    *,
    names: list | None = None,
) -> dict:
    """Overlay Optuna-suggested proposal/truncation settings onto the
    archived run's fixed ``proposal_kwargs`` (see :func:`suggest_configs`).

    ``overrides["latent_radius"]`` is merged, not substituted wholesale, so
    that archived-config keys this module never searches over (``max_radius``,
    ``min_radius``, ``compute_radius_with_all``) survive untouched.

    ``overrides["flatten_distance_prior"]`` puts ``luminosity_distance`` on the
    power-law distance converter (power 2, i.e. uniform in Euclidean volume), so
    the flow sees a nearly flat prior for that parameter instead of the ~d^2
    curvature the run's ``UniformSourceFrame`` prior otherwise leaves it.
    """
    kwargs = dict(proposal_kwargs)
    kwargs["latent_temperature"] = overrides["latent_temperature"]
    kwargs["truncation_methods"] = ["latent_radius"]
    kwargs["truncation_kwargs"] = {
        "latent_radius": {
            **kwargs.get("truncation_kwargs", {}).get("latent_radius", {}),
            **overrides["latent_radius"],
        }
    }
    if overrides.get("flatten_distance_prior") and "luminosity_distance" in (
        names or []
    ):
        reparams = dict(kwargs.get("reparameterisations") or {})
        distance = dict(reparams.get("luminosity_distance") or {})
        distance.setdefault("reparameterisation", "distance")
        distance["prior"] = "power-law"
        distance["converter_kwargs"] = {
            **distance.get("converter_kwargs", {}),
            "power": 2.0,
        }
        reparams["luminosity_distance"] = distance
        kwargs["reparameterisations"] = reparams
    return kwargs


def resolve_reparameterisations(
    run_config: dict, proposal_class, names: list
) -> dict | None:
    """Decide which reparameterisations to replay with.

    An explicit setting in the run's config always wins.  Otherwise the
    proposal class supplies its own defaults, unless the run used a GW
    proposal that is not installed here -- then fall back on the transcribed
    GW setup, which is far closer than z-scoring everything.
    """
    if run_config.get("reparameterisations") is not None:
        return run_config["reparameterisations"]

    run_class = str(run_config.get("flow_proposal_class") or "")
    if "gw" not in run_class.lower() or "gw" in proposal_class.__name__.lower():
        return None

    reparameterisations = {
        name: cfg
        for name, cfg in GW_FALLBACK_REPARAMETERISATIONS.items()
        if name in names
    }
    logger.warning(
        "The run used %s, but %s is installed here.  Replaying with the "
        "transcribed GW reparameterisations for %s; install `nessai-gw` for "
        "an exact replay.",
        run_class,
        proposal_class.__name__,
        sorted(reparameterisations),
    )
    return reparameterisations or None


def get_proposal_class(augmented: bool = False, non_gw: bool = False):
    """The proposal class to replay with.

    ``GWFlowProposal`` when ``nessai-gw`` is installed (so the real GW
    reparameterisations are used), otherwise the core ``FlowProposal``.  With
    ``augmented=True`` the augmented variant is returned, to match a run that
    used noise augmentation.  With ``non_gw=True`` the GW proposals are skipped
    entirely and the core ``FlowProposal`` (or ``AugmentedFlowProposal``) is
    used -- the right choice for a run whose parameters are not GW parameters.
    """
    if non_gw:
        if augmented:
            from nessai.proposal import AugmentedFlowProposal

            return AugmentedFlowProposal
        return FlowProposal
    try:
        from nessai_gw import proposals as _gw

        return _gw.AugmentedGWFlowProposal if augmented else _gw.GWFlowProposal
    except ImportError:
        pass
    try:
        from nessai.gw.proposal import GWFlowProposal

        return GWFlowProposal
    except ImportError:
        pass
    if augmented:
        from nessai.proposal import AugmentedFlowProposal

        return AugmentedFlowProposal
    return FlowProposal


# Parameter names that mark a run as gravitational-wave.  Used only to pick a
# sensible default for ``--non-gw`` when the user does not say; the flag always
# wins over this guess.
_GW_PARAMETER_NAMES = frozenset(GW_FALLBACK_REPARAMETERISATIONS) | {
    "mass_1", "mass_2", "luminosity_distance", "geocent_time", "phase",
    "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl", "dec",
}


def run_looks_like_gw(run: ArchivedRun) -> bool:
    """Best guess at whether an archived run is a GW run.

    A GW run is replayed with ``GWFlowProposal`` and the GW reparameterisations;
    anything else wants the plain ``FlowProposal``.  The run does not record
    this directly, so guess from whether its parameters are GW parameters.
    """
    return any(name in _GW_PARAMETER_NAMES for name in run.names)


# --------------------------------------------------------------------------
# ET-triangle group-mixture proposal (``--group``)
# --------------------------------------------------------------------------
#
# ``nessai_gw.group_mixture`` provides an eight-element discrete symmetry group
# (Z4 x Z2) for a single triangular detector such as one Einstein Telescope
# site: the antipodal / quarter-turn sky-and-polarisation degeneracy that makes
# the archived ET BNS posteriors octomodal (see the module's docstring and the
# ``diagnose_degeneracy`` script).  ``nessai.flowmodel.group_mixture`` wraps a
# base flow so that ``q(x) = sum_g pi_g q0(g^-1 x)``: the base flow only has to
# model one mode of the orbit and the closed-form mixture weights cover the
# rest.  ``GroupFlowProposalMixin`` keeps the group action in physical
# coordinates by wiring the proposal's reparameterisation into the flow as a
# ``CoordinateBridge``, so the replay uses the core ``FlowProposal``.
#
# ``DiscreteGroupMixtureFlowWrapper`` assumes the *physical* group action is
# measure-preserving (unit Jacobian).  The nessai-gw fix makes that true by
# acting on ``sin_dec = sin(dec)`` and ``cos_theta_jn = cos(theta_jn)`` rather
# than the raw angles.  ``_to_group_frame`` rewrites the archived run in exactly
# those coordinates before anything else touches it: the bilby ``Cosine`` dec
# prior and ``Sine`` theta_jn prior are uniform in ``sin_dec`` /
# ``cos_theta_jn``, so it is a clean relabelling.
#
# A bounded ``logit`` map for the five acted parameters was tried first (see
# git history): rescale to [0, 1] on the known prior bounds, then ``logit``
# to an unbounded prime, exactly invertible via ``sigmoid``. It made every
# proposed point land strictly inside the physical box the ET action needs,
# but it also made the flow's importance weights collapse (ESS ~1): the
# posterior for ``sin_dec`` / ``cos_theta_jn`` has real, non-vanishing
# density right up to +-1 (a face-on/face-off or polar sky mode is a normal
# part of the octomodal posterior, not a rare edge case), and ``logit``'s
# derivative diverges there -- an ordinary, moderately-probable physical
# point lands far in the *prime*-space tail the base flow is trained on, so
# it gets a wildly underestimated density and hence a wildly overestimated
# weight. 83% of the 200 largest weights came from within 0.02 of a
# ``sin_dec`` / ``cos_theta_jn`` edge, vs. 3.7% of all proposed points.
#
# So every parameter is z-scored instead: affine, so the mixin's fast
# ``AffineBridge`` applies (no ``ReparamBridge`` round trip, no conjugation
# Jacobian to carry) and, critically, no artificial density singularity at
# the physical edges. The trade-off is that z-scoring is unbounded, so the
# flow can occasionally propose a point just outside the physical box (e.g.
# ``sin_dec`` slightly past +-1); ``ETTriangleGroupAction`` clamps such
# inputs internally rather than raising, and the group-mixture wrapper's
# non-injective-action fallback (see ``DiscreteGroupMixtureFlowWrapper``)
# keeps ``sample_and_log_prob`` / ``log_prob`` consistent for those points
# regardless, so this is an ordinary (harmless) rejection-sampling cost, not
# a correctness problem.


# The parameters ``ETTriangleGroupAction`` acts on.  Kept in sync with
# ``nessai_gw.group_mixture.ET_TRIANGLE_PARAMETERS`` (the quarter-turn sky
# rotation also carries ``geocent_time`` along the change in the
# geocenter-to-detector light-travel delay).  ``make_group_proposal_class``
# takes the authoritative list straight off the action instance and only
# falls back to this when checking coordinate availability.
ET_GROUP_ACTED_PARAMETERS = (
    "ra",
    "sin_dec",
    "cos_theta_jn",
    "psi",
    "phase",
    "geocent_time",
)


def _group_reparameterisations(names) -> dict:
    """Reparameterisations for the ``--group`` replay (see the note above).

    Every parameter is z-scored, so the mixin installs the affine bridge.
    """
    return {name: {"reparameterisation": "zscore"} for name in names}

# Physical angle -> measure-preserving coordinate used by the ET group action.
_GROUP_COORD_RENAME = {"dec": "sin_dec", "theta_jn": "cos_theta_jn"}
_GROUP_COORD_FN = {"dec": np.sin, "theta_jn": np.cos}


def _to_group_frame(run: ArchivedRun) -> ArchivedRun:
    """Rewrite an archived run in the ET group action's native coordinates.

    ``dec -> sin_dec`` and ``theta_jn -> cos_theta_jn`` (values, name, prior and
    bounds).  The action is then measure-preserving in the coordinates the flow
    z-scores, which is what the group-mixture wrapper assumes.  ``--group`` only.
    """
    missing = [p for p in _GROUP_COORD_RENAME if p not in run.names]
    if missing:
        raise RuntimeError(
            f"The run is missing {missing}, which the ET group action needs. "
            "The --group proposal only applies to a single-site ET-triangle "
            "BNS/BBH run with the standard extrinsic parameters."
        )
    from bilby.core.prior import PriorDict, Uniform

    names = [_GROUP_COORD_RENAME.get(n, n) for n in run.names]
    theta = run.theta.copy()
    for old, new in _GROUP_COORD_RENAME.items():
        theta[:, run.names.index(old)] = _GROUP_COORD_FN[old](
            run.theta[:, run.names.index(old)]
        )

    priors = PriorDict()
    for n in run.names:
        new = _GROUP_COORD_RENAME.get(n)
        if new is not None:
            priors[new] = Uniform(-1.0, 1.0, name=new)
        else:
            priors[n] = run.priors[n]

    return replace(run, names=names, theta=theta, priors=priors)


def _reference_time_from_run(run: ArchivedRun) -> float:
    """Geocentric GPS time for the ET group action.

    Only the sidereal time it implies is used (to rotate between the equatorial
    and Earth-fixed frames), so the posterior median of the time parameter --
    or the midpoint of its prior -- is plenty.
    """
    for key in ("geocent_time", "geocentric_time", "time"):
        if key in run.names:
            return float(np.median(run.theta[:, run.names.index(key)]))
    prior = None
    try:
        prior = run.priors["geocent_time"]
    except Exception:
        prior = None
    if prior is not None:
        return float((prior.minimum + prior.maximum) / 2.0)
    raise RuntimeError(
        "Cannot determine a reference GPS time for the ET group action; the "
        "run has no geocent_time parameter or prior."
    )


class _PassthroughGroupAction:
    """Adapt an :class:`ETTriangleGroupAction` to the full parameter vector.

    The ET group action is defined on ``acted`` (``action.parameters``, i.e.
    ``(ra, sin_dec, cos_theta_jn, psi, phase, geocent_time)``; the run is put
    in that frame by ``_to_group_frame``); the flow sees every sampled
    parameter.  This wrapper feeds the acted-on subset through and leaves the
    rest (masses, spins, distance) untouched.
    """

    def __init__(self, action, names, acted):
        self._action = action
        self._names = list(names)
        self._acted = list(acted)

    def __call__(self, point_dict, modes, inverse: bool = False) -> dict:
        sub = {k: point_dict[k] for k in self._acted}
        mapped = self._action(sub, modes, inverse=inverse)
        return {n: mapped.get(n, point_dict[n]) for n in self._names}

    def in_fundamental_domain(self, point_dict):
        return self._action.in_fundamental_domain(
            {k: point_dict[k] for k in self._acted}
        )


def make_group_proposal_class(run: ArchivedRun):
    """Build a ``FlowProposal`` wrapped in the ET-triangle group mixture."""
    from nessai_gw.group_mixture import ETTriangleGroupAction
    from nessai.flowmodel.group_mixture import (
        GroupFlowProposalMixin,
        make_group_mixture_flow,
    )

    action = ETTriangleGroupAction(
        reference_time=_reference_time_from_run(run)
    )
    acted = list(getattr(action, "parameters", ET_GROUP_ACTED_PARAMETERS))
    missing = sorted(set(acted) - set(run.names))
    if missing:
        raise RuntimeError(
            f"The run is missing {missing}, which the ET group action needs. "
            "The --group proposal only applies to a single-site ET-triangle "
            "BNS/BBH run with the standard extrinsic parameters."
        )

    passthrough = _PassthroughGroupAction(action, run.names, acted)
    flow_model_cls = make_group_mixture_flow(
        group_action_fn=passthrough,
        group_size=action.group_size,
        param_names=list(run.names),
        in_fundamental_domain=passthrough.in_fundamental_domain,
    )

    class GroupETFlowProposal(GroupFlowProposalMixin, FlowProposal):
        """Core FlowProposal wired for the ET-triangle group-mixture flow."""

        _FlowModelClass = flow_model_cls

    return GroupETFlowProposal


# --------------------------------------------------------------------------
# Scoring a single checkpoint
# --------------------------------------------------------------------------


def _coverage_mask(proposal, x) -> np.ndarray:
    """Which held-out points does the trained proposal treat as reachable?

    For latent-radius truncation this means "the flow maps the point inside the
    latent ball"; for ``min_log_q`` truncation it means "the flow's log-density
    is above the live-set minimum".
    Boundary inversion with ``inversion_type="duplicate"`` gives a point several
    representations in the prime space; the point is reachable if any of them
    is, so reduce over the duplicates with the most permissive one (smallest
    radius / largest log q).
    """
    radius_rule = proposal.truncation.get_rule("latent_radius")
    min_log_q_rule = proposal.truncation.get_rule("min_log_q")

    z, log_q = proposal.forward_pass(x, rescale=True)
    ratio, remainder = divmod(len(z), len(x))
    if remainder:
        raise RuntimeError(
            f"Rescaling produced {len(z)} prime points for {len(x)} "
            "points, which is not a whole multiple."
        )

    if radius_rule is not None:
        radius = np.sqrt(np.sum(z**2.0, axis=-1))
        radius = np.nanmin(radius.reshape(ratio, len(x)), axis=0)
        return radius <= float(radius_rule.threshold)
    if min_log_q_rule is not None:
        log_q = np.nanmax(log_q.reshape(ratio, len(x)), axis=0)
        return log_q > float(min_log_q_rule.min_log_q)
    raise RuntimeError(
        "The proposal has neither a latent_radius nor a min_log_q truncation "
        "rule, so held-out coverage cannot be measured."
    )


def _support_statistics(proposal, rng) -> dict:
    """Prior mass of the flow's support and the rejection-sampling efficiency.

    Mirrors the draw loop of :meth:`FlowProposal.populate`, stopping before the
    likelihood is evaluated.
    """
    z = proposal.sample_latent_distribution(N_POPULATION_DRAWS)
    z = proposal.truncation.apply_latent(proposal, z)
    n_in_ball = len(z)
    if n_in_ball == 0:
        raise optuna.TrialPruned("No latent samples inside the radius")
    # Empirical latent mass inside the ball; equals volume_fraction when
    # constant_volume_mode is on, but this also covers the other modes.
    latent_mass = n_in_ball / N_POPULATION_DRAWS

    x, log_q, z = proposal.backward_pass(
        z,
        rescale=True,
        return_z=True,
        return_unit_hypercube=proposal.map_to_unit_hypercube,
    )
    x, log_q, z = proposal.truncation.apply_after_backward(
        proposal, x, log_q, z
    )
    if not len(x):
        raise optuna.TrialPruned("No samples survived the backward pass")

    log_w = proposal.compute_weights(x, log_q)
    log_w = log_w[np.isfinite(log_w)]
    if not len(log_w):
        raise optuna.TrialPruned("All rejection-sampling weights were invalid")

    # Kish effective sample size of the importance weights.  Zero-weight draws
    # (dropped above) leave both sums unchanged, so this is computed on the
    # finite weights alone.  When it is small the support-mass sum below is
    # carried by a few latent points and m(S) is not measurable.
    log_ess = 2.0 * logsumexp(log_w) - logsumexp(2.0 * log_w)
    effective_sample_size = float(np.exp(log_ess))
    if effective_sample_size < MIN_EFFECTIVE_SAMPLE_SIZE:
        raise optuna.TrialPruned(
            f"Support-mass estimate has effective sample size "
            f"{effective_sample_size:.2f} (< {MIN_EFFECTIVE_SAMPLE_SIZE}); "
            "the configuration's cost cannot be estimated here."
        )

    # Points dropped for being out of bounds or non-finite have zero prior
    # mass, so they contribute nothing to the sum but do count in the mean.
    log_support_mass = (
        np.log(latent_mass) + logsumexp(log_w) - np.log(n_in_ball)
    )
    acceptance = float(np.mean(np.exp(log_w - log_w.max())))
    return dict(
        log_support_mass=float(log_support_mass),
        rejection_acceptance=acceptance,
        effective_sample_size=effective_sample_size,
        latent_mass=float(latent_mass),
        # nessai's own definition: accepted / drawn from the latent space.
        population_efficiency=acceptance * len(log_w) / N_POPULATION_DRAWS,
    )


def evaluate_checkpoint(
    run: ArchivedRun,
    model: ReplayModel,
    iteration: int,
    flow_config: dict,
    training_config: dict,
    proposal_class,
    rng: np.random.Generator,
    output: str,
    proposal_overrides: dict | None = None,
    use_group: bool = False,
) -> dict:
    """Train a flow on the live set at ``iteration`` and score it."""
    indices = run.live_indices(iteration)
    # indices are sorted, so the first is the worst point -- the one the
    # sampler is about to replace.  It defines the likelihood threshold and
    # must stay in the training set, exactly as in a real run.
    worst = indices[:1]
    rest = indices[1:]

    n_heldout = int(round(HELDOUT_FRACTION * len(indices)))
    shuffled = rng.permutation(rest)
    heldout_idx = np.sort(shuffled[:n_heldout])
    train_idx = np.sort(np.concatenate([worst, shuffled[n_heldout:]]))

    x_train = run.live_points(train_idx, model)
    x_heldout = run.live_points(heldout_idx, model)
    worst_point = run.live_points(worst, model)

    if proposal_overrides is not None:
        proposal_kwargs = _apply_proposal_overrides(
            run.proposal_kwargs,
            proposal_overrides,
            names=run.names,
        )
    else:
        proposal_kwargs = run.proposal_kwargs
    if use_group:
        # ``_group_reparameterisations`` z-scores every parameter (see its
        # docstring for why); ``map_to_unit_hypercube``'s CDF-based prior
        # isn't used by that path, so it is dropped here.
        proposal_kwargs = {
            k: v
            for k, v in proposal_kwargs.items()
            if k != "map_to_unit_hypercube"
        }
        proposal_kwargs["reparameterisations"] = _group_reparameterisations(
            run.names
        )
        proposal_kwargs["fallback_reparameterisation"] = "zscore"
    # The group proposal has no augmented variant, so the run's noise
    # augmentation is dropped for it (see make_group_proposal_class).
    augment_kwargs = {} if use_group else run.augment_kwargs
    proposal = proposal_class(
        model,
        flow_config=dict(flow_config),
        training_config=dict(training_config),
        output=output,
        plot=False,
        rng=rng,
        **proposal_kwargs,
        **augment_kwargs,
    )
    proposal.initialise()

    start = time.perf_counter()
    proposal.train(x_train, plot=False)
    train_time = time.perf_counter() - start

    proposal.truncation.prepare(proposal, worst_point)

    # --- fidelity: the insertion indices the proposal would produce ---------
    inside = _coverage_mask(proposal, x_heldout)
    coverage = float(np.mean(inside))
    if coverage <= 0:
        raise optuna.TrialPruned(
            f"Flow covers no live points at iteration {iteration}"
        )

    live_logl = np.sort(run.log_likelihood[indices])
    # Matches NestedSampler.insert_live_point, which subtracts one so that
    # index 0 is reachable.
    insertion_indices = (
        np.searchsorted(live_logl, run.log_likelihood[heldout_idx][inside]) - 1
    )
    insertion_indices = np.clip(insertion_indices, 0, run.nlive - 1)

    # --- cost: likelihood calls the sampler would need per iteration --------
    support = _support_statistics(proposal, rng)
    log_ratio = (
        support["log_support_mass"]
        - run.log_volume(iteration)
        - np.log(coverage)
    )
    likelihood_calls = float(np.exp(log_ratio))

    ks_d, ks_p = compute_indices_ks_test(insertion_indices, run.nlive)

    n_params = sum(
        p.numel() for p in proposal.flow.model.parameters() if p.requires_grad
    )
    del proposal

    return dict(
        iteration=int(iteration),
        insertion_indices=insertion_indices,
        coverage=coverage,
        likelihood_calls_per_iteration=likelihood_calls,
        ks_statistic=float(ks_d) if ks_d is not None else float("nan"),
        ks_p_value=float(ks_p) if ks_p is not None else float("nan"),
        train_time=train_time,
        n_flow_parameters=int(n_params),
        **support,
    )


def evaluate_config(
    run: ArchivedRun,
    model: ReplayModel,
    flow_config: dict,
    training_config: dict,
    proposal_class,
    seed: int,
    trial: optuna.Trial | None = None,
    proposal_overrides: dict | None = None,
    use_group: bool = False,
) -> dict:
    """Score a configuration across every checkpoint of the run.

    Checkpoints are visited deepest-first.  The objective is the *mean* over
    checkpoints -- the worst checkpoints are meant to dominate, because a real
    run must pass through every iteration and its cost is set by where the flow
    struggles most -- and the deepest checkpoint is both the worst and the one
    most likely to fail the effective-sample-size guard in
    :func:`_support_statistics`.  Evaluating it first prunes a hopeless
    configuration after one trained flow instead of eight.  The mean and the
    pooled KS test are order-independent, so this does not change any score.
    """
    rng = np.random.default_rng(seed)
    results = []
    with tempfile.TemporaryDirectory(prefix="nessai-replay-") as output:
        for iteration in run.checkpoints(N_CHECKPOINTS)[::-1]:
            result = evaluate_checkpoint(
                run,
                model,
                iteration,
                flow_config,
                training_config,
                proposal_class,
                rng,
                os.path.join(output, f"it_{iteration}"),
                proposal_overrides=proposal_overrides,
                use_group=use_group,
            )
            results.append(result)
            logger.info(
                "  trial=%s it=%-7d coverage=%.3f  KS D=%.4f  "
                "log10 L-calls/it=%.2f  ESS=%.1f  train=%.1fs",
                "?" if trial is None else trial.number,
                result["iteration"],
                result["coverage"],
                result["ks_statistic"],
                np.log10(result["likelihood_calls_per_iteration"]),
                result["effective_sample_size"],
                result["train_time"],
            )
            if trial is not None:
                trial.set_user_attr(
                    f"checkpoint_{iteration}",
                    {
                        k: v
                        for k, v in result.items()
                        if k != "insertion_indices"
                    },
                )

    # Pooling the indices mirrors the final (non-rolling) KS test, which is
    # the one that failed for this run.
    pooled = np.concatenate([r["insertion_indices"] for r in results])
    ks_d, ks_p = compute_indices_ks_test(pooled, run.nlive)

    calls = np.array(
        [r["likelihood_calls_per_iteration"] for r in results], dtype=float
    )
    return dict(
        ks_statistic=float(ks_d),
        ks_p_value=float(ks_p),
        # The mean is the right estimator of the total: checkpoints are
        # uniformly spaced in log prior volume, so each stands for an equal
        # slice of the run.
        likelihood_calls_per_iteration=float(np.mean(calls)),
        likelihood_calls_median=float(np.median(calls)),
        likelihood_calls_max=float(np.max(calls)),
        coverage=float(np.mean([r["coverage"] for r in results])),
        effective_sample_size_min=float(
            np.min([r["effective_sample_size"] for r in results])
        ),
        train_time=float(np.sum([r["train_time"] for r in results])),
        n_flow_parameters=int(results[0]["n_flow_parameters"]),
        n_checkpoints=len(results),
    )


# --------------------------------------------------------------------------
# Search space
# --------------------------------------------------------------------------


def suggest_configs(
    trial: optuna.Trial,
    *,
    allow_nsf: bool = True,
    use_group: bool = False,
) -> tuple[dict, dict, dict]:
    """Sample the flow architecture, the training settings, and the
    proposal/truncation settings that control the latent ball the flow's
    support is measured against (see :func:`_apply_proposal_overrides`).

    ``allow_nsf`` is ``False`` when replaying an augmented run: the augmented
    proposal injects a custom coupling ``mask`` into the flow config, which
    nessai's neural-spline flow does not accept (it hard-codes an alternating
    mask), so only ``realnvp`` can be used there.
    """
    choices = ["realnvp", "nsf"] if allow_nsf else ["realnvp"]
    ftype = trial.suggest_categorical("ftype", choices)
    flow_config = dict(
        ftype=ftype,
        n_blocks=trial.suggest_int("n_blocks", 2, 16),
        n_layers=trial.suggest_int("n_layers", 1, 4),
        n_neurons=trial.suggest_int("n_neurons", 16, 256, log=True),
        batch_norm_between_layers=trial.suggest_categorical(
            "batch_norm_between_layers", [True, False]
        ),
        dropout_probability=trial.suggest_float(
            "dropout_probability", 0.0, 0.2
        ),
        linear_transform=trial.suggest_categorical(
            "linear_transform", ["lu", "permutation"]
        ),
    )
    if ftype == "nsf":
        # Spline flows are the reason to be here for complex surfaces: the bin
        # count sets how much structure a single transform can represent.
        flow_config["num_bins"] = trial.suggest_int("num_bins", 4, 16)
        flow_config["tail_bound"] = trial.suggest_float(
            "tail_bound", 3.0, 10.0
        )

    training_config = dict(
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        batch_size=trial.suggest_categorical(
            "batch_size", [200, 500, 1000, 2000]
        ),
        patience=trial.suggest_int("patience", 10, 100),
        max_epochs=MAX_EPOCHS,
        annealing=trial.suggest_categorical("annealing", [True, False]),
        optimiser=trial.suggest_categorical("optimiser", ["adam", "adamw"]),
        optimiser_kwargs={
            "weight_decay": trial.suggest_float(
                "weight_decay", 1e-6, 1e-2, log=True
            )
        },
        val_size=0.1,
        clip_grad_norm=5.0,
    )
    # Noise augmentation smooths the density the flow learns, which is the
    # standard lever against the over-tight proposals that starve insertion
    # indices.
    noise_scale = trial.suggest_float("noise_scale", 1e-3, 0.5, log=True)
    if noise_scale > 1e-3:
        training_config["noise_type"] = "adaptive"
        training_config["noise_scale"] = noise_scale

    # Rescales the latent draws the proposal samples from (see
    # FlowProposal.sample_latent_distribution); temperature=1 is a no-op.
    latent_temperature = trial.suggest_float(
        "latent_temperature", 0.5, 2.0, log=True
    )

    # LatentRadiusTruncation.configure() makes volume_fraction and
    # fuzz/expansion_fraction mutually exclusive: under constant_volume_mode
    # the radius comes from volume_fraction and fuzz is forced back to 1.0;
    # otherwise expansion_fraction, whenever it is set, silently overwrites
    # fuzz.  So exactly one of the three is ever a live lever -- branch on
    # which, rather than suggesting all three and letting two of them be
    # ignored.
    constant_volume_mode = trial.suggest_categorical(
        "constant_volume_mode", [True, False]
    )
    if constant_volume_mode:
        # Fraction of the flow's latent mass kept inside the ball -- the
        # direct control on the fidelity/cost tradeoff these two objectives
        # measure (see the module docstring).
        latent_radius_config = dict(
            constant_volume_mode=True,
            fixed_radius=False,
            volume_fraction=trial.suggest_float(
                "volume_fraction", 0.8, 0.999
            ),
            fuzz=1.0,
            expansion_fraction=None,
        )
    else:
        use_expansion_fraction = trial.suggest_categorical(
            "use_expansion_fraction", [True, False]
        )
        if use_expansion_fraction:
            latent_radius_config = dict(
                constant_volume_mode=False,
                fixed_radius=False,
                fuzz=1.0,
                expansion_fraction=trial.suggest_float(
                    "expansion_fraction", 0.5, 8.0, log=True
                ),
            )
        else:
            latent_radius_config = dict(
                constant_volume_mode=False,
                fixed_radius=False,
                fuzz=trial.suggest_float("fuzz", 0.5, 2.0),
                expansion_fraction=None,
            )

    # The run left `luminosity_distance` on nessai-gw's identity distance
    # converter (its `UniformSourceFrame` prior is not one nessai-gw recognises),
    # so the flow has to absorb the ~d^2 prior curvature.  Toggle the power-law
    # converter (power 2, uniform in Euclidean volume) to hand it a nearly flat
    # prior instead; `False` reproduces the production run.  It uses a
    # non-affine distance converter, so it is off for the group proposal
    # (whose mixin requires an affine reparameterisation).
    flatten_distance_prior = not use_group

    proposal_overrides = dict(
        latent_temperature=latent_temperature,
        latent_radius=latent_radius_config,
        flatten_distance_prior=flatten_distance_prior,
    )

    return flow_config, training_config, proposal_overrides


def baseline_params() -> dict:
    """The archived run's settings, expressed in the search space above."""
    return dict(
        ftype="realnvp",
        n_blocks=BASELINE_FLOW_CONFIG["n_blocks"],
        n_layers=BASELINE_FLOW_CONFIG["n_layers"],
        n_neurons=BASELINE_FLOW_CONFIG["n_neurons"],
        batch_norm_between_layers=True,
        dropout_probability=0.0,
        linear_transform="lu",
        lr=1e-3,
        batch_size=1000,
        patience=BASELINE_TRAINING_CONFIG["patience"],
        annealing=False,
        optimiser="adamw",
        weight_decay=1e-6,
        noise_scale=1e-3,
        latent_temperature=1.0,
        constant_volume_mode=True,
        volume_fraction=FALLBACK_PROPOSAL_KWARGS["volume_fraction"],
        flatten_distance_prior=False,
    )


# --------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------


def make_objective(
    run: ArchivedRun,
    proposal_class,
    use_group: bool = False,
):
    # The group proposal drops the run's noise augmentation, so NSF is allowed
    # again (the augmented proposal was the only thing that forbade it).
    allow_nsf = use_group or not bool(run.augment_kwargs)

    def objective(trial: optuna.Trial):
        flow_config, training_config, proposal_overrides = suggest_configs(
            trial, allow_nsf=allow_nsf, use_group=use_group
        )
        logger.info("Trial %s: %s", trial.number, trial.params)
        seed = SEED + trial.number
        # A fresh model per trial, rather than one shared across trials, so
        # that concurrent trials (--n-jobs > 1) do not race on its RNG.
        model = ReplayModel(run.priors, run.names, rng=np.random.default_rng(seed))
        summary = evaluate_config(
            run,
            model,
            flow_config,
            training_config,
            proposal_class,
            seed=seed,
            trial=trial,
            proposal_overrides=proposal_overrides,
            use_group=use_group,
        )
        for key, value in summary.items():
            trial.set_user_attr(key, value)
        log_likelihood_calls = float(
            np.log10(summary["likelihood_calls_per_iteration"])
        )
        logger.info(
            "Trial %s: KS p=%.4g (D=%.4f), log10 L-calls/it=%.2f "
            "(L-calls/it=%.2f), min ESS=%.1f",
            trial.number,
            summary["ks_p_value"],
            summary["ks_statistic"],
            log_likelihood_calls,
            summary["likelihood_calls_per_iteration"],
            summary["effective_sample_size_min"],
        )
        return (
            summary["ks_p_value"],
            log_likelihood_calls,
        )

    return objective


def _run_worker(
    result_path: Path,
    config_path: Path | None,
    n_checkpoints: int,
    n_trials: int,
    pytorch_threads: int,
    log_level: str,
    non_gw: bool | None = None,
    storage: str = STORAGE,
    study_name: str = STUDY_NAME,
    use_group: bool = False,
) -> None:
    """Entry point for one ``--n-jobs`` worker process.

    Runs ``n_trials`` against the study in ``STORAGE``, which every worker
    (and the parent process) shares.  Each worker reloads the archived run
    itself rather than inheriting it from the parent, since the process pool
    uses the ``spawn`` start method (safer with PyTorch than ``fork``) and so
    nothing is inherited across the fork boundary anyway.
    """
    global N_CHECKPOINTS

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    logging.getLogger("nessai").setLevel(logging.WARNING)
    configure_threads(pytorch_threads=pytorch_threads)

    N_CHECKPOINTS = n_checkpoints
    run = load_archived_run(result_path, config_path)
    if non_gw is None:
        non_gw = not run_looks_like_gw(run)
    if use_group:
        run = _to_group_frame(run)
        proposal_class = make_group_proposal_class(run)
    else:
        proposal_class = get_proposal_class(
            augmented=bool(run.augment_kwargs), non_gw=non_gw
        )

    study = optuna.load_study(
        study_name=study_name,
        storage=_make_storage(storage),
        sampler=_make_sampler(),
    )
    study.optimize(
        make_objective(
            run, proposal_class, use_group=use_group
        ),
        n_trials=n_trials,
        catch=(RuntimeError, ValueError),
    )


def main():
    global N_CHECKPOINTS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Score the archived configuration and exit.",
    )
    parser.add_argument(
        "--n-checkpoints", type=int, default=N_CHECKPOINTS
    )
    parser.add_argument(
        "--result",
        type=Path,
        required=True,
        help="Path to the bilby result JSON file of the run to replay.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to the sampler's config.json.  Overrides the proposal "
            "settings otherwise read from the result file's sampler_kwargs; "
            "FALLBACK_PROPOSAL_KWARGS is used only if neither is available."
        ),
    )
    parser.add_argument(
        "--non-gw",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Replay with the core FlowProposal instead of GWFlowProposal "
            "(and skip the GW fallback reparameterisations).  Use this for a "
            "run whose parameters are not GW parameters.  Left unset, it is "
            "guessed from the parameter names."
        ),
    )
    parser.add_argument(
        "--group",
        action="store_true",
        help=(
            "Replay with the core FlowProposal wrapped in nessai-gw's "
            "eight-element ET-triangle symmetry group mixture "
            "(nessai.flowmodel.group_mixture + "
            "nessai_gw.group_mixture.ETTriangleGroupAction).  Replays in "
            "(ra, sin_dec, cos_theta_jn, psi, ...) coordinates (the ET action's "
            "measure-preserving frame) with GROUP_REPARAMETERISATIONS and drops "
            "the run's noise augmentation.  Unless --storage / "
            "--study-name are given, a separate study "
            f"({GROUP_STUDY_NAME} in {GROUP_STORAGE}) is used."
        ),
    )
    parser.add_argument(
        "--storage",
        default=None,
        help=f"Optuna storage URL (default: {STORAGE}, or {GROUP_STORAGE} with --group).",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help=f"Optuna study name (default: {STUDY_NAME}, or {GROUP_STUDY_NAME} with --group).",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Number of worker *processes* to run trials in concurrently "
            "(threads don't scale for this workload, see the module "
            "docstring). Pass -1 to use all available CPUs "
            "(os.cpu_count()=%d on this machine)."
        )
        % (os.cpu_count() or 1),
    )
    parser.add_argument(
        "--pytorch-threads",
        type=int,
        default=1,
        help=(
            "Max PyTorch intra-op threads per worker. Kept low by default so "
            "that --n-jobs concurrent workers don't oversubscribe the CPU "
            "between them; a single flow is usually too small to benefit "
            "from many threads anyway."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    logging.getLogger("nessai").setLevel(logging.WARNING)

    configure_threads(pytorch_threads=args.pytorch_threads)

    N_CHECKPOINTS = args.n_checkpoints

    if args.group:
        default_storage, default_study = GROUP_STORAGE, GROUP_STUDY_NAME
    else:
        default_storage, default_study = STORAGE, STUDY_NAME
    storage = args.storage or default_storage
    study_name = args.study_name or default_study

    run = load_archived_run(args.result, args.config)
    non_gw = (not run_looks_like_gw(run)) if args.non_gw is None else args.non_gw
    if args.group:
        run = _to_group_frame(run)
        proposal_class = make_group_proposal_class(run)
    else:
        proposal_class = get_proposal_class(
            augmented=bool(run.augment_kwargs), non_gw=non_gw
        )
    logger.info(
        "Using %s (non_gw=%s%s), study %r in %s",
        proposal_class.__name__,
        non_gw,
        ", guessed" if args.non_gw is None else "",
        study_name,
        storage,
    )

    if args.baseline:
        model = ReplayModel(run.priors, run.names)
        summary = evaluate_config(
            run,
            model,
            BASELINE_FLOW_CONFIG,
            dict(BASELINE_TRAINING_CONFIG, max_epochs=MAX_EPOCHS),
            proposal_class,
            seed=SEED,
            use_group=args.group,
        )
        print("\nArchived configuration")
        print(f"  flow     : {BASELINE_FLOW_CONFIG}")
        print(f"  training : {BASELINE_TRAINING_CONFIG}")
        for key, value in summary.items():
            print(f"  {key:32s} {value}")
        return

    study = optuna.create_study(
        study_name=study_name,
        storage=_make_storage(storage),
        directions=["maximize", "minimize"],
        load_if_exists=True,
        sampler=_make_sampler(),
    )
    n_trials = args.n_trials
    if not study.trials:
        # Put the status quo on the Pareto front so every result has a
        # reference point.  Enqueued and run here, synchronously, before any
        # worker starts: Optuna pops a WAITING trial from storage with a
        # read-then-write that isn't atomic across processes on the sqlite
        # backend, so if this were left for the worker pool below, more than
        # one worker could see it as waiting and run it concurrently.
        study.enqueue_trial(baseline_params())
        study.optimize(
            make_objective(
                run, proposal_class, use_group=args.group
            ),
            n_trials=1,
            catch=(RuntimeError, ValueError),
        )
        n_trials = max(n_trials - 1, 0)

    n_jobs = (os.cpu_count() or 1) if args.n_jobs == -1 else args.n_jobs
    if n_jobs <= 1:
        study.optimize(
            make_objective(
                run, proposal_class, use_group=args.group
            ),
            n_trials=n_trials,
            catch=(RuntimeError, ValueError),
        )
    else:
        counts = [n_trials // n_jobs] * n_jobs
        for i in range(n_trials % n_jobs):
            counts[i] += 1
        ctx = get_context("spawn")
        processes = [
            ctx.Process(
                target=_run_worker,
                args=(
                    args.result,
                    args.config,
                    args.n_checkpoints,
                    n,
                    args.pytorch_threads,
                    args.log_level,
                    non_gw,
                    storage,
                    study_name,
                    args.group,
                ),
            )
            for n in counts
            if n > 0
        ]
        logger.info(
            "Starting %d worker processes for %d trials",
            len(processes),
            n_trials,
        )
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        failed = [p for p in processes if p.exitcode]
        if failed:
            raise RuntimeError(
                f"{len(failed)}/{len(processes)} worker process(es) failed; "
                "see the logs above for the tracebacks."
            )
        # `study` re-queries the shared storage on every access below, so it
        # already reflects what the workers wrote -- no need to reload it.

    print(
        "\nPareto-optimal trials (KS p-value, log10 likelihood calls / it):"
    )
    for t in sorted(study.best_trials, key=lambda t: t.values[0], reverse=True):
        print(
            f"  trial {t.number:4d}  p={t.values[0]:.4g}  "
            f"log10 calls/it={t.values[1]:6.2f}  "
            f"(calls/it={10 ** t.values[1]:,.0f})  {t.params}"
        )


if __name__ == "__main__":
    main()
