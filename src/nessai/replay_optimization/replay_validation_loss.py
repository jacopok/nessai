#!/usr/bin/env python
"""
Optimise nessai's flow and training settings by *replaying* a finished nested
sampling run, without ever calling the likelihood again.

This variant of ``replay_optimisation.py`` collapses the optimisation to a
**single** cost function: the flow's minimum validation loss (validation NLL),
aggregated across checkpoints with a soft-max (:func:`_val_loss_softmax`) so the
worst checkpoint dominates without the aggregate being decided by it alone.
Optuna runs single-objective and *minimises* it.  Everything else -- the replay
mechanics, the archived-run reconstruction, the plateau-stopped burst trainer,
the ``--group`` ET-triangle mixture -- is unchanged from the multi-objective
script; the coverage / support-mass / insertion-index machinery is simply not
scored here.

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

3. *Calibration.*  ``m(S)`` is an importance sum over latent draws with
   weights ``exp(log_w)``; its Kish effective sample size measures how
   well-calibrated the flow's density is against the prior over its own
   support.  A low ESS means nessai's ``populate`` loop rejection-samples
   inefficiently (many draws per accepted pool point) and the cost estimate
   itself is noisy -- both real inefficiencies a hard floor
   (``MIN_EFFECTIVE_SAMPLE_SIZE``) only rules out at the extreme.  Objective 3
   is a smooth minimum of ``log10(ESS)`` over checkpoints
   (:func:`_log10_ess_softmin`), *maximised*.

The three objectives are genuinely complementary: a flow that is too *narrow*
looks cheap but wrecks the insertion indices, one that is too *broad* keeps the
indices clean and burns likelihood calls, and one whose density is poorly
calibrated against its support samples inefficiently regardless.  Optuna is run
in multi-objective mode and the archived configuration is enqueued as the first
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
* One flow is built per trial and retrained at each checkpoint, warm-starting
  from the previous checkpoint's weights -- the checkpoints are visited in
  iteration order (shallow to deep), exactly as a real run reuses the flow
  between iterations until ``reset_flow``.  Each retrain is stopped by a
  trailing-median-validation-NLL plateau rule rather than nessai's own patience
  (:func:`_install_staged_training`), which the noisy validation curve defeats.
* ``m(S)`` is an importance sum over ``N_POPULATION_DRAWS`` latent draws whose
  weights are heavy-tailed at the deepest checkpoints, so its Kish effective
  sample size can be small there.  A small ESS is not rejected -- it only makes
  objective 2 (cost) noisier and pulls objective 3 (softmin log10 ESS) down,
  which is the soft pressure towards a well-calibrated proposal.  A checkpoint
  is only rejected when it cannot be scored at all (empty support, no
  coverage, all-invalid weights); see :class:`ConfigRejected`.

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
from dataclasses import dataclass, field
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

logger = logging.getLogger("replay_validation_loss")

# --------------------------------------------------------------------------
# Configuration -- the run being replayed is passed in via ``--result`` (and,
# optionally, ``--config`` for the sampler's ``config.json``); everything
# else generic to the optimisation lives here.
# --------------------------------------------------------------------------

STUDY_NAME = "nessai-replay-valloss"
STORAGE = "sqlite:///nessai_replay_valloss.sqlite3"
# Separate study/storage for the ET-triangle group-mixture proposal
# (``--group``): the flow is wrapped in a discrete 16-element symmetry mixture
# with its own (triangular_group_reparameterisations) prime space, so its
# trials are not comparable with the plain GW-reparameterised latent-ball runs.
GROUP_STUDY_NAME = "nessai-replay-group-valloss"
GROUP_STORAGE = "sqlite:///nessai_replay_group_valloss.sqlite3"
N_TRIALS = 2000

# Soft-max (in nat units) that aggregates the per-checkpoint minimum validation
# loss into the single cost:
#
#     softmax_b(v) = 1/b * log( mean_i exp(b * v_i) )
#
# -> ``max(v)`` as ``b -> inf`` and ``mean(v)`` as ``b -> 0``.  ``b = 2`` leans
# towards the worst (deepest) checkpoint without being decided by it.
VAL_LOSS_SOFTMAX_BETA = 2.0

# Cost returned for a configuration that cannot be scored at some checkpoint
# (:class:`ConfigRejected` -- e.g. the first-checkpoint divergence probe). Far
# above any real validation NLL, so Optuna learns to avoid the region.
REJECT_VAL_LOSS = 1.0e6


def _make_storage(storage: str = STORAGE) -> optuna.storages.RDBStorage:
    """SQLite storage with a longer busy timeout for concurrent workers.

    With several worker *processes* writing at once, SQLite's default
    behaviour of failing immediately on a locked database becomes a real
    concern; a longer busy timeout makes it wait and retry instead.
    """
    return optuna.storages.RDBStorage(
        storage, engine_kwargs={"connect_args": {"timeout": 30}}
    )


def _make_sampler(seed: int | None = None) -> optuna.samplers.BaseSampler:
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

    ``seed`` **must differ between worker processes**.  During TPE's random
    startup phase (``n_startup_trials``, default 10) the sampler just draws
    from its seeded RNG, and ``constant_liar`` -- which only steers the TPE
    model-fitting step, not the random fallback -- does nothing to
    decorrelate them.  Identically-seeded workers therefore propose byte-for
    -byte identical configurations until the study has enough completed
    trials to leave startup.  :func:`_run_worker` is passed ``SEED + 1 + i``.
    """
    # multivariate/group/constant_liar are all still flagged "experimental" by
    # Optuna but are the right choice here (see the docstring); silence the
    # per-construction warnings rather than carry the noise through every run.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        return optuna.samplers.TPESampler(
            seed=SEED if seed is None else seed,
            multivariate=True,
            group=True,
            constant_liar=True,
        )


# Checkpoints are spaced uniformly in iteration, i.e. uniformly in log prior
# volume.  The first FIRST_CHECKPOINT_FRACTION of the run is skipped: early on
# the sampler is still using the uninformed proposal.
N_CHECKPOINTS = 10
FIRST_CHECKPOINT_FRACTION = 0.1
HELDOUT_FRACTION = 0.2

# Latent draws used to estimate the prior mass of the flow's support and the
# rejection-sampling efficiency.  The support-mass estimate is an importance
# sum whose weights become very heavy-tailed at the deepest checkpoints (the
# constrained prior is tiny and the flow's tails dominate); 20k draws left an
# effective sample size of order 1 there, so the cost objective was decided by
# a handful of latent points.  50k is a compromise: with the hard ESS floor
# gone, a noisier m(S) here only softens objectives 2 and 3, it no longer
# rejects the trial, so the extra draws buy less than they used to.
N_POPULATION_DRAWS = 50_000

# Kish effective sample size below which the support-mass estimate at a
# checkpoint is not a measurement but noise from a few dominant weights.  Kept
# deliberately low: the intent is only to reject configurations that cannot be
# scored at all, not to demand a tight estimate.  Checkpoints are evaluated
# deepest-first (see evaluate_config) so a configuration that will trip this is
# usually rejected on its first checkpoint, before the other seven are trained.
# Above this hard floor, the *soft* pressure towards a well-calibrated proposal
# comes from objective 3 (see ESS_SOFTMIN_BETA / make_objective).
MIN_EFFECTIVE_SAMPLE_SIZE = 5.0


class ConfigRejected(Exception):
    """A configuration that cannot be scored at a checkpoint (empty support,
    all-invalid weights, no coverage, or ESS below MIN_EFFECTIVE_SAMPLE_SIZE).

    Raised out of :func:`_support_statistics` / :func:`evaluate_checkpoint` and
    caught in :func:`evaluate_config`, which then returns the worst-case
    objective sentinels below.  It is deliberately *not* ``optuna.TrialPruned``:
    a pruned multi-objective trial has ``values = None``, and Optuna 4.9's
    ``_calculate_weights_below_for_multi_objective`` does
    ``np.asarray([t.values for t in below_trials])`` without filtering those
    out, so a single pruned trial in the TPE "below" set raises an inhomogeneous
    -array ``ValueError`` on *every* subsequent ``sample_relative`` call and
    kills the whole study.  A COMPLETE trial pinned to the sentinel teaches TPE
    to avoid the region and never enters that code path.
    """


# Worst-case objective values for a rejected configuration: KS p-value -> 0,
# log10 likelihood-calls/it -> well above any real trial, softmin log10 ESS -> 0
# (ESS ~ 1).  All three are strictly worse than any *scored* trial can be -- a
# scored trial has every checkpoint ESS >= MIN_EFFECTIVE_SAMPLE_SIZE, so its
# softmin log10 ESS is >= log10(5) > 0 -- so a rejected trial is dominated by
# essentially any real one and never reaches the reported Pareto front.  Finite
# so the multi-objective hypervolume maths stays well posed.
REJECT_KS_P_VALUE = 0.0
REJECT_LOG10_LIKELIHOOD_CALLS = 20.0
REJECT_LOG10_ESS_SOFTMIN = 0.0

# Objective 3 rewards a high rejection-sampling effective sample size at *every*
# checkpoint.  A hard ``min`` over checkpoints would chase the noise in a single
# deep-checkpoint estimate, so the aggregate is a smooth minimum in
# ``log10(ESS)`` units:
#
#     softmin_b(v) = -1/b * log( mean_i exp(-b * v_i) )
#
# which tends to ``min(v)`` as ``b -> inf`` and to ``mean(v)`` (the log of the
# geometric-mean ESS) as ``b -> 0``.  ``b = 2`` in log10 units leans towards the
# worst checkpoint without being decided by it.
ESS_SOFTMIN_BETA = 2.0

SEED = 1234

# Flow training is stopped by a plateau rule (see :func:`_install_staged_training`),
# not by nessai's "epochs since the single best validation loss" patience: the
# per-epoch validation NLL is noisy enough (0.5-1 nat swings) that a chance dip
# resets that counter and drags training 3-6x past the point the flow stopped
# improving -- straight into the overfitting regime, which makes the *proposal*
# worse (see the baseline loss-curve experiment).  Instead the flow is trained
# in short bursts and stopped when the trailing-median validation NLL has not
# improved by more than PLATEAU_TOL nats over the last PLATEAU_WINDOW epochs.
# MAX_EPOCHS is only a backstop.
MAX_EPOCHS = 120
TRAIN_CHUNK_EPOCHS = 8         # burst length between plateau checks
PLATEAU_WINDOW = 15           # epochs the improvement is measured over
PLATEAU_TOL = 0.15            # nats; below this over the window -> stop
PLATEAU_SMOOTH = 8            # trailing-median window applied before the test
PLATEAU_MIN_EPOCHS = 24       # never stop (or prune) before this many epochs
# Probe: on the *first* (cold) checkpoint only, the trailing-median validation
# NLL gain from the post-initial-transient level (epochs 3-7) to the plateau is
# recorded as ``probe_drop`` (a per-checkpoint user attr).  A config whose flow
# *diverges* over the probe (gain <= PROBE_DIVERGE_DROP, i.e. the plateau is no
# better than -- or worse than -- epoch 3-7) is rejected; anything that makes
# real progress is kept.  The gain is logged for every first checkpoint so the
# threshold can be tightened once the distribution over real trials is known --
# an earlier, more aggressive "< 0.75 nats" cut rejected ~90% of trials,
# including near-baseline ones, because epochs 1-2 carry most of the drop and
# 24 epochs is not enough tail for a low-lr config.
PROBE_PRUNE_REF_EPOCHS = (2, 7)  # slice used as the reference level
PROBE_DIVERGE_DROP = 0.0

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
# ``patience`` / ``annealing`` are absent on purpose: the plateau rule replaces
# patience, and cosine annealing is incompatible with burst training (each burst
# would restart the schedule -- see :func:`_install_staged_training`).
BASELINE_TRAINING_CONFIG = dict(max_epochs=MAX_EPOCHS)

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
# ``nessai_gw.group_mixture`` provides the discrete symmetry group for a single
# triangular detector such as one Einstein Telescope site: the antipodal /
# quarter-turn sky-and-polarisation degeneracy that makes the archived ET BNS
# posteriors multimodal.  ``--group`` replays with the packaged factory
# ``make_et_group_flow_proposal`` (``phase_reflection=True`` -> the 16-element
# group with ``phase -> phase + pi`` folded in; ``boundary_reflection=True`` ->
# the base flow is symmetrised across the sky octant faces).  It builds a
# ``GWReparamMixin + GroupFlowProposalMixin + FlowProposal`` class whose base
# flow only has to model one orbit representative, with the detector-frame
# rotated sky octant and the unit-shell sky radial coordinate.
#
# It is paired with ``triangular_group_reparameterisations`` (see
# :func:`_group_reparameterisations`), which keeps the acted parameters
# ``(ra, sin_dec, cos_theta_jn, psi, phase, geocent_time)`` isometric in the
# flow's prime space: rotated ``sky-ra-dec`` octant, ``angle-sine`` theta_jn
# with fixed symmetric bounds (so the ``theta_jn -> pi - theta_jn`` reflection
# stays an exact prime sign flip), a single ``polarisation-phase`` coordinate
# ``delta_phase = phase + sign(cos theta_jn) * psi``, and a fixed
# ``reference_time`` shift for ``geocent_time`` (keeps the GPS epoch off the
# flow).  Every other parameter is left to the GW defaults.  The run's noise
# augmentation is dropped (the group proposal has no augmented variant).


def _group_reparameterisations(run: ArchivedRun) -> dict:
    """Reparameterisations for the ``--group`` replay (see the note above).

    Delegates to :func:`nessai_gw.group_mixture.triangular_group_reparameterisations`.
    """
    from nessai_gw.group_mixture import triangular_group_reparameterisations

    return triangular_group_reparameterisations(
        list(run.names), _reference_time_from_run(run)
    )


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


def make_group_proposal_class(run: ArchivedRun):
    """Build a ``FlowProposal`` wrapped in the ET-triangle group mixture.

    Delegates to ``nessai_gw.group_mixture.make_et_group_flow_proposal``, the
    packaged factory that wires the 16-element (``phase_reflection=True``,
    ``phase -> phase + pi`` folded in) ET-EMR group action into a
    ``GWReparamMixin + GroupFlowProposalMixin + FlowProposal`` class, with the
    detector-frame-rotated sky octant, the unit-shell sky radial coordinate and
    the octant-face ``boundary_reflection`` symmetrisation.  Pair it with the
    reparameterisations from :func:`_group_reparameterisations`.
    """
    from nessai_gw.group_mixture import make_et_group_flow_proposal

    return make_et_group_flow_proposal(
        sampling_parameters=list(run.names),
        reference_time=_reference_time_from_run(run),
        phase_reflection=True,
        boundary_reflection=True,
    )


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
        raise ConfigRejected("No latent samples inside the radius")
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
        raise ConfigRejected("No samples survived the backward pass")

    log_w = proposal.compute_weights(x, log_q)
    log_w = log_w[np.isfinite(log_w)]
    if not len(log_w):
        raise ConfigRejected("All rejection-sampling weights were invalid")

    # Kish effective sample size of the importance weights.  Zero-weight draws
    # (dropped above) leave both sums unchanged, so this is computed on the
    # finite weights alone.  When it is small the support-mass sum below is
    # carried by a few latent points and m(S) is not measurable.
    log_ess = 2.0 * logsumexp(log_w) - logsumexp(2.0 * log_w)
    effective_sample_size = float(np.exp(log_ess))

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


def _trailing_median(x, w: int) -> np.ndarray:
    """Causal (trailing) rolling median, same length as ``x``."""
    x = np.asarray(x, dtype=float)
    return np.array(
        [np.nanmedian(x[max(0, i - w + 1) : i + 1]) for i in range(len(x))]
    )


def _install_staged_training(proposal) -> None:
    """Replace ``proposal.flow.train`` with a plateau-stopped burst trainer.

    nessai stops training ``patience`` epochs after the single lowest validation
    loss.  The per-epoch validation NLL here swings 0.5-1 nat, so a chance dip
    keeps resetting that counter and training runs 3-6x too long, into the
    regime where ``val`` rises while ``train`` keeps falling -- overfitting that
    makes the proposal *worse* (baseline loss-curve experiment).

    Instead the flow is trained in bursts of ``TRAIN_CHUNK_EPOCHS`` (weights and
    optimiser state persist across ``FlowModel.train`` calls, so this is just
    warm-started continuation) and stopped when the trailing-median validation
    NLL has not improved by more than ``PLATEAU_TOL`` nats over the last
    ``PLATEAU_WINDOW`` epochs.  ``MAX_EPOCHS`` is a backstop.

    On the first checkpoint of a trial (a cold flow on the widest, easiest live
    set) a config that cannot pull the validation NLL down by
    ``PROBE_PRUNE_MIN_DROP`` nats within ``PLATEAU_MIN_EPOCHS`` epochs is
    rejected via :class:`ConfigRejected` -- it will not be usable at depth, and
    this costs ~24 epochs instead of all ten checkpoints.  The probe is skipped
    for warm-started checkpoints, which legitimately start near their plateau.

    The last burst's epoch count and stop reason are stashed on the proposal
    (``_replay_train_epochs`` / ``_replay_train_stop``) for the checkpoint log.
    """
    real_train = proposal.flow.train
    counter = {"n": 0}

    def staged(samples, output=None, plot=False, **_ignored):
        counter["n"] += 1
        first_checkpoint = counter["n"] == 1
        val_hist: list[float] = []
        train_hist: list[float] = []
        stop_reason = "max_epochs"
        while len(train_hist) < MAX_EPOCHS:
            budget = min(TRAIN_CHUNK_EPOCHS, MAX_EPOCHS - len(train_hist))
            h = real_train(
                samples,
                output=output,
                plot=False,
                max_epochs=budget,
                patience=budget + 1,  # disable nessai's own early stop
            )
            if not len(h["loss"]):
                stop_reason = "inner_stop"
                break
            train_hist.extend(float(v) for v in h["loss"])
            val_hist.extend(float(v) for v in h["val_loss"])
            if not np.all(np.isfinite(train_hist)):
                raise ConfigRejected(
                    f"non-finite training loss after {len(train_hist)} epochs"
                )
            series = np.asarray(val_hist, dtype=float)
            if not np.isfinite(series).any():
                series = np.asarray(train_hist, dtype=float)
            smooth = _trailing_median(series, PLATEAU_SMOOTH)
            if len(smooth) < PLATEAU_MIN_EPOCHS:
                continue
            if first_checkpoint:
                lo, hi = PROBE_PRUNE_REF_EPOCHS
                ref = float(np.nanmedian(series[lo:hi])) if len(series) > lo else np.inf
                drop = ref - float(np.nanmin(smooth))
                proposal._replay_probe_drop = drop
                if np.isfinite(ref) and drop <= PROBE_DIVERGE_DROP:
                    raise ConfigRejected(
                        f"flow diverging: trailing-median validation NLL did not "
                        f"improve over epochs {lo + 1}-{len(smooth)} "
                        f"(gain {drop:.2f} nats <= {PROBE_DIVERGE_DROP})"
                    )
            if len(smooth) > PLATEAU_WINDOW:
                gain = float(
                    np.nanmin(smooth[:-PLATEAU_WINDOW])
                    - np.nanmin(smooth[-PLATEAU_WINDOW:])
                )
                if gain < PLATEAU_TOL:
                    stop_reason = "plateau"
                    break
        proposal._replay_train_epochs = len(train_hist)
        proposal._replay_train_stop = stop_reason
        # The single cost function: the lowest validation NLL this burst
        # reached (falls back to the training NLL if validation was all-NaN).
        val_series = np.asarray(val_hist, dtype=float)
        if not np.isfinite(val_series).any():
            val_series = np.asarray(train_hist, dtype=float)
        proposal._replay_val_min = (
            float(np.nanmin(val_series))
            if np.isfinite(val_series).any()
            else float("nan")
        )
        return {"loss": train_hist, "val_loss": val_hist}

    proposal.flow.train = staged


def _build_trial_proposal(
    run: ArchivedRun,
    model: ReplayModel,
    flow_config: dict,
    training_config: dict,
    proposal_class,
    rng: np.random.Generator,
    output: str,
    proposal_overrides: dict | None = None,
    use_group: bool = False,
):
    """Construct and initialise the proposal a trial reuses across checkpoints.

    One proposal is built per trial and retrained at each checkpoint (see
    :func:`evaluate_config`), so the flow warm-starts from the previous
    checkpoint's weights -- ``FlowModel.train`` does not reset the model or the
    optimiser between calls -- exactly as a real nessai run reuses the flow
    between iterations until ``reset_flow``.  Training itself is plateau-stopped
    (:func:`_install_staged_training`).
    """
    if proposal_overrides is not None:
        proposal_kwargs = _apply_proposal_overrides(
            run.proposal_kwargs,
            proposal_overrides,
            names=run.names,
        )
    else:
        proposal_kwargs = run.proposal_kwargs
    if use_group:
        # The group proposal supplies its own GW reparameterisations by name
        # (see ``_group_reparameterisations`` / the nessai-gw factory); the
        # ``map_to_unit_hypercube`` CDF prior isn't used on that path.
        proposal_kwargs = {
            k: v
            for k, v in proposal_kwargs.items()
            if k != "map_to_unit_hypercube"
        }
        proposal_kwargs["reparameterisations"] = _group_reparameterisations(run)
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
    _install_staged_training(proposal)
    return proposal


def evaluate_checkpoint(
    run: ArchivedRun,
    model: ReplayModel,
    iteration: int,
    proposal,
    rng: np.random.Generator,
) -> dict:
    """Retrain ``proposal`` on the live set at ``iteration`` and record the
    flow's minimum validation loss.

    ``proposal`` is the per-trial object from :func:`_build_trial_proposal`;
    calling ``train`` again warm-starts the flow from the previous checkpoint.
    Only training is done here -- the coverage / support-mass / insertion-index
    scoring of the multi-objective script is not needed for the single cost.
    """
    indices = run.live_indices(iteration)
    # Train on the full live set, exactly as a real run does; nessai holds out
    # its own ``val_size`` fraction internally for the validation NLL.
    x_train = run.live_points(np.sort(indices), model)

    start = time.perf_counter()
    proposal.train(x_train, plot=False)
    train_time = time.perf_counter() - start

    n_params = sum(
        p.numel() for p in proposal.flow.model.parameters() if p.requires_grad
    )

    return dict(
        iteration=int(iteration),
        val_loss_min=float(getattr(proposal, "_replay_val_min", float("nan"))),
        train_time=train_time,
        n_train_epochs=int(getattr(proposal, "_replay_train_epochs", -1)),
        train_stop=str(getattr(proposal, "_replay_train_stop", "?")),
        probe_drop=float(getattr(proposal, "_replay_probe_drop", float("nan"))),
        n_flow_parameters=int(n_params),
    )


def _val_loss_softmax(val_losses: np.ndarray) -> float:
    """Smooth maximum of the per-checkpoint validation loss (see
    VAL_LOSS_SOFTMAX_BETA).

    ``1/b * log(mean_i exp(b * v_i))``: between ``max`` and ``mean`` of the
    per-checkpoint minimum validation NLL, leaning towards the worst (deepest)
    checkpoint.
    """
    v = np.asarray(val_losses, dtype=float)
    b = VAL_LOSS_SOFTMAX_BETA
    return float((logsumexp(b * v) - np.log(len(v))) / b)


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

    One flow is built for the trial (:func:`_build_trial_proposal`) and
    retrained at each checkpoint, visited in iteration order (shallow to deep)
    so the flow warm-starts from the previous checkpoint -- the same trajectory
    a real run's flow follows.  The single cost is the soft-max over checkpoints
    of the flow's minimum validation loss (:func:`_val_loss_softmax`).  A
    :class:`ConfigRejected` from any checkpoint (the first-checkpoint divergence
    probe) stops the sweep and the summary comes back flagged ``rejected`` with
    the sentinel cost :data:`REJECT_VAL_LOSS` (see :func:`make_objective`).
    """
    rng = np.random.default_rng(seed)
    results = []
    with tempfile.TemporaryDirectory(prefix="nessai-replay-") as output:
        proposal = _build_trial_proposal(
            run,
            model,
            flow_config,
            training_config,
            proposal_class,
            rng,
            os.path.join(output, "flow"),
            proposal_overrides=proposal_overrides,
            use_group=use_group,
        )
        for iteration in run.checkpoints(N_CHECKPOINTS):
            try:
                result = evaluate_checkpoint(
                    run, model, iteration, proposal, rng
                )
            except ConfigRejected as exc:
                return dict(
                    rejected=True,
                    reject_reason=str(exc),
                    reject_iteration=int(iteration),
                    n_checkpoints_evaluated=len(results),
                    val_loss_softmax=REJECT_VAL_LOSS,
                    val_loss_max=float(
                        max(
                            [r["val_loss_min"] for r in results],
                            default=REJECT_VAL_LOSS,
                        )
                    ),
                )
            results.append(result)
            logger.info(
                "  trial=%s it=%-7d val_loss_min=%.3f  train=%.1fs (%dep, %s)",
                "?" if trial is None else trial.number,
                result["iteration"],
                result["val_loss_min"],
                result["train_time"],
                result["n_train_epochs"],
                result["train_stop"],
            )
            if trial is not None:
                trial.set_user_attr(
                    f"checkpoint_{iteration}", dict(result)
                )

    val_losses = [r["val_loss_min"] for r in results]
    return dict(
        # The single cost: soft-max over checkpoints of the minimum val loss.
        val_loss_softmax=_val_loss_softmax(val_losses),
        val_loss_mean=float(np.mean(val_losses)),
        val_loss_max=float(np.max(val_losses)),
        val_loss_per_checkpoint=[float(v) for v in val_losses],
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
        n_blocks=trial.suggest_int("n_blocks", 2, 10),
        n_layers=trial.suggest_int("n_layers", 1, 4),
        n_neurons=trial.suggest_int("n_neurons", 16, 128, log=True),
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
        flow_config["num_bins"] = trial.suggest_int("num_bins", 4, 8)
        flow_config["tail_bound"] = trial.suggest_float(
            "tail_bound", 3.0, 10.0
        )

    training_config = dict(
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        batch_size=trial.suggest_categorical(
            "batch_size", [500, 1000, 2000]
        ),
        # No ``patience`` / ``annealing``: the plateau rule in
        # :func:`_install_staged_training` stops training, and cosine annealing
        # would restart its schedule on every burst.
        max_epochs=MAX_EPOCHS,
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
        # direct control on the fidelity/cost tradeoff objectives 1 and 2
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
        if summary.get("rejected"):
            logger.info(
                "Trial %s: REJECTED at it=%s (%s) -> cost %.3g",
                trial.number,
                summary.get("reject_iteration"),
                summary.get("reject_reason"),
                REJECT_VAL_LOSS,
            )
            return REJECT_VAL_LOSS
        logger.info(
            "Trial %s: val_loss softmax=%.4f (mean=%.4f, max=%.4f) "
            "per-checkpoint=%s",
            trial.number,
            summary["val_loss_softmax"],
            summary["val_loss_mean"],
            summary["val_loss_max"],
            ["%.3f" % v for v in summary["val_loss_per_checkpoint"]],
        )
        return summary["val_loss_softmax"]

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
    sampler_seed: int = SEED,
) -> None:
    """Entry point for one ``--n-jobs`` worker process.

    Runs ``n_trials`` against the study in ``STORAGE``, which every worker
    (and the parent process) shares.  Each worker reloads the archived run
    itself rather than inheriting it from the parent, since the process pool
    uses the ``spawn`` start method (safer with PyTorch than ``fork``) and so
    nothing is inherited across the fork boundary anyway.

    ``sampler_seed`` is distinct per worker (see :func:`_make_sampler`): with a
    shared seed every worker proposes identical configurations through TPE's
    random startup phase.
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
        proposal_class = make_group_proposal_class(run)
    else:
        proposal_class = get_proposal_class(
            augmented=bool(run.augment_kwargs), non_gw=non_gw
        )

    study = optuna.load_study(
        study_name=study_name,
        storage=_make_storage(storage),
        sampler=_make_sampler(sampler_seed),
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
            "Replay with nessai-gw's 16-element ET-triangle symmetry group "
            "mixture (nessai_gw.group_mixture.make_et_group_flow_proposal, "
            "phase_reflection=True, boundary_reflection=True) paired with "
            "triangular_group_reparameterisations.  Drops the run's noise "
            "augmentation.  Unless --storage / --study-name are given, a "
            f"separate study ({GROUP_STUDY_NAME} in {GROUP_STORAGE}) is used."
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
        direction="minimize",
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
                    # Distinct per worker: a shared sampler seed makes every
                    # worker propose identical configs through TPE startup.
                    SEED + 1 + i,
                ),
            )
            for i, n in enumerate(counts)
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

    complete = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    print("\nBest trials (softmax-over-checkpoints minimum validation loss):")
    for t in sorted(complete, key=lambda t: t.value)[:10]:
        vc = t.user_attrs.get("val_loss_per_checkpoint")
        print(
            f"  trial {t.number:4d}  cost={t.value:.4f}  "
            f"(mean={t.user_attrs.get('val_loss_mean', float('nan')):.4f}, "
            f"max={t.user_attrs.get('val_loss_max', float('nan')):.4f})  "
            f"{t.params}"
        )
        if vc:
            print(f"             per-checkpoint: {['%.3f' % v for v in vc]}")


if __name__ == "__main__":
    main()
