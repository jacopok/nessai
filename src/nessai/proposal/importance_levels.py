# -*- coding: utf-8 -*-
"""
Importance nested sampling with the standard sampler's flow proposals.

:class:`LevelFlowProposal` is a proposal for
:class:`~nessai.samplers.importancesampler.ImportanceNestedSampler` whose
levels are instances of a :class:`~nessai.proposal.flowproposal.FlowProposal`
class (any subclass: its reparameterisations, flows and group mixtures), in
place of :class:`~nessai.proposal.importance.ImportanceFlowProposal`'s flows
on the logit of the unit hypercube.

Each level is a frozen copy of the proposal, trained on the samples above the
level's likelihood threshold weighted by their importance weights, and kept
for the rest of the run: the meta-proposal is the mixture of the prior and
every level. The importance sampler needs each component's exact, normalised
density in the unit hypercube; for a level:

* the density of the physical point ``x`` is the proposal's draw density
  (:meth:`FlowProposal.mirror_log_q`: the flow's density in its prime space
  with the reparameterisations' Jacobians, summed over the sides of a
  boundary inversion), at the latent temperature the level draws with;
* it is truncated to the prior support (draws outside the prior bounds or
  with a zero prior are rejected) and divided by the mass of the support,
  measured from ``n_pilot`` draws after the training (no likelihood calls);
* it is mapped to the unit hypercube with the Jacobian of the prior's
  marginal CDFs, ``sum_i ln pi_i(x_i)``.

The proposal's pool machinery (truncation, weight caps, latent radius,
rejection against the prior) is not used: draws come from the flow itself.
"""

import copy
import datetime
import logging
import os
from typing import Optional

import numpy as np
from scipy.special import logsumexp

from .. import config
from ..livepoint import (
    empty_structured_array,
    get_dtype,
    live_points_to_array,
)
from .base import Proposal

logger = logging.getLogger(__name__)

#: points scored per call of a level's density
_CHUNK = 20_000


def _ess(weights):
    """Kish effective sample size."""
    return float(weights.sum() ** 2 / (weights**2).sum())


#: a prime point maps back to the physical point it came from within this
#: fraction of each parameter's prior range (else it is another point's copy)
_ROUND_TRIP = 1e-6


class LevelFlowProposal(Proposal):
    """Meta-proposal of frozen :class:`FlowProposal` levels for the
    importance nested sampler.

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
        The model. Its unit-hypercube map must be the prior's marginal CDFs
        (bilby's priors in nessai-bilby), see :meth:`log_jacobian`.
    output : str
        Output directory; level ``k`` writes into ``level_k``.
    rng : numpy.random.Generator, optional
        Random number generator.
    flow_proposal_class : str or type
        The :class:`FlowProposal` class, as ``flow_proposal_class`` of
        :class:`~nessai.samplers.nestedsampler.NestedSampler`.
    flow_config, training_config : dict, optional
        The flow and training configurations of the proposal.
    flow_proposal_kwargs : dict, optional
        Keyword arguments of the proposal class.
    reset_flow : int
        Train every ``reset_flow``-th level from a fresh copy of the
        untrained proposal; the others start from the previous level's
        trained proposal (warm). 0: always warm.
    n_pilot : int
        Draws measuring each level's mass inside the prior support.
    weighted_training : bool
        Train on the samples' importance weights (``logW``) when the sampler
        gives none.
    training_ess_fraction : float, optional
        Temper the training weights to ``w ** alpha``, with the largest
        ``alpha`` in ``[0, 1]`` that keeps their effective sample size at
        this fraction of the samples (none: ``alpha = 1``). The meta-proposal's
        densities, and so the importance weights of the samples, are exact
        whatever a level is trained on; a level trained on a few heavy
        samples is narrow and misses the tails those came from, which makes
        the next training set heavier still.
    """

    def __init__(
        self,
        model,
        output: str,
        rng: Optional[np.random.Generator] = None,
        flow_proposal_class=None,
        flow_config: Optional[dict] = None,
        training_config: Optional[dict] = None,
        flow_proposal_kwargs: Optional[dict] = None,
        reset_flow: int = 0,
        n_pilot: int = 20_000,
        weighted_training: bool = True,
        training_ess_fraction: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(model, rng=rng)
        if kwargs:
            logger.debug(
                "LevelFlowProposal ignores the keyword arguments of "
                f"ImportanceFlowProposal: {sorted(kwargs)}"
            )
        self.output = output
        self.flow_proposal_class = flow_proposal_class
        self._level_flow_config = copy.deepcopy(flow_config)
        self.training_config = copy.deepcopy(training_config)
        self.flow_proposal_kwargs = dict(flow_proposal_kwargs or {})
        self.reset_flow = int(reset_flow)
        self.n_pilot = int(n_pilot)
        self.weighted_training = weighted_training
        self.training_ess_fraction = training_ess_fraction
        self.level_count = -1
        self._weights = {-1: 1.0}
        self.dtype = get_dtype(self.model.names)
        self.template = None
        self.levels = []
        self.log_mass = []
        self.level_history = []
        self.scoring_time = datetime.timedelta()

    # The importance sampler's interface
    # ----------------------------------

    @property
    def flow_config(self) -> dict:
        return self._level_flow_config

    @property
    def weights(self) -> dict:
        """Weights of the prior (-1) and of each level in the meta-proposal."""
        return self._weights

    @property
    def weights_array(self) -> np.ndarray:
        return np.fromiter(self._weights.values(), dtype=float)

    @property
    def n_proposals(self) -> int:
        return len(self._weights)

    def _check_fields(self):
        for name in ("logQ", "logW", "logU"):
            if name not in config.livepoints.non_sampling_parameters:
                raise RuntimeError(
                    f"{name} field missing in non-sampling parameters."
                )

    def initialise(self) -> None:
        """Build the untrained proposal every cold level is copied from."""
        self._check_fields()
        if self.initialised:
            return
        if self.template is None:
            self.template = self._new_template()
        self.initialised = True

    def _new_template(self):
        from .utils import check_proposal_kwargs, get_flow_proposal_class

        ProposalClass = get_flow_proposal_class(self.flow_proposal_class)
        kwargs = check_proposal_kwargs(
            ProposalClass, dict(self.flow_proposal_kwargs)
        )
        # the pool is never used, but the class needs a size
        if kwargs.get("poolsize") is None:
            kwargs["poolsize"] = 10_000
        output = os.path.join(self.output, "template", "")
        os.makedirs(output, exist_ok=True)
        logger.info(f"Levels: {ProposalClass.__name__} with {sorted(kwargs)}")
        proposal = ProposalClass(
            self.model,
            rng=self._child_rng(),
            flow_config=copy.deepcopy(self._level_flow_config),
            training_config=copy.deepcopy(self.training_config),
            output=output,
            plot=False,
            **kwargs,
        )
        proposal.initialise()
        if getattr(proposal, "map_to_unit_hypercube", False):
            raise ValueError(
                "LevelFlowProposal: map_to_unit_hypercube is not supported"
            )
        return proposal

    def _child_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.rng.integers(2**63))

    def update_output(self, output: str) -> None:
        super().update_output(output)
        for k, prop in enumerate(self.levels):
            prop.output = os.path.join(output, f"level_{k}", "")

    def update_proposal_weights(self, weights: dict) -> None:
        self._weights.update(weights)
        w_sum = np.sum(np.fromiter(self._weights.values(), float))
        if not np.isclose(w_sum, 1.0):
            raise RuntimeError(f"Weights must sum to 1! Actual value: {w_sum}")

    # Levels
    # ------

    def _copy(self, source, output):
        """A copy of ``source`` writing into ``output``, sharing the model.

        The proposal and its flow model leave the model, the flow, the torch
        module and the optimiser out of their pickled state (and so out of
        a deepcopy): those are copied (or rebuilt) here.
        """
        memo = {id(self.model): self.model}
        flow = source.flow
        prop = copy.deepcopy(source, memo)
        new_flow = copy.deepcopy(flow, memo)
        new_flow.model = copy.deepcopy(flow.model, memo)
        new_flow.flow_config = copy.deepcopy(flow.flow_config, memo)
        new_flow._optimiser = new_flow.get_optimiser()
        new_flow.initialised = True
        prop.model = self.model
        prop.flow = new_flow
        prop._flow_config = copy.deepcopy(source._flow_config, memo)
        prop.initialised = True
        prop.rng = self._child_rng()
        os.makedirs(output, exist_ok=True)
        prop.output = output
        return prop

    def _cold(self) -> bool:
        k = self.level_count
        return not self.levels or (self.reset_flow and not k % self.reset_flow)

    def train(
        self,
        samples: np.ndarray,
        plot: bool = False,
        output: Optional[str] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """Add a level trained on ``samples`` (unit hypercube), weighted by
        ``weights`` or, with ``weighted_training``, by their ``logW``."""
        self.level_count += 1
        self._weights[self.level_count] = np.nan
        output = self.output if output is None else output
        level_output = os.path.join(output, f"level_{self.level_count}", "")
        cold = self._cold()
        prop = self._copy(
            self.template if cold else self.levels[-1], level_output
        )
        x = self.model.from_unit_hypercube(samples.copy())
        x["logP"] = self.model.batch_evaluate_log_prior(x)
        if weights is None and self.weighted_training:
            log_w = samples["logW"] - logsumexp(samples["logW"])
            weights = np.exp(log_w)
        alpha = 1.0
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if not np.all(np.isfinite(weights)):
                raise ValueError("Training weights are not finite")
            ess_full = _ess(weights)
            weights, alpha = self._temper(weights)
            weights = weights / weights.mean()
        st = datetime.datetime.now()
        self._fit(prop, x, weights)
        train_time = datetime.datetime.now() - st
        self._check_level(prop)
        self._clear(prop)
        log_mass, n_support = self._log_support_mass(prop)
        self.levels.append(prop)
        self.log_mass.append(log_mass)
        ess = len(x) if weights is None else _ess(weights)
        if weights is None:
            ess_full = ess
        self.level_history.append(
            dict(
                level=self.level_count,
                cold=bool(cold),
                n_train=len(x),
                ess_train=float(ess),
                ess_weights=float(ess_full),
                alpha=float(alpha),
                log_mass=float(log_mass),
                train_seconds=train_time.total_seconds(),
            )
        )
        tempered = f", tempered {alpha:.2f}: {ess:.0f}" if alpha < 1 else ""
        logger.info(
            f"Level {self.level_count}: {'cold' if cold else 'warm'}, "
            f"{len(x)} samples (ESS {ess_full:.0f}{tempered}), trained in "
            f"{train_time.total_seconds():.0f} s, support mass "
            f"{np.exp(log_mass):.4f} ({n_support} of {self.n_pilot})"
        )

    def _temper(self, weights):
        """``weights ** alpha`` with the largest ``alpha`` in ``[0, 1]``
        keeping their ESS at ``training_ess_fraction`` of their number."""
        target = self.training_ess_fraction
        if not target or _ess(weights) >= target * len(weights):
            return weights, 1.0
        log_w = np.log(np.maximum(weights, np.finfo(float).tiny))
        log_w -= log_w.max()
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if _ess(np.exp(mid * log_w)) >= target * len(weights):
                lo = mid
            else:
                hi = mid
        return np.exp(lo * log_w), lo

    @staticmethod
    def _fit(prop, x, weights):
        """Train ``prop`` on ``x`` with per-sample ``weights`` (repeated over
        a boundary inversion's mirror copies, which the rescale appends in
        blocks of ``len(x)`` rows): the flow's loss, and the group mixtures'
        element weights."""
        flow = prop.flow
        own = "train" in flow.__dict__
        train = flow.train

        def weighted_train(data, *args, **kwargs):
            if weights is not None:
                kwargs["weights"] = np.tile(weights, len(data) // len(x))
            return train(data, *args, **kwargs)

        flow.train = weighted_train
        prop.current_live_points = x
        # the group mixtures' element weights count the points by these
        # (nessai.flowmodel.group_mixture.GroupFlowProposalMixin.check_state)
        prop.training_sample_weights = weights
        try:
            prop.train(x, plot=False)
        finally:
            if own:
                flow.train = train
            else:
                del flow.__dict__["train"]
            prop.current_live_points = None
            prop.training_sample_weights = None

    @staticmethod
    def _clear(prop):
        """Drop the training data and pools a frozen level keeps."""
        for name in ("training_data", "training_data_prime", "x", "samples"):
            if hasattr(prop, name):
                setattr(prop, name, None)
        prop.indices = []
        prop.populated = False

    def _draw_physical(self, prop, n):
        """``n`` latent draws of ``prop`` pushed to the physical space: the
        ones inside the prior support (structured, ``logP`` set)."""
        z = prop.sample_latent_distribution(n)
        x = prop.backward_pass(z, rescale=True)[0]
        if not len(x):
            return x
        x = x[np.isfinite(live_points_to_array(x, self.model.names)).all(1)]
        x["logP"] = self.model.batch_evaluate_log_prior(x)
        return x[np.isfinite(x["logP"])]

    def _log_support_mass(self, prop):
        """ln of the level's draws' share inside the prior support."""
        kept, made = 0, 0
        while made < self.n_pilot:
            m = min(_CHUNK, self.n_pilot - made)
            kept += len(self._draw_physical(prop, m))
            made += m
        if not kept:
            raise RuntimeError(
                f"Level {self.level_count}: no draw inside the prior support"
            )
        return float(np.log(kept / made)), kept

    # Densities
    # ---------

    def log_jacobian(self, x: np.ndarray) -> np.ndarray:
        """``ln |du/dx|`` of the unit-hypercube map at the physical points
        ``x``: the sum of the marginal priors' log densities."""
        priors = getattr(self.model, "bilby_priors", None)
        if priors is None:
            fn = getattr(self.model, "log_unit_hypercube_jacobian", None)
            if fn is None:
                raise RuntimeError(
                    "LevelFlowProposal needs the model's unit-hypercube "
                    "Jacobian: bilby_priors or log_unit_hypercube_jacobian"
                )
            return np.asarray(fn(x), dtype=float)
        log_j = np.zeros(len(x))
        for name in self.model.names:
            log_j += np.asarray(priors[name].ln_prob(x[name]), dtype=float)
        return log_j

    def _physical(self, samples):
        """Physical points of unit-hypercube samples, and ``ln |du/dx|``."""
        x = self.model.from_unit_hypercube(samples.copy())
        return x, self.log_jacobian(x)

    def _level_log_q_physical(self, k, x):
        """ln of level ``k``'s normalised density at the physical points
        ``x`` (without the unit-hypercube Jacobian)."""
        prop = self.levels[k]
        out = np.empty(len(x))
        for start in range(0, len(x), _CHUNK):
            xc = x[start : start + _CHUNK].copy()
            n = len(xc)
            if n == 1:
                # some reparameterisations' table lookups fail on one point
                xc = np.concatenate([xc, xc])
            with np.errstate(invalid="ignore", divide="ignore"):
                lq = self._preimage_log_q(prop, xc)[:n]
            out[start : start + n] = np.where(np.isnan(lq), -np.inf, lq)
        inside = self.model.in_bounds(x)
        out[~inside] = -np.inf
        return out - self.log_mass[k]

    def _preimage_log_q(self, prop, x):
        """The draw density of ``prop`` at the physical points ``x``: the
        flow's density summed over the prime-space pre-images of each point,
        with the rescale's Jacobian.

        The rescale gives one prime point per copy it makes (the mirror
        copies of a boundary inversion with duplication, in blocks of
        ``len(x)`` rows); a copy counts only if the inverse rescale maps it
        back to the point. A boundary inversion reflects about the bound of
        its training data, so the prior's range beyond it is never drawn:
        there the point's own copy maps back to its reflection instead, and
        the density is zero.
        """
        n = len(x)
        x_prime, log_j = prop.rescale(x.copy())
        arr = live_points_to_array(
            x_prime, names=prop.prime_parameters, copy=True
        )
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        if len(arr) % n:
            raise RuntimeError(
                f"{len(arr)} prime points for {n} points: not whole blocks "
                "of mirror copies"
            )
        k = len(arr) // n
        log_q = np.full(len(arr), -np.inf)
        finite = np.isfinite(arr).all(axis=1) & np.isfinite(log_j)
        if finite.any():
            log_q[finite] = (
                np.asarray(prop.flow.log_prob(arr[finite]), dtype=float)
                + np.asarray(log_j, dtype=float)[finite]
            )
        back, _ = prop.inverse_rescale(x_prime.copy())
        same = np.ones(len(arr), dtype=bool)
        for name in self.model.names:
            lo, hi = self.model.bounds[name]
            width = hi - lo
            # a periodic angle may come back a period away
            d = np.abs(back[name] - np.tile(x[name], k))
            d = np.minimum(d, np.abs(d - width))
            same &= d <= _ROUND_TRIP * width
        log_q[~same] = -np.inf
        return logsumexp(log_q.reshape(k, n), axis=0)

    def _check_level(self, prop):
        """Settings under which a level's density is not its draws'."""
        temperature = getattr(prop, "latent_temperature", None)
        if temperature not in (None, 1.0):
            raise ValueError(
                "LevelFlowProposal needs latent_temperature = 1 (the density "
                f"is scored untempered), not {temperature}"
            )
        for reparam in getattr(prop, "_reparameterisation", {}).values():
            inversion = getattr(reparam, "boundary_inversion", None) or {}
            if isinstance(inversion, dict) and "split" in inversion.values():
                raise ValueError(
                    f"{type(reparam).__name__}: a boundary inversion of "
                    "type 'split' scores one side only; use 'duplicate'"
                )

    def _log_q_levels(self, x, log_j, levels):
        """Columns of ``ln q_k`` in the unit hypercube for ``levels``."""
        st = datetime.datetime.now()
        out = np.empty((len(x), len(levels)))
        for i, k in enumerate(levels):
            out[:, i] = self._level_log_q_physical(k, x) - log_j
        self.scoring_time += datetime.datetime.now() - st
        return out

    def _log_q_all(self, samples):
        """``ln q`` of every component (prior first) at unit-hypercube
        samples."""
        log_q = np.zeros((len(samples), self.n_proposals))
        if self.levels:
            x, log_j = self._physical(samples)
            log_q[:, 1:] = self._log_q_levels(
                x, log_j, range(len(self.levels))
            )
        return log_q

    def compute_meta_proposal_from_log_q(self, log_q):
        return logsumexp(log_q, b=self.weights_array, axis=1)

    def compute_meta_proposal_samples(self, samples):
        if any(np.isnan(w) for w in self._weights.values()):
            raise RuntimeError(f"Weights are not set: {self._weights}")
        log_q = self._log_q_all(samples)
        return self.compute_meta_proposal_from_log_q(log_q), log_q

    def update_log_q(self, samples, log_q):
        """``log_q`` with the current level's column appended."""
        if log_q.shape[1] == self.n_proposals:
            raise ValueError("log_q array already contains current proposal")
        x, log_j = self._physical(samples)
        new = self._log_q_levels(x, log_j, [self.level_count])
        return np.concatenate([log_q, new], axis=1)

    # Drawing
    # -------

    def _to_unit(self, x):
        """Unit-hypercube samples (the sampler's dtype) of physical points."""
        u = self.model.to_unit_hypercube(x)
        samples = empty_structured_array(len(x), dtype=self.dtype)
        for name in self.model.names:
            samples[name] = u[name]
        samples = samples[self.model.in_unit_hypercube(samples)]
        # Drop the draws whose round trip lands where a marginal prior's
        # density vanishes (e.g. a Sine prior at 0, an aligned-spin prior at
        # +-chi_max): there ln q in the hypercube is -inf - -inf, undefined.
        # The set has measure ~1e-16, so no level's normalisation changes.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_j = self._physical(samples)[1]
        return samples[np.isfinite(log_j)]

    def _draw_level(self, k, n):
        """``n`` unit-hypercube samples from level ``k`` (truncated to the
        prior support), without their densities."""
        prop = self.levels[k]
        rate = max(np.exp(self.log_mass[k]), 1e-3)
        out = empty_structured_array(0, dtype=self.dtype)
        while len(out) < n:
            m = int(min(max(1.02 * (n - len(out)) / rate, 100), 1e6))
            x = self._draw_physical(prop, m)
            if len(x):
                out = np.concatenate([out, self._to_unit(x)])
        return out[:n]

    def _finish(self, samples):
        """Densities, prior and weights of unit-hypercube samples."""
        log_Q, log_q = self.compute_meta_proposal_samples(samples)
        samples["logQ"] = log_Q
        samples["logP"] = self.model.batch_evaluate_log_prior(
            samples, unit_hypercube=True
        )
        samples["logU"] = self.model.batch_evaluate_log_prior_unit_hypercube(
            samples
        )
        samples["logW"] = samples["logU"] - samples["logQ"]
        if np.isposinf(samples["logW"]).any() or np.isnan(log_Q).any():
            raise RuntimeError(
                "Samples with an infinite weight or an undefined density: "
                "a draw outside its own level's support"
            )
        return samples, log_q

    def draw(self, n: int, flow_number: Optional[int] = None, **kwargs):
        """``n`` samples from level ``flow_number`` (default: the latest),
        with ``logQ``, ``logP``, ``logU``, ``logW`` and every component's
        ``ln q``."""
        k = self.level_count if flow_number is None else flow_number
        return self._finish(self._draw_level(k, n))

    def draw_from_prior(self, n: int):
        samples = self.model.sample_unit_hypercube(n)
        samples = self._finish(samples)
        return samples

    def draw_from_flows(self, n: int, weights=None, counts=None):
        """``n`` samples from the meta-proposal (multinomial counts from
        ``weights``, default the current ones, unless ``counts``)."""
        if counts is None:
            weights = self.weights_array if weights is None else weights
            weights = np.asarray(weights, float) / np.sum(weights)
            counts = self.rng.multinomial(n, weights)
        counts = np.asarray(counts, dtype=int)
        parts = []
        for i, m in enumerate(counts):
            if not m:
                continue
            if i == 0:
                part = self.model.sample_unit_hypercube(m)
            else:
                part = self._draw_level(i - 1, m)
            part["it"] = i - 1
            parts.append(part)
        samples = np.concatenate(parts)
        samples, log_q = self._finish(samples)
        return samples, log_q, counts

    # Checkpointing
    # -------------

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["model"]
        return state

    def resume(
        self, model, flow_config=None, training_config=None, weights_path=None
    ):
        """Resume the template and every level with their own weights."""
        self.model = model
        for prop in [self.template] + self.levels:
            if prop is None:
                continue
            prop.resume(model, copy.deepcopy(self._level_flow_config))
        self.initialised = self.template is not None
