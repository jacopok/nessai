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
import torch
from scipy.special import logsumexp

from .. import config
from ..livepoint import (
    empty_structured_array,
    get_dtype,
    live_points_to_array,
)
from ..flowmodel.heavy_kernel import HeavyKernel, KernelMixFlow
from .base import Proposal

logger = logging.getLogger(__name__)

#: points scored per call of a level's density
_CHUNK = 20_000
#: the fewest pilot draws inside a level's prime box for the box to be kept
_MIN_BOX_SUPPORT = 100


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
    kernel_threshold : float, optional
        Heavy-sample kernels (none: off). Training samples holding more than
        this share of the training weight (``logW``: the constrained prior
        above the threshold, the level's target) mark where the
        meta-proposal is too thin; the level then
        draws a fraction ``kernel_fraction`` of its samples from a mixture
        of multivariate Student-t kernels centred on them (at most
        ``kernel_max``) in the flow's coordinates (a group mixture's
        standardised base frame, else its prime space), each with the
        covariance of its ``kernel_neighbours`` nearest training samples
        there. The flow's density becomes ``(1 - f) q + f K``: exact, so the
        draws keep their weights, and they join the next level's training
        set.
    """

    #: the full mixture, also for checkpoints from before the option
    mixture_block = None
    defensive_fraction = 0.0
    #: no prime-space box (levels from before the option)
    box_sigma = None
    prime_boxes = ()
    #: no heavy-sample kernels (levels from before the option)
    kernel_threshold = None
    kernels = ()

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
        mixture_block: Optional[int] = None,
        defensive_fraction: float = 0.0,
        box_sigma: Optional[float] = None,
        kernel_threshold: Optional[float] = None,
        kernel_fraction: float = 0.1,
        kernel_max: int = 5,
        kernel_neighbours: int = 100,
        kernel_dof: float = 4.0,
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
        if mixture_block is not None and int(mixture_block) < 1:
            raise ValueError(f"mixture_block must be >= 1: {mixture_block}")
        self.mixture_block = (
            None if mixture_block is None else int(mixture_block)
        )
        if not 0.0 <= defensive_fraction < 1.0:
            raise ValueError(
                f"defensive_fraction must be in [0, 1): {defensive_fraction}"
            )
        self.defensive_fraction = float(defensive_fraction)
        if box_sigma is not None and not box_sigma > 0:
            raise ValueError(f"box_sigma must be > 0: {box_sigma}")
        self.box_sigma = None if box_sigma is None else float(box_sigma)
        self.prime_boxes = []
        if kernel_threshold is not None and not 0 < kernel_threshold < 1:
            raise ValueError(
                f"kernel_threshold must be in (0, 1): {kernel_threshold}"
            )
        if not 0 < kernel_fraction < 1:
            raise ValueError(
                f"kernel_fraction must be in (0, 1): {kernel_fraction}"
            )
        self.kernel_threshold = (
            None if kernel_threshold is None else float(kernel_threshold)
        )
        self.kernel_fraction = float(kernel_fraction)
        self.kernel_max = int(kernel_max)
        self.kernel_neighbours = int(kernel_neighbours)
        self.kernel_dof = float(kernel_dof)
        self.kernels = []
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
        self._detach_kernel(prop)
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
        kernel = self._heavy_kernel(
            prop,
            samples,
            x,
            kwargs.get("all_samples"),
            kwargs.get("log_likelihood_threshold"),
        )
        self._attach_kernel(prop, kernel)
        box = self._prime_box(prop, x, weights)
        log_mass, n_support = self._log_support_mass(prop, box, strict=box is None)
        if box is not None and n_support < _MIN_BOX_SUPPORT:
            # a box from a handful of effective training samples (or a reset
            # expert's broad flow) can hold almost none of the level's draws:
            # its mass would be a noisy estimate from a few draws, or none
            logger.warning(
                f"Level {self.level_count}: {n_support} of {self.n_pilot} "
                f"draws inside the box: the level without it"
            )
            box = None
            log_mass, n_support = self._log_support_mass(prop, box)
        self.levels.append(prop)
        self.prime_boxes = list(self.prime_boxes)
        while len(self.prime_boxes) < len(self.levels) - 1:
            self.prime_boxes.append(None)
        self.prime_boxes.append(box)
        self.kernels = list(self.kernels)
        while len(self.kernels) < len(self.levels) - 1:
            self.kernels.append(None)
        self.kernels.append(kernel)
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
                n_kernels=0 if kernel is None else sum(
                    len(p.centres) for _, p in kernel["parts"]
                ),
                kernel_shares=[] if kernel is None else kernel["shares"],
            )
        )
        tempered = f", tempered {alpha:.2f}: {ess:.0f}" if alpha < 1 else ""
        logger.info(
            f"Level {self.level_count}: {'cold' if cold else 'warm'}, "
            f"{len(x)} samples (ESS {ess_full:.0f}{tempered}), trained in "
            f"{train_time.total_seconds():.0f} s, support mass "
            f"{np.exp(log_mass):.4f} ({n_support} of {self.n_pilot})"
        )
        if kernel is not None:
            logger.info(
                f"Level {self.level_count}: heavy-sample kernels on "
                + ", ".join(
                    f"{len(p.centres)} in "
                    + ("prime space" if j is None else f"expert {j}")
                    for j, p in kernel["parts"]
                )
                + " (training-weight shares "
                + ", ".join(f"{v:.3f}" for v in kernel["shares"])
                + f"), fraction {self.kernel_fraction}"
            )

    # Heavy-sample kernels (``kernel_threshold``): a sample holding a large
    # share of the posterior weight has a meta-proposal density far below
    # the posterior's there; no flow fitted to the samples sees it (one point
    # among thousands), so the level blends a Student-t mixture centred on
    # such points into its flow, in the flow's own coordinates: for a group
    # mixture, in each expert's standardised base frame (the kernel then
    # covers every group image of the point, with the mixture's branch
    # weights, sign patterns and domain), else in the flow's prime space.
    # The level's density stays exact; the kernel's draws train the next
    # levels.

    @staticmethod
    def _group_experts(module):
        """The group-mixture experts of a flow module (active ones) and the
        expert routing of prime points, or None for a plain flow."""
        experts = getattr(module, "experts", None)
        if experts is not None:
            act = module._active()
            return list(experts[:act]), (
                (lambda xt: module._route(xt)) if act > 1
                else (lambda xt: torch.zeros(len(xt), dtype=torch.long))
            )
        if hasattr(module, "_assign_branch") and hasattr(
            module, "_standardise"
        ):
            return [module], (
                lambda xt: torch.zeros(len(xt), dtype=torch.long)
            )
        return None

    @staticmethod
    def _base_frame(expert, xt):
        """The standardised base-flow coordinates of prime points."""
        assigned, pre, _ = expert._assign_branch(xt)
        canon = pre[assigned, torch.arange(len(xt))]
        t, _ = expert._canon_to_base(canon)
        return expert._standardise(t, assigned)

    def _kernel_part(self, u, heavy, metric=None):
        """A :class:`HeavyKernel` on rows ``heavy`` of ``u``, each with the
        covariance of its nearest rows (in ``metric``, default Euclidean)."""
        n, d = u.shape
        k = min(max(self.kernel_neighbours, 2 * d + 2), n)
        floor = np.atleast_2d(np.cov(u.T)) if n > 1 else np.eye(d)
        floor = 1e-3 * floor + 1e-9 * np.eye(d)
        centres, chols = [], []
        for i in heavy:
            delta = u - u[i]
            if metric is None:
                dist = np.sum(delta**2, axis=1)
            else:
                dist = np.einsum("ij,jk,ik->i", delta, metric, delta)
            near = np.argsort(dist)[:k]
            c = np.atleast_2d(np.cov(u[near].T)) + floor
            nu = self.kernel_dof
            if nu > 2:
                # a t's covariance is nu / (nu - 2) times its scale
                c = c * (nu - 2) / nu
            centres.append(u[i])
            chols.append(np.linalg.cholesky(c))
        return HeavyKernel(centres, chols, self.kernel_dof, self.kernel_fraction)

    def _heavy_kernel(
        self, prop, samples, x, all_samples=None, threshold=None
    ):
        """Kernels on the heavy samples of the capped posterior: weights
        ``W_i min(L_i, L*)`` over every sample (the training target above
        the threshold ``L*``, the posterior below it, and the posterior once
        ``L*`` reaches its bulk). The heavy samples of a run are draws in a
        shell already passed, below every later threshold, so the training
        set alone never holds them; and a plain posterior share is the
        highest-likelihood point's while ``L*`` is far below the bulk."""
        if self.kernel_threshold is None or len(samples) < 2:
            return None
        pool = samples if all_samples is None else all_samples
        log_l = np.asarray(pool["logL"], dtype=float)
        if threshold is None or not np.isfinite(threshold):
            threshold = np.min(np.asarray(samples["logL"], dtype=float))
        log_p = np.asarray(pool["logW"], dtype=float) + np.minimum(
            log_l, threshold
        )
        ok = np.isfinite(log_p)
        if not ok.any():
            return None
        share = np.zeros(len(pool))
        share[ok] = np.exp(log_p[ok] - logsumexp(log_p[ok]))
        top = np.argsort(share)[::-1][: self.kernel_max]
        top = top[share[top] > self.kernel_threshold]
        if not len(top):
            return None
        # the heavy points appended to the training points: their
        # neighbours and the experts' frames come from these (a heavy
        # training sample appears twice, which changes nothing)
        # and their nearest samples of the whole run (in the unit hypercube),
        # the region a heavy point below the threshold lives in
        shares = share[top]
        u_pool = np.column_stack(
            [np.asarray(pool[n], dtype=float) for n in self.model.names]
        )
        near = set()
        m = min(5 * max(self.kernel_neighbours, 1), len(pool))
        for i in top:
            d2 = np.sum((u_pool - u_pool[i]) ** 2, axis=1)
            near.update(np.argpartition(d2, m - 1)[:m].tolist())
        near = np.array(sorted(near.difference(top.tolist())), dtype=int)
        ex = self.model.from_unit_hypercube(
            np.concatenate([pool[top], pool[near]]).copy()
        )
        ex["logP"] = self.model.batch_evaluate_log_prior(ex)
        heavy = len(x) + np.arange(len(top))
        x = np.concatenate([x, ex])
        x_prime, _ = prop.rescale(x.copy())
        arr = live_points_to_array(
            x_prime, names=prop.prime_parameters, copy=True
        )
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        arr = arr[: len(x)]  # the points' own copies (not their mirrors)
        module = prop.flow.model
        group = self._group_experts(module)
        parts = []
        if group is None:
            good = np.isfinite(arr).all(axis=1)
            heavy = heavy[good[heavy]]
            rows = np.flatnonzero(good)
            u = arr[rows]
            metric = np.linalg.inv(
                np.atleast_2d(np.cov(u.T)) + 1e-12 * np.eye(u.shape[1])
            )
            pos = np.searchsorted(rows, heavy)
            if len(pos):
                parts.append((None, self._kernel_part(u, pos, metric)))
        else:
            experts, route = group
            dtype = next(module.parameters()).dtype
            with torch.no_grad():
                xt = torch.as_tensor(arr, dtype=dtype)
                good = torch.isfinite(xt).all(dim=1).numpy()
                rows = np.flatnonzero(good)
                r = np.full(len(arr), -1)
                r[rows] = route(xt[rows]).cpu().numpy()
                for j, expert in enumerate(experts):
                    mine = np.flatnonzero(r == j)
                    hj = [h for h in heavy if r[h] == j]
                    if not hj or len(mine) < 2:
                        continue
                    u = self._base_frame(expert, xt[mine]).cpu().numpy()
                    fin = np.isfinite(u).all(axis=1)
                    pos = [
                        int(np.flatnonzero(mine == h)[0]) for h in hj
                    ]
                    pos = [q for q in pos if fin[q]]
                    if not pos:
                        continue
                    keep = np.flatnonzero(fin)
                    u = u[keep]
                    pos = np.searchsorted(keep, pos)
                    parts.append((j, self._kernel_part(u, pos)))
        if not parts:
            return None
        return dict(parts=parts, shares=[float(v) for v in shares])

    def _attach_kernel(self, prop, kernel):
        """Blend ``kernel`` (see :meth:`_heavy_kernel`) into ``prop``'s flow."""
        if kernel is None:
            return
        module = prop.flow.model
        group = self._group_experts(module)
        masses = kernel.setdefault("domain_log_mass", {})
        for j, part in kernel["parts"]:
            if j is None:
                prop.flow.model = KernelMixFlow(prop.flow.model, part)
                continue
            expert = group[0][j]
            expert.heavy_kernel = part
            # the base mass inside the domain changes with the kernel:
            # measured once, then restored exactly (resume)
            if j in masses:
                expert._log_domain_mass.fill_(masses[j])
                expert._domain_mass_seen = True
            else:
                masses[j] = expert.measure_domain_mass()

    def _detach_kernel(self, prop):
        """``prop``'s flow without any kernel (a warm copy of a level)."""
        while isinstance(prop.flow.model, KernelMixFlow):
            prop.flow.model = prop.flow.model.flow
        experts = getattr(prop.flow.model, "experts", None)
        for e in list(experts or []) + [prop.flow.model]:
            if "heavy_kernel" in getattr(e, "__dict__", {}):
                e.heavy_kernel = None

    def _kernel(self, k):
        kernels = getattr(self, "kernels", ())
        return kernels[k] if k < len(kernels) else None

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

    # Prime-space box (``box_sigma``): level k's density is its flow's
    # truncated to a box in the flow's own (prime) coordinates, the weighted
    # mean of its training data +- ``box_sigma`` weighted standard deviations
    # per coordinate (the training weights, so the old samples' light tails
    # do not set it). Too small a box only costs variance and rejected
    # draws, as a flow with light tails would: the prior, a component of
    # every mixture, still covers everything. Draws
    # outside are rejected and the support mass counts them, so the level
    # stays normalised; scoring gives -inf to the copies outside after the
    # rescale alone, skipping the flow and the round-trip inverse, which are
    # most of the cost for the old samples far from a new level.

    def _prime_box(self, prop, x, weights=None):
        if self.box_sigma is None:
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            x_prime, _ = prop.rescale(x.copy())
        arr = live_points_to_array(
            x_prime, names=prop.prime_parameters, copy=True
        )
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        # the rescale's mirror copies come in blocks of len(x) rows
        w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
        w = np.tile(w, len(arr) // len(x))
        keep = np.isfinite(arr).all(axis=1)
        arr, w = arr[keep], w[keep]
        mean = np.average(arr, axis=0, weights=w)
        sd = np.sqrt(np.average((arr - mean) ** 2, axis=0, weights=w))
        return np.stack([mean - self.box_sigma * sd, mean + self.box_sigma * sd])

    def _box(self, k):
        boxes = self.prime_boxes
        return boxes[k] if k < len(boxes) else None

    @staticmethod
    def _in_box(arr, box):
        if box is None:
            return np.ones(len(arr), dtype=bool)
        return ((arr >= box[0]) & (arr <= box[1])).all(axis=1)

    def _draw_physical(self, prop, n, box=None):
        """``n`` latent draws of ``prop`` pushed to the physical space: the
        ones inside the prior support (structured, ``logP`` set), and with a
        ``box`` the ones whose drawn prime point is inside it."""
        z = prop.sample_latent_distribution(n)
        if box is None:
            x = prop.backward_pass(z, rescale=True)[0]
        else:
            # the drawn prime points, boxed, then inverted: the mirror
            # density backward_pass would add is recomputed when scoring
            x_prime = prop.backward_pass(z, rescale=False)[0]
            if not len(x_prime):
                return x_prime
            arr = live_points_to_array(
                x_prime, names=prop.prime_parameters, copy=True
            )
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            x_prime = x_prime[self._in_box(arr, box)]
            if len(x_prime) < 2:
                return empty_structured_array(0, dtype=self.dtype)
            with np.errstate(invalid="ignore", divide="ignore"):
                x = prop.inverse_rescale(x_prime)[0]
            x = x[self.model.in_bounds(x)]
        if not len(x):
            return x
        x = x[np.isfinite(live_points_to_array(x, self.model.names)).all(1)]
        x["logP"] = self.model.batch_evaluate_log_prior(x)
        return x[np.isfinite(x["logP"])]

    def _log_support_mass(self, prop, box=None, strict=True):
        """ln of the level's draws' share inside the prior support (and the
        box), and their number; with ``strict``, none is an error, else the
        share is -inf."""
        kept, made = 0, 0
        while made < self.n_pilot:
            m = min(_CHUNK, self.n_pilot - made)
            kept += len(self._draw_physical(prop, m, box))
            made += m
        if not kept and not strict:
            return -np.inf, 0
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
                lq = self._preimage_log_q(prop, xc, self._box(k))[:n]
            out[start : start + n] = np.where(np.isnan(lq), -np.inf, lq)
        inside = self.model.in_bounds(x)
        out[~inside] = -np.inf
        return out - self.log_mass[k]

    def _preimage_log_q(self, prop, x, box=None):
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
        if box is not None:
            finite &= self._in_box(arr, box)
        rows = np.flatnonzero(finite)
        if box is not None and len(rows) == 1:
            # some reparameterisations' table lookups fail on one point
            rows = np.flatnonzero(finite | (np.arange(len(arr)) == (rows[0] + 1) % len(arr)))
        if not len(rows):
            return np.full(n, -np.inf)
        log_q[rows] = (
            np.asarray(prop.flow.log_prob(arr[rows]), dtype=float)
            + np.asarray(log_j, dtype=float)[rows]
        )
        if box is None:
            back, _ = prop.inverse_rescale(x_prime.copy())
        else:
            back, _ = prop.inverse_rescale(x_prime[rows].copy())
        target = np.tile(x, k)[rows] if box is not None else None
        same = np.ones(len(rows) if box is not None else len(arr), dtype=bool)
        for name in self.model.names:
            lo, hi = self.model.bounds[name]
            width = hi - lo
            # a periodic angle may come back a period away
            ref = target[name] if box is not None else np.tile(x[name], k)
            d = np.abs(back[name] - ref)
            d = np.minimum(d, np.abs(d - width))
            same &= d <= _ROUND_TRIP * width
        if box is None:
            log_q[~same] = -np.inf
        else:
            log_q[rows[~same]] = -np.inf
            log_q[~finite] = -np.inf
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
        eps = self.defensive_fraction
        if eps:
            # each level is the defensive mixture (1 - eps) q_k + eps prior
            # (density 1 in the unit hypercube), as it draws
            out = np.logaddexp(np.log1p(-eps) + out, np.log(eps))
        self.scoring_time += datetime.datetime.now() - st
        return out

    # Partial deterministic mixture (Elvira et al. 2015): the components
    # (prior = column 0, level k = column k + 1) are split into consecutive
    # blocks of ``mixture_block``, and a sample is weighted by the mixture of
    # its own origin's block only, with that block's weights renormalised.
    # The block depends on where a sample was drawn from, not where it lies,
    # so the estimator stays unbiased; each level is scored on its block's
    # samples only, so the cost is linear in the number of levels. Columns
    # outside a sample's block are NaN. ``mixture_block=None`` is the full
    # mixture.

    def _block(self, column):
        column = np.asarray(column)
        if self.mixture_block is None:
            return np.zeros_like(column)
        return column // self.mixture_block

    def _origin_columns(self, samples, origin=None):
        if origin is None:
            origin = samples["it"]
        return np.asarray(origin, dtype=int) + 1

    def _log_q_all(self, samples, origin=None):
        """``ln q`` of every component (prior first) at unit-hypercube
        samples, NaN outside each sample's block."""
        log_q = np.zeros((len(samples), self.n_proposals))
        if self.mixture_block is not None:
            row_block = self._block(self._origin_columns(samples, origin))
            col_block = self._block(np.arange(self.n_proposals))
            log_q[row_block[:, None] != col_block[None, :]] = np.nan
        if self.levels:
            x, log_j = self._physical(samples)
            for k in range(len(self.levels)):
                rows = ~np.isnan(log_q[:, k + 1])
                if rows.any():
                    log_q[rows, k + 1] = self._log_q_levels(
                        x[rows], log_j[rows], [k]
                    )[:, 0]
        return log_q

    def compute_meta_proposal_from_log_q(self, log_q, weights=None):
        weights = self.weights_array if weights is None else np.asarray(
            weights, dtype=float
        )
        if self.mixture_block is None:
            return logsumexp(log_q, b=weights, axis=1)
        inside = ~np.isnan(log_q)
        w = np.where(inside, weights[None, :], 0.0)
        w = w / w.sum(axis=1, keepdims=True)
        return logsumexp(np.where(inside, log_q, -np.inf), b=w, axis=1)

    def compute_meta_proposal_samples(self, samples, origin=None):
        if any(np.isnan(w) for w in self._weights.values()):
            raise RuntimeError(f"Weights are not set: {self._weights}")
        log_q = self._log_q_all(samples, origin)
        return self.compute_meta_proposal_from_log_q(log_q), log_q

    def update_log_q(self, samples, log_q):
        """``log_q`` with the current level's column appended (scored only
        on the samples of its block)."""
        if log_q.shape[1] == self.n_proposals:
            raise ValueError("log_q array already contains current proposal")
        column = self.level_count + 1
        new = np.full((len(samples), 1), np.nan)
        rows = self._block(self._origin_columns(samples)) == self._block(
            column
        )
        if self.mixture_block is None:
            rows[:] = True
        if rows.any():
            x, log_j = self._physical(samples[rows])
            new[rows] = self._log_q_levels(x, log_j, [self.level_count])
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
        eps = self.defensive_fraction
        if eps:
            m = self.rng.binomial(n, eps)
            return np.concatenate(
                [
                    self._draw_flow_level(k, n - m),
                    self.model.sample_unit_hypercube(m),
                ]
            )
        return self._draw_flow_level(k, n)

    def _draw_flow_level(self, k, n):
        """``n`` draws of level ``k``'s flow (truncated to the prior
        support)."""
        prop = self.levels[k]
        rate = max(np.exp(self.log_mass[k]), 1e-3)
        out = empty_structured_array(0, dtype=self.dtype)
        while len(out) < n:
            m = int(min(max(1.02 * (n - len(out)) / rate, 100), 1e6))
            x = self._draw_physical(prop, m, self._box(k))
            if len(x):
                out = np.concatenate([out, self._to_unit(x)])
        return out[:n]

    def _finish(self, samples, origin=None):
        """Densities, prior and weights of unit-hypercube samples drawn from
        the components ``origin`` (-1 the prior; default ``samples['it']``)."""
        log_Q, log_q = self.compute_meta_proposal_samples(samples, origin)
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
        samples = self._draw_level(k, n)
        return self._finish(samples, np.full(len(samples), k))

    def draw_from_prior(self, n: int):
        samples = self.model.sample_unit_hypercube(n)
        return self._finish(samples, np.full(len(samples), -1))

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
        for k, prop in enumerate(self.levels):
            self._detach_kernel(prop)
            self._attach_kernel(prop, self._kernel(k))
        self.initialised = self.template is not None
