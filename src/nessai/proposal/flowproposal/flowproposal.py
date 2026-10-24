"""Proposal class that uses normalising flows and rejection sampling."""

from __future__ import annotations

import datetime
import logging
import math
import warnings

import numpy as np
from scipy.special import logsumexp

from ... import config
from ...livepoint import empty_structured_array, live_points_to_array
from ...utils.structures import get_subset_arrays
from .base import BaseFlowProposal
from .truncation import (
    TruncationScheme,
    apply_default_truncation_config,
    build_truncation_methods,
    get_deprecated_latent_radius_arguments,
    get_deprecated_latent_radius_kwargs,
    get_truncation_rule_class,
    normalise_truncation_kwargs,
)

logger = logging.getLogger(__name__)


def _clip_weights(weights: np.ndarray) -> np.ndarray:
    """Clip the largest ``ceil(sqrt(N))`` weights to their mean."""
    weights = weights.copy()
    num_clip = math.ceil(math.sqrt(len(weights)))
    if not num_clip:
        return weights
    if num_clip >= len(weights):
        clip_idx = np.arange(len(weights))
    else:
        clip_idx = np.argpartition(-weights, num_clip - 1)[:num_clip]
    weights[clip_idx] = weights[clip_idx].mean()
    weights /= weights.mean()
    return weights


def _weight_cap_for_mass(weights: np.ndarray, mass: float) -> float:
    """Smallest cap ``c`` whose excess ``sum(max(w - c, 0))`` is at most
    ``mass * sum(w)``.

    Rejection sampling with the bound ``c`` instead of ``max(w)`` accepts a
    draw with probability ``min(w, c) / c``, so the population follows
    ``p * min(1, c / w)``: exactly the prior except on the draws above the
    cap, which lose the fraction ``mass`` of the prior mass of the support
    (as estimated by the draws themselves).
    """
    w = np.sort(weights[np.isfinite(weights)])[::-1]
    if not len(w):
        return np.nan
    total = w.sum()
    top = np.cumsum(w)
    k = np.arange(1, len(w) + 1)
    # excess over a cap equal to the k-th largest weight: increasing in k
    excess = top - k * w
    n = int(np.searchsorted(excess, mass * total, side="right"))
    # the cap lies between the (n+1)-th and n-th largest weights
    return float(max((top[n - 1] - mass * total) / n, w[min(n, len(w) - 1)]))


def _weight_cap_for_yield(weights: np.ndarray, n_yield: float) -> float:
    """Largest cap ``c`` whose expected rejection-sampling yield
    ``sum(min(w, c)) / c`` is at least ``n_yield``.
    """
    w = np.sort(weights[np.isfinite(weights)])[::-1]
    if not len(w) or n_yield <= 0:
        return np.inf
    if n_yield >= len(w):
        return float(w[-1])
    total = w.sum()
    # with the k largest weights above the cap: k + (total - top_k) / c = Y
    k = np.arange(int(np.ceil(n_yield)))
    top = np.concatenate([[0.0], np.cumsum(w)])[k]
    cap = (total - top) / (n_yield - k)
    ok = w[k] <= cap
    return float(cap[np.argmax(ok)]) if ok.any() else float(w[-1])


class FlowProposal(BaseFlowProposal):
    """Proposal that samples in latent space using the trained flow.

    Parameters
    ----------
    model : nessai.model.Model
        The model to use for the proposal.
    poolsize : int, optional
        The number of samples to draw from the flow when populating the pool.
    latent_prior : str, optional
        The prior to use for the latent space. This argument is deprecated and
        only 'flow' is supported.
    latent_temperature : float, optional
        The temperature to use for the latent space. If None, no scaling is
        applied.
    constant_volume_mode : bool, optional
        Whether to use constant volume mode for the latent radius. This argument
        is deprecated and should be configured via :code:`truncation_methods`
        and :code:`truncation_kwargs` instead.
    volume_fraction : float, optional
        The volume fraction to use for the latent radius. This argument is
        deprecated and should be configured via :code:`truncation_methods` and
        :code:`truncation_kwargs` instead.
    fuzz : float, optional
        The fuzz to use for the latent radius. This argument is deprecated and
        should be configured via :code:`truncation_methods` and
        :code:`truncation_kwargs` instead.
    fixed_radius : float, optional
        The fixed radius to use for the latent radius. This argument is
        deprecated and should be configured via :code:`truncation_methods` and
        :code:`truncation_kwargs` instead.
    radius_mode : str, optional
        The radius mode to use for the latent radius. This argument is deprecated
        and should be configured via :code:`truncation_methods` and
        :code:`truncation_kwargs` instead.
    drawsize : int, optional
        The number of samples to draw from the flow per batch when populating
        the pool, at most the pool size (default: the initial pool size).
    truncate_log_q : bool, optional
        Whether to truncate the log q values when populating the pool. This
        argument is deprecated and should be configured via
        :code:`truncation_methods` and :code:`truncation_kwargs` instead.
    expansion_fraction : float, optional
        The expansion fraction to use for the latent radius. This argument is
        deprecated and should be configured via :code:`truncation_methods` and
        :code:`truncation_kwargs` instead.
    min_radius : float, optional
        The minimum radius to use for the latent radius. This argument is
        deprecated and should be configured via :code:`truncation_methods` and
        :code:`truncation_kwargs` instead.
    max_radius : float, optional
        The maximum radius to use for the latent radius. This argument is
        deprecated and should be configured via :code:`truncation_methods` and
        :code:`truncation_kwargs` instead.
    compute_radius_with_all : bool, optional
        Whether to compute the latent radius using all samples. This argument is
        deprecated and should be configured via :code:`truncation_methods` and
        :code:`truncation_kwargs` instead.
    enforce_likelihood_threshold : bool, optional
        Whether to enforce a likelihood threshold when populating the pool. This
        argument is deprecated and should be configured via
        :code:`truncation_methods` and :code:`truncation_kwargs` instead.
    truncation_method : str, optional
        The truncation method to use when populating the pool.
    truncation_methods : list of str, optional
        The truncation methods to use when populating the pool.
    truncation_kwargs : dict, optional
        The keyword arguments to use for the truncation methods when populating
        the pool. When using :code:`truncation_methods`, the keys of this
        dictionary should match the names of the truncation methods.
    clip_population_weights : bool, optional
        Clip the largest ``ceil(sqrt(N))`` rejection weights to their mean.
    population_weight_cap_mass : float, optional
        If set, the rejection bound is the smallest weight cap under which the
        draws above it lose at most this fraction of the prior mass of the
        proposal's support (instead of the largest weight). Trades a
        controlled under-sampling of the lowest-density fringe of the
        support for population efficiency. Best combined with
        :code:`accumulate_weights=True`, which sets the cap from every draw
        of the populate call rather than batch by batch. If a heavy-tailed
        weight distribution (a poorly fitted flow) would make that cap yield
        fewer than ``N * n_samples / max_samples`` points after ``N`` draws
        (counted before truncation, as ``max_samples`` is), the cap is
        lowered to that yield, so the pool always fills within
        ``max_samples`` draws; the prior mass then under-sampled is logged.
    sum_mirror_density : bool, optional
        With a reparameterisation that duplicates the points by boundary
        inversion, a draw on either side of the boundary is folded back to the
        same physical point, so the pool's density there is the flow's
        density summed over the point and its mirror copies. If true, the
        draws (and the live points the truncation rules are set on) are
        scored with that sum, at one more flow evaluation per draw; if false,
        a draw is scored with the density of the side it was drawn on, which
        overstates its weight by ``1 + q_mirror / q_side``.
    threshold_from_live_points : bool
        If true, the truncation thresholds are taken from the sampler's
        current live points (set as ``current_live_points`` before each
        populate) instead of the flow's training data, the live points at
        the last training: those lag the current ones by up to a training
        interval, so the threshold is stale.
    **kwargs
        Additional keyword arguments to pass to the base class.
    """

    def __init__(
        self,
        model,
        poolsize=None,
        latent_prior=None,
        latent_temperature=None,
        constant_volume_mode=None,
        volume_fraction=None,
        fuzz=None,
        fixed_radius=None,
        radius_mode=None,
        drawsize=None,
        truncate_log_q=False,
        expansion_fraction=None,
        min_radius=None,
        max_radius=None,
        compute_radius_with_all=None,
        enforce_likelihood_threshold=False,
        truncation_method=None,
        truncation_methods=None,
        truncation_kwargs=None,
        clip_population_weights=False,
        population_weight_cap_mass=None,
        sum_mirror_density=False,
        threshold_from_live_points=False,
        **kwargs,
    ):
        super().__init__(model, poolsize=poolsize, **kwargs)
        self.sum_mirror_density = bool(sum_mirror_density)
        self.threshold_from_live_points = bool(threshold_from_live_points)
        self.current_live_points = None
        self.population_truncation_history = []
        logger.debug("Initialising FlowProposal")
        self._warn_for_deprecated_latent_prior(latent_prior)

        self.configure_population(
            drawsize,
            latent_prior=latent_prior,
            latent_temperature=latent_temperature,
            clip_population_weights=clip_population_weights,
            population_weight_cap_mass=population_weight_cap_mass,
        )

        self._truncation_scheme = TruncationScheme()
        self.truncation_method = truncation_method
        self.truncation_methods = []
        self.truncation_kwargs = {}
        deprecated_latent_radius_arguments = dict(
            fixed_radius=fixed_radius,
            radius_mode=radius_mode,
            min_radius=min_radius,
            max_radius=max_radius,
            compute_radius_with_all=compute_radius_with_all,
            constant_volume_mode=constant_volume_mode,
            volume_fraction=volume_fraction,
            fuzz=fuzz,
            expansion_fraction=expansion_fraction,
        )
        self._warn_for_deprecated_truncation_arguments(
            **deprecated_latent_radius_arguments
        )
        self._warn_for_deprecated_likelihood_threshold(
            enforce_likelihood_threshold
        )
        latent_radius_kwargs = get_deprecated_latent_radius_kwargs(
            **deprecated_latent_radius_arguments
        )

        self.configure_truncation(
            truncation_method=truncation_method,
            truncation_methods=truncation_methods,
            truncation_kwargs=truncation_kwargs,
            truncate_log_q=truncate_log_q,
            enforce_likelihood_threshold=enforce_likelihood_threshold,
            latent_radius_kwargs=latent_radius_kwargs or None,
            default_latent_radius=True,
        )
        self._log_configuration()

    @property
    def truncation(self) -> TruncationScheme:
        return self._truncation_scheme

    def get_truncation_rule(self, name: str):
        return self.truncation.get_rule(name)

    def _get_latent_radius_rule(self):
        return self._truncation_scheme.get_rule("latent_radius")

    def _sync_truncation_state(self) -> None:
        self.truncation_methods = self._truncation_scheme.rule_names
        self.truncate_log_q = "min_log_q" in self.truncation_methods
        self.enforce_likelihood_threshold = (
            "likelihood_threshold" in self.truncation_methods
        )

    def _log_configuration(self) -> None:
        logger.info(
            "FlowProposal configuration: drawsize=%s, latent_prior=%s, "
            "latent_temperature=%s, truncation_methods=%s",
            self.drawsize,
            self.latent_prior,
            self.latent_temperature,
            self.truncation_methods,
        )
        for rule in self._truncation_scheme.rules:
            if hasattr(rule, "to_kwargs"):
                logger.info(
                    "FlowProposal truncation rule %s: %s",
                    rule.name,
                    rule.to_kwargs(),
                )
            else:
                logger.info(
                    "FlowProposal truncation rule enabled: %s",
                    rule.name,
                )

    def _warn_for_deprecated_truncation_arguments(self, **kwargs) -> None:
        deprecated = get_deprecated_latent_radius_arguments(**kwargs)
        if not deprecated:
            return

        warnings.warn(
            "The following FlowProposal arguments are deprecated and should "
            "be configured via truncation_kwargs['latent_radius'] instead: "
            f"{', '.join(deprecated)}.",
            FutureWarning,
            stacklevel=3,
        )

    @staticmethod
    def _warn_for_deprecated_latent_prior(latent_prior) -> None:
        if latent_prior != "flow":
            return
        warnings.warn(
            "`latent_prior` is deprecated and will be removed. "
            "FlowProposal always uses the flow latent distribution.",
            FutureWarning,
            stacklevel=3,
        )

    @staticmethod
    def _warn_for_deprecated_likelihood_threshold(
        enforce_likelihood_threshold: bool,
    ) -> None:
        if not enforce_likelihood_threshold:
            return
        warnings.warn(
            "`enforce_likelihood_threshold` is deprecated and should be "
            "configured via "
            "`truncation_methods=['likelihood_threshold']` instead.",
            FutureWarning,
            stacklevel=3,
        )

    def configure_population(
        self,
        drawsize,
        latent_prior=None,
        latent_temperature=None,
        clip_population_weights=False,
        population_weight_cap_mass=None,
    ) -> None:
        """Configure settings related to population."""
        if drawsize is None:
            drawsize = self.poolsize

        if latent_prior not in (None, "flow"):
            raise ValueError(
                "Only the flow latent prior is supported. "
                "Use truncation rules to filter latent samples."
            )
        if latent_temperature is not None:
            if isinstance(latent_temperature, bool) or not isinstance(
                latent_temperature, (int, float)
            ):
                raise TypeError("latent_temperature must be a float")
            latent_temperature = float(latent_temperature)
            if latent_temperature <= 0.0:
                raise ValueError("latent_temperature must be positive")

        self.drawsize = drawsize
        self.latent_prior = "flow"
        self.latent_temperature = latent_temperature
        self.clip_population_weights = clip_population_weights
        if population_weight_cap_mass is not None:
            population_weight_cap_mass = float(population_weight_cap_mass)
            if not 0.0 < population_weight_cap_mass < 1.0:
                raise ValueError(
                    "population_weight_cap_mass must be in (0, 1)"
                )
            if clip_population_weights:
                raise ValueError(
                    "clip_population_weights and population_weight_cap_mass "
                    "are mutually exclusive"
                )
        self.population_weight_cap_mass = population_weight_cap_mass

    def _get_population_log_weights(
        self, log_weights, min_yield=None
    ) -> np.ndarray:
        """Return log-weights used in the rejection step during population."""
        log_weights = np.asarray(log_weights, dtype=float)
        log_weights = log_weights - np.nanmax(log_weights)
        cap_mass = getattr(self, "population_weight_cap_mass", None)
        if cap_mass is not None:
            w = np.exp(log_weights)
            cap = _weight_cap_for_mass(w, cap_mass)
            if not np.isfinite(cap) or cap <= 0:
                return log_weights
            self._population_cap_lost_mass = None
            if min_yield:
                # a heavy tail must not stall the population: relax the cap
                # until the draws so far yield their share of the pool
                floor = _weight_cap_for_yield(w, min_yield)
                if floor < cap:
                    cap = floor
                    finite = np.isfinite(w)
                    self._population_cap_lost_mass = float(
                        np.maximum(w[finite] - cap, 0.0).sum()
                        / w[finite].sum()
                    )
            return np.minimum(log_weights - np.log(cap), 0.0)
        if not self.clip_population_weights:
            return log_weights

        weights = _clip_weights(np.exp(log_weights))
        return np.log(weights) - np.log(np.nanmax(weights))

    def configure_truncation(
        self,
        truncation_method=None,
        truncation_methods=None,
        truncation_kwargs=None,
        truncate_log_q=False,
        enforce_likelihood_threshold=False,
        latent_radius_kwargs=None,
        default_latent_radius=False,
    ) -> None:
        use_default_latent_radius = default_latent_radius and (
            truncation_method is None and truncation_methods is None
        )
        existing_rule = self._get_latent_radius_rule()
        if (
            latent_radius_kwargs is None
            and truncation_method is None
            and truncation_methods is None
            and existing_rule is not None
        ):
            latent_radius_kwargs = existing_rule.to_kwargs()

        methods = build_truncation_methods(
            truncation_method=truncation_method,
            truncation_methods=truncation_methods,
            truncate_log_q=truncate_log_q,
            enforce_likelihood_threshold=enforce_likelihood_threshold,
            latent_radius_kwargs=latent_radius_kwargs,
            default_latent_radius=use_default_latent_radius,
        )
        self.truncation_method = truncation_method

        truncation_kwargs = normalise_truncation_kwargs(
            truncation_method=truncation_method,
            truncation_methods=truncation_methods,
            truncation_kwargs=truncation_kwargs,
        )
        if latent_radius_kwargs is not None and "latent_radius" in methods:
            truncation_kwargs["latent_radius"] = {
                **latent_radius_kwargs,
                **truncation_kwargs.get("latent_radius", {}),
            }

        methods, self.truncation_kwargs = apply_default_truncation_config(
            methods,
            truncation_kwargs=truncation_kwargs,
            default_latent_radius=use_default_latent_radius,
        )

        rules = []
        for method in methods:
            rule_cls = get_truncation_rule_class(method)
            raw_kwargs = self.truncation_kwargs.get(method, {})
            if not isinstance(raw_kwargs, dict):
                raise TypeError(
                    f"Truncation kwargs for {method} must be a dictionary"
                )

            kwargs = dict(raw_kwargs)
            rules.append(rule_cls(**kwargs))

        self._truncation_scheme = TruncationScheme(rules)
        self._sync_truncation_state()

    def set_rescaling(self) -> None:
        """Set the rescaling functions."""
        super().set_rescaling()
        self._truncation_scheme.configure(self)

    def backward_pass(
        self,
        z,
        rescale=True,
        discard_nans=True,
        return_z=False,
        return_unit_hypercube=False,
    ):
        """Apply the inverse flow and inverse rescaling."""
        try:
            x, log_j = self.flow.inverse(z)
            log_prob = self.latent_log_prob(z, self.latent_temperature) - log_j
        except AssertionError as e:
            logger.warning(
                "Assertion error raised when sampling from the flow. Error: %s",
                e,
            )
            if return_z:
                return np.array([]), np.array([]), np.array([])
            return np.array([]), np.array([])

        if discard_nans:
            valid = np.isfinite(log_prob)
            x, log_prob, z = get_subset_arrays(valid, x, log_prob, z)

        x_array = np.asarray(x, dtype=config.livepoints.default_float_dtype)
        if x_array.ndim == 1:
            x_array = x_array[np.newaxis, :]
        x = empty_structured_array(
            x_array.shape[0], dtype=self.x_prime_internal_dtype
        )
        for i, parameter in enumerate(self.prime_parameters):
            x[parameter] = x_array[:, i]

        # proposals pickled before the option existed have no attribute
        sum_sides = (
            getattr(self, "sum_mirror_density", False)
            and rescale
            and not return_unit_hypercube
        )
        if rescale:
            x, log_J = self.inverse_rescale(
                x, return_unit_hypercube=return_unit_hypercube
            )
            log_prob -= log_J
            if sum_sides:
                x, z, log_prob, x_array = self.check_prior_bounds(
                    x, z, log_prob, x_array
                )
            elif not return_unit_hypercube:
                x, z, log_prob = self.check_prior_bounds(x, z, log_prob)

        if sum_sides and len(x):
            log_q_sides = self.mirror_log_q(
                x, drawn_prime=x_array, drawn_log_q=log_prob, z=z
            )
            if log_q_sides.shape[0] > 1:
                log_prob = logsumexp(log_q_sides, axis=0)

        if return_z:
            return x, log_prob, z
        return x, log_prob

    def mirror_log_q(self, x, drawn_prime=None, drawn_log_q=None, z=None):
        """Pool density of each physical point of ``x`` on each side of the
        boundary inversion: an ``(n_sides, len(x))`` array, one row per copy
        the rescale makes of ``x`` (one without duplication).

        Scored at the pool's latent temperature, as ``backward_pass`` scores
        the draws (the flow's own density is untempered). With
        ``drawn_prime`` (the prime-space points the draws were generated at,
        ``[len(x), dims]``) and ``drawn_log_q`` (their density), each draw's
        own side keeps that density and only the other sides go through the
        flow (``z``, their latent points, is unused by a plain flow).
        """
        n = len(x)
        x_prime, log_j = self.rescale(x.copy())
        arr = live_points_to_array(
            x_prime, names=self.prime_parameters, copy=True
        )
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        log_j = np.asarray(log_j, dtype=float)
        if len(arr) % n:
            raise RuntimeError(
                f"{len(arr)} rows of log q for {n} points: not whole blocks "
                "of mirror copies"
            )
        # boundary inversion with duplication appends the mirror copies in
        # blocks of ``len(x)`` rows
        k = len(arr) // n
        todo = np.ones((k, n), dtype=bool)
        log_q = np.empty((k, n))
        if drawn_prime is not None and k > 1:
            drawn_prime = np.asarray(drawn_prime, dtype=float).reshape(n, -1)
            dist = np.sum(
                (arr.reshape(k, n, -1) - drawn_prime[None]) ** 2, axis=-1
            )
            own = np.argmin(dist, axis=0)
            log_q[own, np.arange(n)] = drawn_log_q
            todo[own, np.arange(n)] = False
        rows = todo.ravel()
        temperature = getattr(self, "latent_temperature", None)
        if temperature in (None, 1.0):
            lq = self.flow.log_prob(arr[rows])
        else:
            z, lq = self.flow.forward_and_log_prob(arr[rows])
            lq = (
                lq
                + self.latent_log_prob(z, temperature)
                - self.latent_log_prob(z, None)
            )
        log_q[todo] = np.asarray(lq, dtype=float) + log_j[rows]
        return log_q

    def _record_truncation_diagnostics(self, n_clipped, log_cap) -> None:
        """Per pool: the clipped share of the pool (clip mode) and the live
        points outside the log-proposal cut."""
        n = len(self.x) if self.x is not None else 0
        self.population_clipped_share = (
            n_clipped / n if np.isfinite(log_cap) and n else np.nan
        )
        rule = self._truncation_scheme.get_rule("log_proposal_threshold")
        entry = dict(
            populated_count=int(self.populated_count),
            training_count=int(self.training_count),
            clipped_share=float(self.population_clipped_share),
            excluded_live_fraction=float(
                getattr(rule, "excluded_live_fraction", np.nan)
            ),
            excluded_live_mean_rank=float(
                getattr(rule, "excluded_live_mean_rank", np.nan)
            ),
            lost_live_mass=float(getattr(rule, "lost_live_mass", np.nan)),
            roundtrip_discarded=int(getattr(rule, "roundtrip_discarded", 0)),
        )
        if not hasattr(self, "population_truncation_history"):
            self.population_truncation_history = []
        self.population_truncation_history.append(entry)
        logger.debug("Pool truncation diagnostics: %s", entry)

    def _warn_max_samples(self, max_samples):
        """Warn that a pool stopped at ``max_samples`` draws, once per
        training round: the pools of one flow all hit it alike."""
        if getattr(self, "_max_samples_warned", None) == self.training_count:
            logger.debug("Reached max samples (%s)", max_samples)
            return
        self._max_samples_warned = self.training_count
        logger.warning(
            "Reached max samples (%s); again for this flow at DEBUG",
            max_samples,
        )

    def populate(
        self,
        worst_point,
        n_samples=10000,
        plot=True,
        r=None,
        max_samples=1_000_000,
    ) -> None:
        """Populate a pool of samples using staged truncation."""
        st = datetime.datetime.now()
        if not self.initialised:
            raise RuntimeError(
                "Proposal has not been initialised. Try calling `initialise()` "
                "first."
            )

        self._truncation_scheme.prepare(self, worst_point, radius=r)

        if self.indices:
            logger.debug(
                "Existing pool of samples is not empty. Discarding existing "
                "samples."
            )
        self.indices = []

        if self.accumulate_weights:
            samples = empty_structured_array(0, dtype=self.population_dtype)
        else:
            samples = empty_structured_array(
                n_samples, dtype=self.population_dtype
            )

        log_n = np.log(n_samples)
        log_n_expected = -np.inf
        n_proposed = 0
        log_weights = np.empty(0)
        n_accepted = 0
        accept = None

        log_cap = self._truncation_scheme.log_weight_cap
        clipped_all = np.empty(0, dtype=bool)
        n_clipped = 0

        while n_accepted < n_samples:
            # a batch is never larger than the pool: ``drawsize`` caps it
            z = self.sample_latent_distribution(min(self.drawsize, n_samples))
            n_batch = z.shape[0]
            n_proposed += n_batch
            z = self._truncation_scheme.apply_latent(self, z)
            if not len(z):
                if n_proposed > max_samples:
                    self._warn_max_samples(max_samples)
                    break
                continue

            x, log_q, z = self.backward_pass(
                z,
                rescale=True,
                return_z=True,
                return_unit_hypercube=self.map_to_unit_hypercube,
            )
            # backward_pass returns plain empty arrays when the flow inverse
            # fails: skip the truncation stages for them
            if len(x):
                x, log_q, z = self._truncation_scheme.apply_after_backward(
                    self, x, log_q, z
                )
            if not len(x):
                if n_proposed > max_samples:
                    self._warn_max_samples(max_samples)
                    break
                continue

            if self._truncation_scheme.requires_log_likelihood:
                x["logL"] = self.model.batch_evaluate_log_likelihood(
                    x, unit_hypercube=self.map_to_unit_hypercube
                )
                x, log_q, z = self._truncation_scheme.apply_after_likelihood(
                    self, x, log_q, z
                )
                if not len(x):
                    if n_proposed > max_samples:
                        self._warn_max_samples(max_samples)
                        break
                    continue

            log_w = self.compute_weights(x, log_q)
            # clip mode of a truncation rule: weights above the cap are
            # clipped to it instead of the draws being dropped
            clipped = np.zeros(len(log_w), dtype=bool)
            if np.isfinite(log_cap):
                clipped = log_w > log_cap
                log_w = np.minimum(log_w, log_cap)

            if self.accumulate_weights:
                samples = np.concatenate([samples, x])
                log_weights = np.concatenate([log_weights, log_w])
                clipped_all = np.concatenate([clipped_all, clipped])
                log_weights_rejection = self._get_population_log_weights(
                    log_weights,
                    min_yield=n_proposed * n_samples / max_samples,
                )
                log_n_expected = logsumexp(log_weights_rejection)

                logger.debug(
                    "Drawn %s - n expected: %s / %s",
                    samples.size,
                    np.exp(log_n_expected),
                    n_samples,
                )

                if log_n_expected >= log_n:
                    log_u = np.log(self.rng.random(len(log_weights)))
                    accept = log_weights_rejection > log_u
                    n_accepted = np.sum(accept)
                if n_proposed > max_samples:
                    self._warn_max_samples(max_samples)
                    break
            else:
                log_w = self._get_population_log_weights(
                    log_w, min_yield=n_batch * n_samples / max_samples
                )
                log_u = np.log(self.rng.random(len(log_w)))
                accept = log_w > log_u
                n_accept_batch = accept.sum()
                m = min(n_samples - n_accepted, n_accept_batch)
                samples[n_accepted : n_accepted + m] = x[accept][:m]
                n_clipped += int(clipped[accept][:m].sum())
                n_accepted += n_accept_batch
                logger.debug("n accepted: %s / %s", n_accepted, n_samples)
                if n_proposed > max_samples:
                    self._warn_max_samples(max_samples)
                    break

        if self.accumulate_weights:
            if accept is None or len(accept) != len(samples):
                log_weights_rejection = self._get_population_log_weights(
                    log_weights,
                    min_yield=n_proposed * n_samples / max_samples,
                )
                log_u = np.log(self.rng.random(len(log_weights)))
                accept = log_weights_rejection > log_u
            logger.debug("Total number of samples: %s", samples.size)
            n_accepted = np.sum(accept)
            self.x = samples[accept][:n_samples]
            n_clipped = int(clipped_all[accept][:n_samples].sum())
        else:
            self.x = samples[: min(n_accepted, n_samples)]
        self._record_truncation_diagnostics(n_clipped, log_cap)

        self.samples = self.convert_to_samples(self.x, plot=plot)
        if self._plot_pool and plot:
            self.plot_pool(self.samples)

        self.population_time += datetime.datetime.now() - st
        if not self._truncation_scheme.requires_log_likelihood:
            logger.debug("Evaluating log-likelihoods")
            self.samples["logL"] = self.model.batch_evaluate_log_likelihood(
                self.samples
            )
        if self.check_acceptance:
            self.acceptance.append(
                self.compute_acceptance(worst_point["logL"])
            )
            logger.debug("Current acceptance %s", self.acceptance[-1])

        self.indices = self.rng.permutation(self.samples.size).tolist()
        self.population_acceptance = n_accepted / n_proposed
        lost = getattr(self, "_population_cap_lost_mass", None)
        if getattr(self, "population_weight_cap_mass", None) and lost:
            logger.info(
                "Population weight cap lowered to fill the pool within "
                "max_samples (heavy-tailed weights): %.2g of the support's "
                "prior mass under-sampled (target %.2g)",
                lost,
                self.population_weight_cap_mass,
            )
            self._population_cap_lost_mass = None
        self.populated_count += 1
        self.populated = True
        self._checked_population = False

    def reset(self) -> None:
        """Reset the proposal."""
        super().reset()
        self._truncation_scheme.reset()

    def __getstate__(self):
        state = super().__getstate__()
        # a reference to the sampler's live points, which it pickles itself
        state["current_live_points"] = None
        return state

    def __setstate__(self, state):
        # Back-fill the attributes of resume files pickled before the
        # options existed: ``configure_population`` is not re-run on resume
        state.setdefault("threshold_from_live_points", False)
        state.setdefault("current_live_points", None)
        state.setdefault("population_truncation_history", [])
        self.__dict__.update(state)
