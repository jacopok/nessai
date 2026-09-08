"""Proposal class that uses normalising flows and rejection sampling."""

from __future__ import annotations

import datetime
import logging
import warnings

import numpy as np
from scipy.special import logsumexp

from ... import config
from ...livepoint import empty_structured_array
from ...utils.structures import get_subset_arrays
from ...utils.temperature import optimise_latent_temperature
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
    adapt_latent_temperature : bool, optional
        If True, the latent temperature is re-estimated after every call to
        :meth:`populate` by retroactively reweighting the drawn batch of latent
        radii and importance weights (see
        :func:`nessai.utils.temperature.optimise_latent_temperature`). The
        temperature is initialised to 1.0 if not otherwise set.
    latent_temperature_kwargs : dict, optional
        Extra keyword arguments forwarded to
        :func:`nessai.utils.temperature.optimise_latent_temperature`, e.g.
        ``criterion`` ("efficiency" or "ess"), ``tau`` or ``grid_range``.
    latent_temperature_validate : bool, optional
        If True (default) a candidate temperature is only adopted when a fresh
        validation batch drawn at that temperature has a higher rejection
        efficiency than one drawn at the current temperature; otherwise the
        geometric midpoint is tried, and failing that the temperature is left
        unchanged. This guards against the retroactive estimate being
        optimistically biased away from the draw temperature.
    latent_temperature_validation_size : int, optional
        Number of latent samples per validation batch. Defaults to
        ``drawsize``.
    latent_temperature_warmup : int, optional
        Number of flow trainings to wait before adapting the temperature; the
        density model is unreliable early in the run. Defaults to 2.
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
        The number of samples to draw from the flow when populating the pool.
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
    **kwargs
        Additional keyword arguments to pass to the base class.
    """

    def __init__(
        self,
        model,
        poolsize=None,
        latent_prior=None,
        latent_temperature=None,
        adapt_latent_temperature=False,
        latent_temperature_kwargs=None,
        latent_temperature_validate=True,
        latent_temperature_validation_size=None,
        latent_temperature_warmup=2,
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
        **kwargs,
    ):
        super().__init__(model, poolsize=poolsize, **kwargs)
        logger.debug("Initialising FlowProposal")
        self._warn_for_deprecated_latent_prior(latent_prior)

        self.configure_population(
            drawsize,
            latent_prior=latent_prior,
            latent_temperature=latent_temperature,
            adapt_latent_temperature=adapt_latent_temperature,
            latent_temperature_kwargs=latent_temperature_kwargs,
            latent_temperature_validate=latent_temperature_validate,
            latent_temperature_validation_size=latent_temperature_validation_size,
            latent_temperature_warmup=latent_temperature_warmup,
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
        adapt_latent_temperature=False,
        latent_temperature_kwargs=None,
        latent_temperature_validate=True,
        latent_temperature_validation_size=None,
        latent_temperature_warmup=2,
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

        if adapt_latent_temperature and latent_temperature is None:
            latent_temperature = 1.0

        self.drawsize = drawsize
        self.latent_prior = "flow"
        self.latent_temperature = latent_temperature
        self.adapt_latent_temperature = adapt_latent_temperature
        self.latent_temperature_kwargs = {
            **(latent_temperature_kwargs or {})
        }
        self.latent_temperature_validate = latent_temperature_validate
        self.latent_temperature_validation_size = (
            latent_temperature_validation_size
        )
        self.latent_temperature_warmup = int(latent_temperature_warmup)
        self.latent_temperature_history = []

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

        if rescale:
            x, log_J = self.inverse_rescale(
                x, return_unit_hypercube=return_unit_hypercube
            )
            log_prob -= log_J
            if not return_unit_hypercube:
                x, z, log_prob = self.check_prior_bounds(x, z, log_prob)

        if return_z:
            return x, log_prob, z
        return x, log_prob

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
        log_constant = -np.inf
        n_accepted = 0
        accept = None

        adapt_full_z = []
        adapt_surv_z = []
        adapt_surv_log_w = []

        while n_accepted < n_samples:
            z = self.sample_latent_distribution(self.drawsize)
            n_proposed += z.shape[0]
            if self.adapt_latent_temperature:
                adapt_full_z.append(np.asarray(z, dtype=float))
            z = self._truncation_scheme.apply_latent(self, z)
            if not len(z):
                if n_proposed > max_samples:
                    logger.warning("Reached max samples (%s)", max_samples)
                    break
                continue

            x, log_q, z = self.backward_pass(
                z,
                rescale=True,
                return_z=True,
                return_unit_hypercube=self.map_to_unit_hypercube,
            )
            x, log_q, z = self._truncation_scheme.apply_after_backward(
                self, x, log_q, z
            )
            if not len(x):
                if n_proposed > max_samples:
                    logger.warning("Reached max samples (%s)", max_samples)
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
                        logger.warning("Reached max samples (%s)", max_samples)
                        break
                    continue

            log_w = self.compute_weights(x, log_q)

            if self.adapt_latent_temperature:
                adapt_surv_z.append(np.asarray(z, dtype=float))
                adapt_surv_log_w.append(np.asarray(log_w, dtype=float))

            if self.accumulate_weights:
                samples = np.concatenate([samples, x])
                log_weights = np.concatenate([log_weights, log_w])
                log_constant = max(np.nanmax(log_w), log_constant)
                log_n_expected = logsumexp(log_weights - log_constant)

                logger.debug(
                    "Drawn %s - n expected: %s / %s",
                    samples.size,
                    np.exp(log_n_expected),
                    n_samples,
                )

                if log_n_expected >= log_n:
                    log_u = np.log(self.rng.random(len(log_weights)))
                    accept = (log_weights - log_constant) > log_u
                    n_accepted = np.sum(accept)
                if n_proposed > max_samples:
                    logger.warning("Reached max samples (%s)", max_samples)
                    break
            else:
                log_w -= log_w.max()
                log_u = np.log(self.rng.random(len(log_w)))
                accept = log_w > log_u
                n_accept_batch = accept.sum()
                m = min(n_samples - n_accepted, n_accept_batch)
                samples[n_accepted : n_accepted + m] = x[accept][:m]
                n_accepted += n_accept_batch
                logger.debug("n accepted: %s / %s", n_accepted, n_samples)
                if n_proposed > max_samples:
                    logger.warning("Reached max samples (%s)", max_samples)
                    break

        if self.accumulate_weights:
            if accept is None or len(accept) != len(samples):
                log_u = np.log(self.rng.random(len(log_weights)))
                accept = (log_weights - log_constant) > log_u
            logger.debug("Total number of samples: %s", samples.size)
            n_accepted = np.sum(accept)
            self.x = samples[accept][:n_samples]
        else:
            self.x = samples[: min(n_accepted, n_samples)]

        if self.adapt_latent_temperature and adapt_full_z:
            full_z = np.concatenate(adapt_full_z)
            r_full = np.sqrt(np.sum(full_z**2, axis=-1))
            log_w_full = np.full(full_z.shape[0], -np.inf)
            if adapt_surv_z:
                surv_z = np.concatenate(adapt_surv_z)
                surv_log_w = np.concatenate(adapt_surv_log_w)
                index = {
                    row.tobytes(): i for i, row in enumerate(full_z)
                }
                surv_idx = []
                surv_w = []
                for row, w in zip(surv_z, surv_log_w):
                    i = index.get(row.tobytes())
                    if i is not None:
                        surv_idx.append(i)
                        surv_w.append(w)
                log_w_full[surv_idx] = surv_w
            self._update_latent_temperature(r_full, log_w_full)

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
        self.populated_count += 1
        self.populated = True
        self._checked_population = False

    def _latent_truncation_radius(self) -> float:
        """Return the hard latent-space truncation radius, or ``inf``.

        The ``latent_radius`` rule applies a fixed ``|z| <= threshold`` cut for
        the batch just drawn; that threshold is the radius the reweighting must
        assume when extrapolating to another temperature (the fraction of the
        base density inside it, ``F_T``, then depends on the temperature). If
        no ``latent_radius`` rule is active the latent space is untruncated.
        """
        rule = self._get_latent_radius_rule()
        if rule is None:
            return np.inf
        threshold = getattr(rule, "threshold", np.nan)
        if threshold is not None and np.isfinite(threshold):
            return float(threshold)
        fixed = getattr(rule, "fixed_radius", False)
        if fixed is not False and np.isfinite(fixed):
            return float(fixed)
        return np.inf

    def _draw_latent_efficiency(self, temperature, n):
        """Rejection efficiency of a fresh batch drawn at ``temperature``.

        Draws ``n`` latent samples at ``temperature``, pushes them through the
        flow and the same truncation stages used by :meth:`populate`, and
        returns ``mean(w) / max(w)`` with every draw lost to truncation /
        prior bounds counted as zero weight. Returns ``-inf`` if the batch is
        entirely rejected.
        """
        original = self.latent_temperature
        self.latent_temperature = float(temperature)
        try:
            z_full = np.asarray(
                self.sample_latent_distribution(n), dtype=float
            )
            n_drawn = z_full.shape[0]
            z = self._truncation_scheme.apply_latent(self, z_full)
            if not len(z):
                return -np.inf
            x, log_q, z = self.backward_pass(
                z,
                rescale=True,
                return_z=True,
                return_unit_hypercube=self.map_to_unit_hypercube,
            )
            x, log_q, z = self._truncation_scheme.apply_after_backward(
                self, x, log_q, z
            )
            if not len(x):
                return -np.inf
            log_w = self.compute_weights(x, log_q)
        finally:
            self.latent_temperature = original

        log_w = log_w[np.isfinite(log_w)]
        if log_w.size < 2:
            return -np.inf
        # mean over *all* draws (lost ones contribute zero), divided by max.
        log_eff = (
            logsumexp(log_w) - np.log(n_drawn) - np.max(log_w)
        )
        return float(np.exp(log_eff))

    def _update_latent_temperature(self, r, log_w) -> None:
        """Update ``latent_temperature`` from the drawn batch.

        A candidate temperature is proposed by retroactively reweighting the
        just-completed batch within the secondary-ESS trust region
        (:func:`nessai.utils.temperature.optimise_latent_temperature`). When
        ``latent_temperature_validate`` is True the candidate (and, failing
        that, the geometric midpoint) is only accepted if a *fresh* validation
        batch drawn at that temperature has a higher rejection efficiency than
        one drawn at the current temperature -- so the temperature never moves
        to a value that is measurably worse.

        No update is made until the flow has been trained more than
        ``latent_temperature_warmup`` times, since the density model (and hence
        the estimate) is unreliable early in the run.
        """
        if self.training_count <= self.latent_temperature_warmup:
            self.latent_temperature_history.append(
                self.latent_temperature or 1.0
            )
            return
        # Keep zero-weight (-inf) draws: they represent samples lost to
        # truncation / prior bounds and must count against the efficiency of a
        # candidate temperature. Only genuine NaNs are dropped.
        valid = ~np.isnan(log_w) & np.isfinite(r)
        r, log_w = r[valid], log_w[valid]
        if np.sum(np.isfinite(log_w)) < 2:
            return
        current = self.latent_temperature or 1.0
        try:
            result = optimise_latent_temperature(
                log_w,
                r,
                self.prime_dims,
                temperature=current,
                radius=self._latent_truncation_radius(),
                **self.latent_temperature_kwargs,
            )
        except RuntimeError as exc:
            logger.warning("Latent-temperature adaptation skipped: %s", exc)
            return
        candidate = result["temperature"]

        if not self.latent_temperature_validate:
            self.latent_temperature = candidate
            self.latent_temperature_history.append(candidate)
            return

        if np.isclose(candidate, current, rtol=1e-3):
            self.latent_temperature_history.append(current)
            return

        n_val = self.latent_temperature_validation_size or self.drawsize
        eff_current = self._draw_latent_efficiency(current, n_val)
        eff_candidate = self._draw_latent_efficiency(candidate, n_val)
        chosen = current
        if eff_candidate > eff_current:
            chosen = candidate
        else:
            midpoint = float(np.sqrt(current * candidate))
            eff_mid = self._draw_latent_efficiency(midpoint, n_val)
            if eff_mid > eff_current:
                chosen = midpoint

        logger.info(
            "Latent-temperature validation: current=%.4f (eff=%.4f), "
            "candidate=%.4f (eff=%.4f) -> %.4f",
            current,
            eff_current,
            candidate,
            eff_candidate,
            chosen,
        )
        self.latent_temperature = chosen
        self.latent_temperature_history.append(chosen)

    def reset(self) -> None:
        """Reset the proposal."""
        super().reset()
        self._truncation_scheme.reset()

    def __getstate__(self):
        return super().__getstate__()

    def __setstate__(self, state):
        # Back-fill the latent-temperature-adaptation attributes for resume
        # files pickled before the feature existed: ``configure_population`` is
        # not re-run on resume, and ``populate`` reads
        # ``self.adapt_latent_temperature`` unconditionally.
        state.setdefault("adapt_latent_temperature", False)
        state.setdefault("latent_temperature_kwargs", {})
        state.setdefault("latent_temperature_validate", True)
        state.setdefault("latent_temperature_validation_size", None)
        state.setdefault("latent_temperature_warmup", 2)
        state.setdefault("latent_temperature_history", [])
        self.__dict__.update(state)
