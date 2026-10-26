"""Truncation rules for flow proposals."""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
from scipy.special import logsumexp

from ...livepoint import empty_structured_array
from ...utils import compute_radius
from ...utils.structures import get_subset_arrays

logger = logging.getLogger(__name__)

DEFAULT_TRUNCATION_METHODS = ["latent_radius"]
DEFAULT_TRUNCATION_KWARGS = {
    "latent_radius": {
        "radius_mode": "constant_volume",
        "volume_fraction": 0.95,
    }
}

LEGACY_LATENT_RADIUS_ARGUMENTS = (
    "constant_volume_mode",
    "volume_fraction",
    "fuzz",
    "expansion_fraction",
    "fixed_radius",
    "radius_mode",
    "min_radius",
    "max_radius",
    "compute_radius_with_all",
)


def get_deprecated_latent_radius_arguments(**kwargs) -> list[str]:
    """Return deprecated latent-radius arguments that were explicitly set."""
    return [
        name
        for name in LEGACY_LATENT_RADIUS_ARGUMENTS
        if kwargs[name] is not None
    ]


def get_deprecated_latent_radius_kwargs(**kwargs) -> dict:
    """Build sparse latent-radius kwargs from deprecated proposal arguments."""
    return {
        name: kwargs[name]
        for name in LEGACY_LATENT_RADIUS_ARGUMENTS
        if kwargs[name] is not None
    }


def normalise_truncation_methods(
    truncation_method=None, truncation_methods=None
) -> list[str]:
    """Normalise truncation-method input into an ordered unique list."""
    methods = (
        truncation_methods
        if truncation_methods is not None
        else truncation_method
    )
    if methods is None:
        return []
    if isinstance(methods, str):
        methods = [methods]
    return list(dict.fromkeys(methods))


def should_enable_latent_radius(latent_radius_kwargs=None) -> bool:
    """Check if latent-radius truncation should be enabled from kwargs."""
    return bool(latent_radius_kwargs)


def build_truncation_methods(
    truncation_method=None,
    truncation_methods=None,
    truncate_log_q=False,
    enforce_likelihood_threshold=False,
    latent_radius_kwargs=None,
    default_latent_radius: bool = False,
) -> list[str]:
    """Build the effective truncation-method list from legacy and new inputs."""
    if truncation_method is not None and truncation_methods is not None:
        raise ValueError(
            "Specify only one of truncation_method or truncation_methods"
        )

    methods = normalise_truncation_methods(
        truncation_method, truncation_methods
    )

    if default_latent_radius and "latent_radius" not in methods:
        methods.insert(0, "latent_radius")
    elif (
        should_enable_latent_radius(latent_radius_kwargs)
        and "latent_radius" not in methods
    ):
        methods.insert(0, "latent_radius")
    if truncate_log_q and "min_log_q" not in methods:
        methods.append("min_log_q")
    if enforce_likelihood_threshold and "likelihood_threshold" not in methods:
        methods.append("likelihood_threshold")
    return methods


def apply_default_truncation_config(
    methods,
    truncation_kwargs=None,
    *,
    default_latent_radius: bool = False,
):
    """Apply canonical default truncation configuration."""
    if default_latent_radius and not methods:
        methods = list(DEFAULT_TRUNCATION_METHODS)
    else:
        methods = list(methods)

    kwargs = deepcopy(truncation_kwargs or {})

    for name, default_kwargs in DEFAULT_TRUNCATION_KWARGS.items():
        if name not in methods:
            continue
        kwargs.setdefault(name, {})
        if not isinstance(kwargs[name], dict):
            continue
        for key, value in default_kwargs.items():
            kwargs[name].setdefault(key, value)

    return methods, kwargs


def normalise_truncation_kwargs(
    truncation_method=None,
    truncation_methods=None,
    truncation_kwargs=None,
):
    """Normalise truncation kwargs into the canonical method-keyed form."""
    if truncation_kwargs is None:
        return {}

    kwargs = deepcopy(truncation_kwargs)

    if (
        isinstance(truncation_method, str)
        and truncation_methods is None
        and truncation_method not in kwargs
        and not any(isinstance(value, dict) for value in kwargs.values())
    ):
        return {truncation_method: kwargs}

    return kwargs


class BaseTruncationRule:
    """Base class for truncation rules."""

    name = "base"
    _transient_defaults = {}

    def __init__(self) -> None:
        self.reset()

    @property
    def requires_log_likelihood(self) -> bool:
        """Indicate if the rule needs log-likelihood values."""
        return False

    def configure(self, proposal) -> None:
        """Apply any proposal-level configuration needed by the rule."""
        return None

    def prepare(self, proposal, worst_point, radius=None):
        """Prepare per-population data for the rule."""
        return None

    def apply_latent(self, proposal, z):
        """Apply truncation in latent space before the inverse pass."""
        return z

    def apply_after_backward(self, proposal, x, log_q, z):
        """Apply truncation after the inverse pass and rescaling."""
        return x, log_q, z

    def apply_after_likelihood(self, proposal, x, log_q, z):
        """Apply truncation after likelihood evaluation."""
        return x, log_q, z

    def reset(self) -> None:
        """Reset transient state."""
        for key, value in self._transient_defaults.items():
            setattr(self, key, value)

    def __getstate__(self):
        state = self.__dict__.copy()
        for key, value in self._transient_defaults.items():
            state[key] = value
        return state


class LatentRadiusTruncation(BaseTruncationRule):
    """Filter latent samples using a radial threshold."""

    name = "latent_radius"
    _transient_defaults = {"_radius": np.nan, "_threshold": np.nan}

    def __init__(
        self,
        radius_mode: str | None = None,
        fixed_radius: float | bool = False,
        min_radius: float | bool = False,
        max_radius: float | bool = 50.0,
        compute_radius_with_all: bool = False,
        constant_volume_mode: bool = False,
        volume_fraction: float = 0.95,
        fuzz: float = 1.0,
        expansion_fraction: float | None = 4.0,
    ) -> None:
        super().__init__()
        self.fixed_radius = self._coerce_radius(
            fixed_radius, name="fixed_radius"
        )
        self.min_radius = self._coerce_radius(min_radius, name="min_radius")
        self.max_radius = self._coerce_radius(max_radius, name="max_radius")
        self.compute_radius_with_all = compute_radius_with_all
        self.volume_fraction = float(volume_fraction)
        self.fuzz = float(fuzz)
        self.expansion_fraction = expansion_fraction
        self.radius_mode = self._resolve_radius_mode(
            radius_mode,
            fixed_radius=self.fixed_radius,
            constant_volume_mode=constant_volume_mode,
        )

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def threshold(self) -> float:
        return self._threshold

    @staticmethod
    def _coerce_radius(value, *, name):
        if value in (False, None):
            return False
        if not isinstance(value, (int, float)):
            raise RuntimeError(f"{name} must be an int or float")
        return float(value)

    @staticmethod
    def _resolve_radius_mode(
        radius_mode, *, fixed_radius, constant_volume_mode
    ):
        if radius_mode is None:
            if constant_volume_mode:
                radius_mode = "constant_volume"
            elif fixed_radius is not False:
                radius_mode = "fixed"
            else:
                radius_mode = "adaptive"

        if radius_mode not in {"adaptive", "fixed", "constant_volume"}:
            raise ValueError(
                "Unknown radius mode: "
                f"{radius_mode}. Choose from: adaptive, fixed, constant_volume"
            )
        if radius_mode == "fixed" and fixed_radius is False:
            raise ValueError(
                "Fixed radius mode requires `fixed_radius` to be specified"
            )
        return radius_mode

    @property
    def constant_volume_mode(self) -> bool:
        return self.radius_mode == "constant_volume"

    def to_kwargs(self) -> dict:
        """Return keyword arguments that reconstruct the rule."""
        return {
            "radius_mode": self.radius_mode,
            "fixed_radius": self.fixed_radius,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
            "compute_radius_with_all": self.compute_radius_with_all,
            "constant_volume_mode": self.constant_volume_mode,
            "volume_fraction": self.volume_fraction,
            "fuzz": self.fuzz,
            "expansion_fraction": self.expansion_fraction,
        }

    def configure(self, proposal) -> None:
        if self.expansion_fraction and self.expansion_fraction is not None:
            logger.info(
                "Overwriting latent-radius fuzz factor with expansion fraction"
            )
            self.fuzz = (1 + self.expansion_fraction) ** (
                1 / proposal.prime_dims
            )
            logger.info(f"New latent-radius fuzz factor: {self.fuzz}")

        if not self.constant_volume_mode:
            return

        self.fixed_radius = compute_radius(
            proposal.prime_dims, self.volume_fraction
        )
        self.fuzz = 1.0

        if self.max_radius and self.max_radius < self.fixed_radius:
            logger.warning(
                "Max radius is less than the constant-volume radius. "
                "Disabling max radius."
            )
            self.max_radius = False
        if self.min_radius and self.min_radius > self.fixed_radius:
            logger.warning(
                "Min radius is greater than the constant-volume radius. "
                "Disabling min radius."
            )
            self.min_radius = False

    def _compute_radius(self, proposal, worst_point, radius=None) -> float:
        if radius is not None:
            return float(radius)

        if self.radius_mode in {"fixed", "constant_volume"}:
            if self.fixed_radius is False:
                raise RuntimeError(
                    "Fixed radius mode requires `fixed_radius` to be specified"
                )
            return float(self.fixed_radius)

        if self.compute_radius_with_all:
            if proposal.training_data is None or not len(
                proposal.training_data
            ):
                raise RuntimeError(
                    "compute_radius_with_all requires training_data to be set"
                )
            worst_point = proposal.training_data

        worst_z = proposal.forward_pass(
            worst_point, rescale=True, compute_radius=True
        )[0]
        radius = float(np.sqrt(np.sum(worst_z**2.0, axis=-1)).max())
        if self.max_radius and radius > self.max_radius:
            radius = self.max_radius
        if self.min_radius and radius < self.min_radius:
            radius = self.min_radius
        return float(radius)

    def prepare(self, proposal, worst_point, radius=None):
        radius = self._compute_radius(proposal, worst_point, radius=radius)
        self._radius = radius
        self._threshold = self.fuzz * radius

    def apply_latent(self, proposal, z):
        radius = np.sqrt(np.sum(z**2.0, axis=-1))
        keep = radius <= self.threshold
        logger.debug(
            "Discarding %s latent samples outside radius threshold",
            len(z) - int(keep.sum()),
        )
        return z[keep]


class MinLogQTruncation(BaseTruncationRule):
    """Truncate samples using the minimum live-point log q."""

    name = "min_log_q"
    _transient_defaults = {"_min_log_q": np.nan}

    @property
    def min_log_q(self) -> float:
        return self._min_log_q

    def prepare(self, proposal, worst_point, radius=None):
        if proposal.training_data is None or not len(proposal.training_data):
            raise RuntimeError(
                "min_log_q truncation requires training_data to be set"
            )

        self._min_log_q = float(
            proposal.forward_pass(proposal.training_data)[1].min()
        )
        logger.debug("Truncating with log_q=%0.3f", self.min_log_q)

    def apply_after_backward(self, proposal, x, log_q, z):
        keep = log_q > self.min_log_q
        logger.debug(
            "Discarding %s samples below log_q_min",
            len(log_q) - int(keep.sum()),
        )
        return get_subset_arrays(keep, x, log_q, z)


def _threshold_points(proposal):
    """Live points to set thresholds from: the current ones with
    ``threshold_from_live_points``, else the flow's training data."""
    live = getattr(proposal, "current_live_points", None)
    if getattr(proposal, "threshold_from_live_points", False) is True and (
        live is not None and len(live)
    ):
        return live
    return proposal.training_data


def _with_auxiliary_parameters(proposal, points):
    """``points`` with the auxiliary parameters of the reparameterisations
    (e.g. an angle's radius) drawn once, so their prior can be evaluated."""
    names = getattr(getattr(proposal, "x_dtype", None), "names", None) or ()
    present = points.dtype.names or ()
    missing = [n for n in names if n not in present]
    if not missing or not len(points):
        return points
    x_prime, _ = proposal.rescale(points.copy())
    x, _ = proposal.inverse_rescale(x_prime)
    out = empty_structured_array(len(points), dtype=proposal.x_dtype)
    for name in present:
        if name in names:
            out[name] = points[name]
    # rows past ``len(points)`` are boundary-inversion mirror copies
    for name in missing:
        out[name] = x[name][: len(points)]
    return out


def _live_log_q(proposal):
    """The threshold live points and their ``log q``, scored as the draws
    are (latent temperature, summed mirror density)."""
    live_points = _with_auxiliary_parameters(
        proposal, _threshold_points(proposal).copy()
    )
    if getattr(proposal, "sum_mirror_density", False) is True and len(
        live_points
    ):
        sides = proposal.mirror_log_q(live_points)
        return live_points, logsumexp(sides, axis=0)
    z, log_q = proposal.forward_pass(live_points)
    temperature = getattr(proposal, "latent_temperature", None)
    if temperature not in (None, 1.0):
        log_q = (
            log_q
            + proposal.latent_log_prob(z, temperature)
            - proposal.latent_log_prob(z, None)
        )
    return live_points, log_q


class LogProposalThresholdTruncation(BaseTruncationRule):
    """Truncate draws below a quantile of the live points' ``log q``.

    With ``prior_relative`` the level is ``log q - log prior``, which caps
    the rejection weight at ``exp(-threshold)``. ``mode="drop"`` discards
    draws above the cap; ``mode="clip"`` keeps them with the weight clipped.
    ``lost_mass`` sets the threshold so that this share of the live points'
    mass is lost, instead of from ``quantile``.
    """

    name = "log_proposal_threshold"
    _transient_defaults = {"_threshold": np.nan}
    _modes = ("drop", "clip")

    def __init__(
        self,
        quantile: float = 0.005,
        prior_relative: bool = True,
        mode: str = "clip",
        lost_mass: float | None = None,
    ) -> None:
        super().__init__()
        self.quantile = float(quantile)
        self.prior_relative = bool(prior_relative)
        if mode not in self._modes:
            raise ValueError(f"mode must be one of {self._modes}")
        if mode == "clip" and not self.prior_relative:
            raise ValueError("mode='clip' requires prior_relative=True")
        if lost_mass is not None:
            if not self.prior_relative:
                raise ValueError("lost_mass requires prior_relative=True")
            if not 0 < lost_mass < 1:
                raise ValueError("lost_mass must be in (0, 1)")
            lost_mass = float(lost_mass)
        self.lost_mass = lost_mass
        self.mode = mode
        self.lost_live_mass = np.nan

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def clip(self) -> bool:
        return self.mode == "clip"

    @property
    def log_weight_cap(self) -> float:
        """The cap on the rejection log-weights in clip mode, else ``inf``."""
        if self.clip and np.isfinite(self._threshold):
            return -self._threshold
        return np.inf

    def _levels(self, proposal, x, log_q):
        log_q = np.asarray(log_q, dtype=float)
        if not self.prior_relative or not len(x):
            return log_q
        log_w = np.asarray(proposal.compute_weights(x, log_q), dtype=float)
        return np.where(np.isfinite(log_w), -log_w, -np.inf)

    def prepare(self, proposal, worst_point, radius=None):
        live_points, log_q = _live_log_q(proposal)
        log_q = np.asarray(log_q)
        if self.prior_relative and len(live_points):
            # mirror copies come in blocks of ``len(live_points)`` rows
            n = len(live_points)
            if len(log_q) % n:
                raise RuntimeError(
                    f"{len(log_q)} rows of log q for {n} live points: not "
                    "whole blocks of mirror copies"
                )
            live_points = np.tile(live_points, len(log_q) // n)
        level = self._levels(proposal, live_points, log_q)
        if self.lost_mass is not None and len(level):
            self._threshold = self._threshold_for_lost_mass(
                level, self.lost_mass
            )
        else:
            self._threshold = float(np.quantile(level, self.quantile))
        if self.prior_relative and len(level):
            self.lost_live_mass = self._lost_mass(level, self._threshold)

    def _lost_mass(self, level, threshold):
        """Mean share of the live points' mass lost at ``threshold``."""
        level = np.asarray(level, dtype=float)
        if not self.clip:
            return float(np.mean(level < threshold))
        return float(np.mean(-np.expm1(np.minimum(level - threshold, 0.0))))

    def _threshold_for_lost_mass(self, level, target, tol=1e-6):
        """Bisect for the threshold at which :meth:`_lost_mass` is
        ``target``."""
        level = np.asarray(level, dtype=float)
        finite = level[np.isfinite(level)]
        if not len(finite):
            return float(np.quantile(level, self.quantile))
        if not self.clip:
            return float(np.quantile(level, target))
        lo, hi = float(finite.min()), float(finite.max())
        if self._lost_mass(level, lo) >= target:
            return lo
        if self._lost_mass(level, hi) <= target:
            return hi
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if self._lost_mass(level, mid) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def apply_after_backward(self, proposal, x, log_q, z):
        levels = self._levels(proposal, x, log_q)
        if self.clip:
            # non-finite weights are dropped in either mode
            keep = np.isfinite(levels)
        else:
            keep = levels > self.threshold
        logger.debug(
            "Accepting %s / %s samples (log-proposal threshold, %s)",
            int(keep.sum()),
            len(x),
            self.mode,
        )
        return get_subset_arrays(keep, x, log_q, z)


class LikelihoodThresholdTruncation(BaseTruncationRule):
    """Truncate samples using the current likelihood threshold."""

    name = "likelihood_threshold"
    _transient_defaults = {"_threshold": np.nan}

    @property
    def requires_log_likelihood(self) -> bool:
        return True

    @property
    def threshold(self) -> float:
        return self._threshold

    def prepare(self, proposal, worst_point, radius=None):
        threshold = np.asarray(worst_point["logL"], dtype=float).reshape(-1)[0]
        if not np.isfinite(threshold):
            logger.debug(
                "Worst point does not have a finite log-likelihood. "
                "Disabling likelihood-threshold truncation."
            )
            threshold = -np.inf
        self._threshold = float(threshold)

    def apply_after_likelihood(self, proposal, x, log_q, z):
        keep = x["logL"] > self.threshold
        logger.debug(
            "Accepting %s / %s samples above logL threshold",
            int(keep.sum()),
            len(x),
        )
        return get_subset_arrays(keep, x, log_q, z)


class LogWeightThresholdTruncation(BaseTruncationRule):
    """Truncate samples using the current log-weight threshold.

    The threshold is set based on the current live points.
    """

    name = "log_weight_threshold"
    _transient_defaults = {"_threshold": np.nan}

    def __init__(
        self, quantile: float = 0.05, enlargement: float = 0.0
    ) -> None:
        super().__init__()
        self.quantile = quantile
        self.enlargement = enlargement

    @property
    def threshold(self) -> float:
        return self._threshold

    def prepare(self, proposal, worst_point, radius=None):
        live_points = proposal.training_data.copy()
        log_q = proposal.forward_pass(live_points)[1]
        if proposal.map_to_unit_hypercube:
            live_points = proposal.model.to_unit_hypercube(live_points)
        log_w = proposal.compute_weights(
            live_points,
            log_q=log_q,
        )

        self._threshold = np.quantile(log_w, self.quantile) - self.enlargement

    def apply_after_backward(self, proposal, x, log_q, z):
        log_w = proposal.compute_weights(
            x,
            log_q=log_q,
        )
        keep = log_w > self.threshold
        logger.debug(
            "Accepting %s / %s samples above log-weight threshold",
            int(keep.sum()),
            len(x),
        )
        return get_subset_arrays(keep, x, log_q, z)


TRUNCATION_REGISTRY = {
    "latent_radius": LatentRadiusTruncation,
    "min_log_q": MinLogQTruncation,
    "likelihood_threshold": LikelihoodThresholdTruncation,
    "log_proposal_threshold": LogProposalThresholdTruncation,
    "log_weight_threshold": LogWeightThresholdTruncation,
}


def get_truncation_rule_class(name: str):
    """Get the truncation rule class for a configured method name."""
    try:
        return TRUNCATION_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown truncation method: {name}") from exc


class TruncationScheme:
    """Apply an ordered set of truncation rules."""

    def __init__(self, rules: list[BaseTruncationRule] | None = None) -> None:
        self.rules = []
        for rule in rules or []:
            self.add_rule(rule)

    @property
    def rule_names(self) -> list[str]:
        return [rule.name for rule in self.rules]

    @property
    def requires_log_likelihood(self) -> bool:
        return any(rule.requires_log_likelihood for rule in self.rules)

    def has_rule(self, name: str) -> bool:
        return any(rule.name == name for rule in self.rules)

    def get_rule(self, name: str):
        for rule in self.rules:
            if rule.name == name:
                return rule
        return None

    def add_rule(
        self, rule: BaseTruncationRule, index: int | None = None
    ) -> None:
        if self.has_rule(rule.name):
            raise ValueError(f"Duplicate truncation rule: {rule.name}")
        if index is None:
            self.rules.append(rule)
        else:
            self.rules.insert(index, rule)

    def configure(self, proposal) -> None:
        for rule in self.rules:
            rule.configure(proposal)

    def prepare(self, proposal, worst_point, radius=None):
        self.reset()
        for rule in self.rules:
            rule.prepare(proposal, worst_point, radius=radius)

    def apply_latent(self, proposal, z):
        for rule in self.rules:
            z = rule.apply_latent(proposal, z)
        return z

    def apply_after_backward(self, proposal, x, log_q, z):
        for rule in self.rules:
            x, log_q, z = rule.apply_after_backward(proposal, x, log_q, z)
        return x, log_q, z

    def apply_after_likelihood(self, proposal, x, log_q, z):
        for rule in self.rules:
            x, log_q, z = rule.apply_after_likelihood(proposal, x, log_q, z)
        return x, log_q, z

    @property
    def log_weight_cap(self) -> float:
        """Tightest rejection log-weight cap of the rules (``inf`` if none)."""
        caps = [getattr(rule, "log_weight_cap", np.inf) for rule in self.rules]
        return float(min(caps, default=np.inf))

    def reset(self) -> None:
        for rule in self.rules:
            rule.reset()
