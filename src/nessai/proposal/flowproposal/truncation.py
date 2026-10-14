"""Truncation rules for flow proposals."""

from __future__ import annotations

import inspect
import logging
from copy import deepcopy

import numpy as np
from scipy.special import logsumexp

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

    @staticmethod
    def _real_mask(proposal):
        """The Gaussian (non-circular) latent dimensions, or ``None`` when
        the proposal does not say (every dimension is Gaussian)."""
        real = getattr(proposal, "latent_real_mask", None)
        return real if isinstance(real, np.ndarray) else None

    def _real_dims(self, proposal, z):
        real = self._real_mask(proposal)
        if real is None or real.all():
            return z
        return z[..., real]

    def _n_real_dims(self, proposal):
        real = self._real_mask(proposal)
        return proposal.prime_dims if real is None else int(real.sum())

    def configure(self, proposal) -> None:
        n_real = self._n_real_dims(proposal)
        if n_real < proposal.prime_dims:
            logger.info(
                "Latent radius over the %s non-circular latent dimensions",
                n_real,
            )
        if self.expansion_fraction and self.expansion_fraction is not None:
            logger.info(
                "Overwriting latent-radius fuzz factor with expansion fraction"
            )
            self.fuzz = (1 + self.expansion_fraction) ** (1 / max(n_real, 1))
            logger.info(f"New latent-radius fuzz factor: {self.fuzz}")

        if not self.constant_volume_mode:
            return

        if n_real == 0:
            self.fixed_radius = np.inf
            self.fuzz = 1.0
            return

        self.fixed_radius = compute_radius(n_real, self.volume_fraction)
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
        worst_z = self._real_dims(proposal, worst_z)
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
        radius = np.sqrt(np.sum(self._real_dims(proposal, z) ** 2.0, axis=-1))
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


def _accepts_keyword(fn, name):
    """Whether ``fn`` takes the keyword argument ``name``."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        p.name == name or p.kind is inspect.Parameter.VAR_KEYWORD
        for p in params
    )


def _threshold_points(proposal):
    """The live points the truncation thresholds are taken from: the
    sampler's current live points with ``threshold_from_live_points``,
    otherwise the flow's training data (the live points at the last
    training, which lag the current ones by up to a training interval)."""
    live = getattr(proposal, "current_live_points", None)
    if getattr(proposal, "threshold_from_live_points", False) is True and (
        live is not None and len(live)
    ):
        return live
    return proposal.training_data


def _live_log_q(proposal):
    """``log q`` of the live points (:func:`_threshold_points`) under the
    density the pool is drawn from.

    ``backward_pass`` scores draws with the *tempered* latent density
    (``latent_log_prob(z, latent_temperature)``) while ``forward_pass``
    returns the untempered flow density, so with ``latent_temperature != 1``
    the live points are rescored at the same temperature before a threshold
    is taken from them.
    """
    live_points = _threshold_points(proposal).copy()
    if getattr(proposal, "sum_mirror_density", False) is True and len(
        live_points
    ):
        # the draws are scored with the density summed over the sides of the
        # boundary inversion: so are the live points
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
    """Truncate samples using the current log-proposal threshold.

    With ``prior_relative`` the thresholded quantity is ``log q - log prior``
    (minus the rejection-sampling log-weight) instead of ``log q``: the
    support then caps the weights at ``exp(-threshold)``, dropping first the
    draws where the flow is broad but the prior large.

    ``mode`` (prior-relative only) is what happens to the draws above the
    cap: ``"drop"`` discards them, so the pool follows the prior on the
    support only, and the region where the flow undercovers the live points
    (``quantile`` of them) is never proposed; ``"clip"`` keeps them with
    their weight clipped to the cap, so the pool follows ``min(p, c q)``
    everywhere: no excluded region, at the cost of the likelihood calls
    spent on the clipped part of the pool (``clipped_share``). In clip mode
    a clipped draw whose latent the flow's forward map does not return to
    (``roundtrip_tolerance`` on the latent norm; a numerical failure of the
    inverse, whose density is then wrong) is discarded.

    ``prepare`` records the live points outside the cap:
    ``excluded_live_fraction`` and ``excluded_live_mean_rank`` (their mean
    likelihood rank in [0, 1]; 0.5 if spread evenly). A rank well below 0.5
    means the excluded region lies at the low-likelihood edge, which biases
    the compression of a drop cut.
    """

    name = "log_proposal_threshold"
    _transient_defaults = {"_threshold": np.nan}
    _modes = ("drop", "clip")

    def __init__(
        self,
        quantile: float = 0.05,
        prior_relative: bool = False,
        mode: str = "drop",
        roundtrip_tolerance: float | None = 2.0,
    ) -> None:
        super().__init__()
        self.quantile = float(quantile)
        self.prior_relative = bool(prior_relative)
        if mode not in self._modes:
            raise ValueError(f"mode must be one of {self._modes}")
        if mode == "clip" and not self.prior_relative:
            raise ValueError("mode='clip' requires prior_relative=True")
        self.mode = mode
        self.roundtrip_tolerance = (
            None if roundtrip_tolerance is None else float(roundtrip_tolerance)
        )
        self.excluded_live_fraction = np.nan
        self.excluded_live_mean_rank = np.nan
        self.roundtrip_discarded = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def clip(self) -> bool:
        # rules pickled before the option existed have no attribute
        return getattr(self, "mode", "drop") == "clip"

    @property
    def log_weight_cap(self) -> float:
        """The cap on the rejection log-weights in clip mode, else ``inf``."""
        if self.clip and np.isfinite(self._threshold):
            return -self._threshold
        return np.inf

    def _levels(self, proposal, x, log_q):
        log_q = np.asarray(log_q, dtype=float)
        # rules pickled before the option existed have no attribute
        if not getattr(self, "prior_relative", False) or not len(x):
            return log_q
        log_w = np.asarray(proposal.compute_weights(x, log_q), dtype=float)
        return np.where(np.isfinite(log_w), -log_w, -np.inf)

    def prepare(self, proposal, worst_point, radius=None):
        live_points, log_q = _live_log_q(proposal)
        log_q = np.asarray(log_q)
        if getattr(self, "prior_relative", False) and len(live_points):
            # Boundary inversion with duplication appends mirror copies of
            # the live points in blocks of ``len(live_points)`` rows (row r
            # is a copy of live point r mod n). A draw is weighted with the
            # density of the side it was drawn on, so the quantile is over
            # every side's density, each paired with its live point's prior.
            n = len(live_points)
            if len(log_q) % n:
                raise RuntimeError(
                    f"{len(log_q)} rows of log q for {n} live points: not "
                    "whole blocks of mirror copies"
                )
            live_points = np.tile(live_points, len(log_q) // n)
        level = self._levels(proposal, live_points, log_q)
        self._threshold = float(np.quantile(level, self.quantile))
        self._record_excluded(live_points, level)

    def _record_excluded(self, live_points, level):
        """Fraction and mean likelihood rank of the live points (and mirror
        copies) below the threshold."""
        self.excluded_live_fraction = np.nan
        self.excluded_live_mean_rank = np.nan
        if not len(level) or "logL" not in (live_points.dtype.names or ()):
            return
        logl = np.asarray(live_points["logL"], dtype=float)
        rank = np.argsort(np.argsort(logl, kind="stable"), kind="stable")
        rank = rank / max(len(rank) - 1, 1)
        if len(level) != len(rank):
            # mirror copies: blocks of ``len(live_points)`` rows
            if len(level) % len(rank):
                return
            rank = np.tile(rank, len(level) // len(rank))
        out = level < self._threshold
        self.excluded_live_fraction = float(out.mean())
        if out.any():
            self.excluded_live_mean_rank = float(rank[out].mean())

    def apply_after_backward(self, proposal, x, log_q, z):
        levels = self._levels(proposal, x, log_q)
        keep = levels > self.threshold
        if self.clip:
            # non-finite weights are dropped in either mode
            keep = np.isfinite(levels) & self._roundtrip_keep(
                proposal, x, z, ~keep
            )
        logger.debug(
            "Accepting %s / %s samples (log-proposal threshold, %s)",
            int(keep.sum()),
            len(x),
            getattr(self, "mode", "drop"),
        )
        return get_subset_arrays(keep, x, log_q, z)

    def _roundtrip_keep(self, proposal, x, z, above):
        """Keep every draw except the ones above the cap that fail the
        round-trip check (only those: below the cap a wrong density costs
        nothing, it is rejected or dropped like any other)."""
        keep = np.ones(len(x), dtype=bool)
        tol = getattr(self, "roundtrip_tolerance", None)
        if tol is None or z is None or not above.any():
            return keep
        from .latent_distortion import roundtrip_gap

        idx = np.flatnonzero(above)
        gap = roundtrip_gap(proposal, x[idx], np.asarray(z)[idx])
        if gap is None:
            return keep
        bad = idx[~(gap <= tol)]
        keep[bad] = False
        self.roundtrip_discarded = (
            getattr(self, "roundtrip_discarded", 0) + len(bad)
        )
        return keep


class LogLevelThresholdTruncation(BaseTruncationRule):
    """Truncate on the proposal density with each mixture piece's weight
    divided out, optionally tightening over-represented pieces.

    For a mixture proposal ``q(x) = pi_p(x) f_p(x)`` whose pieces ``p`` tile
    the space (the group elements of a group-mixture flow, times its experts),
    the support is ``{log q(x) - log pi_p(x) > t}`` with ``t`` the
    ``quantile`` of the same level over the live points.  Unlike
    :class:`LogProposalThresholdTruncation` the support does not depend on the
    piece weights, so re-weighting the pieces (e.g. importance weights) moves
    draws between pieces without moving any piece's boundary -- and without
    moving the threshold, which a piece with a small weight otherwise drags
    down through its live points' low ``log q``.

    With ``share_cap``, a piece whose share of the prior mass of the support
    (estimated from the draws of the previous populate) exceeds ``share_cap``
    times its share of the live points gets its own, higher level floor: the
    level above which its prior mass is ``share_cap`` times its live share,
    but never above its lowest live point.  A symmetry image the likelihood is
    abandoning then stops holding a full-size copy of the support.  Floors are
    relaxed by ``relax`` when a piece falls well inside the cap, and are
    reset whenever the flow is retrained.  With ``pilot_size``, a retrained
    flow's floors are set before its first pool from that many latent draws
    (flow draws only, no likelihood evaluations); otherwise its first pool
    is drawn without floors.

    With ``prior_relative``, the level is ``log q(x) - log pi_p(x) -
    log prior(x)``: the support is a threshold on the piece's density
    relative to the prior, the region where ``f_p`` concentrates the most
    mass per unit prior mass.  The pool is rejection-sampled to the prior on
    the support, so on a plain ``log q`` threshold the draws from where the
    flow is broad but the prior is large (e.g. the far distance tail) get
    the largest weights and fill the pool; thresholding ``f_p / prior``
    instead removes those first, and bounds the weights by ``exp(-t)``.

    The proposal supplies ``mixture_levels(x, log_q) -> (pieces, levels)``,
    both of shape ``(n, k)``: one candidate (piece, level) per mixture
    component that can generate ``x`` (e.g. per overlapping expert), with
    level ``-inf`` for a component whose piece has zero weight.  A point is
    in the support if any candidate clears its piece's threshold, and is
    counted towards its highest-level candidate.  Proposals without the
    method are a single piece with level ``log q``: the rule is then
    :class:`LogProposalThresholdTruncation`.
    """

    name = "log_level_threshold"
    # ``_draws`` (the last pool's kept draws, for the share cap) must survive
    # the scheme's reset at the start of the next ``prepare``, which uses them
    _transient_defaults = {"_threshold": np.nan}

    def __init__(
        self,
        quantile: float = 0.005,
        share_cap: float | None = None,
        relax: float = 0.5,
        pilot_size: int | None = None,
        prior_relative: bool = False,
    ) -> None:
        super().__init__()
        self.quantile = float(quantile)
        if share_cap is not None and share_cap < 1:
            raise ValueError("share_cap must be >= 1")
        self.share_cap = None if share_cap is None else float(share_cap)
        self.relax = float(relax)
        self.pilot_size = int(pilot_size) if pilot_size else 0
        self.prior_relative = bool(prior_relative)
        self._floors = {}
        self._floors_training = None
        self._tightened = {}
        self._draws = []

    @property
    def threshold(self) -> float:
        return self._threshold

    def __getstate__(self):
        state = super().__getstate__()
        state["_draws"] = []   # up to a pool's worth of draws: not checkpointed
        return state

    @property
    def floors(self) -> dict:
        """Per-piece level floors above the global threshold."""
        return dict(self._floors)

    def _levels(self, proposal, x, log_q, z=None, log_w=None):
        """Candidate (piece, level) pairs of ``x``; ``log_w`` (the weights
        ``compute_weights`` gives ``x``) saves recomputing the prior."""
        log_q = np.asarray(log_q, dtype=float)
        fn = getattr(proposal, "mixture_levels", None)
        if not len(x):
            # an empty batch (a failed backward pass) cannot be reshaped
            # with -1 below: give it one (empty) candidate column
            return np.zeros((0, 1), dtype=int), log_q.reshape(0, 1)
        if fn is None:
            pieces, levels = np.zeros(len(x), dtype=int), log_q
        elif z is not None and _accepts_keyword(fn, "z"):
            pieces, levels = fn(x, log_q, z=z)
        else:
            pieces, levels = fn(x, log_q)
        pieces = np.asarray(pieces, dtype=int).reshape(len(x), -1)
        levels = np.asarray(levels, dtype=float).reshape(len(x), -1)
        # rules pickled before the option existed have no attribute
        if getattr(self, "prior_relative", False):
            if log_w is None:
                log_w = proposal.compute_weights(x, log_q)
            log_p = np.asarray(log_w, dtype=float) + log_q
            levels = np.where(
                np.isfinite(log_p)[:, None], levels - log_p[:, None], -np.inf
            )
        return pieces, levels

    @staticmethod
    def _best(pieces, levels):
        """Each point's highest-level candidate."""
        k = np.argmax(levels, axis=1)
        rows = np.arange(len(k))
        return pieces[rows, k], levels[rows, k]

    def _point_thresholds(self, pieces):
        thr = np.full(pieces.shape, self._threshold)
        for piece, floor in self._floors.items():
            thr[pieces == piece] = max(self._threshold, floor)
        return thr

    def prepare(self, proposal, worst_point, radius=None):
        _, log_q = _live_log_q(proposal)
        n = len(proposal.training_data)
        # rows past ``n`` are boundary-inversion mirror copies
        pieces, level = self._best(
            *self._levels(proposal, proposal.training_data, log_q[:n])
        )
        self._threshold = float(np.quantile(level, self.quantile))
        if self.share_cap is not None:
            self._update_floors(proposal, pieces, level)
        self._draws = []

    def _update_floors(self, proposal, live_pieces, live_level):
        training = getattr(proposal, "training_count", None)
        if training != self._floors_training:
            # a new flow: levels are not comparable with the old ones
            self._log_floors()
            self._floors = {}
            self._tightened = {}
            self._floors_training = training
            if not getattr(self, "pilot_size", 0):
                return
            self._draws = self._pilot_draws(proposal)
        if not self._draws:
            return
        d_piece = np.concatenate([d[0] for d in self._draws])
        d_level = np.concatenate([d[1] for d in self._draws])
        d_logw = np.concatenate([d[2] for d in self._draws])
        ok = np.isfinite(d_logw) & (d_piece >= 0)
        if not ok.any():
            return
        d_piece, d_level = d_piece[ok], d_level[ok]
        w = np.exp(d_logw[ok] - d_logw[ok].max())
        total = w.sum()
        valid = live_pieces >= 0
        all_pieces = np.union1d(np.unique(d_piece), live_pieces[valid])
        n_live = int(valid.sum())
        changed = []
        for piece in all_pieces:
            in_live = valid & (live_pieces == piece)
            live_share = (in_live.sum() + 0.5) / (n_live + 0.5 * len(all_pieces))
            sel = d_piece == piece
            share = w[sel].sum() / total
            floor = self._floors.get(piece, -np.inf)
            if share > self.share_cap * live_share:
                order = np.argsort(-d_level[sel])
                cum = np.cumsum(w[sel][order])
                idx = min(
                    int(np.searchsorted(cum, self.share_cap * live_share * total)),
                    len(order) - 1,
                )
                floor = max(floor, float(d_level[sel][order][idx]))
            elif share < 0.5 * self.share_cap * live_share:
                floor = floor - self.relax
            if in_live.any():
                floor = min(floor, float(live_level[in_live].min()))
            if floor > self._threshold:
                if self._floors.get(piece) != floor:
                    changed.append((int(piece), share, live_share))
                    self._tightened[int(piece)] = (share, live_share)
                self._floors[piece] = floor
            else:
                self._floors.pop(piece, None)
        if changed:
            logger.debug(
                "Level threshold %.3f: %d piece(s) tightened (pool/live share: "
                "%s)",
                self._threshold,
                len(self._floors),
                ", ".join(
                    f"{p}: {s:.3g}/{l:.3g}" for p, s, l in changed[:8]
                ),
            )

    def _pilot_draws(self, proposal):
        """Draws from a newly trained flow, kept and recorded as
        :meth:`apply_after_backward` does, to set its floors before its
        first pool.  Only this rule's support applies (the threshold is set
        and there are no floors yet)."""
        draws = []
        batch = int(getattr(proposal, "drawsize", None) or 10_000)
        n_drawn = n_kept = 0
        while n_drawn < self.pilot_size:
            n = min(batch, self.pilot_size - n_drawn)
            z = proposal.sample_latent_distribution(n)
            n_drawn += n
            x, log_q, z = proposal.backward_pass(
                z,
                rescale=True,
                return_z=True,
                return_unit_hypercube=getattr(
                    proposal, "map_to_unit_hypercube", False
                ),
            )
            if not len(x):
                continue
            log_w = np.asarray(proposal.compute_weights(x, log_q))
            cand_pieces, cand_levels = self._levels(
                proposal, x, log_q, z=z, log_w=log_w
            )
            keep = np.any(
                cand_levels > self._point_thresholds(cand_pieces), axis=1
            )
            if not keep.any():
                continue
            pieces, level = self._best(cand_pieces[keep], cand_levels[keep])
            draws.append((pieces, level, log_w[keep]))
            n_kept += int(keep.sum())
        logger.debug(
            "Level threshold pilot: %d / %d draws in the support",
            n_kept,
            n_drawn,
        )
        return draws

    def _log_floors(self):
        """Summarise the ending flow's floors (logged once per retrain)."""
        tightened = getattr(self, "_tightened", None)
        if not tightened:
            return
        worst = sorted(tightened.items(), key=lambda kv: -kv[1][0] / kv[1][1])
        logger.info(
            "Level threshold %.3f: %d piece(s) tightened during the last flow, "
            "%d floor(s) at retrain (pool/live share at the last tightening, "
            "most over-covered first: %s)",
            self._threshold,
            len(tightened),
            len(self._floors),
            ", ".join(f"{p}: {s:.3g}/{l:.3g}" for p, (s, l) in worst[:8]),
        )

    def apply_after_backward(self, proposal, x, log_q, z):
        log_w = None
        if getattr(self, "prior_relative", False) and len(x):
            log_w = np.asarray(proposal.compute_weights(x, log_q))
        cand_pieces, cand_levels = self._levels(
            proposal, x, log_q, z=z, log_w=log_w
        )
        keep = np.any(
            cand_levels > self._point_thresholds(cand_pieces), axis=1
        )
        if self.share_cap is not None and keep.any():
            pieces, level = self._best(cand_pieces[keep], cand_levels[keep])
            if log_w is None:
                log_w = proposal.compute_weights(x[keep], log_q[keep])
            else:
                log_w = log_w[keep]
            if self._draws is None:
                self._draws = []
            self._draws.append((pieces, level, np.asarray(log_w)))
        logger.debug(
            "Accepting %s / %s samples above the level threshold",
            int(keep.sum()),
            len(x),
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
        live_points, log_q = _live_log_q(proposal)
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
    "log_level_threshold": LogLevelThresholdTruncation,
    "weights_threshold": LogWeightThresholdTruncation,
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
        """The tightest cap on the rejection log-weights set by a rule in
        clip mode (``inf`` if none)."""
        caps = [getattr(rule, "log_weight_cap", np.inf) for rule in self.rules]
        return float(min(caps, default=np.inf))

    def reset(self) -> None:
        for rule in self.rules:
            rule.reset()
