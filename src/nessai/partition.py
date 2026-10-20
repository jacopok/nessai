# -*- coding: utf-8 -*-
"""Partitioned nested sampling: the geometry of a split.

When the live points separate along a few coordinates where a multimodality
is expected (e.g. the sky azimuth around a baseline), the nested sampler can
split the prior into frozen regions and continue with an independent run in
each, in lockstep (see :py:meth:`nessai.samplers.nestedsampler.NestedSampler.configure_partition`).
This module holds what defines a split:

* :func:`hold_out_bit`, the persistent random half of the live points that
  estimates the regions' shares of the prior volume, never used to choose the
  split;
* coordinates (:class:`ParameterCoordinates`), the few named functions of the
  parameters the split is made in, frozen at the split;
* splitters (:class:`GapSplitter`) that propose a cut from the other half of
  the live points, and the cuts themselves (:class:`IntervalCut`,
  :class:`LinearCut`);
* :class:`RegionTruncation`, the truncation rule that keeps a region's
  proposal inside it;
* :func:`plot_partition`, the diagnostic plot.

Any frozen partition gives an unbiased evidence; the choice of the cut only
affects the efficiency, through how well each region's flow can cover it.
"""

import importlib
import logging

import numpy as np

from .proposal.flowproposal.truncation import BaseTruncationRule
from .utils.structures import get_subset_arrays

logger = logging.getLogger(__name__)

__all__ = [
    "hold_out_bit",
    "SplitCoordinates",
    "ParameterCoordinates",
    "Cut",
    "IntervalCut",
    "LinearCut",
    "GapSplitter",
    "CoreSplitter",
    "FirstOf",
    "Region",
    "RegionMembership",
    "RegionTruncation",
    "resolve",
    "plot_partition",
    "respawn_generators",
    "check_clone",
]

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def hold_out_bit(points):
    """A fixed random bit for each live point: 1 for the held-out half.

    A splitmix64 hash of the bits of ``logL``, mixed with the iteration of
    birth ``it``: independent of where the point is, fixed for its whole
    life, and the same after a resume. The splitters only see the points
    with bit 0; the shares of the prior volume are estimated from the points
    with bit 1, which no earlier check has looked at, so the share is not
    biased by when the split is made.
    """
    logl = np.ascontiguousarray(points["logL"], dtype=np.float64)
    x = logl.view(np.uint64).copy()
    names = points.dtype.names or ()
    with np.errstate(over="ignore"):
        if "it" in names:
            it = np.asarray(points["it"]).astype(np.int64).view(np.uint64)
            x ^= (it + np.uint64(1)) * _GOLDEN
        x ^= x >> np.uint64(30)
        x *= _M1
        x ^= x >> np.uint64(27)
        x *= _M2
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(63)).astype(int)


def resolve(spec, default_module=None):
    """A class from ``"module:Class"`` (or a name in ``default_module``),
    or ``spec`` itself if it is not a string."""
    if not isinstance(spec, str):
        return spec
    if ":" in spec:
        module, name = spec.split(":", 1)
        return getattr(importlib.import_module(module), name)
    if default_module is not None and hasattr(default_module, spec):
        return getattr(default_module, spec)
    raise ValueError(f"Cannot resolve {spec!r}: use 'module:Class'")


def build(spec, default, default_module=None):
    """An instance from a JSON-able ``spec``: ``None`` (``default()``), a
    class (or ``"module:Class"``), a dictionary with the class under
    ``"class"`` and its keyword arguments, or an instance."""
    if spec is None:
        return default()
    if isinstance(spec, dict):
        kwargs = dict(spec)
        cls = resolve(kwargs.pop("class", default), default_module)
        return cls(**kwargs)
    if isinstance(spec, str) or isinstance(spec, type):
        return resolve(spec, default_module)()
    return spec


# -- coordinates --------------------------------------------------------------
class SplitCoordinates:
    """The coordinates a split is made in: a few functions of the
    parameters.

    ``__call__(points)`` returns an ``(n, d)`` array; :attr:`names` and
    :attr:`periods` (``None`` or the period of each coordinate, ``None`` if
    not periodic) describe it. :meth:`freeze` returns the map as it is now,
    independent of the sampler from then on: a region is defined by a fixed
    function of the parameters, so anything the map takes from the proposal
    (a frame that moves between trainings) must be copied there.
    """

    names = ()
    periods = None

    def configure(self, model):
        """Called once with the model, before the first check."""

    def freeze(self, proposal=None):
        return self

    def __call__(self, points):
        raise NotImplementedError

    @property
    def dims(self):
        return len(self.names)

    def period_list(self):
        if self.periods is None:
            return [None] * self.dims
        return list(self.periods)


class ParameterCoordinates(SplitCoordinates):
    """Model parameters as the split coordinates.

    Parameters
    ----------
    names : list of str
        The parameters.
    periods : list, optional
        The period of each (``None`` if not periodic). By default taken from
        the model's bilby priors (``boundary="periodic"``) when it has them,
        else none.
    """

    def __init__(self, names, periods=None):
        self.names = tuple(names)
        self.periods = None if periods is None else list(periods)

    def configure(self, model):
        missing = set(self.names) - set(model.names)
        if missing:
            raise ValueError(f"Split coordinates {missing} are not parameters")
        if self.periods is not None:
            return
        priors = getattr(model, "bilby_priors", None)
        periods = []
        for name in self.names:
            prior = priors.get(name) if priors is not None else None
            if getattr(prior, "boundary", None) == "periodic":
                periods.append(float(prior.maximum - prior.minimum))
            else:
                periods.append(None)
        self.periods = periods

    def __call__(self, points):
        return np.column_stack(
            [np.asarray(points[n], dtype=float) for n in self.names]
        ).reshape(len(points), len(self.names))


# -- cuts ---------------------------------------------------------------------
class Cut:
    """A partition of the split coordinates into :attr:`n_labels` parts."""

    n_labels = 2

    def classify(self, coords):
        raise NotImplementedError

    def describe(self):
        return type(self).__name__

    def draw(self, ax, dims):
        """Mark the cut on a histogram (one dimension) or a scatter plot
        (two)."""


class IntervalCut(Cut):
    """Cut points along one coordinate (column ``dim``).

    Without a period, the labels are the ``len(points) + 1`` intervals in
    order. With one, they are the ``len(points)`` arcs of the circle, arc
    ``i`` starting at cut point ``i`` (the last wraps round to the first).
    """

    def __init__(self, points, period=None, dim=0):
        self.points = np.sort(np.asarray(points, dtype=float))
        self.period = None if period is None else float(period)
        self.dim = int(dim)
        if self.period is not None:
            self.points = np.mod(self.points, self.period)
            self.points.sort()
            if len(self.points) < 2:
                raise ValueError("A circle needs two cut points")
        elif not len(self.points):
            raise ValueError("No cut points")
        self.info = {}

    @property
    def n_labels(self):
        if self.period is None:
            return len(self.points) + 1
        return len(self.points)

    def classify(self, coords):
        c = np.asarray(coords, dtype=float)
        c = c[:, self.dim] if c.ndim == 2 else c
        if self.period is None:
            return np.searchsorted(self.points, c, side="right")
        c = np.mod(c, self.period)
        i = np.searchsorted(self.points, c, side="right") - 1
        # before the first cut point: the arc that wraps round
        return np.where(i < 0, len(self.points) - 1, i)

    def describe(self):
        pts = ", ".join(f"{p:.4g}" for p in self.points)
        kind = f"period {self.period:.4g}" if self.period else "intervals"
        return f"cuts at [{pts}] ({kind})"

    def draw(self, ax, dims):
        if dims == 1 or self.dim == 0:
            for p in self.points:
                ax.axvline(p, color="k", ls="--", lw=1)
        else:
            for p in self.points:
                ax.axhline(p, color="k", ls="--", lw=1)


class LinearCut(Cut):
    """A half-plane: label 1 where ``coords @ normal > offset``."""

    def __init__(self, normal, offset):
        self.normal = np.asarray(normal, dtype=float)
        self.offset = float(offset)
        self.info = {}

    def classify(self, coords):
        c = np.asarray(coords, dtype=float).reshape(len(coords), -1)
        return (c @ self.normal > self.offset).astype(int)

    def describe(self):
        terms = " + ".join(f"{n:.4g} x{i}" for i, n in enumerate(self.normal))
        return f"{terms} > {self.offset:.4g}"

    def draw(self, ax, dims):
        if dims != 2:
            return
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        a, b = self.normal
        if abs(b) > 1e-12:
            xs = np.array([x0, x1])
            ax.plot(xs, (self.offset - a * xs) / b, "k--", lw=1)
        else:
            ax.axvline(self.offset / a, color="k", ls="--", lw=1)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)


# -- splitters ----------------------------------------------------------------
class GapSplitter:
    """Cut one coordinate at its empty gaps.

    A gap between consecutive values (on a circle, including the gap across
    the seam) is empty if it is wider than ``gap_factor`` times the local
    spacing: the mean spacing of the ``k`` values on its wider side. Among
    ``n`` values drawn from a smooth density the largest gap is about
    ``ln n`` local spacings; the default factor ``ln n + 12`` (with ``k =
    20``, whose noise the margin absorbs) gave no cut in 21000 trials of
    Gaussian, Cauchy and edge-truncated samples of 180 to 2000 values,
    while the gap between two detached modes is typically ten times
    wider. The cut points are the midpoints of
    the empty gaps (wider than ``min_gap`` too). A part with fewer than
    ``min_points`` values is merged into its neighbour across the narrower
    of its two gaps. A cut is proposed if at least two parts remain and the
    smallest holds at most ``max_share`` of the values.

    Parameters
    ----------
    max_share : float
        Split only once some part holds at most this share.
    min_points : int
        Minimum number of values (of the half that chooses the split) in a
        part.
    gap_factor : float, optional
        Minimum width of an empty gap in local spacings (default
        ``ln n + 12``).
    min_gap : float
        Minimum width of an empty gap in the coordinate's units.
    k : int
        Number of values on each side for the local spacing.
    dim : int
        The coordinate (column) to cut.
    """

    def __init__(
        self,
        max_share=0.25,
        min_points=25,
        gap_factor=None,
        min_gap=0.0,
        k=20,
        dim=0,
    ):
        self.max_share = float(max_share)
        self.min_points = int(min_points)
        self.gap_factor = None if gap_factor is None else float(gap_factor)
        self.min_gap = float(min_gap)
        self.k = int(k)
        self.dim = int(dim)

    def _local_spacing(self, v, period):
        """Mean spacing of the ``k`` values on the wider side of each
        gap."""
        n = len(v)
        k = min(self.k, n - 1)
        i = np.arange(n)
        if period is not None:
            ext = np.concatenate([v - period, v, v + period])
            lo = ext[n + i] - ext[n + i - k]
            hi = ext[n + i + 1 + k] - ext[n + i + 1]
        else:
            lo = np.full(n, np.inf)
            hi = np.full(n, np.inf)
            j = i[:-1]
            ok = j - k >= 0
            lo[:-1][ok] = v[j[ok]] - v[j[ok] - k]
            ok = j + 1 + k < n
            hi[:-1][ok] = v[j[ok] + 1 + k] - v[j[ok] + 1]
            # at the ends: the side that exists
            lo = np.where(np.isfinite(lo), lo, hi)
            hi = np.where(np.isfinite(hi), hi, lo)
            lo, hi = lo[:-1], hi[:-1]
        return np.maximum(lo, hi) / k

    def propose(
        self,
        coords,
        periods,
        points=None,
        proposal=None,
        max_labels=None,
        force=False,
    ):
        """The cut, or None. ``force`` skips the ``max_share`` test (to show
        the candidate cut in the diagnostics)."""
        c = np.asarray(coords, dtype=float)
        c = c[:, self.dim] if c.ndim == 2 else c
        period = periods[self.dim] if periods is not None else None
        n = len(c)
        if n < 2 * self.min_points:
            return None
        if period is not None:
            v = np.sort(np.mod(c, period))
            gaps = np.diff(np.concatenate([v, [v[0] + period]]))
            # gap i runs from v[i] to v[i + 1] (the last across the seam)
            mids = np.mod(v + 0.5 * gaps, period)
        else:
            v = np.sort(c)
            gaps = np.diff(v)
            mids = v[:-1] + 0.5 * gaps
        factor = (
            self.gap_factor if self.gap_factor is not None else np.log(n) + 12
        )
        spacing = self._local_spacing(v, period)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = gaps / spacing
        wide = np.flatnonzero((ratio > factor) & (gaps > self.min_gap))
        cuts = list(wide)
        while True:
            parts = self._parts(cuts, n, period is not None)
            if parts is None:
                return None
            sizes = np.array([e - s for s, e in parts])
            small = np.flatnonzero(sizes < self.min_points)
            too_many = max_labels is not None and len(parts) > max_labels
            if not len(small) and not too_many:
                break
            # merge the smallest part (or, with too many parts, the one
            # behind the narrowest gap) across its narrower bounding gap
            j = int(small[np.argmin(sizes[small])]) if len(small) else None
            cuts = self._merge(cuts, gaps, j, period is not None)
        share = sizes.min() / n
        if share > self.max_share and not force:
            return None
        cut = IntervalCut(mids[cuts], period=period, dim=self.dim)
        counts = np.bincount(cut.classify(c), minlength=cut.n_labels)
        cut.info = dict(
            shares=(counts / n).tolist(),
            gaps=sorted(gaps[cuts].tolist()),
            gap_ratios=sorted(ratio[cuts].tolist()),
            gap_factor=float(factor),
        )
        return cut

    @staticmethod
    def _parts(cuts, n, periodic):
        """``(start, end)`` index ranges into the sorted values of each part
        (on a circle, a range may wrap past ``n``); ``None`` if fewer than
        two."""
        cuts = sorted(cuts)
        if periodic:
            if len(cuts) < 2:
                return None
            return [
                (
                    cuts[i] + 1,
                    cuts[(i + 1) % len(cuts)]
                    + 1
                    + (n if i == len(cuts) - 1 else 0),
                )
                for i in range(len(cuts))
            ]
        if not cuts:
            return None
        bounds = [0] + [c + 1 for c in cuts] + [n]
        return list(zip(bounds[:-1], bounds[1:]))

    @staticmethod
    def _merge(cuts, gaps, j, periodic):
        """Remove a cut: one of part ``j``'s two (the narrower gap), or the
        narrowest of all if ``j`` is None."""
        cuts = sorted(cuts)
        if j is None:
            return [c for c in cuts if c != min(cuts, key=lambda c: gaps[c])]
        if periodic:
            bounding = [cuts[j], cuts[(j + 1) % len(cuts)]]
        else:
            bounding = [cuts[k] for k in (j - 1, j) if 0 <= k < len(cuts)]
        drop = min(bounding, key=lambda c: gaps[c])
        return [c for c in cuts if c != drop]


class CoreSplitter:
    """Cut a concentrated core of one coordinate from the rest.

    The core is the shortest arc (interval, without a period) holding
    ``coverage`` of the values, widened ``widen`` times about its middle;
    a cut is proposed once it is at most ``max_width`` of the period (or of
    the values' range) wide and the rest holds at least ``min_points`` and at
    most ``max_share`` of the values. It separates a mode from a diffuse
    tail it is not detached from by an empty gap (in the stalled two-L runs,
    the ring of sky positions round the baseline, at 1% of the live points
    over the whole circle of ``sky_u``, against a core 0.1 wide). Any frozen
    cut gives an unbiased evidence: widening keeps the core's own tails out
    of the rest, where they would make a sharp edge for its flow.

    Parameters
    ----------
    coverage : float
        Share of the values in the arc that defines the core.
    widen : float
        Factor applied to the arc's width about its middle.
    max_width : float
        Maximum width of the widened core, as a share of the period.
    max_share : float
        Maximum share of the values outside the core.
    min_points : int
        Minimum number of values outside the core (of the half that chooses
        the split).
    dim : int
        The coordinate (column) to cut.
    """

    def __init__(
        self,
        coverage=0.9,
        widen=3.0,
        max_width=0.25,
        max_share=0.25,
        min_points=10,
        dim=0,
    ):
        self.coverage = float(coverage)
        self.widen = float(widen)
        self.max_width = float(max_width)
        self.max_share = float(max_share)
        self.min_points = int(min_points)
        self.dim = int(dim)

    def propose(
        self,
        coords,
        periods,
        points=None,
        proposal=None,
        max_labels=None,
        force=False,
    ):
        c = np.asarray(coords, dtype=float)
        c = c[:, self.dim] if c.ndim == 2 else c
        period = periods[self.dim] if periods is not None else None
        n = len(c)
        m = int(np.ceil(self.coverage * n))
        if (
            n < 2 * self.min_points
            or m >= n
            or (max_labels is not None and max_labels < 2)
        ):
            return None
        if period is not None:
            v = np.sort(np.mod(c, period))
            ext = np.concatenate([v, v + period])
            widths = ext[np.arange(n) + m - 1] - v
            i = int(np.argmin(widths))
            lo, hi = v[i], ext[i + m - 1]
            scale = period
        else:
            v = np.sort(c)
            widths = v[m - 1 :] - v[: n - m + 1]
            i = int(np.argmin(widths))
            lo, hi = v[i], v[i + m - 1]
            scale = v[-1] - v[0]
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * self.widen
        if not force and 2 * half > self.max_width * scale:
            return None
        if period is not None:
            if 2 * half >= period:
                return None
            cut = IntervalCut(
                [mid - half, mid + half], period=period, dim=self.dim
            )
            core = int(cut.classify(np.array([mid]))[0])
        else:
            # a tail with too few values joins the core
            left = int((c < mid - half).sum())
            right = int((c > mid + half).sum())
            keep = [
                (side, p)
                for side, p, count in (
                    ("left", mid - half, left),
                    ("right", mid + half, right),
                )
                if count >= self.min_points
            ]
            if max_labels is not None and len(keep) > max_labels - 1:
                keep = [
                    max(keep, key=lambda k: left if k[0] == "left" else right)
                ]
            if not keep:
                return None
            cut = IntervalCut([p for _, p in keep], dim=self.dim)
            core = 1 if keep[0][0] == "left" else 0
        labels = cut.classify(c)
        outside = labels != core
        n_out = int(outside.sum())
        if n_out < self.min_points:
            return None
        if not force and n_out > self.max_share * n:
            return None
        counts = np.bincount(labels, minlength=cut.n_labels)
        cut.info = dict(
            shares=(counts / n).tolist(),
            core_width=float(2 * half),
            core_label=int(core),
        )
        return cut


class FirstOf:
    """The first cut any of ``splitters`` proposes."""

    def __init__(self, splitters):
        import sys

        module = sys.modules[__name__]
        self.splitters = [build(s, None, module) for s in splitters]

    def propose(
        self,
        coords,
        periods,
        points=None,
        proposal=None,
        max_labels=None,
        force=False,
    ):
        for splitter in self.splitters:
            cut = splitter.propose(
                coords,
                periods,
                points=points,
                proposal=proposal,
                max_labels=max_labels,
            )
            if cut is not None:
                return cut
        if force:
            for splitter in self.splitters:
                cut = splitter.propose(
                    coords,
                    periods,
                    points=points,
                    proposal=proposal,
                    max_labels=max_labels,
                    force=True,
                )
                if cut is not None:
                    return cut
        return None


# -- regions ------------------------------------------------------------------
class Region:
    """One region of a partitioned nested sampling run: an independent run
    over its share of the prior, with its own live points, proposal and
    retrain decision, killed in lockstep with the others.

    ``iterations`` counts the region's own dead points (its retrain
    decision and checks run on that clock). The timing accumulators collect
    the time spent drawing and training for this region only, for its
    retrain decision's unit costs.
    """

    def __init__(
        self,
        label,
        parent,
        membership,
        nlive,
        live_points,
        proposal=None,
        retrain_decision=None,
        acceptance_length=None,
    ):
        from collections import deque

        self.label = label
        self.parent = parent
        self.membership = membership
        self.nlive = int(nlive)
        self.live_points = live_points
        self.proposal = proposal
        self.retrain_decision = retrain_decision
        self.active = True
        self.iterations = 0
        self.last_check = 0
        self.training_iterations = []
        self.insertion_indices = []
        self.block_acceptance = 1.0
        self.block_iteration = 0
        self.mean_block_acceptance = 1.0
        self.acceptance_history = deque(
            maxlen=acceptance_length or max(1, self.nlive // 10)
        )
        self.completed_training = True
        self.time = 0.0
        self.likelihood_time = 0.0
        self.likelihood_evaluations = 0
        self._retrain_snapshot = None
        self.final_p_value = None

    @property
    def mean_acceptance(self):
        if self.acceptance_history:
            return float(np.mean(self.acceptance_history))
        return np.nan

    @property
    def last_trained(self):
        return self.training_iterations[-1] if self.training_iterations else 0


class RegionMembership:
    """The region a set of (frozen coordinates, cut, label) constraints
    defines: their conjunction, one per split on the way from the whole
    prior to the region."""

    def __init__(self, constraints=()):
        self.constraints = list(constraints)

    def child(self, coordinates, cut, label):
        return RegionMembership(self.constraints + [(coordinates, cut, label)])

    def contains(self, points):
        keep = np.ones(len(points), dtype=bool)
        for coordinates, cut, label in self.constraints:
            if not keep.any():
                break
            idx = np.flatnonzero(keep)
            inside = cut.classify(coordinates(points[idx])) == label
            keep[idx[~inside]] = False
        return keep


class RegionTruncation(BaseTruncationRule):
    """Keep a proposal's draws inside its region of a partitioned run.

    Applied after the inverse pass, before the likelihood: a draw outside
    the region costs no likelihood evaluation.
    """

    name = "region"

    def __init__(self, membership, label=None):
        self.membership = membership
        self.label = label
        self.n_rejected = 0
        self.n_seen = 0
        super().__init__()

    def apply_after_backward(self, proposal, x, log_q, z):
        if not len(x):
            return x, log_q, z
        points = x
        if getattr(proposal, "map_to_unit_hypercube", False):
            points = proposal.model.from_unit_hypercube(x.copy())
        keep = self.membership.contains(points)
        self.n_seen += len(keep)
        self.n_rejected += int((~keep).sum())
        return get_subset_arrays(keep, x, log_q, z)


# -- diagnostics --------------------------------------------------------------
def plot_partition(
    coords,
    labels,
    names,
    periods=None,
    cut=None,
    legend=None,
    title=None,
    filename=None,
):
    """Histograms of the split coordinates, one colour per region.

    Parameters
    ----------
    coords : numpy.ndarray
        ``(n, d)`` coordinates of the live points.
    labels : array_like
        The region (or candidate part) of each point; ``None`` for one
        colour.
    names : list of str
        Coordinate names.
    periods : list, optional
        Periods, used for the histogram range.
    cut : :class:`Cut`, optional
        Drawn on the plot.
    legend : dict, optional
        Legend text for each label.
    title : str, optional
    filename : str, optional
        If given the figure is saved and closed, else returned.
    """
    import matplotlib.pyplot as plt

    coords = np.asarray(coords, dtype=float).reshape(len(coords), -1)
    d = coords.shape[1]
    labels = (
        np.zeros(len(coords), dtype=int)
        if labels is None
        else np.asarray(labels)
    )
    keys = list(dict.fromkeys(sorted(labels.tolist())))
    legend = legend or {}
    periods = periods or [None] * d
    ncol = d + (1 if d == 2 else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(4.5 * ncol, 3.6))
    axes = np.atleast_1d(axes)
    for i in range(d):
        ax = axes[i]
        rng = (0.0, periods[i]) if periods[i] else None
        data = [coords[labels == k, i] for k in keys]
        ax.hist(
            data,
            bins=60,
            range=rng,
            stacked=True,
            label=[legend.get(k, str(k)) for k in keys],
            color=[f"C{j % 10}" for j in range(len(keys))],
        )
        ax.set_xlabel(names[i])
        if cut is not None and d == 1:
            cut.draw(ax, 1)
        elif isinstance(cut, IntervalCut) and cut.dim == i:
            cut.draw(ax, 1)
    if d == 2:
        ax = axes[2]
        for j, k in enumerate(keys):
            m = labels == k
            ax.scatter(
                coords[m, 0],
                coords[m, 1],
                s=2,
                color=f"C{j % 10}",
                label=legend.get(k, str(k)),
            )
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        if cut is not None:
            cut.draw(ax, 2)
    axes[0].legend(frameon=False, fontsize="small")
    if title:
        fig.suptitle(title, fontsize="medium")
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
        plt.close(fig)
        return None
    return fig


# -- copies of a proposal -----------------------------------------------------
def respawn_generators(obj, rng, max_depth=4):
    """Replace every :class:`numpy.random.Generator` reachable from ``obj``
    (attributes, dictionaries, lists; not inside torch modules) with a new
    one seeded from ``rng``, the same object by the same new one: a copied
    proposal must not repeat the random stream of the original."""
    import torch

    new = {}

    def replace(value):
        key = id(value)
        if key not in new:
            new[key] = np.random.default_rng(int(rng.integers(2**63)))
        return new[key]

    seen = set()

    def visit(o, depth):
        if depth > max_depth or id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, (torch.nn.Module, np.ndarray, str, bytes)):
            return
        if isinstance(o, dict):
            items = list(o.items())
            for k, v in items:
                if isinstance(v, np.random.Generator):
                    o[k] = replace(v)
                else:
                    visit(v, depth + 1)
            return
        if isinstance(o, list):
            for i, v in enumerate(o):
                if isinstance(v, np.random.Generator):
                    o[i] = replace(v)
                else:
                    visit(v, depth + 1)
            return
        if isinstance(o, tuple):
            for v in o:
                visit(v, depth + 1)
            return
        d = getattr(o, "__dict__", None)
        if d is not None:
            visit(d, depth)

    visit(obj, 0)
    return len(new)


def check_clone(source, clone, points, n=200):
    """Warn if a copied proposal's density differs from the source's at a
    few of ``points``: a copy is resumed from the pickled state and the
    weights, and must start from the same flow."""
    if points is None or not len(points):
        return None
    sub = points[: min(n, len(points))].copy()
    try:
        _, log_q_source = source.forward_pass(sub.copy())
        _, log_q_clone = clone.forward_pass(sub.copy())
    except Exception as e:  # a diagnostic must not stop the run
        logger.debug(f"Clone check skipped: {type(e).__name__}: {e}")
        return None
    a, b = np.asarray(log_q_source), np.asarray(log_q_clone)
    ok = np.isfinite(a) & np.isfinite(b)
    gap = float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0
    if gap > 1e-4 or (np.isfinite(a) != np.isfinite(b)).any():
        logger.warning(
            f"Partition: the copied proposal's log q differs from the "
            f"original's by up to {gap:.3g} at the region's live points"
        )
    else:
        logger.debug(f"Clone check: max |delta log q| {gap:.2g}")
    return gap
