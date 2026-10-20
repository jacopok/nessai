# -*- coding: utf-8 -*-
"""
Functions related to computing the evidence.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from .plot import nessai_style

logger = logging.getLogger(__name__)


def logsubexp(x, y):
    """
    Helper function to compute the exponential
    of a difference between two numbers

    Computes: ``x + np.log1p(-np.exp(y-x))``

    Parameters
    ----------
    x, y : float or array_like
        Inputs
    """
    if np.any(x < y):
        raise RuntimeError(
            f"cannot take log of negative number {str(x)!s} - {str(y)!s}"
        )

    return x + np.log1p(-np.exp(y - x))


def log_integrate_log_trap(log_func, log_support):
    """
    Trapezoidal integration of given log(func). Returns log of the integral.

    Parameters
    ----------
    log_func : array_like
        Log values of the function to integrate over.
    log_support : array_like
        Log prior-volumes for each value.

    Returns
    -------
    float
        Log of the result of the integral.
    """
    log_func_sum = np.logaddexp(log_func[:-1], log_func[1:]) - np.log(2)
    log_dxs = logsubexp(log_support[:-1], log_support[1:])

    return logsumexp(log_func_sum + log_dxs)


class _BaseNSIntegralState(ABC):
    """Base class for the nested sampling integral."""

    @property
    @abstractmethod
    def log_evidence(self):
        """The current log-evidence."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def log_evidence_error(self):
        """The current error on the log-evidence."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def log_posterior_weights(self):
        """The log-posterior weights."""
        raise NotImplementedError()

    @property
    def effective_n_posterior_samples(self):
        """Kish's effective sample size for the posterior weights.

        Returns
        -------
        float
            The effective sample size. Returns zero if the posterior weights
            are empty.
        """
        log_p = self.log_posterior_weights
        if not len(log_p):
            return 0
        log_p -= logsumexp(log_p)
        n = np.exp(-logsumexp(2 * log_p))
        return float(n)

    ess = effective_n_posterior_samples
    """Alias for effective_n_posterior_samples"""


class _NSIntegralState(_BaseNSIntegralState):
    """
    Stores the state of the nested sampling integrator

    Parameters
    ----------
    nlive : int
        Number of live points
    track_gradients : bool, optional
        If true the gradient of the change in logL w.r.t logX is saved each
        time `increment` is called.
    expectation : str, {logt, t}
        Method used to compute the expectation value for the shrinkage t.
        Choose between log <t> or <log t>. Defaults to <log t>.
    """

    def __init__(self, nlive, track_gradients=True, expectation="logt"):
        self.base_nlive = nlive
        self.track_gradients = track_gradients

        if expectation.lower() not in ["t", "logt"]:
            raise ValueError(
                f"Expectation must be t or logt, got: {expectation}"
            )
        self.expectation = expectation.lower()

        # Initial state of the integral
        self.logZ = -np.inf
        self.oldZ = -np.inf
        self.logw = 0
        self.info = [0.0]
        # Initially contain all the prior volume
        self.logLs = [-np.inf]  # Likelihoods sampled
        self.log_vols = [0.0]  # Volumes enclosed by contours
        self.nlive = []
        self.gradients = [0]
        # prior volume of each dead point (log), in the order of ``logLs``
        self.log_dx = []
        # partitioned run (:meth:`split`): log-volume of each active region,
        # and the retired regions' live points, added as dead points
        self.region_logw = None
        self.retired = dict(logL=[], log_dx=[], region=[])
        self.threshold_logLs = None

    @property
    def partitioned(self):
        """True once the prior has been split into regions."""
        return getattr(self, "region_logw", None) is not None

    @property
    def log_evidence(self):
        """The current log-evidence."""
        return self.logZ

    @property
    def log_evidence_error(self):
        """The current error on the log-evidence."""
        return np.sqrt(self.info[-1] / self.base_nlive)

    def _logt(self, nlive):
        if self.expectation == "logt":
            # <logt> approx -1 / N
            return -1.0 / nlive
        # <t> = N / (N + 1)
        return -np.log1p(1 / nlive)

    def _add_weight(self, Wt, logL):
        """Add a dead point of log-weight ``Wt`` to the evidence and the
        information."""
        oldZ = self.logZ
        self.logZ = np.logaddexp(self.logZ, Wt)
        if np.isfinite(oldZ) and np.isfinite(self.logZ) and np.isfinite(logL):
            info = (
                np.exp(Wt - self.logZ) * logL
                + np.exp(oldZ - self.logZ) * (self.info[-1] + oldZ)
                - self.logZ
            )
            self.info.append(info)

    def _active_logw(self):
        values = list(self.region_logw.values())
        return float(logsumexp(values)) if values else -np.inf

    def increment(self, logL, nlive=None, region=None):
        """
        Increment the state of the evidence integrator
        Simply uses rectangle rule for initial estimate

        In a partitioned run (:meth:`split`) ``region`` is the region the
        dead point belongs to: only its volume shrinks, by ``1 / nlive``
        with ``nlive`` that region's number of live points.
        """
        if logL <= self.logLs[-1]:
            logger.warning(
                "NS integrator received non-monotonic logL."
                f"{self.logLs[-1]:.5f} -> {logL:.5f}"
            )
        if nlive is None:
            nlive = self.base_nlive
        partitioned = getattr(self, "region_logw", None) is not None
        if partitioned and region is None:
            raise RuntimeError("A partitioned run needs the region")

        self.nlive.append(nlive)
        if self.expectation == "logt":
            # <logt> approx -1 / N
            logt = -1.0 / nlive
        else:
            # <t> = N / (N + 1)
            logt = -np.log1p(1 / nlive)
        logw = self.region_logw[region] if partitioned else self.logw
        log_dx = logw + np.log1p(-np.exp(logt))
        Wt = log_dx + logL
        oldZ = self.logZ
        self.logZ = np.logaddexp(self.logZ, Wt)
        # Update information estimate
        if np.isfinite(oldZ) and np.isfinite(self.logZ) and np.isfinite(logL):
            info = (
                np.exp(Wt - self.logZ) * logL
                + np.exp(oldZ - self.logZ) * (self.info[-1] + oldZ)
                - self.logZ
            )
            self.info.append(info)
        if not hasattr(self, "log_dx"):
            self.log_dx = []
        self.log_dx.append(log_dx)
        if partitioned:
            self.log_dx_region.append(region)

        # Update history
        if partitioned:
            self.region_logw[region] += logt
            self.logw = self._active_logw()
        else:
            self.logw += logt
        self.logLs.append(logL)
        self.log_vols.append(self.logw)
        if self.track_gradients:
            self.gradients.append(
                (self.logLs[-1] - self.logLs[-2])
                / (self.log_vols[-1] - self.log_vols[-2])
            )

    def get_logx_live_points(self, nlive: int):
        """Get the log-prior volume of the current live points.

        Computes the values assuming the run were to terminate at the current
        iteration.

        Parameters
        ----------
        nlive : int
            The number of live points.

        Returns
        -------
        numpy.ndarray
            Array of log-prior volumes of length :code:`nlive`.
        """
        nlive_per_iteration = np.arange(nlive, 0, -1, dtype=float)
        if self.expectation.lower() == "logt":
            logt = -1.0 / nlive_per_iteration
        elif self.expectation.lower() == "t":
            logt = -np.log1p(1.0 / nlive_per_iteration)
        log_x = self.logw + np.cumsum(logt)
        return log_x

    # -- partitioned runs ---------------------------------------------------
    def split(self, parent, children, log_shares):
        """Split the volume of region ``parent`` into ``children``.

        Each child gets ``log_shares[i]`` of the parent's current volume
        (natural log, the shares summing to one). Before the first split the
        whole run is the region ``parent``.
        """
        if not self.partitioned:
            self.region_logw = {parent: float(self.logw)}
            self.retired = dict(logL=[], log_dx=[], region=[])
            self.log_dx_region = [None] * len(getattr(self, "log_dx", []))
        if parent not in self.region_logw:
            raise ValueError(f"Region {parent} is not active")
        log_shares = np.asarray(log_shares, dtype=float)
        if len(children) != len(log_shares):
            raise ValueError("One share per child")
        if not np.isclose(logsumexp(log_shares), 0.0, atol=1e-9):
            raise ValueError("The shares must sum to one")
        logw = self.region_logw.pop(parent)
        for child, log_share in zip(children, log_shares):
            if child in self.region_logw:
                raise ValueError(f"Region {child} already exists")
            self.region_logw[child] = logw + float(log_share)
        self.logw = self._active_logw()

    def retire(self, region, logLs):
        """Close ``region``: its live points ``logLs`` become dead points
        with the volumes of the final live points of a run (``nlive`` from
        ``len(logLs)`` down to 1), and the rest of its volume, below the last
        of them, is given to that last point.

        The entries are kept apart from :attr:`logLs`, which stays in the
        order of the iterations; :meth:`finalise` merges them.
        """
        logLs = np.sort(np.asarray(logLs, dtype=float))
        logw = self.region_logw.pop(region)
        n = len(logLs)
        for i, logL in enumerate(logLs):
            logt = self._logt(n - i)
            log_dx = logw + np.log1p(-np.exp(logt))
            logw += logt
            if i == n - 1:
                # the volume left inside the last contour
                log_dx = np.logaddexp(log_dx, logw)
            self._add_weight(log_dx + logL, logL)
            self.retired["logL"].append(float(logL))
            self.retired["log_dx"].append(float(log_dx))
            self.retired["region"].append(region)
        self.logw = self._active_logw()

    def region_log_evidence(self):
        """Evidence of each region's dead points (log), keyed by region;
        ``None`` holds the points from before the first split."""
        regions = list(self.log_dx_region) + list(self.retired["region"])
        log_w = np.concatenate(
            [
                np.asarray(self.log_dx, dtype=float)
                + np.asarray(self._kill_logLs()[1:], dtype=float),
                np.asarray(self.retired["log_dx"], dtype=float)
                + np.asarray(self.retired["logL"], dtype=float),
            ]
        )
        out = {}
        for r in dict.fromkeys(regions):
            mask = np.array([q == r for q in regions], dtype=bool)
            out[r] = float(logsumexp(log_w[mask]))
        return out

    def _kill_logLs(self):
        return (
            self.threshold_logLs
            if getattr(self, "threshold_logLs", None) is not None
            else self.logLs
        )

    def finalise_partitioned(self, live_logLs=None):
        """Merge the dead points of a partitioned run in likelihood order.

        ``live_logLs`` maps each still-active region to its live points'
        log-likelihoods, which are retired first. Rebuilds :attr:`logLs` and
        :attr:`log_vols` (the volume above each point, the sum over the
        regions) and the evidence by the rectangle rule (each point's own
        volume times its likelihood), and returns the permutation that sorts
        the dead points, in the order of the iterations followed by the
        retired points.
        """
        for region, logLs in (live_logLs or {}).items():
            self.retire(region, logLs)
        if self.region_logw:
            raise RuntimeError(
                f"Regions {list(self.region_logw)} are still active"
            )
        self.threshold_logLs = list(self.logLs)
        logL = np.concatenate(
            [np.asarray(self.logLs[1:], dtype=float), self.retired["logL"]]
        )
        log_dx = np.concatenate(
            [np.asarray(self.log_dx, dtype=float), self.retired["log_dx"]]
        )
        order = np.argsort(logL, kind="stable")
        logL, log_dx = logL[order], log_dx[order]
        # volume above point k: the sum of the volumes of the points after it
        above = np.logaddexp.accumulate(log_dx[::-1])[::-1]
        log_vols = np.concatenate([above[1:], [-np.inf]])
        self.logLs = [-np.inf] + logL.tolist()
        self.log_vols = [float(above[0])] + log_vols.tolist()
        self._sorted_log_dx = log_dx
        if self.track_gradients:
            with np.errstate(divide="ignore", invalid="ignore"):
                grad = np.diff(self.logLs) / np.diff(self.log_vols)
            grad = np.where(np.isfinite(grad), grad, 0.0)
            self.gradients = [0] + grad.tolist()
        log_w = logL + log_dx
        self.logZ = float(logsumexp(log_w))
        p = np.exp(log_w - self.logZ)
        self.info.append(float(np.sum(p * logL) - self.logZ))
        self._finalised_partitioned = True
        return order

    def finalise(self):
        """
        Compute the final evidence with more accurate integrator
        Call at end of sampling run to refine estimate
        """
        if getattr(self, "_finalised_partitioned", False):
            return self.logZ
        # Trapezoidal rule
        # Extra point represents X=0 and assume max(L) = L[-1]
        self.logZ = log_integrate_log_trap(
            np.array(self.logLs + [self.logLs[-1]]),
            np.array(self.log_vols + [-np.inf]),
        )
        return self.logZ

    @nessai_style()
    def plot(self, filename=None):
        """
        Plot the logX vs logL

        Parameters
        ----------
        filename : str, optional
            Filename name for saving the figure. If not specified the figure
            is returned.
        """
        fig = plt.figure()
        plt.plot(self.log_vols, self.logLs)
        plt.title(
            f"log Z={self.logZ:.2f} H={self.info[-1] * np.log2(np.e):.2f} bits"
        )
        plt.grid(which="both")
        plt.xlabel("log prior-volume")
        plt.ylabel("log-likelihood")
        plt.xlim([self.log_vols[-1], self.log_vols[0]])

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")
            plt.close()
            logger.debug(f"Saved nested sampling plot as {filename}")
        else:
            return fig

    @property
    def log_posterior_weights(self):
        """Compute the log-posterior weights."""
        if getattr(self, "_finalised_partitioned", False):
            return np.asarray(self.logLs[1:]) + self._sorted_log_dx - self.logZ
        log_L = np.array(self.logLs + [self.logLs[-1]])
        log_vols = np.array(self.log_vols + [-np.inf])
        log_Z = log_integrate_log_trap(log_L, log_vols)
        log_w = logsubexp(log_vols[:-1], log_vols[1:])
        log_post_w = log_L[1:-1] + log_w[:-1] - log_Z
        return log_post_w


class _INSIntegralState(_BaseNSIntegralState):
    """
    Object to handle computing the evidence for importance nested sampling.
    """

    def __init__(self) -> None:
        self._n_ns = 0
        self._n_lp = 0
        self._n = 0
        self._previous_logZ = -np.inf
        self._logZ = -np.inf
        self._weights_ns = None
        self._weights_lp = None
        self._weights = None

    @property
    def logZ(self) -> float:
        """The current log-evidence."""
        return self._logZ - np.log(self._n)

    log_evidence = logZ
    """Alias for logZ"""

    @property
    def log_evidence_error(self) -> float:
        """Alias for compute_uncertainty"""
        return self.compute_uncertainty(log_evidence=True)

    @property
    def evidence(self) -> float:
        """The current evidence"""
        return np.exp(self.log_evidence)

    @property
    def evidence_error(self) -> float:
        """The evidence error"""
        return self.compute_uncertainty(log_evidence=False)

    @property
    def log_evidence_live_points(self) -> float:
        """Log-evidence in the live points."""
        if self._weights_lp is None:
            raise RuntimeError("Live points are not set")
        return logsumexp(self._weights_lp) - np.log(self._weights_lp.size)

    @property
    def log_evidence_nested_samples(self) -> float:
        """Log-evidence in the nested samples"""
        return logsumexp(self._weights_ns) - np.log(self._weights_ns.size)

    @property
    def fractional_error(self) -> float:
        """Fractional error on the evidence."""
        return self.evidence_error / self.evidence

    @property
    def difference_log_evidence(self) -> float:
        """(Absolute) difference in log evidence"""
        return np.abs(self.logZ - self._previous_logZ)

    @property
    def log_posterior_weights(self) -> np.ndarray:
        """Compute the log posterior weights.

        If the live points have been specified, then weights will be computed
        for these as well.
        """
        return self._weights - self.logZ

    @property
    def log_evidence_ratio(self) -> float:
        """Log evidence ratio of the live points to all the samples."""
        return float(self.compute_log_evidence_ratio(ns_only=False))

    @property
    def log_evidence_ratio_nested_samples(self) -> float:
        """Log evidence ratio of the live points to the nested samples."""
        return float(self.compute_log_evidence_ratio(ns_only=True))

    def update_evidence(
        self,
        nested_samples: np.ndarray,
        live_points: Optional[np.ndarray] = None,
    ) -> None:
        """Update the evidence.

        Parameters
        ----------
        nested_samples
            Array of nested samples.
        live_points
            Optional array of live points, if included the evidence will
            include both live points and nested samples. If not, the evidence
            will only include the nested samples.
        """
        self._previous_logZ = self.log_evidence
        self._weights_ns = nested_samples["logL"] + nested_samples["logW"]
        if live_points is not None:
            self._weights_lp = live_points["logL"] + live_points["logW"]
            self._weights = np.concatenate(
                [
                    self._weights_ns,
                    self._weights_lp,
                ]
            )
        else:
            self._weights = self._weights_ns
            self._weights_lp = None
        self._logZ = logsumexp(self._weights)
        self._n = self._weights.size

    def compute_log_evidence_ratio(self, ns_only: bool = False) -> float:
        """
        Compute the log ratio of the evidence in the live points to total
        (or nested) samples.

        .. versionadded:: 0.15.0

        Parameters
        ----------
        ns_only
            If True only the evidence from the nested samples is used in the
            denominator of the ratio.

        Returns
        -------
        float
            Log ratio of the evidence
        """
        if ns_only:
            return (
                self.log_evidence_live_points
                - self.log_evidence_nested_samples
            )
        else:
            return self.log_evidence_live_points - self.logZ

    def compute_evidence_ratio(self, ns_only: bool = False) -> float:
        """Compute the evidence ratio of the live points to the nested samples.

        .. deprecated:: 0.15.0
            Use :meth:`compute_log_evidence_ratio` instead.
            This will be removed in a future version.
        """

        warn(
            (
                "compute_evidence_ratio is deprecated, "
                "use compute_log_evidence_ratio instead"
            ),
            DeprecationWarning,
        )
        return self.compute_log_evidence_ratio(ns_only=ns_only)

    def compute_uncertainty(self, log_evidence: bool = True) -> float:
        """Compute the uncertainty on the current estimate of the evidence.

        Parameters
        ----------
        log_evidence :
            If true, return the uncertainty for the log-evidence rather than
            the evidence.
        """
        n = self._n
        Z_hat = np.exp(self.logZ, dtype=np.longdouble)
        Z = np.exp(self._weights, dtype=np.longdouble)
        # Standard error sqrt(Var[Z] / n)
        u = np.sqrt(np.sum((Z - Z_hat) ** 2) / (n * (n - 1)))
        if log_evidence:
            # sigma[ln Z] = |sigma[Z] / Z|
            return float(np.abs(u / Z_hat))
        else:
            return u


def log_evidence_from_ins_samples(samples: np.ndarray) -> float:
    """Compute the log-evidence for a given set of samples from the importance
    nested sampler.

    Parameters
    ----------
    samples : numpy.ndarray
        Array of samples from the importance nested samples. Must have the logW
        field.

    Returns
    -------
    float
        The log-evidence
    """
    return logsumexp(samples["logL"] + samples["logW"]) - np.log(len(samples))
