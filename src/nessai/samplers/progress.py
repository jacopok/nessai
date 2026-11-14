"""How far a nested sampling run has to go.

:class:`~nessai.samplers.nestedsampler.NestedSampler` logs a progress line
each time -logX (X the prior volume inside the live points) passes a whole
number: the -logX at which the run is expected to stop and the time to get
there, as the product of its factors.  The assumptions are documented here,
not in the log.

Where the run stops
    The stop is when dlogZ = ln(1 + Lmax X / Z) falls below the tolerance:
    the live points could at most add that much to the evidence.  How fast
    that happens depends on how the likelihood threshold L(X) keeps rising.
    The rise so far is read off the evidence (each step adds L dX to Z) over
    the last ``FIT_SPAN`` of -logX; ln Lmax - ln L is fitted there as an
    exponential in -logX, which is exact for a Gaussian peak (in d
    dimensions ln Lmax - ln L grows as X^(2/d)), and extrapolated until dlogZ
    is below the tolerance.  Real peaks are wider than Gaussian in the tails
    the extrapolation covers, so it is lengthened by ``SCALE``.

How long it takes
    ETA = time/eval x eval/it x iterations left, as logged.  The first two
    are measured over the last ``RATE_SPAN`` of -logX and assumed to stay:
    time/eval is the elapsed sampling time (all workers in parallel,
    training and pool drawing included) per likelihood evaluation, eval/it
    is 1 / acceptance.  The iterations left are the -logX left times the
    iterations per unit of -logX.

A logged line reads e.g.::

    Progress: -logX 37/~49, ETA 1 d 14 h = 6.9 ms/eval x 122 eval/it x 60k it
"""

import math

import numpy as np

from ..stopping_criteria import DifferenceLogEvidence

#: -logX span over which the rise of the likelihood threshold is fitted.
FIT_SPAN = 4.0
#: -logX span over which the acceptance and the time per evaluation are
#: measured.
RATE_SPAN = 1.0
#: -logX grid of the fit and of the extrapolation.
STEP = 0.25
#: Factor on the extrapolated -logX still to go.  On 19 finished runs (BNS,
#: ~20 dimensions), stopped 6-30 short of their end, the extrapolation gave
#: 0.8-0.9 of what actually remained.
SCALE = 1.2
#: Furthest the extrapolation looks ahead, in -logX.
HORIZON = 200.0


def remaining_log_volume(log_volume, log_z, log_l_max, tolerance):
    """The -logX still to go before dlogZ is below ``tolerance``.

    Parameters
    ----------
    log_volume : array_like
        -logX at each recorded iteration, increasing.
    log_z : array_like
        ln Z at those iterations.
    log_l_max : array_like
        The largest ln L found by those iterations.
    tolerance : float
        The dlogZ at which the run stops.

    Returns
    -------
    float
        The -logX still to go (scaled by ``SCALE``); NaN if -logX has not
        advanced by ``FIT_SPAN`` yet, if the evidence does not grow enough
        to fit, or if the extrapolation does not stop within ``HORIZON``.
    """
    x, log_z, log_l_max = (
        np.asarray(a, dtype=float) for a in (log_volume, log_z, log_l_max)
    )
    if len(x) < 2 or x[-1] - x[0] < FIT_SPAN:
        return np.nan
    grid = np.arange(x[-1] - FIT_SPAN, x[-1] + 1e-9, STEP)
    z_grid = np.interp(grid, x, log_z)
    dz = np.diff(z_grid)
    # ln of the share of the prior volume one grid step removes
    log_dx = np.log(-np.expm1(-STEP))
    # the threshold over each step from the evidence it added: Z grows by
    # L * X * (1 - e^-STEP), so ln L = ln(Z_next - Z) - ln(X) - log_dx
    log_l = (
        z_grid[1:]
        + np.log(np.expm1(np.maximum(dz, 1e-300)))
        - log_dx
        + grid[:-1]
    )
    below_peak = log_l_max[-1] - log_l
    ok = (dz > 1e-4) & (below_peak > 0)
    if ok.sum() < 3:
        return np.nan
    slope, intercept = np.polyfit(grid[1:][ok], np.log(below_peak[ok]), 1)
    z, xx = log_z[-1], x[-1]
    while xx < x[-1] + HORIZON:
        log_l_step = log_l_max[-1] - np.exp(
            intercept + slope * (xx + STEP / 2)
        )
        z = np.logaddexp(z, log_l_step - xx + log_dx)
        xx += STEP
        if np.logaddexp(0, log_l_max[-1] - xx - z) < tolerance:
            return SCALE * (xx - x[-1])
    return np.nan


def estimate_progress(sampler):
    """The current -logX and what is left of the run, from the sampler's
    history.

    Returns a dictionary with ``log_volume`` (the current -logX),
    ``dlogz_stop`` (whether the run stops on dlogZ), ``remaining`` (the -logX still to go, see :func:`remaining_log_volume`;
    NaN if it cannot be estimated, and also if the stopping criterion is not
    dlogZ), ``iterations``, ``acceptance``, ``seconds_per_evaluation`` (both
    over the last ``RATE_SPAN`` of -logX) and ``seconds`` (the sampling time
    left).  Quantities that cannot be measured yet are NaN.
    """
    dlogz_stop = isinstance(sampler.stopping_criterion, DifferenceLogEvidence)
    out = {
        "log_volume": float(-sampler.state.logw),
        "dlogz_stop": dlogz_stop,
        "remaining": np.nan,
        "iterations": np.nan,
        "acceptance": np.nan,
        "seconds_per_evaluation": np.nan,
        "seconds": np.nan,
    }
    h = sampler.history or {}
    keys = (
        "iterations",
        "logZ",
        "max_log_likelihood",
        "sampling_time",
        "likelihood_evaluations",
    )
    if any(not h.get(k) for k in keys):
        return out
    # keys added to the history in a later version are shorter after a
    # resume: align the rows at the end
    n = min(len(h[k]) for k in keys)
    its, log_z, log_l_max, t, evals = (
        np.asarray(h[k][-n:], dtype=float) for k in keys
    )
    log_vols = np.asarray(sampler.state.log_vols, dtype=float)
    x = -log_vols[np.minimum(its.astype(int), len(log_vols) - 1)]
    if n < 2:
        return out

    if dlogz_stop:
        out["remaining"] = remaining_log_volume(
            x, log_z, log_l_max, sampler.tolerance
        )

    j = min(int(np.searchsorted(x, x[-1] - RATE_SPAN)), n - 2)
    d_it, d_x = its[-1] - its[j], x[-1] - x[j]
    d_evals, d_t = evals[-1] - evals[j], t[-1] - t[j]
    if d_evals > 0 and d_it > 0:
        out["acceptance"] = d_it / d_evals
        out["seconds_per_evaluation"] = d_t / d_evals
    if d_x > 0 and np.isfinite(out["remaining"]):
        out["iterations"] = out["remaining"] * d_it / d_x
        out["seconds"] = (
            out["iterations"]
            / out["acceptance"]
            * out["seconds_per_evaluation"]
        )
    return out


def format_duration(seconds):
    """``seconds`` as e.g. ``"2 d 3 h"``, ``"5.2 h"``, ``"14 min"``,
    ``"230 ms"``."""
    if not np.isfinite(seconds):
        return "unknown"
    if seconds >= 2 * 86400:
        days, hours = divmod(round(seconds / 3600), 24)
        return f"{days} d {hours} h"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{math.ceil(seconds / 60)} min"
    if seconds >= 1:
        return f"{seconds:.3g} s"
    return f"{1e3 * seconds:.3g} ms"


def _count(n):
    """``n`` to two significant figures, as e.g. ``"9800"``, ``"100k"``,
    ``"1.2M"``."""
    if not np.isfinite(n) or n <= 0:
        return "0"
    n = round(n, 1 - math.floor(math.log10(n)))
    if n >= 1e6:
        return f"{n / 1e6:.2g}M"
    if n >= 1e4:
        return f"{n / 1e3:.0f}k"
    return f"{n:.0f}"


def format_progress(p):
    """The progress line :meth:`NestedSampler.log_progress` logs, for the
    output of :func:`estimate_progress`: the ETA as the product of the time
    per likelihood evaluation, the evaluations per iteration (1 /
    acceptance) and the iterations left."""
    x = p["log_volume"]
    acc = p["acceptance"]
    if np.isfinite(acc):
        rate = (
            f"{format_duration(p['seconds_per_evaluation'])}/eval x "
            f"{1 / acc:.3g} eval/it"
        )
    else:
        rate = "rate not measured yet"
    if np.isfinite(p["remaining"]):
        return (
            f"Progress: -logX {x:.0f}/~{x + p['remaining']:.0f}, ETA "
            f"{format_duration(p['seconds'])} = {rate} x "
            f"{_count(p['iterations'])} it"
        )
    if not p["dlogz_stop"]:
        why = "needs the dlogZ stop"
    elif x < FIT_SPAN:
        why = f"before -logX {FIT_SPAN:g}"
    else:
        why = "irregular logL rise"
    return f"Progress: -logX {x:.0f}, {rate}; no ETA ({why})"
