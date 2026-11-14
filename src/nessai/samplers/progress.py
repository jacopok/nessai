"""How far a nested sampling run has to go.

:class:`~nessai.samplers.nestedsampler.NestedSampler` logs a progress line
after each training of the flow: -logX (X the prior volume inside the live
points), the -logX at which the run is expected to stop and the time to get
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
    are the long run of the retrain cycle
    (:meth:`~nessai.samplers.retrain.RetrainDecision.long_run_breakdown`):
    eval/it is 1 / acceptance averaged over a cycle of the retrain model
    (the fresh acceptance it predicts now, decaying at its fitted rate
    until the next training), time/eval the measured wall-clock cost of an
    evaluation (all workers in parallel), pool drawing and the training
    spread over the cycle included.  Both are assumed to stay: a later drop
    of the acceptance is not foreseen.  Without the model (before the flow
    proposal, in a partitioned run, or before it has fitted a decay) both
    are measured over the last ``RATE_SPAN`` of -logX instead.  The
    iterations left are the -logX left times the iterations per unit of
    -logX over that span.

    A measurement over a short span is no substitute for the model: the
    likelihood is evaluated a pool at a time, so a span shorter than a few
    cycles holds a whole number of pools and of trainings, and its
    eval/it and time/eval jump between reports.

A logged line reads e.g.::

    Progress: -logX 37.2/~49.3, ETA 1 d 14 h = 6.9 ms/eval x 122 eval/it x 60k it
"""

import math

import numpy as np

from ..stopping_criteria import DifferenceLogEvidence

#: -logX span over which the rise of the likelihood threshold is fitted.
FIT_SPAN = 4.0
#: -logX span over which the iterations per unit of -logX are measured, and
#: the acceptance and the time per evaluation when the retrain model does not
#: provide them.
RATE_SPAN = 4.0
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
    # L * X * (1 - e^-STEP), so ln L = ln(Z_next - Z) - ln(X) - log_dx, with
    # ln(Z_next - Z) = z + ln(expm1(dz)) = z + dz + ln(1 - e^-dz) (dz is in
    # the thousands early on, where expm1 overflows)
    dz_pos = np.maximum(dz, 1e-300)
    log_l = (
        z_grid[1:] + dz_pos + np.log(-np.expm1(-dz_pos)) - log_dx + grid[:-1]
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
    ``dlogz_stop`` (whether the run stops on dlogZ), ``remaining`` (the
    -logX still to go, see :func:`remaining_log_volume`; NaN if it cannot
    be estimated, and also if the stopping criterion is not dlogZ),
    ``iterations`` (left), ``evaluations_per_iteration``,
    ``seconds_per_evaluation``, ``from_model`` (whether those two come from
    the retrain model, see the module's docstring) and ``seconds`` (the
    sampling time left).  Quantities that cannot be measured yet are NaN.
    """
    dlogz_stop = isinstance(sampler.stopping_criterion, DifferenceLogEvidence)
    out = {
        "log_volume": float(-sampler.state.logw),
        "dlogz_stop": dlogz_stop,
        "remaining": np.nan,
        "iterations": np.nan,
        "evaluations_per_iteration": np.nan,
        "seconds_per_evaluation": np.nan,
        "from_model": False,
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
    model = _model_rate(sampler)
    if model is not None:
        out["evaluations_per_iteration"] = model["evaluations_per_iteration"]
        out["seconds_per_evaluation"] = model["seconds_per_evaluation"]
        out["from_model"] = True
    elif d_evals > 0 and d_it > 0:
        out["evaluations_per_iteration"] = d_evals / d_it
        out["seconds_per_evaluation"] = d_t / d_evals
    if d_x > 0 and np.isfinite(out["remaining"]):
        out["iterations"] = out["remaining"] * d_it / d_x
        out["seconds"] = (
            out["iterations"]
            * out["evaluations_per_iteration"]
            * out["seconds_per_evaluation"]
        )
    return out


def _model_rate(sampler):
    """The retrain model's long-run breakdown, or None without one (or in a
    partitioned run, whose regions have their own)."""
    if getattr(sampler, "regions", None) is not None:
        return None
    decision = getattr(sampler, "retrain_decision", None)
    if decision is None:
        return None
    return decision.long_run_breakdown()


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
    epi = p["evaluations_per_iteration"]
    if np.isfinite(epi):
        rate = (
            f"{format_duration(p['seconds_per_evaluation'])}/eval x "
            f"{epi:.3g} eval/it"
        )
    else:
        rate = "rate not measured yet"
    if np.isfinite(p["remaining"]):
        return (
            f"Progress: -logX {x:.1f}/~{x + p['remaining']:.1f}, ETA "
            f"{format_duration(p['seconds'])} = {rate} x "
            f"{_count(p['iterations'])} it"
        )
    if not p["dlogz_stop"]:
        why = "needs the dlogZ stop"
    elif x < FIT_SPAN:
        why = f"before -logX {FIT_SPAN:.1f}"
    else:
        why = "irregular logL rise"
    return f"Progress: -logX {x:.1f}, {rate}; no ETA ({why})"
