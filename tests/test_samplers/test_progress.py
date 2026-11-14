"""Tests for the progress estimate (nessai.samplers.progress)."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from nessai.samplers import progress
from nessai.samplers.nestedsampler import NestedSampler
from nessai.stopping_criteria import ESS, DifferenceLogEvidence


def gaussian_run(dims=10, log_l_max=50.0, k=1e3, x_end=80.0, dx=1e-3):
    """-logX, ln Z and ln Lmax along a run on a Gaussian peak in ``dims``
    dimensions: ln L = ln Lmax - k X^(2/d)."""
    x = np.arange(0.0, x_end, dx)
    log_l = log_l_max - k * np.exp(-2 * x / dims)
    # Z(x) = int_x^0 L dX, X = e^-x
    log_dz = log_l - x + np.log(-np.expm1(-dx))
    log_z = np.logaddexp.accumulate(log_dz)
    return x, log_z, np.full_like(x, log_l_max)


def true_remaining(x, log_z, log_l_max, x0, tolerance):
    """-logX from ``x0`` to where dlogZ = ln(1 + Lmax X / Z) < tolerance."""
    dlogz = np.logaddexp(0, log_l_max - x - log_z)
    stop = x[np.argmax((dlogz < tolerance) & (x > x0))]
    return stop - x0


@pytest.mark.parametrize("x0", [10.0, 20.0, 30.0])
def test_remaining_log_volume_gaussian(x0):
    """On a Gaussian peak the extrapolation, before ``SCALE``, finds the stop
    to within a grid step."""
    x, log_z, log_l_max = gaussian_run()
    keep = x <= x0
    got = progress.remaining_log_volume(
        x[keep][::100], log_z[keep][::100], log_l_max[keep][::100], 0.1
    )
    want = true_remaining(x, log_z, log_l_max[0], x0, 0.1)
    assert got / progress.SCALE == pytest.approx(want, abs=2 * progress.STEP)


def test_remaining_log_volume_too_early():
    x, log_z, log_l_max = gaussian_run()
    keep = x < progress.FIT_SPAN - 0.5
    assert np.isnan(
        progress.remaining_log_volume(
            x[keep], log_z[keep], log_l_max[keep], 0.1
        )
    )


def make_sampler(x0, criterion=None, rate=(0.01, 0.005)):
    """A stand-in sampler at -logX = ``x0`` on a Gaussian run with nlive
    1000, acceptance ``rate[0]`` and ``rate[1]`` s per evaluation."""
    nlive = 1000
    x, log_z, log_l_max = gaussian_run()
    keep = (x <= x0) & (np.arange(len(x)) % 100 == 0)
    its = np.round(x[keep] * nlive).astype(int)
    evals = its / rate[0]
    log_vols = -np.arange(its[-1] + 1) / nlive
    return SimpleNamespace(
        state=SimpleNamespace(logw=log_vols[-1], log_vols=list(log_vols)),
        history={
            "iterations": list(its),
            "logZ": list(log_z[keep]),
            "max_log_likelihood": list(log_l_max[keep]),
            "sampling_time": list(evals * rate[1]),
            "likelihood_evaluations": list(evals),
        },
        stopping_criterion=criterion or DifferenceLogEvidence(0.1),
        tolerance=0.1,
    )


def test_estimate_progress():
    p = progress.estimate_progress(make_sampler(20.0))
    assert p["log_volume"] == pytest.approx(20.0, abs=1e-2)
    assert p["evaluations_per_iteration"] == pytest.approx(100)
    assert p["seconds_per_evaluation"] == pytest.approx(0.005)
    assert p["iterations"] == pytest.approx(1000 * p["remaining"], rel=1e-2)
    assert p["seconds"] == pytest.approx(
        p["iterations"] / 0.01 * 0.005, rel=1e-6
    )


def test_estimate_progress_other_criterion():
    p = progress.estimate_progress(make_sampler(20.0, criterion=ESS()))
    assert not p["dlogz_stop"]
    assert np.isnan(p["remaining"])
    assert np.isnan(p["seconds"])
    assert p["evaluations_per_iteration"] == pytest.approx(100)


def test_estimate_progress_no_history():
    s = make_sampler(20.0)
    s.history = None
    p = progress.estimate_progress(s)
    assert np.isnan(p["remaining"]) and np.isnan(
        p["evaluations_per_iteration"]
    )


def test_format_progress_with_estimate():
    p = progress.estimate_progress(make_sampler(20.0))
    line = progress.format_progress(p)
    assert line.startswith("Progress: -logX 20.0/~")
    assert f"ETA {progress.format_duration(p['seconds'])} = " in line
    assert "= 5 ms/eval x 100 eval/it x " in line
    assert line.endswith(" it")


@pytest.mark.parametrize(
    "x0, criterion, expected",
    [
        (3.0, None, "no ETA (before -logX 4.0)"),
        (20.0, ESS(), "no ETA (needs the dlogZ stop)"),
    ],
)
def test_format_progress_without_estimate(x0, criterion, expected):
    line = progress.format_progress(
        progress.estimate_progress(make_sampler(x0, criterion=criterion))
    )
    assert (
        line
        == f"Progress: -logX {x0:.1f}, 5 ms/eval x 100 eval/it; {expected}"
    )


@pytest.mark.parametrize(
    "n, expected",
    [(9812, "9800"), (101234, "100k"), (1.234e6, "1.2M"), (np.nan, "0")],
)
def test_count(n, expected):
    assert progress._count(n) == expected


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0.23, "230 ms"),
        (5.0, "5 s"),
        (125.0, "3 min"),
        (5.2 * 3600, "5.2 h"),
        (2 * 86400 + 3 * 3600, "2 d 3 h"),
        (np.nan, "unknown"),
    ],
)
def test_format_duration(seconds, expected):
    assert progress.format_duration(seconds) == expected


def test_log_progress_logs_one_line(caplog):
    sampler = make_sampler(20.0)
    with caplog.at_level(logging.INFO, logger="nessai.samplers.nestedsampler"):
        NestedSampler.log_progress(sampler)
    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) == 1
    assert lines[0].startswith("Progress: -logX 20.0/~")


def test_estimate_progress_uses_the_retrain_model():
    """With a retrain model the rate is its long-run cycle, not the
    measured window."""
    sampler = make_sampler(20.0)
    sampler.retrain_decision = MagicMock()
    sampler.retrain_decision.long_run_breakdown.return_value = {
        "evaluations_per_iteration": 40.0,
        "seconds_per_evaluation": 0.002,
        "cycle": 3000.0,
    }
    p = progress.estimate_progress(sampler)
    assert p["from_model"]
    assert p["evaluations_per_iteration"] == 40.0
    assert p["seconds_per_evaluation"] == 0.002
    assert p["seconds"] == pytest.approx(p["iterations"] * 40.0 * 0.002)


def test_estimate_progress_falls_back_without_a_model():
    sampler = make_sampler(20.0)
    sampler.retrain_decision = MagicMock()
    sampler.retrain_decision.long_run_breakdown.return_value = None
    p = progress.estimate_progress(sampler)
    assert not p["from_model"]
    assert p["evaluations_per_iteration"] == pytest.approx(100)


def test_remaining_log_volume_large_evidence_steps():
    """Early on ln Z grows by thousands per step: no overflow."""
    x, log_z, log_l_max = gaussian_run(log_l_max=1e5, k=1e5)
    keep = x <= 10.0
    r = progress.remaining_log_volume(
        x[keep][::100], log_z[keep][::100], log_l_max[keep][::100], 0.1
    )
    assert np.isfinite(r)


def test_remaining_log_volume_capped_dimension():
    """A local dimension above ``max_dimension`` is capped, which brings
    the end closer; a cap above it changes nothing."""
    x, log_z, log_l_max = gaussian_run(dims=30)
    keep = x <= 20.0
    args = (x[keep][::100], log_z[keep][::100], log_l_max[keep][::100], 0.1)
    free = progress.remaining_log_volume(*args)
    assert progress.remaining_log_volume(*args, max_dimension=10) < free
    assert progress.remaining_log_volume(*args, max_dimension=60) == free


def test_estimate_progress_caps_at_the_sampled_parameters():
    sampler = make_sampler(20.0)
    sampler.model = SimpleNamespace(names=["a", "b"])
    capped = progress.estimate_progress(sampler)["remaining"]
    sampler.model = SimpleNamespace(names=None)
    assert capped < progress.estimate_progress(sampler)["remaining"]
