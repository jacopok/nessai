"""Tests for the clip-compression term and correction."""

import numpy as np
import pytest
from scipy.integrate import quad

from nessai.evidence import _NSIntegralState
from nessai.samplers.nestedsampler import (
    CLIP_COMPRESSION_MIN_STRETCH,
    clip_compression_stretches,
    clip_compression_term,
)


@pytest.mark.parametrize("log_excess", [-np.inf, -1.0, 0.0])
def test_term_zero_below_the_cap(log_excess):
    assert clip_compression_term(log_excess, 10, 100) == 0.0


def test_term_value():
    nlive, index = 100, 74
    u = 1 - (index + 0.5) / nlive
    expected = (np.e - 1) * (1 + np.log(u))
    assert clip_compression_term(1.0, index, nlive) == pytest.approx(expected)


def test_term_sign():
    """Over the cap at the low-likelihood edge (u near 1) the term is
    positive; near the peak (u small) negative."""
    assert clip_compression_term(0.5, 0, 1000) > 0
    assert clip_compression_term(0.5, 999, 1000) < 0


def test_stretches_spread_each_term_over_the_window():
    terms = np.zeros(20)
    terms[[3, 7]] = [0.4, -0.2]
    stretch = clip_compression_stretches(terms, 5)
    # each term is spread over the 5 iterations after its own
    assert np.allclose(stretch[:4], 1.0)
    assert np.allclose(stretch[4:8], 1.08)
    assert np.allclose(stretch[8:9], 1.08 - 0.04)
    assert np.allclose(stretch[9:13], 1.0 - 0.04)
    assert np.allclose(stretch[13:], 1.0)
    assert np.sum(stretch - 1) == pytest.approx(terms.sum())


def test_stretches_floor():
    stretch = clip_compression_stretches([-100.0, 0.0], 1)
    assert stretch[1] == CLIP_COMPRESSION_MIN_STRETCH


@pytest.mark.parametrize("expectation", ["logt", "t"])
def test_increment_stretch(expectation):
    state = _NSIntegralState(100, expectation=expectation)
    reference = _NSIntegralState(100 / 1.5, expectation=expectation)
    state.increment(-1.0, stretch=1.5)
    reference.increment(-1.0)
    assert state.nlive[-1] == pytest.approx(100 / 1.5)
    assert state.logw == pytest.approx(reference.logw)
    assert state.logZ == pytest.approx(reference.logZ)


@pytest.mark.parametrize("stretch", [0.0, -1.0])
def test_increment_stretch_positive(stretch):
    state = _NSIntegralState(100)
    with pytest.raises(ValueError, match="stretch must be positive"):
        state.increment(-1.0, stretch=stretch)


@pytest.mark.parametrize("edge", [True, False])
def test_mean_term_is_the_compression_bias(edge):
    """New points from a pool capped at c, density proportional to
    a(u) = min(1, c / w(u)) on the contour: the mean term over them is
    b = int (1 - a)(1 + ln u) du / int a du."""
    rng = np.random.default_rng(1234)
    nlive = 10_000

    def log_w(u):
        # the flow under-covers the low-likelihood edge (u near 1), or the
        # peak (u near 0)
        return 3 * u if edge else 3 * (1 - u)

    log_c = 2.5

    def a(u):
        return np.minimum(1.0, np.exp(log_c - log_w(u)))

    b = quad(lambda u: (1 - a(u)) * (1 + np.log(u)), 0, 1, points=[5 / 6, 1 / 6])[0]
    b /= quad(a, 0, 1, points=[5 / 6, 1 / 6])[0]

    u = rng.random(2_000_000)
    u = u[rng.random(u.size) < a(u)]
    # the insertion index of a point at u among nlive - 1 survivors
    index = np.clip(np.floor((1 - u) * nlive), 0, nlive - 1)
    log_excess = log_w(u) - log_c
    terms = np.array(
        [clip_compression_term(e, k, nlive) for e, k in zip(log_excess, index)]
    )
    err = terms.std() / np.sqrt(terms.size)
    assert (b > 0) == edge
    assert terms.mean() == pytest.approx(b, abs=4 * err + 1e-4)
