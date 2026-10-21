# -*- coding: utf-8 -*-
"""Unit tests for the geometry of partitioned nested sampling."""

import pickle

import numpy as np
import pytest
from scipy.special import logsumexp

from nessai.evidence import _NSIntegralState
from nessai.livepoint import empty_structured_array
from nessai.posterior import compute_weights
from nessai.partition import (
    CoreSplitter,
    FirstOf,
    GapSplitter,
    IntervalCut,
    LinearCut,
    ParameterCoordinates,
    RegionMembership,
    build,
    hold_out_bit,
    respawn_generators,
)


@pytest.fixture
def points(rng):
    p = empty_structured_array(4000, names=["x", "y"])
    p["x"] = rng.uniform(0, 1, 4000)
    p["y"] = rng.normal(size=4000)
    p["logL"] = rng.normal(size=4000)
    p["it"] = rng.integers(0, 10_000, 4000)
    return p


def test_hold_out_bit_balanced_and_fixed(points):
    bit = hold_out_bit(points)
    assert set(np.unique(bit)) == {0, 1}
    assert abs(bit.mean() - 0.5) < 0.04
    np.testing.assert_array_equal(bit, hold_out_bit(points.copy()))


def test_hold_out_bit_independent_of_position(points):
    bit = hold_out_bit(points)
    left = points["x"] < 0.5
    assert abs(bit[left].mean() - bit[~left].mean()) < 0.06


def test_gap_splitter_circle_across_seam(rng):
    bulk = np.mod(rng.normal(1.0, 0.04, 900), 1.0)
    island = rng.uniform(0.45, 0.5, 100)
    c = np.r_[bulk, island][:, None]
    cut = GapSplitter().propose(c, [1.0])
    assert cut is not None
    labels = cut.classify(c)
    counts = np.bincount(labels)
    assert sorted(counts.tolist()) == [100, 900]
    # the same arc on both sides of the seam
    assert len(np.unique(labels[:900])) == 1


def test_gap_splitter_single_blob(rng):
    c = np.mod(rng.normal(0.0, 0.05, 1000), 1.0)[:, None]
    assert GapSplitter().propose(c, [1.0]) is None
    assert GapSplitter().propose(c, [1.0], force=True) is None


def test_gap_splitter_max_share(rng):
    c = np.r_[rng.uniform(0.1, 0.2, 600), rng.uniform(0.6, 0.7, 400)][:, None]
    assert GapSplitter(max_share=0.25).propose(c, [1.0]) is None
    assert GapSplitter(max_share=0.25).propose(c, [1.0], force=True)
    cut = GapSplitter(max_share=0.5).propose(c, [1.0])
    assert sorted(np.bincount(cut.classify(c)).tolist()) == [400, 600]


def test_gap_splitter_merges_stragglers(rng):
    c = np.r_[rng.normal(0, 1, 900), rng.normal(10, 0.5, 100), [5.0, 5.1]]
    cut = GapSplitter().propose(c[:, None], [None])
    assert cut.n_labels == 2


def test_gap_splitter_min_points(rng):
    c = np.r_[rng.normal(0, 1, 990), rng.normal(10, 0.5, 10)]
    assert GapSplitter(min_points=25).propose(c[:, None], [None]) is None


def test_gap_splitter_max_labels(rng):
    c = np.r_[
        rng.normal(0, 0.3, 500),
        rng.normal(5, 0.3, 300),
        rng.normal(10, 0.3, 200),
    ][:, None]
    assert GapSplitter(max_share=0.5).propose(c, [None]).n_labels == 3
    cut = GapSplitter(max_share=0.5).propose(c, [None], max_labels=2)
    assert cut.n_labels == 2


def test_interval_cut_periodic_labels():
    cut = IntervalCut([0.2, 0.7], period=1.0)
    labels = cut.classify(np.array([0.1, 0.3, 0.8, 1.1, -0.2])[:, None])
    np.testing.assert_array_equal(labels, [1, 0, 1, 1, 1])


def test_linear_cut():
    cut = LinearCut([1.0, -1.0], 0.0)
    labels = cut.classify(np.array([[1.0, 0.0], [0.0, 1.0]]))
    np.testing.assert_array_equal(labels, [1, 0])


def test_membership_conjunction_and_pickle(points):
    coords = ParameterCoordinates(["x"], periods=[None])
    first = RegionMembership().child(coords, IntervalCut([0.5]), 1)
    second = first.child(
        ParameterCoordinates(["y"], periods=[None]), LinearCut([1.0], 0.0), 0
    )
    inside = second.contains(points)
    expected = (points["x"] > 0.5) & (points["y"] <= 0.0)
    np.testing.assert_array_equal(inside, expected)
    again = pickle.loads(pickle.dumps(second))
    np.testing.assert_array_equal(again.contains(points), expected)


def test_build_from_spec():
    splitter = build({"max_share": 0.4}, GapSplitter)
    assert isinstance(splitter, GapSplitter) and splitter.max_share == 0.4
    assert isinstance(build("nessai.partition:GapSplitter", None), GapSplitter)
    assert isinstance(build(None, GapSplitter), GapSplitter)


def test_respawn_generators():
    rng = np.random.default_rng(0)

    class Holder:
        pass

    a = Holder()
    shared = np.random.default_rng(1)
    a.rng = shared
    a.child = Holder()
    a.child.rng = shared
    a.items = {"g": np.random.default_rng(2)}
    n = respawn_generators(a, rng)
    assert n == 2
    assert a.rng is not shared and a.rng is a.child.rng
    assert a.items["g"] is not a.rng


def _run_state(state, logLs, nlive):
    for logL in logLs:
        state.increment(logL, nlive=nlive)


def test_unsplit_finalise_unchanged(rng):
    logLs = np.sort(rng.normal(size=300))
    state = _NSIntegralState(50)
    _run_state(state, logLs[:250], 50)
    for i, logL in enumerate(logLs[250:]):
        state.increment(logL, nlive=50 - i)
    logZ = state.finalise()
    assert not state.partitioned
    assert len(state.log_dx) == 300
    assert np.isfinite(logZ)


def test_split_state_volumes():
    state = _NSIntegralState(10)
    for logL in [-5.0, -4.0]:
        state.increment(logL, nlive=10)
    logw = state.logw
    state.split(0, [1, 2], np.log([0.25, 0.75]))
    assert state.partitioned
    assert np.isclose(state.region_logw[1], logw + np.log(0.25))
    assert np.isclose(state.logw, logw)
    state.increment(-3.0, nlive=4, region=1)
    assert np.isclose(state.region_logw[1], logw + np.log(0.25) - 0.25)
    assert np.isclose(state.region_logw[2], logw + np.log(0.75))
    assert np.isclose(
        state.logw, np.logaddexp(state.region_logw[1], state.region_logw[2])
    )
    with pytest.raises(RuntimeError):
        state.increment(-2.0)


def test_split_state_total_volume_and_evidence(rng):
    """Every dead point's volume adds up to the whole prior, and the
    evidence is the rectangle rule over the merged points."""
    state = _NSIntegralState(20)
    for logL in np.sort(rng.normal(-10, 1, 40)):
        state.increment(logL, nlive=20)
    state.split(0, [1, 2], np.log([0.4, 0.6]))
    a = np.sort(rng.normal(-5, 1, 30))
    b = np.sort(rng.normal(-5, 1, 30))
    # the lowest of either region first
    events = sorted([(x, 1) for x in a[:20]] + [(x, 2) for x in b[:20]])
    for logL, r in events:
        state.increment(logL, nlive=8 if r == 1 else 12, region=r)
    state.retire(1, a[20:28])
    order = state.finalise_partitioned({2: b[20:32]})
    log_dx = np.r_[state.log_dx, state.retired["log_dx"]]
    assert np.isclose(logsumexp(log_dx), 0.0)
    assert np.all(np.diff(state.logLs[1:]) >= 0)
    assert len(order) == len(state.logLs) - 1
    logL = np.r_[state.threshold_logLs[1:], state.retired["logL"]]
    assert np.isclose(state.logZ, logsumexp(logL + log_dx))
    assert np.isclose(logsumexp(state.log_posterior_weights), 0.0)
    assert state.log_vols[-1] == -np.inf
    assert np.isclose(state.log_vols[0], 0.0)
    assert np.all(np.diff(state.log_vols) <= 0)
    assert set(state.region_log_evidence()) == {None, 1, 2}
    # the weights rebuilt from the effective nlive (as nessai-bilby does)
    _, log_w = compute_weights(logL[order], np.array(state.nlive))
    w = np.exp(log_w - logsumexp(log_w))
    expected = np.exp(state.log_posterior_weights)
    # equal up to the last point (whose volume nlive 1 cuts to 1 - 1/e)
    np.testing.assert_allclose(w[:-1] / w[0], expected[:-1] / expected[0])


def test_split_shares_must_sum_to_one():
    state = _NSIntegralState(10)
    state.increment(-1.0)
    with pytest.raises(ValueError):
        state.split(0, [1, 2], np.log([0.3, 0.3]))


def test_core_splitter_ring_tail(rng):
    """A narrow core with a diffuse tail round the circle: the core is cut
    from the rest; no cut without a tail or for a broad distribution."""
    core = np.mod(rng.normal(0.44, 0.015, 1220), 1.0)
    tail = rng.uniform(0, 1, 18)
    c = np.r_[core, tail][:, None]
    cut = CoreSplitter().propose(c, [1.0])
    assert cut is not None
    labels = cut.classify(c)
    inside = labels == cut.info["core_label"]
    assert inside[:1220].mean() > 0.999
    assert (~inside[1220:]).sum() >= 15
    assert CoreSplitter().propose(core[:, None], [1.0]) is None
    assert (
        CoreSplitter().propose(rng.uniform(0, 1, 1200)[:, None], [1.0]) is None
    )


def test_core_splitter_line(rng):
    c = np.r_[rng.normal(0, 0.1, 1000), rng.uniform(5, 10, 50)][:, None]
    cut = CoreSplitter().propose(c, [None])
    assert cut is not None and cut.n_labels == 2
    assert (
        CoreSplitter().propose(rng.normal(size=1000)[:, None], [None]) is None
    )


def test_first_of(rng):
    core = np.mod(rng.normal(0.44, 0.015, 1220), 1.0)
    c = np.r_[core, rng.uniform(0, 1, 18)][:, None]
    splitter = build(
        {"splitters": [{"class": "GapSplitter"}, {"class": "CoreSplitter"}]},
        FirstOf,
    )
    assert isinstance(splitter.splitters[1], CoreSplitter)
    cut = splitter.propose(c, [1.0])
    assert "core_width" in cut.info
    gap = np.r_[
        np.mod(rng.normal(1.0, 0.04, 900), 1), rng.uniform(0.45, 0.5, 100)
    ]
    assert "gap_ratios" in splitter.propose(gap[:, None], [1.0]).info
