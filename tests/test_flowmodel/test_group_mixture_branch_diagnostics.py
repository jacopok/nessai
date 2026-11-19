"""Per-branch draw diagnostics of the group-mixture proposal."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nessai.flowmodel.group_mixture import GroupFlowProposalMixin
from nessai.livepoint import empty_structured_array


class _Stub(GroupFlowProposalMixin):
    """The mixin on a flow whose branch is floor(x) (x in [0, k))."""

    def __init__(self, k):
        def assign(xp):
            g = torch.floor(xp[:, 0]).long()
            return g, None, torch.ones(len(g), dtype=torch.bool)

        self.flow = SimpleNamespace(
            model=SimpleNamespace(group_size=k, _assign_branch=assign)
        )
        self.training_count = 3

    def _training_data_as_prime_tensor(self, x):
        return torch.as_tensor(np.array(x["x"], dtype=float))[:, None]


def _points(xs, logl=None):
    out = empty_structured_array(len(xs), names=["x"])
    out["x"] = xs
    out["logL"] = np.arange(len(xs), dtype=float) if logl is None else logl
    return out


def _draw(stub, g, accepted, over=False):
    stub.last_draw_element = g
    stub.last_draw_log_excess = 1.0 if over else -1.0
    stub.record_draw_outcome(accepted)


def test_counts_per_branch():
    stub = _Stub(3)
    _draw(stub, 0, True)
    _draw(stub, 0, False)
    _draw(stub, 2, True, over=True)
    _draw(stub, -1, True)  # unclaimed: not counted
    assert stub._branch_counts.tolist() == [[2, 0, 1], [1, 0, 1], [0, 0, 1]]


def test_summary_excess_and_poor_branches():
    stub = _Stub(2)
    # branch 0: 1000 live points, accepts half; branch 1: 10 live points,
    # accepts 1 in 100
    for i in range(1000):
        _draw(stub, 0, i % 2 == 0)
    for i in range(1000):
        _draw(stub, 1, i % 100 == 0)
    live = _points(np.r_[np.full(1000, 0.5), np.full(10, 1.5)])
    rec = stub._branch_summary(live)
    assert rec["reference_acceptance"] == pytest.approx(0.5)
    assert rec["acceptance"] == pytest.approx(510 / 2000)
    assert rec["excess"] == pytest.approx(1 - (510 / 2000) / 0.5)
    assert rec["n_poor"] == 1
    assert rec["poor_evaluated"] == pytest.approx(0.5)
    assert rec["poor_inserted"] == pytest.approx(10 / 510)
    assert rec["live"] == [1000, 10]
    assert stub.branch_diagnostics_history == [rec]
    # the counts restart
    assert stub._branch_counts is None


def test_summary_insertion_ratio_flags_an_undersampled_branch():
    stub = _Stub(2)
    # equal live shares, but branch 1 gets a fifth of branch 0's insertions
    for i in range(500):
        _draw(stub, 0, True)
    for i in range(100):
        _draw(stub, 1, True)
    live = _points(np.r_[np.full(500, 0.5), np.full(500, 1.5)])
    rec = stub._branch_summary(live)
    assert rec["worst_insertion_branch"] == 1
    assert rec["worst_insertion_ratio"] == pytest.approx(100 / 300)


def test_summary_over_cap_per_branch():
    stub = _Stub(2)
    for i in range(400):
        _draw(stub, 0, True, over=i < 4)
        _draw(stub, 1, True, over=i < 40)
    rec = stub._branch_summary(_points(np.r_[np.full(10, 0.5), np.full(10, 1.5)]))
    assert rec["over_cap_rate"] == pytest.approx(44 / 800)
    assert rec["worst_over_cap"] == pytest.approx(0.1)
    assert rec["worst_over_cap_branch"] == 1


def test_summary_without_draws_is_none():
    stub = _Stub(2)
    assert stub._branch_summary(_points(np.array([0.5]))) is None
