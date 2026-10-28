# -*- coding: utf-8 -*-
"""Fitted group-element weights: the wrapper's override and the fit."""

import numpy as np
import pytest
import torch

from nessai.flowmodel.group_mixture import DiscreteGroupMixtureFlowWrapper
from nessai.flows.utils import configure_model
from nessai.proposal.flowproposal import branch_weights as BW

GROUP_SIZE = 4


def shift_group_action(point_dict, modes, inverse=False):
    shift = modes.to(point_dict["x"].dtype)
    x = point_dict["x"] - shift if inverse else point_dict["x"] + shift
    return {"x": x, "y": point_dict["y"]}


def in_fundamental_domain(point_dict):
    return (point_dict["x"] >= 0.0) & (point_dict["x"] < 1.0)


@pytest.fixture()
def wrapper():
    base_flow = configure_model(
        {
            "n_inputs": 2,
            "ftype": "realnvp",
            "n_blocks": 2,
            "n_neurons": 4,
            "n_layers": 1,
            "batch_norm_between_layers": False,
        }
    )
    flow = DiscreteGroupMixtureFlowWrapper(
        base_flow=base_flow,
        num_features=2,
        group_action_fn=shift_group_action,
        group_size=GROUP_SIZE,
        param_names=["x", "y"],
        in_fundamental_domain=in_fundamental_domain,
    )
    flow.eval()
    return flow


def _points(rng, n, elements):
    k = rng.choice(elements, size=n)
    x = rng.uniform(0.0, 1.0, n) + k
    y = rng.normal(0.0, 1.0, n)
    return torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32), k


def test_override_replaces_proposal_weights(wrapper):
    w = np.array([0.7, 0.1, 0.1, 0.1])
    wrapper.set_branch_weights(w)
    assert np.allclose(wrapper.proposal_weights.numpy(), w)
    assert np.allclose(wrapper.fitted_branch_weights, w)
    wrapper.set_branch_weights(None)
    assert wrapper.fitted_branch_weights is None
    assert np.allclose(wrapper.proposal_weights.numpy(), 0.25)


def test_override_checks_shape_and_support(wrapper):
    with pytest.raises(ValueError, match="shape"):
        wrapper.set_branch_weights(np.ones(3) / 3)
    wrapper.weights.copy_(torch.tensor([0.5, 0.5, 0.0, 0.0]))
    with pytest.raises(ValueError, match="inactive"):
        wrapper.set_branch_weights(np.ones(4) / 4)


def test_reweighting_shifts_log_q_by_the_claiming_element(wrapper):
    """With a tiling domain, new weights shift each point's log q by the
    log-ratio of its claiming element's weights, as the fit assumes."""
    rng = np.random.default_rng(0)
    x, k = _points(rng, 200, np.arange(GROUP_SIZE))
    with torch.no_grad():
        lq0 = wrapper.log_prob(x).numpy()
        assigned, _, claimed = wrapper._assign_branch(x)
    assert claimed.all() and np.array_equal(assigned.numpy(), k)
    w0 = wrapper.proposal_weights.numpy().astype(float)
    w = np.array([0.55, 0.25, 0.15, 0.05])
    wrapper.set_branch_weights(w)
    with torch.no_grad():
        lq = wrapper.log_prob(x).numpy()
    lp = lq0[None, :, None].astype(float)
    a = k[None, :, None]
    lev = BW.levels_of(lp, a, np.log(w0)[None], np.log(w)[None],
                       np.zeros(len(k)))
    assert np.allclose(lev, lq, atol=1e-5)


def test_fit_moves_weight_to_the_occupied_elements():
    """Live points in two of four equal-support elements: the fit raises
    their weights and the low tail of the levels."""
    rng = np.random.default_rng(1)
    n = 2000
    k = rng.choice([0, 1], size=n, p=[0.8, 0.2])
    w0 = np.full((1, GROUP_SIZE), 0.25)
    lp = (np.log(0.25) + rng.normal(0.0, 0.3, n))[None, :, None]
    a = k[None, :, None]
    log_p = np.zeros(n)
    w = BW.fit(lp, a, w0, log_p)
    assert w.shape == w0.shape and np.isclose(w.sum(), 1.0)
    assert w[0, 0] > w[0, 1] > w[0, 2]
    assert np.isclose(w[0, 2], w[0, 3])
    lev0 = BW.levels_of(lp, a, np.log(w0), np.log(w0), log_p)
    lev = BW.levels_of(lp, a, np.log(w0), np.log(w), log_p)
    assert np.quantile(lev, 0.01) > np.quantile(lev0, 0.01)


def test_fit_keeps_inactive_elements_at_zero():
    rng = np.random.default_rng(2)
    n = 500
    k = rng.choice([0, 1], size=n)
    w0 = np.array([[0.5, 0.3, 0.2, 0.0]])
    lp = rng.normal(0.0, 0.3, n)[None, :, None]
    w = BW.fit(lp, k[None, :, None], w0, np.zeros(n))
    assert w[0, 3] == 0.0 and np.isclose(w.sum(), 1.0)


def test_unclaimed_points_are_not_shifted():
    lp = np.zeros((1, 3, 1))
    a = np.array([[[0], [-1], [1]]])
    log_w0 = np.log(np.array([[0.5, 0.5]]))
    log_w = np.log(np.array([[0.9, 0.1]]))
    lev = BW.levels_of(lp, a, log_w0, log_w, np.zeros(3))
    assert lev[1] == pytest.approx(0.0)
    assert lev[0] == pytest.approx(np.log(0.9 / 0.5))
    assert lev[2] == pytest.approx(np.log(0.1 / 0.5))
