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


# keep / refit (branch_weights_keep, branch_weights_refit)

from functools import partial  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock, create_autospec  # noqa: E402

from nessai.proposal.flowproposal.base import BaseFlowProposal  # noqa: E402
from nessai.proposal.flowproposal.flowproposal import FlowProposal  # noqa: E402


def _proposal(wrapper, params=None, keep=True):
    p = SimpleNamespace(
        branch_weights=True,
        branch_weights_keep=keep,
        branch_weights_kwargs={},
        branch_weights_params=params,
        branch_weights_history=[],
        flow=SimpleNamespace(model=wrapper),
        training_count=3,
        _truncation_scheme=None,
        current_live_points=np.zeros(10),
        training_data=None,
    )
    for name in ("_project_branch_weights", "_install_branch_weights",
                 "_clear_branch_weights", "_update_branch_weights"):
        setattr(p, name, partial(getattr(BaseFlowProposal, name), p))
    if params is not None:
        wrapper.set_branch_weights(params)
    return p


@pytest.fixture()
def fit_stub(monkeypatch):
    """fit_and_check returning ``stub.result`` and recording the weights in
    place it was called with."""
    stub = SimpleNamespace(result=None, seen=[])

    def fake(proposal, x, quantile, **kwargs):
        w0 = proposal.flow.model.proposal_weights.detach().numpy().copy()
        stub.seen.append(w0)
        if stub.result is None:
            return None, dict(accepted=False, reason="decomposition error")
        return stub.result[None, :], dict(accepted=True, weights_in_place=w0[None, :],
                                          heldout_gain=1.0, heldout_excluded=0.0)

    monkeypatch.setattr(BW, "fit_and_check", fake)
    monkeypatch.setattr(BW, "supports_branch_weights", lambda model: True)
    return stub


def test_projection_follows_the_active_elements(wrapper):
    wrapper.weights.copy_(torch.tensor([0.5, 0.5, 0.0, 0.0]))
    p = _proposal(wrapper)
    w = p._project_branch_weights(np.array([0.6, 0.0, 0.4, 0.0]))
    # element 1 active only now: its live-point weight; element 2 inactive now
    assert np.allclose(w, np.array([0.6, 0.5, 0.0, 0.0]) / 1.1)


def test_projection_none_when_the_group_changed(wrapper):
    p = _proposal(wrapper)
    assert p._project_branch_weights(np.ones(3) / 3) is None


def test_rejected_fit_after_training_keeps_the_previous_weights(wrapper, fit_stub):
    wrapper.weights.copy_(torch.tensor([0.25, 0.25, 0.25, 0.25]))
    previous = np.array([0.7, 0.1, 0.1, 0.1])
    p = _proposal(wrapper)  # cleared by the training
    p._update_branch_weights(previous=previous)
    assert np.allclose(wrapper.fitted_branch_weights, previous)
    assert p.branch_weights_history[-1]["kept_previous"]


def test_rejected_fit_without_keep_leaves_none(wrapper, fit_stub):
    p = _proposal(wrapper, keep=False)
    p._update_branch_weights(previous=np.array([0.7, 0.1, 0.1, 0.1]))
    assert wrapper.fitted_branch_weights is None
    assert p.branch_weights_params is None


def test_refit_starts_from_the_projected_weights_in_place(wrapper, fit_stub):
    wrapper.weights.copy_(torch.tensor([0.5, 0.5, 0.0, 0.0]))
    p = _proposal(wrapper, params=np.array([0.5, 0.5, 0.0, 0.0]))
    # element 1 lost its live points since the last fit
    wrapper.weights.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    p._update_branch_weights(refit=True)
    assert np.allclose(fit_stub.seen[-1], [1.0, 0.0, 0.0, 0.0])
    # a rejected refit keeps them
    assert np.allclose(wrapper.fitted_branch_weights, [1.0, 0.0, 0.0, 0.0])
    assert p.branch_weights_history[-1]["refit"]


def test_accepted_refit_installs_the_new_weights(wrapper, fit_stub):
    wrapper.weights.copy_(torch.tensor([0.25, 0.25, 0.25, 0.25]))
    p = _proposal(wrapper, params=np.array([0.25, 0.25, 0.25, 0.25]))
    fit_stub.result = np.array([0.4, 0.4, 0.1, 0.1])
    p._update_branch_weights(refit=True)
    assert np.allclose(wrapper.fitted_branch_weights, [0.4, 0.4, 0.1, 0.1])
    assert np.allclose(p.branch_weights_params, [0.4, 0.4, 0.1, 0.1])


class _Stop(Exception):
    pass


@pytest.mark.parametrize("refit, pilot, called", [
    (True, False, True), (False, False, False), (True, True, False)])
def test_populate_refits_before_the_pool(refit, pilot, called):
    proposal = create_autospec(FlowProposal)
    proposal.initialised = True
    proposal.branch_weights = True
    proposal.branch_weights_refit = refit
    proposal.training_count = 2
    proposal.current_live_points = np.zeros(5)
    proposal._truncation_scheme = MagicMock()
    proposal._truncation_scheme.prepare.side_effect = _Stop
    with pytest.raises(_Stop):
        FlowProposal.populate(proposal, None, n_samples=10, pilot=pilot)
    if called:
        proposal._update_branch_weights.assert_called_once_with(refit=True)
    else:
        proposal._update_branch_weights.assert_not_called()


# coverage weights (method="coverage")


def _coverage_setup(monkeypatch, u_by_element, live_w, w0=None):
    """One expert, no mirror copies: points with base levels ``u`` (log b -
    log pi) in the given elements, the weights in place ``w0``."""
    k = len(live_w)
    w0 = np.full(k, 1.0 / k) if w0 is None else np.asarray(w0, float)
    g = np.concatenate([np.full(len(v), e) for e, v in u_by_element.items()])
    u = np.concatenate(list(u_by_element.values()))
    lp = (u + np.log(w0[g]))[None, :, None]  # log p = 0
    a = g[None, :, None]
    monkeypatch.setattr(BW, "branch_terms", lambda p, x: (lp, a, w0[None, :]))
    wrapper = SimpleNamespace(weights=torch.tensor(live_w, dtype=torch.float64))
    model = SimpleNamespace(experts=None)
    prop = SimpleNamespace(
        flow=SimpleNamespace(model=wrapper),
        log_prior=lambda x: np.zeros(len(u)),
    )
    monkeypatch.setattr(BW, "_experts", lambda m: [wrapper])
    return prop, np.zeros(len(u))


def test_coverage_equalises_the_alpha_quantile(monkeypatch):
    rng = np.random.default_rng(0)
    u = {0: rng.normal(0.0, 1.0, 4000), 1: rng.normal(2.0, 1.0, 4000)}
    prop, x = _coverage_setup(monkeypatch, u, [0.5, 0.5, 0.0, 0.0])
    w, info = BW.coverage_weights(prop, x, 0.01, estimator="quantile")
    assert info["accepted"]
    q = {e: np.quantile(v, 0.01) for e, v in u.items()}
    # the level log w + u has the same 1% quantile in both elements
    assert np.log(w[0, 0]) + q[0] == pytest.approx(np.log(w[0, 1]) + q[1])
    assert w[0, 2] == 0 and w[0, 3] == 0


def test_coverage_does_not_depend_on_the_weights_in_place(monkeypatch):
    rng = np.random.default_rng(1)
    u = {0: rng.normal(0.0, 1.0, 1000), 1: rng.normal(1.0, 1.0, 1000)}
    prop, x = _coverage_setup(monkeypatch, u, [0.5, 0.5, 0, 0])
    w_a, _ = BW.coverage_weights(prop, x, 0.01)
    prop, x = _coverage_setup(monkeypatch, u, [0.5, 0.5, 0, 0], w0=[0.9, 0.05, 0.03, 0.02])
    w_b, _ = BW.coverage_weights(prop, x, 0.01)
    assert np.allclose(w_a, w_b)


def test_coverage_small_element_from_its_minimum(monkeypatch):
    rng = np.random.default_rng(2)
    big = rng.normal(0.0, 1.0, 5000)
    small = np.array([3.0, 3.5, 4.0])  # 3 points: minimum at the 1/4 quantile
    prop, x = _coverage_setup(monkeypatch, {0: big, 1: small}, [0.5, 0.5, 0, 0])
    w, info = BW.coverage_weights(prop, x, 0.01, estimator="quantile")
    ref = big - np.quantile(big, 0.01)
    q_small = small.min() - np.quantile(ref, 0.25)
    assert info["n_small_elements"] == 1
    assert np.log(w[0, 0] / w[0, 1]) == pytest.approx(q_small - np.quantile(big, 0.01))


def test_coverage_skips_inactive_elements(monkeypatch):
    rng = np.random.default_rng(3)
    u = {0: rng.normal(0.0, 1.0, 500), 1: rng.normal(0.0, 1.0, 500)}
    prop, x = _coverage_setup(monkeypatch, u, [1.0, 0.0, 0.0, 0.0])
    w, _ = BW.coverage_weights(prop, x, 0.01)
    assert w[0, 1] == 0 and w[0, 0] == pytest.approx(1.0)


def test_pooled_coverage_shifts_by_the_medians(monkeypatch):
    """Elements that differ only in location get weights in the ratio of
    their medians' shift, up to the median's standard error."""
    rng = np.random.default_rng(4)
    u = {0: rng.normal(0.0, 1.0, 60), 1: rng.normal(2.0, 1.0, 60)}
    prop, x = _coverage_setup(monkeypatch, u, [0.5, 0.5, 0, 0])
    w, info = BW.coverage_weights(prop, x, 0.01, estimator="pooled")
    assert info["estimator"] == "pooled" and info["accepted"]
    assert np.log(w[0, 0] / w[0, 1]) == pytest.approx(np.median(u[1]) - np.median(u[0]))


def test_pooled_coverage_favours_the_poorly_measured(monkeypatch):
    """At equal medians, an element with few points gets more weight than a
    well-measured one (its median's standard error is larger)."""
    rng = np.random.default_rng(5)
    big = rng.normal(0.0, 1.0, 2000)
    small = np.array([-0.1, 0.0, 0.1])
    prop, x = _coverage_setup(monkeypatch, {0: big - np.median(big), 1: small}, [0.5, 0.5, 0, 0])
    w, _ = BW.coverage_weights(prop, x, 0.01, estimator="pooled")
    sigma_term = lambda n: 1.25 / np.sqrt(n)
    assert w[0, 1] > w[0, 0]
    assert np.log(w[0, 1] / w[0, 0]) == pytest.approx(
        sigma_term(3) - sigma_term(2000), rel=0.15)


def test_pooled_coverage_tail_from_the_pooled_residuals(monkeypatch):
    """The pooled alpha-quantile sets the level: with one large element it
    reduces to that element's own quantile (up to the standard error)."""
    rng = np.random.default_rng(6)
    u = {0: rng.normal(0.0, 1.0, 20000), 1: rng.normal(0.0, 1.0, 20000)}
    prop, x = _coverage_setup(monkeypatch, u, [0.5, 0.5, 0, 0])
    w_p, _ = BW.coverage_weights(prop, x, 0.01, estimator="pooled")
    w_q, _ = BW.coverage_weights(prop, x, 0.01, estimator="quantile")
    assert np.allclose(w_p, w_q, atol=0.02)
