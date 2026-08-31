"""
Test the discrete group-mixture flow model.
"""

from unittest.mock import create_autospec

import numpy as np
import pytest
import torch

from nessai.flowmodel.group_mixture import (
    AffineBridge,
    DiscreteGroupMixtureFlowWrapper,
    GroupFlowProposalMixin,
    GroupMixtureFlowModel,
    ReparamBridge,
    make_group_mixture_flow,
)
from nessai.flows.utils import configure_model
from nessai.livepoint import empty_structured_array
from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.proposal import FlowProposal

GROUP_SIZE = 4
PARAM_NAMES = ["x", "y"]


def shift_group_action(point_dict, modes, inverse=False):
    """Integer shift of ``x`` by the group-element index."""
    shift = modes.to(point_dict["x"].dtype)
    x = point_dict["x"] - shift if inverse else point_dict["x"] + shift
    return {"x": x, "y": point_dict["y"]}


def in_fundamental_domain(point_dict):
    """Canonical representative: ``x`` in the first period ``[0, 1)``."""
    return (point_dict["x"] >= 0.0) & (point_dict["x"] < 1.0)


N_PERIODS = 5


class PeriodicModel(Model):
    """Periodic multimodal target: ``N_PERIODS`` narrow modes in ``x``."""

    def __init__(self):
        self.names = ["x", "y"]
        self.bounds = {"x": (0.0, float(N_PERIODS)), "y": (-5.0, 5.0)}
        self._centres = np.arange(N_PERIODS) + 0.5

    def log_prior(self, x):
        log_p = np.log(self.in_bounds(x), dtype="float")
        log_p -= np.log(N_PERIODS) + np.log(10.0)
        return log_p

    def log_likelihood(self, x):
        x_val = np.atleast_1d(x["x"])[:, np.newaxis]
        modes = np.exp(-0.5 * ((x_val - self._centres) / 0.1) ** 2).sum(axis=1)
        return np.log(modes) - 0.5 * np.atleast_1d(x["y"]) ** 2


PeriodicGroupFlowModel = make_group_mixture_flow(
    shift_group_action, N_PERIODS, ["x", "y"], in_fundamental_domain
)


class PeriodicGroupFlowProposal(GroupFlowProposalMixin, FlowProposal):
    _FlowModelClass = PeriodicGroupFlowModel


@pytest.fixture()
def base_flow():
    return configure_model(
        {
            "n_inputs": 2,
            "ftype": "realnvp",
            "n_blocks": 2,
            "n_neurons": 4,
            "n_layers": 1,
            "batch_norm_between_layers": False,
        }
    )


@pytest.fixture()
def wrapper(base_flow):
    flow = DiscreteGroupMixtureFlowWrapper(
        base_flow=base_flow,
        num_features=2,
        group_action_fn=shift_group_action,
        group_size=GROUP_SIZE,
        param_names=PARAM_NAMES,
        in_fundamental_domain=in_fundamental_domain,
    )
    flow.eval()
    return flow


def points_in_element(k, n, rng):
    """``n`` points whose canonical representative is group element ``k``."""
    x = rng.uniform(0.0, 1.0, n) + k
    y = rng.normal(0.0, 1.0, n)
    return torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)


def test_make_group_mixture_flow_sets_attributes():
    cls = make_group_mixture_flow(
        shift_group_action, GROUP_SIZE, PARAM_NAMES, in_fundamental_domain
    )
    assert issubclass(cls, GroupMixtureFlowModel)
    assert cls.group_size == GROUP_SIZE
    assert cls.param_names == PARAM_NAMES
    assert cls.group_action_fn(
        {"x": torch.zeros(1), "y": torch.zeros(1)}, torch.ones(1)
    )["x"].item() == pytest.approx(1.0)
    assert cls.in_fundamental_domain({"x": torch.tensor([0.5])}).item()


def test_make_group_mixture_flow_without_domain():
    cls = make_group_mixture_flow(shift_group_action, GROUP_SIZE, PARAM_NAMES)
    assert cls.in_fundamental_domain is None


def test_get_model_requires_group_config():
    model = create_autospec(GroupMixtureFlowModel)
    model.group_action_fn = None
    model.group_size = None
    with pytest.raises(ValueError, match="requires `group_action_fn`"):
        GroupMixtureFlowModel.get_model(model, {"n_inputs": 2})


def test_get_model_builds_wrapper():
    model = create_autospec(GroupMixtureFlowModel)
    model.group_action_fn = staticmethod(shift_group_action)
    model.group_size = GROUP_SIZE
    model.param_names = PARAM_NAMES
    model.in_fundamental_domain = staticmethod(in_fundamental_domain)
    config = {
        "n_inputs": 2,
        "ftype": "realnvp",
        "n_blocks": 2,
        "n_neurons": 4,
        "n_layers": 1,
        "model": "realnvp",
    }
    flow = GroupMixtureFlowModel.get_model(model, config)
    assert isinstance(flow, DiscreteGroupMixtureFlowWrapper)
    assert flow.group_size == GROUP_SIZE
    # The original config is not mutated.
    assert config["model"] == "realnvp"


def test_default_param_names(base_flow):
    wrapper = DiscreteGroupMixtureFlowWrapper(
        base_flow, 2, shift_group_action, GROUP_SIZE
    )
    assert wrapper.param_names == ["p_0", "p_1"]


def test_initial_weights_uniform(wrapper):
    assert torch.allclose(
        wrapper.weights, torch.full((GROUP_SIZE,), 1.0 / GROUP_SIZE)
    )


def test_assign_branch_assigns_to_correct_element(wrapper, rng):
    x = torch.cat([points_in_element(k, 5, rng) for k in range(GROUP_SIZE)])
    assigned, _, claimed = wrapper._assign_branch(x)
    expected = torch.arange(GROUP_SIZE).repeat_interleave(5)
    assert claimed.all()
    assert torch.equal(assigned, expected)


def test_update_mixture_weights_matches_assigned_fractions(wrapper, rng):
    counts = [10, 30, 20, 40]
    x = torch.cat([points_in_element(k, n, rng) for k, n in enumerate(counts)])
    wrapper.update_mixture_weights(x, smoothing=0.0)
    expected = torch.tensor(counts, dtype=torch.float32) / sum(counts)
    assert torch.allclose(wrapper.weights, expected, atol=1e-6)
    assert wrapper.weights.sum() == pytest.approx(1.0)


def test_update_mixture_weights_drops_persistently_empty_element(wrapper, rng):
    x = torch.cat([points_in_element(k, 10, rng) for k in range(3)])
    for _ in range(wrapper._weight_empty_patience):
        assert wrapper.weights[3] > 0.0
        wrapper.update_mixture_weights(x)
    assert wrapper.weights[3] == 0.0
    assert wrapper.weights.sum() == pytest.approx(1.0)
    # Element recovers once points return to it.
    x_all = torch.cat([x, points_in_element(3, 10, rng)])
    wrapper.update_mixture_weights(x_all)
    assert wrapper.weights[3] > 0.0


def test_set_affine_maps_updates_buffers(wrapper):
    scale = torch.tensor([2.0, 3.0])
    shift = torch.tensor([1.0, -1.0])
    wrapper.set_affine_maps(scale, shift)
    assert torch.equal(wrapper._prime_scale, scale)
    assert torch.equal(wrapper._prime_shift, shift)


def test_set_affine_maps_reexpresses_canonical_buffers(wrapper, rng):
    x = torch.cat([points_in_element(k, 50, rng) for k in range(GROUP_SIZE)])
    wrapper.update_base_standardisation(x)
    seen_mean = wrapper._canon_mean.clone()
    wrapper.set_affine_maps(torch.tensor([2.0, 2.0]), torch.tensor([0.0, 0.0]))
    # prime frame halved -> stored prime-coordinate means double.
    assert torch.allclose(wrapper._canon_mean, seen_mean / 2.0, atol=1e-5)


def test_log_prob_shape_and_finiteness(wrapper, rng):
    x = points_in_element(1, 16, rng)
    log_prob = wrapper.log_prob(x)
    assert log_prob.shape == (16,)
    assert torch.isfinite(log_prob).all()


def test_log_prob_tracks_mixture_weights(wrapper, rng):
    """A point in element k should get more likely as pi_k grows."""
    x = points_in_element(2, 8, rng)
    wrapper.weights.copy_(torch.tensor([0.7, 0.1, 0.1, 0.1]))
    low = wrapper.log_prob(x)
    wrapper.weights.copy_(torch.tensor([0.1, 0.1, 0.7, 0.1]))
    high = wrapper.log_prob(x)
    assert (high > low).all()


def test_sample_and_log_prob_consistent(wrapper):
    x, log_q = wrapper.sample_and_log_prob(32)
    assert x.shape == (32, 2)
    assert log_q.shape == (32,)
    assert torch.allclose(log_q, wrapper.log_prob(x), atol=1e-4)


def test_forward_and_log_prob_shapes(wrapper, rng):
    x = points_in_element(0, 10, rng)
    z, log_prob = wrapper.forward_and_log_prob(x)
    assert z.shape == (10, 2)
    assert torch.allclose(log_prob, wrapper.log_prob(x), atol=1e-5)


def test_inverse_log_j_matches_log_prob(wrapper):
    z = torch.randn(64, 2)
    x, log_j = wrapper.inverse(z)
    latent_log_prob = wrapper.base_flow.base_distribution_log_prob(z)
    assert torch.allclose(
        latent_log_prob - log_j, wrapper.log_prob(x), atol=1e-4
    )


def _struct(array, names):
    out = empty_structured_array(len(array), names=list(names))
    for i, name in enumerate(names):
        out[name] = array[:, i]
    return out


def _logit_reparam(prime_names, physical_names):
    """A non-affine (logit) prime<->physical map on ``x``; ``y`` affine.

    physical x in (0, 1), prime x = logit(physical x). ``y`` unchanged.
    """

    def _sig(v):
        return 1.0 / (1.0 + np.exp(-v))

    def inverse_rescale(struct):  # prime -> physical
        px, py = struct[prime_names[0]], struct[prime_names[1]]
        sx = _sig(px)
        out = _struct(np.stack([sx, py], axis=1), physical_names)
        log_j = np.log(sx) + np.log1p(-sx)  # log|dphys/dprime|
        return out, log_j

    def rescale(struct):  # physical -> prime
        x, y = struct[physical_names[0]], struct[physical_names[1]]
        x = np.clip(x, 1e-9, 1 - 1e-9)
        px = np.log(x) - np.log1p(-x)
        out = _struct(np.stack([px, y], axis=1), prime_names)
        log_j = -(np.log(x) + np.log1p(-x))  # log|dprime/dphys|
        return out, log_j

    return rescale, inverse_rescale


def _logit_bridge():
    rescale, inverse_rescale = _logit_reparam(["x", "y"], ["x", "y"])
    return ReparamBridge(
        prime_names=["x", "y"],
        physical_names=["x", "y"],
        rescale_fn=rescale,
        inverse_rescale_fn=inverse_rescale,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def test_reparam_bridge_matches_analytic_logdet():
    bridge = _logit_bridge()
    prime = torch.tensor([[0.3, 1.0], [-1.2, -0.4], [2.0, 0.1]])
    phys, L, aux = bridge.to_physical(prime)
    assert aux is None
    sx = torch.sigmoid(prime[:, 0])
    assert torch.allclose(phys[:, 0], sx, atol=1e-5)
    assert torch.allclose(L, torch.log(sx) + torch.log1p(-sx), atol=1e-5)
    # Round trip is the identity (deterministic part) and L is consistent.
    prime2, L2 = bridge.to_prime(phys)
    assert torch.allclose(prime2, prime, atol=1e-4)
    assert torch.allclose(L2, L, atol=1e-4)


def test_reparam_bridge_reproduces_affine():
    affine = AffineBridge(torch.tensor([2.0, 3.0]), torch.tensor([1.0, -1.0]))

    def rescale(struct):
        arr = np.stack([struct["x"], struct["y"]], axis=1)
        prime = (arr - np.array([1.0, -1.0])) / np.array([2.0, 3.0])
        return _struct(prime, ["x", "y"]), np.full(len(arr), -np.log(6.0))

    def inverse_rescale(struct):
        arr = np.stack([struct["x"], struct["y"]], axis=1)
        phys = arr * np.array([2.0, 3.0]) + np.array([1.0, -1.0])
        return _struct(phys, ["x", "y"]), np.full(len(arr), np.log(6.0))

    bridge = ReparamBridge(
        ["x", "y"], ["x", "y"], rescale, inverse_rescale,
        torch.float32, torch.device("cpu"),
    )
    prime = torch.randn(8, 2)
    p_a, L_a, _ = affine.to_physical(prime)
    p_r, L_r, _ = bridge.to_physical(prime)
    assert torch.allclose(p_a, p_r, atol=1e-4)
    assert torch.allclose(L_a, L_r, atol=1e-4)


def _nonlinear_wrapper(base_flow, group_size=1, **kw):
    w = DiscreteGroupMixtureFlowWrapper(
        base_flow=base_flow,
        num_features=2,
        group_action_fn=lambda d, m, inverse=False: d,
        group_size=group_size,
        param_names=["x", "y"],
        in_fundamental_domain=lambda d: torch.ones_like(
            d["x"], dtype=torch.bool
        ),
        **kw,
    )
    w.set_coordinate_bridge(_logit_bridge())
    w.eval()
    return w


def test_conj_logdet_zero_for_identity_action(base_flow):
    w = _nonlinear_wrapper(base_flow)
    z = torch.randn(16, 2)
    _, delta = w._apply_group_action(
        z, torch.zeros(16, dtype=torch.long), inverse=True
    )
    assert torch.allclose(delta, torch.zeros(16), atol=1e-4)


def test_nonlinear_bridge_sample_and_log_prob_consistent(base_flow):
    w = _nonlinear_wrapper(base_flow)
    x, log_q = w.sample_and_log_prob(64)
    assert torch.isfinite(log_q).all()
    assert torch.allclose(log_q, w.log_prob(x), atol=1e-4)


def test_conj_logdet_matches_numeric_jacobian(base_flow):
    """conj log-det of ``T^-1 . g . T`` vs a finite-difference Jacobian."""

    def prime_shift_action(d, m, inverse=False):
        s = 0.1 * m.to(d["x"].dtype)
        return {"x": d["x"] - s if inverse else d["x"] + s, "y": d["y"]}

    # Physical-space shift on the logit axis, via the bridge.
    w = DiscreteGroupMixtureFlowWrapper(
        base_flow=base_flow,
        num_features=2,
        group_action_fn=prime_shift_action,
        group_size=2,
        param_names=["x", "y"],
        in_fundamental_domain=lambda d: torch.ones_like(
            d["x"], dtype=torch.bool
        ),
    )
    w.set_coordinate_bridge(_logit_bridge())
    w.eval()

    z = torch.tensor([[0.2, 0.5], [-0.25, 1.1]], dtype=torch.float32)
    modes = torch.ones(2, dtype=torch.long)

    def mapped(zz):
        out, _ = w._apply_group_action(zz, modes, inverse=True)
        return out.double()

    eps = 1e-3
    base = mapped(z)
    _, delta = w._apply_group_action(z, modes, inverse=True)
    jac = torch.zeros(2, 2, 2, dtype=torch.float64)
    for j in range(2):
        dz = z.clone()
        dz[:, j] += eps
        jac[:, :, j] = (mapped(dz) - base) / eps
    logdet_numeric = torch.logdet(jac)
    assert torch.allclose(delta.double(), logdet_numeric, atol=2e-2)


def test_physical_action_returning_log_det(base_flow):
    """A non-measure-preserving physical action supplies its own log-det.

    Mode 1 is the involution ``x -> 1 / x`` (log-det ``-2 log|x|``, so *not*
    measure preserving). The action returns the log-determinant of the map it
    applied; both ``_apply_group_action`` and the
    ``sample_and_log_prob`` / ``log_prob`` round trip must account for it.
    """

    def invert_action(d, m, inverse=False):
        flip = (m == 1).to(d["x"].dtype)
        x_out = torch.where(m == 1, 1.0 / d["x"], d["x"])
        log_det = flip * (-2.0 * torch.log(d["x"].abs()))
        return {"x": x_out, "y": d["y"]}, log_det

    w = DiscreteGroupMixtureFlowWrapper(
        base_flow=base_flow,
        num_features=2,
        group_action_fn=invert_action,
        group_size=2,
        param_names=["x", "y"],
        # |x| < 1 is the representative; its orbit partner 1/x has |1/x| > 1.
        in_fundamental_domain=lambda d: d["x"].abs() < 1.0,
    )
    w.eval()

    z = torch.randn(32, 2)
    modes = torch.ones(32, dtype=torch.long)
    _, delta = w._apply_group_action(z, modes, inverse=True)
    assert torch.allclose(delta, -2.0 * z[:, 0].abs().log(), atol=1e-5)

    x, log_q = w.sample_and_log_prob(256)
    finite = torch.isfinite(log_q)
    assert finite.sum() > 128
    assert torch.allclose(log_q[finite], w.log_prob(x)[finite], atol=1e-4)


def test_prime_space_action_escape_hatch(base_flow):
    """Dimension-agnostic path: action + domain given directly in prime coords."""

    def reflect(d, m, inverse=False):
        sign = torch.where(m == 1, -1.0, 1.0).to(d["x"].dtype)
        return {"x": d["x"] * sign, "y": d["y"]}, torch.zeros_like(d["x"])

    def prime_in_domain(d):
        return d["x"] >= 0.0

    w = DiscreteGroupMixtureFlowWrapper(
        base_flow=base_flow,
        num_features=2,
        group_action_fn=None,
        group_size=2,
        param_names=["x", "y"],
        prime_space_action=reflect,
        prime_space_in_domain=prime_in_domain,
    )
    w.eval()
    pos = torch.tensor([[0.3, 0.1], [1.2, -0.5]])
    neg = torch.tensor([[-0.3, 0.1], [-1.2, -0.5]])
    assigned, _, claimed = w._assign_branch(torch.cat([pos, neg]))
    assert claimed.all()
    assert torch.equal(assigned, torch.tensor([0, 0, 1, 1]))
    x, log_q = w.sample_and_log_prob(48)
    assert torch.isfinite(log_q).all()
    assert torch.allclose(log_q, w.log_prob(x), atol=1e-4)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The wrapper assumes the group action is a clean bijection on the "
        "whole prime space. An action that is measure preserving only on a "
        "box and saturates outside it (e.g. nessai_gw's ETTriangleGroupAction, "
        "which does asin(clamp(sin_dec, -1, 1))) makes sample_and_log_prob "
        "and log_prob disagree once the canonical draw leaves the box."
    ),
)
def test_sample_and_log_prob_consistent_with_saturating_action(base_flow):
    """log q from the generator must match log q recomputed by log_prob.

    ``clamped_shift`` is an integer shift of ``x`` that is exactly measure
    preserving for ``|x| <= 1`` but clamps ``x`` into ``[-1, 1]`` first, so it
    is not injective outside that box -- the minimal stand-in for a group
    action defined via ``clamp`` / ``asin`` / ``atan2`` on physical angles.
    With the base standardisation active the canonical draw routinely lands
    outside the box, and the two log-density paths then diverge.
    """

    def clamped_shift(point_dict, modes, inverse=False):
        shift = modes.to(point_dict["x"].dtype)
        x = point_dict["x"].clamp(-1.0, 1.0)
        x = x - shift if inverse else x + shift
        return {"x": x, "y": point_dict["y"]}

    def fundamental_domain(point_dict):
        return (point_dict["x"] >= 0.0) & (point_dict["x"] < 1.0)

    w = DiscreteGroupMixtureFlowWrapper(
        base_flow=base_flow,
        num_features=2,
        group_action_fn=clamped_shift,
        group_size=3,
        param_names=["x", "y"],
        in_fundamental_domain=fundamental_domain,
    )
    w.eval()

    rng = np.random.default_rng(1)
    data = np.concatenate(
        [
            np.stack([rng.uniform(0.0, 1.0, 300) + k, rng.normal(0.0, 1.0, 300)], axis=1)
            for k in range(3)
        ]
    )
    x_train = torch.tensor(data, dtype=torch.float32)
    w.update_mixture_weights(x_train)
    w.update_base_standardisation(x_train)

    torch.manual_seed(0)
    x, log_q = w.sample_and_log_prob(4000)
    finite = torch.isfinite(log_q)
    assert finite.sum() > 2000
    assert torch.allclose(log_q[finite], w.log_prob(x)[finite], atol=1e-3)


@pytest.mark.slow_integration_test
def test_sampling_with_group_mixture_flow(tmp_path):
    """Sample a periodic multimodal target with the group-mixture proposal."""
    fs = FlowSampler(
        PeriodicModel(),
        output=tmp_path / "group_mixture",
        flow_proposal_class=PeriodicGroupFlowProposal,
        flow_config={"model": "realnvp", "n_blocks": 2, "n_neurons": 8},
        nlive=500,
        maximum_uninformed=500,
        plot=False,
        resume=False,
        seed=1234,
    )
    fs.run(plot=False)

    weights = fs.ns._flow_proposal.flow.model.weights.detach().cpu().numpy()
    assert np.isclose(weights.sum(), 1.0)
    assert np.isfinite(fs.log_evidence)


@pytest.mark.slow_integration_test
def test_sampling_with_nonaffine_reparameterisation(tmp_path):
    """Same target through a non-affine (logit) reparameterisation.

    Exercises the ``ReparamBridge`` path end to end: the group action /
    fundamental domain stay in physical coords while the flow sees logit
    coords, and the conjugation Jacobian is carried each batch.
    """
    fs = FlowSampler(
        PeriodicModel(),
        output=tmp_path / "group_mixture_logit",
        flow_proposal_class=PeriodicGroupFlowProposal,
        flow_config={"model": "realnvp", "n_blocks": 2, "n_neurons": 8},
        reparameterisations={"x": "logit", "y": "zscore"},
        nlive=500,
        maximum_uninformed=500,
        plot=False,
        resume=False,
        seed=1234,
    )
    fs.run(plot=False)

    model = fs.ns._flow_proposal.flow.model
    from nessai.flowmodel.group_mixture import ReparamBridge

    assert isinstance(model._bridge, ReparamBridge)
    weights = model.weights.detach().cpu().numpy()
    assert np.isclose(weights.sum(), 1.0)
    assert np.isfinite(fs.log_evidence)
