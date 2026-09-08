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
    # log_q equals the full-mixture density, except it is floored at the
    # honest single-branch generative value (>= never below log_prob).
    assert torch.isfinite(log_q).all()
    assert (log_q >= wrapper.log_prob(x) - 1e-4).all()


def test_forward_and_log_prob_shapes(wrapper, rng):
    x = points_in_element(0, 10, rng)
    z, log_prob = wrapper.forward_and_log_prob(x)
    assert z.shape == (10, 2)
    assert torch.allclose(log_prob, wrapper.log_prob(x), atol=1e-5)


def test_inverse_log_j_matches_log_prob(wrapper):
    z = torch.randn(64, 2)
    x, log_j = wrapper.inverse(z)
    latent_log_prob = wrapper.base_flow.base_distribution_log_prob(z)
    # latent_log_prob - log_j is the proposal density used by
    # FlowProposal.backward_pass. With truncation on (default) a draw whose
    # canonical representative leaves the fundamental domain is discarded
    # (-inf); the rest carry the single-branch generative density.
    log_q = latent_log_prob - log_j
    finite = torch.isfinite(log_q)
    assert finite.any()
    assert (log_q[finite] >= wrapper.log_prob(x)[finite] - 1e-4).all()


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
    # The physical side is carried in float64 so a large-magnitude physical
    # parameter survives the round trip (see ``ReparamBridge``).
    assert phys.dtype is torch.float64
    sx = torch.sigmoid(prime[:, 0])
    assert torch.allclose(phys[:, 0], sx.double(), atol=1e-5)
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
    assert torch.allclose(p_a.double(), p_r, atol=1e-4)
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
    honest = w.log_prob(x)
    ok = torch.isfinite(log_q) & torch.isfinite(honest)
    assert ok.sum() > 128
    # log_q tracks the non-measure-preserving log_prob, floored at the honest
    # single-branch generative density.
    assert (log_q[ok] >= honest[ok] - 1e-4).all()
    assert (log_q[ok] <= honest[ok] + 1e-3).float().mean() > 0.5


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
    honest = w.log_prob(x)
    assert torch.isfinite(log_q).all()
    # floored at the single-branch generative density (q0 has mass outside the
    # prime domain, so the single term can exceed the pruned full mixture).
    assert (log_q >= honest - 1e-4).all()


def test_sample_and_log_prob_floored_at_single_branch(base_flow):
    """log q from the generator is floored at the honest single-branch value.

    ``clamped_shift`` is an integer shift of ``x`` that is exactly measure
    preserving for ``|x| <= 1`` but clamps ``x`` into ``[-1, 1]`` first, so it
    is not injective outside that box -- the minimal stand-in for a group
    action defined via ``clamp`` / ``asin`` / ``atan2`` on physical angles.
    With the base standardisation active the canonical draw routinely lands
    outside the box; the full-mixture recompute can then degenerate, so
    ``_mixture_log_prob_from_canonical`` floors it at the single-branch
    generative density and never reports a spuriously tiny log q.
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
        truncate_base_to_domain=False,
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

    # The non-injective action forces the single-branch shortcut to fall back
    # for a sizeable fraction of the batch; that fraction is recorded, split
    # into its two disjoint causes.
    assert 0.0 < w._last_leakage_fraction <= 1.0
    assert w._last_leakage_fraction == pytest.approx(
        w._last_leakage_fraction_domain + w._last_leakage_fraction_roundtrip
    )
    # ``canon`` routinely leaks past the domain here; the clamp is the
    # identity inside it, so that leak is what drives the fallback.
    assert w._last_leakage_fraction_domain > 0.0

    # Every draw keeps a finite log q, floored at the single-branch value so it
    # is never a downward spike below what log_prob gives.
    honest = w.log_prob(x)
    assert torch.isfinite(log_q).all()
    assert (log_q >= honest - 1e-3).all()

    # Where the full-mixture recompute is well behaved (the majority) the two
    # still agree; only the degenerate minority is lifted above it.
    assert (log_q <= honest + 1e-3).float().mean() > 0.5


def test_truncate_base_to_domain_default_on(base_flow):
    """Truncation is on by default when a domain predicate is available."""
    w = DiscreteGroupMixtureFlowWrapper(
        base_flow=base_flow,
        num_features=2,
        group_action_fn=shift_group_action,
        group_size=GROUP_SIZE,
        param_names=PARAM_NAMES,
        in_fundamental_domain=in_fundamental_domain,
    )
    assert w._truncate is True


def test_truncate_base_to_domain_noop_without_predicate(base_flow, caplog):
    """Without a domain predicate truncation silently disables itself."""
    with caplog.at_level("INFO"):
        w = DiscreteGroupMixtureFlowWrapper(
            base_flow=base_flow,
            num_features=2,
            group_action_fn=shift_group_action,
            group_size=GROUP_SIZE,
            param_names=PARAM_NAMES,
        )
    assert w._truncate is False
    assert any(
        "truncation is disabled" in r.getMessage() for r in caplog.records
    )
    # sample_and_log_prob still works, via the leaky fallback path.
    x, log_q = w.sample_and_log_prob(16)
    assert torch.isfinite(log_q).all()


def test_truncate_base_to_domain_normalises_and_is_consistent(base_flow):
    """Rejecting out-of-domain draws makes the single-branch density exact.

    The base flow is untrained (~unit normal) while the fundamental domain is
    ``x in [0, 1)``, so a large fraction of canonical draws leak out. With
    ``truncate_base_to_domain`` those are rejected, ``log Z`` is estimated from
    the acceptance rate, and every surviving sample lands in the domain.
    """
    w = DiscreteGroupMixtureFlowWrapper(
        base_flow=base_flow,
        num_features=2,
        group_action_fn=shift_group_action,
        group_size=GROUP_SIZE,
        param_names=PARAM_NAMES,
        in_fundamental_domain=in_fundamental_domain,
        truncate_base_to_domain=True,
    )
    w.eval()

    torch.manual_seed(0)
    x, log_q = w.sample_and_log_prob(2000)

    # Every returned sample's canonical representative is in the domain.
    _, _, claimed = w._assign_branch(x)
    assert claimed.all()
    assert x.shape[0] == 2000

    # log Z was estimated from the rejection rate; real leakage -> Z < 1.
    assert w._domain_mass_seen
    assert w._log_domain_mass.item() < -0.05

    # The estimate matches the empirical base-flow mass in the domain.
    torch.manual_seed(1)
    u = w.base_flow.sample(20000)
    modes = torch.zeros(20000, dtype=torch.long)
    canon = w._destandardise(u, modes)
    emp = w._in_domain(canon).float().mean().item()
    assert w._log_domain_mass.exp().item() == pytest.approx(emp, abs=0.05)

    # sample_and_log_prob agrees with log_prob on its own draws, and the
    # -log Z offset is applied consistently to both paths.
    lp = w.log_prob(x)
    assert torch.std(log_q - lp).item() < 1e-3
    assert torch.allclose(lp, w._raw_log_prob(x) - w._log_domain_mass)

    # No residual leakage flagged inside the scored (already-filtered) batch.
    assert w._last_leakage_fraction == 0.0


def test_truncate_base_to_domain_shifts_log_prob_by_log_z(base_flow):
    """``log_prob`` with truncation is the raw mixture density minus ``log Z``."""
    common = dict(
        base_flow=base_flow,
        num_features=2,
        group_action_fn=shift_group_action,
        group_size=GROUP_SIZE,
        param_names=PARAM_NAMES,
        in_fundamental_domain=in_fundamental_domain,
    )
    plain = DiscreteGroupMixtureFlowWrapper(**common)
    trunc = DiscreteGroupMixtureFlowWrapper(
        **common, truncate_base_to_domain=True
    )
    plain.eval()
    trunc.eval()
    trunc._log_domain_mass.fill_(-1.1)
    trunc._domain_mass_seen = True

    rng = np.random.default_rng(0)
    x = points_in_element(0, 64, rng)
    assert torch.allclose(trunc.log_prob(x), plain.log_prob(x) + 1.1, atol=1e-5)


def test_truncate_base_to_domain_discards_on_inverse_path(base_flow):
    """The backward_pass / ``inverse`` path discards out-of-domain draws too.

    ``FlowProposal.populate`` samples through ``inverse``, not
    ``sample_and_log_prob``, so truncation has to bite there: an out-of-domain
    canonical representative gets ``log_q = -inf`` (dropped by
    ``backward_pass``'s finite-log_prob filter) and the leak rate still feeds
    the ``log Z`` estimate.
    """
    w = DiscreteGroupMixtureFlowWrapper(
        base_flow=base_flow,
        num_features=2,
        group_action_fn=shift_group_action,
        group_size=GROUP_SIZE,
        param_names=PARAM_NAMES,
        in_fundamental_domain=in_fundamental_domain,
    )
    w.eval()
    torch.manual_seed(0)
    z = torch.randn(4000, 2)
    x, log_j = w.inverse(z)
    latent = w.base_flow.base_distribution_log_prob(z)
    log_q = latent - log_j

    # A non-trivial slice is discarded, the rest are finite and match log_prob
    # up to the single-branch/full-mixture gap.
    frac_discarded = float((~torch.isfinite(log_q)).float().mean())
    assert 0.05 < frac_discarded < 0.95
    finite = torch.isfinite(log_q)
    assert (log_q[finite] >= w.log_prob(x)[finite] - 1e-4).all()
    # The leak rate fed the log Z estimate.
    assert w._domain_mass_seen
    assert w._log_domain_mass.item() < 0.0
    assert w._log_domain_mass.exp().item() == pytest.approx(
        1.0 - frac_discarded, abs=0.05
    )


def test_min_canon_std_plumbed_through_factory():
    cls = make_group_mixture_flow(
        shift_group_action,
        GROUP_SIZE,
        PARAM_NAMES,
        in_fundamental_domain,
        min_canon_std=1e-5,
    )
    assert cls.min_canon_std == 1e-5


def test_update_base_standardisation_warns_when_floor_binds(base_flow, caplog):
    """A prime dim far narrower than the floor triggers a one-off warning."""
    w = DiscreteGroupMixtureFlowWrapper(
        base_flow=base_flow,
        num_features=2,
        group_action_fn=shift_group_action,
        group_size=GROUP_SIZE,
        param_names=["x", "y"],
        in_fundamental_domain=in_fundamental_domain,
        min_canon_std=1e-2,
    )
    rng = np.random.default_rng(0)
    # ``y`` is pinned ~1e-6 wide, well below the 1e-2 floor.
    data = np.stack(
        [rng.uniform(0.0, 1.0, 400), rng.normal(0.0, 1e-6, 400)], axis=1
    )
    x_train = torch.tensor(data, dtype=torch.float32)
    with caplog.at_level("WARNING"):
        w.update_base_standardisation(x_train)
        w.update_base_standardisation(x_train)
    hits = sum(
        "canonical std floored" in r.getMessage() for r in caplog.records
    )
    assert hits == 1


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


def _reflect_wrapper(base_flow, reflect_parameters):
    def reflect(d, m, inverse=False):
        sign = torch.where(m == 1, -1.0, 1.0).to(d["x"].dtype)
        return {"x": d["x"] * sign, "y": d["y"]}, torch.zeros_like(d["x"])

    return DiscreteGroupMixtureFlowWrapper(
        base_flow=base_flow,
        num_features=2,
        group_action_fn=None,
        group_size=2,
        param_names=["x", "y"],
        prime_space_action=reflect,
        prime_space_in_domain=lambda d: d["x"] >= 0.0,
        reflect_parameters=reflect_parameters,
    )


def test_reflect_parameters_only_prime_space(base_flow, caplog):
    """reflect_parameters is ignored (with a warning) off the prime-space path."""
    with caplog.at_level("WARNING"):
        w = DiscreteGroupMixtureFlowWrapper(
            base_flow=base_flow,
            num_features=2,
            group_action_fn=shift_group_action,
            group_size=GROUP_SIZE,
            param_names=PARAM_NAMES,
            in_fundamental_domain=in_fundamental_domain,
            reflect_parameters=["x"],
        )
    assert w._sign_patterns is None
    assert "only supported on the prime-space" in caplog.text


def test_reflect_base_log_prob_normalised_over_domain(base_flow):
    """sum_s q0(s.u) integrates to 1 over the x>=0 half-plane (fresh buffers,
    identity standardisation)."""
    w = _reflect_wrapper(base_flow, ["x"])
    w.eval()
    # grid over x in [0, 8], y in [-8, 8]
    xs = torch.linspace(1e-3, 8.0, 400)
    ys = torch.linspace(-8.0, 8.0, 400)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    canon = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    modes = torch.zeros(canon.shape[0], dtype=torch.long)
    with torch.no_grad():
        dens = torch.exp(
            w._base_log_prob(canon, modes) + w._canon_log_det(modes)
        )
    integral = dens.sum() * (xs[1] - xs[0]) * (ys[1] - ys[0])
    assert integral == pytest.approx(1.0, abs=0.03)

    # matches an explicit two-term logsumexp
    with torch.no_grad():
        pt = torch.tensor([[0.4, -0.7]])
        m = torch.zeros(1, dtype=torch.long)
        manual = torch.logsumexp(
            torch.stack([
                base_flow.log_prob(torch.tensor([[0.4, -0.7]])),
                base_flow.log_prob(torch.tensor([[-0.4, -0.7]])),
            ]),
            dim=0,
        )
        got = w._base_log_prob(pt, m)
    assert float(got) == pytest.approx(float(manual), abs=1e-5)


def test_reflect_standardisation_pinned_and_samples_in_domain(base_flow):
    w = _reflect_wrapper(base_flow, ["x"])
    w.eval()
    rng = np.random.default_rng(0)
    # half-normal-ish data hard against the x=0 wall
    data = np.stack(
        [np.abs(rng.normal(0.0, 0.6, 2000)), rng.normal(0.0, 1.0, 2000)],
        axis=1,
    )
    x_train = torch.tensor(data, dtype=torch.float32)
    w.update_mixture_weights(x_train)
    w.update_base_standardisation(x_train)
    # x is a reflect dim -> centre pinned to 0, scale to RMS about 0
    assert w._canon_mean[:, 0].abs().max() == 0.0
    assert (w._canon_std[:, 0] > 0).all()
    # y untouched -> ordinary mean/std
    assert w._canon_mean[0, 1].abs() < 0.2

    torch.manual_seed(0)
    xs, log_q = w.sample_and_log_prob(2000)
    assert torch.isfinite(log_q).all()
    # the mixture proposes across all tiles; only the canonical rep is folded
    with torch.no_grad():
        assigned, pre, claimed = w._assign_branch(xs)
        canon = pre[assigned, torch.arange(xs.shape[0])]
    assert (canon[claimed, 0] >= -1e-6).all()
    honest = w.log_prob(xs)
    assert torch.isfinite(honest).all()
    assert (log_q >= honest - 1e-3).all()


# --------------------------------------------------------------------------
# canonical_transform


class _SinhTransform:
    """Toy canonical transform: ``t0 = sinh(canon0)`` on the first dim only."""

    def __init__(self):
        self.idx = 0

    def bind(self, param_names):
        self.idx = 0

    def forward(self, canon):
        t = canon.clone()
        c = canon[:, self.idx]
        t[:, self.idx] = torch.sinh(c)
        # log|dt0/dcanon0| = log cosh(canon0)
        return t, torch.log(torch.cosh(c))

    def inverse(self, t):
        canon = t.clone()
        v = t[:, self.idx]
        canon[:, self.idx] = torch.asinh(v)
        # log|dcanon0/dt0| = -0.5 log(1 + t0**2)
        return canon, -0.5 * torch.log1p(v * v)


def _ct_wrapper(base_flow, transform):
    def reflect(d, m, inverse=False):
        sign = torch.where(m == 1, -1.0, 1.0).to(d["x"].dtype)
        return {"x": d["x"] * sign, "y": d["y"]}, torch.zeros_like(d["x"])

    w = DiscreteGroupMixtureFlowWrapper(
        base_flow=base_flow,
        num_features=2,
        group_action_fn=None,
        group_size=2,
        param_names=["x", "y"],
        prime_space_action=reflect,
        prime_space_in_domain=lambda d: d["x"] >= 0.0,
        canonical_transform=transform,
    )
    w.eval()
    return w


def test_canonical_transform_requires_prime_space(base_flow):
    with pytest.raises(ValueError, match="prime_space_action"):
        DiscreteGroupMixtureFlowWrapper(
            base_flow=base_flow,
            num_features=2,
            group_action_fn=shift_group_action,
            group_size=GROUP_SIZE,
            param_names=PARAM_NAMES,
            in_fundamental_domain=in_fundamental_domain,
            canonical_transform=_SinhTransform(),
        )


def test_canonical_transform_roundtrip_and_jacobian():
    t = _SinhTransform()
    canon = torch.linspace(-1.5, 1.5, 20).unsqueeze(1)
    canon = torch.cat([canon, torch.zeros_like(canon)], dim=1)
    base, ljf = t.forward(canon)
    back, lji = t.inverse(base)
    assert torch.allclose(back, canon, atol=1e-5)
    assert torch.allclose(ljf, -lji, atol=1e-5)
    # forward log-Jacobian vs finite difference
    eps = 1e-4
    d = (
        torch.sinh(canon[:, 0] + eps) - torch.sinh(canon[:, 0] - eps)
    ) / (2 * eps)
    assert torch.allclose(ljf, torch.log(d.abs()), atol=1e-3)


def test_canonical_transform_identity_matches_no_transform(base_flow, rng):
    class _Id:
        def bind(self, names):
            pass

        def forward(self, c):
            return c, c.new_zeros(c.shape[0])

        def inverse(self, t):
            return t, t.new_zeros(t.shape[0])

    x = points_in_element(0, 16, rng)
    torch.manual_seed(0)
    w_none = _ct_wrapper(base_flow, None)
    lp_none = w_none.log_prob(x)
    w_id = _ct_wrapper(base_flow, _Id())
    lp_id = w_id.log_prob(x)
    assert torch.allclose(lp_none, lp_id, atol=1e-6)


def test_canonical_transform_sample_and_log_prob_consistent(base_flow):
    w = _ct_wrapper(base_flow, _SinhTransform())
    torch.manual_seed(0)
    x, log_q = w.sample_and_log_prob(64)
    assert x.shape == (64, 2)
    honest = w.log_prob(x)
    assert torch.isfinite(log_q).all()
    assert (log_q >= honest - 1e-4).all()

    z = torch.randn(64, 2)
    x2, log_j = w.inverse(z)
    log_q2 = w.base_flow.base_distribution_log_prob(z) - log_j
    finite = torch.isfinite(log_q2)
    assert finite.any()
    assert (log_q2[finite] >= w.log_prob(x2)[finite] - 1e-4).all()




# ---------------------------------------------------------------------------
# Clustered group mixture
# ---------------------------------------------------------------------------
from nessai.flowmodel.group_mixture import (  # noqa: E402
    ClusteredGroupMixtureFlowWrapper,
    ClusteredGroupMixtureFlowModel,
    make_clustered_group_mixture_flow,
)


def _expert(base_flow_cfg=None):
    cfg = base_flow_cfg or {
        "n_inputs": 2, "ftype": "realnvp", "n_blocks": 2, "n_neurons": 4,
        "n_layers": 1, "batch_norm_between_layers": False,
    }
    return DiscreteGroupMixtureFlowWrapper(
        base_flow=configure_model(cfg),
        num_features=2,
        group_action_fn=shift_group_action,
        group_size=GROUP_SIZE,
        param_names=PARAM_NAMES,
        in_fundamental_domain=in_fundamental_domain,
    )


def _clustered_wrapper(k=2):
    w = ClusteredGroupMixtureFlowWrapper(
        [_expert() for _ in range(k)], num_features=2, min_cluster_size=10,
    )
    w.eval()
    return w


def test_clustered_k1_returns_plain_wrapper():
    cls = make_clustered_group_mixture_flow(
        n_clusters_max=1,
        group_action_fn=shift_group_action,
        group_size=GROUP_SIZE,
        param_names=PARAM_NAMES,
        in_fundamental_domain=in_fundamental_domain,
    )
    assert issubclass(cls, ClusteredGroupMixtureFlowModel)
    fm = cls(flow_config={"n_inputs": 2, "ftype": "realnvp", "n_blocks": 2,
                          "n_neurons": 4, "n_layers": 1,
                          "batch_norm_between_layers": False})
    fm.initialise()
    assert isinstance(fm.model, DiscreteGroupMixtureFlowWrapper)
    assert not isinstance(fm.model, ClusteredGroupMixtureFlowWrapper)


def test_clustered_wrapper_defaults_to_first_expert():
    w = _clustered_wrapper(2)
    assert int(w._n_active.item()) == 1
    x = points_in_element(0, 32, np.random.default_rng(0))
    # deterministic paths: log_prob and forward are exactly experts[0]
    assert torch.allclose(w.log_prob(x), w.experts[0].log_prob(x))
    torch.manual_seed(0)
    z0, j0 = w.forward(x)
    torch.manual_seed(0)
    z1, j1 = w.experts[0].forward(x)
    assert torch.allclose(z0, z1) and torch.allclose(j0, j1)
    # inverse is stochastic in both; just check the reconstruction identity
    x_i, log_j = w.inverse(torch.randn(64, 2))
    recon = w.base_distribution_log_prob(torch.zeros(64, 2))  # smoke
    assert x_i.shape == (64, 2) and log_j.shape == (64,)


def test_clustered_log_prob_is_logsumexp_of_experts():
    w = _clustered_wrapper(2)
    with torch.no_grad():
        w._n_active.fill_(2)
        w._clustering_seen.fill_(True)
        w.cluster_weights.copy_(torch.tensor([0.7, 0.3]))
    rng = np.random.default_rng(1)
    x = points_in_element(0, 64, rng)
    expected = torch.logsumexp(
        torch.stack([
            w.experts[0].log_prob(x) + np.log(0.7),
            w.experts[1].log_prob(x) + np.log(0.3),
        ]), dim=0,
    )
    assert torch.allclose(w.log_prob(x), expected, atol=1e-5)


def test_clustered_inverse_log_j_matches_log_prob():
    w = _clustered_wrapper(2)
    with torch.no_grad():
        w._n_active.fill_(2)
        w._clustering_seen.fill_(True)
        w.cluster_weights.copy_(torch.tensor([0.5, 0.5]))
        w._centroids[:2].copy_(torch.tensor([[0.0, -2.0], [0.0, 2.0]]))
    torch.manual_seed(0)
    z = torch.randn(256, 2)
    x, log_j = w.inverse(z)
    recon = w.base_distribution_log_prob(z) - log_j
    finite = torch.isfinite(recon) & torch.isfinite(w.log_prob(x))
    assert finite.float().mean() > 0.5
    assert torch.allclose(recon[finite], w.log_prob(x)[finite], atol=1e-4)


def test_clustered_clusters_bimodal_folded_data():
    w = _clustered_wrapper(3)
    rng = np.random.default_rng(0)
    # folded x in [0,1); y bimodal at +/-3
    n = 800
    x = rng.uniform(0, 1, n)
    y = np.where(rng.random(n) < 0.6, rng.normal(3, 0.4, n),
                 rng.normal(-3, 0.4, n))
    data = torch.tensor(np.stack([x, y], 1), dtype=torch.float32)
    w.update_mixture_weights(data)
    w.update_base_standardisation(data)
    assert int(w._n_active.item()) == 2
    wts = w.cluster_weights[:2].detach().numpy()
    assert abs(wts.sum() - 1.0) < 1e-5
    # routing recovers the two y-blobs
    r = w._route(data).numpy()
    hi = y[r == 0].mean(), y[r == 1].mean()
    assert abs(hi[0] - hi[1]) > 3.0


def test_clustered_stays_k1_for_unimodal_data():
    w = _clustered_wrapper(3)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 600)
    y = rng.normal(0, 1, 600)
    data = torch.tensor(np.stack([x, y], 1), dtype=torch.float32)
    w.update_mixture_weights(data)
    assert int(w._n_active.item()) == 1


def _unimodal(n, rng):
    return torch.tensor(
        np.stack([rng.uniform(0, 1, n), rng.normal(0, 1, n)], 1),
        dtype=torch.float32,
    )


def _bimodal(n, rng):
    y = np.where(rng.random(n) < 0.6, rng.normal(3, 0.4, n),
                 rng.normal(-3, 0.4, n))
    return torch.tensor(
        np.stack([rng.uniform(0, 1, n), y], 1), dtype=torch.float32
    )


def test_clustered_k_evolves_unimodal_to_bimodal_with_warm_start():
    w = _clustered_wrapper(2)
    rng = np.random.default_rng(0)

    w._cluster(_unimodal(800, rng))
    assert int(w._n_active.item()) == 1

    # a fresh (distinct data_ptr) bimodal round: k rises to 2
    w._cluster(_bimodal(800, rng))
    assert int(w._n_active.item()) == 2
    # the newly activated expert was warm-started from expert 0
    sd0 = w.experts[0].base_flow.state_dict()
    sd1 = w.experts[1].base_flow.state_dict()
    assert all(torch.allclose(sd0[k], sd1[k]) for k in sd0)
    # ... and its per-cluster standardisation is re-bootstrapped
    assert not bool(w.experts[1]._canon_seen.any())


def test_clustered_k_shrink_needs_hysteresis():
    w = ClusteredGroupMixtureFlowWrapper(
        [_expert() for _ in range(2)], num_features=2, min_cluster_size=10,
        max_cluster_overlap=0.1, k_shrink_patience=3,
    )
    rng = np.random.default_rng(1)
    w._cluster(_bimodal(800, rng))
    assert int(w._n_active.item()) == 2
    # one unimodal round -> k held at 2 (hysteresis)
    w._cluster(_unimodal(800, rng))
    assert int(w._n_active.item()) == 2
    w._cluster(_unimodal(800, rng))
    assert int(w._n_active.item()) == 2
    # third consecutive -> drops to 1
    w._cluster(_unimodal(800, rng))
    assert int(w._n_active.item()) == 1


class _FoldedBimodalModel(PeriodicModel):
    def log_likelihood(self, x):
        base = super().log_likelihood(x)
        y = np.atleast_1d(x["y"])
        blob = np.logaddexp(-0.5 * ((y - 2.5) / 0.5) ** 2,
                            -0.5 * ((y + 2.5) / 0.5) ** 2)
        return base + 0.5 * y**2 + blob  # cancel the parent's -0.5 y^2


_ClusteredPeriodicFlowModel = make_clustered_group_mixture_flow(
    n_clusters_max=2, min_cluster_size=50, max_cluster_overlap=0.15,
    group_action_fn=shift_group_action, group_size=N_PERIODS,
    param_names=["x", "y"], in_fundamental_domain=in_fundamental_domain,
)


class ClusteredPeriodicGroupFlowProposal(GroupFlowProposalMixin, FlowProposal):
    _FlowModelClass = _ClusteredPeriodicFlowModel


@pytest.mark.slow_integration_test
def test_sampling_with_clustered_group_mixture_flow(tmp_path):
    fs = FlowSampler(
        _FoldedBimodalModel(),
        output=tmp_path / "clustered",
        flow_proposal_class=ClusteredPeriodicGroupFlowProposal,
        flow_config={"model": "realnvp", "n_blocks": 2, "n_neurons": 8},
        nlive=500,
        maximum_uninformed=500,
        plot=False,
        resume=False,
        seed=1234,
    )
    fs.run(plot=False)
    model = fs.ns._flow_proposal.flow.model
    if isinstance(model, ClusteredGroupMixtureFlowWrapper):
        assert np.isclose(
            model.cluster_weights[: int(model._n_active.item())]
            .detach().cpu().numpy().sum(), 1.0)
    assert np.isfinite(fs.log_evidence)
