"""
Test the discrete group-mixture flow model.
"""

from unittest.mock import create_autospec

import numpy as np
import pytest
import torch

from nessai.flowmodel.group_mixture import (
    DiscreteGroupMixtureFlowWrapper,
    GroupFlowProposalMixin,
    GroupMixtureFlowModel,
    make_group_mixture_flow,
)
from nessai.flows.utils import configure_model
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
