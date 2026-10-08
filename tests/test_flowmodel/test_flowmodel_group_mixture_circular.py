"""
Group mixtures with circular base-frame coordinates and base symmetries.
"""

import math

import numpy as np
import pytest
import torch

from nessai.flowmodel.group_mixture import (
    ClusteredGroupMixtureFlowWrapper,
    DiscreteGroupMixtureFlowWrapper,
    make_clustered_group_mixture_flow,
    make_group_mixture_flow,
)
from nessai.flows.circular import CircularNeuralSplineFlow

TWO_PI = 2.0 * math.pi


@pytest.fixture(autouse=True)
def float64():
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(dtype)


def _wrap(t):
    return torch.remainder(t + math.pi, TWO_PI) - math.pi


# -- half-period shift of an angle --------------------------------------------
# prime coordinates (a, y), a in [-1, 1) with period 2; the group shifts a by
# one (half a period); fundamental domain a in [-1, 0).


def half_shift(point_dict, modes, inverse=False):
    a = torch.remainder(point_dict["a"] + 1.0 + modes.to(point_dict["a"]), 2.0)
    return {"a": a - 1.0, "y": point_dict["y"]}


def half_shift_domain(point_dict):
    return point_dict["a"] < 0.0


class AngleOfA:
    """``a in [-1, 0) -> t in [-pi, pi)``: the quotient circle."""

    def forward(self, canon):
        t = canon.clone()
        t[:, 0] = _wrap(TWO_PI * (canon[:, 0] + 1.0) - math.pi)
        return t, canon.new_full((canon.shape[0],), math.log(TWO_PI))

    def inverse(self, t):
        canon = t.clone()
        canon[:, 0] = (_wrap(t[:, 0]) + math.pi) / TWO_PI - 1.0
        return canon, t.new_full((t.shape[0],), -math.log(TWO_PI))


def perturb(module, scale=0.3):
    torch.manual_seed(3)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn_like(p))


def circular_base_flow(circular_features=(0,)):
    torch.manual_seed(0)
    return CircularNeuralSplineFlow(
        2, 16, 4, 1, circular_features=list(circular_features), mask_seed=0
    )


@pytest.fixture
def shift_wrapper():
    base = circular_base_flow()
    perturb(base)
    base.eval()
    return DiscreteGroupMixtureFlowWrapper(
        base,
        2,
        group_action_fn=half_shift,
        group_size=2,
        param_names=["a", "y"],
        prime_space_action=half_shift,
        prime_space_in_domain=half_shift_domain,
        canonical_transform=AngleOfA(),
        circular_parameters=["a"],
    )


def _grid_integral(wrapper, a_lo, a_hi, n=800, y_max=10.0):
    a = torch.linspace(a_lo, a_hi, n + 1)[:-1] + (a_hi - a_lo) / (2 * n)
    y = torch.linspace(-y_max, y_max, n + 1)[:-1] + y_max / n
    grid = torch.stack(torch.meshgrid(a, y, indexing="ij"), -1).reshape(-1, 2)
    with torch.no_grad():
        lp = wrapper.log_prob(grid)
    return float(lp.exp().sum() * (a_hi - a_lo) / n * 2 * y_max / n)


def test_circular_features(shift_wrapper):
    assert shift_wrapper.circular_features == [0]


def test_shift_density_normalised(shift_wrapper):
    assert _grid_integral(shift_wrapper, -1.0, 1.0) == pytest.approx(
        1.0, abs=2e-3
    )


def test_shift_density_continuous_across_fold(shift_wrapper):
    e = 1e-6
    x = torch.tensor([[-e, 0.3], [e, 0.3], [-1 + e, 0.3], [1 - e, 0.3]])
    with torch.no_grad():
        lp = shift_wrapper.log_prob(x)
    assert lp[0].item() == pytest.approx(lp[1].item(), abs=1e-3)
    assert lp[2].item() == pytest.approx(lp[3].item(), abs=1e-3)


def test_shift_sample_and_log_prob(shift_wrapper):
    with torch.no_grad():
        x, log_q = shift_wrapper.sample_and_log_prob(2000)
        np.testing.assert_allclose(
            log_q, shift_wrapper.log_prob(x), atol=1e-8
        )
    # every base draw lands in the fundamental domain: no truncation loss
    assert float(shift_wrapper._log_domain_mass) == pytest.approx(0.0)


def test_shift_standardisation_pinned(shift_wrapper):
    x = torch.stack(
        [torch.rand(500) * 2 - 1, 3.0 + 0.5 * torch.randn(500)], dim=1
    )
    shift_wrapper.update_base_standardisation(x)
    assert float(shift_wrapper._canon_mean[0]) == 0.0
    assert float(shift_wrapper._canon_std[0]) == 1.0
    assert float(shift_wrapper._canon_mean[1]) == pytest.approx(3.0, abs=0.2)


def test_get_model_needs_circular_flow():
    model_cls = make_group_mixture_flow(
        half_shift,
        2,
        ["a", "y"],
        prime_space_action=half_shift,
        prime_space_in_domain=half_shift_domain,
        canonical_transform=AngleOfA(),
        circular_parameters=["a"],
    )
    model = model_cls.__new__(model_cls)
    config = {"n_inputs": 2, "n_neurons": 8, "n_blocks": 2, "n_layers": 1}
    with pytest.raises(ValueError, match="circular base flow"):
        model.get_model(dict(config, ftype="realnvp"))
    wrapper = model.get_model(dict(config, ftype="circular"))
    assert isinstance(wrapper.base_flow, CircularNeuralSplineFlow)
    assert wrapper.base_flow.circular_features == [0]
    assert wrapper.circular_features == [0]


def test_clustered_routing_ignores_circular_dims():
    model_cls = make_clustered_group_mixture_flow(
        n_clusters_max=2,
        group_action_fn=half_shift,
        group_size=2,
        param_names=["a", "y"],
        prime_space_action=half_shift,
        prime_space_in_domain=half_shift_domain,
        canonical_transform=AngleOfA(),
        circular_parameters=["a"],
    )
    model = model_cls.__new__(model_cls)
    wrapper = model.get_model(
        {"n_inputs": 2, "n_neurons": 8, "n_blocks": 2, "n_layers": 1,
         "ftype": "circular"}
    )
    assert isinstance(wrapper, ClusteredGroupMixtureFlowWrapper)
    assert wrapper.circular_features == [0]
    x = torch.stack([torch.rand(10) * 2 - 1, torch.randn(10)], dim=1)
    t = wrapper._fold_to_base(x)
    assert (t[:, 0] == 0).all()
    np.testing.assert_allclose(t[:, 1], x[:, 1])


# -- half-turn with a twist ---------------------------------------------------
# prime coordinates (u, x), u in [0, 1) with period 1; the group element
# h : (u, x) -> (u + 1/2, -x); fundamental domain u in [0, 1/2). The base flow
# models the whole u circle, summed over h.


def half_turn(point_dict, modes, inverse=False):
    flip = modes.to(torch.bool)
    u = torch.remainder(point_dict["u"] + 0.5 * flip.to(point_dict["u"]), 1.0)
    x = torch.where(flip, -point_dict["x"], point_dict["x"])
    return {"u": u, "x": x}


def half_turn_domain(point_dict):
    return point_dict["u"] < 0.5


class AngleOfU:
    """``u in [0, 1) -> t in [-pi, pi)`` over the whole circle."""

    def forward(self, canon):
        t = canon.clone()
        t[:, 0] = _wrap(TWO_PI * canon[:, 0] - math.pi)
        return t, canon.new_full((canon.shape[0],), math.log(TWO_PI))

    def inverse(self, t):
        canon = t.clone()
        canon[:, 0] = torch.remainder(
            (_wrap(t[:, 0]) + math.pi) / TWO_PI, 1.0
        )
        return canon, t.new_full((t.shape[0],), -math.log(TWO_PI))


class HalfTurnSymmetry:
    """``h`` in the base frame: ``t_u -> t_u + pi``, ``x -> -x``."""

    @staticmethod
    def _h(t):
        out = t.clone()
        out[:, 0] = _wrap(t[:, 0] + math.pi)
        out[:, 1] = -t[:, 1]
        return out

    def images(self, t):
        return [self._h(t)]

    def fold(self, t):
        # canonical part: t_u in [-pi, 0) (u in [0, 1/2))
        return torch.where((t[:, 0] >= 0)[:, None], self._h(t), t)


@pytest.fixture
def twist_wrapper():
    base = circular_base_flow()
    perturb(base)
    base.eval()
    return DiscreteGroupMixtureFlowWrapper(
        base,
        2,
        group_action_fn=half_turn,
        group_size=2,
        param_names=["u", "x"],
        prime_space_action=half_turn,
        prime_space_in_domain=half_turn_domain,
        canonical_transform=AngleOfU(),
        circular_parameters=["u"],
        base_symmetry=HalfTurnSymmetry(),
    )


def test_twist_density_normalised(twist_wrapper):
    assert _grid_integral(twist_wrapper, 0.0, 1.0) == pytest.approx(
        1.0, abs=2e-3
    )


def test_twist_density_continuous_across_seams(twist_wrapper):
    e = 1e-6
    x = torch.tensor(
        [[0.5 - e, 0.7], [0.5 + e, 0.7], [e, -0.4], [1 - e, -0.4]]
    )
    with torch.no_grad():
        lp = twist_wrapper.log_prob(x)
    assert lp[0].item() == pytest.approx(lp[1].item(), abs=1e-3)
    assert lp[2].item() == pytest.approx(lp[3].item(), abs=1e-3)


def test_twist_sample_and_log_prob(twist_wrapper):
    with torch.no_grad():
        x, log_q = twist_wrapper.sample_and_log_prob(2000)
        np.testing.assert_allclose(
            log_q, twist_wrapper.log_prob(x), atol=1e-8
        )
        z = twist_wrapper.sample_latent_distribution(500)
        x, log_j = twist_wrapper.inverse(z)
        log_q = twist_wrapper.base_distribution_log_prob(z) - log_j
        np.testing.assert_allclose(
            log_q, twist_wrapper.log_prob(x), atol=1e-8
        )
    assert float(twist_wrapper._log_domain_mass) == pytest.approx(0.0)


def test_twist_draws_fill_both_halves_with_weights(twist_wrapper):
    with torch.no_grad():
        twist_wrapper.weights.copy_(torch.tensor([0.8, 0.2]))
        x, _ = twist_wrapper.sample_and_log_prob(20000)
    frac = float((x[:, 0] < 0.5).double().mean())
    assert frac == pytest.approx(0.8, abs=0.02)


def test_base_symmetry_rejects_reflection():
    with pytest.raises(ValueError, match="reflect_parameters"):
        DiscreteGroupMixtureFlowWrapper(
            circular_base_flow(),
            2,
            group_action_fn=half_turn,
            group_size=2,
            param_names=["u", "x"],
            prime_space_action=half_turn,
            prime_space_in_domain=half_turn_domain,
            reflect_parameters=["x"],
            base_symmetry=HalfTurnSymmetry(),
        )


def test_rebinding_to_another_order_is_refused(shift_wrapper):
    """The base flow fixed its circular inputs when it was built."""
    shift_wrapper.set_param_names(["a", "y"])
    with pytest.raises(RuntimeError, match="parameter order"):
        shift_wrapper.set_param_names(["y", "a"])
