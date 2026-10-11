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


# -- Möbius standardisation of the circular dims ------------------------------


def test_mobius_inverse_and_jacobian():
    from nessai.flowmodel.group_mixture import (
        mobius_destandardise,
        mobius_standardise,
    )

    psi = torch.linspace(-math.pi, math.pi, 1001)[:, None]
    scale = torch.tensor([0.03])
    phi, log_det = mobius_standardise(psi, scale)
    assert phi.min() >= -math.pi and phi.max() <= math.pi
    back, log_det_inv = mobius_destandardise(phi, scale)
    np.testing.assert_allclose(back, psi, atol=1e-10)
    np.testing.assert_allclose(log_det_inv, -log_det, atol=1e-8)
    h = 1e-6
    fd = (
        mobius_standardise(psi + h, scale)[0]
        - mobius_standardise(psi - h, scale)[0]
    ) / (2 * h)
    inner = psi.abs() < math.pi - 1e-3
    np.testing.assert_allclose(
        log_det[inner], torch.log(fd[inner]), rtol=1e-5, atol=1e-6
    )
    # the identity at scale 1
    phi, log_det = mobius_standardise(psi, torch.tensor([1.0]))
    np.testing.assert_allclose(phi, psi, atol=1e-12)
    np.testing.assert_allclose(log_det, 0.0, atol=1e-12)


def test_fit_mobius_scale():
    from nessai.flowmodel.group_mixture import (
        fit_mobius_scale,
        mobius_standardise,
    )

    torch.manual_seed(1)
    sigma = torch.tensor([0.01, 0.3, 1.0])
    psi = _wrap(sigma * torch.randn(20000, 3))
    scale = fit_mobius_scale(psi)
    phi, _ = mobius_standardise(psi, scale)
    np.testing.assert_allclose(torch.cos(phi).mean(0), 0.0, atol=1e-6)
    # scale-equivariant for narrow peaks
    assert float(scale[0] / sigma[0]) == pytest.approx(
        float(scale[1] / sigma[1]), rel=0.05
    )
    # a sample spread over the circle stays (close to) the identity
    flat = torch.rand(5000, 1) * TWO_PI - math.pi
    assert float(fit_mobius_scale(flat)) == pytest.approx(1.0, abs=0.1)
    # one with zero mean cosine about its mean direction keeps it exactly
    pair = torch.tensor([[-math.pi / 2], [math.pi / 2]])
    assert float(fit_mobius_scale(pair)) == pytest.approx(1.0)


def _standardised(wrapper, a_centre, a_width, y_mean=0.4, n=4000):
    """Turn on the Möbius standardisation, fitted to a narrow peak in ``a``."""
    torch.manual_seed(5)
    wrapper.circular_standardisation = True
    a = torch.remainder(a_centre + a_width * torch.randn(n) + 1.0, 2.0) - 1.0
    x = torch.stack([a, y_mean + 0.5 * torch.randn(n)], dim=1)
    wrapper.update_base_standardisation(x)
    return wrapper


def test_standardised_shift_fit(shift_wrapper):
    # a ~ -0.5 is the middle of the domain [-1, 0): t = 2 pi (a + 1) - pi = 0
    _standardised(shift_wrapper, -0.5, 0.01)
    mu, scale = shift_wrapper._canon_mean[0], shift_wrapper._canon_std[0]
    assert float(mu) == pytest.approx(0.0, abs=0.01)
    # t has width 2 pi * 0.01; a Gaussian peak gets scale ~0.3 sigma (the
    # median of |psi| / 2)
    assert float(scale) / (TWO_PI * 0.01) == pytest.approx(0.3, abs=0.05)
    assert float(shift_wrapper._canon_mean[1]) == pytest.approx(0.4, abs=0.05)


@pytest.mark.parametrize("centre", [-0.5, -0.02])
def test_standardised_shift_density(shift_wrapper, centre):
    """Normalised, consistent with sampling and the latent round trip, also
    with the peak next to the fold (the base-frame seam)."""
    _standardised(shift_wrapper, centre, 0.05)
    assert _grid_integral(shift_wrapper, -1.0, 1.0, n=1200) == pytest.approx(
        1.0, abs=3e-3
    )
    with torch.no_grad():
        x, log_q = shift_wrapper.sample_and_log_prob(2000)
        np.testing.assert_allclose(
            log_q, shift_wrapper.log_prob(x), atol=1e-8
        )
        z = shift_wrapper.sample_latent_distribution(500)
        x, log_j = shift_wrapper.inverse(z)
        log_q = shift_wrapper.base_distribution_log_prob(z) - log_j
        np.testing.assert_allclose(
            log_q, shift_wrapper.log_prob(x), atol=1e-8
        )


def test_standardised_twist_density(twist_wrapper):
    _standardised(twist_wrapper, 0.2, 0.05)
    assert _grid_integral(twist_wrapper, 0.0, 1.0, n=1200) == pytest.approx(
        1.0, abs=3e-3
    )
    with torch.no_grad():
        x, log_q = twist_wrapper.sample_and_log_prob(2000)
        np.testing.assert_allclose(
            log_q, twist_wrapper.log_prob(x), atol=1e-8
        )


def test_standardised_ema_follows_the_circle(shift_wrapper):
    """The location is averaged along the circle, not across it."""
    # t = 2 pi (a + 1) - pi: a = -0.99 / -0.01 sit either side of t = -/+ pi
    _standardised(shift_wrapper, -0.99, 0.002)
    first = float(shift_wrapper._canon_mean[0])
    _standardised(shift_wrapper, -0.01, 0.002)
    mu = float(shift_wrapper._canon_mean[0])
    assert abs(first) > 3.0 and abs(mu) > 3.0


def test_standardisation_off_by_default(shift_wrapper):
    assert not shift_wrapper.circular_standardisation
    model_cls = make_group_mixture_flow(
        half_shift,
        2,
        ["a", "y"],
        prime_space_action=half_shift,
        prime_space_in_domain=half_shift_domain,
        canonical_transform=AngleOfA(),
        circular_parameters=["a"],
        circular_standardisation=True,
    )
    model = model_cls.__new__(model_cls)
    wrapper = model.get_model(
        {"n_inputs": 2, "n_neurons": 8, "n_blocks": 2, "n_layers": 1,
         "ftype": "circular"}
    )
    assert wrapper.circular_standardisation
