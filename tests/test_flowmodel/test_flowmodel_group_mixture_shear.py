"""
The curvature shear of the group mixture's standardised frame.
"""

import math

import numpy as np
import pytest
import torch

from nessai.flowmodel.group_mixture import DiscreteGroupMixtureFlowWrapper
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


# prime coordinates (a, s, y): a an angle in [-1, 1) shifted by half a period
# by the group (fundamental domain a in [-1, 0)), s and y unchanged.


def half_shift(point_dict, modes, inverse=False):
    a = torch.remainder(point_dict["a"] + 1.0 + modes.to(point_dict["a"]), 2.0)
    return {"a": a - 1.0, "s": point_dict["s"], "y": point_dict["y"]}


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


def ridge(n=4000, seed=1, width=0.05):
    """A U-shaped ridge ``y = 2 + 3 s**2`` mirrored about ``s = 0``, with the
    angle uniform."""
    g = torch.Generator().manual_seed(seed)
    s = torch.rand(n, generator=g) * 2 - 1
    y = 2.0 + 3.0 * s**2 + width * torch.randn(n, generator=g)
    a = torch.rand(n, generator=g) * 2 - 1
    return torch.stack([a, s, y], dim=1)


@pytest.fixture
def shear_wrapper():
    torch.manual_seed(0)
    base = CircularNeuralSplineFlow(
        3, 16, 4, 1, circular_features=[0], mask_seed=0
    )
    torch.manual_seed(3)
    with torch.no_grad():
        for p in base.parameters():
            p.add_(0.3 * torch.randn_like(p))
    base.eval()
    return DiscreteGroupMixtureFlowWrapper(
        base,
        3,
        group_action_fn=half_shift,
        group_size=2,
        param_names=["a", "s", "y"],
        prime_space_action=half_shift,
        prime_space_in_domain=half_shift_domain,
        canonical_transform=AngleOfA(),
        circular_parameters=["a"],
        curvature_shear="s",
    )


def test_shear_straightens_ridge(shear_wrapper):
    x = ridge()
    shear_wrapper.update_base_standardisation(x)
    coef = shear_wrapper._shear_coef
    # neither the circular angle nor s itself is sheared
    assert torch.all(coef[:, :2] == 0)
    t, _ = shear_wrapper._canon_to_base(x)
    sheared = shear_wrapper._shear(t)
    # the U is gone: y is the noise alone, uncorrelated with s**2
    assert float(sheared[:, 2].std()) == pytest.approx(0.05, rel=0.1)
    assert abs(np.corrcoef(t[:, 1] ** 2, sheared[:, 2])[0, 1]) < 0.05
    # and the standardisation is that of the sheared band
    u = shear_wrapper.canonical_to_standardised(x)
    assert float(u[:, 2].std()) == pytest.approx(1.0, rel=0.05)


def test_shear_round_trip(shear_wrapper):
    x = ridge()
    shear_wrapper.update_base_standardisation(x)
    t, _ = shear_wrapper._canon_to_base(x)
    back = shear_wrapper._destandardise(shear_wrapper._standardise(t))
    np.testing.assert_allclose(back, t, atol=1e-12)


def test_shear_is_unit_jacobian(shear_wrapper):
    """The sheared density at ``x`` is the unsheared one (same base flow and
    standardisation) at the sheared point: the shear only moves the points,
    so the density stays normalised."""
    x = ridge()
    shear_wrapper.update_base_standardisation(x)
    plain = DiscreteGroupMixtureFlowWrapper(
        shear_wrapper.base_flow,
        3,
        group_action_fn=half_shift,
        group_size=2,
        param_names=["a", "s", "y"],
        prime_space_action=half_shift,
        prime_space_in_domain=half_shift_domain,
        canonical_transform=AngleOfA(),
        circular_parameters=["a"],
    )
    plain.load_state_dict(shear_wrapper.state_dict())
    plain._shear_coef.zero_()
    moved = x.clone()
    moved[:, 2] = shear_wrapper._shear(x)[:, 2]
    with torch.no_grad():
        np.testing.assert_allclose(
            shear_wrapper.log_prob(x), plain.log_prob(moved), atol=1e-10
        )


def test_shear_sample_and_log_prob(shear_wrapper):
    shear_wrapper.update_base_standardisation(ridge())
    with torch.no_grad():
        x, log_q = shear_wrapper.sample_and_log_prob(2000)
        np.testing.assert_allclose(
            log_q, shear_wrapper.log_prob(x), atol=1e-8
        )


def _wrapper(powers=None):
    torch.manual_seed(0)
    base = CircularNeuralSplineFlow(
        3, 16, 4, 1, circular_features=[0], mask_seed=0
    )
    base.eval()
    return DiscreteGroupMixtureFlowWrapper(
        base,
        3,
        group_action_fn=half_shift,
        group_size=2,
        param_names=["a", "s", "y"],
        prime_space_action=half_shift,
        prime_space_in_domain=half_shift_domain,
        canonical_transform=AngleOfA(),
        circular_parameters=["a"],
        curvature_shear="s",
        curvature_shear_powers=powers,
    )


def skewed_ridge(n=4000, seed=2, width=0.05, layout="mirrored"):
    """A ridge with odd terms, ``y = 2 + 4 s + 3 s**2 - 2 s**3``: on ``s <=
    0`` with its mirror copies about ``s = 0`` (``mirrored``), alone
    (``one_sided``), or on ``-1 <= s <= 1`` (``two_sided``: a boundary not
    inverted, the parameter rescaled to [-1, 1])."""
    g = torch.Generator().manual_seed(seed)
    s = -torch.rand(n, generator=g)
    if layout == "two_sided":
        s = 2 * s + 1
    noise = width * torch.randn(n, generator=g)
    y = 2.0 + 4.0 * s + 3.0 * s**2 - 2.0 * s**3 + noise
    a = torch.rand(n, generator=g) * 2 - 1
    x = torch.stack([a, s, y], dim=1)
    if layout != "mirrored":
        return x
    mirror = x.clone()
    mirror[:, 1] = -s
    return torch.cat([x, mirror])


def test_default_powers():
    w = _wrapper()
    assert w.shear_powers == (2, 4)
    assert w._shear_coef.shape == (2, 3)
    assert w._shear_exponents.tolist() == [2, 4]


@pytest.mark.parametrize("layout", ["mirrored", "one_sided", "two_sided"])
def test_odd_powers_straighten_a_skewed_ridge(layout):
    """Even powers leave the odd terms; with the odd powers (of ``|s|`` on
    mirrored points, else of ``s``) the ridge is straight."""
    x = skewed_ridge(layout=layout)
    even, odd = _wrapper(), _wrapper([1, 2, 3, 4])
    for w in (even, odd):
        w.update_base_standardisation(x)
    assert bool(odd._shear_abs) == (layout == "mirrored")
    spread = {}
    for name, w in (("even", even), ("odd", odd)):
        t, _ = w._canon_to_base(x)
        spread[name] = float(w._shear(t)[:, 2].std())
    assert spread["odd"] == pytest.approx(0.05, rel=0.1)
    assert spread["even"] > 2 * spread["odd"]


def test_even_powers_ignore_the_mirroring():
    w = _wrapper()
    w.update_base_standardisation(skewed_ridge())
    assert not bool(w._shear_abs)


def test_change_of_mirroring_refits():
    """Fitted on mirrored points (``|s|``), then on unmirrored ones (``s``):
    the coefficients of the other basis are refitted, not averaged."""
    w = _wrapper([1, 2, 3, 4])
    w.update_base_standardisation(skewed_ridge())
    x = skewed_ridge(seed=3, layout="two_sided")
    w.update_base_standardisation(x)
    fresh = _wrapper([1, 2, 3, 4])
    fresh.update_base_standardisation(x)
    assert not bool(w._shear_abs)
    np.testing.assert_allclose(w._shear_coef, fresh._shear_coef)
    # the same mirroring again: averaged
    w.update_base_standardisation(skewed_ridge(seed=4, layout="two_sided"))
    assert not torch.allclose(w._shear_coef, fresh._shear_coef)


def test_odd_powers_commute_with_the_mirror():
    w = _wrapper([1, 2, 3, 4])
    w.update_base_standardisation(skewed_ridge())
    t, _ = w._canon_to_base(skewed_ridge(n=100, seed=5, layout="one_sided"))
    m = t.clone()
    m[:, 1] = -t[:, 1]
    np.testing.assert_allclose(w._shear_offset(m), w._shear_offset(t))


def test_odd_powers_sample_and_log_prob():
    w = _wrapper([1, 2, 3, 4])
    w.update_base_standardisation(skewed_ridge())
    with torch.no_grad():
        x, log_q = w.sample_and_log_prob(2000)
        np.testing.assert_allclose(log_q, w.log_prob(x), atol=1e-8)


@pytest.mark.parametrize("named", [True, False])
def test_checkpoint_with_other_powers(named):
    """A checkpoint of the default powers (naming them, or written before
    they were stored) loads into a wrapper with more: the shared powers'
    coefficients are kept, the new ones start at zero, and the next
    standardisation refits them rather than averaging with the loaded
    ones."""
    old = _wrapper()
    old.update_base_standardisation(skewed_ridge())
    state = old.state_dict()
    if not named:
        del state["_shear_exponents"], state["_shear_abs"]
    new = _wrapper([1, 2, 3, 4])
    new.load_state_dict(state)
    assert new._shear_exponents.tolist() == [1, 2, 3, 4]
    np.testing.assert_array_equal(new._shear_coef[[1, 3]], old._shear_coef)
    assert torch.all(new._shear_coef[[0, 2]] == 0)
    x = skewed_ridge(seed=7)
    new.update_base_standardisation(x)
    fresh = _wrapper([1, 2, 3, 4])
    fresh.update_base_standardisation(x)
    np.testing.assert_allclose(new._shear_coef, fresh._shear_coef)
    # and back: the odd rows dropped
    back = _wrapper()
    back.load_state_dict(new.state_dict())
    np.testing.assert_array_equal(back._shear_coef, new._shear_coef[[1, 3]])


def test_checkpoint_with_same_powers_is_averaged():
    a = _wrapper([1, 2, 3, 4])
    a.update_base_standardisation(skewed_ridge())
    b = _wrapper([1, 2, 3, 4])
    b.load_state_dict(a.state_dict())
    np.testing.assert_array_equal(b._shear_coef, a._shear_coef)
    assert bool(b._shear_abs)
    assert not b._shear_refit
    t, _ = a._canon_to_base(skewed_ridge(seed=6))
    np.testing.assert_array_equal(b._shear_offset(t), a._shear_offset(t))


@pytest.mark.parametrize("powers", [[], [0, 2], [1.5], ["2"]])
def test_invalid_powers(powers):
    with pytest.raises(ValueError, match="positive integers"):
        _wrapper(powers)


def test_shear_needs_known_parameter():
    base = CircularNeuralSplineFlow(
        3, 16, 4, 1, circular_features=[0], mask_seed=0
    )
    with pytest.raises(ValueError, match="cannot be circular"):
        DiscreteGroupMixtureFlowWrapper(
            base,
            3,
            group_action_fn=half_shift,
            group_size=2,
            param_names=["a", "s", "y"],
            prime_space_action=half_shift,
            prime_space_in_domain=half_shift_domain,
            canonical_transform=AngleOfA(),
            circular_parameters=["a"],
            curvature_shear="a",
        )
