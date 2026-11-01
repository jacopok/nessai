"""
The per-mode shift of the group mixture's standardisation centre.
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


OFFSET, WIDTH = 0.5, 0.05


def broken_images(n=4000, seed=1):
    """A blob and its image under the group, the image displaced by
    ``OFFSET`` in ``y``: an approximate symmetry. Points with ``a < 0`` (the
    identity's) at ``y ~ 2``, those with ``a >= 0`` at ``y ~ 2 + OFFSET``."""
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(n, generator=g) * 2 - 1
    s = 0.1 * torch.randn(n, generator=g)
    y = 2.0 + OFFSET * (a >= 0) + WIDTH * torch.randn(n, generator=g)
    return torch.stack([a, s, y], dim=1)


def make_wrapper(mode_shift=("y",), **kwargs):
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
        mode_shift=list(mode_shift) if mode_shift else None,
        **kwargs,
    )


def test_mode_shift_stacks_the_images():
    x = broken_images()
    wrapper = make_wrapper()
    wrapper.update_base_standardisation(x)
    shift = wrapper._mode_shift
    # only y is shifted, by the images' offset, about the pooled centre
    assert torch.all(shift[:, :2] == 0)
    assert float(shift[1, 2] - shift[0, 2]) == pytest.approx(OFFSET, abs=0.01)
    # the aligned points are one blob: the standardisation is its width
    assert float(wrapper._canon_std[2]) == pytest.approx(WIDTH, rel=0.1)
    assigned, pre, _ = wrapper._assign_branch(x)
    canon = pre[assigned, torch.arange(len(x))]
    t, _ = wrapper._canon_to_base(canon)
    u = wrapper._standardise(t, assigned)
    assert float(u[:, 2].std()) == pytest.approx(1.0, rel=0.05)
    assert abs(float(u[assigned == 0, 2].mean())) < 0.1
    assert abs(float(u[assigned == 1, 2].mean())) < 0.1


def test_without_mode_shift_the_images_stay_apart():
    wrapper = make_wrapper(mode_shift=None)
    wrapper.update_base_standardisation(broken_images())
    assert torch.all(wrapper._mode_shift == 0)
    assert float(wrapper._canon_std[2]) > 4 * WIDTH


def test_mode_shift_round_trip():
    x = broken_images()
    wrapper = make_wrapper()
    wrapper.update_base_standardisation(x)
    assigned, pre, _ = wrapper._assign_branch(x)
    t, _ = wrapper._canon_to_base(pre[assigned, torch.arange(len(x))])
    back = wrapper._destandardise(wrapper._standardise(t, assigned), assigned)
    np.testing.assert_allclose(back, t, atol=1e-12)


def test_mode_shift_on_a_circular_parameter_wraps():
    """A shift of the angle itself: the images are offset by a quarter of
    the quotient circle, across its seam."""
    x = broken_images()
    x[:, 2] = 2.0 + WIDTH * torch.randn(len(x))
    # element 1's points rotated by +0.2 in a (0.4 pi of the quotient angle)
    second = x[:, 0] >= 0
    x[second, 0] = torch.remainder(x[second, 0] + 1.2, 2.0) - 1.0
    wrapper = make_wrapper(mode_shift=("a",))
    wrapper.update_base_standardisation(x)
    d = float(_wrap(wrapper._mode_shift[1, 0] - wrapper._mode_shift[0, 0]))
    assert d == pytest.approx(0.4 * math.pi, abs=0.05)
    assigned, pre, _ = wrapper._assign_branch(x)
    t, _ = wrapper._canon_to_base(pre[assigned, torch.arange(len(x))])
    back = wrapper._destandardise(wrapper._standardise(t, assigned), assigned)
    np.testing.assert_allclose(_wrap(back[:, 0] - t[:, 0]), 0, atol=1e-10)


def test_mode_shift_sample_and_log_prob():
    """The draws' density is the model's: each element's shift is applied
    where its points are generated and where they are scored."""
    wrapper = make_wrapper()
    wrapper.update_base_standardisation(broken_images())
    with torch.no_grad():
        x, log_q = wrapper.sample_and_log_prob(2000)
        np.testing.assert_allclose(log_q, wrapper.log_prob(x), atol=1e-8)
    # the draws follow the displaced images: element 1's are OFFSET higher
    second = x[:, 0] >= 0
    gap = float(x[second, 2].mean() - x[~second, 2].mean())
    assert gap == pytest.approx(
        float(wrapper._mode_shift[1, 2] - wrapper._mode_shift[0, 2]), abs=0.05
    )


def test_mode_shift_is_unit_jacobian():
    """The shifted density at ``x`` is the unshifted one (same base flow and
    standardisation) at the point moved back by its element's shift."""
    x = broken_images()
    wrapper = make_wrapper()
    wrapper.update_base_standardisation(x)
    plain = make_wrapper(mode_shift=None)
    plain.load_state_dict(wrapper.state_dict())
    plain._mode_shift.zero_()
    # points of one element only (a < 0, the identity's), so each has a
    # single non-negligible branch
    x0 = x[x[:, 0] < 0]
    moved = x0.clone()
    moved[:, 2] = x0[:, 2] - wrapper._mode_shift[0, 2]
    with torch.no_grad():
        lp = wrapper._branch_log_probs(x0)[0]
        lp_plain = plain._branch_log_probs(moved)[0]
    np.testing.assert_allclose(lp, lp_plain, atol=1e-10)


def test_mode_shift_resumes_from_an_older_checkpoint():
    wrapper = make_wrapper()
    state = wrapper.state_dict()
    del state["_mode_shift"]
    wrapper.load_state_dict(state)
    assert torch.all(wrapper._mode_shift == 0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(mode_shift=("z",)), "not in the parameters"),
        (dict(mode_shift=("s",), curvature_shear="s"), "curvature shear"),
    ],
)
def test_mode_shift_invalid(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_wrapper(**kwargs)
