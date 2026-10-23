# -*- coding: utf-8 -*-
"""End-to-end test of the populate method"""

import numpy as np
import pytest
import torch

from nessai.livepoint import numpy_array_to_live_points
from nessai.proposal import FlowProposal
from nessai.utils.testing import assert_structured_arrays_equal

torch.set_num_threads(1)


@pytest.fixture()
def flow_config():
    return dict(
        n_neurons=1,
        n_blocks=1,
        n_layers=1,
    )


@pytest.mark.parametrize("expansion_fraction", [0.0, 1.0, None])
@pytest.mark.parametrize("check_acceptance", [False, True])
@pytest.mark.integration_test
@pytest.mark.timeout(30)
def test_flowproposal_populate(
    tmp_path,
    model,
    flow_config,
    expansion_fraction,
    check_acceptance,
):
    """
    Test the populate method in the FlowProposal class with a range of
    parameters
    """
    output = tmp_path / "flowproposal"
    output.mkdir()
    n_draw = 10
    fp = FlowProposal(
        model,
        output=output,
        flow_config=flow_config,
        plot=False,
        poolsize=10,
        fixed_radius=1.0,
        radius_mode="fixed",
        expansion_fraction=expansion_fraction,
        check_acceptance=check_acceptance,
        max_radius=1.0,
    )

    fp.initialise()
    worst = numpy_array_to_live_points(0.01 * np.ones(fp.dims), fp.parameters)
    fp.populate(worst, n_samples=n_draw)

    assert fp.x.size == n_draw


@pytest.mark.parametrize(
    "truncation_methods",
    [[], ["likelihood_threshold"]],
)
@pytest.mark.integration_test
@pytest.mark.timeout(30)
def test_flowproposal_populate_edge_cases(
    tmp_path, model, flow_config, truncation_methods
):
    """Tests some less common settings for flowproposal"""
    output = tmp_path / "flowproposal"
    output.mkdir()
    n_draw = 2
    fp = FlowProposal(
        model,
        output=output,
        flow_config=flow_config,
        plot=False,
        poolsize=10,
        expansion_fraction=None,
        fixed_radius=0.1,
        radius_mode="fixed",
        truncation_methods=truncation_methods,
        fallback_reparameterisation=None,
    )

    fp.initialise()
    assert fp.parameters == fp.prime_parameters
    worst = numpy_array_to_live_points(0.01 * np.ones(fp.dims), fp.parameters)
    fp.populate(worst, n_samples=n_draw)

    assert fp.x.size == n_draw


@pytest.mark.integration_test
def test_flowproposal_populate_min_log_q_requires_training_data(
    tmp_path, model, flow_config
):
    """min_log_q truncation should fail without training data."""
    output = tmp_path / "flowproposal"
    output.mkdir()
    fp = FlowProposal(
        model,
        output=output,
        flow_config=flow_config,
        plot=False,
        poolsize=10,
        expansion_fraction=None,
        fixed_radius=0.1,
        radius_mode="fixed",
        truncation_methods=["min_log_q"],
        fallback_reparameterisation=None,
    )

    fp.initialise()
    worst = numpy_array_to_live_points(0.01 * np.ones(fp.dims), fp.parameters)
    with pytest.raises(
        RuntimeError, match="min_log_q truncation requires training_data"
    ):
        fp.populate(worst, n_samples=2)


@pytest.mark.parametrize("plot", [False, True])
@pytest.mark.integration_test
def test_training(tmpdir, model, plot):
    """Integration test to test training the flow with and without plotting."""
    output = str(tmpdir.mkdir("test_train"))
    config = dict(n_layers=1)
    training_config = dict(max_epochs=10)
    fp = FlowProposal(
        model,
        output=output,
        plot="min",
        poolsize=100,
        flow_config=config,
        training_config=training_config,
    )

    fp.initialise()

    x = model.new_point(500)
    fp.train(x, plot=plot)

    assert fp.training_count == 1
    assert fp.populated is False


@pytest.mark.parametrize("check_acceptance", [False, True])
@pytest.mark.integration_test
@pytest.mark.timeout(30)
def test_constant_volume_mode(
    tmpdir,
    model,
    flow_config,
    check_acceptance,
):
    """Integration test for constant volume mode.

    With q=0.8647 should get a radius of ~2.
    """
    output = str(tmpdir.mkdir("flowproposal"))
    expected_radius = 2.0
    fp = FlowProposal(
        model,
        output=output,
        flow_config=flow_config,
        plot=False,
        poolsize=10,
        constant_volume_mode=True,
        volume_fraction=0.8647,
        check_acceptance=check_acceptance,
    )
    fp.initialise()
    worst = numpy_array_to_live_points(0.5 * np.ones(fp.dims), fp.parameters)
    fp.populate(worst, n_samples=10)
    assert fp.x.size == 10

    rule = fp.get_truncation_rule("latent_radius")
    np.testing.assert_approx_equal(rule.radius, expected_radius, 4)
    np.testing.assert_approx_equal(rule.fixed_radius, expected_radius, 4)


@pytest.mark.parametrize(
    "reparameterisations",
    ["rescaletobounds", "inversion", "inversion-duplicate", "zscore"],
)
@pytest.mark.integration_test
def test_verify_rescaling_integration(tmp_path, model, reparameterisations):
    """Assert verify rescaling passes."""
    output = tmp_path / "test"
    output.mkdir()

    fp = FlowProposal(
        model,
        output=output,
        poolsize=10,
        reparameterisations=reparameterisations,
    )
    fp.set_rescaling()
    fp.verify_rescaling()


@pytest.mark.integration_test
def test_rescaling_integration_with_zscore(tmp_path, model):
    """Assert zscore is configured correctly"""
    output = tmp_path / "test"
    output.mkdir()

    fp = FlowProposal(
        model,
        output=output,
        poolsize=10,
    )
    fp.set_rescaling()

    n = 10
    x = model.new_point(n)

    x_prime, log_j = fp.rescale(x)

    x_recon, log_j_inv = fp.inverse_rescale(x_prime)

    assert len(log_j) == n
    np.testing.assert_array_almost_equal(log_j, -log_j_inv, decimal=15)

    assert_structured_arrays_equal(x_recon, x, atol=1e-15)


@pytest.mark.integration_test
def test_rescaling_integration_with_rescaletobounds(tmp_path, model):
    """Assert rescaletobounds is configured correctly"""
    output = tmp_path / "test"
    output.mkdir()

    fp = FlowProposal(
        model,
        output=output,
        poolsize=10,
        reparameterisations="rescaletobounds",
    )
    fp.set_rescaling()

    n = 10
    x = model.new_point(n)

    x_prime, log_j = fp.rescale(x)

    x_recon, log_j_inv = fp.inverse_rescale(x_prime)

    x_prime_expected = x_prime.copy()
    for name in model.names:
        x_prime_expected[name + "_prime"] = (
            2
            * (
                (x[name] - model.bounds[name][0])
                / (model.bounds[name][1] - model.bounds[name][0])
            )
            - 1
        )
    expected_log_j = np.sum(
        [np.log(2 / np.ptp(model.bounds[n])) for n in model.names]
    )
    assert len(log_j) == n
    np.testing.assert_allclose(log_j, expected_log_j)
    np.testing.assert_array_almost_equal(log_j, -log_j_inv, decimal=15)

    assert_structured_arrays_equal(x_prime, x_prime_expected, atol=1e-15)
    assert_structured_arrays_equal(x_recon, x, atol=1e-15)


@pytest.mark.integration_test
def test_rescaling_integration_no_rescaling(tmp_path, model):
    """Assert setting rescale_parameters=False does not rescale any
    parameters.
    """
    output = tmp_path / "test"
    output.mkdir()

    fp = FlowProposal(
        model,
        output=output,
        poolsize=10,
        fallback_reparameterisation=None,
    )
    fp.set_rescaling()

    n = 10
    x = model.new_point(n)

    x_prime, log_j = fp.rescale(x)

    x_recon, log_j_inv = fp.inverse_rescale(x_prime)

    assert len(log_j) == n
    assert (log_j == 0.0).all()
    np.testing.assert_array_equal(log_j, -log_j_inv)

    assert_structured_arrays_equal(x_prime, x)
    assert_structured_arrays_equal(x_recon, x_prime)


@pytest.mark.parametrize("sum_mirror_density", [False, True])
@pytest.mark.integration_test
@pytest.mark.timeout(60)
def test_sum_mirror_density(tmp_path, model, flow_config, sum_mirror_density):
    """With boundary inversion by duplication a draw is folded back from any
    side: with ``sum_mirror_density`` it is scored with the flow's density
    summed over the sides (as are the live points the truncation rules use),
    otherwise with the side it was drawn on."""
    from scipy.special import logsumexp

    from nessai.proposal.flowproposal.truncation import _live_log_q

    output = tmp_path / "flowproposal"
    output.mkdir()
    fp = FlowProposal(
        model,
        output=output,
        flow_config=flow_config,
        plot=False,
        poolsize=100,
        reparameterisations="inversion-duplicate",
        sum_mirror_density=sum_mirror_density,
    )
    fp.initialise()
    live = model.new_point(100)
    fp.train(live, plot=False)

    z = fp.sample_latent_distribution(200)
    x, log_q = fp.backward_pass(z)
    sides = fp.mirror_log_q(x)
    assert sides.shape[0] > 1
    if sum_mirror_density:
        np.testing.assert_allclose(log_q, logsumexp(sides, axis=0), atol=1e-5)
    else:
        # the density of the side the draw came from
        assert np.all(np.min(np.abs(sides - log_q), axis=0) < 1e-3)
        assert np.all(log_q < logsumexp(sides, axis=0))

    live_points, live_log_q = _live_log_q(fp)
    expected = (
        logsumexp(fp.mirror_log_q(live_points), axis=0)
        if sum_mirror_density
        else fp.mirror_log_q(live_points).ravel()
    )
    np.testing.assert_allclose(live_log_q, expected)


@pytest.mark.parametrize("sum_mirror_density", [False, True])
@pytest.mark.integration_test
def test_prior_relative_threshold_with_auxiliary_radius(
    tmp_path, rng, sum_mirror_density
):
    """Live points do not carry the auxiliary radius of an angle (e.g.
    GWFlowProposal's sky and phase): the prior-relative rules draw it once,
    so that the prior of the live points can be evaluated, and the flow
    density and the mirror copies of the boundary inversion see the same
    radius."""
    from scipy.special import logsumexp
    from scipy.stats import norm

    from nessai.model import Model
    from nessai.proposal.flowproposal.truncation import _live_log_q

    class AngleModel(Model):
        def __init__(self):
            self.names = ["x", "phi"]
            self.bounds = {"x": [-5, 5], "phi": [0, 2 * np.pi]}

        def log_prior(self, x):
            log_p = np.log(self.in_bounds(x), dtype="float")
            return log_p - np.log(10 * 2 * np.pi)

        def log_likelihood(self, x):
            return norm.logpdf(x["x"]) + 2 * np.cos(x["phi"])

    model = AngleModel()
    model.set_rng(rng)
    reparameterisations = {"phi": "angle-2pi"}
    if sum_mirror_density:
        reparameterisations["x"] = "inversion-duplicate"
    output = tmp_path / "flowproposal"
    output.mkdir()
    fp = FlowProposal(
        model,
        output=output,
        flow_config=dict(n_neurons=2, n_blocks=1, n_layers=1),
        plot=False,
        poolsize=100,
        reparameterisations=reparameterisations,
        truncation_methods=["log_proposal_threshold"],
        truncation_kwargs={
            "log_proposal_threshold": {
                "quantile": 0.1,
                "prior_relative": True,
                "mode": "clip",
            }
        },
        sum_mirror_density=sum_mirror_density,
    )
    fp.initialise()
    assert "phi_radial" in fp.x_dtype.names
    live = model.new_point(200)
    live["logL"] = model.batch_evaluate_log_likelihood(live)
    fp.train(live, plot=False)

    points, log_q = _live_log_q(fp)
    assert np.all(points["phi_radial"] > 0)
    # the radius is the point's own: scoring the points again reuses it
    if sum_mirror_density:
        again = logsumexp(fp.mirror_log_q(points), axis=0)
    else:
        again = fp.forward_pass(points)[1]
    np.testing.assert_allclose(again, log_q, rtol=1e-6, atol=1e-6)

    fp.populate(live[np.argmin(live["logL"])], n_samples=50, plot=False)
    assert fp.populated
    assert np.isfinite(
        fp.get_truncation_rule("log_proposal_threshold").threshold
    )
