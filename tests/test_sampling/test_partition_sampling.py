# -*- coding: utf-8 -*-
"""Integration tests for partitioned nested sampling."""

import os

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.posterior import compute_weights
from nessai.proposal import FlowProposal
from nessai.samplers.nestedsampler import NestedSampler

torch.set_num_threads(1)


class TwoModes(Model):
    """Two Gaussian modes along ``x``, of different weights and widths, in
    a uniform prior: the evidence and the modes' posterior shares are
    known."""

    def __init__(self, w=(0.7, 0.3), mu=(-5.0, 5.0), sd=(0.5, 0.3)):
        self.names = ["x", "y"]
        self.bounds = {"x": [-10, 10], "y": [-10, 10]}
        self.w, self.mu, self.sd = map(np.array, (w, mu, sd))
        self.logZ = np.log(self.w.sum()) - np.log(400)
        self.share = self.w[1] / self.w.sum()

    def log_prior(self, x):
        log_p = np.log(self.in_bounds(x), dtype="float")
        return log_p - np.log(400)

    def log_likelihood(self, x):
        out = [
            np.log(w)
            - 0.5 * ((x["x"] - m) ** 2 + x["y"] ** 2) / s**2
            - np.log(2 * np.pi * s**2)
            for w, m, s in zip(self.w, self.mu, self.sd)
        ]
        return logsumexp(out, axis=0)


FLOW = dict(n_blocks=2, n_neurons=16)


def _check_result(ns, model, n_sigma=4):
    assert ns.partitions, "no split"
    assert ns.finalised
    err = max(ns.state.log_evidence_error, 0.05)
    assert abs(ns.log_evidence - model.logZ) < n_sigma * err
    samples = np.array(ns.nested_samples)
    assert np.all(np.diff(samples["logL"]) >= 0)
    w = np.exp(ns.state.log_posterior_weights)
    assert np.isclose(w.sum(), 1.0)
    share = w[samples["x"] > 0].sum()
    # the scatter of the share at nlive 500 is ~0.035 (40 seeds)
    assert abs(share - model.share) < 0.12
    # the weights rebuilt from the effective nlive, as nessai-bilby does
    _, log_w = compute_weights(samples["logL"], np.array(ns.state.nlive))
    rebuilt = np.exp(log_w - logsumexp(log_w))
    np.testing.assert_allclose(rebuilt[:-1] / rebuilt[0], w[:-1] / w[0])
    regions = ns.nested_samples_region
    assert len(regions) == len(samples)
    assert set(np.unique(regions)) == {-1, 1, 2}
    # the regions are the two modes
    assert np.all(samples["x"][regions == 1] < 0)
    assert np.all(samples["x"][regions == 2] > 0)


@pytest.mark.slow_integration_test
def test_partition_sampling(tmp_path):
    model = TwoModes()
    output = str(tmp_path / "partition")
    fs = FlowSampler(
        model,
        output=output,
        nlive=500,
        seed=1234,
        plot=True,
        checkpointing=False,
        flow_config=FLOW,
        partition={"coordinates": ["x"]},
    )
    # the logX-logL plot: a partitioned run's last volume is zero
    fs.run(plot=True, plot_posterior=False, plot_indices=False, save=True)
    _check_result(fs.ns, model)
    assert os.path.exists(os.path.join(output, "logXlogL.png"))
    assert os.path.exists(os.path.join(output, "diagnostics", "partition.png"))
    result = fs.ns.get_result_dictionary()
    assert len(result["partitions"]) == 2
    assert np.isclose(result["partitions"]["share"].sum(), 1.0)
    assert os.path.exists(os.path.join(output, "result.hdf5"))


@pytest.mark.slow_integration_test
def test_partition_resume_after_split(tmp_path):
    model = TwoModes()
    output = str(tmp_path / "partition_resume")
    ns = NestedSampler(
        model,
        output=output,
        nlive=500,
        seed=4321,
        plot=False,
        checkpointing=True,
        flow_config=FLOW,
        max_iteration=2000,
        # split as soon as the modes detach, well before the checkpoint
        partition={"coordinates": ["x"], "splitter": {"max_share": 0.5}},
    )
    ns.nested_sampling_loop()
    assert ns.regions is not None, "no split before the checkpoint"
    assert not ns.finalised
    resume_file = os.path.join(output, "nested_sampler_resume.pkl")
    ns = NestedSampler.resume(resume_file, model, flow_config=FLOW)
    ns.initialise()
    assert ns.regions is not None
    for region in ns.active_regions:
        assert region.proposal.flow is not None
        assert region.proposal.truncation.get_rule("region") is not None
    ns.max_iteration = np.inf
    ns.nested_sampling_loop()
    _check_result(ns, model)


@pytest.mark.slow_integration_test
def test_partition_enabled_on_resume(tmp_path):
    """A run started without the partition splits once it is resumed with
    it."""
    model = TwoModes()
    output = str(tmp_path / "partition_late")
    ns = NestedSampler(
        model,
        output=output,
        nlive=500,
        seed=99,
        plot=False,
        checkpointing=True,
        flow_config=FLOW,
        max_iteration=600,
    )
    ns.nested_sampling_loop()
    resume_file = os.path.join(output, "nested_sampler_resume.pkl")
    ns = NestedSampler.resume(
        resume_file,
        model,
        flow_config=FLOW,
        partition={"coordinates": ["x"]},
    )
    ns.initialise()
    ns.max_iteration = np.inf
    ns.nested_sampling_loop()
    _check_result(ns, model)


@pytest.mark.slow_integration_test
def test_partition_nested_split(tmp_path):
    """Three modes: the far one detaches first, the two close ones later,
    by a split of a region."""
    model = TwoModes(
        w=(0.4, 0.3, 0.3), mu=(-6.0, -3.0, 6.0), sd=(0.4, 0.4, 0.4)
    )
    model.share = 0.3
    output = str(tmp_path / "partition_nested")
    fs = FlowSampler(
        model,
        output=output,
        nlive=400,
        seed=7,
        plot=False,
        checkpointing=False,
        flow_config=FLOW,
        partition={"coordinates": ["x"], "splitter": {"max_share": 0.5}},
    )
    fs.run(plot=False, save=False)
    ns = fs.ns
    parents = {p["parent"] for p in ns.partitions}
    assert len(parents) >= 2, "no split of a region"
    err = max(ns.state.log_evidence_error, 0.05)
    assert abs(ns.log_evidence - model.logZ) < 4 * err
    samples = np.array(ns.nested_samples)
    w = np.exp(ns.state.log_posterior_weights)
    for m, share in zip(model.mu, model.w / model.w.sum()):
        near = np.abs(samples["x"] - m) < 1.5
        assert abs(w[near].sum() - share) < 0.1


@pytest.mark.slow_integration_test
def test_partition_n_min_refuses_small_parts(tmp_path):
    """A cut that would leave a part with fewer than ``n_min`` live points
    is not used: the run goes on unsplit."""
    model = TwoModes()
    fs = FlowSampler(
        model,
        output=str(tmp_path / "partition_n_min"),
        nlive=500,
        seed=1234,
        plot=False,
        checkpointing=False,
        flow_config=FLOW,
        # the smaller mode holds 30% of the 500 live points
        partition={"coordinates": ["x"], "n_min": 200},
    )
    fs.run(plot=False, save=False)
    assert fs.ns.regions is None
    assert not fs.ns.partitions
    err = max(fs.ns.state.log_evidence_error, 0.05)
    assert abs(fs.ns.log_evidence - model.logZ) < 4 * err


class FreshProposalForPositiveX:
    """``region_proposal`` factory: the region at ``x > 0`` gets a new
    flow proposal with its own flow, the other keeps its copy."""

    def __init__(self):
        self.calls = []
        self.made = []

    def __call__(self, clone, region, sampler):
        self.calls.append(region.label)
        if np.median(region.live_points["x"]) < 0:
            return None
        proposal = FlowProposal(
            sampler.model,
            flow_config=dict(n_blocks=3, n_neurons=8),
            poolsize=clone._poolsize,
            plot=False,
        )
        self.made.append(proposal)
        return proposal


@pytest.mark.slow_integration_test
def test_partition_region_proposal(tmp_path):
    """A region can sample with a proposal of its own (``region_proposal``),
    restricted to the region like the copies."""
    model = TwoModes()
    factory = FreshProposalForPositiveX()
    fs = FlowSampler(
        model,
        output=str(tmp_path / "partition_region_proposal"),
        nlive=500,
        seed=1234,
        plot=False,
        checkpointing=False,
        flow_config=FLOW,
        partition={"coordinates": ["x"], "region_proposal": factory},
    )
    fs.run(plot=False, save=False)
    _check_result(fs.ns, model)
    assert sorted(factory.calls) == [1, 2]
    (proposal,) = factory.made
    assert proposal.training_count > 0
    assert proposal.truncation.get_rule("region") is not None


class FinalFlowProposal(FlowProposal):
    """A region proposal that declares its region final."""

    splittable = False


class FinalProposalForNegativeX:
    """``region_proposal`` factory: the region at ``x < 0`` gets a new flow
    proposal that is not split again."""

    def __init__(self):
        self.calls = []

    def __call__(self, clone, region, sampler):
        self.calls.append(region.label)
        if np.median(region.live_points["x"]) > 0:
            return None
        return FinalFlowProposal(
            sampler.model,
            flow_config=FLOW,
            poolsize=clone._poolsize,
            plot=False,
        )


@pytest.mark.slow_integration_test
def test_partition_region_proposal_final_and_resumed(tmp_path):
    """A region proposal with ``splittable = False`` keeps its region whole
    (two modes, which the copies split: see test_partition_nested_split),
    and the run resumes from a checkpoint taken after it was installed,
    with the partition passed again (as xg_pe does)."""
    model = TwoModes(
        w=(0.4, 0.3, 0.3), mu=(-6.0, -3.0, 6.0), sd=(0.4, 0.4, 0.4)
    )
    model.share = 0.3
    output = str(tmp_path / "partition_final")

    def partition():
        return {
            "coordinates": ["x"],
            "splitter": {"max_share": 0.5},
            "region_proposal": FinalProposalForNegativeX(),
        }

    ns = NestedSampler(
        model,
        output=output,
        nlive=400,
        seed=7,
        plot=False,
        checkpointing=True,
        flow_config=FLOW,
        max_iteration=2500,
        partition=partition(),
    )
    ns.nested_sampling_loop()
    assert ns.regions is not None, "no split before the checkpoint"
    assert not ns.finalised
    (final,) = [
        r for r in ns.active_regions if isinstance(r.proposal, FinalFlowProposal)
    ]
    assert np.all(final.live_points["x"] < 0)
    ns = NestedSampler.resume(
        os.path.join(output, "nested_sampler_resume.pkl"),
        model,
        flow_config=FLOW,
        partition=partition(),
    )
    ns.initialise()
    (final,) = [
        r for r in ns.active_regions if isinstance(r.proposal, FinalFlowProposal)
    ]
    assert final.proposal.training_count > 0
    assert final.proposal.truncation.get_rule("region") is not None
    ns.max_iteration = np.inf
    ns.nested_sampling_loop()
    assert ns.finalised
    # one split, of the whole prior
    assert {p["parent"] for p in ns.partitions} == {0}
    err = max(ns.state.log_evidence_error, 0.05)
    assert abs(ns.log_evidence - model.logZ) < 4 * err
    samples = np.array(ns.nested_samples)
    w = np.exp(ns.state.log_posterior_weights)
    for m, share in zip(model.mu, model.w / model.w.sum()):
        near = np.abs(samples["x"] - m) < 1.5
        assert abs(w[near].sum() - share) < 0.1
