# -*- coding: utf-8 -*-
"""Tests for the guard on a training of the flow (``training_guard`` of
:class:`nessai.samplers.retrain.RetrainDecision`)."""

import logging
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nessai.samplers.nestedsampler import NestedSampler
from nessai.samplers.retrain import RetrainDecision

from .test_retrain import make_decision, run_episodes


def test_guard_rejects():
    d = make_decision(guard_tolerance=0.1)
    assert d.guard_rejects(-10.0, -9.5)
    assert not d.guard_rejects(-10.0, -9.95)
    assert not d.guard_rejects(-10.0, -11.0)
    assert not d.guard_rejects(None, -9.0)
    assert not d.guard_rejects(-10.0, None)
    d.training_guard = False
    assert not d.guard_rejects(-10.0, -9.0)


def test_rejected_training_lowers_the_fresh_level(rng, caplog):
    """A discarded training counts as a start level for the prediction of
    the next one, and the running episode goes on."""
    caplog.set_level(logging.INFO)
    d = make_decision()
    it = run_episodes(d, rng, [np.log(0.3)] * 4, 300)
    ep = d.current
    before = d._warm_level(at=it)[0]
    d.reset_next = True
    d.record_rejected_training(it, 1.0)
    assert d.current is ep and len(d.episodes) == 4
    assert d.reset_next is False
    after = d._warm_level(at=it)[0]
    assert after < before - 0.2
    assert d.log[-1]["reason"] == "guard"
    assert d.log[-1]["excess"] == pytest.approx(1.0)
    assert "discarded" in caplog.text
    # the fit of the running episode is not affected
    np.testing.assert_allclose(d.current_fit()[0][0], np.log(0.3), atol=0.3)
    # and it is forgotten with the lineage
    d.start_episode(it, True, 50, d.nlive)
    for _ in range(300):
        d.record_iteration(3)
    assert len(d._level_history()) == 1
    pickle.loads(pickle.dumps(d))


def test_guard_defaults_after_unpickling(rng):
    d = make_decision()
    run_episodes(d, rng, [np.log(0.3)] * 2, 200)
    for attr in ("training_guard", "guard_tolerance", "_rejected"):
        del d.__dict__[attr]
    d2 = pickle.loads(pickle.dumps(d))
    assert d2.training_guard is True
    d2.record_rejected_training(400, 0.5)
    assert len(d2._rejected) == 1
    d2.decide(400, 1000)


class _Model(torch.nn.Linear):
    structure_changed = False
    hold_structure = False


class _Flow:
    weights_file = None

    def __init__(self):
        self.model = _Model(2, 2)
        self._optimiser = torch.optim.Adam(self.model.parameters())


class _Proposal:
    """Stand-in whose 'support mass' is the first weight of its flow; each
    training adds the next of ``steps`` to it."""

    def __init__(self, steps, structural=()):
        self.flow = _Flow()
        with torch.no_grad():
            self.flow.model.weight.zero_()
        self.steps = list(steps)
        self.structural = list(structural)
        self.training_count = 1
        self.training_data = "old"
        self._reparameterisation = {"scale": 1.0}
        self.rounds = []
        self.held = []
        self.max_next_poolsize = 123

    def train(self, x):
        model = self.flow.model
        self.held.append(model.hold_structure)
        model.structure_changed = bool(
            self.structural.pop(0) if self.structural else False
        ) and not model.hold_structure
        model.hold_structure = False
        with torch.no_grad():
            model.weight += self.steps.pop(0)
        self.training_data = x
        self._reparameterisation["scale"] += 1.0
        self.training_count += 1

    def log_support_mass(self, worst_point, n_samples=5000, rounds=3):
        self.rounds.append(rounds)
        return float(self.flow.model.weight[0, 0])

    def reset(self):
        with torch.no_grad():
            self.flow.model.weight += 5.0


def _sampler():
    ns = object.__new__(NestedSampler)
    ns._rollback_flow = "saved"
    return ns


def _train(ns, proposal, decision, flow_reset=lambda: None):
    return ns._guarded_train(proposal, decision, "new", flow_reset, None, 1200)


def test_guarded_train_keeps_a_good_training(rng):
    d = make_decision()
    run_episodes(d, rng, [np.log(0.3)] * 3, 300)
    p = _Proposal([-0.5])
    ns = _sampler()
    assert _train(ns, p, d) == (True, False)
    assert p.training_data == "new" and p.training_count == 2
    assert p.rounds == [1, d.guard_rounds]
    assert ns._rollback_flow == "saved"
    assert not d._rejected


def test_guarded_train_discards_a_bad_training(rng):
    d = make_decision()
    run_episodes(d, rng, [np.log(0.3)] * 3, 300)
    p = _Proposal([0.7])
    model, reparam = p.flow.model, p._reparameterisation
    ns = _sampler()
    assert _train(ns, p, d) == (False, False)
    # the proposal is as it was: copies of the flow and reparameterisations
    assert float(p.flow.model.weight[0, 0]) == 0.0
    assert p.flow.model is not model
    opt = p.flow._optimiser
    assert opt.param_groups[0]["params"][0] is p.flow.model.weight
    assert p._reparameterisation == {"scale": 1.0}
    assert p._reparameterisation is not reparam
    assert p.training_data == "old" and p.training_count == 1
    assert p.max_next_poolsize is None
    assert ns._rollback_flow is None
    assert d._rejected[-1][0] == 1200
    assert d.log[-1]["excess"] == pytest.approx(0.7)


def test_guarded_train_retries_without_the_structural_change(rng):
    d = make_decision()
    run_episodes(d, rng, [np.log(0.3)] * 3, 300)
    p = _Proposal([0.7, -0.2], structural=[True])
    assert _train(_sampler(), p, d) == (True, False)
    assert p.held == [False, True]
    assert float(p.flow.model.weight[0, 0]) == pytest.approx(-0.2)
    assert p.training_count == 2
    assert not d._rejected


class _FrameProposal(_Proposal):
    """The structural change is the proposal's own (not its flow's)."""

    structure_changed = False
    hold_structure = False

    def train(self, x):
        self.held_frame = getattr(self, "held_frame", [])
        self.held_frame.append(self.hold_structure)
        moved = not self.hold_structure
        self.hold_structure = False
        super().train(x)
        self.structure_changed = moved


def test_guarded_train_retries_without_the_proposal_s_change(rng):
    d = make_decision()
    run_episodes(d, rng, [np.log(0.3)] * 3, 300)
    p = _FrameProposal([0.7, -0.2])
    assert _train(_sampler(), p, d) == (True, False)
    assert p.held_frame == [False, True]
    assert float(p.flow.model.weight[0, 0]) == pytest.approx(-0.2)
    assert p.training_count == 2
    assert not d._rejected


def test_guarded_train_retries_a_reset_warm(rng):
    """A scheduled reset whose flow is worse is discarded at once, and the
    flow from before it is trained instead."""
    d = make_decision()
    run_episodes(d, rng, [np.log(0.3)] * 3, 300)
    d.reset_next = True
    d._attempt_next = (0.0, 1.0)
    p = _Proposal([0.0, -0.2])
    assert _train(_sampler(), p, d, flow_reset=p.reset) == (True, False)
    assert float(p.flow.model.weight[0, 0]) == pytest.approx(-0.2)
    assert d.reset_next is False and d._attempt_next is None


def test_guarded_train_discards_both_attempts(rng):
    d = make_decision()
    run_episodes(d, rng, [np.log(0.3)] * 3, 300)
    p = _Proposal([0.7, 0.4], structural=[True])
    assert _train(_sampler(), p, d) == (False, False)
    assert float(p.flow.model.weight[0, 0]) == 0.0
    assert p.training_count == 1
    assert d.log[-1]["excess"] == pytest.approx(0.4)


@pytest.mark.parametrize("case", ["off", "first", "no_decision", "no_cap"])
def test_guarded_train_without_guard(rng, case):
    d = make_decision(training_guard=case != "off")
    run_episodes(d, rng, [np.log(0.3)] * 3, 300)
    p = _Proposal([0.7])
    if case == "first":
        p.training_count = 0
    if case == "no_cap":
        p.log_support_mass = lambda *a, **k: None
    out = _train(_sampler(), p, None if case == "no_decision" else d)
    assert out == (True, case == "first")
    assert float(p.flow.model.weight[0, 0]) == pytest.approx(0.7)


def test_restore_saves_the_weights_file(tmp_path):
    ns = _sampler()
    p = SimpleNamespace(flow=_Flow(), training_count=1)
    p.flow.weights_file = str(tmp_path / "model.pt")
    p.flow.save_weights = lambda f: torch.save(p.flow.model.state_dict(), f)
    before = p.flow.model.weight.detach().clone()
    saved = ns._training_snapshot(p)
    with torch.no_grad():
        p.flow.model.weight += 1.0
    p.flow.save_weights(p.flow.weights_file)
    ns._restore_training_snapshot(saved)
    on_disk = torch.load(p.flow.weights_file, weights_only=True)
    assert torch.equal(on_disk["weight"], before)


def test_snapshot_keeps_shared_state():
    """What a deep copy leaves shared is kept through the proposal's hooks."""
    ns = _sampler()
    shared = {"fit": 1.0}
    p = SimpleNamespace(
        flow=_Flow(),
        training_count=1,
        _shared_training_state=lambda: dict(shared),
        _restore_shared_training_state=lambda state: shared.update(state),
    )
    saved = ns._training_snapshot(p)
    shared["fit"] = 2.0
    ns._restore_training_snapshot(saved)
    assert shared == {"fit": 1.0}
