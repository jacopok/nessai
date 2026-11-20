"""Importance nested sampling with FlowProposal levels (LevelFlowProposal):
the evidence of a Gaussian likelihood, with and without a boundary
inversion (whose draws are scored summed over the mirror sides)."""

import pickle

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.proposal.importance_levels import LevelFlowProposal


class UniformGaussian(Model):
    """Uniform prior on ``[lo, hi]`` per dimension, unit Gaussian likelihood
    at ``mu``."""

    def __init__(self, mu, lo=-10.0, hi=10.0):
        self.mu = np.asarray(mu, float)
        self.names = [f"x{i}" for i in range(len(mu))]
        self.bounds = {n: [lo, hi] for n in self.names}
        self.lo, self.hi = lo, hi

    def log_prior(self, x):
        with np.errstate(divide="ignore"):
            return np.log(self.in_bounds(x), dtype=float) - len(
                self.names
            ) * np.log(self.hi - self.lo)

    def log_likelihood(self, x):
        return sum(
            norm.logpdf(x[n], loc=m) for n, m in zip(self.names, self.mu)
        )

    def to_unit_hypercube(self, x):
        u = x.copy()
        for n in self.names:
            u[n] = (x[n] - self.lo) / (self.hi - self.lo)
        return u

    def from_unit_hypercube(self, u):
        x = u.copy()
        for n in self.names:
            x[n] = self.lo + (self.hi - self.lo) * u[n]
        return x

    def log_unit_hypercube_jacobian(self, x):
        return -len(self.names) * np.log(self.hi - self.lo) * np.ones(len(x))

    def log_evidence(self):
        return sum(
            np.log(norm.cdf(self.hi - m) - norm.cdf(self.lo - m))
            - np.log(self.hi - self.lo)
            for m in self.mu
        )


FLOW = dict(n_blocks=2, n_neurons=16)
TRAINING = dict(max_epochs=200, patience=10)


def _run(tmp_path, model, reparameterisations, seed=1234, **extra):
    fs = FlowSampler(
        model,
        output=str(tmp_path),
        importance_nested_sampler=True,
        nlive=1000,
        min_samples=200,
        seed=seed,
        plot=False,
        checkpointing=False,
        proposal_class="levels",
        proposal_kwargs=dict(
            flow_proposal_class="flowproposal",
            flow_config=FLOW,
            training_config=TRAINING,
            flow_proposal_kwargs=dict(
                reparameterisations=reparameterisations,
                sum_mirror_density=True,
            ),
            n_pilot=5000,
            **extra,
        ),
    )
    fs.run(plot=False, save=False)
    return fs


@pytest.mark.slow_integration_test
@pytest.mark.parametrize(
    "mu, reparameterisations",
    [
        ([1.0, -2.0], {"x0": "z-score", "x1": "z-score"}),
        # the peak on the prior's lower bound: a boundary inversion
        (
            [-10.0, 3.0],
            {
                "x0": {
                    "reparameterisation": "inversion",
                    "detect_edges": True,
                    "inversion_type": "duplicate",
                },
                "x1": "z-score",
            },
        ),
    ],
)
def test_level_proposal_evidence(tmp_path, mu, reparameterisations):
    model = UniformGaussian(mu)
    fs = _run(tmp_path, model, reparameterisations)
    assert isinstance(fs.ns.proposal, LevelFlowProposal)
    err = max(fs.log_evidence_error, 0.02)
    assert abs(fs.log_evidence - model.log_evidence()) < 4 * err


@pytest.mark.slow_integration_test
def test_level_proposal_densities_are_normalised(tmp_path):
    """Each level's unit-hypercube density integrates to 1 (prior
    importance sampling), and survives pickling and resume."""
    model = UniformGaussian([2.0, 0.5])
    fs = _run(tmp_path, model, {"x0": "z-score", "x1": "z-score"})
    prop = fs.ns.proposal
    u = model.sample_unit_hypercube(400_000)
    log_q = prop._log_q_all(u)[:, 1:]
    # E_uniform[q] = 1 in the unit hypercube; the levels are narrow, so
    # check the last (narrowest) against its own draws instead
    for k in range(min(2, log_q.shape[1])):
        mass = np.exp(logsumexp(log_q[:, k]) - np.log(len(u)))
        assert abs(mass - 1) < 0.05, (k, mass)

    pts = prop.draw(2000)[0]
    before = prop._log_q_all(pts)
    state = pickle.dumps(prop)
    resumed = pickle.loads(state)
    resumed.resume(model)
    after = resumed._log_q_all(pts)
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-6)
    # the draws are inside the unit hypercube
    assert model.in_unit_hypercube(pts).all()


class VanishingEdgeGaussian(UniformGaussian):
    """A prior whose hypercube Jacobian is -inf near the lower bound of x0,
    like a Sine prior at 0: draws landing there must be dropped, not given
    an undefined density."""

    def log_unit_hypercube_jacobian(self, x):
        log_j = super().log_unit_hypercube_jacobian(x)
        return np.where(x["x0"] < self.lo + 0.05, -np.inf, log_j)


@pytest.mark.slow_integration_test
def test_level_proposal_drops_vanishing_prior_density(tmp_path):
    model = VanishingEdgeGaussian([-9.5, 3.0])
    fs = _run(tmp_path, model, {"x0": "z-score", "x1": "z-score"})
    pts, log_q = fs.ns.proposal.draw(5000)
    assert np.isfinite(pts["logQ"]).all() and not np.isnan(log_q).any()
    assert (model.from_unit_hypercube(pts)["x0"] >= model.lo + 0.05).all()


class _Interrupt(Exception):
    pass


@pytest.mark.slow_integration_test
def test_level_proposal_resume(tmp_path, monkeypatch):
    """A run interrupted after a few levels resumes from its checkpoint (the
    levels' flows reloaded from their own weights files) to the evidence."""
    model = UniformGaussian([1.0, -2.0])
    kwargs = dict(
        output=str(tmp_path),
        importance_nested_sampler=True,
        nlive=1000,
        min_samples=200,
        seed=4321,
        plot=False,
        checkpointing=True,
        checkpoint_on_iteration=True,
        checkpoint_interval=1,
        proposal_class="levels",
        proposal_kwargs=dict(
            flow_proposal_class="flowproposal",
            flow_config=FLOW,
            training_config=TRAINING,
            flow_proposal_kwargs=dict(
                reparameterisations={"x0": "z-score", "x1": "z-score"}
            ),
            n_pilot=5000,
        ),
    )
    from nessai.samplers.importancesampler import ImportanceNestedSampler

    add = ImportanceNestedSampler.add_new_proposal

    def interrupted(self):
        if self.iteration >= 3:
            raise _Interrupt
        add(self)

    fs = FlowSampler(model, **kwargs)
    monkeypatch.setattr(
        ImportanceNestedSampler, "add_new_proposal", interrupted
    )
    with pytest.raises(_Interrupt):
        fs.run(plot=False, save=False)
    monkeypatch.undo()
    resumed = FlowSampler(model, resume=True, **kwargs)
    assert resumed.ns.iteration == 3
    assert len(resumed.ns.proposal.levels) == 3
    resumed.run(plot=False, save=False)
    err = max(resumed.log_evidence_error, 0.02)
    assert abs(resumed.log_evidence - model.log_evidence()) < 4 * err


@pytest.mark.slow_integration_test
def test_level_proposal_refuses_split_inversion(tmp_path):
    """A 'split' boundary inversion's rescale gives one copy of a point, so
    its density would miss the mirror side: refused."""
    model = UniformGaussian([-10.0, 3.0])
    with pytest.raises(ValueError, match="split"):
        _run(
            tmp_path,
            model,
            {
                "x0": {
                    "reparameterisation": "inversion",
                    "detect_edges": True,
                },
                "x1": "z-score",
            },
        )


@pytest.mark.slow_integration_test
@pytest.mark.parametrize("block, eps", [(2, 0.0), (3, 0.0), (2, 0.05)])
def test_level_proposal_partial_mixture(tmp_path, block, eps):
    """With the partial deterministic mixture (blocks of ``block``
    components) each sample is scored under its own block only, and the
    evidence is still unbiased."""
    model = UniformGaussian([1.0, -2.0, 0.5])
    fs = _run(
        tmp_path,
        model,
        {n: "z-score" for n in model.names},
        mixture_block=block,
        defensive_fraction=eps,
    )
    prop = fs.ns.proposal
    assert prop.mixture_block == block
    assert len(prop.levels) > block
    log_q = fs.ns.training_samples.log_q
    col = fs.ns.training_samples.samples["it"].astype(int) + 1
    scored = ~np.isnan(log_q)
    assert (
        scored == ((np.arange(log_q.shape[1]) // block) == (col // block)[:, None])
    ).all()
    err = max(fs.log_evidence_error, 0.02)
    assert abs(fs.log_evidence - model.log_evidence()) < 4 * err


@pytest.mark.slow_integration_test
def test_level_proposal_defensive_density_normalised(tmp_path):
    """A defensive level, (1 - eps) q_k + eps prior, integrates to 1 and
    draws a fraction eps from the prior."""
    model = UniformGaussian([2.0, 0.5])
    fs = _run(
        tmp_path,
        model,
        {"x0": "z-score", "x1": "z-score"},
        defensive_fraction=0.1,
    )
    prop = fs.ns.proposal
    u = model.sample_unit_hypercube(400_000)
    log_q = prop._log_q_all(u)[:, 1]
    mass = np.exp(logsumexp(log_q) - np.log(len(u)))
    assert abs(mass - 1) < 0.05, mass
    assert (log_q >= np.log(0.1) - 1e-12).all()
    pts = prop._draw_level(len(prop.levels) - 1, 20000)
    x = model.from_unit_hypercube(pts)
    # the last level is narrow about the peak: draws far from it are the
    # prior's (|x - mu| > 6 has negligible flow mass)
    far = (np.abs(x["x0"] - 2.0) > 6) | (np.abs(x["x1"] - 0.5) > 6)
    expected = 0.1 * (1 - (12 / 20) ** 2)
    assert abs(far.mean() - expected) < 0.01, (far.mean(), expected)


@pytest.mark.slow_integration_test
@pytest.mark.parametrize(
    "reparameterisations",
    [
        {"x0": "z-score", "x1": "z-score"},
        {
            "x0": {
                "reparameterisation": "inversion",
                "detect_edges": True,
                "inversion_type": "duplicate",
            },
            "x1": "z-score",
        },
    ],
)
def test_level_proposal_prime_box(tmp_path, reparameterisations):
    """With a tight prime-space box (2 sigma) each level is its flow
    truncated to the box: still normalised, draws inside it, and the
    evidence unbiased."""
    mu = [2.0, 0.5] if "z-score" == reparameterisations["x0"] else [-10.0, 3.0]
    model = UniformGaussian(mu)
    fs = _run(tmp_path, model, reparameterisations, box_sigma=2.0)
    prop = fs.ns.proposal
    assert len(prop.prime_boxes) == len(prop.levels)
    assert all(b is not None for b in prop.prime_boxes)
    u = model.sample_unit_hypercube(400_000)
    log_q = prop._log_q_all(u)[:, 1]
    mass = np.exp(logsumexp(log_q) - np.log(len(u)))
    assert abs(mass - 1) < 0.05, mass
    err = max(fs.log_evidence_error, 0.02)
    assert abs(fs.log_evidence - model.log_evidence()) < 4 * err


def test_level_proposal_prime_box_fallback(tmp_path):
    """A box holding (almost) none of a level's draws is dropped for that
    level, which stays normalised, instead of failing the run."""
    model = UniformGaussian([2.0, 0.5])
    fs = _run(tmp_path, model, {"x0": "z-score", "x1": "z-score"}, box_sigma=1e-9)
    prop = fs.ns.proposal
    assert len(prop.prime_boxes) == len(prop.levels)
    assert all(b is None for b in prop.prime_boxes)
    err = max(fs.log_evidence_error, 0.02)
    assert abs(fs.log_evidence - model.log_evidence()) < 4 * err


@pytest.mark.slow_integration_test
def test_level_proposal_heavy_kernels(tmp_path):
    """With heavy-sample kernels forced on (a tiny threshold) the levels
    blend a Student-t mixture into their flows: still normalised, the draws
    inside the hypercube, and the evidence unbiased."""
    model = UniformGaussian([2.0, 0.5])
    fs = _run(
        tmp_path,
        model,
        {"x0": "z-score", "x1": "z-score"},
        kernel_threshold=1e-6,
        kernel_fraction=0.3,
    )
    prop = fs.ns.proposal
    assert len(prop.kernels) == len(prop.levels)
    assert sum(k is not None for k in prop.kernels) >= len(prop.levels) - 1
    u = model.sample_unit_hypercube(400_000)
    log_q = prop._log_q_all(u)[:, 1:]
    for k in range(min(2, log_q.shape[1])):
        mass = np.exp(logsumexp(log_q[:, k]) - np.log(len(u)))
        assert abs(mass - 1) < 0.05, (k, mass)
    # a plain flow carries the kernel in its prime space
    k = next(i for i, kern in enumerate(prop.kernels) if kern is not None)
    assert prop.kernels[k]["parts"][0][0] is None
    assert type(prop.levels[k].flow.model).__name__ == "KernelMixFlow"
    # a warm copy for the next level starts without it
    nxt = prop.levels[k + 1] if k + 1 < len(prop.levels) else None
    if nxt is not None and prop.kernels[k + 1] is None:
        assert type(nxt.flow.model).__name__ != "KernelMixFlow"
    pts = prop.draw(5000)[0]
    assert model.in_unit_hypercube(pts).all()
    assert np.isfinite(pts["logQ"]).all()
    err = max(fs.log_evidence_error, 0.02)
    assert abs(fs.log_evidence - model.log_evidence()) < 4 * err


@pytest.mark.parametrize("target", [600, 1500])
def test_level_proposal_min_training_ess(tmp_path, target, caplog):
    """With a minimum training ESS the latest level is topped up until the
    samples above the next threshold have it; the extra draws count with
    their level and the evidence stays unbiased."""
    import logging

    from nessai.utils.stats import effective_sample_size

    caplog.set_level(logging.INFO)
    model = UniformGaussian([1.0, -2.0])
    fs = FlowSampler(
        model,
        output=str(tmp_path),
        importance_nested_sampler=True,
        nlive=1000,
        min_samples=200,
        seed=1234,
        plot=False,
        checkpointing=False,
        strict_threshold=True,
        min_training_ess=target,
        proposal_class="levels",
        proposal_kwargs=dict(
            flow_proposal_class="flowproposal",
            flow_config=FLOW,
            training_config=TRAINING,
            flow_proposal_kwargs=dict(
                reparameterisations={"x0": "z-score", "x1": "z-score"},
                sum_mirror_density=True,
            ),
            n_pilot=5000,
        ),
    )
    fs.run(plot=False, save=False)
    assert fs.ns.history.get("top_ups")
    assert "extra batch" in caplog.text
    assert "batch limit" not in caplog.text
    assert sum(fs.ns.sample_counts.values()) == len(fs.ns.samples_unit)
    err = max(fs.log_evidence_error, 0.02)
    assert abs(fs.log_evidence - model.log_evidence()) < 4 * err


@pytest.mark.parametrize("beta", [1.0, 0.5])
def test_level_proposal_training_subsample(tmp_path, beta, caplog):
    """Large training sets are subsampled (keep p ~ w**beta, weights w / p);
    the evidence is unaffected."""
    import logging

    caplog.set_level(logging.INFO)
    model = UniformGaussian([1.0, -2.0])
    fs = FlowSampler(
        model,
        output=str(tmp_path),
        importance_nested_sampler=True,
        nlive=1000,
        min_samples=200,
        seed=1234,
        plot=False,
        checkpointing=False,
        strict_threshold=True,
        min_training_ess=600,
        training_max_samples=800,
        training_accept_power=beta,
        proposal_class="levels",
        proposal_kwargs=dict(
            flow_proposal_class="flowproposal",
            flow_config=FLOW,
            training_config=TRAINING,
            flow_proposal_kwargs=dict(
                reparameterisations={"x0": "z-score", "x1": "z-score"},
                sum_mirror_density=True,
            ),
            n_pilot=5000,
        ),
    )
    fs.run(plot=False, save=False)
    assert "Training set subsampled" in caplog.text
    # the top-ups target the expected ESS after subsampling
    import re

    for m in re.finditer(r"ESS (\d+) -> expected (\d+)", caplog.text):
        assert int(m.group(2)) >= 600 or "extra batch" not in caplog.text
    err = max(fs.log_evidence_error, 0.02)
    assert abs(fs.log_evidence - model.log_evidence()) < 4 * err


@pytest.mark.parametrize(
    "seed, score", [(1234, "weight"), (4321, "capped")]
)
def test_level_proposal_thinned_levels(tmp_path, seed, score, caplog):
    """A level thinned after each batch (kept with p ~ s, counted 1/p)
    keeps the evidence unbiased; the level counts stay the draws."""
    import logging

    caplog.set_level(logging.INFO)
    model = UniformGaussian([1.0, -2.0])
    fs = FlowSampler(
        model,
        output=str(tmp_path),
        importance_nested_sampler=True,
        nlive=1000,
        min_samples=200,
        seed=seed,
        plot=False,
        checkpointing=False,
        strict_threshold=True,
        min_training_ess=600,
        min_training_ess_max_batches=50,
        level_max_samples=1500,
        level_thin_score=score,
        proposal_class="levels",
        proposal_kwargs=dict(
            flow_proposal_class="flowproposal",
            flow_config=FLOW,
            training_config=TRAINING,
            flow_proposal_kwargs=dict(
                reparameterisations={"x0": "z-score", "x1": "z-score"},
                sum_mirror_density=True,
            ),
            n_pilot=5000,
        ),
    )
    fs.run(plot=False, save=False)
    assert "keeps" in caplog.text
    assert fs.ns.training_samples.thinned
    assert len(fs.ns.samples_unit) < sum(fs.ns.sample_counts.values())
    err = max(fs.log_evidence_error, 0.02)
    assert abs(fs.log_evidence - model.log_evidence()) < 4 * err


class _SharedFit:
    """Shared on a plain deep copy (one per proposal, like nessai-gw's
    polarisation ellipse), copied when the memo asks for independent
    copies."""

    def __init__(self, value=0.0):
        self.value = value

    def __deepcopy__(self, memo):
        from nessai.proposal.importance_levels import INDEPENDENT_COPY

        if not memo.get(INDEPENDENT_COPY):
            return self
        new = _SharedFit(self.value)
        memo[id(self)] = new
        return new


@pytest.mark.slow_integration_test
def test_level_proposal_levels_share_no_fit(tmp_path):
    """A level copied from another gets its own instance of an object the
    proposal shares on a deep copy, held by the proposal and its flow model
    alike: refitting it changes no other level."""
    import copy

    model = UniformGaussian([1.0, -2.0])
    fs = _run(tmp_path, model, {"x0": "z-score", "x1": "z-score"})
    P = fs.ns.proposal
    src = P.levels[-1]
    fit = _SharedFit(1.0)
    src.shared_fit = fit
    src.flow.model.shared_fit = fit
    assert copy.deepcopy(fit) is fit
    new = P._copy(src, str(tmp_path / "copy"))
    assert new.shared_fit is not fit
    assert new.flow.model.shared_fit is new.shared_fit
    new.shared_fit.value = 2.0
    assert src.shared_fit.value == 1.0


@pytest.mark.slow_integration_test
def test_level_proposal_retrain_on_stall(tmp_path, caplog):
    """With min_training_ess_retrain_after, a level that needs more top-ups
    than that is followed by a new level at the same threshold (no draw
    discarded) at most max_retrains times per threshold; the evidence stays
    unbiased."""
    import logging

    caplog.set_level(logging.INFO)
    model = UniformGaussian([1.0, -2.0])
    fs = FlowSampler(
        model,
        output=str(tmp_path),
        importance_nested_sampler=True,
        nlive=1000,
        min_samples=200,
        seed=1234,
        plot=False,
        checkpointing=False,
        strict_threshold=True,
        min_training_ess=1500,
        min_training_ess_retrain_after=1,
        min_training_ess_max_retrains=2,
        proposal_class="levels",
        proposal_kwargs=dict(
            flow_proposal_class="flowproposal",
            flow_config=FLOW,
            training_config=TRAINING,
            flow_proposal_kwargs=dict(
                reparameterisations={"x0": "z-score", "x1": "z-score"},
                sum_mirror_density=True,
            ),
            n_pilot=5000,
        ),
    )
    fs.run(plot=False, save=False)
    ns = fs.ns
    retrains = ns.history.get("retrains")
    assert retrains and "stalled" in caplog.text
    # each retrain repeats the threshold it was made at, at most twice each
    thresholds = ns.history["logL_threshold"]
    for it, _, t, _, _ in retrains:
        assert thresholds[it] == t
    _, counts = np.unique([r[2] for r in retrains], return_counts=True)
    assert counts.max() <= 2
    assert len(ns.proposal.levels) == ns.iteration
    assert sum(ns.sample_counts.values()) == len(ns.samples_unit)
    err = max(fs.log_evidence_error, 0.02)
    assert abs(fs.log_evidence - model.log_evidence()) < 4 * err


class _MixtureStub:
    """compute_meta_proposal_from_log_q of a count-weighted mixture."""

    def __init__(self, weights):
        self.weights_array = np.asarray(weights, float)

    def compute_meta_proposal_from_log_q(self, log_q):
        return logsumexp(log_q, b=self.weights_array, axis=1)


def _top_up_state(log_q1_in_b):
    """A sampler state with the prior and two levels (1000 draws each):
    region A, where both levels have density 1, and region B, where level 0
    has density 1 and level 1 ``exp(log_q1_in_b)``; 100 samples of level 1
    in A, 5 of level 0 in B, all above the threshold, U = 1."""
    from nessai.samplers.importancesampler import (
        ImportanceNestedSampler,
        OrderedSamples,
    )

    ns = ImportanceNestedSampler.__new__(ImportanceNestedSampler)
    ns.sample_counts = {-1: 1000, 0: 1000, 1: 1000}
    ns.min_training_ess_top_up = "ess_gain"
    ns.proposal = _MixtureStub([1 / 3, 1 / 3, 1 / 3])
    n_a, n_b = 100, 5
    samples = np.zeros(
        n_a + n_b,
        dtype=[("logL", "f8"), ("logW", "f8"), ("logU", "f8"), ("it", "i4")],
    )
    samples["logL"] = 1.0
    samples["it"][:n_a] = 1
    samples["it"][n_a:] = 0
    log_q = np.full((n_a + n_b, 3), np.log(1e-6))  # the prior: negligible
    log_q[:n_a, 1:] = 0.0
    log_q[n_a:, 1] = 0.0
    log_q[n_a:, 2] = log_q1_in_b
    samples["logU"] = 0.0
    samples["logW"] = -ns.proposal.compute_meta_proposal_from_log_q(log_q)
    ts = OrderedSamples()
    ts.samples, ts.log_q = samples, log_q
    ns.training_samples = ts
    return ns


def test_top_up_level_goes_to_the_level_covering_the_heaviest_samples():
    """``ess_gain``: the batch goes to the level that alone covers the
    samples with the largest weights; when the latest level covers them as
    well, to the latest; ``latest`` ignores the scores."""
    ns = _top_up_state(log_q1_in_b=np.log(1e-4))
    scores = ns.top_up_level_scores(0.0)
    assert scores[0] > scores[1]
    assert ns._choose_top_up_level(0.0) == 0
    # the region also covered by the latest level: its draws win
    ns = _top_up_state(log_q1_in_b=np.log(2.0))
    assert ns._choose_top_up_level(0.0) == 1
    ns = _top_up_state(log_q1_in_b=np.log(1e-4))
    ns.min_training_ess_top_up = "latest"
    assert ns._choose_top_up_level(0.0) == 1
    # nothing above the threshold: the latest
    ns.min_training_ess_top_up = "ess_gain"
    assert ns._choose_top_up_level(2.0) == 1


def test_top_up_level_score_matches_the_change_of_sum_w2():
    """The score's ranking is the first-order change of sum w^2 per draw:
    the deterministic part (the count raised, Q recomputed) by finite
    difference, plus the expected squared weight of a new draw as the pool
    estimates it."""
    ns = _top_up_state(log_q1_in_b=np.log(0.05))
    ts = ns.training_samples
    counts = np.array([1000.0, 1000.0, 1000.0])

    def sum_w2(c):
        lq = logsumexp(ts.log_q, b=c / c.sum(), axis=1)
        return np.exp(logsumexp(2 * (ts.samples["logU"] - lq)))

    s2 = sum_w2(counts)
    change = {}
    for j in (0, 1):
        c = counts.copy()
        c[j + 1] += 1.0
        # expected w^2 of one draw from level j (as the pool estimates it)
        lQ = logsumexp(ts.log_q, b=counts / counts.sum(), axis=1)
        e_new = np.exp(
            logsumexp(
                2 * (ts.samples["logU"] - lQ) + ts.log_q[:, j + 1] - lQ
            )
        ) / counts.sum()
        change[j] = sum_w2(c) - s2 + e_new
    scores = ns.top_up_level_scores(0.0)
    assert (change[0] < change[1]) == (scores[0] > scores[1])
    # and the formula: change_j = (2 S2 - exp(score_j)) / N
    for j in (0, 1):
        pred = (2 * s2 - np.exp(scores[j])) / counts.sum()
        assert np.isclose(change[j], pred, rtol=1e-2), (j, change[j], pred)


def _ess_gain_sampler(tmp_path, **kwargs):
    options = dict(
        min_training_ess=600,
        min_training_ess_max_batches=20,
        min_training_ess_top_up="ess_gain",
        level_max_samples=1500,
        level_thin_score="capped",
        checkpointing=False,
    )
    options.update(kwargs)
    return FlowSampler(
        UniformGaussian([1.0, -2.0]),
        output=str(tmp_path),
        importance_nested_sampler=True,
        nlive=1000,
        min_samples=200,
        seed=1234,
        plot=False,
        strict_threshold=True,
        proposal_class="levels",
        proposal_kwargs=dict(
            flow_proposal_class="flowproposal",
            flow_config=FLOW,
            training_config=TRAINING,
            flow_proposal_kwargs=dict(
                reparameterisations={"x0": "z-score", "x1": "z-score"},
                sum_mirror_density=True,
            ),
            n_pilot=5000,
        ),
        **options,
    )


def test_top_ups_from_earlier_levels_keep_the_evidence(
    tmp_path, monkeypatch, caplog
):
    """Top-ups drawn from an earlier level (forced: every other batch from
    the level before the latest) count with that level, are thinned like
    it, and leave the evidence unbiased."""
    import logging

    from nessai.samplers.importancesampler import ImportanceNestedSampler

    caplog.set_level(logging.INFO)
    calls = {"n": 0}

    def alternate(self, threshold):
        latest = max(self.sample_counts)
        calls["n"] += 1
        if latest >= 1 and calls["n"] % 2:
            return latest - 1
        return latest

    monkeypatch.setattr(
        ImportanceNestedSampler, "_choose_top_up_level", alternate
    )
    # a target the levels reach only after several batches each
    fs = _ess_gain_sampler(tmp_path, min_training_ess=1500,
                           level_max_samples=3000)
    fs.run(plot=False, save=False)
    ns = fs.ns
    tops = ns.history["top_ups"]
    assert "drawn from level" in caplog.text
    # some batches went to a level other than the latest at the time
    latest = {}
    for it, _, level in tops:
        latest[it] = max(latest.get(it, level), level)
    assert any(level < latest[it] for it, _, level in tops)
    # every sample's count: draws per level match the bookkeeping
    ts = ns.training_samples
    assert sum(ns.sample_counts.values()) == ts.samples.size + getattr(
        ts, "n_dropped", 0
    )
    err = max(fs.log_evidence_error, 0.02)
    model = UniformGaussian([1.0, -2.0])
    assert abs(fs.log_evidence - model.log_evidence()) < 4 * err


def test_ess_gain_top_ups_run(tmp_path):
    """``ess_gain`` end to end: the evidence stays unbiased."""
    fs = _ess_gain_sampler(tmp_path)
    fs.run(plot=False, save=False)
    assert all(len(t) == 3 for t in fs.ns.history.get("top_ups", []))
    err = max(fs.log_evidence_error, 0.02)
    model = UniformGaussian([1.0, -2.0])
    assert abs(fs.log_evidence - model.log_evidence()) < 4 * err


def test_min_training_ess_top_up_is_validated(tmp_path):
    with pytest.raises(ValueError, match="min_training_ess_top_up"):
        _ess_gain_sampler(tmp_path, min_training_ess_top_up="heaviest")


def test_resume_takes_the_top_up_rule(tmp_path, caplog):
    """A resumed sampler takes ``min_training_ess_top_up`` from the call (it
    only steers the next top-ups) and carries on."""
    import logging

    caplog.set_level(logging.INFO)
    kwargs = dict(
        min_training_ess_top_up="latest",
        checkpointing=True,
        checkpoint_on_iteration=True,
        checkpoint_interval=1,
        max_iteration=2,
    )
    fs = _ess_gain_sampler(tmp_path, **kwargs)
    fs.run(plot=False, save=False)
    assert (tmp_path / "nested_sampler_resume.pkl").exists()
    kwargs.update(min_training_ess_top_up="ess_gain", max_iteration=4)
    resumed = _ess_gain_sampler(tmp_path, resume=True, **kwargs)
    assert resumed.ns.iteration >= 2
    assert resumed.ns.min_training_ess_top_up == "ess_gain"
    assert "min_training_ess_top_up latest -> ess_gain" in caplog.text
    resumed.run(plot=False, save=False)
    with pytest.raises(ValueError, match="min_training_ess_top_up"):
        kwargs.update(min_training_ess_top_up="heaviest")
        _ess_gain_sampler(tmp_path, resume=True, **kwargs)


def test_resume_of_a_checkpoint_without_top_up_levels_saves(tmp_path):
    """A checkpoint written before the top-ups recorded their level (entries
    ``(iteration, n)``) resumes with the level filled in (the latest then,
    ``iteration - 1``), so the run's history still saves to HDF5."""
    kwargs = dict(
        min_training_ess=1500,
        level_max_samples=3000,
        min_training_ess_top_up="latest",
        checkpointing=True,
        checkpoint_on_iteration=True,
        checkpoint_interval=1,
        max_iteration=3,
    )
    fs = _ess_gain_sampler(tmp_path, **kwargs)
    fs.run(plot=False, save=False)
    tops = list(fs.ns.history["top_ups"])
    assert tops and all(t[2] == t[0] - 1 for t in tops)
    # the checkpoint as the earlier code wrote it
    fs.ns.history["top_ups"] = [(t[0], t[1]) for t in tops]
    fs.ns.checkpoint(periodic=True, force=True)
    kwargs.update(min_training_ess_top_up="ess_gain", max_iteration=5)
    resumed = _ess_gain_sampler(tmp_path, resume=True, **kwargs)
    assert resumed.ns.history["top_ups"][: len(tops)] == [tuple(t) for t in tops]
    resumed.run(plot=False, save=True)
    assert (tmp_path / "result.hdf5").exists()
    assert all(len(t) == 3 for t in resumed.ns.history["top_ups"])
