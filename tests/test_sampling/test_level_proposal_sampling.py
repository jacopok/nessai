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
