"""Instrumented nessai runs for the retraining-decision investigation.

Records per-iteration draw counts, pool populations and training events so
that the acceptance model and the cost model can be studied offline.
"""

import json
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)

from nessai.flowmodel.base import FlowModel  # noqa: E402
from nessai.flowsampler import FlowSampler  # noqa: E402
from nessai.model import Model  # noqa: E402
from nessai.proposal.flowproposal.flowproposal import FlowProposal as BaseFlowProposal  # noqa: E402
from nessai.samplers.nestedsampler import NestedSampler  # noqa: E402

LOG = {}
SAMPLER = {}


def _reset_log():
    LOG.clear()
    LOG.update(
        iterations=[],  # (iteration, draws, logLmin)
        populations=[],  # dict
        trainings=[],  # dict
    )


_orig_consume = NestedSampler.consume_sample


def consume_sample(self):
    SAMPLER["s"] = self
    _orig_consume(self)
    count = int(round(1.0 / self.acceptance_history[-1]))
    LOG["iterations"].append((self.iteration, count, float(self.logLmin)))


NestedSampler.consume_sample = consume_sample

_orig_populate = BaseFlowProposal.populate


def populate(self, worst_point, n_samples=10000, **kwargs):
    s = SAMPLER.get("s")
    m = self.model
    n0 = m.likelihood_evaluations
    tl0 = m.likelihood_evaluation_time.total_seconds()
    t0 = time.perf_counter()
    _orig_populate(self, worst_point, n_samples=n_samples, **kwargs)
    dt = time.perf_counter() - t0
    dtl = m.likelihood_evaluation_time.total_seconds() - tl0
    LOG["populations"].append(
        dict(
            iteration=s.iteration if s is not None else -1,
            n_samples=int(self.samples.size),
            n_like=int(m.likelihood_evaluations - n0),
            t_like=dtl,
            t_pop=dt - dtl,
            pop_acc=float(self.population_acceptance),
            training_count=int(self.training_count),
        )
    )


BaseFlowProposal.populate = populate

_orig_fm_train = FlowModel.train
_orig_reset = FlowModel.reset_model


def fm_train(self, samples, *args, **kwargs):
    t0 = time.perf_counter()
    hist = _orig_fm_train(self, samples, *args, **kwargs)
    LOG["_last_train"] = dict(
        epochs=len(hist["loss"]),
        n_train=int(samples.shape[0]),
        t_fit=time.perf_counter() - t0,
        final_val=float(np.min(hist["val_loss"])),
    )
    return hist


def reset_model(self, weights=True, permutations=False):
    if weights and permutations:
        LOG["_reset"] = True
    return _orig_reset(self, weights=weights, permutations=permutations)


FlowModel.train = fm_train
FlowModel.reset_model = reset_model

_orig_train_proposal = NestedSampler.train_proposal


def train_proposal(self, force=False):
    n0 = len(self.history["training_iterations"])
    LOG["_reset"] = False
    t0 = time.perf_counter()
    _orig_train_proposal(self, force=force)
    dt = time.perf_counter() - t0
    if len(self.history["training_iterations"]) > n0:
        d = dict(
            iteration=self.iteration,
            t_train=dt,
            reset=bool(LOG.get("_reset", False)),
            logLmin=float(self.logLmin),
        )
        d.update(LOG.get("_last_train", {}))
        LOG["trainings"].append(d)


NestedSampler.train_proposal = train_proposal


# ---------------------------------------------------------------- models
class Gaussian(Model):
    def __init__(self, dims=4, sigma=1.0, bound=10.0):
        self.names = [f"x_{d}" for d in range(dims)]
        self.bounds = {n: [-bound, bound] for n in self.names}
        self.sigma = sigma
        self._lp = -dims * np.log(2 * bound)

    def log_prior(self, x):
        return np.log(self.in_bounds(x), dtype="float") + self._lp

    def log_likelihood(self, x):
        x = self.unstructured_view(x)
        return -0.5 * np.sum((x / self.sigma) ** 2, axis=-1)


class Rosenbrock(Model):
    def __init__(self, dims=4):
        self.names = [f"x_{d}" for d in range(dims)]
        self.bounds = {n: [-5.0, 5.0] for n in self.names}
        self._lp = -dims * np.log(10.0)

    def log_prior(self, x):
        return np.log(self.in_bounds(x), dtype="float") + self._lp

    def log_likelihood(self, x):
        x = self.unstructured_view(x)
        return -np.sum(
            100.0 * (x[..., 1:] - x[..., :-1] ** 2.0) ** 2.0
            + (1.0 - x[..., :-1]) ** 2.0,
            axis=-1,
        )


class Bimodal(Model):
    """Two well-separated correlated Gaussians."""

    def __init__(self, dims=4, sep=4.0, bound=10.0):
        self.names = [f"x_{d}" for d in range(dims)]
        self.bounds = {n: [-bound, bound] for n in self.names}
        self._lp = -dims * np.log(2 * bound)
        self.mu = np.zeros(dims)
        self.mu[0] = sep
        cov = 0.5 * np.eye(dims) + 0.5
        cov *= 0.3
        self.icov = np.linalg.inv(cov)

    def log_prior(self, x):
        return np.log(self.in_bounds(x), dtype="float") + self._lp

    def log_likelihood(self, x):
        x = self.unstructured_view(x)
        out = []
        for s in (+1, -1):
            d = x - s * self.mu
            out.append(-0.5 * np.einsum("...i,ij,...j->...", d, self.icov, d))
        return np.logaddexp(out[0], out[1])


MODELS = dict(gaussian=Gaussian, rosenbrock=Rosenbrock, bimodal=Bimodal)


def run(name, model_name, dims, outdir, seed=1234, nlive=1000, **kwargs):
    _reset_log()
    SAMPLER.clear()
    model = MODELS[model_name](dims)
    flow_config = kwargs.pop(
        "flow_config", dict(n_blocks=4, n_neurons=16, n_layers=2)
    )
    out = os.path.join(outdir, name)
    fs = FlowSampler(
        model,
        output=out,
        nlive=nlive,
        resume=False,
        seed=seed,
        plot=False,
        flow_config=flow_config,
        checkpointing=False,
        **kwargs,
    )
    t0 = time.perf_counter()
    fs.run(plot=False, save=False)
    wall = time.perf_counter() - t0
    ns = fs.ns
    res = dict(
        name=name,
        model=model_name,
        dims=dims,
        nlive=nlive,
        seed=seed,
        kwargs={k: repr(v) for k, v in kwargs.items()},
        wall=wall,
        log_evidence=float(ns.log_evidence),
        log_evidence_error=float(ns.log_evidence_error),
        n_like=int(ns.total_likelihood_evaluations),
        t_like=ns.likelihood_evaluation_time.total_seconds(),
        iterations=LOG["iterations"],
        populations=LOG["populations"],
        trainings=LOG["trainings"],
        uninformed_until=int(getattr(ns, "maximum_uninformed", 0)),
        retrain_log=(ns.retrain_decision.log if getattr(ns, "retrain_decision", None) is not None else None),
        retrain_summary=(ns.retrain_decision.summary() if getattr(ns, "retrain_decision", None) is not None else None),
    )
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{name}.json"), "w") as f:
        json.dump(res, f)
    return res


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    r = run(**cfg)
    print(
        r["name"],
        f"wall={r['wall']:.1f}s nlike={r['n_like']} "
        f"ntrain={len(r['trainings'])} logZ={r['log_evidence']:.3f}",
    )
