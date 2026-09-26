# -*- coding: utf-8 -*-
"""Measure the unit costs used by the retrain decision on this hardware.

The costs are measured with the same model, flow configuration and proposal
settings as the run, so the result can be passed directly as
:code:`retrain_costs`. This can be run independently of a sampling run::

    python -m nessai.samplers.retrain_benchmark my_module:MyModel \\
        --nlive 1000 --config run_config.json --output costs.json

where ``run_config.json`` contains the keyword arguments of the run
(``flow_config``, ``poolsize``, ``reparameterisations``, ...). Keyword
arguments that are not used by the proposal are ignored.

The measurement takes roughly the time of one training plus one pool
population. The per-iteration overhead of the sampler, which the in-run
measurement includes in the population cost, is not measured, so the
population cost is slightly underestimated.
"""

import argparse
import copy
import datetime
import importlib
import inspect
import json
import logging
import os
import tempfile
import time

import numpy as np

from .retrain import COST_KEYS

logger = logging.getLogger(__name__)


def _proposal_kwargs(ProposalClass, kwargs):
    """Keep the keyword arguments accepted by the proposal class."""
    from ..proposal.utils import check_proposal_kwargs

    accepted = set()
    for cls in inspect.getmro(ProposalClass):
        if "__init__" in cls.__dict__:
            accepted |= set(inspect.signature(cls.__init__).parameters)
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    for key in ("model", "rng", "output", "plot", "flow_config"):
        kwargs.pop(key, None)
    return check_proposal_kwargs(ProposalClass, kwargs)


def measure_retrain_costs(
    model,
    nlive=2000,
    flow_config=None,
    flow_proposal_class=None,
    epochs=20,
    n_likelihood=None,
    seed=0,
    filename=None,
    output=None,
    **kwargs,
):
    """Measure the unit costs of likelihood, pool population and training.

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
        The model of the run. Its likelihood evaluation counters and random
        number generator are restored afterwards.
    nlive : int
        Number of live points of the run (sets the training-set size and the
        default pool size).
    flow_config : dict, optional
        Flow configuration of the run.
    flow_proposal_class : str or class, optional
        Proposal class of the run.
    epochs : int
        Number of training epochs to time.
    n_likelihood : int, optional
        Number of likelihood evaluations to time. Defaults to ``nlive``.
    seed : int
        Seed of the random number generator used for the benchmark.
    filename : str, optional
        If given, the costs are saved in the format read by
        :py:class:`nessai.samplers.retrain.RetrainCostModel`.
    output : str, optional
        Directory for the proposal files. A temporary directory is used by
        default.
    kwargs :
        Other keyword arguments of the run. Those accepted by the proposal
        class (e.g. ``poolsize``, ``reparameterisations``) are used.

    Returns
    -------
    dict
        Unit costs with keys ``likelihood``, ``population`` and
        ``training``, in seconds.
    """
    from ..proposal.utils import get_flow_proposal_class

    ProposalClass = get_flow_proposal_class(flow_proposal_class)
    kwargs = dict(kwargs)
    if kwargs.get("poolsize") is None:
        kwargs["poolsize"] = nlive
    kwargs = _proposal_kwargs(ProposalClass, kwargs)
    n_likelihood = n_likelihood or nlive

    n_evals = model.likelihood_evaluations
    t_evals = model.likelihood_evaluation_time
    rng = np.random.default_rng(seed)
    model_rng = getattr(model, "rng", None)
    model.rng = rng
    tmp = None
    if output is None:
        tmp = tempfile.TemporaryDirectory()
        output = tmp.name
    try:
        proposal = ProposalClass(
            model,
            rng=rng,
            flow_config=copy.deepcopy(flow_config),
            output=os.path.join(output, ""),
            plot=False,
            **kwargs,
        )
        proposal.initialise()

        # Likelihood
        x = model.new_point(N=max(n_likelihood, nlive))
        x["logP"] = model.batch_evaluate_log_prior(x)
        t0 = model.likelihood_evaluation_time
        st = time.perf_counter()
        x["logL"] = model.batch_evaluate_log_likelihood(x)
        wall = time.perf_counter() - st
        t_like = (model.likelihood_evaluation_time - t0).total_seconds()
        # Batch evaluation may not be timed by the model
        t_like = t_like if t_like > 0 else wall
        likelihood = t_like / x.size

        # Training, with a fixed number of epochs
        x = x[:nlive]
        proposal.flow.training_config["max_epochs"] = epochs
        proposal.flow.training_config["patience"] = epochs + 1
        st = time.perf_counter()
        proposal.train(x, plot=False)
        t_train = time.perf_counter() - st
        n_epochs = getattr(proposal, "last_training_epochs", None) or epochs
        training = t_train / (n_epochs * x.size)

        # Population, excluding likelihood time. The worst point is the
        # lowest likelihood, as at the start of a run.
        worst = x[np.argmin(x["logL"])]
        t0 = model.likelihood_evaluation_time
        st = time.perf_counter()
        proposal.populate(worst, n_samples=kwargs["poolsize"], plot=False)
        t_pop = time.perf_counter() - st
        t_like_pop = (model.likelihood_evaluation_time - t0).total_seconds()
        n_points = max(int(proposal.samples.size), 1)
        population = max(t_pop - t_like_pop, 0.0) / n_points
    finally:
        model.likelihood_evaluations = n_evals
        model.likelihood_evaluation_time = t_evals
        model.rng = model_rng
        if tmp is not None:
            tmp.cleanup()

    costs = dict(
        likelihood=likelihood, population=population, training=training
    )
    logger.info(
        "Measured retrain unit costs: "
        + ", ".join(f"{k}={costs[k]:.3g} s" for k in COST_KEYS)
    )
    if filename is not None:
        save_costs(costs, filename)
    return costs


def save_costs(costs, filename):
    """Save unit costs in the format read by ``RetrainCostModel``."""
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(
            dict(
                unit_costs={k: costs[k] for k in COST_KEYS},
                details=dict(
                    source="benchmark",
                    date=datetime.datetime.now().isoformat(),
                ),
            ),
            f,
            indent=2,
        )


def _load_object(path):
    module, _, name = path.partition(":")
    obj = importlib.import_module(module)
    for attr in name.split("."):
        obj = getattr(obj, attr)
    return obj


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Measure the unit costs used by nessai's retrain decision. "
            "The output can be passed as `retrain_costs`."
        )
    )
    parser.add_argument(
        "model",
        help=(
            "Model as 'module:attribute'. A class is instantiated without "
            "arguments, a callable returning a model is called."
        ),
    )
    parser.add_argument("--nlive", type=int, default=None)
    parser.add_argument(
        "--config",
        help="JSON file with the keyword arguments of the run",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="retrain_costs.json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    model = _load_object(args.model)
    if callable(model):
        model = model()
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    nlive = args.nlive or config.pop("nlive", 2000)
    config.pop("nlive", None)
    costs = measure_retrain_costs(
        model,
        nlive=nlive,
        epochs=args.epochs,
        seed=args.seed,
        filename=args.output,
        **config,
    )
    print(json.dumps(costs, indent=2))
    return costs


if __name__ == "__main__":  # pragma: no cover
    main()
