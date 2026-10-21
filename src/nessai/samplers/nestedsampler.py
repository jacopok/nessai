# -*- coding: utf-8 -*-
"""
Functions and objects related to the main nested sampling algorithm.
"""

import datetime
import logging
import math
import os
import pickle
from collections import deque
from copy import copy, deepcopy
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.special import logsumexp

from .. import config
from .. import partition as partition_module
from ..evidence import _NSIntegralState
from ..livepoint import empty_structured_array
from ..plot import nessai_style, plot_indices, plot_trace, sanitise_array
from ..proposal.utils import (
    check_proposal_kwargs,
    get_flow_proposal_class,
)
from ..stopping_criteria import StoppingCriterionRegistry
from ..utils import (
    compute_indices_ks_test,
    rolling_mean,
)
from .base import BaseNestedSampler
from .retrain import RetrainDecision

logger = logging.getLogger(__name__)


class NestedSampler(BaseNestedSampler):
    """
    Nested Sampler class.

    Parameters
    ----------
    model: :obj:`nessai.model.Model`
        User defined model
    nlive : int, optional
        Number of live points.
    output : str
        Path for output
    stopping : float, optional
        Stop when remaining samples wouldn't change logZ estimate by this much.
    max_iteration : int, optional
        Maximum number of iterations to run before force sampler to stop.
        If stopping criteria is met before max. is reached sampler will stop.
    checkpointing : bool, optional
        Boolean to toggle checkpointing, must be enabled to resume the sampler.
        If false the sampler is still saved at the end of sampling.
    checkpoint_interval : int
        The interval used for checkpointing. By default this is a time interval
        in seconds. If :code:`checkpoint_on_iteration=True` this corresponds to
        the number of iterations between checkpointing.
    checkpoint_on_iteration : bool
        If true the checkpointing interval is checked against the number of
        iterations
    logging_interval : int, optional
        The interval in seconds used for periodic logging. If not specified,
        then periodic logging is disabled.
    log_on_iteration : bool
        If true logging will occur based on the iteration. If false logging
        will be periodic if `logging_interval` is set. In case where neither
        logging is enabled, `log_on_iteration` will be set to true with an
        interval of :code:`nlive`.
    resume_file : str, optional
        If specified sampler will be resumed from this file. Still requires
        correct model.
    seed : int, optional
        seed for the initialisation of the pseudorandom chain
    n_pool : int, optional
        Number of threads to when for creating the multiprocessing pool.
    pool : object
        User defined multiprocessing pool that will be used when evaluating
        the likelihood.
    close_pool : bool
        Boolean to indicated if the pool should be closed at the end of the
        nested sampling loop. If False, the user must manually close the pool.
    plot : bool (True)
        Boolean to toggle plotting
    proposal_plots : bool (True)
        Boolean to enable additional plots for the population stage of the
        sampler. Overwritten by plot.
    prior_sampling : bool (False)
        produce nlive samples from the prior.
    analytic_priors : bool (False)
        Boolean that indicates that the `new_point` method in the model
        draws directly from the priors meaning rejection sampling is not
        needed.
    maximum_uninformed : int (1000)
        Maximum number of iterations before forcing the sampler to switch to
        using the proposal method with the flow.
    uninformed_proposal : :obj:`nessai.proposal.Proposal`: (None)
        Class to use for initial sampling before training the flow. If
        None RejectionProposal or AnalyticProposal are used depending if
        `analytic_priors` is False or True.
    uninformed_acceptance_threshold : float (None)
        Acceptance threshold for initialising sampling, if acceptance falls
        below this value sampler switches to flow-based proposal. If None
        then value is set to 10 times `acceptance_threshold`
    uninformed_proposal_kwargs : dict, ({})
        Dictionary of keyword argument to pass to the class use for
        the initial sampling when it is initialised.
    flow_class : :obj:`nessai.proposal.FlowProposal`
        Class to use for flow-based proposal method
    flow_config : dict ({})
        Dictionary used to configure instance of `nessai.flowmodel.FlowModel`,
        this includes configuring the normalising flow and the training.
    training_frequency : int (None)
        Number of iterations between re-training the flow. If None flow
        is only re-trained based on other criteria.
    train_on_empty : bool (True)
        If true the flow is retrained every time the proposal pool is
        empty. If false it is only training according to the other criteria.
    cooldown : int (100)
        Minimum number of iterations between training. Can be overridden if
        `train_on_empty=True` and the pool is empty.
    memory : int, False (False)
        Number of old live points to use in training. If False only the current
        live points are used.
    reset_weights : bool, int, (False)
        Boolean to toggle resetting the flow weights whenever re-training.
        If an integer is specified the flow is reset every nth time it is
        trained.
    reset_permutations: bool, int, (False)
        Boolean to toggle resetting the permutation layers in the flow whenever
        re-training. If an integer is specified the flow is reset every nth
        time it is trained.
    reset_flow : bool, int
        Boolean to toggle resetting the entire flow whenever re-training. If
        an integer is specified the flow is reset every nth time it trains.
        This option overrides :code:`reset_weights` and
        :code:`reset_permutations`.
    reset_acceptance : bool, (True)
        If true use mean acceptance of samples produced with current flow
        as a criteria for retraining
    retrain_acceptance : bool (False)
        Force the flow to be reset if the acceptance falls below the acceptance
        threshold. Requires `reset_acceptance=True`
    acceptance_threshold : float (0.01)
        Threshold to determine if the flow should be retrained, will not
        retrain if cooldown is not satisfied.
    shrinkage_expectation : str, {"t", "logt"}
        Method used to compute the expectation value for the shrinkage t.
        Choose between log <t> or <log t>. Defaults to <log t>.
    trace_parameters : Optional[list[str]]
        List of parameters to include in the trace plot. If None, all model
        parameters are included.
    retrain_decision : bool or dict, optional
        If true (the default, or a dictionary of keyword arguments for
        :py:class:`nessai.samplers.retrain.RetrainDecision`), the flow is
        only retrained when the pool is empty and the cost-based decision
        model predicts that retraining reduces the run time. This replaces
        :code:`training_frequency`, :code:`cooldown`,
        :code:`retrain_acceptance` and :code:`reset_acceptance`. Set to
        false to use those criteria instead.
    retrain_costs : dict or str, optional
        Unit costs used by the retrain decision (see
        :py:class:`nessai.samplers.retrain.RetrainCostModel`). Defaults to
        the :code:`'gw'` preset, which makes the training schedule
        deterministic. Can be a dictionary, the name of a preset, the path
        to a JSON file (e.g. :code:`retrain_costs.json` written by a
        previous run, or by :code:`python -m
        nessai.samplers.retrain_benchmark`), :code:`'benchmark'` to measure
        the costs on this hardware before the run, or None to measure them
        during the run. The last two make the schedule hardware-dependent.
    partition : dict, optional
        Partitioned nested sampling (see :py:meth:`configure_partition`):
        once the live points separate along the given coordinates, the prior
        is split into frozen regions, each continued as an independent run
        with its own live points, proposal and stopping, in lockstep with
        the others.
    kwargs :
        Keyword arguments passed to the flow proposal class
    """

    def __init__(
        self,
        model,
        nlive=2000,
        output=None,
        stopping=0.1,
        max_iteration=None,
        checkpointing=True,
        checkpoint_interval=600,
        checkpoint_on_iteration=False,
        checkpoint_on_training=False,
        checkpoint_callback=None,
        logging_interval=None,
        log_on_iteration=True,
        resume_file=None,
        seed=None,
        rng=None,
        pool=None,
        close_pool=False,
        n_pool=None,
        plot=True,
        proposal_plots=False,
        prior_sampling=False,
        analytic_priors=False,
        maximum_uninformed=None,
        uninformed_proposal=None,
        uninformed_acceptance_threshold=None,
        uninformed_proposal_kwargs=None,
        flow_proposal_class=None,
        flow_config=None,
        training_frequency=None,
        train_on_empty=True,
        cooldown=200,
        memory=False,
        reset_weights=False,
        reset_permutations=False,
        reset_flow=False,
        retrain_acceptance=True,
        reset_acceptance=False,
        acceptance_threshold=0.01,
        shrinkage_expectation="logt",
        trace_parameters=None,
        retrain_decision=True,
        retrain_costs="gw",
        partition=None,
        **kwargs,
    ):
        super().__init__(
            model,
            nlive,
            output=output,
            seed=seed,
            rng=rng,
            checkpointing=checkpointing,
            checkpoint_interval=checkpoint_interval,
            checkpoint_on_iteration=checkpoint_on_iteration,
            checkpoint_callback=checkpoint_callback,
            logging_interval=logging_interval,
            log_on_iteration=log_on_iteration,
            resume_file=resume_file,
            plot=plot,
            n_pool=n_pool,
            pool=pool,
        )

        self._close_pool = close_pool

        self.live_points = None
        self.prior_sampling = prior_sampling
        self.accepted = 0
        self.rejected = 1
        self.initialised = False

        self.checkpoint_on_training = checkpoint_on_training
        self.iteration = 0
        self.acceptance_history = deque(maxlen=(nlive // 10))
        self.block_acceptance = 1.0
        self.mean_block_acceptance = 1.0
        self.block_iteration = 0
        self.retrain_acceptance = retrain_acceptance
        self.reset_acceptance = reset_acceptance
        self.trace_parameters = (
            trace_parameters if trace_parameters is not None else model.names
        )

        self.insertion_indices = []
        self.rolling_p = []
        self.final_p_value = None
        self.final_ks_statistic = None

        self.resumed = False
        self.condition = np.inf
        self.logLmin = -np.inf
        self.logLmax = -np.inf
        self.nested_samples = []
        self.logZ = None
        self.state = _NSIntegralState(
            self.nlive,
            track_gradients=plot,
            expectation=shrinkage_expectation,
        )

        self.stopping_criterion = StoppingCriterionRegistry.get(
            "dlogZ", tolerance=stopping
        )

        # Timing
        self.training_time = datetime.timedelta()

        # Resume flags
        self.completed_training = True

        self.acceptance_threshold = acceptance_threshold

        self.train_on_empty = train_on_empty
        self.cooldown = cooldown
        self.memory = memory

        self.configure_max_iteration(max_iteration)
        self.configure_flow_reset(
            reset_weights, reset_permutations, reset_flow
        )
        self.configure_training_frequency(training_frequency)
        self.configure_retrain_decision(retrain_decision, retrain_costs)
        self.configure_partition(partition)

        if uninformed_proposal_kwargs is None:
            uninformed_proposal_kwargs = {}
        self.configure_uninformed_proposal(
            uninformed_proposal,
            analytic_priors,
            maximum_uninformed,
            uninformed_acceptance_threshold,
            **uninformed_proposal_kwargs,
        )

        self.configure_flow_proposal(
            flow_proposal_class, flow_config, proposal_plots, **kwargs
        )

        # Uninformed proposal is used for prior sampling
        # If maximum uninformed is greater than 0, the it will be used for
        # another n iterations or until it becomes inefficient

    @property
    def log_evidence(self):
        return self.state.logZ

    @property
    def log_evidence_error(self):
        return self.state.log_evidence_error

    @property
    def information(self):
        return self.state.info[-1]

    @property
    def proposal_population_time(self):
        t = self._uninformed_proposal.population_time
        t += self._flow_proposal.population_time
        # the proposals of regions that were split again
        split = vars(self).get("_split_population_time")
        if split is not None:
            t += split
        for region in getattr(self, "regions", None) or []:
            if region.proposal is not None:
                t += region.proposal.population_time
            else:
                t += getattr(region, "population_time", datetime.timedelta())
        return t

    @property
    def acceptance(self):
        return self.iteration / self.likelihood_calls

    @property
    def last_updated(self):
        """Last time the normalising flow was retrained"""
        if self.history and self.history["training_iterations"]:
            return self.history["training_iterations"][-1]
        else:
            return 0

    @property
    def mean_acceptance(self):
        """
        Mean acceptance of the last nlive // 10 points
        """
        if self.acceptance_history:
            return np.mean(self.acceptance_history)
        else:
            return np.nan

    @property
    def posterior_effective_sample_size(self):
        """The effective sample size of the posterior distribution"""
        return self.state.effective_n_posterior_samples

    @property
    def birth_log_likelihoods(self):
        """Return the birth log-likelihoods for the nested samples"""
        # the thresholds in the order of the iterations (a partitioned run's
        # ``logLs`` are sorted at the end)
        threshold = getattr(self.state, "threshold_logLs", None)
        logLs = np.array(
            threshold if isinstance(threshold, list) else self.state.logLs
        )
        its = np.array(self.nested_samples)["it"]
        return logLs[its].flatten()

    @property
    def tolerance(self):
        """Return the stopping criterion tolerance"""
        return self.stopping_criterion.tolerance

    def configure_max_iteration(self, max_iteration):
        """Configure the maximum iteration.

        If None then no maximum is set.

        Parameters
        ----------
        max_iteration : int, None
            Maximum iteration.
        """
        if max_iteration is None:
            self.max_iteration = np.inf
        else:
            self.max_iteration = max_iteration

    def configure_training_frequency(self, training_frequency):
        """Configure the training frequency.

        If None, 'inf' or 'None' flow will only train when empty.
        """
        if training_frequency in [None, "inf", "None"]:
            logger.debug("Proposal will only train when empty")
            self.training_frequency = np.inf
        else:
            self.training_frequency = training_frequency

    def configure_retrain_decision(self, retrain_decision, retrain_costs):
        """Configure the cost-based retrain decision."""
        if not retrain_decision:
            self.retrain_decision = None
            return
        kwargs = (
            retrain_decision if isinstance(retrain_decision, dict) else {}
        )
        self._retrain_benchmark = retrain_costs == "benchmark"
        self._retrain_decision_kwargs = dict(kwargs)
        self.retrain_decision = RetrainDecision(
            self.nlive,
            costs=None if self._retrain_benchmark else retrain_costs,
            **kwargs,
        )
        self._retrain_snapshot = None

    def benchmark_retrain_costs(self):
        """Measure the retrain unit costs with the run's configuration."""
        from .retrain_benchmark import measure_retrain_costs

        ProposalClass, flow_config, kwargs = self._flow_proposal_config
        filename = None
        if self.output:
            filename = os.path.join(self.output, "retrain_costs_benchmark.json")
        logger.info("Measuring the retrain unit costs on this hardware")
        costs = measure_retrain_costs(
            self.model,
            nlive=self.nlive,
            flow_config=flow_config,
            flow_proposal_class=ProposalClass,
            filename=filename,
            **kwargs,
        )
        self.retrain_decision.cost.provided.update(costs)
        self._retrain_benchmark = False

    def _retrain_time_snapshot(self):
        return (
            (self.current_sampling_time - self.training_time).total_seconds(),
            self.likelihood_evaluation_time.total_seconds(),
            self.total_likelihood_evaluations,
        )

    def _record_retrain_populations(self):
        """Record the cost of the pool that has just been used."""
        snap = self._retrain_time_snapshot()
        if self._retrain_snapshot is not None and self.proposal.samples is not None:
            dt, dl, dn = (a - b for a, b in zip(snap, self._retrain_snapshot))
            self.retrain_decision.record_populations(
                dt, self.proposal.samples.size, dl, dn
            )
        self._retrain_snapshot = snap

    def _next_poolsize(self):
        """Size of the next pool if drawn with the current flow."""
        proposal = self.proposal
        if not getattr(proposal, "update_poolsize", False):
            return proposal.poolsize
        acc = self.mean_block_acceptance
        scale = 1.0 / acc if acc > 0 else proposal.max_poolsize_scale
        scale = min(max(scale, 1.0), proposal.max_poolsize_scale)
        return int(scale * proposal._poolsize)

    def _remaining_iterations(self):
        """Rough estimate of the number of iterations left."""
        tol = self.tolerance
        if not np.isfinite(self.condition) or tol <= 0:
            return np.inf
        if self.condition <= tol:
            return 0.0
        # log(expm1(c)) = c + log(-expm1(-c)), stable for large c
        c = self.condition
        return self.nlive * (
            c + np.log(-np.expm1(-c)) - np.log(np.expm1(tol))
        )

    def configure_uninformed_proposal(
        self,
        uninformed_proposal,
        analytic_priors,
        maximum_uninformed,
        uninformed_acceptance_threshold,
        **kwargs,
    ):
        """
        Setup the uninformed proposal method (is NOT trained)

        Parameters
        ----------
        uninformed_proposal : None or obj
            Class to use for uninformed proposal
        analytic_priors : bool
            If True `AnalyticProposal` is used to directly sample from the
            priors rather than using rejection sampling.
        maximum_uninformed : {False, None, int, float}
            Maximum number of iterations before switching to FlowProposal.
            If None, two times nlive is used. If False uninformed sampling is
            not used.
        uninformed_acceptance_threshold :  float or None:
            Threshold to use for uninformed proposal, once reached proposal
            method will switch. If None acceptance_threshold is used if
            greater than 0.1 else 10 x acceptance_threshold is used.
        kwargs
            Kwargs are passed to init method for uninformed proposal class
        """
        if maximum_uninformed is None:
            self.uninformed_sampling = True
            self.maximum_uninformed = 2 * self.nlive
        elif not maximum_uninformed:
            self.uninformed_sampling = False
            self.maximum_uninformed = 0
        else:
            self.uninformed_sampling = True
            self.maximum_uninformed = float(maximum_uninformed)

        if uninformed_acceptance_threshold is None:
            if self.acceptance_threshold < 0.1:
                self.uninformed_acceptance_threshold = (
                    10 * self.acceptance_threshold
                )
            else:
                self.uninformed_acceptance_threshold = (
                    self.acceptance_threshold
                )
        else:
            self.uninformed_acceptance_threshold = (
                uninformed_acceptance_threshold
            )

        if uninformed_proposal is None:
            if analytic_priors:
                from ..proposal import AnalyticProposal as uninformed_proposal
            else:
                from ..proposal import RejectionProposal as uninformed_proposal

                kwargs["poolsize"] = self.nlive

        logger.debug(f"Using uninformed proposal: {uninformed_proposal}")
        logger.debug(f"Parsing kwargs to uninformed proposal: {kwargs}")
        self._uninformed_proposal = uninformed_proposal(
            self.model, rng=self.rng, **kwargs
        )

    def configure_flow_proposal(
        self, flow_proposal_class, flow_config, proposal_plots, **kwargs
    ):
        """
        Set up the flow-based proposal method

        Parameters
        ----------
        flow_proposal_class : None or Callable or str
            Class to use for proposal. If None FlowProposal is used.
        flow_config : dict
            Configuration dictionary passed to the class.
        proposal_plots : bool or str
            Configuration of plotting in proposal class.
        **kwargs :
            Kwargs passed to init function.
        """
        proposal_output = os.path.join(self.output, "proposal", "")

        if not self.plot:
            proposal_plots = False

        ProposalClass = get_flow_proposal_class(flow_proposal_class)

        if kwargs.get("poolsize", None) is None:
            kwargs["poolsize"] = self.nlive

        logger.debug(f"Using flow class: {ProposalClass}")

        kwargs = check_proposal_kwargs(ProposalClass, kwargs)

        logger.info(f"Passing kwargs to {ProposalClass.__name__}: {kwargs}")
        self._flow_proposal_config = (
            ProposalClass,
            flow_config,
            dict(kwargs),
        )
        self._flow_proposal = ProposalClass(
            self.model,
            rng=self.rng,
            flow_config=flow_config,
            output=proposal_output,
            plot=proposal_plots,
            **kwargs,
        )

    def configure_output(
        self, output: Union[str, None], resume_file: Union[str, None] = None
    ):
        """Configure the output folder

        Creates a "diagnostics" directory plotting is enabled.

        Parameters
        ----------
        output : str
            Directory where the results will be stored
        resume_file : str, optional
            Specific file to use for checkpointing. If not specified the
            default is used (nested_sampler_resume.pkl)
        """
        super().configure_output(output, resume_file)
        if self.plot:
            os.makedirs(os.path.join(output, "diagnostics"), exist_ok=True)

    def update_output(self, output: str) -> None:
        """Update the output directory.

        Also creates a "diagnostics" directory for plotting.

        Parameters
        ----------
        output : str
            Path to the output directory.
        """
        super().update_output(output)
        if self.plot:
            os.makedirs(os.path.join(output, "diagnostics"), exist_ok=True)
        if self._flow_proposal is not None:
            # FlowProposal uses a subdirectory
            subdir = os.path.basename(
                os.path.normpath(self._flow_proposal.output)
            )
            self._flow_proposal.update_output(os.path.join(output, subdir, ""))
        for region in getattr(self, "regions", None) or []:
            if region.proposal is not None:
                region.proposal.update_output(
                    os.path.join(
                        output, "proposal", f"region_{region.label}", ""
                    )
                )

    def configure_flow_reset(
        self, reset_weights, reset_permutations, reset_flow
    ):
        """Configure how often the flow parameters are reset.

        Values are converted to floats.

        Parameters
        ----------
        reset_weights : int, float or bool
            Frequency with which the weights will be reset.
        reset_permutations : int, float or bool
            Frequency with which the permutations will be reset.
        reset_flow : int, float or bool
            Frequency with which the entire flow will be reset. Will overwrite
            the value for the other resets.
        """
        if isinstance(reset_weights, (int, float)):
            self.reset_weights = float(reset_weights)
        else:
            raise TypeError("`reset_weights` must be a bool, int or float")
        if isinstance(reset_permutations, (int, float)):
            self.reset_permutations = float(reset_permutations)
        else:
            raise TypeError(
                "`reset_permutations` must be a bool, int or float"
            )
        if isinstance(reset_flow, (int, float)):
            self.reset_flow = float(reset_flow)
        else:
            raise TypeError("`reset_flow` must be a bool, int or float")

        if self.reset_flow:
            self.reset_weights = self.reset_flow
            self.reset_permutations = self.reset_flow

    def initialise_history(self):
        if not self.history:
            super().initialise_history()
            self.history.update(
                dict(
                    iterations=[],
                    min_log_likelihood=[],
                    max_log_likelihood=[],
                    logZ=[],
                    dlogZ=[],
                    mean_acceptance=[],
                    training_iterations=[],
                    population_acceptance=[],
                    population_radii=[],
                    population_iterations=[],
                    rolling_p=[],
                )
            )

    def update_history(self):
        super().update_history()
        self.history["iterations"].append(self.iteration)
        self.history["min_log_likelihood"].append(self.logLmin)
        self.history["max_log_likelihood"].append(self.logLmax)
        self.history["logZ"].append(self.state.logZ)
        self.history["dlogZ"].append(self.condition)
        self.history["mean_acceptance"].append(self.mean_acceptance)

    def log_state(self):
        """Log the current state of the sampler"""
        logger.info(
            f"it: {self.iteration:5d}: "
            f"n eval: {self.likelihood_calls} "
            f"H: {self.state.info[-1]:.2f} "
            f"dlogZ: {self.condition:.3f} logZ: {self.state.logZ:.3f} "
            f"+/- {self.state.log_evidence_error:.3f} "
            f"logLmax: {self.logLmax:.2f}"
        )
        for r in self.active_regions:
            logger.info(
                f"  region {r.label}: nlive {r.nlive}, ln X "
                f"{self.state.region_logw[r.label]:.3f}, logL "
                f"{r.live_points['logL'][0]:.2f} - "
                f"{r.live_points['logL'][-1]:.2f}, iterations "
                f"{r.iterations}, acceptance {r.mean_acceptance:.3g}, "
                f"trainings {len(r.training_iterations)}"
            )

    def check_insertion_indices(self, rolling=True, filename=None):
        """
        Checking the distribution of the insertion indices either during
        the nested sampling run (rolling=True) or for the whole run
        (rolling=False).
        """
        if getattr(self, "regions", None) is not None:
            return self._check_region_insertion_indices(rolling, filename)
        if rolling:
            indices = self.insertion_indices[-self.nlive :]
        else:
            indices = self.insertion_indices

        D, p = compute_indices_ks_test(indices, self.nlive)

        if p is not None:
            if rolling:
                logger.info(
                    f"it: {self.iteration:5d}: "
                    f"Rolling KS test: D={D:.4}, p-value={p:.4}"
                )
                self.history["rolling_p"].append(p)
            else:
                logger.info(f"Final KS test: D={D:.4}, p-value={p:.4}")
                self.final_p_value = p
                self.final_ks_statistic = D
                pt = 0.05
                if self.final_p_value < pt:
                    logger.warning(
                        f"Final p-value for the insertion indices is less "
                        f"than {pt}, this could be an indication of problems "
                        "during sampling. Consider checking the diagnostic "
                        "plots."
                    )

        if filename is not None:
            np.savetxt(
                os.path.join(self.output, filename),
                self.insertion_indices,
                newline="\n",
                delimiter=" ",
            )

    def _check_region_insertion_indices(self, rolling=True, filename=None):
        """The insertion-index KS test of each region (its own indices and
        number of live points); the run's p-value is the smallest."""
        regions = self.active_regions if rolling else self.regions
        results = []
        for r in regions:
            indices = r.insertion_indices
            if rolling:
                indices = indices[-r.nlive :]
            D, p = compute_indices_ks_test(indices, r.nlive)
            if p is None:
                continue
            if not rolling:
                r.final_p_value = p
            results.append((p, D, r.label))
            logger.info(
                f"it: {self.iteration:5d}: region {r.label} "
                f"{'rolling' if rolling else 'final'} KS test: "
                f"D={D:.4}, p-value={p:.4}"
            )
        if results:
            p, D, _ = min(results)
            if rolling:
                self.history["rolling_p"].append(p)
            else:
                self.final_p_value = p
                self.final_ks_statistic = D
        if filename is not None:
            for r in self.regions:
                np.savetxt(
                    os.path.join(
                        self.output,
                        filename.replace(".", f"_region_{r.label}.", 1),
                    ),
                    r.insertion_indices,
                    newline="\n",
                    delimiter=" ",
                )

    def yield_sample(self, oldparam):
        """
        Draw points and applying rejection sampling
        """
        while True:
            counter = 0
            while True:
                counter += 1
                if hasattr(self.proposal, "current_live_points"):
                    # the flow proposal's truncation thresholds can be taken
                    # from the current live points (a reference: the live
                    # points are updated in place)
                    self.proposal.current_live_points = getattr(
                        self, "live_points", None
                    )
                newparam = self.proposal.draw(copy(oldparam))

                # Prior is computed in the proposal
                if newparam["logP"] != -np.inf:
                    if not newparam["logL"]:
                        newparam["logL"] = self.model.evaluate_log_likelihood(
                            newparam
                        )
                    if newparam["logL"] > self.logLmin:
                        self.logLmax = max(self.logLmax, newparam["logL"])
                        oldparam = newparam.copy()
                        break
                # Only here if proposed and then empty
                # This returns the old point and allows for a training check
                if not self.proposal.populated:
                    break
            yield counter, oldparam

    def insert_live_point(self, live_point):
        """
        Insert a live point
        """
        # This is the index including the current worst point, so final index
        # is one less, otherwise index=0 would never be possible
        index = np.searchsorted(self.live_points["logL"], live_point["logL"])
        self.live_points[: index - 1] = self.live_points[1:index]
        self.live_points[index - 1] = live_point
        return index - 1

    def consume_sample(self):
        """
        Replace a sample for single thread
        """
        worst = self.live_points[0].copy()
        self.logLmin = worst["logL"]
        self.state.increment(worst["logL"])
        self.nested_samples.append(worst)

        self.condition = (
            np.logaddexp(
                self.state.logZ,
                self.logLmax - self.iteration / float(self.nlive),
            )
            - self.state.logZ
        )

        # Replace the points we just consumed with the next acceptable ones
        # Make sure we are mixing the chains
        self.iteration += 1
        self.block_iteration += 1
        count = 0

        while True:
            c, proposed = next(self.yield_sample(worst))
            count += c

            if proposed["logL"] > self.logLmin:
                # Assuming point was proposed
                # replace worst point with new one
                proposed["it"] = self.iteration
                index = self.insert_live_point(proposed)
                self.insertion_indices.append(index)
                self.accepted += 1
                self.block_acceptance += 1 / count
                self.acceptance_history.append(1 / count)
                if (
                    self.retrain_decision is not None
                    and self.proposal is self._flow_proposal
                ):
                    self.retrain_decision.record_iteration(count)
                break
            else:
                # Only get here if the yield sample returns worse point
                # which can only happen if the pool is empty
                self.rejected += 1
                self.check_state()
                # if retrained whilst proposing a sample then update the
                # iteration count since will be zero otherwise
                if not self.block_iteration:
                    self.block_iteration += 1

        self.mean_block_acceptance = (
            self.block_acceptance / self.block_iteration
        )

        if self.debug_enabled and self.log_on_iteration:
            logger.debug(
                f"{self.iteration:5d}: n: {count:3d} "
                f"b_acc: {self.mean_block_acceptance:.3f} "
                f"H: {self.state.info[-1]:.2f} "
                f"logL: {self.logLmin:.5f} --> {proposed['logL']:.5f} "
                f"dlogZ: {self.condition:.3f} "
                f"logZ: {self.state.logZ:.3f} "
                f"+/- {self.state.log_evidence_error:.3f} "
                f"logLmax: {self.logLmax:.2f}"
            )

    def populate_live_points(self):
        """
        Initialise the pool of live points.
        """
        accepted = 0
        live_points = empty_structured_array(
            self.nlive, names=self.model.names
        )

        logger.info("Populating initial live points")
        while accepted < self.nlive:
            _, live_point = next(self.yield_sample(None))
            if live_point is None:
                continue
            if np.isnan(live_point["logL"]):
                logger.error(
                    "Likelihood function returned NaN for "
                    f"live_point {live_point}"
                )
                logger.error("You may want to check your likelihood function")
                continue
            if np.isfinite(live_point["logP"]) and np.isfinite(
                live_point["logL"]
            ):
                live_points[accepted] = live_point
                accepted += 1
            # Round up to avoid issues in nlive < 10
            if accepted % math.ceil(self.nlive / 10) == 0:
                logger.info(f"Populated {accepted} / {self.nlive} live points")

        self.live_points = np.sort(live_points, order="logL")

        _, logl_counts = np.unique(
            self.live_points["logL"], return_counts=True
        )
        if np.any(logl_counts > 1):
            logger.warning(
                "Initial live points contain repeated log-likelihood values! "
                "This will likely lead to issues in the sampling process. "
                "See the FAQs for more information."
            )
        self.live_points["it"] = 0

    def initialise(self, live_points=True):
        """
        Initialise the nested sampler

        Parameters
        ----------
        live_points : bool, optional (True)
            If true and there are no live points, new live points are
            drawn using `populate_live_points` else all other initialisation
            steps are complete but live points remain empty.
        """
        flags = [False] * 3
        if getattr(self, "_retrain_benchmark", False):
            self.benchmark_retrain_costs()
        if not self._flow_proposal.initialised:
            self._flow_proposal.initialise()
            flags[0] = True

        if not self._uninformed_proposal.initialised:
            self._uninformed_proposal.initialise()
            flags[1] = True

        if (
            self.iteration < self.maximum_uninformed
            and self.uninformed_sampling
        ):
            self.proposal = self._uninformed_proposal
        else:
            self.proposal = self._flow_proposal

        if live_points and self.live_points is None and not self.finalised:
            self.populate_live_points()
            flags[2] = True

        if not self.stopping_criterion.is_met(self.condition):
            self.finalised = False

        self.initialise_history()

        if all(flags):
            self.initialised = True

    def check_proposal_switch(self, force=False):
        """
        Check if the proposal should be switch from uninformed to
        flowproposal given the current state.

        If the flow proposal is already in use, no changes are made.

        Parameters
        ----------
        force : bool, optional
            If True proposal is forced to switch.

        Returns
        -------
        bool
            Flag to indicated if proposal was switched
        """
        if (
            (self.mean_acceptance < self.uninformed_acceptance_threshold)
            or (self.iteration >= self.maximum_uninformed)
            or force
        ):
            if self.proposal is self._flow_proposal:
                logger.info("Already using flowproposal")
                self.uninformed_sampling = False
                return True
            logger.info("Switching to FlowProposal")
            self.proposal = self._flow_proposal
            self.proposal.ns_acceptance = self.mean_block_acceptance
            self.uninformed_sampling = False
            return True
        # If using uninformed sampling, don't check training
        else:
            return False

    # ------------------------------------------------------------------
    # Partitioned nested sampling
    # ------------------------------------------------------------------
    _partition_defaults = dict(
        coordinates=None,
        splitter=None,
        n_min=None,
        max_regions=4,
        check_every=None,
        plot_every=None,
        plot=True,
    )

    def configure_partition(self, partition):
        """Configure partitioned nested sampling.

        Every ``check_every`` iterations (in each region, counted in its own
        iterations) once its flow is trained, the splitter looks at the live points
        with hold-out bit 0 (:func:`nessai.partition.hold_out_bit`) along the
        split coordinates. When it proposes a cut, the region (at first the
        whole prior) is split into frozen parts: each part gets the share of
        its volume that the live points with bit 1 estimate, keeps its live
        points (a cut is used only if every part has at least ``n_min``) and
        its own copy of the proposal, restricted to the part
        (:class:`nessai.partition.RegionTruncation`) and trained on its
        points. The regions then run in lockstep: the lowest live point of
        all is killed, only its region's volume shrinks (by one over that
        region's number of live points) and it is replaced from its region's
        proposal. A region stops (its live points are added as for the end
        of a run) once it could add less than the stopping tolerance to the
        evidence; the run ends when every region has stopped. The dead points
        are merged in likelihood order.

        Parameters
        ----------
        partition : dict or None
            None or False disables it. Keys:

            * ``coordinates`` (required): a list of parameter names, or a
              :class:`nessai.partition.SplitCoordinates` (an instance, a
              ``"module:Class"`` string, or a dictionary with ``"class"``
              and its keyword arguments);
            * ``splitter``: the splitter, as for ``coordinates`` (default
              :class:`nessai.partition.GapSplitter`, which takes one
              coordinate);
            * ``n_min`` (``nlive // 5``): minimum number of live points of a
              region; a cut that leaves fewer in a part is not used (a
              region with few points trains a poor flow and draws slowly);
            * ``max_regions`` (4): maximum number of active regions;
            * ``check_every`` (``nlive // 20`` of the region): iterations
              between checks;
            * ``plot_every`` (``nlive``): iterations between the diagnostic
              plots (``diagnostics/partition.png``), also made at every
              split;
            * ``plot`` (True): make them.
        """
        if not hasattr(self, "regions"):
            self.regions = None
            self.partitions = []
            self.retired_samples = []
            self.retired_regions = []
            self.nested_regions = []
            self._partition_start = None
            # the first check at the first chance (e.g. a run resumed with
            # the partition switched on)
            self._partition_last_check = -np.inf
            self._next_region_label = 1
            self._partition_last_cut = None
            self._checkpoint_pending = False
            self._splitting = False
        if not partition:
            self.partition = None
            return
        if not isinstance(partition, dict):
            raise TypeError(
                "partition must be a dictionary with at least 'coordinates'"
            )
        unknown = set(partition) - set(self._partition_defaults)
        if unknown:
            raise ValueError(f"Unknown partition options: {unknown}")
        config = dict(self._partition_defaults)
        config.update(partition)
        coordinates = config["coordinates"]
        if coordinates is None:
            raise ValueError("partition needs the split 'coordinates'")
        if isinstance(coordinates, (list, tuple)) and all(
            isinstance(c, str) for c in coordinates
        ):
            coordinates = partition_module.ParameterCoordinates(coordinates)
        else:
            coordinates = partition_module.build(
                coordinates,
                partition_module.ParameterCoordinates,
                partition_module,
            )
        coordinates.configure(self.model)
        splitter = partition_module.build(
            config["splitter"], partition_module.GapSplitter, partition_module
        )
        if config["n_min"] is None:
            config["n_min"] = max(1, self.nlive // 5)
        if config["plot_every"] is None:
            config["plot_every"] = self.nlive
        if config["max_regions"] < 2:
            raise ValueError("partition max_regions must be at least 2")
        self.partition = config
        self._partition_coordinates = coordinates
        self._partition_splitter = splitter
        logger.info(
            f"Partition: coordinates {list(coordinates.names)}, splitter "
            f"{type(splitter).__name__}, n_min {config['n_min']}, max "
            f"regions {config['max_regions']}"
        )

    @property
    def active_regions(self):
        """The regions still running (empty before the first split)."""
        if getattr(self, "regions", None) is None:
            return []
        return [r for r in self.regions if r.active]

    def _root_region(self):
        """The unsplit run as the region every first split starts from."""
        root = partition_module.Region(
            0,
            None,
            partition_module.RegionMembership(),
            self.nlive,
            self.live_points,
            proposal=self._flow_proposal,
            retrain_decision=self.retrain_decision,
        )
        root.iterations = self.iteration
        return root

    def _region_ready(self, proposal, completed_training):
        """A trained flow, not in the middle of a population. The checks
        are made between iterations, at most every ``check_every``; a split
        discards the region's pool (once)."""
        return bool(
            proposal is not None
            and completed_training
            and getattr(proposal, "training_count", 0)
            and not getattr(proposal, "populating", False)
        )

    def _maybe_partition(self):
        """Check the unsplit run, or each active region, for a split (see
        :meth:`configure_partition`). Returns True if a split was made."""
        config = self.partition
        if self.regions is None:
            if (
                self.uninformed_sampling
                or self.proposal is not self._flow_proposal
                or not self._region_ready(
                    self._flow_proposal, self.completed_training
                )
            ):
                return False
            every = config["check_every"] or max(1, self.nlive // 20)
            if self.iteration - self._partition_last_check < every:
                return False
            self._partition_last_check = self.iteration
            return self._try_split(self._root_region())
        for region in self.active_regions:
            if not self._region_ready(
                region.proposal, region.completed_training
            ):
                continue
            every = config["check_every"] or max(1, region.nlive // 20)
            if region.iterations - region.last_check < every:
                continue
            region.last_check = region.iterations
            if self._try_split(region):
                return True
        return False

    def _try_split(self, parent):
        n_active = len(self.active_regions) if self.regions else 1
        max_labels = self.partition["max_regions"] - n_active + 1
        if max_labels < 2:
            return False
        live = parent.live_points
        # the check changes nothing, so a failure in it (e.g. in a
        # splitter's frame) leaves the run unsplit rather than stopping it
        try:
            coordinates = self._partition_coordinates.freeze(parent.proposal)
            coords = coordinates(live)
            bit = partition_module.hold_out_bit(live)
            sel = bit == 0
            cut = self._partition_splitter.propose(
                coords[sel],
                coordinates.period_list(),
                points=live[sel],
                proposal=parent.proposal,
                max_labels=max_labels,
            )
        except Exception as e:
            logger.warning(
                f"Partition check at iteration {self.iteration} (region "
                f"{parent.label}) failed, no split: {e!r}"
            )
            return False
        logger.debug(
            f"Partition check at iteration {self.iteration} (region "
            f"{parent.label}): "
            + ("no cut" if cut is None else f"{cut.describe()} {cut.info}")
        )
        if cut is None:
            return False
        return self._split(parent, coordinates, cut, coords, bit)

    def _split(self, parent, coordinates, cut, coords, bit):
        """Split ``parent`` by ``cut`` (see :meth:`configure_partition`)."""
        labels = cut.classify(coords)
        n_labels = cut.n_labels
        held = labels[bit == 1]
        counts = np.bincount(labels, minlength=n_labels)
        if not len(held):
            return False
        shares = np.bincount(held, minlength=n_labels) / len(held)
        if np.any(shares == 0) or np.any(counts == 0):
            logger.warning(
                f"Partition: cut {cut.describe()} leaves a part without "
                f"held-out live points (counts {counts.tolist()}, shares "
                f"{shares.tolist()}): not splitting"
            )
            return False
        n_min = self.partition["n_min"]
        if counts.min() < n_min:
            # a region with few live points trains a poor flow and draws
            # slowly: keep them with the rest
            logger.debug(
                f"Partition: cut {cut.describe()} leaves a part with fewer "
                f"than {n_min} live points (counts {counts.tolist()}): not "
                "splitting"
            )
            return False
        first = self.regions is None
        if first:
            self.regions = []
            self._partition_start = len(self.nested_samples)
            self.nested_regions = []
        log_x = (
            float(self.state.logw)
            if first
            else float(self.state.region_logw[parent.label])
        )
        child_labels = list(
            range(self._next_region_label, self._next_region_label + n_labels)
        )
        self._next_region_label += n_labels
        self.state.split(parent.label, child_labels, np.log(shares))
        children = []
        for k, label in enumerate(child_labels):
            child = partition_module.Region(
                label,
                parent.label,
                parent.membership.child(coordinates, cut, k),
                max(int(counts[k]), n_min),
                parent.live_points[labels == k].copy(),
                acceptance_length=max(1, self.nlive // 10),
            )
            children.append(child)
            self.partitions.append(
                dict(
                    iteration=self.iteration,
                    logL=float(self.logLmin),
                    parent=parent.label,
                    region=label,
                    share=float(shares[k]),
                    n_points=int(counts[k]),
                    n_held_out=int((held == k).sum()),
                    nlive=child.nlive,
                    log_x=log_x + float(np.log(shares[k])),
                    cut=cut.describe(),
                )
            )
        logger.info(
            f"Partition at iteration {self.iteration} (logL > "
            f"{self.logLmin:.3f}): region {parent.label} (ln X {log_x:.3f}) "
            f"split by {list(coordinates.names)} {cut.describe()} into "
            + ", ".join(
                f"region {c.label} ({counts[k]} live points, share "
                f"{shares[k]:.4f} from {(held == k).sum()} held out, nlive "
                f"{c.nlive})"
                for k, c in enumerate(children)
            )
        )
        if self.memory:
            logger.warning("Partition: the regions train without memory")
        if not first:
            self.regions.remove(parent)
            if parent.proposal is not None:
                self._split_population_time = (
                    getattr(self, "_split_population_time", datetime.timedelta())
                    + parent.proposal.population_time
                )
        parent.active = False
        self.regions.extend(children)
        self._partition_last_cut = (coordinates, cut)
        # a region that cannot hold enough evidence stops at once
        self._retire_regions()
        self._splitting = True
        try:
            for child in children:
                if not child.active:
                    continue
                child.proposal = self._clone_proposal(parent.proposal, child)
                child.retrain_decision = self._child_retrain_decision(
                    parent.retrain_decision, child.nlive
                )
                if len(child.live_points) < child.nlive:
                    # train on the points it has, so the top-up is drawn
                    # from a flow that covers the region
                    self._train_region(child, force=True)
                    self._fill_region(child)
                self._train_region(child, force=True)
        finally:
            self._splitting = False
        if not first:
            parent.proposal = None
        self._update_partition_condition()
        self._merge_live_points()
        self._plot_partition_diagnostic(split=True)
        if self.checkpointing:
            self.checkpoint(periodic=True, force=True)
        return True

    def _clone_proposal(self, source, region):
        """A copy of ``source`` for ``region``: the pickled state (as for a
        checkpoint) resumed with the source's current flow weights, in its
        own output directory, with its own random generators and restricted
        to the region."""
        import torch

        output = os.path.join(
            self.output, "proposal", f"region_{region.label}", ""
        )
        os.makedirs(output, exist_ok=True)
        weights = os.path.join(output, "model.pt")
        torch.save(source.flow.model.state_dict(), weights)
        clone = pickle.loads(pickle.dumps(source))
        clone.output = output
        clone.indices = []
        clone.populated = False
        clone.resume_populated = False
        clone.samples = None
        clone.x = None
        clone.population_time = datetime.timedelta()
        partition_module.respawn_generators(clone, self.rng)
        flow_config = getattr(self, "_resume_flow_config", None)
        if flow_config is None:
            flow_config = self._flow_proposal_config[1]
        clone.resume(self.model, deepcopy(flow_config or {}), weights)
        scheme = clone.truncation
        # a copy of a region's proposal: its region becomes the child's
        scheme.rules = [r for r in scheme.rules if r.name != "region"]
        scheme.add_rule(
            partition_module.RegionTruncation(region.membership, region.label),
            index=0,
        )
        if hasattr(clone, "_sync_truncation_state"):
            clone._sync_truncation_state()
        clone.current_live_points = region.live_points
        partition_module.check_clone(source, clone, region.live_points)
        # e.g. a mixture keeps the expert of the region's side of the cut
        restrict = getattr(clone, "restrict_to_region", None)
        if restrict is not None:
            restrict(region.live_points)
        return clone

    def _child_retrain_decision(self, parent, nlive):
        """A new retrain decision with the parent's unit costs."""
        if parent is None:
            return None
        decision = RetrainDecision(
            nlive, **getattr(self, "_retrain_decision_kwargs", {})
        )
        decision.cost = deepcopy(parent.cost)
        return decision

    def _region_clock(self):
        return self._retrain_time_snapshot()

    def _charge_region(self, region, before):
        """Add the time and likelihood calls since ``before`` to
        ``region``."""
        dt, dl, dn = (a - b for a, b in zip(self._region_clock(), before))
        region.time += dt
        region.likelihood_time += dl
        region.likelihood_evaluations += dn

    def _region_yield(self, region, oldparam):
        """:meth:`yield_sample` for a region."""
        proposal = region.proposal
        while True:
            counter = 0
            while True:
                counter += 1
                proposal.current_live_points = region.live_points
                try:
                    newparam = proposal.draw(copy(oldparam))
                except IndexError:
                    # a pool with no draw inside the region
                    proposal.indices = []
                    proposal.populated = False
                    proposal.populating = False
                    region.empty_pools = getattr(region, "empty_pools", 0) + 1
                    if region.empty_pools >= 50:
                        raise RuntimeError(
                            f"Partition: 50 pools without a draw inside "
                            f"region {region.label}"
                        )
                    break
                if newparam["logP"] != -np.inf:
                    if not newparam["logL"]:
                        newparam["logL"] = self.model.evaluate_log_likelihood(
                            newparam
                        )
                    if newparam["logL"] > self.logLmin:
                        self.logLmax = max(self.logLmax, newparam["logL"])
                        region.empty_pools = 0
                        oldparam = newparam.copy()
                        break
                if not proposal.populated:
                    break
            yield counter, oldparam

    def _replace_in_region(self, region, worst):
        """Draw the replacement of ``worst`` in ``region``, training when
        the pool runs out (as :meth:`consume_sample`)."""
        count = 0
        while True:
            c, proposed = next(self._region_yield(region, worst))
            count += c
            if proposed["logL"] > self.logLmin:
                return count, proposed
            self.rejected += 1
            self._region_check_state(region)
            if not region.block_iteration:
                region.block_iteration += 1

    def _fill_region(self, region):
        """Top ``region`` up to its number of live points with independent
        draws above the current contour (no dead points)."""
        missing = region.nlive - len(region.live_points)
        logger.info(f"Partition: drawing {missing} live points in region "
                    f"{region.label}")
        template = region.live_points[0].copy()
        template["logL"] = self.logLmin
        before = self._region_clock()
        drawn, added = 0, 0
        report = max(1, missing // 5)
        while len(region.live_points) < region.nlive:
            count, proposed = next(self._region_yield(region, template.copy()))
            drawn += count
            if proposed["logL"] > self.logLmin:
                proposed["it"] = self.iteration
                index = np.searchsorted(
                    region.live_points["logL"], proposed["logL"]
                )
                region.live_points = np.insert(
                    region.live_points, index, proposed
                )
                added += 1
                if not added % report:
                    logger.info(
                        f"Partition: region {region.label}: {added} / "
                        f"{missing} drawn (acceptance {added / drawn:.3g})"
                    )
        self._charge_region(region, before)
        region.proposal.current_live_points = region.live_points

    def _partitioned_iteration(self):
        """One iteration of a partitioned run (see
        :meth:`configure_partition`)."""
        region = min(
            self.active_regions, key=lambda r: r.live_points["logL"][0]
        )
        self.proposal = region.proposal
        worst = region.live_points[0].copy()
        self.logLmin = float(worst["logL"])
        self.state.increment(
            self.logLmin, nlive=region.nlive, region=region.label
        )
        self.nested_samples.append(worst)
        self.nested_regions.append(region.label)
        self.iteration += 1
        self.block_iteration += 1
        region.iterations += 1
        region.block_iteration += 1

        before = self._region_clock()
        count, proposed = self._replace_in_region(region, worst)
        self._charge_region(region, before)

        proposed["it"] = self.iteration
        live = region.live_points
        index = np.searchsorted(live["logL"], proposed["logL"])
        live[: index - 1] = live[1:index]
        live[index - 1] = proposed
        region.insertion_indices.append(index - 1)
        self.accepted += 1
        region.block_acceptance += 1 / count
        region.acceptance_history.append(1 / count)
        self.acceptance_history.append(1 / count)
        if region.retrain_decision is not None:
            region.retrain_decision.record_iteration(count)
        region.mean_block_acceptance = (
            region.block_acceptance / region.block_iteration
        )
        self.mean_block_acceptance = region.mean_block_acceptance
        region.proposal.ns_acceptance = region.mean_block_acceptance

        self._retire_regions()
        self._update_partition_condition()
        self._merge_live_points()
        if self._checkpoint_pending:
            self._checkpoint_pending = False
            self.checkpoint(periodic=True, force=True)

    def _region_condition(self, region):
        """``ln(1 + X_r Lmax_r / Z)``: the most the region can still add."""
        log_x = self.state.region_logw[region.label]
        return float(
            np.logaddexp(
                0.0, log_x + region.live_points["logL"][-1] - self.state.logZ
            )
        )

    def _retire_regions(self):
        for region in self.active_regions:
            condition = self._region_condition(region)
            if condition < self.tolerance:
                self._retire_region(region, condition)

    def _retire_region(self, region, condition=None):
        """Stop ``region``: its live points become dead points."""
        live = region.live_points
        log_x = self.state.region_logw[region.label]
        self.state.retire(region.label, live["logL"])
        self.retired_samples.extend(list(live))
        self.retired_regions.extend([region.label] * len(live))
        region.active = False
        if len(region.insertion_indices) >= region.nlive:
            _, region.final_p_value = compute_indices_ks_test(
                region.insertion_indices, region.nlive
            )
        logger.info(
            f"Partition: region {region.label} stops at iteration "
            f"{self.iteration} after {region.iterations} of its own "
            f"(ln X {log_x:.3f}, max logL {live['logL'][-1]:.3f}"
            + (f", dlogZ {condition:.3g}" if condition is not None else "")
            + f"); {len(self.active_regions)} region(s) left"
        )
        region.live_points = live[:0]
        if region.proposal is not None:
            region.population_time = region.proposal.population_time
        region.proposal = None

    def _update_partition_condition(self):
        terms = [
            self.state.region_logw[r.label] + r.live_points["logL"][-1]
            for r in self.active_regions
        ]
        if terms:
            self.condition = float(
                np.logaddexp(self.state.logZ, logsumexp(terms))
                - self.state.logZ
            )
        else:
            self.condition = 0.0

    def _merge_live_points(self):
        """The active regions' live points as one array sorted by logL, for
        plots and anything that reads ``live_points``."""
        active = self.active_regions
        if not active:
            self.live_points = self.live_points[:0]
            return
        live = np.concatenate([r.live_points for r in active])
        self.live_points = live[np.argsort(live["logL"], kind="stable")]

    def _region_next_poolsize(self, region):
        proposal = region.proposal
        if not getattr(proposal, "update_poolsize", False):
            return proposal.poolsize
        acc = region.mean_block_acceptance
        scale = 1.0 / acc if acc > 0 else proposal.max_poolsize_scale
        scale = min(max(scale, 1.0), proposal.max_poolsize_scale)
        return int(scale * proposal._poolsize)

    def _region_remaining(self, region):
        """Rough number of the region's own iterations left."""
        tol = self.tolerance
        c = self._region_condition(region)
        if not np.isfinite(c) or tol <= 0:
            return np.inf
        if c <= tol:
            return 0.0
        return region.nlive * (
            c + np.log(-np.expm1(-c)) - np.log(np.expm1(tol))
        )

    def _record_region_populations(self, region):
        snap = (
            region.time,
            region.likelihood_time,
            region.likelihood_evaluations,
        )
        samples = getattr(region.proposal, "samples", None)
        if region._retrain_snapshot is not None and samples is not None:
            dt, dl, dn = (a - b for a, b in zip(snap, region._retrain_snapshot))
            region.retrain_decision.record_populations(
                dt, samples.size, dl, dn
            )
        region._retrain_snapshot = snap

    def _region_check_state(self, region):
        """:meth:`check_training` and :meth:`train_proposal` for a
        region."""
        proposal = region.proposal
        decision = region.retrain_decision
        if not region.completed_training:
            self._train_region(region, force=True)
        elif decision is not None:
            if not proposal.populated and not proposal.populating:
                self._record_region_populations(region)
                retrain = decision.decide(
                    region.iterations,
                    self._region_next_poolsize(region),
                    remaining=self._region_remaining(region),
                )
                proposal.max_next_poolsize = decision.next_poolsize
                if retrain:
                    self._train_region(region, force=True)
        elif (
            not proposal.populated
            and self.train_on_empty
            and not proposal.populating
        ):
            self._train_region(region, force=True)
        elif (
            region.mean_block_acceptance < self.acceptance_threshold
            and self.retrain_acceptance
        ):
            self._train_region(region)
        elif (
            region.iterations - region.last_trained
        ) == self.training_frequency:
            self._train_region(region)

    def _region_flow_reset(self, region):
        """:meth:`check_flow_model_reset` for a region."""
        proposal = region.proposal
        decision = region.retrain_decision
        if not proposal.training_count:
            return
        if decision is not None and decision.allow_reset:
            if decision.reset_next:
                proposal.reset_model_weights(weights=True, permutations=True)
            return
        if (
            self.reset_acceptance
            and region.mean_block_acceptance < self.acceptance_threshold
        ):
            proposal.reset_model_weights(weights=True, permutations=True)
            return
        proposal.reset_model_weights(
            weights=(
                self.reset_weights
                and not (proposal.training_count % self.reset_weights)
            ),
            permutations=(
                self.reset_permutations
                and not (proposal.training_count % self.reset_permutations)
            ),
        )

    def _train_region(self, region, force=False):
        """:meth:`train_proposal` for a region, on its own live points."""
        if (
            region.iterations - region.last_trained < self.cooldown
            and region.training_iterations
            and not force
        ):
            return
        proposal = region.proposal
        decision = region.retrain_decision
        region.completed_training = False
        self._region_flow_reset(region)
        training_data = region.live_points.copy()
        proposal.current_live_points = region.live_points
        reset = proposal.training_count == 0 or (
            decision is not None and decision.reset_next
        )
        st = datetime.datetime.now()
        proposal.train(training_data)
        dt = datetime.datetime.now() - st
        self.training_time += dt
        self.history["training_iterations"].append(self.iteration)
        region.training_iterations.append(region.iterations)
        if decision is not None:
            epochs = getattr(proposal, "last_training_epochs", None)
            n_train = len(training_data)
            decision.record_training_time(dt.total_seconds(), epochs, n_train)
            decision.start_episode(region.iterations, reset, epochs, n_train)
            region._retrain_snapshot = (
                region.time,
                region.likelihood_time,
                region.likelihood_evaluations,
            )
        region.block_iteration = 0
        region.block_acceptance = 0.0
        region.completed_training = True
        if self.checkpoint_on_training and not self._splitting:
            # after the iteration: a checkpoint between the kill and its
            # replacement would count the dead point twice on resume
            self._checkpoint_pending = True

    def _finalise_partitioned(self):
        """Stop the regions still running and merge the dead points in
        likelihood order."""
        for region in self.active_regions:
            self._retire_region(region)
        order = self.state.finalise_partitioned()
        samples = list(self.nested_samples) + list(self.retired_samples)
        regions = (
            [-1] * self._partition_start
            + list(self.nested_regions)
            + list(self.retired_regions)
        )
        self.nested_samples = [samples[i] for i in order]
        self.nested_samples_region = np.asarray(regions, dtype=int)[order]
        self.retired_samples = []
        self.retired_regions = []
        self.nested_regions = []
        self.live_points = None

    def plot_partition(self, filename=None):
        """Histograms of the split coordinates of the live points, coloured
        by region, or, before the first split, by the parts of the cut the
        splitter would make (see :func:`nessai.partition.plot_partition`)."""
        if self.live_points is None or not len(self.live_points):
            return None
        if self.regions is None:
            proposal = self._flow_proposal
            coordinates = self._partition_coordinates.freeze(proposal)
            live = self.live_points
            coords = coordinates(live)
            sel = partition_module.hold_out_bit(live) == 0
            cut = None
            try:
                cut = self._partition_splitter.propose(
                    coords[sel],
                    coordinates.period_list(),
                    points=live[sel],
                    proposal=proposal,
                    max_labels=self.partition["max_regions"],
                    force=True,
                )
            except Exception as e:  # a diagnostic must not stop the run
                logger.debug(f"Partition plot: no candidate cut ({e})")
            labels = cut.classify(coords) if cut is not None else None
            if cut is not None:
                shares = np.bincount(labels, minlength=cut.n_labels) / len(
                    labels
                )
                title = (
                    f"iteration {self.iteration}, not split: candidate "
                    f"{cut.describe()}, shares "
                    + ", ".join(f"{s:.3f}" for s in shares)
                )
                legend = {k: f"part {k} ({s:.3f})" for k, s in enumerate(shares)}
            else:
                title = f"iteration {self.iteration}, not split: no candidate cut"
                legend = {0: "live points"}
        else:
            coordinates, cut = self._partition_last_cut
            active = self.active_regions
            live = np.concatenate([r.live_points for r in active])
            labels = np.concatenate(
                [np.full(len(r.live_points), r.label) for r in active]
            )
            coords = coordinates(live)
            legend = {
                r.label: (
                    f"region {r.label}: n {r.nlive}, ln X "
                    f"{self.state.region_logw[r.label]:.2f}, max logL "
                    f"{r.live_points['logL'][-1]:.1f}"
                )
                for r in active
            }
            stopped = [r.label for r in self.regions if not r.active]
            title = (
                f"iteration {self.iteration}: {len(active)} active region(s)"
                + (f", stopped {stopped}" if stopped else "")
            )
        return partition_module.plot_partition(
            coords,
            labels,
            list(coordinates.names),
            coordinates.period_list(),
            cut=cut,
            legend=legend,
            title=title,
            filename=filename,
        )

    def _plot_partition_diagnostic(self, split=False):
        if (
            getattr(self, "partition", None) is None
            or not self.plot
            or not self.partition["plot"]
            or not self.output
        ):
            return
        directory = os.path.join(self.output, "diagnostics")
        os.makedirs(directory, exist_ok=True)
        names = ["partition.png"]
        if split:
            names.append(f"partition_split_{self.iteration}.png")
        try:
            for name in names:
                self.plot_partition(filename=os.path.join(directory, name))
        except Exception as e:  # a diagnostic must not stop the run
            logger.warning(f"Partition plot failed: {type(e).__name__}: {e}")

    def partition_summary(self):
        """The regions as a structured array (one row each)."""
        fields = [
            ("region", "i8"),
            ("parent", "i8"),
            ("active", "?"),
            ("nlive", "i8"),
            ("iterations", "i8"),
            ("trainings", "i8"),
            ("log_x", "f8"),
            ("mean_acceptance", "f8"),
            ("final_p_value", "f8"),
        ]
        rows = []
        for r in self.regions or []:
            log_x = (
                self.state.region_logw.get(r.label, -np.inf)
                if self.state.region_logw is not None
                else -np.inf
            )
            rows.append(
                (
                    r.label,
                    -1 if r.parent is None else r.parent,
                    r.active,
                    r.nlive,
                    r.iterations,
                    len(r.training_iterations),
                    log_x,
                    r.mean_acceptance,
                    np.nan if r.final_p_value is None else r.final_p_value,
                )
            )
        return np.array(rows, dtype=fields)

    def check_training(self):
        """
        Check if the normalising flow should be trained

        Checks that can force training:
            - Training was previously stopped before completion
            - The pool is empty and the proposal was not in the process
              of populating when stopped.
        Checks that cannot force training is still on cooldown:
            - Acceptance falls below threshold and `retrain_acceptance` is
              true
            - The number of iterations since last training is equal to the
              training frequency

        Returns
        -------
        train : bool
            Try to train if true
        force : bool
            Force the training irrespective of cooldown
        """
        if not self.completed_training:
            logger.debug("Training flow (resume)")
            return True, True
        elif (
            self.retrain_decision is not None
            and not self.proposal.populated
            and not self.proposal.populating
        ):
            self._record_retrain_populations()
            retrain = self.retrain_decision.decide(
                self.iteration,
                self._next_poolsize(),
                remaining=self._remaining_iterations(),
            )
            self.proposal.max_next_poolsize = (
                self.retrain_decision.next_poolsize
            )
            if retrain:
                logger.debug("Training flow (retrain decision)")
                return True, True
            return False, False
        elif self.retrain_decision is not None:
            return False, False
        elif (
            not self.proposal.populated
            and self.train_on_empty
            and not self.proposal.populating
        ):
            logger.debug("Training flow (proposal empty)")
            return True, True
        elif (
            self.mean_block_acceptance < self.acceptance_threshold
            and self.retrain_acceptance
        ):
            logger.debug("Training flow (acceptance)")
            return True, False
        elif (self.iteration - self.last_updated) == self.training_frequency:
            logger.debug("Training flow (iteration)")
            return True, False
        else:
            return False, False

    def check_flow_model_reset(self):
        """
        Check if the normalising flow model should be reset.

        Checks acceptance if `reset_acceptance` is True and always checks
        how many times the flow has been trained.

        Flow will not be reset if it has not been trained. To force a reset
        manually call `proposal.reset_model_weights`.
        """
        if not self.proposal.training_count:
            return

        if self.retrain_decision is not None and (
            self.retrain_decision.allow_reset
        ):
            if self.retrain_decision.reset_next:
                self.proposal.reset_model_weights(
                    weights=True, permutations=True
                )
            return

        if (
            self.reset_acceptance
            and self.mean_block_acceptance < self.acceptance_threshold
        ):
            self.proposal.reset_model_weights(weights=True, permutations=True)
            return

        self.proposal.reset_model_weights(
            weights=(
                self.reset_weights
                and not (self.proposal.training_count % self.reset_weights)
            ),
            permutations=(
                self.reset_permutations
                and not (
                    self.proposal.training_count % self.reset_permutations
                )
            ),
        )

    def train_proposal(self, force=False):
        """
        Try to train the proposal. Proposal will not train if cooldown is not
        exceeded unless force is True.

        Parameters
        ----------
        force : bool
            Override training checks
        """
        if self.iteration - self.last_updated < self.cooldown and not force:
            logger.debug("Not training, still cooling down!")
        else:
            self.completed_training = False
            self.check_flow_model_reset()

            training_data = self.live_points.copy()
            if hasattr(getattr(self, "_flow_proposal", None),
                       "current_live_points"):
                self._flow_proposal.current_live_points = self.live_points
            if self.memory and (len(self.nested_samples) >= self.memory):
                training_data = np.concatenate(
                    [training_data, self.nested_samples[-self.memory :].copy()]
                )

            reset = self.proposal.training_count == 0 or (
                self.retrain_decision is not None
                and self.retrain_decision.reset_next
            )
            st = datetime.datetime.now()
            self.proposal.train(training_data)
            dt = datetime.datetime.now() - st
            self.training_time += dt
            self.history["training_iterations"].append(self.iteration)
            if self.retrain_decision is not None:
                epochs = getattr(self.proposal, "last_training_epochs", None)
                n_train = len(training_data)
                self.retrain_decision.record_training_time(
                    dt.total_seconds(), epochs, n_train
                )
                self.retrain_decision.start_episode(
                    self.iteration, reset, epochs, n_train
                )
                self._retrain_snapshot = self._retrain_time_snapshot()

            self.block_iteration = 0
            self.block_acceptance = 0.0
            self.completed_training = True
            if self.checkpoint_on_training:
                # forced: a periodic checkpoint is throttled by
                # checkpoint_interval, so the resume file would still lag
                # the flow just saved
                self.checkpoint(periodic=True, force=True)

    def check_state(self, force=False):
        """
        Check if state should be updated prior to drawing a new sample

        Force will override the cooldown mechanism.
        """

        if self.uninformed_sampling:
            if self.check_proposal_switch():
                force = True
            else:
                return
        # General override
        train = False
        if force:
            train = True
            logger.debug("Training flow (force)")
        elif not train:
            train, force = self.check_training()

        if train or force:
            self.train_proposal(force=force)

    @nessai_style()
    def plot_state(self, filename=None):
        """
        Produce plots with the current state of the nested sampling run.
        Plots are saved to the output directory specified at initialisation.

        Parameters
        ----------
        filename : str, optional
            If specified the figure will be saved, otherwise the figure is
            returned.
        """

        fig, ax = plt.subplots(7, 1, sharex=True, figsize=(12, 12))
        ax = ax.ravel()
        it = np.array(self.history["iterations"])

        for i in self.history["checkpoint_iterations"]:
            for a in ax:
                a.axvline(i, ls=":", color="#66ccff")

        for a in ax:
            a.axvline(self.iteration, c="#ff9900", ls="-.")

        ax[0].plot(
            it,
            sanitise_array(self.history["min_log_likelihood"]),
            label="Min log L",
        )
        ax[0].plot(
            it,
            sanitise_array(self.history["max_log_likelihood"]),
            label="Max log L",
        )
        ax[0].set_ylabel(r"$\log L$")
        ax[0].legend(frameon=False)

        ax_logl_diff = plt.twinx(ax[0])
        logl_diff = sanitise_array(
            np.array(self.history["max_log_likelihood"])
            - np.array(self.history["min_log_likelihood"]),
        )
        ax_logl_diff.plot(
            it,
            logl_diff,
            c="C2",
            ls=config.plotting.line_styles[2],
            label=r"$\Delta \log L$ ",
        )
        ax_logl_diff.set_yscale("log")
        ax_logl_diff.set_ylabel(r"$\Delta \log L$")
        handles, labels = ax[0].get_legend_handles_labels()
        handles_diff, labels_diff = ax_logl_diff.get_legend_handles_labels()
        ax[0].legend(
            handles + handles_diff, labels + labels_diff, frameon=False
        )

        logX_its = np.arange(len(self.state.log_vols))
        ax[1].plot(logX_its, self.state.log_vols, label="log X")
        ax[1].set_ylabel(r"$\log X$")
        ax[1].legend(frameon=False)

        if self.state.track_gradients:
            ax_logX_grad = plt.twinx(ax[1])
            # Use dotted linestyle (ls[2]) because dashed isn't clear
            ax_logX_grad.plot(
                logX_its,
                rolling_mean(np.abs(self.state.gradients), self.nlive // 10),
                c="C1",
                ls=config.plotting.line_styles[1],
                label="Gradient",
            )
            ax_logX_grad.set_ylabel(r"$|d\log L/d \log X|$")
            ax_logX_grad.set_yscale("log")
            handles, labels = ax[1].get_legend_handles_labels()
            handles_tw, labels_tw = ax_logX_grad.get_legend_handles_labels()
            ax[1].legend(
                handles + handles_tw, labels + labels_tw, frameon=False
            )

        ax[2].plot(
            it, self.history["likelihood_evaluations"], label="Evaluations"
        )
        ax[2].set_ylabel("Likelihood\n evaluations")
        ax[2].set_yscale("log")
        ax[2].legend()

        ax_time = plt.twinx(ax[2])
        ax_time.plot(
            it,
            np.array(self.history["sampling_time"]) / 60,
            ls=config.plotting.line_styles[1],
            color="C1",
            label="Time",
        )
        ax_time.set_ylabel("Sampling \ntime [min]")
        handles, labels = ax[2].get_legend_handles_labels()
        handles_time, labels_time = ax_time.get_legend_handles_labels()
        ax[2].legend(
            handles + handles_time, labels + labels_time, frameon=False
        )

        ax[3].plot(it, sanitise_array(self.history["logZ"]), label="logZ")
        ax[3].set_ylabel(r"$\log Z$")
        ax[3].legend(frameon=False)

        ax_dz = plt.twinx(ax[3])
        ax_dz.plot(
            it,
            self.history["dlogZ"],
            label="dlogZ",
            c="C1",
            ls=config.plotting.line_styles[1],
        )
        ax_dz.set_yscale("log")
        ax_dz.set_ylabel(r"$d \log Z$")
        handles, labels = ax[3].get_legend_handles_labels()
        handles_dz, labels_dz = ax_dz.get_legend_handles_labels()
        ax[3].legend(handles + handles_dz, labels + labels_dz, frameon=False)

        ax[4].plot(it, self.history["mean_acceptance"], label="Proposal")
        ax[4].plot(
            self.history["population_iterations"],
            self.history["population_acceptance"],
            label="Population",
        )
        ax[4].set_ylabel("Acceptance")
        handles, labels = ax[4].get_legend_handles_labels()

        ax_r = plt.twinx(ax[4])
        ax_r.plot(
            self.history["population_iterations"],
            self.history["population_radii"],
            label="Radius",
            color="C2",
            ls=config.plotting.line_styles[2],
        )
        ax_r.set_ylabel("Population radius")
        handles_r, labels_r = ax_r.get_legend_handles_labels()
        ax[4].legend(handles + handles_r, labels + labels_r, frameon=False)
        ax[4].set_yscale("log")
        ax[4].set_ylim(top=1.1)
        dtrain = np.array(self.history["training_iterations"][1:]) - np.array(
            self.history["training_iterations"][:-1]
        )
        ax[5].plot(self.history["training_iterations"][1:], dtrain)
        if self.history["training_iterations"]:
            ax[5].axvline(
                self.history["training_iterations"][0],
                ls="-",
                color="lightgrey",
            )
        ax[5].set_ylabel(r"$\Delta$ train")

        if len(self.history["rolling_p"]):
            it = (np.arange(len(self.history["rolling_p"])) + 1) * self.nlive
            ax[6].plot(it, self.history["rolling_p"], "o", label="p-value")
        ax[6].set_ylabel("p-value")
        ax[6].set_ylim([-0.1, 1.1])

        ax[-1].set_xlabel("Iteration")

        fig.suptitle(
            f"Sampling time: {self.current_sampling_time}", fontsize=16
        )

        handles = [
            Line2D(
                [0],
                [0],
                color="#ff9900",
                linestyle="-.",
                label="Current iteration",
            ),
            Line2D(
                [0], [0], color="#66ccff", linestyle=":", label="Checkpoint"
            ),
        ]
        fig.legend(handles=handles, frameon=False, ncol=3, loc=(0.6, 0.0))

        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        if filename is not None:
            fig.savefig(filename)
            plt.close(fig)
        else:
            return fig

    def plot_trace(self, filename=None, **kwargs):
        """
        Make trace plots for the nested samples.

        Parameters
        ----------
        filename : str, optional
            If filename is None, the figure is returned. Else the figure
            is saved with that file name.
        kwargs :
            Additional keyword arguments passed to
            :py:func:`nessai.plot.plot_trace`.
        """
        if self.nested_samples:
            if self.live_points is not None:
                log_x_live_points = self.state.get_logx_live_points(
                    len(self.live_points)
                )
            else:
                log_x_live_points = None
            fig = plot_trace(
                self.state.log_vols[1:],
                self.nested_samples,
                parameters=self.trace_parameters,
                live_points=self.live_points,
                log_x_live_points=log_x_live_points,
                filename=filename,
                **kwargs,
            )
            return fig
        else:
            logger.warning("Could not produce trace plot. No nested samples!")

    def plot_insertion_indices(self, filename=None, **kwargs):
        """
        Make a plot of all the insertion indices.

        Parameters
        ----------
        filename : str, optional
            If filename is None, the figure is returned. Else the figure
            is saved with that file name.
        kwargs :
            Keyword arguments passed to `nessai.plot.plot_indices`.
        """
        return plot_indices(
            self.insertion_indices, self.nlive, filename=filename, **kwargs
        )

    def update_state(self, force=False):
        """
        Update state after replacing a live point
        """
        # Check if acceptance is not None, this indicates the proposal
        # was populated
        if not self.proposal._checked_population:
            self.history["population_acceptance"].append(
                self.proposal.population_acceptance
            )
            self.history["population_radii"].append(self.proposal.r)
            self.history["population_iterations"].append(self.iteration)
            self.proposal._checked_population = True

        if not (self.iteration % (self.nlive // 10)) or force:
            self.update_history()

        if not (self.iteration % self.nlive) or force:
            if not force:
                self.check_insertion_indices()
                if self.plot and getattr(self, "regions", None) is None:
                    plot_indices(
                        self.insertion_indices[-self.nlive :],
                        self.nlive,
                        plot_breakdown=False,
                        filename=os.path.join(
                            self.output,
                            "diagnostics",
                            f"insertion_indices_{self.iteration}.png",
                        ),
                    )

            if self.plot:
                self.plot_state(
                    filename=os.path.join(self.output, "state.png")
                )
                self.plot_trace(
                    filename=os.path.join(self.output, "trace.png")
                )

            if self.uninformed_sampling:
                self.block_acceptance = 0.0
                self.block_iteration = 0

        if getattr(self, "partition", None) is not None and (
            force or not (self.iteration % self.partition["plot_every"])
        ):
            self._plot_partition_diagnostic()

        if self.checkpointing:
            self.checkpoint(periodic=True)

        self.proposal.ns_acceptance = self.mean_block_acceptance

    def check_resume(self):
        """
        Check the normalising flow is correctly configured is the sampler
        was resumed.
        """
        if self.resumed:
            if self.uninformed_sampling is False:
                self.check_proposal_switch(force=True)
            # If pool is populated reset the flag since it is set to
            # false during initialisation
            if hasattr(self._flow_proposal, "resume_populated"):
                if (
                    self._flow_proposal.resume_populated
                    and self._flow_proposal.indices
                ):
                    self._flow_proposal.populated = True
                    logger.info("Resumed with populated pool")

            self.resumed = False

    def finalise(self):
        """
        Finalise things after sampling
        """
        logger.info("Finalising")
        if getattr(self, "regions", None) is not None:
            self._finalise_partitioned()
        else:
            for i, p in enumerate(self.live_points):
                self.state.increment(p["logL"], nlive=self.nlive - i)
                self.nested_samples.append(p)
            self.live_points = None

        # Refine evidence estimate
        self.update_state(force=True)
        self.state.finalise()
        if self.retrain_decision is not None:
            self.finalise_retrain_decision()
        # output the chain and evidence
        self.finalised = True

    def finalise_retrain_decision(self):
        """Log and save the unit costs measured during the run."""
        summary = self.retrain_decision.summary()
        for key, value in summary["unit_costs"].items():
            if value["measured"] is not None:
                logger.info(
                    "Retrain decision: measured %s cost %.3g s (provided: %s)",
                    key,
                    value["measured"],
                    value["provided"],
                )
        if "long_run" in summary:
            logger.info(
                "Retrain decision: optimal interval between trainings "
                "~%.0f iterations",
                summary["long_run"]["tau"],
            )
        if self.output:
            filename = os.path.join(self.output, "retrain_costs.json")
            self.retrain_decision.cost.save(filename)
            logger.info(
                "Measured unit costs saved to %s. Pass this file as "
                "`retrain_costs` to use them in a future run.",
                filename,
            )

    def nested_sampling_loop(self):
        """Main nested sampling loop.

        Notes
        -----
        If :code:`prior_sampling=True` then sampling will stop after drawing
        the initial live points.

        Returns
        -------
        log_evidence : float
            The final log evidence.
        nested_samples : numpy.ndarray
            Array of nested samples.
        """
        if self.finalised:
            logger.info("Run has already finished!")
            return self.log_evidence, np.array(self.nested_samples)
        self.sampling_start_time = datetime.datetime.now()
        if not self.initialised:
            self.initialise(live_points=True)

        if self.prior_sampling:
            if self._close_pool:
                self.close_pool()
            self.finalise()
            return self.log_evidence, np.array(self.nested_samples)

        self.check_resume()

        if self.iteration:
            self.update_state()

        logger.info("Starting nested sampling loop")

        while not self.stopping_criterion.is_met(self.condition):
            if getattr(self, "partition", None) is not None:
                self._maybe_partition()
            if getattr(self, "regions", None) is not None:
                if not self.active_regions:
                    self.condition = 0.0
                    continue
                self._partitioned_iteration()
            else:
                self.check_state()
                self.consume_sample()

            self.update_state()

            self.periodically_log_state()

            if self.iteration >= self.max_iteration:
                logger.info("Reached max iteration")
                break
        logger.info(
            f"Stopping with {self.stopping_criterion.name}"
            f"={self.condition:.3f}"
        )

        # final adjustments
        # avoid repeating final adjustments if resuming a completed run.
        if not self.finalised and self.stopping_criterion.is_met(
            self.condition
        ):
            self.finalise()

        logger.info(
            f"Final ln-evidence: {self.state.logZ:.3f} +/- "
            f"{self.state.log_evidence_error:.3f}"
        )
        logger.info(f"Information: {self.state.info[-1]:.2f}")

        self.check_insertion_indices(rolling=False)

        # This includes updating the total sampling time
        self.checkpoint(periodic=True, force=True)

        if self._close_pool:
            self.close_pool()

        logger.info(f"Total sampling time: {self.sampling_time}")
        logger.info(f"Total training time: {self.training_time}")
        logger.info(f"Total population time: {self.proposal_population_time}")
        logger.info(
            f"Total likelihood evaluations: {self.likelihood_calls:3d}"
        )
        logger.info(
            "Time spent evaluating likelihood: "
            f"{self.likelihood_evaluation_time}"
        )

        return self.state.logZ, np.array(self.nested_samples)

    def get_result_dictionary(self):
        """Return a dictionary that contains results"""
        d = super().get_result_dictionary()
        d["insertion_indices"] = self.insertion_indices
        d["final_p_value"] = self.final_p_value
        d["final_ks_statistic"] = self.final_ks_statistic
        d["nested_samples"] = np.array(self.nested_samples)
        d["log_posterior_weights"] = self.state.log_posterior_weights
        d["logL_birth"] = self.birth_log_likelihoods
        d["log_evidence"] = self.log_evidence
        d["log_evidence_error"] = self.state.log_evidence_error
        d["information"] = self.information
        d["training_time"] = self.training_time.total_seconds()
        d["population_time"] = self.proposal_population_time.total_seconds()
        if getattr(self, "regions", None) is not None:
            fields = [
                ("iteration", "i8"),
                ("logL", "f8"),
                ("parent", "i8"),
                ("region", "i8"),
                ("share", "f8"),
                ("n_points", "i8"),
                ("n_held_out", "i8"),
                ("nlive", "i8"),
                ("log_x", "f8"),
                ("cut", "S256"),
            ]
            d["partitions"] = np.array(
                [
                    tuple(
                        r[k].encode()[:256] if k == "cut" else r[k]
                        for k, _ in fields
                    )
                    for r in self.partitions
                ],
                dtype=fields,
            )
            d["regions"] = self.partition_summary()
            region = getattr(self, "nested_samples_region", None)
            if region is not None:
                d["nested_samples_region"] = region
            log_z = self.state.region_log_evidence()
            d["region_log_evidence"] = np.array(
                [(-1 if k is None else k, v) for k, v in log_z.items()],
                dtype=[("region", "i8"), ("log_evidence", "f8")],
            )
        return d

    @classmethod
    def resume_from_pickled_sampler(
        cls,
        sampler,
        model,
        flow_config=None,
        weights_path=None,
        partition=None,
        **kwargs,
    ):
        """Resume from a pickled sampler.

        Parameters
        ----------
        sampler: Any
            Pickled sampler.
        model : :obj:`nessai.model.Model`
            User-defined model
        flow_config : Optional[dict]
            Dictionary for configuring the flow
        weights_path : Optional[str]
            Weights file to use in place of the weights file stored in the
            pickle file.
        partition : Optional[dict]
            If given, replaces the partition configuration (see
            :py:meth:`configure_partition`), e.g. to enable it on a run
            started without it. The regions of a run already split are
            kept.
        kwargs :
            Keyword arguments passed to the parent class's method.

        Returns
        -------
        Instance of NestedSampler
        """
        obj = super(NestedSampler, cls).resume_from_pickled_sampler(
            sampler, model, **kwargs
        )
        obj._uninformed_proposal.resume(model)
        if flow_config is None:
            flow_config = {}
        obj._flow_proposal.resume(model, flow_config, weights_path)
        obj._resume_flow_config = deepcopy(flow_config)
        if partition is not None or not hasattr(obj, "partition"):
            obj.configure_partition(partition)
        for region in obj.active_regions:
            proposal = region.proposal
            proposal.resume(model, deepcopy(flow_config), None)
            if getattr(proposal, "resume_populated", False) and (
                proposal.indices
            ):
                proposal.populated = True
            proposal.current_live_points = region.live_points
        if getattr(obj.state, "_finalised_partitioned", False):
            # a run finished before nessai set the effective nlive
            obj.state.set_partitioned_nlive()
        return obj
