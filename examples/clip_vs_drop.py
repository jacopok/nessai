#!/usr/bin/env python

# Compare the drop and clip modes of the log-proposal truncation rule, and
# clip mode with a latent distortion, on a heavy-tailed likelihood with a
# known evidence: a product of Student-t distributions (3 degrees of
# freedom), which a flow with a Gaussian latent undercovers in the tails.
#
# All arms cap the rejection weight at the 10% quantile of the live points'
# log q - log prior.
#
# - drop: draws above the cap are discarded, so the tails where the flow is
#   too narrow are never proposed. New points are biased inwards, log Z
#   comes out biased and the posterior too narrow.
# - clip: those draws are kept with their weight clipped, so they are only
#   under-weighted. Unbiased, but every clipped draw is a likelihood call.
# - clip + distortion: after each training, z' = mu + sigma z is fitted per
#   latent dimension to raise the low tail of the live points'
#   log q' - log prior (sigma > 1 broadens the flow into the tails), and the
#   pool is drawn and scored with it. The cap then cuts less and more of each
#   flow draw is accepted.
#
# Each arm is run with several seeds. Reported: the log Z error in units of
# the quoted uncertainty ("pull"), the posterior std against the exact one,
# the likelihood calls, the mean population acceptance (new points per flow
# draw) and the share of distortion fits that passed the checks. Takes a
# few minutes.

import numpy as np
from scipy.integrate import quad
from scipy.stats import t

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import configure_logger

N_DIMS = 6
DOF = 3
BOUND = 10
QUANTILE = 0.01
N_SEEDS = 3
NLIVE = 500

output = "./outdir/clip_vs_drop/"
logger = configure_logger(output=output, log_level="WARNING")


class StudentTModel(Model):
    """A Student-t likelihood in each dimension with a uniform prior on
    [-BOUND, BOUND]^n."""

    def __init__(self, n_dims):
        self.names = [f"x_{i}" for i in range(n_dims)]
        self.bounds = {n: [-BOUND, BOUND] for n in self.names}

    def log_prior(self, x):
        log_p = np.log(self.in_bounds(x), dtype="float")
        log_p -= len(self.names) * np.log(2 * BOUND)
        return log_p

    def log_likelihood(self, x):
        log_l = np.zeros(x.size)
        for n in self.names:
            log_l += t.logpdf(x[n], DOF)
        return log_l


names = StudentTModel(N_DIMS).names
# the likelihood is normalised: Z is its mass inside the prior box
mass = t.cdf(BOUND, DOF) - t.cdf(-BOUND, DOF)
true_log_z = N_DIMS * (np.log(mass) - np.log(2 * BOUND))
true_std = np.sqrt(
    quad(lambda x: x**2 * t.pdf(x, DOF), -BOUND, BOUND)[0] / mass
)

arms = {
    "drop": dict(mode="drop"),
    "clip": dict(mode="clip"),
    "clip + distortion": dict(mode="clip", latent_distortion="expert"),
}
results = {arm: [] for arm in arms}
for seed in range(N_SEEDS):
    for arm, opts in arms.items():
        opts = dict(opts)
        mode = opts.pop("mode")
        fs = FlowSampler(
            StudentTModel(N_DIMS),
            output=f"{output}{arm.replace(' + ', '_')}_{seed}",
            resume=False,
            seed=seed,
            nlive=NLIVE,
            truncation_method="log_proposal_threshold",
            truncation_kwargs=dict(quantile=QUANTILE, mode=mode),
            threshold_from_live_points=True,
            **opts,
        )
        fs.run(plot=False)
        ns = fs.ns
        std = np.mean([np.std(fs.posterior_samples[n]) for n in names])
        pull = (fs.log_evidence - true_log_z) / fs.log_evidence_error
        acceptance = np.nanmean(
            np.asarray(ns.history["population_acceptance"], dtype=float)
        )
        fits = getattr(ns.proposal, "latent_distortion_history", [])
        accepted = np.mean([f["accepted"] for f in fits]) if fits else np.nan
        results[arm].append(
            (pull, std, ns.model.likelihood_evaluations, acceptance, accepted)
        )

print(
    f"\n{N_SEEDS} runs per arm; unbiased: mean pull 0, "
    f"posterior std {true_std:.3f}"
)
for arm, rows in results.items():
    pull, std, calls, acceptance, accepted = np.array(rows).T
    line = (
        f"{arm:>18}: log Z pull = {pull.mean():+.2f} "
        f"+/- {pull.std(ddof=1) / np.sqrt(len(pull)):.2f}, "
        f"posterior std = {std.mean():.3f}, "
        f"likelihood calls = {calls.mean():.0f}, "
        f"population acceptance = {acceptance.mean():.3f}"
    )
    if np.isfinite(accepted).all():
        line += f", distortion fits accepted = {accepted.mean():.0%}"
    print(line)
