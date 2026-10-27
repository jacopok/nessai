#!/usr/bin/env python

# Compare the drop and clip modes of the log-proposal truncation rule on a
# Gaussian likelihood with a known evidence.
#
# Both modes cap the rejection weight at the 10% quantile of the live points'
# log q - log prior. Drop mode discards draws above the cap, so the region
# where the flow is narrowest is never proposed: new points are biased
# inwards, log Z comes out biased high and the posterior too narrow. Clip
# mode keeps those draws with the weight clipped, so they are only
# under-weighted. Each mode is run with several seeds and the log Z error is
# reported in units of the quoted uncertainty ("pull"). Takes a few minutes.

import numpy as np
from scipy.stats import norm

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import configure_logger

N_DIMS = 4
QUANTILE = 0.1
N_SEEDS = 6
NLIVE = 300

output = "./outdir/clip_vs_drop/"
logger = configure_logger(output=output, log_level="WARNING")


class GaussianModel(Model):
    """A unit Gaussian likelihood with a uniform prior on [-10, 10]^n."""

    def __init__(self, n_dims):
        self.names = [f"x_{i}" for i in range(n_dims)]
        self.bounds = {n: [-10, 10] for n in self.names}

    def log_prior(self, x):
        log_p = np.log(self.in_bounds(x), dtype="float")
        log_p -= len(self.names) * np.log(20)
        return log_p

    def log_likelihood(self, x):
        log_l = np.zeros(x.size)
        for n in self.names:
            log_l += norm.logpdf(x[n])
        return log_l


names = GaussianModel(N_DIMS).names
# the likelihood is normalised and its mass inside the prior is ~1
true_log_z = -N_DIMS * np.log(20)

results = {"drop": [], "clip": []}
for seed in range(N_SEEDS):
    for mode in results:
        fs = FlowSampler(
            GaussianModel(N_DIMS),
            output=f"{output}{mode}_{seed}",
            resume=False,
            seed=seed,
            nlive=NLIVE,
            truncation_method="log_proposal_threshold",
            truncation_kwargs=dict(quantile=QUANTILE, mode=mode),
            threshold_from_live_points=True,
        )
        fs.run(plot=False)
        std = np.mean([np.std(fs.posterior_samples[n]) for n in names])
        pull = (fs.log_evidence - true_log_z) / fs.log_evidence_error
        results[mode].append((pull, std))

print(f"\n{N_SEEDS} runs per mode; unbiased: mean pull 0, posterior std 1")
for mode, rows in results.items():
    pull, std = np.array(rows).T
    print(
        f"{mode}: log Z pull = {pull.mean():+.2f} "
        f"+/- {pull.std(ddof=1) / np.sqrt(len(pull)):.2f}, "
        f"posterior std = {std.mean():.3f}"
    )
