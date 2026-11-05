#!/usr/bin/env python
"""The clip-compression correction on a Gaussian with a known evidence.

A unit Gaussian likelihood in ``DIMS`` dimensions in a uniform prior on
[-10, 10]^DIMS: ln Z = -DIMS ln 20 (the prior's truncation of the Gaussian
is negligible). Each seed runs nessai's FlowProposal with a small flow and
the log-proposal threshold in clip mode (``quantile`` the cap), with the
clip-compression correction applied, and records ln Z with and without it
(the same run's terms: ``NestedSampler.clip_compression_log_evidence``).

usage: examples/clip_compression_toy.py OUTDIR [--seeds 100] [--quantile 0.05]
       [--nlive 500] [--dims 6] [--jobs 8]
Writes OUTDIR/results.json and prints the mean errors of ln Z.
"""
import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

BOUND = 10.0


def run(seed, outdir, quantile, nlive, dims):
    import torch

    torch.set_num_threads(1)
    from nessai.flowsampler import FlowSampler
    from nessai.model import Model

    class Gaussian(Model):
        def __init__(self):
            self.names = [f"x{i}" for i in range(dims)]
            self.bounds = {n: [-BOUND, BOUND] for n in self.names}

        def log_prior(self, x):
            return np.log(self.in_bounds(x), dtype=float) - dims * np.log(2 * BOUND)

        def log_likelihood(self, x):
            r2 = sum(np.asarray(x[n], float) ** 2 for n in self.names)
            return -0.5 * r2 - 0.5 * dims * np.log(2 * np.pi)

    logging.getLogger("nessai").setLevel(logging.WARNING)
    output = os.path.join(outdir, f"seed{seed:03d}")
    truncation = {"quantile": quantile, "prior_relative": True, "mode": "clip"} \
        if quantile else None
    fs = FlowSampler(
        Gaussian(), output=output, nlive=nlive, seed=seed, plot=False,
        checkpointing=False, resume=False,
        flow_config=dict(n_blocks=2, n_neurons=8, n_layers=1),
        truncation_methods=["log_proposal_threshold"] if quantile else None,
        truncation_kwargs={"log_proposal_threshold": truncation} if quantile else None,
        threshold_from_live_points=True,
        clip_compression_correction=True,
    )
    fs.run(plot=False, save=False)
    ns = fs.ns
    terms = np.asarray(ns.clip_compression_terms)
    return dict(
        seed=seed, ln_z=float(fs.logZ), ln_z_err=float(fs.logZ_error),
        ln_z_corrected=float(ns.clip_compression_log_evidence(True)),
        ln_z_uncorrected=float(ns.clip_compression_log_evidence(False)),
        mean_term=float(terms.mean()), share_over_cap=float(np.mean(terms != 0)),
        iterations=int(ns.iteration), likelihood_calls=int(ns.likelihood_calls),
        insertion_p=float(ns.final_p_value) if ns.final_p_value is not None else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--quantile", type=float, default=0.05)
    ap.add_argument("--nlive", type=int, default=500)
    ap.add_argument("--dims", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    truth = -args.dims * np.log(2 * BOUND)
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    with ProcessPoolExecutor(args.jobs) as pool:
        results = list(pool.map(run, seeds, *zip(*[(args.outdir, args.quantile, args.nlive,
                                                    args.dims)] * len(seeds))))
    out = os.path.join(args.outdir, "results.json")
    with open(out, "w") as f:
        json.dump(dict(vars(args), truth=truth, results=results), f, indent=1)
    for key in ("ln_z_uncorrected", "ln_z_corrected", "ln_z"):
        err = np.array([r[key] for r in results]) - truth
        print(f"{key:18s} error {err.mean():+.4f} +- {err.std(ddof=1) / np.sqrt(len(err)):.4f}"
              f" (scatter {err.std(ddof=1):.3f}, mean reported error "
              f"{np.mean([r['ln_z_err'] for r in results]):.3f})")
    print(f"mean term b {np.mean([r['mean_term'] for r in results]):.2e}, insertions over the "
          f"cap {np.mean([r['share_over_cap'] for r in results]):.2%}, iterations "
          f"{np.mean([r['iterations'] for r in results]):.0f}")
    print("->", out)


if __name__ == "__main__":
    main()
