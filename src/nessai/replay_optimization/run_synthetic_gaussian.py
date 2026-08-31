#!/usr/bin/env python
"""Run bilby + nessai (default settings) on a synthetic 6D correlated Gaussian.

The point is to produce a bilby result JSON that can be fed straight into
``replay_optimisation.py`` (and the other ``replay_optimization`` scripts) as a
small, fast, fully-synthetic test case:

    python run_synthetic_gaussian.py
    python replay_optimisation.py --result outdir_synthetic_gaussian/synthetic_gaussian_result.json

The likelihood is a multivariate normal with a non-trivial covariance (unit
variances, constant pairwise correlation ``RHO``) over ``NDIM`` parameters, each
with a broad uniform prior.  nessai is run with its defaults so the archived run
mirrors a realistic ``sampler_kwargs`` block; ``nlive`` is kept modest so the
whole thing finishes in a couple of minutes on a laptop.
"""

from __future__ import annotations

import argparse
import sys
import types

import numpy as np
from scipy.stats import multivariate_normal


def _install_nessai_bilby_shim() -> None:
    """Make bilby's (deprecated) built-in nessai wrapper work with this checkout.

    This fork of nessai removed ``nessai.utils.bilbyutils`` and renamed
    ``setup_logger`` -> ``configure_logger``; bilby 2.8's wrapper still expects
    the old names.  Rather than depend on the external ``nessai-bilby`` plugin,
    re-expose the handful of symbols bilby imports.
    """
    import nessai.utils
    from nessai.utils.logging import configure_logger
    from nessai.utils.settings import get_all_kwargs, get_run_kwargs_list

    if not hasattr(nessai.utils, "setup_logger"):
        nessai.utils.setup_logger = configure_logger

    if "nessai.utils.bilbyutils" not in sys.modules:
        mod = types.ModuleType("nessai.utils.bilbyutils")
        mod.get_all_kwargs = get_all_kwargs
        mod.get_run_kwargs_list = get_run_kwargs_list
        sys.modules["nessai.utils.bilbyutils"] = mod
        nessai.utils.bilbyutils = mod


_install_nessai_bilby_shim()

import bilby  # noqa: E402

NDIM = 10
PRIOR_HALF_WIDTH = 10.0
COV_SEED = 0  # fix so the target is reproducible across trials


def make_covariance(ndim: int, seed: int = COV_SEED) -> np.ndarray:
    """Random SPD covariance with ~unit-scale variances.

    Draws a random correlation matrix (via a Wishart-style A @ A.T,
    normalised to unit diagonal) then rescales by random standard
    deviations near 1, so the overall scale matches the old
    constant-correlation matrix.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((ndim, ndim + 2))
    m = a @ a.T                      # SPD
    d = np.sqrt(np.diag(m))
    corr = m / np.outer(d, d)        # unit diagonal
    std = rng.uniform(0.7, 1.3, size=ndim)
    return corr * np.outer(std, std)

class CorrelatedGaussianLikelihood(bilby.core.likelihood.Likelihood):
    """Multivariate-normal likelihood with a fixed mean and covariance."""

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.parameter_names = [f"x{i}" for i in range(len(mean))]
        super().__init__(parameters=dict.fromkeys(self.parameter_names))
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self._dist = multivariate_normal(mean=self.mean, cov=self.cov)

    def log_likelihood(self) -> float:
        x = np.array([self.parameters[name] for name in self.parameter_names])
        return float(self._dist.logpdf(x))


def _strip_spurious_augment_kwargs(result, result_path: str) -> None:
    import json

    keys = ("augment_dims", "generate_augment", "marginalise_augment", "n_marg")
    sk = dict(getattr(result, "sampler_kwargs", None) or {})
    if not any(k in sk for k in keys):
        return
    for k in keys:
        sk.pop(k, None)
    result.sampler_kwargs = sk
    with open(result_path) as f:
        data = json.load(f)
    for k in keys:
        data.get("sampler_kwargs", {}).pop(k, None)
    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="outdir_synthetic_gaussian")
    parser.add_argument("--label", default="synthetic_gaussian")
    parser.add_argument("--nlive", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ndim", type=int, default=NDIM)
    args = parser.parse_args()

    bilby.core.utils.random.seed(args.seed)

    mean = np.zeros(args.ndim)
    cov = make_covariance(args.ndim, args.seed)

    likelihood = CorrelatedGaussianLikelihood(mean, cov)

    priors = bilby.core.prior.PriorDict()
    for i in range(args.ndim):
        priors[f"x{i}"] = bilby.core.prior.Uniform(
            -PRIOR_HALF_WIDTH, PRIOR_HALF_WIDTH, name=f"x{i}", latex_label=f"$x_{i}$"
        )

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="nessai",
        nlive=args.nlive,
        outdir=args.outdir,
        label=args.label,
        seed=args.seed,
        resume=False,
        plot=False,
        save="json",
        # nessai defaults everywhere else -- keep the sampler_kwargs realistic.
    )

    # bilby's get_all_kwargs dumps the *signature* defaults of every optional
    # nessai proposal class into sampler_kwargs, including AugmentedFlowProposal's
    # augment_dims=1.  This run used the plain FlowProposal, so strip those keys
    # to stop replay_optimisation.py from replaying with augmentation.
    _strip_spurious_augment_kwargs(result, f"{args.outdir}/{args.label}_result.json")

    # Sanity check: the replay harness needs the birth-iteration column.
    ns = result.nested_samples
    assert ns is not None and "iteration" in ns, (
        "nested samples lack an 'iteration' column; replay_optimisation.py "
        "cannot reconstruct live sets from this result."
    )
    print(f"\nResult written to {args.outdir}/{args.label}_result.json")
    print(f"nested samples: {len(ns)}, log Z = {result.log_evidence:.3f}")


if __name__ == "__main__":
    main()
