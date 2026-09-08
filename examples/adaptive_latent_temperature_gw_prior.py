"""Adaptive latent-temperature with a GW-inspired non-uniform prior.

The prior mimics a gravitational-wave parameter-estimation prior:

* ``chirp_mass``          uniform
* ``mass_ratio``          uniform
* ``luminosity_distance`` p(d) proportional to d**2  (Euclidean volume)
* ``theta_jn``            p(theta) proportional to sin(theta)  (isotropic)
* ``dec``                 p(dec) proportional to cos(dec)      (isotropic)
* ``ra``                  uniform

The likelihood is a tight, correlated Gaussian (a distance-inclination style
degeneracy plus a mass-ratio / chirp-mass anti-correlation), so the posterior
is a thin tilted ridge sitting inside a genuinely structured prior -- the
regime where the retroactive latent-temperature criterion has a real interior
optimum rather than "broaden until the box clips".

Runs baseline vs ``adapt_latent_temperature=True`` for several seeds and
compares proposal acceptance and evidence.

Run: ``python examples/adaptive_latent_temperature_gw_prior.py``
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from nessai.flowsampler import FlowSampler
from nessai.livepoint import dict_to_live_points
from nessai.model import Model
from nessai.utils import configure_logger

SEEDS = [1451, 7, 20240, 99]
BASE_OUTPUT = "./outdir/adaptive_latent_temperature_gw_prior/"
PLOT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "adaptive_latent_temperature_gw_prior.png",
)

configure_logger(output=BASE_OUTPUT, log_level="WARNING")

NAMES = [
    "chirp_mass",
    "mass_ratio",
    "luminosity_distance",
    "theta_jn",
    "dec",
    "ra",
]
BOUNDS = {
    "chirp_mass": [15.0, 45.0],
    "mass_ratio": [0.125, 1.0],
    "luminosity_distance": [100.0, 5000.0],
    "theta_jn": [0.0, np.pi],
    "dec": [-0.5 * np.pi, 0.5 * np.pi],
    "ra": [0.0, 2.0 * np.pi],
}

# Tight correlated Gaussian likelihood.
_MU = np.array([28.0, 0.55, 1600.0, 1.15, 0.20, 3.1])
_SIGMA = np.array([1.6, 0.09, 450.0, 0.32, 0.18, 0.25])
_CORR = np.eye(6)
_CORR[2, 3] = _CORR[3, 2] = 0.82  # distance <-> inclination
_CORR[0, 1] = _CORR[1, 0] = -0.55  # chirp mass <-> mass ratio
_CORR[4, 5] = _CORR[5, 4] = 0.35  # dec <-> ra
_COV = _CORR * np.outer(_SIGMA, _SIGMA)
_LIKELIHOOD = multivariate_normal(mean=_MU, cov=_COV)


class GWInspiredModel(Model):
    def __init__(self):
        self.names = NAMES
        self.bounds = BOUNDS
        a, b = BOUNDS["luminosity_distance"]
        self._dl_norm = np.log(3.0) - np.log(b**3 - a**3)

    def log_prior(self, x):
        log_p = np.log(self.in_bounds(x), dtype="float")
        log_p -= np.log(
            BOUNDS["chirp_mass"][1] - BOUNDS["chirp_mass"][0]
        )
        log_p -= np.log(
            BOUNDS["mass_ratio"][1] - BOUNDS["mass_ratio"][0]
        )
        log_p += self._dl_norm + 2.0 * np.log(x["luminosity_distance"])
        log_p += np.log(np.sin(x["theta_jn"])) - np.log(2.0)
        log_p += np.log(np.cos(x["dec"])) - np.log(2.0)
        log_p -= np.log(2.0 * np.pi)
        return log_p

    def new_point(self, N=1):
        rng = self.rng
        a, b = BOUNDS["luminosity_distance"]
        u = rng.uniform(size=(N, 4))
        d = {
            "chirp_mass": rng.uniform(*BOUNDS["chirp_mass"], N),
            "mass_ratio": rng.uniform(*BOUNDS["mass_ratio"], N),
            "luminosity_distance": (a**3 + u[:, 0] * (b**3 - a**3)) ** (
                1.0 / 3.0
            ),
            "theta_jn": np.arccos(1.0 - 2.0 * u[:, 1]),
            "dec": np.arcsin(2.0 * u[:, 2] - 1.0),
            "ra": rng.uniform(*BOUNDS["ra"], N),
        }
        return dict_to_live_points(d)

    def new_point_log_prob(self, x):
        return self.log_prior(x)

    def log_likelihood(self, x):
        return _LIKELIHOOD.logpdf(self.unstructured_view(x))


def run(label, seed, **proposal_kwargs):
    fs = FlowSampler(
        GWInspiredModel(),
        output=f"{BASE_OUTPUT}{label}_{seed}",
        nlive=1000,
        flow_config=dict(n_blocks=4, n_neurons=24, n_layers=2),
        resume=False,
        seed=seed,
        plot=False,
        **proposal_kwargs,
    )
    fs.run(plot=False)
    ns = fs.ns
    accept = np.asarray(ns.history["population_acceptance"], dtype=float)
    temps = list(
        getattr(fs.ns._flow_proposal, "latent_temperature_history", [])
    )
    return dict(
        accept=accept,
        temps=temps,
        logZ=fs.log_evidence,
        logZ_err=fs.log_evidence_error,
        n_like=ns.total_likelihood_evaluations,
    )




def main():
    baseline, adaptive = [], []
    for seed in SEEDS:
        b = run("baseline", seed)
        a = run(
            "adaptive",
            seed,
            adapt_latent_temperature=True,
            latent_temperature_validation_size=20000,
        )
        baseline.append(b)
        adaptive.append(a)
        print(
            f"seed {seed:>6}: acceptance {np.nanmean(b['accept']):.4f} -> "
            f"{np.nanmean(a['accept']):.4f} | "
            f"logZ {b['logZ']:.3f}+/-{b['logZ_err']:.3f} -> "
            f"{a['logZ']:.3f}+/-{a['logZ_err']:.3f} | "
            f"final T {a['temps'][-1] if a['temps'] else 1.0:.2f}"
        )

    b_acc = np.array([np.nanmean(r["accept"]) for r in baseline])
    a_acc = np.array([np.nanmean(r["accept"]) for r in adaptive])
    b_lnz = np.array([r["logZ"] for r in baseline])
    a_lnz = np.array([r["logZ"] for r in adaptive])
    print(
        f"\nmean acceptance  baseline {b_acc.mean():.4f} +/- {b_acc.std():.4f}"
        f"   adaptive {a_acc.mean():.4f} +/- {a_acc.std():.4f}"
    )
    print(
        f"mean logZ        baseline {b_lnz.mean():.3f} +/- {b_lnz.std():.3f}"
        f"   adaptive {a_lnz.mean():.3f} +/- {a_lnz.std():.3f}"
    )

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].axhline(0, color="k", lw=0.8)
    axs[0].bar(
        np.arange(len(SEEDS)),
        a_acc - b_acc,
        color=["C2" if d > 0 else "C3" for d in a_acc - b_acc],
    )
    axs[0].set_xticks(np.arange(len(SEEDS)))
    axs[0].set_xticklabels(SEEDS)
    axs[0].set_xlabel("seed")
    axs[0].set_ylabel("adaptive - baseline mean acceptance")
    axs[0].set_title(f"acceptance: {b_acc.mean():.3f} -> {a_acc.mean():.3f}")
    for r in adaptive:
        axs[1].plot(r["temps"], marker=".", alpha=0.8)
    axs[1].axhline(1.0, color="k", ls=":", label="baseline T = 1")
    axs[1].set_xlabel("adaptation step")
    axs[1].set_ylabel("latent_temperature")
    axs[1].set_title("adaptive temperature trajectories")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    print(f"saved {PLOT_PATH}")


if __name__ == "__main__":
    main()
