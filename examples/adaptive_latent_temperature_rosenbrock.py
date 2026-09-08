"""Compare FlowProposal with and without adaptive latent-temperature.

Runs nessai on a 5-D Rosenbrock likelihood (uniform prior) for a handful of
seeds, once with the default proposal and once with
``adapt_latent_temperature=True`` (validation-batch gated), and compares the
per-population proposal acceptance and the evidence.

Run: ``python examples/adaptive_latent_temperature_rosenbrock.py``
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import configure_logger

DIMS = 5
SEEDS = [1451, 7, 20240, 99]
BASE_OUTPUT = "./outdir/adaptive_latent_temperature_rosenbrock/"
PLOT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "adaptive_latent_temperature_rosenbrock.png",
)

configure_logger(output=BASE_OUTPUT, log_level="WARNING")


class RosenbrockModel(Model):
    """Rosenbrock function in ``dims`` dimensions on ``[-5, 5]^dims``."""

    def __init__(self, dims):
        self.names = [f"x_{d}" for d in range(dims)]
        self.bounds = {n: [-5.0, 5.0] for n in self.names}

    def log_prior(self, x):
        log_p = np.log(self.in_bounds(x), dtype="float")
        for bounds in self.bounds.values():
            log_p -= np.log(bounds[1] - bounds[0])
        return log_p

    def log_likelihood(self, x):
        x = self.unstructured_view(x)
        return -np.sum(
            100.0 * (x[..., 1:] - x[..., :-1] ** 2.0) ** 2.0
            + (1.0 - x[..., :-1]) ** 2.0,
            axis=-1,
        )


def run(label, seed, **proposal_kwargs):
    model = RosenbrockModel(DIMS)
    fs = FlowSampler(
        model,
        output=f"{BASE_OUTPUT}{label}_{seed}",
        flow_config=dict(n_blocks=4, n_neurons=10, n_layers=3),
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
        f"seed {seed:>6}: "
        f"acceptance {np.nanmean(b['accept']):.4f} -> "
        f"{np.nanmean(a['accept']):.4f} | "
        f"logZ {b['logZ']:.3f} -> {a['logZ']:.3f} | "
        f"n_like {b['n_like']} -> {a['n_like']} | "
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
axs[0].set_title(
    f"acceptance: {b_acc.mean():.3f} -> {a_acc.mean():.3f}"
)
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
