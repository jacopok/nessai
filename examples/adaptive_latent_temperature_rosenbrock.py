"""Compare FlowProposal with and without adaptive latent-temperature.

Runs nessai twice on a 5-D Rosenbrock likelihood (uniform prior): once with the
default proposal and once with ``adapt_latent_temperature=True``, then compares
the per-population proposal acceptance.

Run: ``python examples/adaptive_latent_temperature_rosenbrock.py``
"""

import matplotlib.pyplot as plt
import numpy as np

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import configure_logger

DIMS = 5
SEED = 1451
BASE_OUTPUT = "./outdir/adaptive_latent_temperature_rosenbrock/"

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


def run(label, **proposal_kwargs):
    model = RosenbrockModel(DIMS)
    fs = FlowSampler(
        model,
        output=BASE_OUTPUT + label,
        flow_config=dict(n_blocks=4, n_neurons=10, n_layers=3),
        resume=False,
        seed=SEED,
        plot=False,
        **proposal_kwargs,
    )
    fs.run(plot=False)
    ns = fs.ns
    accept = np.asarray(ns.history["population_acceptance"], dtype=float)
    temps = getattr(fs.ns._flow_proposal, "latent_temperature_history", [])
    print(
        f"[{label}] logZ = {fs.log_evidence:.3f} +/- {fs.log_evidence_error:.3f}"
        f" | likelihood evals = {ns.total_likelihood_evaluations}"
        f" | mean proposal acceptance = {np.nanmean(accept):.4f}"
    )
    if temps:
        print(f"[{label}] final latent_temperature = {temps[-1]:.3f}")
    return accept, temps


baseline_acc, _ = run("baseline")
adaptive_acc, adaptive_temps = run(
    "adaptive", adapt_latent_temperature=True
)

fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=False)
axs[0].plot(baseline_acc, label="baseline", marker=".")
axs[0].plot(adaptive_acc, label="adaptive T", marker=".")
axs[0].set_xlabel("population index")
axs[0].set_ylabel("proposal acceptance")
axs[0].legend()
axs[1].plot(adaptive_temps, marker=".", color="C1")
axs[1].axhline(1.0, color="k", ls=":", label="baseline T = 1")
axs[1].set_xlabel("adaptation step")
axs[1].set_ylabel("latent_temperature")
axs[1].legend()
fig.tight_layout()
fig.savefig("adaptive_latent_temperature_rosenbrock.png", dpi=120)
print("saved adaptive_latent_temperature_rosenbrock.png")
