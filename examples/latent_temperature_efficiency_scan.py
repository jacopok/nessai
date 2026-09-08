"""Diagnostic: realised vs predicted proposal efficiency across latent T.

Trains a single FlowProposal on a fixed Rosenbrock likelihood contour, then:

* draws one batch at T = 1 and uses ``optimise_latent_temperature`` to
  *predict* the rejection efficiency at a grid of temperatures;
* for the same grid, actually re-populates the pool at that fixed
  ``latent_temperature`` and *measures* ``population_acceptance`` and the
  fraction of latent draws lost to nan-discard / prior-bounds rejection.

The gap between the two curves is the point: the predictor only sees samples
that survived those filters, so it over-estimates efficiency as T grows and
more draws fall outside the prior box.

Run: ``python examples/latent_temperature_efficiency_scan.py``
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from nessai.livepoint import numpy_array_to_live_points
from nessai.model import Model
from nessai.proposal import FlowProposal
from nessai.utils import configure_logger
from nessai.utils.temperature import log_reweight_factor

DIMS = 5
NLIVE = 2000
SEED = 1234
OUTPUT = "./outdir/latent_temperature_efficiency_scan/"

configure_logger(output=OUTPUT, log_level="WARNING")
rng = np.random.default_rng(SEED)


class RosenbrockModel(Model):
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


model = RosenbrockModel(DIMS)
model.set_rng(rng)

# Build a likelihood contour: draw from the prior, keep the best NLIVE.
pool = model.new_point(N=40 * NLIVE)
pool["logL"] = model.log_likelihood(pool)
pool["logP"] = model.log_prior(pool)
live_points = np.sort(pool, order="logL")[-NLIVE:]
worst_point = live_points[0].copy()
print(f"contour logL threshold = {worst_point['logL']:.2f}")

proposal = FlowProposal(
    model,
    output=OUTPUT,
    flow_config=dict(n_blocks=4, n_neurons=10, n_layers=3),
    poolsize=5000,
    plot=False,
    rng=rng,
)
proposal.initialise()
proposal.train(live_points)

# One batch at T = 1 -> predicted efficiency curves.
proposal.latent_temperature = 1.0
proposal._truncation_scheme.prepare(proposal, worst_point, radius=None)
z_full = np.asarray(
    proposal.sample_latent_distribution(20000), dtype=float
)
z = proposal._truncation_scheme.apply_latent(proposal, z_full.copy())
x, log_q, z = proposal.backward_pass(z, rescale=True, return_z=True)
x, log_q, z = proposal._truncation_scheme.apply_after_backward(
    proposal, x, log_q, z
)
log_w_surv = proposal.compute_weights(x, log_q)

r_full = np.sqrt(np.sum(z_full**2, axis=-1))
r_surv = np.sqrt(np.sum(np.asarray(z, dtype=float) ** 2, axis=-1))
# align survivors back to the full batch; lost draws get weight 0 (-inf)
idx = {row.tobytes(): i for i, row in enumerate(z_full)}
log_w_full = np.full(z_full.shape[0], -np.inf)
for row, w in zip(np.asarray(z, dtype=float), log_w_surv):
    j = idx.get(row.tobytes())
    if j is not None:
        log_w_full[j] = w

grid = np.geomspace(0.5, 20.0, 25)
pred_eff = []  # survivors only (what the naive predictor sees)
pred_eff_full = []  # counting lost draws as zero weight
for tp in grid:
    ls_s = log_reweight_factor(r_surv, proposal.prime_dims, 1.0, tp)
    lw = log_w_surv - ls_s
    lw = lw[np.isfinite(lw)]
    pred_eff.append(np.mean(np.exp(lw - lw.max())))

    ls_f = log_reweight_factor(r_full, proposal.prime_dims, 1.0, tp)
    lwf = log_w_full - ls_f
    m = np.max(lwf[np.isfinite(lwf)])
    pred_eff_full.append(np.mean(np.exp(lwf - m)))

# Realised efficiency: actually populate at each fixed T.
real_acc, lost_frac = [], []
for tp in grid:
    proposal.adapt_latent_temperature = False
    proposal.latent_temperature = float(tp)
    n_draw = 40000
    zt = proposal.sample_latent_distribution(n_draw)
    zt = proposal._truncation_scheme.apply_latent(proposal, zt)
    xt, log_qt, zt = proposal.backward_pass(zt, rescale=True, return_z=True)
    xt, log_qt, zt = proposal._truncation_scheme.apply_after_backward(
        proposal, xt, log_qt, zt
    )
    lost_frac.append(1.0 - len(xt) / n_draw)
    lw = proposal.compute_weights(xt, log_qt)
    lw = lw[np.isfinite(lw)]
    # efficiency among survivors
    surv_eff = np.mean(np.exp(lw - lw.max()))
    # efficiency including the lost draws as zero-weight
    real_acc.append(surv_eff * len(xt) / n_draw)
    k = len(real_acc) - 1
    print(
        f"T={tp:6.2f}  pred_survivors={pred_eff[k]:.3f}"
        f"  pred_with_losses={pred_eff_full[k]:.3f}"
        f"  realised={real_acc[-1]:.3f}  lost={lost_frac[-1]:.3f}"
    )

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(grid, pred_eff_full, "C2-o", label="predicted (losses as zero weight)")
ax.plot(grid, pred_eff, "C0-o", label="predicted (survivors only, from T=1)")
ax.plot(grid, real_acc, "C1-o", label="realised (incl. lost draws)")
ax.plot(grid, lost_frac, "C3--", label="fraction of draws lost")
ax.axvline(1.0, color="k", ls=":", label="baseline T=1")
ax.set_xlabel("latent_temperature")
ax.set_ylabel("efficiency")
ax.set_xscale("log")
ax.legend()
fig.tight_layout()
plot_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "latent_temperature_efficiency_scan.png",
)
fig.savefig(plot_path, dpi=120)
print(f"saved {plot_path}")
