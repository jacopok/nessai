"""Retroactive latent-temperature optimisation on a toy problem.

A single batch of latent samples is drawn from a truncated isotropic Gaussian
at temperature ``T = 1`` and pushed through an *identity* flow. The target is a
broader isotropic Gaussian (variance ``target_var``). The optimal latent
temperature is exactly ``target_var``; we recover it by reweighting the single
batch, without redrawing.

Run: ``python examples/latent_temperature_optimisation.py``
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from nessai.utils.temperature import (
    log_truncation_fraction,
    optimise_latent_temperature,
)

DIMS = 2
RADIUS = 6.0
TARGET_VAR = 2.5
N = 50000

rng = np.random.default_rng(0)


def draw_batch(temperature):
    """Draw a single batch at ``temperature`` through the identity flow."""
    z = rng.normal(scale=np.sqrt(temperature), size=(3 * N, DIMS))
    r = np.linalg.norm(z, axis=1)
    z, r = z[r <= RADIUS][:N], r[r <= RADIUS][:N]
    log_p = multivariate_normal.logpdf(
        z, mean=np.zeros(DIMS), cov=temperature
    ) - log_truncation_fraction(DIMS, RADIUS, temperature)
    log_rho = multivariate_normal.logpdf(
        z, mean=np.zeros(DIMS), cov=TARGET_VAR
    )
    return r, log_rho - log_p


# Step 6: iterate, redrawing at each estimate, until the optimum sits
# comfortably inside the trust region.
temperature = 1.0
for iteration in range(10):
    r, log_w = draw_batch(temperature)
    result = optimise_latent_temperature(
        log_w,
        r,
        DIMS,
        temperature=temperature,
        radius=RADIUS,
        tau=0.5,
        criterion="efficiency",
    )
    print(
        f"iter {iteration}: T={temperature:.3f} -> T_hat*={result['temperature']:.3f}"
        f"  (boundary={result['at_trust_boundary']})"
    )
    temperature = result["temperature"]
    if not result["at_trust_boundary"]:
        break

print(f"\ntrue optimal temperature : {TARGET_VAR}")
print(f"estimated T_hat*         : {result['temperature']:.3f}")

grid = result["grid"]
trust = result["trust_region"]
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(
    grid, np.exp(result["criterion"]), "C0", label="rejection efficiency"
)
ax1.axvline(TARGET_VAR, color="k", ls=":", label="true optimum")
ax1.axvline(result["temperature"], color="C3", ls="--", label=r"$\hat{T}^*$")
ax1.fill_between(
    grid, 0, 1, where=trust, color="C2", alpha=0.15, label="trust region"
)
ax1.set_xlabel(r"$T'$")
ax1.set_ylabel("rejection efficiency  mean(w) / max(w)")
ax1.set_xscale("log")
ax2 = ax1.twinx()
ax2.plot(grid, result["secondary_ess"] / N, "C1", alpha=0.7)
ax2.set_ylabel(r"ESS$_{\rm sec}$ / N", color="C1")
ax1.legend(loc="upper right")
fig.tight_layout()
fig.savefig("latent_temperature_optimisation.png", dpi=120)
print("saved latent_temperature_optimisation.png")
