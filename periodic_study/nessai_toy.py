"""Nested sampling with nessai on the toy targets.

The likelihood is a toy target density on (phi, psi, x, y) and the prior is
uniform on [0, 2 pi) x [0, pi) x [-6, 6]^2, so the true log-evidence is
``-log(prior volume) + log(mass of the target in the box)`` and ``0`` after
the prior volume is added back (the target's mass outside |x|, |y| < 6 is
below 1e-8).

Configurations (v18 flow size: 6 coupling layers, 2 blocks of 64 neurons,
LU and batch norm between layers):

* ``chi``: ``angle-2pi`` / ``angle-pi`` for phi / psi (an auxiliary chi(2)
  radius each; nessai-gw's default for phase and polarisation), RealNVP;
* ``box``: ``rescale-to-bounds`` for phi / psi, RealNVP;
* ``circular``: ``circular`` for phi / psi, the circular flow (splines),
  rotation from ``atan2``; ``circular-real``: rotation from one output.

nessai's default truncation (a latent ball over the Gaussian latent
dimensions) is used unless the configuration ends in ``-lpt``, which
truncates on the flow density instead.

Usage: python nessai_toy.py OUTDIR CASE CONFIG SEED [NLIVE]
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import configure_logger

sys.path.insert(0, str(Path(__file__).parent))
import toy_benchmark as tb  # noqa: E402

BOUNDS = {"phi": [0, 2 * math.pi], "psi": [0, math.pi], "x": [-6, 6], "y": [-6, 6]}
LOG_VOLUME = float(sum(np.log(np.ptp(b)) for b in BOUNDS.values()))

FLOW = {
    "n_blocks": 6,
    "n_layers": 2,
    "n_neurons": 64,
    "batch_norm_between_layers": True,
    "linear_transform": "lu",
}
TRAINING = {
    "lr": 1e-3,
    "batch_size": 1000,
    "max_epochs": 500,
    "patience": 20,
    "optimiser": "adamw",
    "val_size": 0.1,
    "clip_grad_norm": 5.0,
}
CONFIGS = {
    "chi": (
        {"phi": "angle-2pi", "psi": "angle-pi"},
        {"ftype": "realnvp"},
    ),
    "box": (
        {"phi": "rescale-to-bounds", "psi": "rescale-to-bounds"},
        {"ftype": "realnvp"},
    ),
    "circular": (
        {"phi": "circular", "psi": "circular"},
        {"ftype": "circular", "circular_shift": "angle"},
    ),
    "circular-real": (
        {"phi": "circular", "psi": "circular"},
        {"ftype": "circular", "circular_shift": "real"},
    ),
}


class ToyModel(Model):
    def __init__(self, target):
        self.target = target
        self.names = list(BOUNDS)
        self.bounds = BOUNDS

    def log_prior(self, x):
        return np.log(self.in_bounds(x), dtype=float) - LOG_VOLUME

    def log_likelihood(self, x):
        X = np.stack([x[n] for n in self.names], axis=-1).reshape(-1, 4)
        # the targets exclude the upper end of each angle's range
        X[:, 0] = np.where(X[:, 0] >= 2 * math.pi, 0.0, X[:, 0])
        X[:, 1] = np.where(X[:, 1] >= math.pi, 0.0, X[:, 1])
        return self.target.log_prob(X).reshape(np.shape(x))


def main(outdir, case, config, seed, nlive=1000):
    seed, nlive = int(seed), int(nlive)
    output = Path(outdir) / f"{case}_{config}_{seed}"
    configure_logger(output=str(output), log_level=__import__("os").environ.get("NS_LOG", "WARNING"))
    # "-lpt": truncate on the flow density (the 0.5% quantile of the live
    # points' log q, as the xg_pe runs) instead of the latent ball
    base, lpt = (config[:-4], True) if config.endswith("-lpt") else (config, False)
    reparams, flow = CONFIGS[base]
    truncation = {}
    if lpt:
        truncation = dict(
            truncation_methods=["log_proposal_threshold"],
            truncation_kwargs={"log_proposal_threshold": {"quantile": 0.005}},
        )
    target = tb.TARGETS[case]()
    fs = FlowSampler(
        ToyModel(target),
        output=str(output),
        resume=False,
        seed=seed,
        nlive=nlive,
        reparameterisations=reparams,
        flow_config=FLOW | flow,
        training_config=TRAINING,
        plot=False,
        checkpointing=False,
        **truncation,
    )
    t0 = time.time()
    fs.run(plot=False, save=False)
    wall = time.time() - t0

    post = fs.posterior_samples
    P = np.stack([post[n] for n in BOUNDS], axis=-1)
    T = target.sample(200_000, np.random.default_rng(seed + 1))
    ks = {
        n: float(stats.ks_2samp(P[:, i], T[:, i]).pvalue)
        for i, n in enumerate(BOUNDS)
    }
    # the measured combination of the stripe, phi + 2 psi
    u_p = np.mod(P[:, 0] + 2 * P[:, 1], 2 * math.pi)
    u_t = np.mod(T[:, 0] + 2 * T[:, 1], 2 * math.pi)
    ks["phi+2psi"] = float(stats.ks_2samp(u_p, u_t).pvalue)
    row = dict(
        case=case,
        config=config,
        seed=seed,
        nlive=nlive,
        log_z=float(fs.log_evidence + LOG_VOLUME),
        log_z_err=float(fs.log_evidence_error),
        n_likelihood=int(fs.ns.total_likelihood_evaluations),
        sampling_time=float(fs.ns.sampling_time.total_seconds()),
        wall=wall,
        n_posterior=int(len(post)),
        ks=ks,
    )
    print(json.dumps(row))
    with open(Path(outdir) / f"{case}_{config}_{seed}.json", "w") as f:
        json.dump(row, f)


if __name__ == "__main__":
    main(*sys.argv[1:])
