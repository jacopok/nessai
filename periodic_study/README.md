# Periodic parameters: toy study

Flows for targets with periodic parameters (the gravitational-wave phase and
polarisation angle), compared on four 4-D targets on (phi, psi, x, y) with
phi of period 2 pi and psi of period pi: uniform angles; a sharp phase peak
(kappa 400) whose location sweeps the circle with x; three phase modes whose
weights move with y; and the phi +- 2 psi stripe that winds round the torus.

* `toy_benchmark.py`: flow fits (5000 target draws, the xg_pe `v18` flow
  size), scored by the forward KL and the effective sample size of p / q.
  Methods: RealNVP / NSF on the box (nessai's default), on nessai's `Angle`
  (chi-radius) coordinates, with damped periodic ghost copies (discarded or
  folded), with partition-of-unity ghosts (folded), and the circular flow
  (`nessai.flows.circular.CircularNeuralSplineFlow`).
* `nessai_toy.py`: nested sampling of the same targets as likelihoods
  (true log Z = 0), with the latent-ball or the density truncation.
* `summarise.py`, `sample_figure.py`: tables and figures; `check_circular.py`:
  numerical checks of the circular flow.
* `results/`: the outputs of the runs (`flow_fits.csv`, `nested_sampling.csv`,
  `summary.csv`, `toy_summary.png`, `samples_seed1.png`).

Usage:

    python toy_benchmark.py OUTDIR CASE METHOD SEED
    python nessai_toy.py OUTDIR CASE CONFIG SEED [NLIVE]
    python summarise.py OUTDIR RESULTS_DIR

Summary: on the box, RealNVP bridges the seam and smears the modes. The
circular flow had the best worst case over seeds on every target; ghosts
whose weights form a partition of unity are as good or better on smooth
targets but unstable on sharp peaks. In nested sampling the circular flow
needs the density truncation (`log_proposal_threshold`): a uniform circle has
no latent radius, so the latent ball cannot truncate it.
