# Cost-based retraining decision: investigation

Code: `nessai.samplers.retrain` (enabled with `retrain_decision=True` on
`NestedSampler`/`FlowSampler`).

## Question

Before each nested-sampling iteration: should the flow be retrained (or
reset) now? The target is the total run time, weighted by the cost of the
likelihood, pool population and training.

## Facts established from instrumented runs

1. **The pool is paid for up front.** Every pool point is drawn from the flow
   and its likelihood evaluated when the pool is populated. Retraining throws
   away the unused part of the pool (`FlowProposal.train` sets
   `populated=False`). A retrain decision therefore only makes sense when the
   pool is empty. Fixed-frequency schedules that retrain mid-pool (e.g.
   `training_frequency=500, train_on_empty=False`) used 1.1–1.6× the
   likelihood evaluations of the default train-on-empty schedule with a
   similar number of trainings.

2. **Acceptance decays as `exp(-s / nlive)` between trainings.** Here `s` is
   the number of iterations since training. The measured slope of
   log-acceptance was -1.00 ± 0.04 per `nlive` iterations over three
   e-folds, on every problem tested (`plot_decay.py`). The flow is fixed, so
   its proposal volume is fixed; the acceptance is the fraction of that
   volume inside the contour and scales with X. The exponential model is
   therefore exact, and the rate is known a priori. It is still fitted (with a
   prior at `1/nlive`), because truncation schemes that shrink the proposal
   would change it.

3. **Fresh acceptance after a warm-started retrain is roughly stationary in
   absolute terms.** It does not depend on how long ago the previous training
   was, so it is predicted as a level from recent episodes rather than as a
   multiplicative "improvement" over the pre-training acceptance (the
   improvement would then depend on the gap). The level can drift slowly over
   the run, in both directions.

4. **Training cost grows mildly with the gap** (warm-start epochs ~30 at
   500-iteration gaps vs ~100 at 3000). Under an affine model `T = T0 + T1·gap`,
   the `T1` part cancels exactly in the decision below. Only the fixed part
   matters, so using the mean `T` is slightly conservative (retrains a little
   less often than optimal).

5. **Full resets** cost 3–5× the epochs of a warm start. On the problems
   tested, their fresh acceptance was not better on average and had a much
   larger scatter. There is one exception: in some runs the warm-started
   lineage degrades steadily (bimodal, seed 1: log A from -0.7 to -3.0 over
   the run), and a reset restores it.

## Decision model

With per-pool-point cost `c` (population + likelihood), a flow with fresh
acceptance `A` costs `c/A · exp(k s)` per iteration, with `k ≈ 1/nlive`.
Retraining costs `T`.

**How far to look ahead.** Compare "retrain now, then act optimally" with
"use the current flow for one more pool of `P` points / `n` iterations, then
retrain and act optimally". The number of iterations in a run is fixed by the
stopping criterion, not by the schedule. So after the extra block, the second
scenario is the first one shifted by `n` iterations. The difference between
the two total costs therefore converges to the constant `c·P - g*·n`, where
`g*` is the long-run average cost per iteration of the optimal schedule.
(This is the constant the two exponentials were expected to converge to.
The exponentials only cancel if future retraining is allowed in both
scenarios; if it isn't, both costs diverge.) Because this
is a renewal (machine-replacement) problem, `g*` has a closed form:

    T = ∫_0^∞ (g* - c/A' · e^{k s})^+ ds
    ⇔  x ln x - x + 1 = y,   x = g* A'/c,   y = T k A'/c

`y` is the dimensionless training cost and `τ* = ln(x)/k` is the optimal
interval. For small `y`, `τ* ≈ sqrt(2 T nlive / (c/A'))`, a square-root law.

**Rule.** When the pool is empty, retrain iff `c·P > g*·E[n]`. That is, the
next pool with the current flow would cost more per iteration than the
optimal long-run rate. The effective horizon is therefore one training cycle;
everything beyond it is summarised by `g*`.

**Pool planning.** When `τ*` is shorter than the default pool lasts (expensive
likelihoods), the next pool is capped at the number of points needed to
reach the break-even point. Otherwise the schedule cannot retrain more than
once per pool.

**Finite horizon.** When the estimated number of remaining iterations
(from `dlogZ` and the tolerance) is shorter than `τ*`, the total costs to the
end are compared directly. This stops pointless retraining at the end of the
run.

### Uncertainties (Gaussian)

* **Current flow:** Bayesian weighted least squares on binned log-acceptance
  (`log a = α + β s`), with the prior on `α` taken from the fresh-level
  prediction and the prior on `β` from previous episodes (initially
  `-1/nlive ± 30%`). Draws are geometric, so the variance of a binned value
  is `(1-a)/m`.
* **Fresh flow:** the level is the mean of the recent episodes in the current
  lineage. The predictive variance is calibrated on the errors of past
  predictions in the same run, plus a weak prior.
* **Propagation:** costs are convex in `1/a`, so uncertainties enter through
  log-normal moments (`E[1/A'] = exp(-μ + σ²/2)`) and Gauss–Hermite
  quadrature for the pool yield. `c` and `T` enter linearly, so only their
  means matter.

Calibration over the recorded runs:

| quantity | z-score mean | z-score sd | beyond 2σ |
|---|---|---|---|
| log-acceptance at the start of the next pool (648 blocks) | -0.06 | 1.03 | 5.2 % |
| fresh acceptance after retraining (≈700 episodes) | -0.2 … +0.2 | 0.87 … 1.12 | 3.4 … 6.5 % |

(`calibrate_block.py`, `calibrate_fresh.py`.) Before the calibrated
variance, the fresh-acceptance prediction had a z-score sd of 1.6–1.8,
because a sample variance over five episodes underestimates the spread.

### Reset (bonus)

Model `log A_reset = log A_warm + δ`, with a Gaussian posterior on `δ`
built from two sources:

* the degradation of the current lineage relative to its own from-scratch
  first training (weight decreasing with elapsed iterations);
* the outcome of previous resets relative to the preceding flow.

The flow is reset when `g*` computed with `T_reset` and `A_reset` is lower
than `g*` for a warm retrain. A reset that did not help counts against the
next one.

### Costs and determinism

Costs are deterministic counts × unit costs:

* likelihood evaluations × s/evaluation;
* pool points × s/point;
* epochs × training samples × s/(sample·epoch).

The epoch counts come from the run itself and are deterministic for a fixed
seed. With `retrain_costs=dict(likelihood=..., population=..., training=...)`
(or the path of a previous run's `retrain_costs.json`), the schedule is
deterministic: runs with identical costs reproduce the default trajectory
exactly. Missing costs are measured, which makes the schedule depend on the
hardware. The measured values are always tracked, a warning is logged when
one differs from the provided value by more than 2×, and
`retrain_costs.json` is written at the end for the next run.

## Results

RESULTS_PLACEHOLDER

## Reproducing

    python driver.py <configs.jsonl> 4          # runs harness.py per line
    python compare.py <outdir> -p               # paired comparison
    python calibrate_block.py <outdir>
    python calibrate_fresh.py "<outdir>/default_*.json"

`harness.py` monkeypatches the sampler to record draws per iteration,
populations and trainings. A virtual likelihood cost `t_L` is applied
afterwards (virtual time = wall + n_like·t_L). Decision runs receive the
same `t_L` through `retrain_costs`.
