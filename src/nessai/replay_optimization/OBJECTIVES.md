# Objectives optimised by `replay_optimisation.py`

`replay_optimisation.py` runs a multi-objective Optuna study (`NSGAIISampler`,
`directions=["minimize", "minimize"]`) over flow architecture and training
settings, replaying an archived nested-sampling run instead of calling the
likelihood again. Each trial trains a flow at several checkpoints
(`iteration`s) sampled uniformly in log prior volume across the run, and is
scored on **two objectives**, both of which are predictable from a trained
flow without new likelihood calls.

Both are computed per checkpoint in `evaluate_checkpoint()` and then combined
across checkpoints in `evaluate_config()`.

## Background: what the proposal actually does

nessai's `FlowProposal` draws latent samples `z`, discards those with
`|z| > r` (the latent-radius truncation, `constant_volume_mode`), maps the
survivors back to the physical space through the flow, and rejection-samples
using `log_w = log_prior - log_q`. The accepted points are therefore
distributed as `prior(x) * 1[x in S]`, where `S` is the region of physical
space the flow maps inside the latent ball. Both objectives are consequences
of what `S` looks like relative to the true constrained prior.

## Objective 1 — Fidelity: insertion-index KS statistic

**What it measures:** whether the flow's proposal distribution matches the
true prior constrained to `logL > logL_min`, i.e. whether the sampler would
produce correctly calibrated insertion indices.

**Why it matters:** this was one of the two things that went wrong in the run
the script was written for — the pooled insertion-index KS test failed badly.
A flow whose support `S` fails to cover part of the constrained prior starves
some ranks of the insertion-index distribution, which is exactly this
pathology.

**How it's computed:**

1. A held-out fraction (`HELDOUT_FRACTION`) of the live points at a checkpoint
   is set aside; the rest (plus the current worst point, which defines the
   likelihood threshold) trains the flow.
2. The trained flow's coverage is checked on the held-out points: each is
   pushed through `proposal.forward_pass` and kept if its latent radius falls
   inside the truncation threshold (`_coverage_mask`). Held-out points are
   exact draws from the true constrained prior, so this reconstructs, without
   new likelihood calls, which parts of that prior the flow's support `S`
   actually reaches.
3. The covered held-out points' archived log-likelihoods are ranked against
   the full live set at that checkpoint (`np.searchsorted`), exactly mirroring
   `NestedSampler.insert_live_point`, to produce insertion indices.
4. Indices from all checkpoints are pooled and scored with nessai's own
   `compute_indices_ks_test`, matching the final (non-rolling) KS test used to
   diagnose the original run.

**Objective value:** the KS statistic `D` of the pooled insertion indices
against the uniform distribution they should follow. Lower is better; `D = 0`
means perfect calibration.

## Objective 2 — Cost: likelihood calls per iteration

**What it measures:** how many likelihood evaluations the sampler would need
per nested-sampling iteration if it used this flow, i.e. the run's dominant
computational cost.

**Why it matters:** the second thing that went wrong in the original run was
that it took a long time. The likelihood is evaluated for every point put in
the proposal pool, so the expected number of likelihood calls per iteration is
`1 / P(logL > logL_min | proposed point)`.

**The underlying formula:** with `m(S)` the prior mass of the flow's support
`S`, `X_i = exp(-i / nlive)` the constrained prior volume at iteration `i`,
and `f` the fraction of live points that fall inside `S`:

```
P(logL > logL_min) = X_i * f / m(S)
```

`f` is the `coverage` computed for objective 1 above. `m(S) = v * E[exp(log_w)]`
over latent draws inside the ball, where `v` is the latent mass inside the
ball — both computable from the flow alone via `_support_statistics()`, which
mirrors `FlowProposal.populate()` up to (but not including) the likelihood
call:

1. Draw `N_POPULATION_DRAWS` latent samples, apply the latent-radius
   truncation, and measure `v` (`latent_mass`) as the surviving fraction.
2. Push the survivors through the flow's backward pass to get physical points
   `x` and `log_q`.
3. Compute rejection-sampling weights `log_w = log_prior - log_q`
   (`proposal.compute_weights`) and estimate `m(S)` via a log-sum-exp over
   these weights.

A perfect flow has `m(S) = v * X_i` and `f = v`, giving exactly one likelihood
call per iteration — the theoretical minimum.

**Objective value:** `likelihood_calls_per_iteration`, computed at each
checkpoint as

```
likelihood_calls = exp(log(m(S)) - log(X_i) - log(f))
```

then averaged across checkpoints (checkpoints are spaced uniformly in log
prior volume, so the mean is the right estimator of the total). Lower is
better.

## Why the two objectives trade off against each other

They are genuinely complementary, not redundant:

- A flow whose support `S` is too **narrow** (tight, low `m(S)`) looks cheap
  (few likelihood calls) but under-covers the true constrained prior, wrecking
  the insertion-index KS statistic.
- A flow whose support `S` is too **broad** keeps insertion indices clean
  (good coverage, good KS) but wastes likelihood calls on points that will be
  rejected.

Optuna searches the Pareto front between these two objectives, with the
archived configuration enqueued as the first trial so the resulting front
always contains the status quo as a reference point.

## Caveats affecting the objectives

- Held-out live points are ranked against the live set they belong to, which
  shifts the insertion index by `O(1 / nlive)`. This shift is identical for
  every trial, so it does not affect comparisons between configurations.
- With `inversion_type="duplicate"`, a physical point has two representations
  in the prime space, and nessai's `log_q` only accounts for one of them.
  `m(S)` — and hence objective 2 — inherits this bias, so its absolute scale
  is approximate; the *ranking* between configurations is not affected.
- Each checkpoint trains a flow from scratch, whereas a real run retrains from
  the previous weights until `reset_flow` triggers. Training from scratch is
  the harder problem, so both objectives are conservative (pessimistic)
  relative to a real run.
