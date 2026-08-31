#!/usr/bin/env python
"""Re-score the existing Pareto front on a *different* archived run.

The Pareto front in ``nessai_replay.sqlite3`` was found by replaying one run
(``BNS_result.json``).  This script takes those same configurations -- every
``study.best_trials`` entry, plus the archived default (trial 0) -- and
evaluates them, unchanged, against another finished run, to check whether the
front transfers and how each config compares to the default *on the new run*.

It is deliberately serial and only runs the O(10) front configurations plus
the default; no Optuna study is created or written.

Usage
-----
    python evaluate_front_on_run.py --result BNS_result_v70.json
    python evaluate_front_on_run.py --result BNS_result_v70.json --config config.json
    python evaluate_front_on_run.py --result BNS_result_v70.json --n-checkpoints 4
    python evaluate_front_on_run.py --result BNS_result_v70.json --max-parameters 500000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import optuna

import replay_optimisation as ro
from replay_optimisation import (
    STORAGE,
    STUDY_NAME,
    SEED,
    ReplayModel,
    evaluate_config,
    get_proposal_class,
    load_archived_run,
    run_looks_like_gw,
    suggest_configs,
)

logging.basicConfig(level="WARNING", format="%(name)s %(levelname)s: %(message)s")


def _front_configs(study_name: str, storage: str):
    """Yield ``(label, seed, known_n_params, flow_config, training_config,
    overrides)`` for the archived default and every Pareto-front trial,
    reconstructed through ``suggest_configs`` exactly as the study scored them
    (the default is the study's trial 0, enqueued from ``baseline_params()``).

    ``known_n_params`` is the trainable-parameter count the study recorded for
    that config -- architecture-only, so it carries over to the new run and can
    be used to skip the largest flows before paying to train them.
    """
    study = optuna.load_study(study_name=study_name, storage=storage)
    trial_0 = next(t for t in study.trials if t.number == 0)

    entries = [("default (trial 0)", SEED, trial_0.params, trial_0)]
    entries += [
        (f"trial {t.number}", SEED + t.number, t.params, t)
        for t in sorted(
            study.best_trials, key=lambda t: t.user_attrs["ks_statistic"]
        )
    ]
    for label, seed, trial_params, trial in entries:
        flow_config, training_config, overrides = suggest_configs(
            optuna.trial.FixedTrial(trial_params)
        )
        known = trial.user_attrs.get("n_flow_parameters")
        yield label, seed, known, flow_config, training_config, overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--storage", default=STORAGE)
    parser.add_argument("--study-name", default=STUDY_NAME)
    parser.add_argument("--n-checkpoints", type=int, default=ro.N_CHECKPOINTS)
    parser.add_argument(
        "--non-gw",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Replay with the core FlowProposal instead of GWFlowProposal. "
        "Left unset, it is guessed from the parameter names.",
    )
    parser.add_argument(
        "--max-parameters",
        type=int,
        default=None,
        help="Skip front configs whose flow has more trainable parameters "
        "than this (the largest few take by far the longest to train).",
    )
    args = parser.parse_args()

    ro.N_CHECKPOINTS = args.n_checkpoints

    run = load_archived_run(args.result, args.config)
    non_gw = (
        (not run_looks_like_gw(run)) if args.non_gw is None else args.non_gw
    )
    proposal_class = get_proposal_class(
        augmented=bool(run.augment_kwargs), non_gw=non_gw
    )
    print(
        f"Replaying {args.result.name}: nlive={run.nlive}, "
        f"{run.n_iterations} iterations, {len(run.names)} dimensions, "
        f"{args.n_checkpoints} checkpoints\n"
    )

    header = f"{'config':<18} {'KS D':>8} {'p-value':>9} {'calls/it':>14} {'train [s]':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for label, seed, known, flow_config, training_config, overrides in (
        _front_configs(args.study_name, args.storage)
    ):
        is_default = label.startswith("default")
        if (
            not is_default
            and args.max_parameters is not None
            and known is not None
            and known > args.max_parameters
        ):
            print(f"{label:<18}   skipped: {known:,} parameters")
            continue

        model = ReplayModel(
            run.priors, run.names, rng=np.random.default_rng(seed)
        )
        try:
            summary = evaluate_config(
                run,
                model,
                flow_config,
                training_config,
                proposal_class,
                seed=seed,
                proposal_overrides=overrides,
            )
        except (RuntimeError, ValueError, optuna.TrialPruned) as exc:
            print(f"{label:<18}   failed: {exc}")
            continue

        rows.append((label, summary))
        print(
            f"{label:<18} {summary['ks_statistic']:>8.4f} "
            f"{summary['ks_p_value']:>9.3f} "
            f"{summary['likelihood_calls_per_iteration']:>14,.1f} "
            f"{summary['train_time']:>10.1f}",
            flush=True,
        )

    if not rows or rows[0][0] != "default (trial 0)":
        return

    base = rows[0][1]
    print(f"\nRelative to the default on {args.result.name}:")
    print(f"{'config':<18} {'KS D':>10} {'calls/it':>12}")
    print("-" * 42)
    for label, summary in rows[1:]:
        d_ratio = summary["ks_statistic"] / base["ks_statistic"]
        c_ratio = (
            base["likelihood_calls_per_iteration"]
            / summary["likelihood_calls_per_iteration"]
        )
        print(
            f"{label:<18} {d_ratio:>9.2f}x {c_ratio:>11.2f}x"
        )
    print("\n(KS D: <1 is better; calls/it: >1 means fewer calls than default)")


if __name__ == "__main__":
    main()
