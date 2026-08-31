#!/usr/bin/env python
"""Pick one configuration off the Pareto front found by ``replay_optimisation``.

Selection rule, in decreasing order of importance:

1. **Likelihood efficiency.**  Fewest likelihood calls per nested-sampling
   iteration (objective 2 is the base-10 log of this; it is converted back to
   a raw call count here).
2. **Small flow.**  Fewest trainable parameters.
3. **Small KS statistic** (objective 1), subject to the hard constraint that
   the insertion-index KS test must *pass*: ``ks_p_value >= --p-threshold``.

Criteria 1 and 2 are applied with a relative tolerance (``--tolerance``): a
config within that fraction of the best value is treated as tied on that
criterion, so the choice falls through to the next one.  Without it the
ranking would be decided entirely by criterion 1, since two trials never
have exactly equal call counts.

Usage
-----
    python select_config.py
    python select_config.py --p-threshold 0.05 --tolerance 0.15
    python select_config.py --output selected_config.py
"""

from __future__ import annotations

import argparse
import pprint
from pathlib import Path

import optuna

from replay_optimisation import STORAGE, STUDY_NAME, suggest_configs

BASELINE_NUMBER = 0  # trial 0 is the archived run, enqueued by replay_optimisation


def _improvement(name: str, baseline: float, selected: float, unit: str = "") -> str:
    """One line comparing a selected metric to the baseline's, as a ratio."""
    if selected < baseline:
        change = f"{baseline / selected:.1f}x better"
    elif selected > baseline:
        change = f"{selected / baseline:.1f}x worse"
    else:
        change = "unchanged"
    return (
        f"  {name:<22}: {baseline:>12,.4g}{unit}  ->  "
        f"{selected:>12,.4g}{unit}   ({change})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", default=STORAGE)
    parser.add_argument("--study-name", default=STUDY_NAME)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=0.15)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "If given, write a Python file defining flow_config, "
            "training_config and proposal_overrides for the selected trial, "
            "ready to paste into a run script."
        ),
    )
    args = parser.parse_args()

    study = optuna.load_study(
        study_name=args.study_name, storage=args.storage
    )

    # Objective 2 is stored as log10(likelihood calls / iteration).
    calls = lambda t: 10.0 ** t.values[1]
    n_params = lambda t: t.user_attrs["n_flow_parameters"]
    ks = lambda t: t.user_attrs["ks_statistic"]

    candidates = [
        t
        for t in study.best_trials
        if t.user_attrs["ks_p_value"] >= args.p_threshold
    ]
    if not candidates:
        raise SystemExit(
            f"No Pareto-front trial passes the KS test at "
            f"p >= {args.p_threshold}."
        )

    within = lambda ts, key: [
        t for t in ts if key(t) <= min(key(u) for u in ts) * (1 + args.tolerance)
    ]
    shortlist = within(within(candidates, calls), n_params)
    winner = min(shortlist, key=ks)

    print(
        f"{len(candidates)} of {len(study.best_trials)} front trials pass "
        f"the KS test (p >= {args.p_threshold}).\n"
    )
    print(f"Selected trial {winner.number}:")
    a = winner.user_attrs
    print(f"  KS statistic          : {ks(winner):.4f}  (p = {a['ks_p_value']:.3f})")
    print(f"  likelihood calls / it : {calls(winner):,.1f}")
    print(f"  trainable parameters  : {n_params(winner):,}")
    print(f"  coverage              : {a['coverage']:.3f}")

    baseline = next(
        (t for t in study.trials if t.number == BASELINE_NUMBER), None
    )
    if baseline is not None and baseline.values is not None:
        b = baseline.user_attrs
        print(f"\nImprovement over the archived configuration (trial {BASELINE_NUMBER}):")
        print(_improvement("likelihood calls / it", calls(baseline), calls(winner)))
        print(_improvement("trainable parameters", n_params(baseline), n_params(winner)))
        print(_improvement("KS statistic", ks(baseline), ks(winner)))
        print(_improvement("KS p-value", b["ks_p_value"], a["ks_p_value"]))
        print(
            _improvement(
                "train time / checkpoint",
                b["train_time"] / b["n_checkpoints"],
                a["train_time"] / a["n_checkpoints"],
                unit=" s",
            )
        )

    if args.output is not None:
        import nessai

        flow_config, training_config, overrides = suggest_configs(
            optuna.trial.FixedTrial(winner.params)
        )
        # Expand the replay's compact ``proposal_overrides`` into the kwargs a
        # real FlowProposal / sampler actually takes -- mirrors
        # replay_optimisation._apply_proposal_overrides.
        proposal_kwargs = {
            "latent_temperature": overrides["latent_temperature"],
            "truncation_methods": ["latent_radius"],
            "truncation_kwargs": {"latent_radius": overrides["latent_radius"]},
        }
        fmt = lambda d: pprint.pformat(d, indent=1, sort_dicts=False, width=88)
        header = (
            f'"""Hyperparameters selected by select_config.py from the '
            f"replay-optimisation Pareto front (trial {winner.number}).\n\n"
            f"KS statistic {ks(winner):.4f} (p = {a['ks_p_value']:.3f}), "
            f"{calls(winner):,.0f} likelihood calls/iteration, "
            f"{n_params(winner):,} trainable flow parameters.\n\n"
            f"Tuned against nessai {nessai.__version__}; the proposal settings "
            f"below (latent_temperature alongside constant_volume_mode) are "
            f"only mutually consistent on that version -- run production on the "
            f"same nessai, or re-tune.\n\"\"\"\n"
        )
        args.output.write_text(
            f"{header}\n"
            f"flow_config = {fmt(flow_config)}\n\n"
            f"training_config = {fmt(training_config)}\n\n"
            f"proposal_kwargs = {fmt(proposal_kwargs)}\n"
        )
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
