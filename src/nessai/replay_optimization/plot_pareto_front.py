#!/usr/bin/env python
"""Plot the Pareto front found by ``replay_optimisation.py``.

Reads the completed trials from the shared Optuna storage, highlights the
Pareto-optimal ones and the archived configuration (trial 0, enqueued by
``replay_optimisation.main``), colors every point by its own exact KS-test
p-value, and adds a secondary axis converting the cost objective (likelihood
calls per iteration) into an estimated total cost for the whole run
(likelihood calls per iteration times the run's iteration count).

The p-value is not a function of the KS statistic alone -- it also depends on
the pooled, coverage-masked sample size, which varies trial to trial -- so it
is encoded as a genuine per-point color rather than a second functional axis
(a fixed conversion using e.g. the median sample size is off by orders of
magnitude for low-coverage trials).

Usage
-----
    python plot_pareto_front.py --result run_result.json
    python plot_pareto_front.py --result run_result.json --output front.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

from replay_optimisation import STORAGE, STUDY_NAME, load_archived_run

INK = "#0b0b0b"
MUTED = "#767267"
GRID = "#e1e0d9"
LEGEND_GRAY = "#9a978f"
CUTOFF = 0.05
CUTOFF_COLOR = "#d03b3b"

# Sequential blue ramp (light -> dark), from the validated palette; reversed
# so the darkest step lands on the smallest (most significant) p-values.
P_VALUE_CMAP = LinearSegmentedColormap.from_list(
    "p_value",
    [
        "#0d366b",
        "#184f95",
        "#256abf",
        "#3987e5",
        "#6da7ec",
        "#9ec5f4",
        "#cde2fb",
    ][::-1],
)


def _load_completed_trials(study_name: str, storage: str) -> list:
    study = optuna.load_study(study_name=study_name, storage=storage)
    return [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ], study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        required=True,
        help=(
            "Bilby result file for the replayed run, used only to recover "
            "its iteration count for the total-cost axis."
        ),
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--storage", default=STORAGE)
    parser.add_argument("--study-name", default=STUDY_NAME)
    parser.add_argument(
        "--output", type=Path, default=Path("pareto_front.pdf")
    )
    args = parser.parse_args()

    run = load_archived_run(args.result, args.config)
    trials, study = _load_completed_trials(args.study_name, args.storage)

    numbers = np.array([t.number for t in trials])
    ks = np.array([t.values[0] for t in trials])
    calls = np.array([t.values[1] for t in trials])
    p_value = np.array([t.user_attrs["ks_p_value"] for t in trials])

    pareto_numbers = {t.number for t in study.best_trials}
    on_front = np.array([n in pareto_numbers for n in numbers])
    is_baseline = numbers == 0
    is_dominated = ~on_front & ~is_baseline

    order = np.argsort(ks[on_front])
    front_ks = ks[on_front][order]
    front_calls = calls[on_front][order]
    front_p = p_value[on_front][order]

    norm = LogNorm(vmin=max(p_value.min(), 1e-8), vmax=1.0)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.5,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.7), dpi=300)

    dominated_s = 13
    front_s = 48
    baseline_s = 170

    below_dominated = is_dominated & (p_value < CUTOFF)
    below_front = front_p < CUTOFF
    below_baseline = is_baseline & (p_value < CUTOFF)

    # Halo rings flagging p < cutoff, drawn under each group's own markers so
    # they show as an outer ring rather than covering the fill or edge.
    ax.scatter(
        ks[below_dominated],
        calls[below_dominated],
        s=dominated_s * 2.4,
        facecolors="none",
        edgecolors=CUTOFF_COLOR,
        linewidths=1.0,
        zorder=1.5,
    )
    ax.scatter(
        front_ks[below_front],
        front_calls[below_front],
        s=front_s * 1.9,
        facecolors="none",
        edgecolors=CUTOFF_COLOR,
        linewidths=1.2,
        zorder=3.5,
    )
    ax.scatter(
        ks[below_baseline],
        calls[below_baseline],
        s=baseline_s * 1.7,
        marker="*",
        facecolors="none",
        edgecolors=CUTOFF_COLOR,
        linewidths=1.2,
        zorder=4.5,
    )

    ax.scatter(
        ks[is_dominated],
        calls[is_dominated],
        c=p_value[is_dominated],
        cmap=P_VALUE_CMAP,
        norm=norm,
        s=dominated_s,
        alpha=0.75,
        linewidths=0,
        zorder=2,
    )
    ax.plot(
        front_ks,
        front_calls,
        color=MUTED,
        lw=1.0,
        zorder=3,
    )
    ax.scatter(
        front_ks,
        front_calls,
        c=front_p,
        cmap=P_VALUE_CMAP,
        norm=norm,
        s=front_s,
        edgecolors=INK,
        linewidths=0.7,
        zorder=4,
    )
    ax.scatter(
        ks[is_baseline],
        calls[is_baseline],
        c=p_value[is_baseline],
        cmap=P_VALUE_CMAP,
        norm=norm,
        s=baseline_s,
        marker="*",
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("KS statistic of insertion indices")
    ax.set_ylabel("Likelihood calls / iteration")
    ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    secax = ax.secondary_yaxis(
        "right",
        functions=(
            lambda c: c * run.n_iterations,
            lambda cost: cost / run.n_iterations,
        ),
    )
    secax.set_ylabel(
        f"Estimated total likelihood evaluations\n"
        f"(calls/iteration $\\times$ {run.n_iterations:,} iterations)"
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=P_VALUE_CMAP)
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        location="top",
        pad=0.04,
        fraction=0.055,
        aspect=32,
        shrink=0.92,
    )
    cbar.set_label("KS test p-value (exact, per trial)")
    cbar.outline.set_visible(False)
    cbar.ax.axvline(CUTOFF, color=CUTOFF_COLOR, lw=1.4, zorder=5)
    cbar.ax.text(
        CUTOFF,
        1.35,
        f"{CUTOFF:.2f}",
        transform=cbar.ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8,
        color=CUTOFF_COLOR,
    )

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4.5,
            color=LEGEND_GRAY,
            label="dominated trials",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="-",
            markersize=6,
            markeredgecolor=INK,
            markerfacecolor=LEGEND_GRAY,
            color=MUTED,
            label="Pareto front",
        ),
        Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            markersize=12,
            markeredgecolor="white",
            markeredgewidth=0.8,
            markerfacecolor=LEGEND_GRAY,
            label="archived config (trial 0)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=8,
            markerfacecolor="none",
            markeredgecolor=CUTOFF_COLOR,
            markeredgewidth=1.2,
            label=f"p < {CUTOFF:.2f}",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower left",
        fontsize=8.5,
        handletextpad=0.5,
    )
    fig.suptitle(
        "Fidelity-cost trade-off across replayed flow configurations",
        fontsize=11,
        y=0.985,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
