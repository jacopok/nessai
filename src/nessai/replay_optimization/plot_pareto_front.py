#!/usr/bin/env python
"""Plot the Pareto front found by ``replay_optimisation.py``.

Reads the completed trials from the shared Optuna storage, highlights the
Pareto-optimal ones and the archived configuration (trial 0, enqueued by
``replay_optimisation.main``), and adds a secondary axis converting the cost
objective (likelihood calls per iteration) into an estimated total cost for the
whole run (likelihood calls per iteration times the run's iteration count).

The two objectives are plotted directly: the KS-test p-value of the pooled
insertion indices (objective 1, *maximised*) on a log x-axis, and the
likelihood calls per iteration (objective 2, minimised) on a log y-axis.  A
vertical line marks the ``CUTOFF`` p-value below which the insertion-index KS
test is considered failed.

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
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from replay_optimisation import (
    BASELINE_FLOW_CONFIG,
    STORAGE,
    STUDY_NAME,
    load_archived_run,
)

INK = "#0b0b0b"
MUTED = "#767267"
GRID = "#e1e0d9"
LEGEND_GRAY = "#9a978f"
CUTOFF = 0.05
CUTOFF_COLOR = "#d03b3b"
FRONT_COLOR = "#d03b3b"


def _network_size(params: dict) -> float:
    """Rough coupling-flow parameter count: the transform networks dominate and
    scale as ``n_blocks * n_layers * n_neurons ** 2``.  Used only to rank
    trials by network size for the marker-size encoding."""
    n_blocks = params.get("n_blocks", BASELINE_FLOW_CONFIG["n_blocks"])
    n_layers = params.get("n_layers", BASELINE_FLOW_CONFIG["n_layers"])
    n_neurons = params.get("n_neurons", BASELINE_FLOW_CONFIG["n_neurons"])
    return n_blocks * n_layers * n_neurons ** 2


def _size_to_area(size: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Map network sizes to scatter marker areas (points**2), linearly in
    ``sqrt(size)`` so marker *width* tracks network scale."""
    lo, hi = np.sqrt(ref.min()), np.sqrt(ref.max())
    frac = (np.sqrt(size) - lo) / (hi - lo) if hi > lo else np.zeros_like(size)
    return 12.0 + frac * (240.0 - 12.0)


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
    p_value = np.array([t.user_attrs["ks_p_value"] for t in trials])
    # Objective 2 is stored as log10(likelihood calls / iteration).
    calls = np.array([10.0 ** t.values[1] for t in trials])
    net_size = np.array([_network_size(t.params) for t in trials])
    marker_area = _size_to_area(net_size, net_size)

    pareto_numbers = {t.number for t in study.best_trials}
    on_front = np.array([n in pareto_numbers for n in numbers])
    is_baseline = numbers == 0
    is_dominated = ~on_front & ~is_baseline

    order = np.argsort(p_value[on_front])
    front_p = p_value[on_front][order]
    front_calls = calls[on_front][order]
    front_area = marker_area[on_front][order]

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

    baseline_s = 170

    # Marker size encodes the flow's network size (see ``_network_size``).
    ax.scatter(
        p_value[is_dominated],
        calls[is_dominated],
        s=marker_area[is_dominated],
        color=LEGEND_GRAY,
        alpha=0.75,
        linewidths=0,
        zorder=2,
    )
    ax.plot(
        front_p,
        front_calls,
        color=FRONT_COLOR,
        lw=1.3,
        ls=(0, (5, 3)),
        zorder=3,
    )
    ax.scatter(
        front_p,
        front_calls,
        s=front_area,
        facecolors="white",
        edgecolors=INK,
        linewidths=0.9,
        zorder=4,
    )
    ax.scatter(
        p_value[is_baseline],
        calls[is_baseline],
        s=baseline_s,
        marker="*",
        facecolors=LEGEND_GRAY,
        edgecolors=INK,
        linewidths=0.8,
        zorder=5,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(1e3, 1e9)
    # Minor-tick labels crowd each other on a narrow p-value range; the major
    # decade labels are enough.
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("KS test p-value of insertion indices")
    ax.set_ylabel("Likelihood calls / iteration")
    ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.axvline(CUTOFF, color=CUTOFF_COLOR, lw=1.0, ls=":", zorder=1)
    ax.text(
        CUTOFF,
        1.01,
        f"p = {CUTOFF:.2f}",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8,
        color=CUTOFF_COLOR,
    )

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
            linestyle=(0, (5, 3)),
            markersize=6,
            markeredgecolor=INK,
            markerfacecolor="white",
            color=FRONT_COLOR,
            label="Pareto front",
        ),
        Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            markersize=12,
            markeredgecolor=INK,
            markeredgewidth=0.8,
            markerfacecolor=LEGEND_GRAY,
            label="archived config (trial 0)",
        ),
        Line2D(
            [],
            [],
            color=CUTOFF_COLOR,
            lw=1.0,
            ls=":",
            label=f"KS test fails (p < {CUTOFF:.2f})",
        ),
    ]
    main_legend = ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower left",
        fontsize=8.5,
        handletextpad=0.5,
    )
    ax.add_artist(main_legend)

    # Secondary legend: marker size -> flow network size.
    size_ticks = np.array(
        [net_size.min(), np.sqrt(net_size.min() * net_size.max()), net_size.max()]
    )
    size_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=LEGEND_GRAY,
            markeredgecolor="none",
            markersize=np.sqrt(_size_to_area(np.array([s]), net_size)[0]),
            label=f"~{s / 1e3:,.0f}k params",
        )
        for s in size_ticks
    ]
    ax.legend(
        handles=size_handles,
        title="flow network size",
        frameon=False,
        loc="upper left",
        fontsize=8,
        title_fontsize=8.5,
        labelspacing=1.1,
        handletextpad=0.8,
        borderpad=1.0,
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
