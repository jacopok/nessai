#!/usr/bin/env python
"""Plot how flow *training time* relates to the two replay objectives.

Companion to ``plot_pareto_front.py``.  That figure shows the
fidelity/cost trade-off (KS statistic of the insertion indices vs
likelihood calls per iteration); this one asks a separate question: does a
configuration that scores well on either objective also cost more to
*train*?

Two panels share a common y-axis (mean wall-clock time to train one
checkpoint flow, averaged over the run's checkpoints):

    left   training time vs KS statistic         (objective 1)
    right  training time vs likelihood calls/it  (objective 2)

Every point is colored by the flow's trainable-parameter count, which is
the obvious confounder: bigger flows are slower to train regardless of how
they score.  Each panel is annotated with the Spearman rank correlation
between training time and that panel's objective, so the reader can see at
a glance whether the objectives carry any training-time penalty once the
parameter count is set aside (they largely do not).

Usage
-----
    python plot_training_time.py
    python plot_training_time.py --output training_time.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from replay_optimisation import STORAGE, STUDY_NAME

INK = "#0b0b0b"
MUTED = "#767267"
GRID = "#e1e0d9"
LEGEND_GRAY = "#9a978f"
FIT_COLOR = "#d03b3b"

# Same sequential blue ramp (light -> dark) as plot_pareto_front, here
# mapping the flow's trainable-parameter count.
PARAM_CMAP = LinearSegmentedColormap.from_list(
    "n_params",
    [
        "#cde2fb",
        "#9ec5f4",
        "#6da7ec",
        "#3987e5",
        "#256abf",
        "#184f95",
        "#0d366b",
    ],
)


def _load_completed_trials(study_name: str, storage: str):
    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    return trials, study


def _panel(
    ax,
    x,
    train_time,
    n_params,
    norm,
    on_front,
    is_baseline,
    xlabel,
) -> None:
    is_dominated = ~on_front & ~is_baseline

    ax.scatter(
        x[is_dominated],
        train_time[is_dominated],
        c=n_params[is_dominated],
        cmap=PARAM_CMAP,
        norm=norm,
        s=13,
        alpha=0.75,
        linewidths=0,
        zorder=2,
    )
    ax.scatter(
        x[on_front],
        train_time[on_front],
        c=n_params[on_front],
        cmap=PARAM_CMAP,
        norm=norm,
        s=48,
        edgecolors=INK,
        linewidths=0.7,
        zorder=4,
    )
    ax.scatter(
        x[is_baseline],
        train_time[is_baseline],
        c=n_params[is_baseline],
        cmap=PARAM_CMAP,
        norm=norm,
        s=170,
        marker="*",
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )

    # Least-squares fit in log-log space: a visual guide to the trend, not
    # a model.  Drawn faint and dashed so it never competes with the data.
    lx = np.log10(x)
    ly = np.log10(train_time)
    slope, intercept = np.polyfit(lx, ly, 1)
    xs = np.linspace(lx.min(), lx.max(), 100)
    ax.plot(
        10.0**xs,
        10.0 ** (intercept + slope * xs),
        color=FIT_COLOR,
        lw=1.2,
        ls=(0, (5, 3)),
        zorder=3,
    )

    rho, p_value = spearmanr(x, train_time)
    ax.text(
        0.04,
        0.96,
        f"Spearman $\\rho = {rho:+.2f}$\n$p = {p_value:.2f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=INK,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor=GRID,
            linewidth=0.8,
        ),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", default=STORAGE)
    parser.add_argument("--study-name", default=STUDY_NAME)
    parser.add_argument(
        "--output", type=Path, default=Path("training_time.pdf")
    )
    args = parser.parse_args()

    trials, study = _load_completed_trials(args.study_name, args.storage)

    numbers = np.array([t.number for t in trials])
    ks = np.array([t.values[0] for t in trials])
    calls = np.array([t.values[1] for t in trials])
    n_checkpoints = np.array(
        [t.user_attrs["n_checkpoints"] for t in trials]
    )
    # Stored train_time is the sum over checkpoints; per-checkpoint mean is
    # the figure that transfers to a real run (one flow fit per reset).
    train_time = (
        np.array([t.user_attrs["train_time"] for t in trials]) / n_checkpoints
    )
    n_params = np.array(
        [t.user_attrs["n_flow_parameters"] for t in trials], dtype=float
    )

    pareto_numbers = {t.number for t in study.best_trials}
    on_front = np.array([n in pareto_numbers for n in numbers])
    is_baseline = numbers == 0

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

    fig, axes = plt.subplots(
        1, 2, figsize=(9.0, 4.7), dpi=300, sharey=True
    )
    norm = LogNorm(vmin=n_params.min(), vmax=n_params.max())

    _panel(
        axes[0],
        ks,
        train_time,
        n_params,
        norm,
        on_front,
        is_baseline,
        "KS statistic of insertion indices",
    )
    _panel(
        axes[1],
        calls,
        train_time,
        n_params,
        norm,
        on_front,
        is_baseline,
        "Likelihood calls / iteration",
    )
    axes[0].set_ylabel("Flow training time per checkpoint [s]")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=PARAM_CMAP)
    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="horizontal",
        location="top",
        pad=0.12,
        fraction=0.06,
        aspect=40,
        shrink=0.75,
    )
    cbar.set_label("Trainable flow parameters")
    cbar.outline.set_visible(False)

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
            linestyle="none",
            markersize=6,
            markeredgecolor=INK,
            markeredgewidth=0.7,
            markerfacecolor=LEGEND_GRAY,
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
            color=FIT_COLOR,
            lw=1.2,
            ls=(0, (5, 3)),
            label="log-log least-squares fit",
        ),
    ]
    axes[1].legend(
        handles=legend_handles,
        frameon=False,
        loc="lower right",
        fontsize=8.5,
        handletextpad=0.5,
    )
    fig.suptitle(
        "Training cost vs the two replay objectives",
        fontsize=11,
        y=1.09,
    )

    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
