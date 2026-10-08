"""Summarise the toy benchmark: table and figures.

Usage: python summarise.py GRID_DIR OUT_DIR
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CASES = ["uniform", "peak", "multimodal", "stripe"]
ORDER = [
    "realnvp_box",
    "realnvp_chi",
    "realnvp_ghost_discard",
    "realnvp_ghost_fold",
    "nsf_box",
    "nsf_chi",
    "nsf_ghost_discard",
    "nsf_ghost_fold",
    "realnvp_ghostpou_fold",
    "nsf_ghostpou_fold",
    "circ_affine",
    "circ_affine_realshift",
    "circ_spline",
    "circ_spline_realshift",
    "circ_spline_noshift",
]
FAMILY_COLOR = {"existing": "#8a8984", "ghost": "#eb6834", "circular": "#2a78d6"}


def family(method):
    if "ghost" in method:
        return "ghost"
    if method.startswith("circ"):
        return "circular"
    return "existing"


def load(grid):
    rows = []
    for f in Path(grid).glob("*.json"):
        rows.extend(json.load(open(f)))
    return pd.DataFrame(rows)


def table(df):
    g = df.groupby(["case", "method"])
    t = g.agg(
        kl=("kl", "median"),
        kl_max=("kl", "max"),
        ess=("ess", "median"),
        ess_min=("ess", "min"),
        inside=("inside", "median"),
        zhat=("zhat", "median"),
        epochs=("epochs", "median"),
        time=("train_time", "median"),
        n=("seed", "count"),
    ).reset_index()
    return t


def figure(df, path):
    methods = [m for m in ORDER if m in set(df.method)]
    fig, axes = plt.subplots(
        2, len(CASES), figsize=(13, 6.2), sharey=True, constrained_layout=True
    )
    y = np.arange(len(methods))[::-1]
    for j, case in enumerate(CASES):
        d = df[df.case == case]
        for i, (key, label, log) in enumerate(
            [("kl", "forward KL (nats)", True), ("ess", "ESS / draws", False)]
        ):
            ax = axes[i, j]
            for yy, m in zip(y, methods):
                v = d[d.method == m][key].to_numpy()
                if not len(v):
                    continue
                if log:
                    v = np.clip(v, 1e-3, None)
                c = FAMILY_COLOR[family(m)]
                ax.scatter(v, np.full(len(v), yy), s=22, color=c, zorder=3,
                           edgecolor="white", linewidth=0.8)
                ax.plot([np.median(v)] * 2, [yy - 0.3, yy + 0.3], color=c,
                        lw=2, zorder=2)
            if log:
                ax.set_xscale("log")
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.xaxis.set_major_locator(
                    matplotlib.ticker.FixedLocator(
                        [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
                    )
                )
                ax.xaxis.set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
                )
            else:
                ax.set_xlim(-0.02, 1.02)
            ax.grid(axis="x", color="#e4e3df", lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors="#52514e", labelsize=8)
            if i == 0:
                ax.set_title(case, fontsize=11, color="#0b0b0b")
            ax.set_xlabel(label, fontsize=9, color="#52514e")
            ax.set_yticks(y)
            ax.set_yticklabels(methods, fontsize=8)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, label=lab)
        for lab, c in [
            ("nessai today (box, chi radius)", FAMILY_COLOR["existing"]),
            ("ghost padding", FAMILY_COLOR["ghost"]),
            ("circular flow", FAMILY_COLOR["circular"]),
        ]
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3,
               frameon=False, fontsize=9)
    fig.savefig(path, dpi=150)


def main(grid, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(grid)
    t = table(df)
    t["order"] = t.method.map({m: i for i, m in enumerate(ORDER)})
    t["corder"] = t.case.map({c: i for i, c in enumerate(CASES)})
    t = t.sort_values(["corder", "order"]).drop(columns=["order", "corder"])
    pd.set_option("display.width", 200)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    t.to_csv(out / "summary.csv", index=False)
    figure(df, out / "toy_summary.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
