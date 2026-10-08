"""Scatter of flow draws against the target for a few methods.

Rebuilds each trained flow by replaying ``toy_benchmark.main``'s random
stream with the saved weights in place of training.

Usage: python sample_figure.py GRID_DIR OUT_PNG SEED
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import toy_benchmark as tb  # noqa: E402

METHODS = [
    ("realnvp_box", None, "RealNVP, box"),
    ("realnvp_chi", None, "RealNVP, chi radius"),
    ("nsf_box", None, "NSF, box"),
    ("nsf_ghostpou", "fold", "NSF, PoU ghosts (fold)"),
    ("circ_spline_realshift", None, "circular flow"),
]
# (case, x column, y column, x label, y label)
PANELS = [
    ("peak", 2, 0, "x", "phi"),
    ("multimodal", 2, 0, "x", "phi"),
    ("stripe", 1, 0, "psi", "phi"),
]


def rebuild(grid, case, method_name, seed):
    state = torch.load(Path(grid) / f"{case}_{method_name}_{seed}.pt")

    def load_state(flow, Yt, Yv, wt=None, wv=None):
        expected = set(flow.state_dict())
        if not set(state) <= expected:
            # saved before the parameter-free WrapCircular first layer was
            # added: shift the layer indices by one
            prefix = "_transform._transforms."
            renamed = {}
            for k, v in state.items():
                i, rest = k[len(prefix):].split(".", 1)
                renamed[f"{prefix}{int(i) + 1}.{rest}"] = v
            state.clear()
            state.update(renamed)
        flow.load_state_dict(state)
        flow.eval()
        return []

    tb.train_flow = load_state
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    target = tb.TARGETS[case]()
    X = target.sample(5000, rng)
    method = tb.METHODS[method_name](rng)
    method.train(X)
    return target, method


def main(grid, out, seed=1):
    seed = int(seed)
    n = 4000
    fig, axes = plt.subplots(
        len(PANELS),
        len(METHODS) + 1,
        figsize=(15, 8.4),
        sharex="row",
        sharey="row",
        constrained_layout=True,
    )
    for i, (case, cx, cy, lx, ly) in enumerate(PANELS):
        for j, (m, mode, title) in enumerate([(None, None, "target")] + METHODS):
            ax = axes[i, j]
            if m is None:
                X = tb.TARGETS[case]().sample(n, np.random.default_rng(0))
                color = "#52514e"
            else:
                _, method = rebuild(grid, case, m, seed)
                if mode is not None:
                    method.mode = mode
                X, _ = method.sample(n)
                color = "#eb6834" if "ghost" in m else (
                    "#2a78d6" if m.startswith("circ") else "#8a8984")
            ok = np.isfinite(X).all(axis=1)
            ax.scatter(X[ok, cx], X[ok, cy], s=1.5, color=color, alpha=0.5,
                       linewidths=0, rasterized=True)
            if i == 0:
                ax.set_title(title, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{case}\n{ly}", fontsize=9)
            ax.set_xlabel(lx, fontsize=8, color="#52514e")
            ax.tick_params(labelsize=7, colors="#52514e")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if ly == "phi":
                ax.set_ylim(0, 2 * np.pi)
            if lx == "psi":
                ax.set_xlim(0, np.pi)
            if lx == "x":
                ax.set_xlim(-3.5, 3.5)
    fig.savefig(out, dpi=140)


if __name__ == "__main__":
    main(*sys.argv[1:])
