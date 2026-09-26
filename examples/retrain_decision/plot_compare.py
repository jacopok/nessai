import sys, numpy as np, pandas as pd, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = sys.argv[1] if len(sys.argv) > 1 else "runs7"
df = pd.read_csv(f"{d}_rows.csv")
pols = [("f1000", "fixed, every 1000 it"), ("dec", "decision (costs given)"), ("decreset", "decision + reset")]
tls = [0, 1e-4, 1e-3, 1e-2]
C = {"f1000": "#8a8a8a", "dec": "#2a78d6", "decreset": "#eb6834"}
fig, axs = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
for ax, tl in zip(axs, tls):
    for j, (p, lab) in enumerate(pols):
        sub = df[(df.tl == tl) & (df.policy == p)].dropna(subset=["lr"])
        y = np.full(len(sub), j) + np.random.default_rng(j).uniform(-0.15, 0.15, len(sub))
        ax.scatter(np.exp(sub.lr), y, s=12, color=C[p], alpha=0.45, lw=0)
        if len(sub):
            ax.plot([np.exp(sub.lr.mean())] * 2, [j - 0.3, j + 0.3], color=C[p], lw=3)
    ax.axvline(1, color="#333", lw=1, ls="--")
    ax.set_xscale("log"); ax.set_xlim(0.3, 2.3); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xticks([0.35, 0.5, 0.7, 1, 1.4, 2]); ax.set_xticklabels(["0.35", "0.5", "0.7", "1", "1.4", "2"])
    ax.set_title(f"t_L = {tl:g} s" if tl else "t_L ≈ 0 (training-dominated)", fontsize=10, loc="left")
    ax.grid(alpha=0.2, axis="x"); ax.spines[["top", "right"]].set_visible(False)
axs[0].set_yticks(range(len(pols))); axs[0].set_yticklabels([l for _, l in pols])
fig.supxlabel("run time relative to default schedule (same problem & seed; bar = geometric mean)", fontsize=9)
fig.tight_layout(); fig.savefig("report/fig_compare.pdf")
