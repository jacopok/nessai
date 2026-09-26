import numpy as np, pandas as pd
S = pd.read_csv("runs7_summary.csv")
R = pd.read_csv("runs7_rows.csv")
names = {"default": "default", "f1000": "fixed (1000 it)", "dec": "decision",
         "decdyn": "decision (measured costs)", "decreset": "decision + reset"}
order = ["f1000", "dec", "decdyn", "decreset"]
lines = [r"\begin{table}[t]\centering",
         r"\caption{Run time relative to the default schedule (geometric mean over 3 problems $\times$ 6 seeds, 95\% bootstrap interval), mean number of trainings and resets.}",
         r"\label{tab:results}", r"\begin{tabular}{llcrr}", r"\toprule",
         r"$t_L$ [s] & policy & time ratio & trainings & resets \\", r"\midrule"]
for tl in [0, 1e-4, 1e-3, 1e-2]:
    sub = S[S.tl == tl]
    d = sub[sub.policy == "default"].iloc[0]
    lines.append(f"{tl:g} & default & 1 & {d.ntrain:.1f} & 0 \\\\")
    for p in order:
        r = sub[sub.policy == p]
        if r.empty: continue
        r = r.iloc[0]
        lines.append(f" & {names[p]} & {r.ratio:.2f} [{r.lo:.2f}, {r.hi:.2f}] & {r.ntrain:.1f} & {r.nreset:.1f} \\\\")
    lines.append(r"\midrule" if tl != 1e-2 else r"\bottomrule")
lines += [r"\end{tabular}", r"\end{table}"]
open("report/table_results.tex", "w").write("\n".join(lines))

# per problem
R["lr"] = np.log(R.vtime / R.ref)
pp = R[R.policy.isin(["dec", "decreset", "f1000"])].groupby(["tl", "policy", "problem"]).lr.mean().apply(np.exp).unstack("problem")
lines = [r"\begin{table}[t]\centering", r"\caption{Time ratio to default per problem (geometric mean over 6 seeds).}", r"\label{tab:perproblem}",
         r"\begin{tabular}{llccc}", r"\toprule", r"$t_L$ [s] & policy & Gaussian 8D & Rosenbrock 4D & bimodal 4D \\", r"\midrule"]
for (tl, p), row in pp.iterrows():
    lines.append(f"{tl:g} & {names[p]} & {row['gauss8']:.2f} & {row['rosen4']:.2f} & {row['bimod4']:.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("report/table_perproblem.tex", "w").write("\n".join(lines))

# evidence check (one row per run)
runs = R.drop_duplicates(subset=["policy", "problem", "seed", "logZ"])
truth = {"gauss8": 4 * np.log(2 * np.pi) - 8 * np.log(20)}
cov = (0.5 * np.eye(4) + 0.5) * 0.3
truth["bimod4"] = np.log(2) + 2 * np.log(2 * np.pi) + 0.5 * np.linalg.slogdet(cov)[1] - 4 * np.log(20)
ev = runs.groupby(["problem", "policy"]).logZ.agg(["mean", "std", "count"])
print(ev.round(3)); print(truth)
ev.to_csv("runs7_evidence.csv")
