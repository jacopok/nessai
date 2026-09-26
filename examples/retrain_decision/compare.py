"""Paired comparison of schedules in virtual time (wall + n_like * t_L)."""
import glob, json, re, sys
import numpy as np, pandas as pd

d = sys.argv[1] if len(sys.argv) > 1 else "runs7"
TLS = [0, 1e-4, 1e-3, 1e-2]
rows = []
for f in glob.glob(f"{d}/*.json"):
    r = json.load(open(f))
    pol, prob, seed = re.match(r"(.+)_(\w+?\d)_s(\d)$", r["name"]).groups()
    m = re.match(r"(dec\w*)_tl([\d.e-]+)$", pol)
    tls = [float(m.group(2))] if m else TLS
    pol = m.group(1) if m else pol
    for tl in tls:
        rows.append(dict(policy=pol, problem=prob, seed=int(seed), tl=tl,
                         vtime=r["wall"] + r["n_like"] * tl, nlike=r["n_like"],
                         ntrain=len(r["trainings"]),
                         nreset=sum(t["reset"] for t in r["trainings"][1:]),
                         logZ=r["log_evidence"]))
df = pd.DataFrame(rows)
df = df[~((df.policy == "decdyn") & (df.tl > 0))]
ref = df[df.policy == "default"].set_index(["tl", "problem", "seed"])
df = df.join(ref[["vtime"]].rename(columns=dict(vtime="ref")), on=["tl", "problem", "seed"])
df["lr"] = np.log(df.vtime / df.ref)
rng = np.random.default_rng(0)

def summ(x):
    v = x.lr.dropna().values
    if len(v) == 0:
        return pd.Series(dict(n=0))
    boots = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    return pd.Series(dict(n=len(v), ratio=np.exp(v.mean()),
                          lo=np.exp(np.percentile(boots, 2.5)),
                          hi=np.exp(np.percentile(boots, 97.5)),
                          ntrain=x.ntrain.mean(), nreset=x.nreset.mean()))

pd.set_option("display.width", 200)
out = df.groupby(["tl", "policy"]).apply(summ).round(3)
print(out)
if "-p" in sys.argv:
    print(df.groupby(["tl", "policy", "problem"]).apply(summ)[["n", "ratio", "lo", "hi", "ntrain"]].round(3).unstack("problem"))
out.to_csv(f"{d}_summary.csv")
df.to_csv(f"{d}_rows.csv")
