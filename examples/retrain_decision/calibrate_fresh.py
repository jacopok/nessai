import json, glob, sys, numpy as np
from nessai.samplers.retrain import RetrainDecision
zs=[]; errs=[]
for f in glob.glob(sys.argv[1]):
    r=json.load(open(f)); nl=r["nlive"]
    it=np.array([x[0] for x in r["iterations"]]); cnt=np.array([x[1] for x in r["iterations"]])
    T=r["trainings"]
    d=RetrainDecision(nl, costs=dict(likelihood=1e-3,population=1e-4,training=1e-5))
    preds=[]
    for j,t in enumerate(T):
        preds.append(d.fresh_acceptance(reset=False) if not t["reset"] else None)
        d.start_episode(t["iteration"], t["reset"], t["epochs"], 1000)
        e = T[j+1]["iteration"] if j+1<len(T) else it.max()+1
        m=(it>t["iteration"])&(it<=e)
        for c in cnt[m]: d.record_iteration(int(c))
    d.start_episode(it.max()+1, False, 1, 1)
    for p, ep in zip(preds, d.episodes[:-1]):
        if p is None or not np.isfinite(ep.log_a0): continue
        zs.append((ep.log_a0-p[0])/np.sqrt(p[1]+ep.log_a0_var)); errs.append(ep.log_a0-p[0])
zs=np.array(zs); errs=np.array(errs)
print(f"n={len(zs)} error mean={errs.mean():+.3f} rms={np.sqrt(np.mean(errs**2)):.3f} | z mean={zs.mean():+.2f} sd={zs.std():.2f} frac|z|>2={np.mean(np.abs(zs)>2):.3f}")
