import json, glob, sys, numpy as np
d = sys.argv[1]
zs=[]; lr=[]
for f in glob.glob(f"{d}/dec*.json"):
    r=json.load(open(f))
    it=np.array([x[0] for x in r["iterations"]]); cnt=np.array([x[1] for x in r["iterations"]])
    L=[x for x in r["retrain_log"] if "log_a" in x]
    for a,b in zip(L[:-1],L[1:]):
        if a["retrain"]: continue
        n_real=b["iteration"]-a["iteration"]
        if n_real<20: continue
        lr.append(np.log(n_real/a["n_block"]))
        # realised log acceptance over first 50 iterations of the block vs predicted at block start (slope-corrected)
        m=(it>a["iteration"])&(it<=a["iteration"]+50)
        if m.sum()<50: continue
        k=1/r["nlive"]
        w=np.exp(-(it[m]-a["iteration"]-1)*(-k))  # undo decay back to block start
        la=np.log(m.sum()/np.sum(cnt[m]/w))
        noise=(1-np.exp(la))/m.sum()
        zs.append((la-a["log_a"])/np.sqrt(a["log_a_sd"]**2+noise))
lr=np.array(lr); zs=np.array(zs)
print(f"blocks={len(lr)} log(n_real/n_pred): mean={lr.mean():+.3f} sd={lr.std():.3f}")
print(f"z of start-of-block log acceptance: n={len(zs)} mean={zs.mean():+.2f} sd={zs.std():.2f}  frac|z|>2={np.mean(np.abs(zs)>2):.3f}")
