import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
C=["#2a78d6","#eb6834","#1baf7a"]
fig,ax=plt.subplots(figsize=(7,4.2))
for col,(name,lab) in zip(C,[("f3000_gauss8","Gaussian 8D"),("f3000_rosen4","Rosenbrock 4D"),("f3000_bimod4","Bimodal 4D")]):
    r=json.load(open(f"runs2/{name}.json")); nl=r["nlive"]
    it=np.array([x[0] for x in r["iterations"]]); cnt=np.array([x[1] for x in r["iterations"]])
    tr=[t["iteration"] for t in r["trainings"]]
    first=True
    for a,b in zip(tr, tr[1:]+[it.max()+1]):
        m=(it>a)&(it<=b); c=cnt[m]; B=100; nb=len(c)//B
        if nb<5: continue
        la=np.log(B/np.array([c[i*B:(i+1)*B].sum() for i in range(nb)]))
        x=(np.arange(nb)+0.5)*B/nl
        ax.plot(x, la-(la+x).mean(), color=col, lw=1.2, alpha=0.8, label=lab if first else None); first=False
xx=np.linspace(0,3,10); ax.plot(xx,-xx,color="#555",ls="--",lw=2,label="slope −1 (acceptance ∝ X)")
ax.set_xlabel("iterations since training / nlive"); ax.set_ylabel("log acceptance (offset per episode)")
ax.set_title("Acceptance decays as exp(−s / nlive) between trainings",loc="left",fontsize=11)
ax.grid(alpha=0.25); ax.spines[["top","right"]].set_visible(False); ax.legend(frameon=False,fontsize=9)
fig.tight_layout(); fig.savefig("report/fig_decay.pdf")
