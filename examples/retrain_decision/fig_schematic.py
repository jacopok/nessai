import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from nessai.samplers.retrain import solve_renewal
k=1.0; A=1.0; c=1.0; T=0.35
x=solve_renewal(T*k*A/c); g=x*c/A; tau=np.log(x)/k
fig,ax=plt.subplots(figsize=(7,3.3))
s=np.linspace(0,tau,200)
for i in range(3):
    t0=i*tau
    ax.fill_between(t0+s, c/A*np.exp(k*s), g, color="#2a78d6", alpha=0.15, lw=0, label="savings $=T$" if i==0 else None)
    ax.plot(t0+s, c/A*np.exp(k*s), color="#2a78d6", lw=2, label=r"cost rate $c\,e^{ks}/A$" if i==0 else None)
    ax.axvline(t0+tau, color="#8a8a8a", lw=1, ls=":")
ss=np.linspace(0,2.2*tau,200)
ax.plot(2*tau+ss*0.5-0.0, c/A*np.exp(k*(tau+ss*0.5)), color="#eb6834", lw=1.5, ls="--", label="keep current flow")
ax.axhline(g, color="#333", lw=1.5, label=r"optimal long-run rate $g^*$")
ax.set_xlim(0,3.2*tau); ax.set_ylim(0.9, g*1.35)
ax.set_xlabel("iterations"); ax.set_ylabel("cost per iteration")
ax.set_xticks([tau,2*tau,3*tau]); ax.set_xticklabels([r"$\tau^*$",r"$2\tau^*$",r"$3\tau^*$"])
ax.set_yticks([c/A,g]); ax.set_yticklabels([r"$c/A$",r"$g^*$"])
ax.spines[["top","right"]].set_visible(False); ax.legend(frameon=False,fontsize=8,loc="upper left",ncol=2)
fig.tight_layout(); fig.savefig("report/fig_schematic.pdf")
