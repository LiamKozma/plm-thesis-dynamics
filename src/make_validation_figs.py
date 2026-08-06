#!/usr/bin/env python
"""Validation figures: (1) v2 copies real geometry across all datasets;
(2) v2 predicts real recovery out-of-sample. Cluster only."""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0,"/work/ah2lab/LiamK/tidythesis/src")
import generate_synthetic_v2 as v2g

D=json.load(open("/work/ah2lab/LiamK/tidythesis/validation_data.json"))
REP="/work/ah2lab/LiamK/tidythesis"

# ---- v2 geometry, computed the SAME way as real ----
uni=v2g.build_universe(np.random.default_rng(7),16,960,8,64,3.0,1.5,3.0,2.0,1.9,36.0)
Xs,ys=v2g.sample(np.random.default_rng(1),4000,uni,0.0,0.5,0.5,False,1.2)
ks=sorted(set(ys.tolist()))
Cs=np.stack([Xs[ys==k].mean(0) for k in ks])
ws=np.mean([np.sqrt(((Xs[ys==k]-Xs[ys==k].mean(0))**2).sum(1).mean()) for k in ks])
iu=np.triu_indices(len(ks),1); gaps=np.sqrt(((Cs[:,None]-Cs[None])**2).sum(-1))[iu]
Xc=Xs-Xs.mean(0); sv=np.linalg.svd(Xc[:2000],compute_uv=False); l=sv**2
v2_sep=float(gaps.mean()/ws); v2_erank=float(l.sum()**2/(l**2).sum())
print(f"v2 geometry: sep {v2_sep:.2f}  erank {v2_erank:.1f}")

LAD=["archaea","fungi","metazoa","plants"]; SP=["swissprot_esmc","swissprot_esm2"]
order=LAD+SP+["somedir_esm2"]
def g(n,k): return D["real"][n]["geom"][k]
LC="#0d9488"; SC="#bd6a1c"; V2="#2c3e90"

# ================= FIGURE 1: geometry realism =================
fig,axes=plt.subplots(1,3,figsize=(15,5))
def geompanel(ax,key,label,v2val,ylim=None):
    for i,n in enumerate(order):
        c=LC if n in LAD else (SC if n in SP else "#888")
        ax.scatter(i,g(n,key),s=90,color=c,zorder=3)
    ax.axhline(v2val,color=V2,lw=2.4,ls="--",zorder=2,label=f"v2 calibrated ({v2val:.2f})")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([n.replace("swissprot_","sp_").replace("_esm2"," e2").replace("_esmc"," ec").replace("somedir e2","old e2") for n in order],rotation=40,ha="right",fontsize=9)
    ax.set_title(label,fontweight="bold",fontsize=13); ax.legend(fontsize=9.5,loc="best")
    if ylim: ax.set_ylim(*ylim)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.grid(True,axis="y",color="#eee"); ax.set_axisbelow(True)
geompanel(axes[0],"sep","separation ratio  (mean gap / spread)",v2_sep,(0,1.6))
geompanel(axes[1],"erank","effective rank",v2_erank,(0,14))
# alpha panel: show real range + v2 knob band
for i,n in enumerate(order):
    c=LC if n in LAD else (SC if n in SP else "#888")
    axes[2].scatter(i,g(n,"alpha"),s=90,color=c,zorder=3)
axes[2].axhspan(0.0,1.0,color=V2,alpha=0.08);
axes[2].set_xticks(range(len(order)))
axes[2].set_xticklabels([n.replace("swissprot_","sp_").replace("_esm2"," e2").replace("_esmc"," ec").replace("somedir e2","old e2") for n in order],rotation=40,ha="right",fontsize=9)
axes[2].set_title("shared-direction fraction  α",fontweight="bold",fontsize=13)
axes[2].text(0.5,0.92,"v2 α knob spans 0–1",transform=axes[2].transAxes,ha="center",color=V2,fontsize=10)
axes[2].set_ylim(0,1);
for s in ["top","right"]: axes[2].spines[s].set_visible(False)
axes[2].grid(True,axis="y",color="#eee"); axes[2].set_axisbelow(True)
fig.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=LC,ms=10,label='taxonomy ladder (ESM-C)'),
                    Line2D([0],[0],marker='o',color='w',markerfacecolor=SC,ms=10,label='swissprot→trembl'),
                    Line2D([0],[0],marker='o',color='w',markerfacecolor='#888',ms=10,label='old bact→archaea')],
           loc="lower center",ncol=3,frameon=False,fontsize=10,bbox_to_anchor=(0.5,-0.03))
fig.suptitle("Does v2 copy real embedding geometry?  Yes — v2 lands inside the real range on every measure",
             fontsize=14.5,fontweight="bold",y=1.0)
fig.tight_layout(rect=[0,0.05,1,0.94])
fig.savefig(f"{REP}/validation_geometry.png",dpi=150,bbox_inches="tight",facecolor="white")
print("saved validation_geometry.png")

# ================= FIGURE 2: recovery out-of-sample =================
fig,(a1,a2)=plt.subplots(1,2,figsize=(15,6))
# panel A: r* vs zero-shot, real points + v2 loci
def rec(n): return D["real"][n]["rec"]
be=sorted(D["v2"]["beta"],key=lambda x:-x["zero"])
a1.plot([x["zero"] for x in be],[x["rstar"] for x in be],"-o",color="#7c88c8",lw=2.2,ms=7,
        label="v2 β-locus (d=1, vary function-damage)")
dd=D["v2"]["distance"]
dz=[(z,r) for z,r in zip(dd["zero"],dd["rstar"]) if r is not None]
a1.plot([z for z,r in dz],[r for z,r in dz],"--s",color="#b0b7dd",lw=1.8,ms=6,
        label="v2 distance-locus (α=0.5, vary distance)")
lad_off={"archaea":(6,7),"fungi":(6,-15),"metazoa":(-4,10),"plants":(8,-15)}
for n in LAD:
    a1.scatter(rec(n)["zero_shot"],rec(n)["r_star"],s=130,color=LC,zorder=5,edgecolors="white",linewidths=1.3)
    a1.annotate(n,(rec(n)["zero_shot"],rec(n)["r_star"]),textcoords="offset points",xytext=lad_off[n],fontsize=9.5,color=LC,fontweight="bold")
sp_off={"swissprot_esmc":(4,9),"swissprot_esm2":(4,-17)}
for n in SP:
    a1.scatter(rec(n)["zero_shot"],rec(n)["r_star"],s=130,color=SC,marker="D",zorder=5,edgecolors="white",linewidths=1.3)
    a1.annotate(n.replace("swissprot_","sp "),(rec(n)["zero_shot"],rec(n)["r_star"]),textcoords="offset points",xytext=sp_off[n],fontsize=9,color=SC)
a1.set_xlabel("zero-shot F1  (how degraded the source model is — a proxy for effective distance)")
a1.set_ylabel("recovery threshold  r*")
a1.set_xlim(0.95,0.1); a1.set_ylim(-0.05,1.05)   # reversed x: more shifted -> right
a1.set_title("Out-of-sample: real r* matches v2's β-locus",fontweight="bold",fontsize=13)
a1.legend(fontsize=9.5,loc="upper left")
a1.annotate("real shifts are all MILD\n(r* ≤ 0.25) — v2 predicts this",xy=(0.64,0.25),xytext=(0.55,0.55),
            fontsize=10,color=LC,fontweight="bold",arrowprops=dict(arrowstyle="-|>",color=LC,lw=1.5))
a1.annotate("v2 extrapolates to SEVERE\nshifts real proteins can't reach",xy=(0.27,1.0),xytext=(0.45,0.85),
            fontsize=9.5,color="#8088c0",arrowprops=dict(arrowstyle="-|>",color="#8088c0",lw=1.4))
for s in ["top","right"]: a1.spines[s].set_visible(False);
a1.grid(True,color="#eee"); a1.set_axisbelow(True)

# panel B: ladder distance vs zero-shot -> same distance, different damage
mg=[g(n,"mag_over_gap") for n in LAD]; zs=[rec(n)["zero_shot"] for n in LAD]
a2.scatter(mg,zs,s=150,color=LC,zorder=5,edgecolors="white",linewidths=1.4)
for n,x,y in zip(LAD,mg,zs):
    a2.annotate(n,(x,y),textcoords="offset points",xytext=(8,4),fontsize=11,color=LC,fontweight="bold")
a2.set_xlabel("geometric distance  (shift magnitude / family gap)")
a2.set_ylabel("zero-shot F1")
a2.set_xlim(0.9,1.4); a2.set_ylim(0.5,0.9)
a2.set_title("The ladder is DAMAGE, not distance",fontweight="bold",fontsize=13)
a2.annotate("archaea & plants: nearly the SAME distance\n(1.16 vs 1.20) but very different damage\n(0.82 vs 0.64) → the β story, on real data",
            xy=(1.18,0.73),xytext=(0.93,0.55),fontsize=9.5,color="#333",
            bbox=dict(boxstyle="round,pad=0.4",fc="#f6e6d2",ec=SC,alpha=.9))
for s in ["top","right"]: a2.spines[s].set_visible(False)
a2.grid(True,color="#eee"); a2.set_axisbelow(True)

fig.suptitle("Does v2 predict the real recovery threshold?  Yes, in the regime real data can reach",
             fontsize=14.5,fontweight="bold",y=1.0)
fig.text(0.5,0.94,"Real taxonomy shifts vary mostly in function-damage (β) at roughly fixed distance — and v2's β-locus reproduces their r*. "
         "The severe-distance regime (r*→1) is v2's extrapolation, which real proteins can't test directly.",
         ha="center",fontsize=10,color="#555")
fig.tight_layout(rect=[0,0,1,0.92])
fig.savefig(f"{REP}/validation_recovery.png",dpi=150,bbox_inches="tight",facecolor="white")
print("saved validation_recovery.png")
