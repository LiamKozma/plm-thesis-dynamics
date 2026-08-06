#!/usr/bin/env python
"""Show the 'taxonomy axis': real family shift vectors share a common direction.
Statistic = pairwise cosine between family shift vectors (real vs an independent-shift null).
Run on the cluster (real .npy live there)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations

ROOT="/scratch/lmk04992/taxonomy_ladder"
RUNGS=["archaea","fungi","metazoa","plants"]
rng=np.random.default_rng(0)

def shifts(rung):
    p=f"{ROOT}/{rung}"
    S=np.load(f"{p}/source_Shf0.0_X.npy"); Sy=np.load(f"{p}/source_Shf0.0_y.npy")
    T=np.load(f"{p}/test_Shf0.0_X.npy");   Ty=np.load(f"{p}/test_Shf0.0_y.npy")
    ks=sorted(set(Sy)&set(Ty))
    Cs=np.stack([S[Sy==k].mean(0) for k in ks])
    Ct=np.stack([T[Ty==k].mean(0) for k in ks])
    return Ct-Cs                     # per-family shift vectors

def pairwise_cos(V):
    Vn=V/np.linalg.norm(V,axis=1,keepdims=True)
    iu=np.triu_indices(len(V),1)
    return (Vn@Vn.T)[iu]

def shared_frac(V):                  # |mean shift|^2 / mean|shift|^2  == alpha
    return float(np.linalg.norm(V.mean(0))**2 / (np.linalg.norm(V,axis=1)**2).mean())

# --- per-rung stats ---
stats={}
for r in RUNGS:
    V=shifts(r)
    cos=pairwise_cos(V)
    stats[r]=dict(V=V, cos_mean=float(cos.mean()), alpha=shared_frac(V),
                  mag=float(np.linalg.norm(V,axis=1).mean()))
    print(f"{r:8s} mean pairwise cos {stats[r]['cos_mean']:+.2f}  "
          f"shared-frac(alpha) {stats[r]['alpha']:.2f}  mean|shift| {stats[r]['mag']:.2f}")

# --- cross-rung: are the four clades' mean directions aligned? (one taxonomy axis) ---
means=np.stack([stats[r]['V'].mean(0) for r in RUNGS])
mn=means/np.linalg.norm(means,axis=1,keepdims=True)
xr=(mn@mn.T)[np.triu_indices(4,1)]
print(f"cross-rung mean-direction pairwise cos: mean {xr.mean():+.2f} (min {xr.min():+.2f})")

# --- null: independent random shifts of the same magnitudes -> cosines ~ 0 ---
Vp=stats["plants"]["V"]
null=rng.standard_normal((len(Vp), Vp.shape[1]))
cos_real=pairwise_cos(Vp); cos_null=pairwise_cos(null)

# ---------------- figure ----------------
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12})
fig,(a,b)=plt.subplots(1,2,figsize=(14,5.6))
REAL="#0d9488"; NULL="#9aa4a3"

bins=np.linspace(-1,1,41)
a.hist(cos_null,bins=bins,color=NULL,alpha=0.75,label="if families moved independently (null)")
a.hist(cos_real,bins=bins,color=REAL,alpha=0.8,label="real (bacteria → plants)")
a.axvline(cos_null.mean(),color="#5f6a69",ls="--",lw=2)
a.axvline(cos_real.mean(),color=REAL,ls="--",lw=2.4)
a.annotate(f"mean {cos_real.mean():+.2f}",xy=(cos_real.mean(),0),xytext=(cos_real.mean()+0.05,a.get_ylim()[1]*0.8),
           color=REAL,fontweight="bold",fontsize=12)
a.annotate(f"mean {cos_null.mean():+.2f}",xy=(cos_null.mean(),0),xytext=(cos_null.mean()-0.62,a.get_ylim()[1]*0.62),
           color="#5f6a69",fontweight="bold",fontsize=11)
a.set_xlabel("cosine between two families' shift vectors")
a.set_ylabel("number of family pairs")
a.set_title("Families shift in a shared direction",fontweight="bold",fontsize=14)
a.legend(fontsize=10.5,loc="upper left"); a.set_xlim(-1,1)
for s in ["top","right"]: a.spines[s].set_visible(False)

x=np.arange(len(RUNGS))
al=[stats[r]["alpha"] for r in RUNGS]; cm=[stats[r]["cos_mean"] for r in RUNGS]
b.bar(x-0.19,al,width=0.36,color=REAL,label="shared-direction fraction (α)")
b.bar(x+0.19,cm,width=0.36,color="#7cc4bc",label="mean pairwise cosine")
b.axhline(0,color="#ccc",lw=1)
for xi,(a_,c_) in enumerate(zip(al,cm)):
    b.text(xi-0.19,a_+0.01,f"{a_:.2f}",ha="center",va="bottom",fontsize=10,color=REAL,fontweight="bold")
    b.text(xi+0.19,c_+0.01,f"{c_:.2f}",ha="center",va="bottom",fontsize=10,color="#3f7a73",fontweight="bold")
b.set_xticks(x); b.set_xticklabels([r+f"\n(|shift| {stats[r]['mag']:.1f})" for r in RUNGS])
b.set_ylabel("fraction / cosine (0–1)")
b.set_title("Consistent across the whole ladder",fontweight="bold",fontsize=14)
b.set_ylim(0,max(al+cm)*1.25); b.legend(fontsize=10.5,loc="upper left")
for s in ["top","right"]: b.spines[s].set_visible(False)

fig.suptitle("The taxonomy axis: real domain shift is a shared translation, not identity-swapping",
             fontsize=15.5,fontweight="bold",y=1.0)
fig.text(0.5,0.925,f"A random / family-specific shift would give cosines near 0 (grey). Real shifts average "
         f"positive — families move together, they don't swap identities. And the four clades' mean shift "
         f"directions are themselves aligned (cos {xr.mean():+.2f}): a single shared taxonomy axis.",
         ha="center",fontsize=10.5,color="#555")
fig.tight_layout(rect=[0,0,1,0.90])
out="/work/ah2lab/LiamK/tidythesis/taxonomy_axis.png"
fig.savefig(out,dpi=150,bbox_inches="tight",facecolor="white")
print("saved",out)
