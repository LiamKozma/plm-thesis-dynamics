#!/usr/bin/env python
"""Is real's bimodal cosine distribution real structure, or centroid-estimation noise?
Compare real vs v2-EXACT (noise-free centroids) vs v2-SAMPLED (centroids estimated from
the same # of points as real). Also split real cosines by family size. Run on cluster."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,"/work/ah2lab/LiamK/tidythesis/src")
import generate_synthetic_v2 as v2g
rng=np.random.default_rng(1)

def pcos_pairs(V):
    Vn=V/np.linalg.norm(V,axis=1,keepdims=True)
    iu=np.triu_indices(len(V),1)
    return (Vn@Vn.T)[iu], iu

# ---- real (bacteria -> plants) + family sizes ----
p="/scratch/lmk04992/taxonomy_ladder/plants"
S=np.load(f"{p}/source_Shf0.0_X.npy"); Sy=np.load(f"{p}/source_Shf0.0_y.npy")
T=np.load(f"{p}/test_Shf0.0_X.npy");   Ty=np.load(f"{p}/test_Shf0.0_y.npy")
ks=sorted(set(Sy)&set(Ty))
n_src=np.array([(Sy==k).sum() for k in ks]); n_tgt=np.array([(Ty==k).sum() for k in ks])
Cs=np.stack([S[Sy==k].mean(0) for k in ks]); Ct=np.stack([T[Ty==k].mean(0) for k in ks])
Vreal=Ct-Cs
cos_real,iu=pcos_pairs(Vreal)
print("real target family sizes: min",n_tgt.min(),"max",n_tgt.max(),"median",int(np.median(n_tgt)))

# ---- v2 universe ----
uni=v2g.build_universe(np.random.default_rng(7),16,960,8,64,3.0,1.5,3.0,2.0,1.9,36.0)
Ct2=v2g.target_centroids(uni,1.0,0.5,0.5,1.2)
Cs2=uni["C"]
Vv2_exact=(Ct2-Cs2)@uni["P"]
cos_v2_exact,_=pcos_pairs(Vv2_exact)

# ---- v2 SAMPLED: estimate centroids from the same counts as real ----
def sample_family(center, fam, n, inflate):
    sd=np.sqrt(uni["var"])[None,:]*uni["fam_scale"][fam]*inflate
    lat=center+sd*rng.standard_normal((n,uni["latent_dim"]))
    return (lat@uni["P"]).mean(0)          # estimated centroid in ambient
def v2_sampled_shifts():
    est_s=np.stack([sample_family(Cs2[f],f,int(n_src[i]),1.0)          for i,f in enumerate(range(16))])
    est_t=np.stack([sample_family(Ct2[f],f,int(n_tgt[i]),1.0+0.15*1.0) for i,f in enumerate(range(16))])
    return est_t-est_s
cos_v2_samp,_=pcos_pairs(v2_sampled_shifts())
print(f"mean cos  real {cos_real.mean():+.2f}  v2-exact {cos_v2_exact.mean():+.2f}  "
      f"v2-sampled {cos_v2_samp.mean():+.2f}")

# ---- split REAL pairs by size: both families large vs >=1 small ----
med=np.median(n_tgt)
small=n_tgt<med
pair_has_small=small[iu[0]]|small[iu[1]]
print(f"real cos: pairs w/ a small family {cos_real[pair_has_small].mean():+.2f}  "
      f"both large {cos_real[~pair_has_small].mean():+.2f}")

# ---------------- figure ----------------
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12})
fig,(a,b)=plt.subplots(1,2,figsize=(14,5.6)); bins=np.linspace(-1,1,33)
REAL="#0d9488"; EX="#2c3e90"; SA="#e0993f"
a.hist(cos_v2_exact,bins=bins,color=EX,alpha=0.5,label=f"v2 EXACT centroids (mean {cos_v2_exact.mean():+.2f})")
a.hist(cos_v2_samp,bins=bins,color=SA,alpha=0.55,label=f"v2 SAMPLED like real (mean {cos_v2_samp.mean():+.2f})")
a.hist(cos_real,bins=bins,histtype="step",color=REAL,lw=2.6,label=f"real (mean {cos_real.mean():+.2f})")
a.set_xlabel("cosine between two families' shift vectors"); a.set_ylabel("# family pairs")
a.set_title("Estimation noise broadens the distribution",fontweight="bold",fontsize=13.5)
a.legend(fontsize=9.5,loc="upper left"); a.set_xlim(-1,1)
for s in ["top","right"]: a.spines[s].set_visible(False)

# Right: v2 has ONE shared axis by construction, yet each 16-family draw is lumpy & different.
for i,sd in enumerate([11,12,13,14,15]):
    u=v2g.build_universe(np.random.default_rng(sd),16,960,8,64,3.0,1.5,3.0,2.0,1.9,36.0)
    Vd=(v2g.target_centroids(u,1.0,0.5,0.5,1.2)-u["C"])@u["P"]
    cd,_=pcos_pairs(Vd)
    b.hist(cd,bins=bins,histtype="step",lw=1.6,alpha=0.8,
           label="v2 draws (α=0.5, one shared axis)" if i==0 else None,color="#7c88c8")
b.hist(cos_real,bins=bins,histtype="step",color=REAL,lw=3,label="real (one draw)")
b.set_xlabel("cosine between two families' shift vectors"); b.set_ylabel("# family pairs")
b.set_title("At 16 families the shape is noise — only the mean (α) is stable",fontweight="bold",fontsize=12.5)
b.legend(fontsize=9.5,loc="upper left"); b.set_xlim(-1,1)
for s in ["top","right"]: b.spines[s].set_visible(False)

fig.suptitle("Real's lumpy cosine distribution: sampling noise, not real bimodal structure",fontsize=15,fontweight="bold",y=1.0)
fig.text(0.5,0.925,"Left: exact v2 centroids vs v2 sampled with real's protein counts vs real — same mean (~0.47), noise broadens it. "
         "Right: v2 has ONE shared axis by construction, yet every 16-family draw is lumpy and different — so real's bumps aren't extra modes.",
         ha="center",fontsize=10,color="#555")
fig.tight_layout(rect=[0,0,1,0.90])
out="/work/ah2lab/LiamK/tidythesis/taxonomy_axis_noise.png"
fig.savefig(out,dpi=150,bbox_inches="tight",facecolor="white"); print("saved",out)
