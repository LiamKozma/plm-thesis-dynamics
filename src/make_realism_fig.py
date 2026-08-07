#!/usr/bin/env python
"""Realism check: 2D projection of v1 vs v2 SYNTHETIC vs REAL embeddings, same
amount of data, same projection method. Shows which synthetic looks like reality.
Run on the cluster (needs real .npy + the two generators + sklearn/matplotlib)."""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from itertools import combinations

# Resolve paths relative to this file so the script works from any checkout.
SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
import generate_synthetic_precomputed as v1
import generate_synthetic_v2 as v2

FAMS   = [0,1,2,3,4,5]
NSRC, NTGT = 130, 60          # match the real target density (~62/family)
SEED   = 7
COLORS = ["#e6194B","#3cb44b","#4363d8","#f58231","#911eb4","#00a0b0"]
REAL   = "/scratch/lmk04992/taxonomy_ladder/plants"   # bacteria -> plants (the d=1.0 anchor)
rng = np.random.default_rng(SEED)

# ---------- geometry stats (ambient, on source) -------------------------
def sep_ratio(X, y):
    ks = sorted(set(y))
    C = np.stack([X[y==k].mean(0) for k in ks])
    ws = np.mean([np.sqrt(((X[y==k]-X[y==k].mean(0))**2).sum(1).mean()) for k in ks])
    iu = np.triu_indices(len(ks),1)
    gaps = np.sqrt(((C[:,None]-C[None])**2).sum(-1))[iu]
    return gaps.mean()/ws, gaps.min()/ws
def eff_rank(X):
    Xc = X - X.mean(0); s = np.linalg.svd(Xc[:2000], compute_uv=False); l=s**2
    return float(l.sum()**2/(l**2).sum())

# ---------- v1 synthetic (seat-swap, isotropic, 6x sep) -----------------
def make_v1(d=1.0):
    C_lat,P,perm = v1.build_universe(np.random.default_rng(SEED),16,1280,32,3.0)
    def samp(is_t, fam, n):
        c = (C_lat[fam] + d*(C_lat[perm[fam]]-C_lat[fam])) if is_t else C_lat[fam]
        lat = c + 1.0*rng.standard_normal((n,32))       # within_sigma=1.0 (generator default)
        return lat @ P
    return _assemble(samp, srcC=C_lat, tgtC=(C_lat + d*(C_lat[perm]-C_lat)), P=P)

# ---------- v2 synthetic (signal+nuisance, calibrated) ------------------
def make_v2(d=1.0, alpha=0.5, beta=0.5):
    uni = v2.build_universe(np.random.default_rng(SEED),16,960,8,64,3.0,1.5,3.0,2.0,1.9,36.0)
    Ct  = v2.target_centroids(uni,d,alpha,beta,1.2)
    def samp(is_t, fam, n):
        c = (Ct if is_t else uni["C"])[fam]
        sd = np.sqrt(uni["var"])[None,:]*uni["fam_scale"][fam]
        if is_t: sd = sd*(1.0+0.15*d)
        lat = c + sd*rng.standard_normal((n,uni["latent_dim"]))
        return lat @ uni["P"]
    return _assemble(samp, srcC=uni["C"], tgtC=Ct, P=uni["P"])

def _assemble(samp, srcC, tgtC, P):
    Xs,ys,Xt,yt = [],[],[],[]
    for f in FAMS:
        Xs.append(samp(False,f,NSRC)); ys += [f]*NSRC
        Xt.append(samp(True ,f,NTGT)); yt += [f]*NTGT
    Xs=np.vstack(Xs); Xt=np.vstack(Xt); ys=np.array(ys); yt=np.array(yt)
    Cs=(srcC[FAMS]@P); Ct=(tgtC[FAMS]@P)
    return Xs,ys,Xt,yt,Cs,Ct

# ---------- real (bacteria -> plants) -----------------------------------
def make_real():
    S=np.load(f"{REAL}/source_Shf0.0_X.npy"); Sy=np.load(f"{REAL}/source_Shf0.0_y.npy")
    T=np.load(f"{REAL}/test_Shf0.0_X.npy");   Ty=np.load(f"{REAL}/test_Shf0.0_y.npy")
    Xs,ys,Xt,yt=[],[],[],[]
    for f in FAMS:
        si=np.where(Sy==f)[0]; ti=np.where(Ty==f)[0]
        si=rng.permutation(si)[:NSRC]; ti=rng.permutation(ti)[:NTGT]
        Xs.append(S[si]); ys+=[f]*len(si); Xt.append(T[ti]); yt+=[f]*len(ti)
    Xs=np.vstack(Xs);Xt=np.vstack(Xt);ys=np.array(ys);yt=np.array(yt)
    Cs=np.stack([Xs[ys==f].mean(0) for f in FAMS])
    Ct=np.stack([Xt[yt==f].mean(0) for f in FAMS])
    return Xs,ys,Xt,yt,Cs,Ct

# ---------- project + draw ----------------------------------------------
def panel(ax, data, title, subtitle, tcol):
    Xs,ys,Xt,yt,Cs,Ct = data
    pca = PCA(2).fit(np.vstack([Cs,Ct]))
    for j,f in enumerate(FAMS):
        col=COLORS[j]
        S=pca.transform(Xs[ys==f]); T=pca.transform(Xt[yt==f])
        ax.scatter(S[:,0],S[:,1],s=7,c=col,alpha=0.16,linewidths=0)
        ax.scatter(T[:,0],T[:,1],s=10,c=col,alpha=0.55,marker="^",edgecolors="none")
        sc=pca.transform(Cs[j:j+1])[0]; tc=pca.transform(Ct[j:j+1])[0]
        if np.hypot(*(tc-sc))>0.05:
            ax.annotate("",xy=tc,xytext=sc,arrowprops=dict(arrowstyle="-|>",color=col,lw=1.8,alpha=.9,shrinkA=0,shrinkB=0))
        ax.scatter(*sc,s=85,c=col,edgecolors="white",linewidths=1.3,zorder=5)
        ax.scatter(*tc,s=85,c=col,marker="^",edgecolors="white",linewidths=1.3,zorder=5)
    sr,mr = sep_ratio(Xs,ys)
    ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor("#cccccc")
    ax.set_title(title,fontsize=15,fontweight="bold",color=tcol,pad=8)
    ax.text(0.5,1.005,subtitle,transform=ax.transAxes,ha="center",va="bottom",fontsize=10,color="#666")
    overlap = "families OVERLAP" if mr < 0.7 else "families stay APART"
    ax.text(0.03,0.03,f"mean gap / spread   {sr:.1f}\nmin  gap / spread   {mr:.1f}\n{overlap}",
            transform=ax.transAxes,ha="left",va="bottom",fontsize=10.5,family="monospace",
            bbox=dict(boxstyle="round,pad=0.4",fc="white",ec="#bbbbbb",alpha=.92))
    return sr,mr

fig,axes=plt.subplots(1,3,figsize=(16.5,6.0))
panel(axes[0],make_v1(1.0),"v1 synthetic","derangement · isotropic (default)","#bd6a1c")
panel(axes[1],make_v2(1.0),"v2 synthetic","signal+nuisance · calibrated","#0d9488")
panel(axes[2],make_real(),"REAL (ESM-C)","bacteria → plants","#2b2b2b")

handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='#555',markersize=8,label='source cluster (●, faint)'),
         Line2D([0],[0],marker='^',color='w',markerfacecolor='#555',markersize=9,label='target cluster (▲)'),
         Line2D([0],[0],marker='>',color='#555',markersize=9,lw=2,label='centroid: source → target')]
fig.legend(handles=handles,loc="lower center",ncol=3,frameon=False,fontsize=10.5,bbox_to_anchor=(0.5,0.0))
fig.suptitle("Which synthetic looks real?  2D projection, same data volume, same projection method",
             fontsize=16,fontweight="bold",y=0.99)
fig.text(0.5,0.925,"In real embeddings the family clouds OVERLAP (min gap < spread). v2 reproduces that messy overlap; "
         "v1's tight, well-separated blobs (min gap > 2x spread) look like a textbook toy, not real biology.",ha="center",fontsize=10.5,color="#555")
fig.tight_layout(rect=[0,0.05,1,0.90])
out=os.environ.get("FIG_OUT", os.path.join(ROOT,"docs","figures","realism_v1_v2_real.png"))
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out,dpi=150,bbox_inches="tight",facecolor="white")
print("saved",out)
