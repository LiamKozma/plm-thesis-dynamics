#!/usr/bin/env python
"""2D projection of the synthetic embedding space: source vs. target at growing
distance, v1 (seat-swap) vs v2 (shared drift). For the professor meeting."""
import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

SRC = "/home/dude/sapelo2_files/work/tidythesis/src"
sys.path.insert(0, SRC)
import generate_synthetic_precomputed as v1
import generate_synthetic_v2 as v2

# ---- config -------------------------------------------------------------
DISTS   = [0.0, 0.35, 0.70, 1.0]      # columns
FAMS    = [0, 1, 2, 3, 4, 5]          # show 6 families for legibility
N_PER   = 130                         # points per family per domain
SEED    = 7
V2_ALPHA, V2_BETA = 1.0, 0.15         # pure shared drift -> parallel arrows (cleanest contrast)
COLORS  = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00a0b0"]
rng = np.random.default_rng(SEED)

# ---- v1 universe (latent geometry) --------------------------------------
LATENT_V1 = 32
C_lat, P1, perm = v1.build_universe(np.random.default_rng(SEED), 16, 960,
                                    LATENT_V1, centroid_spread=3.0)
WSIG_V1 = 4.0   # realistic-ceiling setting used in the sweeps (sigma4 run)

def v1_target_centroids(d):
    return C_lat + d * (C_lat[perm] - C_lat)

def v1_sample(d, is_target, fam):
    c = v1_target_centroids(d)[fam] if is_target else C_lat[fam]
    return c + WSIG_V1 * rng.standard_normal((N_PER, LATENT_V1))

# ---- v2 universe (signal + nuisance latent) -----------------------------
uni = v2.build_universe(np.random.default_rng(SEED), n_families=16, dim=960,
                        signal_dim=8, nuisance_dim=64, centroid_spread=3.0,
                        sigma_signal=1.5, nuisance_ratio=3.0, spectrum_exponent=2.0,
                        family_sigma_spread=1.9, family_size_skew=36.0)
D_UNIT = 1.20
SIG_INFLATE = 0.15

def v2_target_centroids(d):
    return v2.target_centroids(uni, d, V2_ALPHA, V2_BETA, D_UNIT)

def v2_sample(d, is_target, fam):
    C = v2_target_centroids(d) if is_target else uni["C"]
    c = C[fam]
    sd = np.sqrt(uni["var"])[None, :] * uni["fam_scale"][fam]
    if is_target:
        sd = sd * (1.0 + SIG_INFLATE * d)
    return c + sd * rng.standard_normal((N_PER, uni["latent_dim"]))

# ---- fixed 2D projection per version ------------------------------------
# Fit PCA on ALL centroids (source + every target distance) so both the family
# layout AND the shift direction live in the projection, and it is IDENTICAL
# across the distance panels (so motion is real, not a re-rotation).
def fit_proj(src_C, tgt_fn, dists):
    stack = [src_C[FAMS]]
    for d in dists:
        stack.append(tgt_fn(d)[FAMS])
    pca = PCA(n_components=2).fit(np.vstack(stack))
    return pca

pca1 = fit_proj(C_lat, v1_target_centroids, DISTS)
pca2 = fit_proj(uni["C"], v2_target_centroids, DISTS)

# ---- plot ---------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(2, len(DISTS), figsize=(4.1*len(DISTS), 8.4),
                         sharex="row", sharey="row")

def panel(ax, version, d):
    if version == 1:
        proj, srcC, tgt_c_fn, samp = pca1, C_lat, v1_target_centroids, v1_sample
    else:
        proj, srcC, tgt_c_fn, samp = pca2, uni["C"], v2_target_centroids, v2_sample
    tgtC = tgt_c_fn(d)
    for j, f in enumerate(FAMS):
        col = COLORS[j]
        S = proj.transform(v1_sample(d, False, f) if version == 1 else v2_sample(d, False, f))
        T = proj.transform(v1_sample(d, True,  f) if version == 1 else v2_sample(d, True,  f))
        ax.scatter(S[:,0], S[:,1], s=7, c=col, alpha=0.16, linewidths=0)
        ax.scatter(T[:,0], T[:,1], s=9, c=col, alpha=0.55, marker="^",
                   edgecolors="none")
        # arrow source-centroid -> target-centroid
        sc = proj.transform(srcC[f:f+1])[0]
        tc = proj.transform(tgtC[f:f+1])[0]
        if d > 0 and np.hypot(*(tc-sc)) > 0.05:
            ax.annotate("", xy=tc, xytext=sc,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.9,
                                        alpha=0.9, shrinkA=0, shrinkB=0))
        ax.scatter(*sc, s=90, c=col, edgecolors="white", linewidths=1.4, zorder=5)
        ax.scatter(*tc, s=90, c=col, marker="^", edgecolors="white",
                   linewidths=1.4, zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#cccccc")
    ax.set_title(f"d = {d:g}", fontsize=12.5,
                 fontweight="bold" if d in (0.0,1.0) else "normal")

for c, d in enumerate(DISTS):
    panel(axes[0, c], 1, d)
    panel(axes[1, c], 2, d)

axes[0,0].set_ylabel("v1  (seat-swap)\n", fontsize=15, fontweight="bold",
                     color="#bd6a1c", labelpad=10)
axes[1,0].set_ylabel("v2  (shared drift)\n", fontsize=15, fontweight="bold",
                     color="#0d9488", labelpad=10)

# row annotations
axes[0,-1].text(0.98, 0.02, "families slide ONTO each\nother → overlap at mid-d",
                transform=axes[0,-1].transAxes, ha="right", va="bottom",
                fontsize=9.5, color="#8a4d12",
                bbox=dict(boxstyle="round,pad=0.35", fc="#f6e6d2", ec="#bd6a1c", alpha=.9))
axes[1,-1].text(0.98, 0.02, "every family shifts the SAME\nway → arrangement preserved",
                transform=axes[1,-1].transAxes, ha="right", va="bottom",
                fontsize=9.5, color="#0a5b53",
                bbox=dict(boxstyle="round,pad=0.35", fc="#d8efec", ec="#0d9488", alpha=.9))

# legend
handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#555', markersize=8,
           label='source cluster  (●, faint)'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='#555', markersize=9,
           label='target cluster  (▲)'),
    Line2D([0],[0], marker='>', color='#555', markersize=9, lw=2,
           label='family centroid: source → target'),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           fontsize=10.5, bbox_to_anchor=(0.5, 0.055))

fig.suptitle("Synthetic embedding space, 2D projection  —  source vs. target as distance grows",
             fontsize=16, fontweight="bold", y=0.995)
fig.text(0.5, 0.945,
         "Same fixed projection across each row. Top: v1 moves each family toward another family's spot "
         "(they collide mid-distance).  Bottom: v2 (α=1) slides every family the same way (relationships preserved).",
         ha="center", fontsize=10.5, color="#555")
fig.text(0.5, 0.028,
         "Min. gap between families  —  v1:  15.6 → 9.8 → 9.2 → 15.6  (collapses mid-distance, "
         "then rebounds → U-shaped ceiling)      v2:  3.5 at every distance  (isometry → flat ceiling)",
         ha="center", fontsize=10, color="#333", fontweight="medium")

fig.tight_layout(rect=[0.02, 0.085, 1, 0.93])
out = os.environ.get("FIG_OUT", "embedding_2d_v1_v2.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("saved", out)

# quick sanity: min pairwise centroid gap vs d, both versions (the U-shape story)
def min_gap(C):
    from itertools import combinations
    d = [np.linalg.norm(C[i]-C[j]) for i,j in combinations(range(16),2)]
    return min(d)
print("v1 min-centroid-gap:", {d: round(min_gap(v1_target_centroids(d)),2) for d in DISTS})
print("v2 min-centroid-gap:", {d: round(min_gap(v2_target_centroids(d)),2) for d in DISTS})
