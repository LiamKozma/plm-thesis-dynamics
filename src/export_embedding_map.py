#!/usr/bin/env python
"""
Export a 2-D map of the real ESM-C embeddings for the dashboard.

The point of the figure this feeds: the same proteins carry three different
labels (taxonomic group, EC number, Pfam family), and they are three ways of
grouping one set of points. Projecting the same cloud two different ways then
shows why the project is hard, because the directions carrying the most variance
are not the directions carrying function.

Two projections, computed on identical points:
  raw : top 2 principal components of the embeddings, i.e. the widest directions
  fun : top 2 directions of separation between EC-class centroids, which is the
        function subspace B defined on the Measures page

Output: docs/figures/embedding_map.json
"""
import json, os, sys
import numpy as np

ROOT = "/scratch/lmk04992/ec_swissprot"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "figures", "embedding_map.json")

N_PLOT = 6000      # points drawn in the browser
N_FIT = 60000      # points used to estimate the projections
SEED = 42
MIN_PER_EC = 150   # an EC L3 class must be this common to be its own colour

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- labels
ids, fam, grp = [], [], []
with open(os.path.join(ROOT, "data", "metadata.tsv")) as fh:
    assert fh.readline().split() == ["id", "family", "group"]
    for line in fh:
        a, b, c = line.rstrip("\n").split("\t")
        ids.append(a); fam.append(b); grp.append(c)

ec_by_id = {}
with open(os.path.join(ROOT, "data", "ec_annotations.tsv")) as fh:
    fh.readline()
    for line in fh:
        p = line.rstrip("\n").split("\t")
        ec_by_id[p[0]] = p[1]

fam = np.array(fam); grp = np.array(grp); ids = np.array(ids)
ec_raw = np.array([ec_by_id.get(i, "") for i in ids])

# EC level 3 only where all three levels are real numbers
def ec3(e):
    p = e.split(".")
    if len(p) < 3 or any(x in ("-", "") for x in p[:3]):
        return ""
    return ".".join(p[:3])

ec3s = np.array([ec3(e) for e in ec_raw])
usable = ec3s != ""
print("proteins: %d, with a clean EC L3: %d" % (len(ids), usable.sum()), flush=True)

emb = np.load(os.path.join(ROOT, "emb_cache_esmc.npy"), mmap_mode="r")
assert emb.shape[0] == len(ids), (emb.shape, len(ids))

# ---------------------------------------------------------------- samples
idx_all = np.flatnonzero(usable)
fit_idx = np.sort(rng.choice(idx_all, size=min(N_FIT, idx_all.size), replace=False))
X = np.asarray(emb[fit_idx], dtype=np.float64)
print("fit matrix", X.shape, flush=True)

mu = X.mean(0)
Xc = X - mu

# --- projection 1: the widest directions (plain PCA)
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
P_raw = Vt[:2].T

# --- projection 2: the directions that separate EC classes
lab = ec3s[fit_idx]
uniq, counts = np.unique(lab, return_counts=True)
keep = uniq[counts >= 30]
cents = np.stack([Xc[lab == u].mean(0) for u in keep])
print("EC L3 centroids used for the function projection: %d" % len(cents), flush=True)
_, _, Vf = np.linalg.svd(cents - cents.mean(0), full_matrices=False)
P_fun = Vf[:2].T

# ---------------------------------------------------------------- plot set
plot_idx = np.sort(rng.choice(fit_idx, size=min(N_PLOT, fit_idx.size), replace=False))
pos = np.searchsorted(fit_idx, plot_idx)
Xp = Xc[pos]

A = Xp @ P_raw
B = Xp @ P_fun

def norm(M):
    M = M - M.mean(0)
    s = np.percentile(np.abs(M), 99)
    return np.clip(M / (s if s else 1.0), -1.6, 1.6)

A, B = norm(A), norm(B)

# ---------------------------------------------------------------- colour keys
g = grp[plot_idx]
e3 = ec3s[plot_idx]
e1 = np.array([x.split(".")[0] for x in e3])
f = fam[plot_idx]

def top_levels(arr, n, minc):
    u, c = np.unique(arr, return_counts=True)
    o = np.argsort(-c)
    return [x for x, k in zip(u[o], c[o]) if k >= minc][:n]

top_fam = top_levels(f, 12, 40)
top_ec3 = top_levels(e3, 12, MIN_PER_EC)

groups = sorted(set(g.tolist()))
rec = {
    "n": int(len(plot_idx)),
    "groups": groups,
    "ec1": sorted(set(e1.tolist())),
    "top_fam": top_fam,
    "top_ec3": top_ec3,
    "x_raw": [round(float(v), 3) for v in A[:, 0]],
    "y_raw": [round(float(v), 3) for v in A[:, 1]],
    "x_fun": [round(float(v), 3) for v in B[:, 0]],
    "y_fun": [round(float(v), 3) for v in B[:, 1]],
    "g":   [groups.index(v) for v in g],
    "e1":  [int(v) if v.isdigit() else -1 for v in e1],
    "e3":  [top_ec3.index(v) if v in top_ec3 else -1 for v in e3],
    "f":   [top_fam.index(v) if v in top_fam else -1 for v in f],
    "meta": {
        "source": "ec_swissprot", "n_total": int(len(ids)),
        "n_clean_ec3": int(usable.sum()), "n_fit": int(len(fit_idx)),
        "model": "ESM-C 300M, 960-D mean-pooled", "seed": SEED,
        "proj_raw": "top 2 principal components of the raw embeddings",
        "proj_fun": "top 2 singular directions of the EC L3 centroid matrix",
        "n_ec_centroids": int(len(cents)),
    },
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(rec, fh, separators=(",", ":"))
print("wrote %s (%.1f KB)" % (OUT, os.path.getsize(OUT) / 1024), flush=True)
