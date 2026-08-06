"""Measure the geometry of REAL PLM embeddings so the synthetic generator can imitate it.

Questions:
  1. How separated are families relative to their own spread? (sets the ceiling)
  2. Is the domain shift a SHARED direction (all families move together) or
     family-specific scatter? (decides the right shift model)
  3. How big is the real shift relative to the inter-family gap? (calibrates d)
  4. Effective rank / anisotropy / family-size skew. (realism knobs)
"""
import json, os, sys
import numpy as np

ROOT = "/home/dude/sapelo2_files/scratch"
SETS = {
    "bact->arch (ESM-2 650M)": f"{ROOT}/somedir/embeddings",
    "swissprot ESM-C":          f"{ROOT}/swissprot_esmc/embeddings",
    "swissprot ESM-2":          f"{ROOT}/swissprot_esm2/embeddings",
    "ladder: archaea":          f"{ROOT}/taxonomy_ladder/archaea",
    "ladder: fungi":            f"{ROOT}/taxonomy_ladder/fungi",
    "ladder: metazoa":          f"{ROOT}/taxonomy_ladder/metazoa",
    "ladder: plants":           f"{ROOT}/taxonomy_ladder/plants",
}

def eff_rank(X):
    """Participation ratio of the covariance spectrum: (sum l)^2 / sum l^2."""
    Xc = X - X.mean(0)
    n = min(len(Xc), 4000)
    s = np.linalg.svd(Xc[:n], compute_uv=False)
    l = s ** 2
    return float(l.sum() ** 2 / (l ** 2).sum())

def centroids(X, y, klass):
    return np.stack([X[y == k].mean(0) for k in klass])

def within_sigma(X, y, klass):
    """RMS distance of a point to its own family centroid (per-family, then mean)."""
    out = []
    for k in klass:
        Z = X[y == k]
        if len(Z) < 2: continue
        out.append(np.sqrt(((Z - Z.mean(0)) ** 2).sum(1).mean()))
    return float(np.mean(out))

for name, path in SETS.items():
    sx, sy = f"{path}/source_Shf0.0_X.npy", f"{path}/source_Shf0.0_y.npy"
    tx, ty = f"{path}/test_Shf0.0_X.npy", f"{path}/test_Shf0.0_y.npy"
    if not all(os.path.exists(p) for p in (sx, sy, tx, ty)):
        print(f"\n### {name}: MISSING ({path})"); continue
    S, Sy = np.load(sx), np.load(sy)
    T, Ty = np.load(tx), np.load(ty)
    klass = sorted(set(np.unique(Sy)) & set(np.unique(Ty)))
    print(f"\n### {name}")
    print(f"  source {S.shape}, target {T.shape}, {len(klass)} shared families")

    Cs, Ct = centroids(S, Sy, klass), centroids(T, Ty, klass)
    ws_s, ws_t = within_sigma(S, Sy, klass), within_sigma(T, Ty, klass)
    iu = np.triu_indices(len(klass), 1)
    gaps = np.sqrt(((Cs[:, None] - Cs[None]) ** 2).sum(-1))[iu]

    print(f"  within-family sigma:  source {ws_s:.2f}  target {ws_t:.2f}")
    print(f"  inter-family gap:     mean {gaps.mean():.2f}  min {gaps.min():.2f}")
    print(f"  SEPARATION RATIO mean_gap/within_sigma = {gaps.mean()/ws_s:.2f}"
          f"   (min_gap/ws = {gaps.min()/ws_s:.2f})")

    # --- the shift ---
    V = Ct - Cs                              # per-family shift vector
    mag = np.linalg.norm(V, axis=1)
    vbar = V.mean(0)
    shared_frac = (np.linalg.norm(vbar) ** 2) / (mag ** 2).mean()
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    cos = (Vn @ Vn.T)[iu]
    print(f"  shift magnitude:      mean {mag.mean():.2f}  (sd {mag.std():.2f})")
    print(f"  SHIFT / within_sigma  = {mag.mean()/ws_s:.2f}")
    print(f"  SHIFT / mean_gap      = {mag.mean()/gaps.mean():.2f}   <-- 'real d'")
    print(f"  shared-direction frac = {shared_frac:.3f} "
          f"(1.0 = pure common translation, 0 = all family-specific)")
    print(f"  pairwise cos(shift_i, shift_j): mean {cos.mean():+.3f}")

    # --- realism knobs ---
    print(f"  effective rank: source {eff_rank(S):.1f} / {S.shape[1]}  "
          f"target {eff_rank(T):.1f}")
    cnt = np.array([(Sy == k).sum() for k in klass])
    print(f"  family sizes: min {cnt.min()} max {cnt.max()} "
          f"max/min {cnt.max()/max(cnt.min(),1):.1f}")
    # per-family within-sigma spread (are families equally tight?)
    per = [np.sqrt(((S[Sy==k] - S[Sy==k].mean(0))**2).sum(1).mean())
           for k in klass if (Sy==k).sum() > 1]
    print(f"  per-family sigma varies: {min(per):.2f} .. {max(per):.2f} "
          f"(ratio {max(per)/min(per):.2f})")
