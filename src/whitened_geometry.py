#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does beta mean what its name says once magnitude is measured in whitened units?

Open item since 6 August: beta is the fraction of a shift's squared length that lands
inside the function subspace B, and on real data it correlates POSITIVELY with retained
F1 (rho = +0.280), which is the wrong sign for a damage measure. The proposed explanation
is that B is built by SVD and is therefore ordered by variance, so "inside B" mostly means
"along a roomy axis". The proposed repair is to define magnitude and beta in whitened
units, where one unit means the same thing in every direction. It was never run.

This runs it. Every geometry predictor is computed twice, once in raw coordinates and once
in coordinates whitened on the SOURCE domain only (the honest choice: at prediction time
you have not seen the target). Both are correlated against the same `retained` values from
the 209-pair linear-probe scan, with a group-level permutation null.

Usage:
  python whitened_geometry.py --out /scratch/lmk04992/whitened/whitened_geometry.json
"""
import argparse, json, itertools, time
import numpy as np

EMB = "/scratch/lmk04992/ec_swissprot/emb_cache_esmc.npy"
META = "/scratch/lmk04992/ec_swissprot/data/metadata.tsv"
ECF = "/scratch/lmk04992/ec_swissprot/data/ec_annotations.tsv"
RETAINED = "/scratch/lmk04992/ec_rstar/rstar_allpairs.json"
DROP = {"other_bacteria", "other_eukaryota"}


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 4:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else float("nan")


def load():
    ids, fams, groups = [], [], []
    with open(META) as f:
        h = f.readline().rstrip("\n").split("\t"); ci = {c: i for i, c in enumerate(h)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            ids.append(p[ci["id"]]); fams.append(p[ci["family"]]); groups.append(p[ci["group"]])
    ec = {}
    with open(ECF) as f:
        h = f.readline().rstrip("\n").split("\t"); ci = {c: i for i, c in enumerate(h)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            parts = p[ci["ec_full"]].split(".")
            if len(parts) >= 3 and all(x.isdigit() for x in parts[:3]):
                ec[p[ci["id"]]] = ".".join(parts[:3])
    return np.array(ids), np.array(groups), ec


def centroids(X, lab, keep):
    return {c: X[lab == c].mean(0) for c in keep}


def procrustes_disparity(A, B):
    """standard orthogonal Procrustes disparity between two centroid configurations"""
    A = A - A.mean(0); B = B - B.mean(0)
    nA, nB = np.linalg.norm(A), np.linalg.norm(B)
    if nA < 1e-12 or nB < 1e-12:
        return float("nan")
    A, B = A / nA, B / nB
    s = np.linalg.svd(A.T @ B, compute_uv=False).sum()
    return float(max(0.0, 1.0 - s ** 2))


def geometry(Cs, Ct, shared, Bbasis, gap):
    V = np.stack([Ct[c] - Cs[c] for c in shared])
    vbar = V.mean(0)
    cos = []
    for i, j in itertools.combinations(range(len(V)), 2):
        a, b = V[i], V[j]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-12 and nb > 1e-12:
            cos.append(float(a @ b / (na * nb)))
    nvb = np.linalg.norm(vbar)
    pb = Bbasis.T @ vbar
    A = np.stack([Cs[c] for c in shared]); Bc = np.stack([Ct[c] for c in shared])
    return {
        "mag_over_gap": float(nvb / gap),
        "mean_abs_over_gap": float(np.mean([np.linalg.norm(v) for v in V]) / gap),
        "alpha": float(np.mean(cos)) if cos else float("nan"),
        "beta_shared": float((pb @ pb) / (nvb ** 2)) if nvb > 1e-12 else float("nan"),
        "diff_abs": float(np.mean([np.linalg.norm(v - vbar) for v in V]) / gap),
        "procrustes": procrustes_disparity(A, Bc),
        "n_shared": int(len(shared)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_n", type=int, default=15)
    ap.add_argument("--var_frac", type=float, default=0.90)
    ap.add_argument("--n_perm", type=int, default=5000)
    a = ap.parse_args()

    ids, groups, ecmap = load()
    have = np.array([i in ecmap for i in ids])
    eclab = np.array([ecmap.get(i, "") for i in ids])
    gs = sorted(set(groups[have]) - DROP)
    log("groups:", len(gs))

    log("loading embeddings")
    X = np.load(EMB, mmap_mode="r")

    idx = {g: np.where((groups == g) & have)[0] for g in gs}
    Xg, Eg = {}, {}
    for g in gs:
        rows = np.sort(idx[g])
        Xg[g] = np.asarray(X[rows], dtype=np.float64)
        Eg[g] = eclab[rows]
        log("  %-24s %6d proteins" % (g, len(rows)))

    ret = {}
    R = json.load(open(RETAINED))["rows"]
    for r in R:
        ret[(r["source"], r["target"])] = r["retained"]
    log("retained values loaded for %d pairs" % len(ret))

    recs = []
    for s in gs:
        Xs, Es = Xg[s], Eg[s]
        # whitening fitted on the SOURCE only
        mu = Xs.mean(0)
        U, S, Vt = np.linalg.svd(Xs - mu, full_matrices=False)
        keep = S > S.max() * 1e-6
        W = Vt[keep].T / (S[keep] / np.sqrt(len(Xs)))
        u, c = np.unique(Es, return_counts=True)
        big_s = set(u[c >= a.min_n])

        for space in ("raw", "whitened"):
            Xs_ = Xs if space == "raw" else (Xs - mu) @ W
            Cs_all = centroids(Xs_, Es, big_s)
            Cm = np.stack([Cs_all[c] for c in sorted(big_s)])
            dists = [np.linalg.norm(Cm[i] - Cm[j])
                     for i, j in itertools.combinations(range(len(Cm)), 2)]
            gap = float(np.mean(dists))
            Cc = Cm - Cm.mean(0)
            _, sv, vt = np.linalg.svd(Cc, full_matrices=False)
            ev = sv ** 2
            k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), a.var_frac) + 1)
            Bb = vt[:k].T

            for t in gs:
                if t == s or (s, t) not in ret:
                    continue
                Xt, Et = Xg[t], Eg[t]
                Xt_ = Xt if space == "raw" else (Xt - mu) @ W
                ut, ct = np.unique(Et, return_counts=True)
                big_t = set(ut[ct >= a.min_n])
                shared = sorted(big_s & big_t)
                if len(shared) < 5:
                    continue
                Ct_ = centroids(Xt_, Et, set(shared))
                g_ = geometry({c: Cs_all[c] for c in shared}, Ct_, shared, Bb, gap)
                g_.update(source=s, target=t, space=space, gap=gap, subspace_dim=int(k),
                          retained=ret[(s, t)])
                recs.append(g_)
        log("  source %-24s done" % s)

    out = {"config": vars(a), "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "n_records": len(recs), "records": recs}

    PRED = ["mag_over_gap", "mean_abs_over_gap", "alpha", "beta_shared", "diff_abs", "procrustes"]
    rng = np.random.default_rng(0)
    corr = {}
    for space in ("raw", "whitened"):
        rs = [r for r in recs if r["space"] == space]
        y = np.array([r["retained"] for r in rs])
        srcs = np.array([r["source"] for r in rs])
        tgts = np.array([r["target"] for r in rs])
        corr[space] = {}
        for p in PRED:
            x = np.array([r[p] for r in rs])
            rho = spearman(x, y)
            # group-level permutation: relabel the 15 groups, keep the pair structure
            cnt = 0
            for _ in range(a.n_perm):
                perm = {g: h for g, h in zip(gs, rng.permutation(gs))}
                key = {(r["source"], r["target"]): r[p] for r in rs}
                xp = np.array([key.get((perm[s_], perm[t_]), np.nan)
                               for s_, t_ in zip(srcs, tgts)])
                rp = spearman(xp, y)
                if np.isfinite(rp) and abs(rp) >= abs(rho):
                    cnt += 1
            corr[space][p] = {"rho": round(rho, 4),
                              "p_group_perm": round((cnt + 1) / (a.n_perm + 1), 4),
                              "n": int(len(rs))}
            log("  %-9s %-20s rho=%+.3f  p=%.4f" % (space, p, rho, corr[space][p]["p_group_perm"]))
    out["correlations"] = corr
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    log("wrote", a.out)


if __name__ == "__main__":
    main()
