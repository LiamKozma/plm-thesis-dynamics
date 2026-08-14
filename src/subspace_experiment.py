#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homology subspace vs function subspace in a protein language model.

Question (from the 12 Aug meeting): the generator splits latent space into a
"signal" subspace that carries the label and a "nuisance" subspace that does not.
On real embeddings there are TWO candidate signal subspaces, because there are two
label axes:

    H = the directions along which Pfam families separate   (homology)
    B = the directions along which EC classes separate      (function)

If H and B were orthogonal, homology and function would be independently readable
and a shift along H would leave function prediction untouched. If H contains B,
then "the EC probe is reading homology" and any homology-directed shift damages
function prediction just as much.

This script measures the relationship three ways:

  PART 1  geometry      principal angles between H and B, against a random-subspace
                        null, plus the variance of each label axis captured by the
                        other's subspace.
  PART 2  ablation      project embeddings onto / out of each subspace and measure
                        what a probe can still read.
  PART 3  displacement  the causal test. Apply a shift of fixed length drawn from
                        the data covariance (matched control, per landmine 1 of
                        BRIEF_ec_recovery_threshold.md), restricted to
                        B, H, H-minus-B, B-minus-H, or the complement of both,
                        and measure retained macro-F1 for EC and for Pfam.

Everything runs on the cached ESM-C embeddings; nothing is re-embedded.

Usage:
  python subspace_experiment.py --outdir /scratch/lmk04992/subspace --domain gammaproteobacteria
"""
import argparse, json, os, sys, time
import numpy as np

EMB = "/scratch/lmk04992/ec_swissprot/emb_cache_esmc.npy"
META = "/scratch/lmk04992/ec_swissprot/data/metadata.tsv"
ECF = "/scratch/lmk04992/ec_swissprot/data/ec_annotations.tsv"


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------- data
def load(domain, ec_level, min_n_ec, min_n_fam, max_n, seed, emb_path=None, ids_path=None):
    log("loading metadata")
    ids, fams, groups = [], [], []
    with open(META) as f:
        head = f.readline().rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(head)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            ids.append(p[ci["id"]]); fams.append(p[ci["family"]]); groups.append(p[ci["group"]])
    ids = np.array(ids); fams = np.array(fams); groups = np.array(groups)
    log("metadata rows:", len(ids))

    ec_map = {}
    with open(ECF) as f:
        head = f.readline().rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(head)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            full = p[ci["ec_full"]] if "ec_full" in ci else p[ci["ec"]]
            parts = full.split(".")
            if len(parts) >= ec_level and all(x.isdigit() for x in parts[:ec_level]):
                ec_map[p[ci["id"]]] = ".".join(parts[:ec_level])
    log("proteins with a clean EC at level %d: %d" % (ec_level, len(ec_map)))

    restrict = None
    if ids_path:
        with open(ids_path) as f:
            restrict = set(x.strip() for x in f if x.strip())
        log("restricting to %d ids from %s" % (len(restrict), ids_path))
        # the alternate cache is stored in the order of ids.txt, so build that index
        alt_order = [x.strip() for x in open(ids_path) if x.strip()]
        alt_pos = {i: k for k, i in enumerate(alt_order)}

    sel = np.where(groups == domain)[0]
    if restrict is not None:
        sel = np.array([i for i in sel if ids[i] in restrict])
    log("%s rows: %d" % (domain, len(sel)))
    keep = [i for i in sel if ids[i] in ec_map]
    keep = np.array(keep)
    log("with EC: %d" % len(keep))

    ec = np.array([ec_map[ids[i]] for i in keep])
    fam = fams[keep]

    # keep classes with enough members on BOTH axes
    def big(labels, m):
        u, c = np.unique(labels, return_counts=True)
        return set(u[c >= m])
    ok_ec, ok_fam = big(ec, min_n_ec), big(fam, min_n_fam)
    m = np.array([(e in ok_ec) and (f in ok_fam) for e, f in zip(ec, fam)])
    keep, ec, fam = keep[m], ec[m], fam[m]
    log("after min_n filter: %d proteins, %d EC classes, %d Pfam families"
        % (len(keep), len(set(ec)), len(set(fam))))

    if max_n and len(keep) > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(keep), max_n, replace=False)
        keep, ec, fam = keep[idx], ec[idx], fam[idx]
        log("subsampled to %d" % len(keep))

    log("memory-mapping embeddings")
    if emb_path:
        X = np.load(emb_path, mmap_mode="r")
        rows = np.array([alt_pos[ids[i]] for i in keep])
        Xs = np.asarray(X[np.sort(rows)], dtype=np.float64)
        order = np.argsort(rows)
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
        Xs = Xs[inv]
    else:
        X = np.load(EMB, mmap_mode="r")
        Xs = np.asarray(X[np.sort(keep)], dtype=np.float64)
        order = np.argsort(keep)
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
        Xs = Xs[inv]
    log("embedding block:", Xs.shape)
    return Xs, ec, fam


# ----------------------------------------------------------------- subspaces
def centroids(X, labels):
    u = np.unique(labels)
    C = np.stack([X[labels == c].mean(0) for c in u])
    return C, u


def subspace(X, labels, var_frac=0.90, kmax=None):
    """Top directions of the class-centroid cloud holding var_frac of between-class variance."""
    C, u = centroids(X, labels)
    Cc = C - C.mean(0)
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    ev = S ** 2
    cum = np.cumsum(ev) / ev.sum()
    k = int(np.searchsorted(cum, var_frac) + 1)
    if kmax:
        k = min(k, kmax)
    return Vt[:k].T, k, ev / ev.sum(), C, u          # (960,k)


def principal_angles(A, B):
    """A (d,ka), B (d,kb) orthonormal. Returns cosines of principal angles, descending."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(s, 0, 1)


def captured(C, u, P):
    """Fraction of between-class variance of centroid set C that lies inside projector basis P."""
    Cc = C - C.mean(0)
    tot = (Cc ** 2).sum()
    inside = ((Cc @ P) ** 2).sum()
    return inside / tot


# ----------------------------------------------------------------- probe
def probe_f1(Xtr, ytr, Xte, yte, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, C=1.0, random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    return f1_score(yte, clf.predict(sc.transform(Xte)), average="macro")


def fit_probe(Xtr, ytr, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, C=1.0, random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    return sc, clf


def score(sc, clf, X, y):
    from sklearn.metrics import f1_score
    return f1_score(y, clf.predict(sc.transform(X)), average="macro")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--domain", default="gammaproteobacteria")
    ap.add_argument("--ec_level", type=int, default=3)
    ap.add_argument("--min_n_ec", type=int, default=30)
    ap.add_argument("--min_n_fam", type=int, default=30)
    ap.add_argument("--max_n", type=int, default=20000)
    ap.add_argument("--var_frac", type=float, default=0.90)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--mags", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--whiten", action="store_true",
                    help="whiten the embedding space first, so one unit means the same in every "
                         "direction; this is the repair proposed for beta but never run")
    ap.add_argument("--emb", default=None,
                    help="alternate embedding cache (.npy), rows ordered as --ids")
    ap.add_argument("--ids", default=None,
                    help="id list restricting the analysis; required with --emb, and usable "
                         "alone to run the default cache on exactly the same proteins")
    ap.add_argument("--pca", type=int, default=0,
                    help="reduce to this many principal components before any analysis. Controls "
                         "for the two models having different ambient dimension, and keeps the "
                         "whitening well conditioned (whitening needs n >> d).")
    ap.add_argument("--tag", default="",
                    help="suffix for the output filename, e.g. _esm2")
    ap.add_argument("--n_rand", type=int, default=5,
                    help="random matched-dimension control subspaces per condition")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    out = {"config": vars(args), "generated": time.strftime("%Y-%m-%d %H:%M:%S")}

    X, ec, fam = load(args.domain, args.ec_level, args.min_n_ec, args.min_n_fam,
                      args.max_n, args.seeds[0], args.emb, args.ids)
    out["embedding"] = args.emb or EMB
    out["dim"] = int(X.shape[1])
    if args.pca:
        k = min(args.pca, X.shape[0] - 1, X.shape[1])
        mu0 = X.mean(0)
        _, _, Vt0 = np.linalg.svd(X - mu0, full_matrices=False)
        X = (X - mu0) @ Vt0[:k].T
        log("PCA to %d components; n/d is now %.1f" % (k, len(X) / k))
    out["pca"] = int(args.pca)
    out["n_over_d"] = round(len(X) / X.shape[1], 2)

    if args.whiten:
        log("whitening: rescaling every direction by its own standard deviation")
        mu = X.mean(0)
        Xc0 = X - mu
        U0, S0, Vt0 = np.linalg.svd(Xc0, full_matrices=False)
        keep0 = S0 > (S0.max() * 1e-6)
        W = Vt0[keep0].T / (S0[keep0] / np.sqrt(len(X)))
        X = Xc0 @ W
        log("  whitened block: %s (kept %d directions)" % (str(X.shape), int(keep0.sum())))
    out["whitened"] = bool(args.whiten)
    out["n_proteins"] = int(len(X))
    out["n_ec"] = int(len(set(ec)))
    out["n_fam"] = int(len(set(fam)))

    # ---------------- PART 1: geometry
    log("PART 1  subspace geometry")
    B, kB, evB, C_ec, u_ec = subspace(X, ec, args.var_frac)
    H, kH, evH, C_fam, u_fam = subspace(X, fam, args.var_frac)
    log("  dim B (EC / function)   = %d" % kB)
    log("  dim H (Pfam / homology) = %d" % kH)

    cos = principal_angles(B, H)
    # random-subspace null: same dims, random orthonormal
    rng = np.random.default_rng(0)
    null = []
    for _ in range(50):
        R1 = np.linalg.qr(rng.standard_normal((X.shape[1], kB)))[0]
        R2 = np.linalg.qr(rng.standard_normal((X.shape[1], kH)))[0]
        null.append(principal_angles(R1, R2).mean())
    out["part1"] = {
        "dim_B_function": kB, "dim_H_homology": kH,
        "principal_angle_cosines": [round(float(c), 4) for c in cos],
        "mean_cos_principal_angles": round(float(cos.mean()), 4),
        "n_cos_above_0.9": int((cos > 0.9).sum()),
        "n_cos_above_0.7": int((cos > 0.7).sum()),
        "random_null_mean_cos": round(float(np.mean(null)), 4),
        "random_null_sd": round(float(np.std(null)), 4),
        "ec_variance_inside_H": round(float(captured(C_ec, u_ec, H)), 4),
        "fam_variance_inside_B": round(float(captured(C_fam, u_fam, B)), 4),
        "ec_variance_inside_B": round(float(captured(C_ec, u_ec, B)), 4),
        "fam_variance_inside_H": round(float(captured(C_fam, u_fam, H)), 4),
    }
    log("  mean cos(principal angles) = %.3f   (random null %.3f)"
        % (cos.mean(), np.mean(null)))
    log("  EC between-class variance inside H = %.3f" % out["part1"]["ec_variance_inside_H"])
    log("  Pfam between-class variance inside B = %.3f" % out["part1"]["fam_variance_inside_B"])

    # matched-dimension version so the comparison is not driven by kH != kB
    k = min(kB, kH)
    Bm, _, _, _, _ = subspace(X, ec, 1.0, kmax=k)
    Hm, _, _, _, _ = subspace(X, fam, 1.0, kmax=k)
    cosm = principal_angles(Bm, Hm)
    out["part1"]["matched_k"] = int(k)
    out["part1"]["matched_mean_cos"] = round(float(cosm.mean()), 4)
    out["part1"]["matched_cosines"] = [round(float(c), 4) for c in cosm]
    log("  matched-k (%d) mean cos = %.3f" % (k, cosm.mean()))

    # ---------------- split train/test once
    rng = np.random.default_rng(args.seeds[0])
    perm = rng.permutation(len(X))
    ntr = int(0.6 * len(X))
    tr, te = perm[:ntr], perm[ntr:]

    # ---------------- PART 2: ablation
    log("PART 2  ablation")
    d = X.shape[1]
    def proj(M):          # projector onto span(M)
        Q, _ = np.linalg.qr(M)
        return Q @ Q.T
    PB, PH = proj(B), proj(H)
    I = np.eye(d)
    variants = {
        "full": I,
        "keep_B_only": PB,
        "keep_H_only": PH,
        "remove_B": I - PB,
        "remove_H": I - PH,
        "remove_both": (I - PB) @ (I - PH),
    }
    part2 = {}
    for name, P in variants.items():
        Xp = X @ P.T
        f_ec = probe_f1(Xp[tr], ec[tr], Xp[te], ec[te])
        f_fm = probe_f1(Xp[tr], fam[tr], Xp[te], fam[te])
        part2[name] = {"ec_f1": round(float(f_ec), 4), "fam_f1": round(float(f_fm), 4)}
        log("  %-14s EC %.3f   Pfam %.3f" % (name, f_ec, f_fm))
    out["part2"] = part2

    # ---------------- PART 3: displacement
    log("PART 3  matched displacement")
    # gap = mean pairwise distance between EC centroids, the project's normaliser
    Cc = C_ec
    dists = []
    for i in range(len(Cc)):
        for j in range(i + 1, len(Cc)):
            dists.append(np.linalg.norm(Cc[i] - Cc[j]))
    gap = float(np.mean(dists))
    out["ec_gap"] = round(gap, 4)
    log("  EC gap = %.4f over %d centroid pairs" % (gap, len(dists)))

    # matched control: draw the direction from the data covariance
    Xc = X - X.mean(0)
    cov_sqrt_basis = np.linalg.svd(Xc, full_matrices=False)
    Vd, Sd = cov_sqrt_basis[2].T, cov_sqrt_basis[1]
    def draw_matched(r):
        z = r.standard_normal(Sd.shape[0])
        v = Vd @ (Sd * z)
        return v / np.linalg.norm(v)

    # subspace bases for the five conditions
    def orth_complement_within(A, Bsub):
        """component of span(A) orthogonal to span(Bsub)"""
        Qa, _ = np.linalg.qr(A)
        Qb, _ = np.linalg.qr(Bsub)
        M = Qa - Qb @ (Qb.T @ Qa)
        # re-orthonormalise, dropping near-zero directions
        U, S, _ = np.linalg.svd(M, full_matrices=False)
        keep = S > 1e-8
        return U[:, keep]

    H_not_B = orth_complement_within(H, B)
    B_not_H = orth_complement_within(B, H)
    conds = {
        "B_function": B,
        "H_homology": H,
        "H_minus_B": H_not_B,
        "B_minus_H": B_not_H,
        "outside_both": None,     # complement handled separately
    }
    out["cond_dims"] = {k: (int(v.shape[1]) if v is not None else int(d - kB - H_not_B.shape[1]))
                        for k, v in conds.items()}
    log("  condition dims: %s" % out["cond_dims"])

    sc_ec, clf_ec = fit_probe(X[tr], ec[tr])
    sc_fm, clf_fm = fit_probe(X[tr], fam[tr])
    base_ec = score(sc_ec, clf_ec, X[te], ec[te])
    base_fm = score(sc_fm, clf_fm, X[te], fam[te])
    out["baseline"] = {"ec_f1": round(float(base_ec), 4), "fam_f1": round(float(base_fm), 4)}
    log("  baseline  EC %.3f   Pfam %.3f" % (base_ec, base_fm))

    PBH = proj(np.concatenate([B, H], axis=1))

    # random subspaces of matched dimension: is "H minus B is worst" about H minus B,
    # or about any subspace of that width?
    rr = np.random.default_rng(7)
    rand_bases = {}
    for nm, kk in [("rand_dim%d" % kB, kB), ("rand_dim%d" % H_not_B.shape[1], H_not_B.shape[1])]:
        rand_bases[nm] = [np.linalg.qr(rr.standard_normal((d, kk)))[0] for _ in range(args.n_rand)]
    out["rand_dims"] = {k: int(v[0].shape[1]) for k, v in rand_bases.items()}

    part3 = {}
    for cname in conds:
        for mag in args.mags:
            key = "%s@%.2f" % (cname, mag)
            e_scores, f_scores, sig_units = [], [], []
            for sd in args.seeds:
                r = np.random.default_rng(1000 + sd)
                v = draw_matched(r)
                if cname == "outside_both":
                    v = v - PBH @ v
                else:
                    Q, _ = np.linalg.qr(conds[cname])
                    v = Q @ (Q.T @ v)
                nv = np.linalg.norm(v)
                if nv < 1e-9:
                    continue
                u = v / nv
                sig = float(np.sqrt(np.mean([np.var(X[ec == c] @ u) for c in np.unique(ec)])))
                v = u * (mag * gap)
                sig_units.append((mag * gap) / sig if sig > 0 else float("nan"))
                Xt = X[te] + v
                e_scores.append(score(sc_ec, clf_ec, Xt, ec[te]))
                f_scores.append(score(sc_fm, clf_fm, Xt, fam[te]))
            part3[key] = {
                "mag_in_within_class_sd": round(float(np.mean(sig_units)), 3),
                "ec_f1": round(float(np.mean(e_scores)), 4),
                "ec_sd": round(float(np.std(e_scores)), 4),
                "ec_retained": round(float(np.mean(e_scores) / base_ec), 4),
                "fam_f1": round(float(np.mean(f_scores)), 4),
                "fam_sd": round(float(np.std(f_scores)), 4),
                "fam_retained": round(float(np.mean(f_scores) / base_fm), 4),
            }
            log("  %-22s EC ret %.3f   Pfam ret %.3f"
                % (key, part3[key]["ec_retained"], part3[key]["fam_retained"]))
    for nm, bases in rand_bases.items():
        for mag in args.mags:
            e_scores, f_scores, sig_units = [], [], []
            for bi, Qr in enumerate(bases):
                r = np.random.default_rng(5000 + bi)
                v = draw_matched(r)
                v = Qr @ (Qr.T @ v)
                nv = np.linalg.norm(v)
                if nv < 1e-9:
                    continue
                u = v / nv
                sig = float(np.sqrt(np.mean([np.var(X[ec == c] @ u) for c in np.unique(ec)])))
                v = u * (mag * gap)
                sig_units.append((mag * gap) / sig if sig > 0 else float("nan"))
                Xt = X[te] + v
                e_scores.append(score(sc_ec, clf_ec, Xt, ec[te]))
                f_scores.append(score(sc_fm, clf_fm, Xt, fam[te]))
            part3["%s@%.2f" % (nm, mag)] = {
                "mag_in_within_class_sd": round(float(np.mean(sig_units)), 3),
                "ec_f1": round(float(np.mean(e_scores)), 4),
                "ec_sd": round(float(np.std(e_scores)), 4),
                "ec_retained": round(float(np.mean(e_scores) / base_ec), 4),
                "fam_f1": round(float(np.mean(f_scores)), 4),
                "fam_sd": round(float(np.std(f_scores)), 4),
                "fam_retained": round(float(np.mean(f_scores) / base_fm), 4),
            }
            log("  %-22s EC ret %.3f   Pfam ret %.3f"
                % ("%s@%.2f" % (nm, mag), part3["%s@%.2f" % (nm, mag)]["ec_retained"],
                   part3["%s@%.2f" % (nm, mag)]["fam_retained"]))
    out["part3"] = part3

    p = os.path.join(args.outdir, "subspace_%s%s%s.json" % (args.domain, args.tag, "_whitened" if args.whiten else ""))
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    log("wrote", p)


if __name__ == "__main__":
    main()
