#!/usr/bin/env python
"""A scorecard for "does the synthetic embedding space look like the real one?"

Why more than magnitude / alpha / beta
--------------------------------------
Those three describe the SHIFT, and all three are computed from class CENTROIDS --
first moments. Two datasets can agree on every one of them and still be nothing
alike, because a centroid says nothing about:

  * the shape of the cloud       (is it low-rank the way ESM-C is? effective rank
                                  3-11 of 960 is a strong constraint the generator
                                  has to satisfy for "orthogonal direction" to mean
                                  the same thing in both)
  * the LOCAL neighbourhood      (a k-NN classifier and a small-sample linear probe
                                  are driven by who is next to whom, not by where
                                  the centroids are)
  * how classes are arranged     (real EC classes are unbalanced and hierarchical;
                                  a random oracle's are neither)

The recovery-threshold experiment the thesis rests on is a SMALL-SAMPLE adaptation
problem, which is precisely the regime where local structure matters more than
centroid geometry. So the scorecard below is deliberately weighted toward
distribution- and neighbourhood-level statistics.

Every statistic is computed identically for real and synthetic input, so the two
columns are directly comparable and the gaps are the to-do list.

Usage
-----
  real:     --real_emb X.npy --real_meta metadata.tsv --real_ec ec.tsv \
            --source_group gammaproteobacteria --target_group vertebrata
  synthetic: --synth_dir <dir with source_Shf0.0_{X,y}.npy and test_Shf1.0_{X,y}.npy>
Give both to get a side-by-side table.
"""
import argparse
import json
import os

import numpy as np


# ------------------------------------------------------------------ helpers
def subsample(X, y, n, rng):
    if len(X) <= n:
        return X, y
    i = rng.choice(len(X), n, replace=False)
    return X[i], y[i]


def pairwise_dist(A):
    """Euclidean distance matrix via the Gram trick.

    The broadcast form (A[:, None] - A[None]) materialises an (n, n, d) array --
    4000 x 4000 x 960 floats is 60 GB. This is the same numbers in n^2 memory.
    """
    sq = (A ** 2).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (A @ A.T)
    np.maximum(D2, 0, out=D2)
    return np.sqrt(D2)


def effective_rank(X):
    """Participation ratio of the eigenvalue spectrum: (sum L)^2 / sum L^2.

    Reads as "how many directions does this cloud really use". Insensitive to the
    long tail of near-zero eigenvalues, unlike a 90%-variance cutoff, so both are
    reported.
    """
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    lam = lam / lam.sum()
    er = float(1.0 / (lam ** 2).sum())
    d90 = int(np.searchsorted(np.cumsum(lam), 0.90) + 1)
    return er, d90


def twonn_id(X, rng, n=3000):
    """Intrinsic dimension by the TwoNN estimator (Facco et al.).

    Uses only the ratio of the 2nd to the 1st nearest-neighbour distance, so it is
    a local measure and does not care about global curvature -- which is what makes
    it complementary to effective rank.
    """
    Xs, _ = subsample(X, np.zeros(len(X)), n, rng)
    D = pairwise_dist(Xs)
    np.fill_diagonal(D, np.inf)
    part = np.partition(D, 1, axis=1)[:, :2]
    r1, r2 = part[:, 0], part[:, 1]
    ok = (r1 > 1e-12) & (r2 > r1)
    mu = r2[ok] / r1[ok]
    return float(len(mu) / np.log(mu).sum())


def knn_stats(X, y, rng, k=10, n=4000):
    """Local label purity and hubness.

    knn_label_consistency: share of a point's k neighbours carrying its own label.
    A synthetic space can match every centroid statistic and still get this wrong,
    and it is what a small-sample probe actually feels.

    hubness_skew: skewness of the k-occurrence count (how often a point appears in
    other points' neighbour lists). High-dimensional real embeddings are hub-heavy;
    isotropic Gaussian mixtures are not, so this separates them sharply.
    """
    Xs, ys = subsample(X, y, n, rng)
    D = pairwise_dist(Xs)
    np.fill_diagonal(D, np.inf)
    nn = np.argpartition(D, k, axis=1)[:, :k]
    cons = float((ys[nn] == ys[:, None]).mean())
    occ = np.bincount(nn.ravel(), minlength=len(Xs)).astype(float)
    sd = occ.std()
    skew = float((((occ - occ.mean()) / sd) ** 3).mean()) if sd > 1e-12 else float("nan")
    return cons, skew


def class_geometry(X, y, min_n):
    """Centroid separation, within-class scatter, and whether classes share a shape."""
    keys = [c for c in np.unique(y) if (y == c).sum() >= min_n]
    if len(keys) < 3:
        return None
    C = np.stack([X[y == c].mean(0) for c in keys])
    iu = np.triu_indices(len(C), 1)
    gap = float(pairwise_dist(C)[iu].mean())
    sig = float(np.mean([np.sqrt(((X[y == c] - X[y == c].mean(0)) ** 2).sum(1).mean())
                         for c in keys]))
    # do the classes share a covariance orientation? mean cos^2 between the leading
    # eigenvector of each pair of class covariances. The generator gives every class
    # the same covariance, which pins this near 1.
    tops = []
    for c in keys:
        Z = X[y == c] - X[y == c].mean(0)
        if len(Z) < 3:
            continue
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        tops.append(Vt[0])
    align = float("nan")
    if len(tops) >= 2:
        Tm = np.stack(tops)
        G = (Tm @ Tm.T) ** 2
        iu2 = np.triu_indices(len(Tm), 1)
        align = float(G[iu2].mean())
    sizes = np.array(sorted([(y == c).sum() for c in keys], reverse=True), dtype=float)
    r = np.arange(1, len(sizes) + 1, dtype=float)
    zipf = float(np.polyfit(np.log(r), np.log(sizes), 1)[0]) if len(sizes) >= 4 else float("nan")
    return dict(n_classes=len(keys), gap=gap, within_sigma=sig,
                separation_ratio=gap / max(sig, 1e-12),
                class_cov_alignment=align, class_size_zipf_slope=zipf)


def procrustes_disparity(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    A = A / max(np.linalg.norm(A), 1e-12)
    B = B / max(np.linalg.norm(B), 1e-12)
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return float(2 - 2 * s.sum())


def shift_geometry(Xs, ys, Xt, yt, min_n):
    """The three knobs the generator already has, plus the configuration distortion."""
    keys = [c for c in np.unique(ys)
            if (ys == c).sum() >= min_n and (yt == c).sum() >= min_n]
    if len(keys) < 3:
        return None
    Cs = np.stack([Xs[ys == c].mean(0) for c in keys])
    Ct = np.stack([Xt[yt == c].mean(0) for c in keys])
    V = Ct - Cs
    iu = np.triu_indices(len(Cs), 1)
    gap = float(pairwise_dist(Cs)[iu].mean())
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    cos = (Vn @ Vn.T)[np.triu_indices(len(V), 1)]
    mag = np.linalg.norm(V, axis=1)
    vbar = V.mean(0)

    # beta against the function subspace of the SOURCE, defined exactly as elsewhere
    Cc = Cs - Cs.mean(0)
    _, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    kdim = int(np.searchsorted(np.cumsum(S ** 2) / (S ** 2).sum(), 0.90) + 1)
    B = Vt[:kdim]
    beta_shared = float((np.linalg.norm(B @ vbar) ** 2) /
                        max(np.linalg.norm(vbar) ** 2, 1e-12))
    return dict(n_shared_classes=len(keys),
                mag_over_gap=float(mag.mean() / gap),
                alpha=float(cos.mean()),
                shared_frac=float(np.linalg.norm(vbar) ** 2 / (mag ** 2).mean()),
                beta_shared=beta_shared,
                function_subspace_dim=kdim,
                differential_over_gap=float(np.linalg.norm(V - vbar, axis=1).mean() / gap),
                procrustes=procrustes_disparity(Cs, Ct))


def additivity_two_group(Xs, ys, Xt, yt, min_n):
    """Interaction share of the (group x class) grid when there are exactly 2 groups.

    With two groups the additive model is "every class moves by the same vector", so
    the interaction share is exactly 1 - shared_frac of the shift. Reported under the
    same name as the multi-group version so the two are comparable.
    """
    keys = [c for c in np.unique(ys)
            if (ys == c).sum() >= min_n and (yt == c).sum() >= min_n]
    if len(keys) < 3:
        return None
    Cs = np.stack([Xs[ys == c].mean(0) for c in keys])
    Ct = np.stack([Xt[yt == c].mean(0) for c in keys])
    Y = np.concatenate([Cs, Ct])
    g = np.array([0] * len(Cs) + [1] * len(Ct))
    cls = np.concatenate([np.arange(len(Cs))] * 2)

    def resid(with_g, with_c):
        blocks = [np.ones((len(Y), 1))]
        if with_g:
            blocks.append(np.stack([(g == v).astype(float) for v in [0, 1]], 1))
        if with_c:
            blocks.append(np.stack([(cls == v).astype(float) for v in range(len(Cs))], 1))
        A = np.concatenate(blocks, 1)
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        return float(((Y - A @ coef) ** 2).sum())

    tot = resid(False, False)
    return dict(frac_explained_by_additive=1 - resid(True, True) / tot,
                frac_interaction_residual=resid(True, True) / tot)


def scorecard(Xs, ys, Xt, yt, min_n, rng, label):
    print(f"\n########## {label} ##########")
    print(f"  source {Xs.shape}, target {Xt.shape}, "
          f"{len(np.unique(ys))} source classes")
    out = {"n_source": int(len(Xs)), "n_target": int(len(Xt)), "dim": int(Xs.shape[1])}

    er, d90 = effective_rank(Xs)
    out["effective_rank"] = er
    out["dims_to_90pct_variance"] = d90
    out["intrinsic_dim_twonn"] = twonn_id(Xs, rng)
    cons, skew = knn_stats(Xs, ys, rng)
    out["knn10_label_consistency"] = cons
    out["knn10_hubness_skew"] = skew
    cg = class_geometry(Xs, ys, min_n)
    if cg:
        out.update({f"src_{k}": v for k, v in cg.items()})
    sg = shift_geometry(Xs, ys, Xt, yt, min_n)
    if sg:
        out.update({f"shift_{k}": v for k, v in sg.items()})
    ad = additivity_two_group(Xs, ys, Xt, yt, min_n)
    if ad:
        out.update({f"add_{k}": v for k, v in ad.items()})
    for k, v in out.items():
        if isinstance(v, float):
            print(f"    {k:38s} {v:10.4f}")
        else:
            print(f"    {k:38s} {v:10d}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real_emb"); ap.add_argument("--real_meta"); ap.add_argument("--real_ec")
    ap.add_argument("--source_group", default=None)
    ap.add_argument("--target_group", default=None)
    ap.add_argument("--ec_level", type=int, default=3)
    ap.add_argument("--synth_dir", default=None)
    ap.add_argument("--synth_source", default="source_Shf0.0")
    ap.add_argument("--synth_target", default="test_Shf1.0")
    ap.add_argument("--min_n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cards = {}

    if args.real_emb:
        from measure_ec_geometry import load_table
        T = load_table(args.real_emb, args.real_meta, args.real_ec, args.ec_level)
        ms = T["dom"] == args.source_group
        mt = T["dom"] == args.target_group
        if ms.sum() == 0 or mt.sum() == 0:
            raise SystemExit(f"empty group: {sorted(set(T['dom']))}")
        cards["real"] = scorecard(T["X"][ms], T["ec"][ms], T["X"][mt], T["ec"][mt],
                                  args.min_n, rng,
                                  f"REAL  {args.source_group} -> {args.target_group}  "
                                  f"(labels = EC level {args.ec_level})")

    if args.synth_dir:
        d = args.synth_dir
        Xs = np.load(os.path.join(d, f"{args.synth_source}_X.npy"))
        ys = np.load(os.path.join(d, f"{args.synth_source}_y.npy"))
        Xt = np.load(os.path.join(d, f"{args.synth_target}_X.npy"))
        yt = np.load(os.path.join(d, f"{args.synth_target}_y.npy"))
        cards["synthetic"] = scorecard(Xs, ys, Xt, yt, args.min_n, rng,
                                       f"SYNTHETIC  {os.path.basename(d.rstrip('/'))}")

    if len(cards) == 2:
        print("\n########## side by side ##########")
        keys = [k for k in cards["real"] if k in cards["synthetic"]]
        print(f"  {'statistic':38s} {'real':>12s} {'synthetic':>12s} {'ratio':>9s}")
        print("  " + "-" * 74)
        for k in keys:
            r, s = cards["real"][k], cards["synthetic"][k]
            if not isinstance(r, (int, float)) or not isinstance(s, (int, float)):
                continue
            ratio = s / r if abs(r) > 1e-9 else float("nan")
            print(f"  {k:38s} {r:12.4f} {s:12.4f} {ratio:9.2f}")
        print("\n  ratio near 1 = the generator reproduces that property.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(cards, f, indent=2)
        print(f"\nWrote {args.out}")
