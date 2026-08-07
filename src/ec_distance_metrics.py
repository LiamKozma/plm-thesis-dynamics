#!/usr/bin/env python
"""Distances between taxonomic groups that can be computed BEFORE any labelling.

The practical payoff of r* is telling a scientist their annotation budget in
advance, so a distance that needs target labels is useless for the purpose. Every
metric here is label-free (or, for the taxonomic rank, free from metadata):

  label-free, from the embeddings
    mmd_rbf          maximum mean discrepancy, RBF kernel, median heuristic
    energy_dist      energy distance (kernel-free, no bandwidth to choose)
    frechet          Frechet/FID-style Gaussian distance
    proxy_a_dist     2(1 - 2 err) of a held-out source-vs-target discriminator;
                     the classic Ben-David et al. proxy for H-divergence
    feat_wasserstein mean per-feature 1-D Wasserstein (what metrics.py already had)
    mean_shift_norm  ||mean_t - mean_s||, and the same over the source gap

  free, from the NCBI lineage in raw/ec_swissprot_raw.tsv
    n_shared_lineage number of taxids shared by the two groups' consensus lineages
    lca_rank         NCBI rank of the deepest shared node ("phylum", "domain", ...)
    same_superkingdom / same_domain

The internal-geometry predictors (|v|/gap, ||v - vbar||/gap, Procrustes, probe
logit spread, beta) are NOT recomputed here -- they already exist for all 210
pairs in analysis/beta_diagnosis.json and are joined in by ec_rstar_regress.py.

Subsampling: MMD/energy/Frechet are O(n^2) in memory, so each group is subsampled
to --n_sub proteins with a fixed seed. Reported n is in the output.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_ec_geometry import load_table, drop_groups  # noqa: E402


# ------------------------------------------------------------ label-free pairs
def _cross_sq(A, B):
    """Squared Euclidean distances between rows of A and rows of B, n^2 memory."""
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
                      - 2.0 * (A @ B.T), 0.0)


def mmd_rbf(A, B, gamma=None):
    """Unbiased MMD^2 with an RBF kernel; bandwidth by the median heuristic."""
    Kaa, Kbb, Kab = _cross_sq(A, A), _cross_sq(B, B), _cross_sq(A, B)
    if gamma is None:
        med = np.median(np.concatenate([Kaa[np.triu_indices(len(A), 1)],
                                        Kbb[np.triu_indices(len(B), 1)]]))
        gamma = 1.0 / max(med, 1e-12)
    na, nb = len(A), len(B)
    kaa = (np.exp(-gamma * Kaa).sum() - na) / (na * (na - 1))
    kbb = (np.exp(-gamma * Kbb).sum() - nb) / (nb * (nb - 1))
    kab = np.exp(-gamma * Kab).mean()
    return float(max(kaa + kbb - 2 * kab, 0.0)), float(gamma)


def energy_distance(A, B):
    """2 E|x-y| - E|x-x'| - E|y-y'|. No bandwidth, so nothing to tune or defend."""
    dab = np.sqrt(_cross_sq(A, B)).mean()
    daa = np.sqrt(_cross_sq(A, A))[np.triu_indices(len(A), 1)].mean()
    dbb = np.sqrt(_cross_sq(B, B))[np.triu_indices(len(B), 1)].mean()
    return float(2 * dab - daa - dbb)


def frechet(A, B, eps=1e-6):
    """Gaussian (FID-style) distance. Covariances are shrunk to stay invertible."""
    from scipy import linalg
    mu1, mu2 = A.mean(0), B.mean(0)
    c1 = np.cov(A, rowvar=False) + eps * np.eye(A.shape[1])
    c2 = np.cov(B, rowvar=False) + eps * np.eye(B.shape[1])
    cov, _ = linalg.sqrtm(c1 @ c2, disp=False)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(c1) + np.trace(c2)
                 - 2 * np.trace(cov))


def proxy_a_distance(A, B, seed=0):
    """2(1 - 2 err) of a held-out linear source-vs-target discriminator.

    0 = indistinguishable, 2 = perfectly separable. This is the standard cheap
    stand-in for the H-divergence in the Ben-David domain-adaptation bound, so it
    is the metric with an actual theoretical link to "how much target label
    effort will this cost".
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.vstack([A, B])
    y = np.concatenate([np.zeros(len(A)), np.ones(len(B))])
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(X))
    X, y = X[p], y[p]
    cut = int(0.7 * len(X))
    sc = StandardScaler().fit(X[:cut])
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(sc.transform(X[:cut]), y[:cut])
    err = float((clf.predict(sc.transform(X[cut:])) != y[cut:]).mean())
    return float(2 * (1 - 2 * err)), err


def feat_wasserstein(A, B):
    from scipy.stats import wasserstein_distance
    return float(np.mean([wasserstein_distance(A[:, i], B[:, i])
                          for i in range(A.shape[1])]))


# --------------------------------------------------------------- taxonomy tree
def parse_lineages(raw_tsv, keep_ids):
    """id -> ordered [(taxid, rank)] from the raw fetch's `lineage_ids` column."""
    out = {}
    with open(raw_tsv) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        try:
            i_id, i_lin = hdr.index("id"), hdr.index("lineage_ids")
        except ValueError:
            raise SystemExit(f"{raw_tsv} lacks id/lineage_ids columns: {hdr}")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) <= i_lin or p[i_id] not in keep_ids:
                continue
            chain = []
            for part in p[i_lin].split(", "):
                part = part.strip()
                if not part:
                    continue
                tid, _, rank = part.partition(" ")
                chain.append((tid, rank.strip("()") or "no rank"))
            out[p[i_id]] = chain
    return out


def consensus_lineage(chains, frac=0.9):
    """Taxids present in >= frac of a group's members, kept in depth order.

    A taxonomic group is not one organism, so it has no single lineage. The
    consensus is the part of the tree essentially all its members share.
    """
    if not chains:
        return []
    n = len(chains)
    cnt, rank_of, depth_of = {}, {}, {}
    for ch in chains:
        for d, (tid, rank) in enumerate(ch):
            cnt[tid] = cnt.get(tid, 0) + 1
            rank_of[tid] = rank
            depth_of[tid] = min(depth_of.get(tid, d), d)
    keep = [t for t, c in cnt.items() if c >= frac * n]
    return [(t, rank_of[t]) for t in sorted(keep, key=lambda t: depth_of[t])]


def taxo_pair(cons_s, cons_t):
    ids_s = [t for t, _ in cons_s]
    ids_t = {t for t, _ in cons_t}
    shared = [(t, r) for t, r in cons_s if t in ids_t]
    n_shared = len(shared)
    lca_tid, lca_rank = (shared[-1] if shared else (None, None))
    named = [r for _, r in shared if r not in ("no rank", "clade", "")]
    return dict(n_shared_lineage=n_shared, lca_taxid=lca_tid,
                lca_rank=lca_rank,
                lca_named_rank=(named[-1] if named else None),
                depth_source=len(ids_s), depth_target=len(cons_t))


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--ec", required=True)
    ap.add_argument("--raw", required=True, help="raw/ec_swissprot_raw.tsv, for lineage")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ec_level", type=int, default=3)
    ap.add_argument("--drop_groups", default="other_bacteria,other_eukaryota")
    ap.add_argument("--n_sub", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source_domain", default="",
                    help="if set, only pairs out of this group (cheap mode)")
    ap.add_argument("--skip_frechet", action="store_true",
                    help="the 960x960 matrix sqrt is the slow part")
    a = ap.parse_args()

    T = load_table(a.emb, a.meta, a.ec, a.ec_level)
    T = drop_groups(T, a.drop_groups.split(","))
    X, dom, acc = T["X"], T["dom"], T["acc"]
    groups = sorted(set(dom))
    print(f"{len(X)} proteins, {len(groups)} groups", flush=True)

    rng = np.random.default_rng(a.seed)
    sub = {}
    for g in groups:
        idx = np.where(dom == g)[0]
        if len(idx) > a.n_sub:
            idx = rng.permutation(idx)[:a.n_sub]
        sub[g] = X[idx].astype(np.float64)
    print("subsample sizes: " + ", ".join(f"{g}={len(v)}" for g, v in sub.items()),
          flush=True)

    print("parsing lineages ...", flush=True)
    lin = parse_lineages(a.raw, set(acc.tolist()))
    cons = {}
    for g in groups:
        ids = acc[dom == g]
        cons[g] = consensus_lineage([lin[i] for i in ids if i in lin])
        print(f"  {g:24s} consensus depth {len(cons[g])} "
              f"(deepest: {cons[g][-1] if cons[g] else None})", flush=True)

    # source-domain EC gap, so the mean shift can be reported in the usual units
    ec = T["ec"]
    gaps = {}
    for g in groups:
        m = dom == g
        keys = [e for e in sorted(set(ec)) if (m & (ec == e)).sum() >= 30]
        if len(keys) < 3:
            gaps[g] = float("nan")
            continue
        C = np.stack([X[m & (ec == e)].mean(0) for e in keys])
        iu = np.triu_indices(len(C), 1)
        gaps[g] = float(np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))[iu].mean())

    pairs = ([(a.source_domain, t) for t in groups if t != a.source_domain]
             if a.source_domain else list(itertools.permutations(groups, 2)))
    rows = []
    for s, t in pairs:
        A, B = sub[s], sub[t]
        mmd, gamma = mmd_rbf(A, B)
        ed = energy_distance(A, B)
        pad, err = proxy_a_distance(A, B, a.seed)
        fw = feat_wasserstein(A, B)
        ms = float(np.linalg.norm(B.mean(0) - A.mean(0)))
        r = dict(source=s, target=t, n_sub_source=len(A), n_sub_target=len(B),
                 mmd_rbf=mmd, mmd_gamma=gamma, energy_dist=ed,
                 proxy_a_dist=pad, disc_err=err, feat_wasserstein=fw,
                 mean_shift_norm=ms, mean_shift_over_gap=ms / gaps[s]
                 if gaps[s] == gaps[s] else None,
                 source_ec_gap=gaps[s], target_ec_gap=gaps[t])
        if not a.skip_frechet:
            r["frechet"] = frechet(A, B)
        r.update(taxo_pair(cons[s], cons[t]))
        rows.append(r)
        print(f"  {s:22s} -> {t:22s} mmd={mmd:.5f} energy={ed:.3f} "
              f"pad={pad:.3f} shared_lineage={r['n_shared_lineage']} "
              f"lca={r['lca_named_rank']}", flush=True)

    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "rows": rows,
                   "consensus_lineage": {g: cons[g] for g in groups},
                   "ec_gaps": gaps}, f, indent=2)
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
