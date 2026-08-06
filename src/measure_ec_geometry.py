#!/usr/bin/env python
"""Measure real ESM-C shift geometry in a FUNCTIONAL label space (EC number).

Why
---
Every earlier measurement conditioned on Pfam FAMILY. Families are defined by
homology, so "family" and "sequence similarity" are nearly the same axis, and a
family-conditioned taxonomic shift cannot separate two very different claims:

    (a) going bacteria -> plants moves proteins somewhere else in embedding space
    (b) going bacteria -> plants changes what the embedding says about FUNCTION

EC number is a second label axis over the same proteins, defined by the reaction
catalysed rather than by homology. The same EC occurs in bacteria and in plants,
and usually in more than one Pfam family. That gives three shift types that can
be compared directly, because all three are measured in the same units:

    TAX|EC     hold EC fixed, move domain A -> domain B
    TAX|FAM    hold Pfam fixed, move domain A -> domain B   (the existing measurement)
    FAM|EC     hold EC fixed, move Pfam f1 -> f2            (same function, different scaffold)

The synthetic generator (generate_synthetic_v2) claims a decomposition into a
shared translation (alpha) that is partly function-damaging (beta). Those knobs
were previously calibrated against TAX|FAM alone. Here they get measured against
a functional label space, and the three shift types get compared to each other.

Inputs
------
  --emb    (N, D) .npy embedding cache, row i == line i+1 of metadata.tsv
  --meta   metadata.tsv  (id, family, group)
  --ec     ec_annotations.tsv  (id, ec, ...)   from fetch_ec_annotations.py

Output
------
  stdout report + --out JSON consumed by the figure scripts.
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np


# ---------------------------------------------------------------- data loading
def load_table(emb_path, meta_path, ec_path, ec_level):
    """Join embeddings + Pfam/domain + EC into one aligned table.

    Proteins with no EC, or with several ECs (moonlighting / multi-domain), are
    dropped: a shift vector conditioned on "EC" is only meaningful if the protein
    has exactly one. Partial ECs ("2.7.-.-") are dropped at the requested level.
    """
    X = np.load(emb_path)
    accs, fams, doms = [], [], []
    with open(meta_path) as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            accs.append(p[0]); fams.append(p[1]); doms.append(p[2])
    if len(accs) != len(X):
        raise SystemExit(f"metadata has {len(accs)} rows but embeddings have {len(X)}")

    ec_of = {}
    n_multi = n_empty = 0
    with open(ec_path) as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            acc, ec = p[0], p[1].strip()
            if not ec:
                n_empty += 1
                continue
            if ";" in ec:
                n_multi += 1
                continue
            ec_of[acc] = ec

    keep, ec_lab = [], []
    for i, a in enumerate(accs):
        ec = ec_of.get(a)
        if ec is None:
            continue
        parts = ec.split(".")
        if len(parts) < ec_level or any(p == "-" for p in parts[:ec_level]):
            continue  # partial EC at the level we are conditioning on
        keep.append(i)
        ec_lab.append(".".join(parts[:ec_level]))

    keep = np.array(keep, dtype=int)
    return dict(
        X=X[keep],
        acc=np.array(accs)[keep],
        fam=np.array(fams)[keep],
        dom=np.array(doms)[keep],
        ec=np.array(ec_lab),
        n_total=len(accs), n_multi=n_multi, n_empty=n_empty,
    )


def drop_groups(T, names):
    """Remove taxonomic groups from a loaded table, in place of a reload.

    The EC-first Swiss-Prot build assigns a few catch-all groups ('other_bacteria',
    'other_eukaryota') to proteins whose lineage matches no named clade. Those are
    not coherent taxa -- they are the remainder -- so a shift measured into one of
    them is not a taxonomic shift and they are excluded from shift analyses.
    """
    names = {n.strip() for n in names if n.strip()}
    if not names:
        return T
    keep = ~np.isin(T["dom"], list(names))
    out = dict(T)
    for k in ("X", "acc", "fam", "dom", "ec"):
        out[k] = T[k][keep]
    return out


# ------------------------------------------------------------------- geometry
def centroid(X, mask):
    return X[mask].mean(0)


def pairwise_cos(V):
    """Cosines between every pair of shift vectors. This is the 'taxonomy axis'
    statistic that alpha is calibrated against (E[cos] = alpha by construction)."""
    if len(V) < 2:
        return np.array([])
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    return (Vn @ Vn.T)[np.triu_indices(len(V), 1)]


def shift_stats(V):
    """Summarise a set of shift vectors the way alpha/beta are defined.

    shared_frac = ||mean(v)||^2 / mean(||v||^2) is the fraction of squared
    displacement carried by a common translation -- the direct analogue of the
    generator's alpha. mean pairwise cosine is the same quantity estimated
    pairwise, and is what the histogram figure plots.
    """
    if len(V) < 2:
        return None
    mag = np.linalg.norm(V, axis=1)
    cos = pairwise_cos(V)
    vbar = V.mean(0)
    return dict(
        n=int(len(V)),
        mag_mean=float(mag.mean()), mag_sd=float(mag.std()),
        mag_cv=float(mag.std() / max(mag.mean(), 1e-12)),
        mean_cos=float(cos.mean()), sd_cos=float(cos.std()),
        # mean |cos| is the sign-free version. It is the one to quote for FAM|EC,
        # where the direction f1->f2 is fixed by alphabetical Pfam order and is
        # therefore an arbitrary convention -- flipping it would flip mean_cos.
        mean_abs_cos=float(np.abs(cos).mean()),
        shared_frac=float(np.linalg.norm(vbar) ** 2 / (mag ** 2).mean()),
        frac_cos_negative=float((cos < 0).mean()),
    )


def axis_alignment(V, u):
    """Sign-free energy fraction of each shift vector along unit axis `u`.

    mean of cos^2, so it does not depend on which end of a shift is called the
    start. 1.0 = the shift lies entirely along the axis, 0 = orthogonal to it.
    """
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    return (Vn @ u) ** 2


def conditioned_shift(T, cond_key, shift_key, a, b, min_n):
    """Shift vectors holding `cond_key` fixed while `shift_key` moves a -> b.

    Returns (V, keys, cell_counts). Only levels of the conditioning variable that
    have >= min_n proteins on BOTH sides contribute, because a centroid estimated
    from a handful of points is mostly noise -- and centroid noise inflates the
    measured shift magnitude and deflates the measured cosine.
    """
    cond, shift = T[cond_key], T[shift_key]
    V, keys, counts = [], [], []
    for c in sorted(set(cond)):
        ma = (cond == c) & (shift == a)
        mb = (cond == c) & (shift == b)
        na, nb = int(ma.sum()), int(mb.sum())
        if na < min_n or nb < min_n:
            continue
        V.append(centroid(T["X"], mb) - centroid(T["X"], ma))
        keys.append(c)
        counts.append((na, nb))
    return (np.stack(V) if V else np.zeros((0, T["X"].shape[1]))), keys, counts


def split_half_shift(T, cond_key, shift_key, a, b, min_n, rng, n_rep=20):
    """Reliability of a single shift vector, for correcting the cosine statistic.

    A centroid estimated from finitely many proteins is the true centroid plus
    independent noise, so every measured shift vector is signal + noise. That
    noise is independent between two different ECs, so it contributes nothing to
    their cosine's numerator but inflates both magnitudes -- it DEFLATES the
    measured mean cosine, biasing alpha downward. (JULY22 open item 1 guessed the
    bias ran the other way; this measures it instead of guessing.)

    Estimate it by splitting each cell in half and computing the same shift twice
    from disjoint proteins: cos(v_half1, v_half2) is the fraction of a single
    shift vector that is reproducible signal. The corrected shared-direction
    estimate is then roughly mean_cos_observed / reliability.
    """
    cond, shift = T[cond_key], T[shift_key]
    X = T["X"]
    rels = []
    for _ in range(n_rep):
        for c in sorted(set(cond)):
            ia = np.where((cond == c) & (shift == a))[0]
            ib = np.where((cond == c) & (shift == b))[0]
            if len(ia) < 2 * min_n or len(ib) < 2 * min_n:
                continue
            ia, ib = rng.permutation(ia), rng.permutation(ib)
            ha, hb = np.array_split(ia, 2), np.array_split(ib, 2)
            v1 = X[hb[0]].mean(0) - X[ha[0]].mean(0)
            v2 = X[hb[1]].mean(0) - X[ha[1]].mean(0)
            d = np.linalg.norm(v1) * np.linalg.norm(v2)
            if d > 1e-12:
                rels.append(float(v1 @ v2 / d))
    return float(np.mean(rels)) if rels else float("nan")


# --------------------------------------------------------- function subspace
def function_subspace(T, domain, var_keep=0.90):
    """Directions along which EC identity is expressed, fit on ONE domain.

    Built from the EC centroids of a single domain (mean-centred, SVD), so it is
    the subspace that separates functions WITHIN a taxonomic domain. Fitting on
    the source domain only keeps it independent of the shift we then project
    onto it.
    """
    X, ec, dom = T["X"], T["ec"], T["dom"]
    m = dom == domain
    keys = sorted(set(ec[m]))
    C = np.stack([X[m & (ec == k)].mean(0) for k in keys
                  if (m & (ec == k)).sum() >= 5])
    C = C - C.mean(0)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    frac = np.cumsum(S ** 2) / (S ** 2).sum()
    k = int(np.searchsorted(frac, var_keep) + 1)
    return Vt[:k], k, len(C)


def beta_of(V, B):
    """Fraction of squared displacement lying in the function subspace B.

    This is the generator's beta measured on real data: beta=0 means the shift is
    orthogonal to everything that encodes function (pure covariate shift, function
    readout survives), beta=1 means the shift moves entirely along the directions
    that distinguish one EC from another (maximally function-damaging).
    """
    if len(V) == 0:
        return None
    proj = V @ B.T
    num = (proj ** 2).sum(1)
    den = (V ** 2).sum(1)
    return (num / np.maximum(den, 1e-12))


def beta_null(T, B, rng, n=2000):
    """Empirical null for beta.

    Comparing beta to k/D (the isotropic expectation) is wrong here: ESM-C
    embeddings have effective rank 3-11 out of 960, so a 'random' direction in
    the DATA distribution is nothing like a random direction in R^960. The honest
    null is the beta of difference vectors between randomly paired proteins,
    which carries the same anisotropy but no systematic shift.
    """
    X = T["X"]
    i = rng.integers(0, len(X), n)
    j = rng.integers(0, len(X), n)
    V = X[i] - X[j]
    return beta_of(V, B)


# -------------------------------------------------------------------- report
def fmt(s):
    if s is None:
        return "  (too few cells)"
    return (f"  n={s['n']:3d} cells | |v| {s['mag_mean']:7.3f} (cv {s['mag_cv']:.2f}) | "
            f"mean cos {s['mean_cos']:+.3f} (sd {s['sd_cos']:.3f}) | "
            f"shared_frac {s['shared_frac']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--ec", required=True)
    ap.add_argument("--ec_level", type=int, default=3,
                    help="EC digits to condition on. 3 = sub-subclass (e.g. 2.7.11), "
                         "the level where there are enough proteins per cell; "
                         "4 = the exact reaction.")
    ap.add_argument("--source_domain", default="bacteria")
    ap.add_argument("--target_domains", default="archaea,fungi,metazoa,plants")
    ap.add_argument("--min_n", type=int, default=15,
                    help="min proteins per (condition, side) cell for a centroid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write measurements as JSON")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    T = load_table(args.emb, args.meta, args.ec, args.ec_level)
    X = T["X"]
    print(f"Loaded {T['n_total']} proteins; {len(X)} kept with a single unambiguous "
          f"EC at level {args.ec_level}")
    print(f"  dropped: {T['n_empty']} no EC, {T['n_multi']} multi-EC, "
          f"rest partial at this level")
    print(f"  {len(set(T['ec']))} distinct EC groups, {len(set(T['fam']))} Pfam families, "
          f"{len(set(T['dom']))} domains")

    # --- how the two label spaces relate (this is the oracle's calibration target)
    print("\n=== EC x Pfam cross-tabulation (the 'label space' structure) ===")
    fam_of_ec = defaultdict(set); ec_of_fam = defaultdict(set)
    for e, f in zip(T["ec"], T["fam"]):
        fam_of_ec[e].add(f); ec_of_fam[f].add(e)
    # purity: within a family, the share taken by its most common EC
    purity = []
    for f in ec_of_fam:
        cnt = Counter(T["ec"][T["fam"] == f])
        purity.append(max(cnt.values()) / sum(cnt.values()))
    promiscuous = np.mean([len(v) > 1 for v in ec_of_fam.values()])
    coverage = np.mean([len(v) for v in fam_of_ec.values()])
    print(f"  within-family EC purity : {100*np.mean(purity):.1f}%  "
          f"(one Pfam -> how single-function is it)")
    print(f"  family promiscuity      : {100*promiscuous:.1f}%  "
          f"(share of Pfams hosting >1 EC)")
    print(f"  class coverage          : {coverage:.2f} families per EC")
    print(f"  ECs spanning >=2 Pfams  : {sum(len(v) >= 2 for v in fam_of_ec.values())}"
          f" of {len(fam_of_ec)}")

    # --- function subspace, fit on the source domain only
    B, k, n_cent = function_subspace(T, args.source_domain)
    print(f"\nFunction subspace: {k} directions (90% of between-EC variance over "
          f"{n_cent} EC centroids in {args.source_domain}), ambient dim {X.shape[1]}")
    bnull = beta_null(T, B, rng)
    print(f"  empirical null beta (random protein-pair differences): "
          f"{bnull.mean():.3f} +- {bnull.std():.3f}")

    # --- reference scales, measured in the source domain
    src = T["dom"] == args.source_domain
    ec_keys = [e for e in sorted(set(T["ec"])) if (src & (T["ec"] == e)).sum() >= args.min_n]
    Cec = np.stack([centroid(X, src & (T["ec"] == e)) for e in ec_keys])
    iu = np.triu_indices(len(Cec), 1)
    ec_gap = float(np.sqrt(((Cec[:, None] - Cec[None]) ** 2).sum(-1))[iu].mean())
    wsig = float(np.mean([np.sqrt(((X[src & (T["ec"] == e)] -
                                    centroid(X, src & (T["ec"] == e))) ** 2).sum(1).mean())
                          for e in ec_keys]))
    print(f"\nReference scales in {args.source_domain}: mean inter-EC gap {ec_gap:.3f}, "
          f"within-EC sigma {wsig:.3f}  (separation ratio {ec_gap/wsig:.2f})")

    results = {
        "ec_level": args.ec_level, "source_domain": args.source_domain,
        "min_n": args.min_n, "n_proteins": int(len(X)),
        "n_ec": len(set(T["ec"])), "n_fam": len(set(T["fam"])),
        "label_space": {"purity": float(np.mean(purity)),
                        "promiscuity": float(promiscuous),
                        "coverage": float(coverage)},
        "function_subspace_dim": k, "ambient_dim": int(X.shape[1]),
        "beta_null_mean": float(bnull.mean()), "beta_null_sd": float(bnull.std()),
        "ec_gap": ec_gap, "within_ec_sigma": wsig,
        "shifts": {},
    }

    def record(name, V, keys):
        s = shift_stats(V)
        entry = {"stats": s, "keys": list(keys)}
        if s is not None:
            bet = beta_of(V, B)
            entry["beta_mean"] = float(bet.mean())
            entry["beta_sd"] = float(bet.std())
            entry["mag_over_ec_gap"] = s["mag_mean"] / ec_gap
            entry["mag_over_within_sigma"] = s["mag_mean"] / wsig
            entry["mean_vector"] = V.mean(0).tolist()
        results["shifts"][name] = entry
        return s

    tgts = [d.strip() for d in args.target_domains.split(",")]

    # ---- 1. taxonomic shift, holding EC fixed  (the new measurement)
    print(f"\n=== TAX|EC : same EC, {args.source_domain} -> X ===")
    for t in tgts:
        V, keys, _ = conditioned_shift(T, "ec", "dom", args.source_domain, t, args.min_n)
        s = record(f"tax|ec:{args.source_domain}->{t}", V, keys)
        rel = split_half_shift(T, "ec", "dom", args.source_domain, t, args.min_n, rng)
        print(f"{t:8s}"); print(fmt(s))
        if s is not None:
            e = results["shifts"][f"tax|ec:{args.source_domain}->{t}"]
            e["reliability"] = rel
            e["mean_cos_corrected"] = s["mean_cos"] / rel if rel == rel and rel > 0 else None
            print(f"          |v|/EC-gap {e['mag_over_ec_gap']:.2f} | "
                  f"beta {e['beta_mean']:.3f} (null {bnull.mean():.3f}) | "
                  f"reliability {rel:+.2f} -> cos corrected "
                  f"{e['mean_cos_corrected'] if e['mean_cos_corrected'] is None else round(e['mean_cos_corrected'],3)}")

    # ---- 2. taxonomic shift, holding Pfam fixed  (the existing measurement, same units)
    print(f"\n=== TAX|FAM : same Pfam, {args.source_domain} -> X  (for comparison) ===")
    for t in tgts:
        V, keys, _ = conditioned_shift(T, "fam", "dom", args.source_domain, t, args.min_n)
        s = record(f"tax|fam:{args.source_domain}->{t}", V, keys)
        rel = split_half_shift(T, "fam", "dom", args.source_domain, t, args.min_n, rng)
        print(f"{t:8s}"); print(fmt(s))
        if s is not None:
            e = results["shifts"][f"tax|fam:{args.source_domain}->{t}"]
            e["reliability"] = rel
            e["mean_cos_corrected"] = s["mean_cos"] / rel if rel == rel and rel > 0 else None
            print(f"          |v|/EC-gap {e['mag_over_ec_gap']:.2f} | "
                  f"beta {e['beta_mean']:.3f} | reliability {rel:+.2f}")

    # ---- 3. family shift, holding EC fixed: same function, different scaffold
    # Pooled over every domain, not just the source one: a (EC, Pfam) cell is only
    # comparable within a single domain (otherwise the taxonomy shift contaminates
    # it), but each domain contributes its own cells, which is the only way to get
    # more than a handful of same-function-different-scaffold pairs.
    print("\n=== FAM|EC : same EC, Pfam f1 -> f2 (within one domain, pooled) ===")
    Vf, kf = [], []
    for dm in sorted(set(T["dom"])):
        md = T["dom"] == dm
        for e in sorted(set(T["ec"])):
            me = T["ec"] == e
            fams_here = [f for f in sorted(set(T["fam"][me & md]))
                         if (me & (T["fam"] == f) & md).sum() >= args.min_n]
            for i in range(len(fams_here)):
                for j in range(i + 1, len(fams_here)):
                    f1, f2 = fams_here[i], fams_here[j]
                    Vf.append(centroid(X, me & (T["fam"] == f2) & md) -
                              centroid(X, me & (T["fam"] == f1) & md))
                    kf.append(f"{dm}/{e}:{f1}->{f2}")
    Vf = np.stack(Vf) if Vf else np.zeros((0, X.shape[1]))
    s = record("fam|ec", Vf, kf)
    print(fmt(s))
    if s is not None:
        e = results["shifts"]["fam|ec"]
        print(f"          |v|/EC-gap {e['mag_over_ec_gap']:.2f} | beta {e['beta_mean']:.3f}")
        for key in kf[:8]:
            print(f"            {key}")

    # ---- 3b. CONTROL: does the taxonomic shift vary with function or with homology?
    #
    # TAX|EC and TAX|FAM having near-identical mean vectors proves little on its own:
    # both averages are dominated by the global domain-mean difference, so they would
    # agree even if the conditioning were irrelevant. The real question is about the
    # RESIDUAL, after the shared translation is removed -- when a family's shift
    # deviates from the common axis, is that deviation predicted by its function or
    # by its homology?
    #
    # Build shift vectors at the (EC, Pfam) cell level, subtract the global mean
    # shift, and ask how much of the residual variance each factor explains. The
    # design is unbalanced and the two factors are correlated (a Pfam is ~69% one
    # EC here), so these eta^2 values are marginal, not additive -- they are read as
    # "which label tracks the residual better", not as a variance budget.
    print("\n=== CONTROL: residual shift variance explained by EC vs by Pfam ===")
    results["residual_anova"] = {}
    for t in tgts:
        cells, cv = [], []
        for e in sorted(set(T["ec"])):
            for f in sorted(set(T["fam"])):
                m = (T["ec"] == e) & (T["fam"] == f)
                ma, mb = m & (T["dom"] == args.source_domain), m & (T["dom"] == t)
                if ma.sum() >= args.min_n and mb.sum() >= args.min_n:
                    cv.append(centroid(X, mb) - centroid(X, ma))
                    cells.append((e, f))
        if len(cv) < 6:
            print(f"{t:8s}  only {len(cv)} (EC,Pfam) cells -- skipped")
            continue
        V = np.stack(cv)
        R = V - V.mean(0)                       # remove the shared translation
        ss_tot = float((R ** 2).sum())

        def eta2(labels):
            """Fraction of residual variance lying between groups of `labels`."""
            ss_b = 0.0
            for g in set(labels):
                idx = [i for i, l in enumerate(labels) if l == g]
                ss_b += len(idx) * float((R[idx].mean(0) ** 2).sum())
            return ss_b / max(ss_tot, 1e-12)

        # With ~17 cells spread over ~12 groups most groups hold a single cell, so
        # raw eta^2 is close to 1 whatever the labels mean. Only the excess over a
        # label-permuted null carries information, and with this few cells that
        # excess is usually not resolvable -- which is itself the honest answer.
        def excess(labels, n_perm=500):
            obs = eta2(labels)
            lab = list(labels)
            null = [eta2(list(rng.permutation(lab))) for _ in range(n_perm)]
            null = np.array(null)
            p = float((null >= obs).mean())
            return obs, float(null.mean()), float(null.std()), p

        ec_lab = [c[0] for c in cells]; fam_lab = [c[1] for c in cells]
        o_e, m_e, s_e, p_e = excess(ec_lab)
        o_f, m_f, s_f, p_f = excess(fam_lab)
        n_ec = len(set(ec_lab)); n_fam = len(set(fam_lab))
        print(f"{t:8s}  {len(cv):3d} cells ({n_ec} ECs x {n_fam} Pfams)")
        print(f"          EC   eta^2 {o_e:.3f}  vs permuted null {m_e:.3f}+-{s_e:.3f}  p={p_e:.3f}")
        print(f"          Pfam eta^2 {o_f:.3f}  vs permuted null {m_f:.3f}+-{s_f:.3f}  p={p_f:.3f}")
        results["residual_anova"][t] = dict(
            n_cells=len(cv), n_ec=n_ec, n_fam=n_fam,
            eta2_ec=o_e, null_ec=m_e, p_ec=p_e,
            eta2_fam=o_f, null_fam=m_f, p_fam=p_f)

    # ---- 4. are these axes the same axis? cosine between mean shift vectors
    print("\n=== Do the shift types share a direction? cos(mean shift, mean shift) ===")
    names = [n for n, e in results["shifts"].items() if e.get("mean_vector")]
    M = np.stack([np.array(results["shifts"][n]["mean_vector"]) for n in names])
    Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    G = Mn @ Mn.T
    w = max(len(n) for n in names) + 1
    print(" " * w + "".join(f"{i:>7d}" for i in range(len(names))))
    for i, n in enumerate(names):
        print(f"{n:<{w}}" + "".join(f"{G[i,j]:+7.2f}" for j in range(len(names))))
    print("\n  index: " + ", ".join(f"{i}={n}" for i, n in enumerate(names)))
    print("  NOTE: the fam|ec row/column is sign-convention dependent (f1->f2 is "
          "alphabetical);\n        read the sign-free table below for that comparison.")
    results["axis_cosine_matrix"] = {"names": names, "matrix": G.tolist()}

    # ---- 4b. sign-free: how much of each shift lies along THE taxonomy axis?
    #
    # The taxonomy axis is defined once, as the unit mean bacteria->plants shift
    # measured with Pfam conditioning (the pre-existing definition). Alignment is
    # mean cos^2, which is invariant to which end of a shift is called the start --
    # the signed cosine above is not, and for FAM|EC it is meaningless.
    #
    # The null is the alignment of difference vectors between randomly paired
    # proteins: in a space with effective rank ~3-11 out of 960, "orthogonal to the
    # taxonomy axis" does NOT mean cos^2 ~ 1/960.
    print("\n=== Sign-free alignment with the taxonomy axis (mean cos^2) ===")
    u_tax = np.array(results["shifts"][f"tax|fam:{args.source_domain}->plants"]["mean_vector"])
    u_tax = u_tax / np.linalg.norm(u_tax)
    i = rng.integers(0, len(X), 3000); j = rng.integers(0, len(X), 3000)
    null_al = axis_alignment(X[i] - X[j], u_tax)
    print(f"  null (random protein pairs): {null_al.mean():.3f} +- {null_al.std():.3f}")
    results["axis_alignment"] = {"null_mean": float(null_al.mean()),
                                 "null_sd": float(null_al.std())}
    for n in names:
        V = None
        if n.startswith("tax|ec:"):
            V, _, _ = conditioned_shift(T, "ec", "dom", args.source_domain,
                                        n.split("->")[1], args.min_n)
        elif n.startswith("tax|fam:"):
            V, _, _ = conditioned_shift(T, "fam", "dom", args.source_domain,
                                        n.split("->")[1], args.min_n)
        elif n == "fam|ec":
            V = Vf
        if V is None or len(V) == 0:
            continue
        al = axis_alignment(V, u_tax)
        print(f"  {n:<28s} {al.mean():.3f} +- {al.std():.3f}")
        results["axis_alignment"][n] = {"mean": float(al.mean()), "sd": float(al.std())}

    # keep the raw cosine samples the histogram figure needs
    results["cos_samples"] = {}
    for n in names:
        V = None
        if n.startswith("tax|ec:"):
            t = n.split("->")[1]
            V, _, _ = conditioned_shift(T, "ec", "dom", args.source_domain, t, args.min_n)
        elif n.startswith("tax|fam:"):
            t = n.split("->")[1]
            V, _, _ = conditioned_shift(T, "fam", "dom", args.source_domain, t, args.min_n)
        elif n == "fam|ec":
            V = Vf
        if V is not None and len(V) >= 2:
            results["cos_samples"][n] = pairwise_cos(V).tolist()
            results["shifts"][n]["beta_samples"] = beta_of(V, B).tolist()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        # mean_vector is 960 floats x ~10 entries; keep it, the figures need it
        with open(args.out, "w") as f:
            json.dump(results, f)
        print(f"\nWrote {args.out}")
