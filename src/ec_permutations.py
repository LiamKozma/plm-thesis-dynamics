#!/usr/bin/env python
"""Every conditioned shift, not just the taxonomic ones -- plus the structure tests.

Motivation
----------
The Aug 4 analysis measured three shift types (TAX|EC, TAX|FAM, FAM|EC) and two of
the three move a taxonomic domain. That over-samples taxonomic shift and leaves the
FUNCTIONAL shift almost unmeasured: nothing so far moves EC e1 -> EC e2 while
holding taxonomy and/or homology fixed. The alpha ~ 0.08 that got read as "function
shift is unstructured" is in fact FAM|EC -- same function, DIFFERENT SCAFFOLD -- a
homology move, not a functional one.

This script measures the full cross-product. For a label triple (domain, Pfam, EC),
pick one axis to MOVE and any subset of the other two to HOLD:

    move    hold            name         question it answers
    ----    ----            ----         -------------------
    dom     ec              TAX|EC       taxonomic shift, function fixed
    dom     fam             TAX|FAM      taxonomic shift, homology fixed
    dom     ec,fam          TAX|EC,FAM   taxonomic shift, both fixed (cleanest)
    fam     ec              FAM|EC       same function, different scaffold
    fam     ec,dom          FAM|EC,DOM   ... within one domain
    ec      dom             EC|DOM       FUNCTIONAL shift, taxonomy fixed   <- new
    ec      fam             EC|FAM       FUNCTIONAL shift, homology fixed   <- new
    ec      dom,fam         EC|DOM,FAM   FUNCTIONAL shift, both fixed       <- new

alpha means the same thing in every row: the extent to which the SAME labelled move
produces the SAME displacement vector when carried out in different contexts. For
TAX|EC that is "does bacteria->plants look the same for every EC". For EC|DOM it is
"does e1->e2 look the same in every domain" -- i.e. whether functional offsets are
transferable constants, which is exactly what the generator assumes when it places
class centroids once and then translates whole domains.

Sign note: for EC|* and FAM|* the ordered pair (a -> b) is fixed by sort order, which
is arbitrary -- but the SAME order is used in every context being compared, so
cosines BETWEEN contexts are sign-meaningful. Only cosines between DIFFERENT label
pairs would not be, and those are never taken here.

Also computed
-------------
  * gap sensitivity   -- every plausible definition of the |v| normaliser (Q: what
                         exactly is "the typical distance between two EC groups"?)
  * additivity        -- how much of the (domain x EC) centroid grid is captured by
                         an additive  mu + a_domain + b_EC  model. The generator IS
                         additive by construction, so the interaction term is the
                         part of reality it cannot express.
  * EC hierarchy      -- does embedding distance track distance in the EC tree? Real
                         labels are hierarchical; a random oracle's are not.
"""
import argparse
import itertools
import json
import os
from collections import defaultdict

import numpy as np

from measure_ec_geometry import (load_table, centroid, pairwise_cos, shift_stats,
                                 drop_groups)


# ----------------------------------------------------------------- cell helpers
def cell_key(T, keys, i):
    return tuple(T[k][i] for k in keys)


def cell_index(T, keys, min_n):
    """Map every combination of `keys` values to its member row indices (>= min_n)."""
    buckets = defaultdict(list)
    n = len(T["X"])
    arrs = [T[k] for k in keys]
    for i in range(n):
        buckets[tuple(a[i] for a in arrs)].append(i)
    return {k: np.array(v) for k, v in buckets.items() if len(v) >= min_n}


def shift_instances(T, move_key, hold_keys, min_n, max_pairs=None, rng=None):
    """All (a -> b) moves of `move_key`, each measured in every `hold_keys` context.

    Returns a list of instances; one instance is a single labelled move (e.g.
    EC 2.7.11 -> EC 3.1.3) together with the >= 2 displacement vectors measured for
    it in different contexts (e.g. in bacteria, in fungi, in plants). alpha is a
    within-instance statistic, so instances with only one context are dropped.
    """
    keys = list(hold_keys) + [move_key]
    cells = cell_index(T, keys, min_n)
    X = T["X"]

    # centroid per cell, and an index  context -> {move_value: centroid}
    by_ctx = defaultdict(dict)
    for k, idx in cells.items():
        ctx, mv = k[:-1], k[-1]
        by_ctx[ctx][mv] = X[idx].mean(0)

    # which (a, b) moves occur in >= 2 contexts?
    move_ctxs = defaultdict(list)
    for ctx, d in by_ctx.items():
        vals = sorted(d)
        for a, b in itertools.combinations(vals, 2):   # a < b, same order everywhere
            move_ctxs[(a, b)].append(ctx)
    pairs = [p for p, c in move_ctxs.items() if len(c) >= 2]
    pairs.sort()
    if max_pairs and len(pairs) > max_pairs:
        # subsample deterministically so a huge EC x EC grid stays tractable
        sel = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in sorted(sel)]

    out = []
    for (a, b) in pairs:
        ctxs = sorted(move_ctxs[(a, b)])
        V = np.stack([by_ctx[c][b] - by_ctx[c][a] for c in ctxs])
        out.append(dict(move=(a, b), ctxs=ctxs, V=V))
    return out


def cross_move_alpha(insts, rng, n=4000):
    """Control: cosine between vectors of DIFFERENT labelled moves.

    This is the statistic that decides whether alpha means anything. alpha_within
    asks "does bacteria->plants look the same for EC a as for EC b" -- the move is
    held fixed and the context varies. If two vectors from two UNRELATED moves are
    already at cosine 0.5, then alpha = 0.5 says nothing about the move; it just
    says the embedding space is low-rank and every displacement points somewhere
    similar. The informative quantity is the GAP between the two.

    This control also settles a comparison the Aug 4 table made by accident. Its
    TAX|EC alpha (+0.59) is a within-move number: the move bacteria->plants is
    fixed and EC varies. Its FAM|EC alpha (+0.08) pooled cosines across DIFFERENT
    family pairs, so it is a cross-move number. The two were never the same
    statistic, which is why scaffold shift looked unstructured next to taxonomic
    shift.
    """
    if len(insts) < 2:
        return None
    flat = [(i, v) for i, it in enumerate(insts) for v in it["V"]]
    if len(flat) < 2:
        return None
    cos = []
    for _ in range(n):
        a, b = rng.integers(0, len(flat), 2)
        if flat[a][0] == flat[b][0]:
            continue                      # same labelled move: that is alpha_within
        va, vb = flat[a][1], flat[b][1]
        d = np.linalg.norm(va) * np.linalg.norm(vb)
        if d > 1e-12:
            cos.append(float(va @ vb / d))
    if not cos:
        return None
    cos = np.array(cos)
    return dict(mean=float(cos.mean()), sd=float(cos.std()),
                mean_abs=float(np.abs(cos).mean()), n=len(cos))


def instance_stats(insts, gap):
    """Pool per-instance alpha / magnitude into one row of the summary table.

    alpha is averaged OVER instances (each instance weighted equally) rather than
    over all cosines pooled, so a single move that happens to occur in many contexts
    cannot dominate. Both are reported.
    """
    if not insts:
        return None
    per_alpha, per_shared, mags, all_cos = [], [], [], []
    for it in insts:
        V = it["V"]
        cos = pairwise_cos(V)
        if len(cos) == 0:
            continue
        per_alpha.append(float(cos.mean()))
        all_cos.extend(cos.tolist())
        m = np.linalg.norm(V, axis=1)
        mags.extend(m.tolist())
        vbar = V.mean(0)
        per_shared.append(float(np.linalg.norm(vbar) ** 2 / (m ** 2).mean()))
    if not per_alpha:
        return None
    per_alpha = np.array(per_alpha); mags = np.array(mags)
    return dict(
        n_instances=len(per_alpha),
        n_vectors=int(sum(len(it["V"]) for it in insts)),
        n_cosines=len(all_cos),
        alpha=float(per_alpha.mean()),           # mean over instances
        alpha_sd_across_instances=float(per_alpha.std()),
        alpha_pooled=float(np.mean(all_cos)),    # mean over all cosines
        alpha_frac_positive=float((np.array(all_cos) > 0).mean()),
        shared_frac=float(np.mean(per_shared)),
        mag_mean=float(mags.mean()),
        mag_over_gap=float(mags.mean() / gap),
    )


def reliability(T, move_key, hold_keys, min_n, rng, n_rep=10, max_inst=200):
    """Split-half cos(v1, v2) for ONE displacement vector of this shift type.

    Every measured centroid is truth + estimation noise, and that noise is
    independent between contexts, so it deflates alpha. Dividing the observed alpha
    by this reliability removes the deflation to first order. Cells need 2*min_n
    members to be splittable, so this is measured on the subset that qualifies.
    """
    keys = list(hold_keys) + [move_key]
    cells = cell_index(T, keys, 2 * min_n)
    X = T["X"]
    by_ctx = defaultdict(dict)
    for k, idx in cells.items():
        by_ctx[k[:-1]][k[-1]] = idx
    # Sample (context, a, b) triples at random rather than walking them in sorted
    # order: on the full Swiss-Prot set there are millions of them, and taking the
    # first few thousand would estimate reliability from whichever labels sort first.
    ctxs = [c for c, d in by_ctx.items() if len(d) >= 2]
    if not ctxs:
        return float("nan")
    rels = []
    budget = max_inst * n_rep
    for _ in range(budget * 4):
        if len(rels) >= budget:
            break
        ctx = ctxs[rng.integers(0, len(ctxs))]
        vals = sorted(by_ctx[ctx])
        i, j = rng.choice(len(vals), 2, replace=False)
        a, b = vals[i], vals[j]
        ia, ib = rng.permutation(by_ctx[ctx][a]), rng.permutation(by_ctx[ctx][b])
        ha, hb = np.array_split(ia, 2), np.array_split(ib, 2)
        v1 = X[hb[0]].mean(0) - X[ha[0]].mean(0)
        v2 = X[hb[1]].mean(0) - X[ha[1]].mean(0)
        den = np.linalg.norm(v1) * np.linalg.norm(v2)
        if den > 1e-12:
            rels.append(float(v1 @ v2 / den))
    return float(np.mean(rels)) if rels else float("nan")


# ------------------------------------------------------------ gap sensitivity
def centroid_dists(C):
    """Upper-triangle pairwise distances via the Gram trick.

    (C[:, None] - C[None]) materialises an (n, n, 960) array. That was fine for the
    18 EC centroids of the taxonomy ladder; on the Swiss-Prot build the Pfam
    centroids number in the thousands and the same expression asks for tens of GB.
    """
    sq = (C ** 2).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (C @ C.T)
    np.maximum(D2, 0, out=D2)
    return np.sqrt(D2)[np.triu_indices(len(C), 1)]


def gap_definitions(T, domain, min_n):
    """Every defensible answer to "what is the typical distance between EC groups?".

    The number the dashboard divides by, `ec_gap`, is ONE of these: the mean over all
    pairwise Euclidean distances between EC-group CENTROIDS, computed inside the
    source domain only, over ECs holding >= min_n proteins there. Alternatives differ
    by a fixed factor, so every published |v|/gap converts between them exactly.
    """
    X, ec, fam, dom = T["X"], T["ec"], T["fam"], T["dom"]
    m = dom == domain
    keys = [e for e in sorted(set(ec)) if (m & (ec == e)).sum() >= min_n]
    C = np.stack([centroid(X, m & (ec == e)) for e in keys])
    D = centroid_dists(C)

    # same thing but pooled over all domains, i.e. EC centroids ignoring taxonomy
    keys_all = [e for e in sorted(set(ec)) if (ec == e).sum() >= min_n]
    Ca = np.stack([centroid(X, ec == e) for e in keys_all])
    Da = centroid_dists(Ca)

    # Pfam centroids, for the homology-side equivalent
    fkeys = [f for f in sorted(set(fam)) if (m & (fam == f)).sum() >= min_n]
    Cf = np.stack([centroid(X, m & (fam == f)) for f in fkeys])
    Df = centroid_dists(Cf)

    # within-EC scatter: RMS distance of a protein from its own EC centroid
    sig = float(np.mean([np.sqrt(((X[m & (ec == e)] - centroid(X, m & (ec == e))) ** 2)
                                 .sum(1).mean()) for e in keys]))
    # distance between two random PROTEINS (not centroids) in the source domain
    rng = np.random.default_rng(0)
    Xs = X[m]
    i, j = rng.integers(0, len(Xs), 5000), rng.integers(0, len(Xs), 5000)
    dpp = float(np.linalg.norm(Xs[i] - Xs[j], axis=1).mean())

    return {
        "ec_gap_centroid_mean_source": float(D.mean()),      # <- the one in use
        "ec_gap_centroid_median_source": float(np.median(D)),
        "ec_gap_centroid_min_source": float(D.min()),
        "ec_gap_centroid_p10_source": float(np.percentile(D, 10)),
        "ec_gap_centroid_mean_alldomains": float(Da.mean()),
        "pfam_gap_centroid_mean_source": float(Df.mean()),
        "within_ec_sigma_source": sig,
        "random_protein_pair_distance_source": dpp,
        "n_ec_centroids": len(C), "n_ec_pairs": int(len(D)),
        "separation_ratio_gap_over_sigma": float(D.mean() / sig),
    }


# --------------------------------------------------------------- additivity
def additivity(T, row_key, col_key, min_n):
    """How much of the (row x col) centroid grid is  mu + a_row + b_col ?

    The generator builds every domain as (shared class geometry) + (one translation
    per domain). That is exactly an additive model, so the interaction term measured
    here is the share of real structure the generator cannot represent no matter how
    its knobs are set. Cells are weighted equally; the design is unbalanced, so the
    additive fit is a least-squares dummy regression rather than a means decomposition.
    """
    cells = cell_index(T, [row_key, col_key], min_n)
    if len(cells) < 6:
        return None
    X = T["X"]
    keys = sorted(cells)
    Y = np.stack([X[cells[k]].mean(0) for k in keys])
    rows = sorted({k[0] for k in keys}); cols = sorted({k[1] for k in keys})
    if len(rows) < 2 or len(cols) < 2:
        return None
    ri = {r: i for i, r in enumerate(rows)}; ci = {c: i for i, c in enumerate(cols)}

    def design(with_row, with_col):
        blocks = [np.ones((len(keys), 1))]
        if with_row:
            M = np.zeros((len(keys), len(rows)))
            for n, k in enumerate(keys):
                M[n, ri[k[0]]] = 1
            blocks.append(M)
        if with_col:
            M = np.zeros((len(keys), len(cols)))
            for n, k in enumerate(keys):
                M[n, ci[k[1]]] = 1
            blocks.append(M)
        return np.concatenate(blocks, 1)

    def ss_resid(A):
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        return float(((Y - A @ coef) ** 2).sum())

    ss_tot = ss_resid(design(False, False))       # about the grand mean
    ss_row = ss_resid(design(True, False))
    ss_col = ss_resid(design(False, True))
    ss_add = ss_resid(design(True, True))
    return dict(
        n_cells=len(keys), n_rows=len(rows), n_cols=len(cols),
        frac_explained_by_row_only=1 - ss_row / ss_tot,
        frac_explained_by_col_only=1 - ss_col / ss_tot,
        frac_explained_by_additive=1 - ss_add / ss_tot,
        frac_interaction_residual=ss_add / ss_tot,
        row_key=row_key, col_key=col_key,
    )


# ------------------------------------------------------------- EC hierarchy
def ec_hierarchy(T, domain, min_n):
    """Does embedding distance between two EC groups track their distance in the EC tree?

    EC is a 4-level tree, so two classes can be siblings (share the first two digits)
    or unrelated (differ at digit 1). If ESM-C respects that hierarchy, sibling
    classes sit closer together. A random oracle label space -- which is what the
    generator currently produces -- has no such structure, so this is a concrete
    property of real functional label spaces that synthetic data can be scored on.
    """
    X, ec, dom = T["X"], T["ec"], T["dom"]
    m = dom == domain
    keys = [e for e in sorted(set(ec)) if (m & (ec == e)).sum() >= min_n]
    if len(keys) < 5:
        return None
    C = np.stack([centroid(X, m & (ec == e)) for e in keys])
    parts = [k.split(".") for k in keys]

    d_emb, d_tree = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            shared = 0
            for lv in range(min(len(parts[i]), len(parts[j]))):
                if parts[i][lv] == parts[j][lv]:
                    shared += 1
                else:
                    break
            d_emb.append(float(np.linalg.norm(C[i] - C[j])))
            d_tree.append(3 - shared)     # 1 = same subclass, 3 = different class
    d_emb, d_tree = np.array(d_emb), np.array(d_tree)

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra, rb = ra - ra.mean(), rb - rb.mean()
        den = np.linalg.norm(ra) * np.linalg.norm(rb)
        return float(ra @ rb / den) if den > 1e-12 else float("nan")

    by_level = {int(t): float(d_emb[d_tree == t].mean())
                for t in sorted(set(d_tree.tolist())) if (d_tree == t).sum() > 0}
    counts = {int(t): int((d_tree == t).sum()) for t in sorted(set(d_tree.tolist()))}
    return dict(n_classes=len(keys), spearman_dist_vs_treedist=spearman(d_emb, d_tree),
                mean_dist_by_tree_distance=by_level, n_pairs_by_tree_distance=counts)


# -------------------------------------------------------------------- driver
SHIFT_TYPES = [
    ("TAX|EC",      "dom", ["ec"]),
    ("TAX|FAM",     "dom", ["fam"]),
    ("TAX|EC,FAM",  "dom", ["ec", "fam"]),
    ("FAM|EC",      "fam", ["ec"]),
    ("FAM|EC,DOM",  "fam", ["ec", "dom"]),
    ("EC|DOM",      "ec",  ["dom"]),
    ("EC|FAM",      "ec",  ["fam"]),
    ("EC|DOM,FAM",  "ec",  ["dom", "fam"]),
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--ec", required=True)
    ap.add_argument("--ec_level", type=int, default=3)
    ap.add_argument("--min_n", type=int, default=15)
    ap.add_argument("--source_domain", default="bacteria")
    ap.add_argument("--max_pairs", type=int, default=4000,
                    help="cap on labelled moves per shift type (EC x EC is quadratic)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop_groups", default="",
                    help="comma list of taxonomic groups to exclude (e.g. the "
                         "'other_*' catch-alls, which are remainders not clades)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    T = load_table(args.emb, args.meta, args.ec, args.ec_level)
    T = drop_groups(T, args.drop_groups.split(","))
    print(f"{len(T['X'])} proteins | {len(set(T['ec']))} ECs | {len(set(T['fam']))} Pfams "
          f"| {len(set(T['dom']))} domains | min_n={args.min_n}\n")

    results = {"min_n": args.min_n, "ec_level": args.ec_level,
               "n_proteins": int(len(T["X"]))}

    # ---- Q1: what exactly is the normaliser?
    print("=== What '|v| / gap' divides by: every candidate definition ===")
    gaps = gap_definitions(T, args.source_domain, args.min_n)
    results["gap_definitions"] = gaps
    base = gaps["ec_gap_centroid_mean_source"]
    for k, v in gaps.items():
        if k.startswith("n_") or k.startswith("separation"):
            continue
        mark = "   <-- the one in use" if k == "ec_gap_centroid_mean_source" else ""
        print(f"  {k:44s} {v:8.3f}   (x{v/base:5.2f} of the one in use){mark}")
    print(f"  built from {gaps['n_ec_centroids']} EC centroids "
          f"= {gaps['n_ec_pairs']} pairwise distances, {args.source_domain} only")
    print(f"  separation ratio gap / within-EC sigma = "
          f"{gaps['separation_ratio_gap_over_sigma']:.2f}")

    # ---- Q4: the full permutation table
    print(f"\n=== Every conditioned shift type "
          f"(alpha = does the same labelled move give the same vector?) ===")
    hdr = (f"{'shift':<13s} {'moves':>6s} {'vecs':>5s} {'cos':>6s} | {'a_within':>8s} "
           f"{'rel':>5s} {'a_corr':>7s} | {'a_cross':>8s} {'excess':>7s} | "
           f"{'shared':>7s} {'|v|/gap':>8s}")
    print(hdr); print("-" * len(hdr))
    results["shift_types"] = {}
    for name, move, hold in SHIFT_TYPES:
        insts = shift_instances(T, move, hold, args.min_n, args.max_pairs, rng)
        st = instance_stats(insts, base)
        if st is None:
            print(f"{name:<13s}   (no labelled move occurs in >= 2 contexts at min_n={args.min_n})")
            results["shift_types"][name] = None
            continue
        rel = reliability(T, move, hold, args.min_n, rng)
        st["reliability"] = rel
        st["alpha_corrected"] = st["alpha"] / rel if rel == rel and rel > 0 else None
        st["alpha_cross_move"] = cross_move_alpha(insts, rng)
        ac = st["alpha_corrected"]
        xm = st["alpha_cross_move"]
        xs = f"{xm['mean']:+8.3f}" if xm else "      --"
        ex = f"{st['alpha'] - xm['mean']:+7.3f}" if xm else "     --"
        print(f"{name:<13s} {st['n_instances']:6d} {st['n_vectors']:5d} {st['n_cosines']:6d} | "
              f"{st['alpha']:+8.3f} {rel:5.2f} "
              f"{(f'{ac:+.3f}' if ac is not None else '   --  '):>7s} | {xs} {ex} | "
              f"{st['shared_frac']:7.3f} {st['mag_over_gap']:8.2f}")
        results["shift_types"][name] = st

    print("\n  moves    = distinct labelled moves (e.g. one EC->EC pair) seen in >=2 contexts")
    print("  a_within = mean cosine between the SAME move measured in different contexts")
    print("  a_cross  = same but for vectors of DIFFERENT moves -- the null. In a")
    print("             low-rank space this is well above zero, so only the EXCESS")
    print("             (a_within - a_cross) says the label of the move matters.")
    print("  rel      = split-half reliability of one vector; a_corr = a_within / rel")

    # ---- Q5a: additivity of the (domain x EC) grid
    print("\n=== Additivity: is the centroid grid  mu + a_row + b_col ? ===")
    print("    (the generator is additive by construction, so the residual is the")
    print("     part of real structure it cannot express at any knob setting)")
    results["additivity"] = {}
    for r, c in [("dom", "ec"), ("dom", "fam"), ("fam", "ec")]:
        a = additivity(T, r, c, args.min_n)
        results["additivity"][f"{r}x{c}"] = a
        if a is None:
            print(f"  {r} x {c}: too few cells")
            continue
        print(f"  {r:4s} x {c:4s}  {a['n_cells']:3d} cells ({a['n_rows']} x {a['n_cols']})  "
              f"{r} alone {a['frac_explained_by_row_only']:.3f} | "
              f"{c} alone {a['frac_explained_by_col_only']:.3f} | "
              f"additive {a['frac_explained_by_additive']:.3f} | "
              f"interaction {a['frac_interaction_residual']:.3f}")

    # ---- Q5b: does the EC tree show up in the geometry?
    print("\n=== EC hierarchy: does embedding distance track distance in the EC tree? ===")
    results["ec_hierarchy"] = {}
    for d in sorted(set(T["dom"])):
        h = ec_hierarchy(T, d, args.min_n)
        results["ec_hierarchy"][d] = h
        if h is None:
            print(f"  {d:9s} too few EC classes")
            continue
        means = ", ".join(f"tree-d {k}: {v:.2f}" for k, v in
                          sorted(h["mean_dist_by_tree_distance"].items()))
        print(f"  {d:9s} {h['n_classes']:3d} classes  Spearman {h['spearman_dist_vs_treedist']:+.3f}"
              f"   ({means})")
    print("  tree-d 1 = same EC subclass (first 2 digits shared), 3 = different EC class")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out}")
