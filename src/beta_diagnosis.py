#!/usr/bin/env python
"""Why does beta have the WRONG SIGN, and what should replace it?

The puzzle
----------
beta is defined as the fraction of a shift's squared length that lies in B, the
subspace spanned by the directions that separate EC groups. By construction those
are "the dimensions that differentiate function", so a shift with high beta ought to
destroy the function readout. Measured over all 20 ordered domain pairs it does the
opposite: Spearman(beta_shared, retained EC F1) = +0.38.

Three candidate explanations, all testable:

  H1  beta is a FRACTION, so it is blind to how far the shift actually travels.
      A shift with beta = 0.9 and length 0.1 gaps moves 0.3 gaps into B; one with
      beta = 0.1 and length 2 gaps moves 0.63 gaps into B and does more damage.
      The damage-relevant quantity is the ABSOLUTE in-subspace displacement,
      sqrt(beta) * |v| / gap -- not beta.

  H2  A SHARED translation inside B moves every EC group together and preserves
      their relative arrangement. What breaks a classifier is DIFFERENTIAL motion:
      groups moving differently from one another. beta measures displacement into
      B; it does not measure distortion OF B. Since beta is computed on the mean
      shift vector, it is measuring precisely the benign component.

  H3  B is fit as the top principal directions of the EC-centroid cloud, which are
      also the highest-variance directions of the data. A shift that stays inside B
      stays on the manifold the probe was trained on; a low-beta shift leaves it,
      and after the probe's StandardScaler divides by a small source-domain sigma,
      off-manifold motion is amplified. Under H3 beta is a proxy for "in
      distribution", which genuinely predicts LESS damage.

Two experiments
---------------
PART A  Controlled injection. Take one domain, train a probe, and hit the test set
        with a SYNTHETIC shift of chosen magnitude and chosen beta. Everything else
        is held fixed, so whatever beta does here it does causally. Shared and
        differential (per-class) modes are both run, which separates H2. The
        off-B direction is drawn two ways -- isotropic, and from the data covariance
        -- which separates H3.

PART B  Predictor battery on the 20 real domain pairs. Eleven candidate geometric
        quantities, each correlated with retained EC F1 under the same domain-level
        permutation null used in ec_allpairs.py. This answers "is magnitude, alpha
        and beta enough, or should we be measuring something else" with data.
"""
import argparse
import itertools
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from measure_ec_geometry import (load_table, conditioned_shift, shift_stats,
                                 function_subspace, beta_of, centroid, drop_groups)
from measure_ec_damage import run_pair


# --------------------------------------------------------------------- stats
def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / d) if d > 1e-12 else float("nan")


def macro_f1(pred, y):
    f1s = []
    for c in np.unique(y):
        tp = float(((pred == c) & (y == c)).sum())
        fp = float(((pred == c) & (y != c)).sum())
        fn = float(((pred != c) & (y == c)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


# ------------------------------------------------------- PART A: injection
def random_unit_in(basis, rng, cov_sqrt=None):
    """Unit vector inside the row space of `basis`.

    cov_sqrt=None samples isotropically WITHIN B, which is not a fair partner for a
    data-weighted off-B direction: B's own directions differ enormously in variance,
    so an isotropic draw over-weights B's weak directions and understates what a
    typical in-B move looks like. Passing the covariance factor draws the in-B
    direction the same way the off-B one is drawn, which is the only version of this
    comparison that isolates "is it in B" from "is it a direction the data uses".
    """
    if cov_sqrt is None:
        c = rng.normal(size=len(basis))
        v = c @ basis
        return v / np.linalg.norm(v)
    for _ in range(100):
        v = cov_sqrt @ rng.normal(size=cov_sqrt.shape[1])
        v = basis.T @ (basis @ v)              # keep only the part inside B
        n = np.linalg.norm(v)
        if n > 1e-8:
            return v / n
    raise RuntimeError("could not draw an in-subspace direction")


def random_unit_off(basis, rng, D, cov_sqrt=None):
    """Unit vector orthogonal to `basis`.

    cov_sqrt=None gives an isotropic direction in R^D -- which, in a space of
    effective rank ~3-11 out of 960, is essentially off-manifold noise. Passing a
    covariance factor instead draws a direction the DATA actually populates, which
    is the honest comparison: it asks whether beta matters once "how typical is this
    direction" is held constant.
    """
    for _ in range(100):
        v = rng.normal(size=D) if cov_sqrt is None else cov_sqrt @ rng.normal(size=D)
        v = v - basis.T @ (basis @ v)          # project out B
        n = np.linalg.norm(v)
        if n > 1e-8:
            return v / n
    raise RuntimeError("could not draw an off-subspace direction")


def part_a(T, domain, min_n, seed, betas, mags, n_rep):
    """Damage as a function of beta with magnitude held fixed, in one domain."""
    rng = np.random.default_rng(seed)
    X, ec, dom = T["X"], T["ec"], T["dom"]
    m = dom == domain
    keys = [e for e in sorted(set(ec)) if (m & (ec == e)).sum() >= 2 * min_n]
    keep = m & np.isin(ec, keys)
    idx = rng.permutation(np.where(keep)[0])
    cut = int(0.7 * len(idx))
    tr, te = idx[:cut], idx[cut:]

    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=3000, random_state=seed)
    clf.fit(sc.transform(X[tr]), ec[tr])
    base_f1 = macro_f1(clf.predict(sc.transform(X[te])), ec[te])

    B, k, _ = function_subspace(T, domain)
    D = X.shape[1]
    # square-root factor of the source covariance, for the on-manifold null direction
    Xc = X[tr] - X[tr].mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    cov_sqrt = (Vt.T * (S / np.sqrt(max(len(Xc) - 1, 1)))) @ Vt

    # the same gap the dashboard normalises by
    C = np.stack([centroid(X, m & (ec == e)) for e in keys])
    iu = np.triu_indices(len(C), 1)
    gap = float(np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))[iu].mean())

    print(f"PART A: injection inside '{domain}'  ({len(tr)} train / {len(te)} test, "
          f"{len(keys)} EC classes)")
    print(f"  function subspace B: {k} dims of {D} | mean inter-EC gap {gap:.3f}")
    print(f"  undisturbed macro-F1 = {base_f1:.3f}\n")

    # "isotropic"    off-B drawn isotropically, in-B drawn isotropically within B
    # "on-manifold"  off-B drawn from the data covariance, in-B isotropic within B
    # "matched"      BOTH drawn from the data covariance -- the fair comparison
    VARIANTS = [("isotropic", None, None),
                ("on-manifold", cov_sqrt, None),
                ("matched", cov_sqrt, cov_sqrt)]
    rows = []
    for mode in ["shared", "differential"]:
        for off, cs, cin in VARIANTS:
            for mag in mags:
                for beta in betas:
                    f1s = []
                    for r in range(n_rep):
                        if mode == "shared":
                            u = (np.sqrt(beta) * random_unit_in(B, rng, cin) +
                                 np.sqrt(1 - beta) * random_unit_off(B, rng, D, cs))
                            delta = mag * gap * u / np.linalg.norm(u)
                            Xte = X[te] + delta
                        else:
                            # every class gets its own direction: same beta, same
                            # length, zero shared component -- this is alpha = 0
                            Xte = X[te].copy()
                            for c in keys:
                                u = (np.sqrt(beta) * random_unit_in(B, rng, cin) +
                                     np.sqrt(1 - beta) * random_unit_off(B, rng, D, cs))
                                d_c = mag * gap * u / np.linalg.norm(u)
                                Xte[ec[te] == c] += d_c
                        f1s.append(macro_f1(clf.predict(sc.transform(Xte)), ec[te]))
                    f1 = float(np.mean(f1s))
                    rows.append(dict(mode=mode, off_direction=off, mag_over_gap=mag,
                                     beta=beta, f1=f1, retained=f1 / base_f1,
                                     f1_sd=float(np.std(f1s))))
    # print as a grid
    for mode in ["shared", "differential"]:
        for off, _, _ in VARIANTS:
            print(f"  retained F1 | mode={mode:<12s} off-B direction={off}")
            print("    " + f"{'|v|/gap':>8s}" + "".join(f"{f'b={b:g}':>9s}" for b in betas))
            for mag in mags:
                sel = [r for r in rows if r["mode"] == mode and r["off_direction"] == off
                       and r["mag_over_gap"] == mag]
                sel = {r["beta"]: r["retained"] for r in sel}
                print("    " + f"{mag:8.2f}" + "".join(f"{sel[b]:9.3f}" for b in betas))
            print()
    return dict(domain=domain, base_f1=base_f1, gap=gap, subspace_dim=k,
                n_classes=len(keys), rows=rows)


# ------------------------------------------------ PART B: predictor battery
def procrustes_disparity(A, Bm):
    """How much the class-centroid CONFIGURATION rearranges, translation+scale free.

    Both centroid clouds are centred and scaled to unit Frobenius norm, then optimally
    rotated onto one another. What is left is pure relative rearrangement: a rigid
    translation of every class (the thing beta is computed on) contributes zero.
    """
    A = A - A.mean(0); Bm = Bm - Bm.mean(0)
    A = A / max(np.linalg.norm(A), 1e-12)
    Bm = Bm / max(np.linalg.norm(Bm), 1e-12)
    U, S, Vt = np.linalg.svd(A.T @ Bm)
    return float(2 - 2 * S.sum())          # 0 = identical configuration


def probe_logit_spread(T, source, target, min_n, seed, max_train=None):
    """First-order damage a translation does to the probe that is actually used.

    A linear probe scores class c as w_c . z + b_c on standardised features. Adding a
    displacement v moves every score by w_c . (v / sigma). If that quantity were the
    same for all c the ranking would be untouched, so the damage is its SPREAD across
    classes. This is the theory-correct predictor: it uses the real decision
    boundaries instead of a proxy subspace.
    """
    X, ec, dom = T["X"], T["ec"], T["dom"]
    keys = [k for k in sorted(set(ec))
            if ((ec == k) & (dom == source)).sum() >= min_n
            and ((ec == k) & (dom == target)).sum() >= min_n]
    if len(keys) < 3:
        return None
    keep = np.isin(ec, keys)
    tr = np.where(keep & (dom == source))[0]
    if max_train and len(tr) > max_train:
        tr = np.random.default_rng(seed).permutation(tr)[:max_train]
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=3000, random_state=seed)
    clf.fit(sc.transform(X[tr]), ec[tr])
    v = X[keep & (dom == target)].mean(0) - X[keep & (dom == source)].mean(0)
    dlogit = clf.coef_ @ (v / sc.scale_)              # (n_classes,)
    # scale by the spread of the source logits, so it is in units of "how far apart
    # the classes already were" rather than raw logit units
    margin = float(clf.decision_function(sc.transform(X[tr])).std())
    return float(dlogit.std() / max(margin, 1e-12))


# Set before the worker pool forks, so children inherit the big arrays copy-on-write
# instead of pickling a few hundred MB per task.
_CTX = {}


def _pair_row(st_pair):
    s, t = st_pair
    T, min_n, seed = _CTX["T"], _CTX["min_n"], _CTX["seed"]
    sub, gaps, sigs = _CTX["sub"], _CTX["gaps"], _CTX["sigs"]
    max_train, max_test = _CTX["max_train"], _CTX["max_test"]
    X, ec, dom = T["X"], T["ec"], T["dom"]

    V, keys, _ = conditioned_shift(T, "ec", "dom", s, t, min_n)
    if len(V) < 3:
        return None
    dmg = run_pair(T, "ec", s, t, min_n, seed, max_train, max_test)
    if dmg is None:
        return None
    st = shift_stats(V)
    Bs, gap = sub[s], gaps[s]
    vbar = V.mean(0)
    b_shared = float(beta_of(vbar[None], Bs)[0])
    b_cell = float(beta_of(V, Bs).mean())
    R = V - vbar                                   # differential component
    mag_g = st["mag_mean"] / gap

    # in-subspace ABSOLUTE displacement of the shared translation (H1)
    inB = float(np.linalg.norm(Bs @ vbar) / gap)
    offB = float(np.sqrt(max(np.linalg.norm(vbar) ** 2 -
                             np.linalg.norm(Bs @ vbar) ** 2, 0)) / gap)
    # differential motion, total and inside B (H2)
    diff_abs = float(np.linalg.norm(R, axis=1).mean() / gap)
    diff_inB = float(np.mean([np.linalg.norm(Bs @ r) for r in R]) / gap)

    # configuration rearrangement of the shared EC centroids
    Cs = np.stack([centroid(X, (dom == s) & (ec == k)) for k in keys])
    Ct = np.stack([centroid(X, (dom == t) & (ec == k)) for k in keys])
    proc = procrustes_disparity(Cs, Ct)

    return dict(
        source=s, target=t, n_cells=len(V), retained=dmg["retained"],
        ceiling_f1=dmg["ceiling_f1"], zeroshot_f1=dmg["zeroshot_f1"],
        n_classes=dmg["n_classes"],
        mag_over_gap=mag_g,
        alpha=st["mean_cos"],
        beta_cell=b_cell, beta_shared=b_shared,
        inB_abs=inB, offB_abs=offB,
        diff_frac=1 - st["shared_frac"],
        diff_abs=diff_abs, diff_inB_abs=diff_inB,
        procrustes=proc,
        logit_spread=probe_logit_spread(T, s, t, min_n, seed, max_train),
        gap_ratio=gaps[t] / gaps[s],
        sep_ratio_target=gaps[t] / sigs[t],
    )


def part_b(T, min_n, seed, n_perm, jobs=1, max_train=None, max_test=None):
    rng = np.random.default_rng(seed)
    X, ec, dom = T["X"], T["ec"], T["dom"]
    domains = sorted(set(dom))

    sub, gaps, sigs = {}, {}, {}
    for d in domains:
        Bm, k, _ = function_subspace(T, d)
        sub[d] = Bm
        m = dom == d
        keys = [e for e in sorted(set(ec)) if (m & (ec == e)).sum() >= min_n]
        C = np.stack([centroid(X, m & (ec == e)) for e in keys])
        iu = np.triu_indices(len(C), 1)
        gaps[d] = float(np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))[iu].mean())
        sigs[d] = float(np.mean([np.sqrt(((X[m & (ec == e)] -
                                          centroid(X, m & (ec == e))) ** 2).sum(1).mean())
                                 for e in keys]))

    _CTX.update(T=T, min_n=min_n, seed=seed, sub=sub, gaps=gaps, sigs=sigs,
                max_train=max_train, max_test=max_test)
    pairs = list(itertools.permutations(domains, 2))
    print(f"  {len(pairs)} ordered group pairs, {jobs} worker(s)", flush=True)
    if jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(jobs) as pool:
            rows = pool.map(_pair_row, pairs, chunksize=1)
    else:
        rows = [_pair_row(p) for p in pairs]
    rows = [r for r in rows if r is not None]

    PREDICTORS = [
        ("|v|/gap                 ", "mag_over_gap",  "current: raw shift length"),
        ("alpha                   ", "alpha",         "current: shared-direction fraction"),
        ("beta (per cell)         ", "beta_cell",     "current"),
        ("beta (shared translation)", "beta_shared",  "current: the generator's definition"),
        ("|P_B v| / gap           ", "inB_abs",       "H1: ABSOLUTE displacement inside B"),
        ("|P_B^perp v| / gap      ", "offB_abs",      "H1/H3: displacement OFF the function subspace"),
        ("differential fraction   ", "diff_frac",     "H2: 1 - shared_frac"),
        ("|v - vbar| / gap        ", "diff_abs",      "H2: absolute differential motion"),
        ("|P_B (v - vbar)| / gap  ", "diff_inB_abs",  "H2: differential motion INSIDE B"),
        ("Procrustes disparity    ", "procrustes",    "rearrangement of the class configuration"),
        ("probe logit spread      ", "logit_spread",  "first-order damage to the actual probe"),
        ("gap ratio target/source ", "gap_ratio",     "asymmetry: does the target spread classes more?"),
    ]

    keep_arr = np.array([r["retained"] for r in rows])
    doms = sorted({r["source"] for r in rows} | {r["target"] for r in rows})
    look = {(r["source"], r["target"]): r for r in rows}

    def perm_p(field, obs):
        """Permute whole DOMAINS, not pairs: the 20 pairs share only 5 domains."""
        cnt = 0
        for _ in range(n_perm):
            mp = {d: e for d, e in zip(doms, rng.permutation(doms))}
            xs, ys = [], []
            for r in rows:
                rr = look.get((mp[r["source"]], mp[r["target"]]))
                if rr is not None and rr[field] is not None:
                    xs.append(rr[field]); ys.append(r["retained"])
            if len(xs) > 5 and abs(spearman(np.array(xs), np.array(ys))) >= obs:
                cnt += 1
        return cnt / n_perm

    print(f"\nPART B: which geometric quantity tracks the damage? "
          f"(n = {len(rows)} ordered group pairs, {len(doms)} groups)\n")
    hdr = f"  {'predictor':<26s} {'note':<44s} {'rho':>7s} {'p_dom':>7s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    stats = {}
    scored = []
    for label, field, note in PREDICTORS:
        vals = [r[field] for r in rows]
        if any(v is None for v in vals):
            continue
        rho = spearman(np.array(vals), keep_arr)
        p = perm_p(field, abs(rho))
        stats[field] = {"rho": rho, "p_domain_perm": p, "note": note}
        scored.append((abs(rho), label, note, rho, p))
    for _, label, note, rho, p in sorted(scored, reverse=True):
        star = " *" if p < 0.05 else ""
        print(f"  {label:<26s} {note:<44s} {rho:+7.3f} {p:7.3f}{star}")
    print("\n  rho   = Spearman with retained EC F1; NEGATIVE = predicts more damage")
    print(f"  p_dom = permutation over the {len(doms)} groups, not over the "
          f"{len(rows)} pairs (the pairs are not independent)")
    print(f"  NOTE: {len(scored)} predictors were tested, so at p<0.05 roughly "
          f"{0.05*len(scored):.1f} false positives are expected by chance.")

    return {"rows": rows, "predictors": stats, "n_pairs": len(rows)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--ec", required=True)
    ap.add_argument("--ec_level", type=int, default=3)
    ap.add_argument("--min_n", type=int, default=15)
    ap.add_argument("--domain", default="bacteria", help="domain for the injection test")
    ap.add_argument("--betas", default="0.0,0.1,0.25,0.5,0.75,0.9,1.0")
    ap.add_argument("--mags", default="0.25,0.5,1.0,2.0")
    ap.add_argument("--n_rep", type=int, default=5)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_a", action="store_true")
    ap.add_argument("--only_a", action="store_true",
                    help="run the injection experiment and stop (skip the 210-pair battery)")
    ap.add_argument("--drop_groups", default="",
                    help="comma list of taxonomic groups to exclude")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel workers for the per-pair probes")
    ap.add_argument("--max_train", type=int, default=0,
                    help="cap probe training set per group (0 = uncapped)")
    ap.add_argument("--max_test", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    T = load_table(args.emb, args.meta, args.ec, args.ec_level)
    T = drop_groups(T, args.drop_groups.split(","))
    print(f"{len(T['X'])} proteins, {len(set(T['ec']))} ECs, "
          f"{len(set(T['dom']))} domains\n")

    out = {}
    if not args.skip_a:
        betas = [float(b) for b in args.betas.split(",")]
        mags = [float(m) for m in args.mags.split(",")]
        out["part_a_injection"] = part_a(T, args.domain, args.min_n, args.seed,
                                         betas, mags, args.n_rep)
        print("=" * 78 + "\n")
    if not args.only_a:
        out["part_b_predictors"] = part_b(T, args.min_n, args.seed, args.n_perm,
                                          jobs=args.jobs,
                                          max_train=args.max_train or None,
                                          max_test=args.max_test or None)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out}")
