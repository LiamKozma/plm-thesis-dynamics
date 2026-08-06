#!/usr/bin/env python
"""Geometry AND damage for EVERY ordered pair of taxonomic domains (n = 20).

Why not just bacteria -> X
--------------------------
With bacteria as the only source there are four measurements, so "does beta predict
damage better than distance does?" rests on four points and cannot be answered --
both correlate weakly and negatively and the ordering is broken by a single domain.

Every ordered pair of the five domains gives 20 shifts instead. Each one has a
measured magnitude, a measured beta (share of the shift lying in the source domain's
EC-discriminative subspace) and a measured cost (retained EC F1 under a linear probe
trained on the source). That is enough to ask which geometric quantity actually
tracks the cost -- which is the premise the whole generator rests on.

Each pair is self-contained: the function subspace is refit on that pair's source
domain, and the label set is restricted to ECs present on both sides, so the ceiling
and the zero-shot number describe the same classification problem.
"""
import argparse
import itertools
import json

import numpy as np

from measure_ec_geometry import (load_table, conditioned_shift, shift_stats,
                                 function_subspace, beta_of, centroid,
                                 axis_alignment)
from measure_ec_damage import run_pair


def spearman(a, b):
    """Rank correlation -- the relationship need not be linear, and with 20 points
    a single extreme domain pair would dominate a Pearson r."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / d) if d > 1e-12 else float("nan")


def partial_spearman(x, y, z):
    """Rank correlation of x and y after removing what z explains of each.

    beta and magnitude are themselves correlated across domain pairs, so a raw
    correlation of either with damage may be inherited from the other.
    """
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rz = np.argsort(np.argsort(z)).astype(float)

    def resid(a, b):
        b1 = np.stack([b, np.ones_like(b)], 1)
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef

    ex, ey = resid(rx, rz), resid(ry, rz)
    d = np.linalg.norm(ex) * np.linalg.norm(ey)
    return float(ex @ ey / d) if d > 1e-12 else float("nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--ec", required=True)
    ap.add_argument("--ec_level", type=int, default=3)
    ap.add_argument("--min_n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    T = load_table(args.emb, args.meta, args.ec, args.ec_level)
    domains = sorted(set(T["dom"]))
    X = T["X"]
    print(f"{len(X)} proteins, {len(set(T['ec']))} ECs, domains: {domains}")
    print(f"Measuring all {len(domains)*(len(domains)-1)} ordered pairs "
          f"(refitting the function subspace per source domain)\n")

    # function subspace + reference scales are properties of the SOURCE domain
    subspace, scales = {}, {}
    for d in domains:
        B, k, n_c = function_subspace(T, d)
        subspace[d] = B
        m = T["dom"] == d
        keys = [e for e in sorted(set(T["ec"])) if (m & (T["ec"] == e)).sum() >= args.min_n]
        C = np.stack([centroid(X, m & (T["ec"] == e)) for e in keys])
        iu = np.triu_indices(len(C), 1)
        gap = float(np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))[iu].mean())
        scales[d] = gap
        print(f"  {d:9s} function subspace {k:2d} dims over {n_c} EC centroids, "
              f"mean inter-EC gap {gap:.3f}")

    rows = []
    for s, t in itertools.permutations(domains, 2):
        V, keys, _ = conditioned_shift(T, "ec", "dom", s, t, args.min_n)
        if len(V) < 3:
            print(f"  {s}->{t}: only {len(V)} shared ECs, skipped")
            continue
        st = shift_stats(V)
        bet = beta_of(V, subspace[s])
        # The generator's beta is defined on the SHARED translation, not on
        # individual family shifts, so the per-cell mean above is not the matching
        # quantity. beta_shared is: project the mean shift vector -- the common
        # component every EC undergoes -- onto the function subspace.
        bet_shared = float(beta_of(V.mean(0)[None], subspace[s])[0])
        dmg = run_pair(T, "ec", s, t, args.min_n, args.seed)
        if dmg is None:
            continue
        rows.append(dict(source=s, target=t, n_ec_cells=len(V),
                         mag=st["mag_mean"], mag_over_gap=st["mag_mean"] / scales[s],
                         mean_cos=st["mean_cos"], shared_frac=st["shared_frac"],
                         beta=float(bet.mean()), beta_shared=bet_shared,
                         n_classes=dmg["n_classes"],
                         ceiling_f1=dmg["ceiling_f1"], zeroshot_f1=dmg["zeroshot_f1"],
                         retained=dmg["retained"]))
        print(f"  {s:9s} -> {t:9s}  cells {len(V):2d}  |v|/gap {rows[-1]['mag_over_gap']:.2f}  "
              f"beta {rows[-1]['beta']:.3f}  alpha {st['mean_cos']:+.2f}  "
              f"ceil {dmg['ceiling_f1']:.3f}  0shot {dmg['zeroshot_f1']:.3f}  "
              f"keep {dmg['retained']:.3f}")

    print(f"\n=== Which geometric quantity tracks the damage? (n = {len(rows)} pairs) ===")
    keep = np.array([r["retained"] for r in rows])
    beta = np.array([r["beta"] for r in rows])
    beta_sh = np.array([r["beta_shared"] for r in rows])
    mag = np.array([r["mag_over_gap"] for r in rows])
    alpha = np.array([r["mean_cos"] for r in rows])
    print(f"  Spearman(beta per-cell,   retained EC F1) = {spearman(beta, keep):+.3f}")
    print(f"  Spearman(beta of SHARED,  retained EC F1) = {spearman(beta_sh, keep):+.3f}"
          f"   <- the generator's definition")
    print(f"  Spearman(|v|/gap,         retained EC F1) = {spearman(mag, keep):+.3f}")
    print(f"  Spearman(alpha,           retained EC F1) = {spearman(alpha, keep):+.3f}")
    print(f"  Spearman(beta_shared, |v|/gap)            = {spearman(beta_sh, mag):+.3f}"
          f"   (are the predictors entangled?)")
    print(f"\n  partial Spearman(beta_shared, retained | magnitude) = "
          f"{partial_spearman(beta_sh, keep, mag):+.3f}")
    print(f"  partial Spearman(magnitude, retained | beta_shared) = "
          f"{partial_spearman(mag, keep, beta_sh):+.3f}")

    # The 20 pairs are NOT independent: they are built from 5 domains, and both
    # beta and the damage look largely like properties of the TARGET domain. A
    # correlation over pairs therefore has far fewer than 20 degrees of freedom.
    # Permuting whole domains (not pairs) respects that structure.
    doms = sorted({r["source"] for r in rows} | {r["target"] for r in rows})
    def domain_perm_null(x, n_perm=2000):
        obs = abs(spearman(x, keep))
        cnt = 0
        for _ in range(n_perm):
            mp = {d: e for d, e in zip(doms, rng.permutation(doms))}
            xs = []
            look = {(r["source"], r["target"]): r for r in rows}
            for r in rows:
                rr = look.get((mp[r["source"]], mp[r["target"]]))
                xs.append(np.nan if rr is None else rr[
                    "mag_over_gap" if x is mag else
                    ("beta_shared" if x is beta_sh else "beta")])
            xs = np.array(xs)
            ok2 = ~np.isnan(xs)
            if ok2.sum() > 5 and abs(spearman(xs[ok2], keep[ok2])) >= obs:
                cnt += 1
        return obs, cnt / n_perm
    o_m, p_m = domain_perm_null(mag)
    o_b, p_b = domain_perm_null(beta_sh)
    print(f"\n  domain-level permutation test (respects the 5-domain structure):")
    print(f"    |v|/gap      |rho| = {o_m:.3f}, p = {p_m:.3f}")
    print(f"    beta_shared  |rho| = {o_b:.3f}, p = {p_b:.3f}")

    out = {"rows": rows,
           "spearman": {"beta_vs_retained": spearman(beta, keep),
                        "beta_shared_vs_retained": spearman(beta_sh, keep),
                        "mag_vs_retained": spearman(mag, keep),
                        "alpha_vs_retained": spearman(alpha, keep),
                        "beta_shared_vs_mag": spearman(beta_sh, mag),
                        "partial_beta_shared": partial_spearman(beta_sh, keep, mag),
                        "partial_mag": partial_spearman(mag, keep, beta_sh),
                        "domain_perm_p_mag": p_m,
                        "domain_perm_p_beta_shared": p_b}}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out}")
