#!/usr/bin/env python3
"""Does the HOMOLOGY component of a real taxonomic shift predict FUNCTION damage?

The observational twin of `subspace_rstar.py`. That script injects a displacement
into a chosen subspace and measures what it costs. This one injects nothing: it
takes the shifts that really occur between taxonomic groups, splits each one into
the part lying along the homology subspace H, the part along the function
subspace B, and the part outside both, and asks which piece tracks the loss of EC
prediction across the 210 ordered group pairs.

Why both are needed
-------------------
The injected version is causal but constructed, and it is open to the objection
that a fixed Euclidean displacement is not a fixed intervention -- two EC gaps is
a different number of within-class standard deviations inside the function
subspace than outside it. The observational version cannot be accused of being a
construction, because nothing is constructed: real archaea really do sit where
they sit. It is correlational instead. Agreement between the two is the argument;
disagreement is the finding.

The decomposition
-----------------
B and H overlap, so "the fraction in B" and "the fraction in H" do not add to
anything. The decomposition used here is orthogonal and exhaustive:

    R^d  =  B  (+)  (H minus B)  (+)  outside both

so the three squared fractions sum to 1 and can be compared directly. The mirror
split, H (+) (B minus H) (+) outside, is reported as well, because which of the
two overlapping subspaces gets credit for the shared directions is a choice and
should be visible rather than silent.

Every fraction is reported **against a matched null**. dim(H-B) is roughly three
times dim(B), so a direction drawn at random from the source-group data
covariance already puts about three times as much of itself in H-B. The raw
fraction is therefore uninterpretable; the excess over that null is the quantity
that means something.

Both coordinate systems are run (`--whiten`), because the whole raw-against-
whitened question is live: the subspace damage result holds in raw coordinates
and largely vanishes in whitened ones, and the generator lives in raw ones.

Outputs
-------
  shift_decomposition<tag>.csv    one row per ordered pair, all fractions and excesses
  shift_decomposition<tag>.json   Spearman of each fraction against retention and r*,
                                  with a group-level permutation null (landmine 4)
"""
import argparse, csv, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from measure_ec_geometry import (load_table, drop_groups, centroid,          # noqa: E402
                                 conditioned_shift)

DEF_EMB = "/scratch/lmk04992/ec_swissprot/emb_cache_esmc.npy"
DEF_META = "/scratch/lmk04992/ec_swissprot/data/metadata.tsv"
DEF_EC = "/scratch/lmk04992/ec_swissprot/data/ec_annotations.tsv"
DEF_JOIN = "/scratch/lmk04992/ec_rstar/rstar_allpairs_flat.csv"


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 4:
        return float("nan"), 0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return (float(ra @ rb / d) if d > 1e-12 else float("nan")), int(len(a))


def partial_spearman(x, y, z):
    """Spearman of x and y with z partialled out, on ranks."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 5:
        return float("nan")
    R = np.stack([np.argsort(np.argsort(v[m])).astype(float) for v in (x, y, z)])
    R = R - R.mean(1, keepdims=True)
    C = np.corrcoef(R)
    den = np.sqrt(max((1 - C[0, 2] ** 2) * (1 - C[1, 2] ** 2), 1e-12))
    return float((C[0, 1] - C[0, 2] * C[1, 2]) / den)


def orth_complement_within(A, Bsub):
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(Bsub)
    M = Qa - Qb @ (Qb.T @ Qa)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, S > 1e-8]


def frac_in(v, Q):
    """Squared fraction of v inside the span of orthonormal columns Q."""
    if Q is None or Q.shape[1] == 0:
        return 0.0
    n2 = float(v @ v)
    if n2 < 1e-24:
        return float("nan")
    p = Q.T @ v
    return float((p @ p) / n2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", default=DEF_EMB)
    ap.add_argument("--meta", default=DEF_META)
    ap.add_argument("--ec", default=DEF_EC)
    ap.add_argument("--join", default=DEF_JOIN,
                    help="CSV with source,target,retained,r_star_budget to correlate against")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ec_level", type=int, default=3)
    ap.add_argument("--min_n", type=int, default=15)
    ap.add_argument("--var_frac", type=float, default=0.90)
    ap.add_argument("--drop", default="other_bacteria,other_eukaryota")
    ap.add_argument("--n_null", type=int, default=400,
                    help="matched directions drawn per source group for the null")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--whiten", action="store_true",
                    help="whiten per source group before decomposing")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    log("loading table")
    T = load_table(a.emb, a.meta, a.ec, a.ec_level)
    T = drop_groups(T, [x for x in a.drop.split(",") if x])
    groups = sorted(set(T["dom"]))
    log("%d proteins, %d groups" % (len(T["X"]), len(groups)))

    # ------------------------------------------------- per-source-group bases
    # Whitening is linear, so it can be applied to centroids and shift vectors
    # rather than to the 231k x 960 block: W is fit on the SOURCE group's own
    # covariance, C_white = (C - mu) W, and V_white = V W because the shared mu
    # cancels in a difference. In whitened coordinates the source covariance is
    # the identity, so a matched random direction is simply isotropic.
    basis = {}
    for g in groups:
        m = T["dom"] == g
        Xg = T["X"][m]
        mu = Xg.mean(0)
        W = None
        if a.whiten:
            _, S0, Vt0 = np.linalg.svd(Xg - mu, full_matrices=False)
            k0 = S0 > (S0.max() * 1e-6)
            W = Vt0[k0].T / (S0[k0] / np.sqrt(len(Xg)))

        def cents(labels, min_c):
            keys = [k for k in sorted(set(labels[m])) if (m & (labels == k)).sum() >= min_c]
            C = np.stack([centroid(T["X"], m & (labels == k)) for k in keys])
            return ((C - mu) @ W) if W is not None else (C - mu)

        C_ec = cents(T["ec"], a.min_n)
        C_fam = cents(T["fam"], 5)

        def top(C, var_keep):
            Cc = C - C.mean(0)
            _, S, Vt = np.linalg.svd(Cc, full_matrices=False)
            frac = np.cumsum(S ** 2) / (S ** 2).sum()
            k = int(np.searchsorted(frac, var_keep) + 1)
            return Vt[:k].T, k

        B, kB = top(C_ec, a.var_frac)
        H, kH = top(C_fam, a.var_frac)
        QB, _ = np.linalg.qr(B)
        QH, _ = np.linalg.qr(H)
        QHmB = orth_complement_within(H, B)
        QBmH = orth_complement_within(B, H)
        QBH, _ = np.linalg.qr(np.concatenate([B, H], axis=1))

        iu = np.triu_indices(len(C_ec), 1)
        gap = float(np.sqrt(((C_ec[:, None] - C_ec[None]) ** 2).sum(-1))[iu].mean())

        rng = np.random.default_rng(11)
        if W is not None:
            d_w = QB.shape[0]
            draws = rng.standard_normal((a.n_null, d_w))
        else:
            _, S, Vt = np.linalg.svd(Xg - mu, full_matrices=False)
            draws = (rng.standard_normal((a.n_null, S.shape[0])) * S) @ Vt
        draws = draws / np.linalg.norm(draws, axis=1, keepdims=True)
        nullm = np.array([[frac_in(v, QB), frac_in(v, QHmB), frac_in(v, QH),
                           frac_in(v, QBmH), 1 - frac_in(v, QBH)] for v in draws])
        nullm = np.nanmean(nullm, axis=0)

        basis[g] = dict(QB=QB, QH=QH, QHmB=QHmB, QBmH=QBmH, QBH=QBH, gap=gap, W=W,
                        kB=int(kB), kH=int(kH), kHmB=int(QHmB.shape[1]),
                        kBmH=int(QBmH.shape[1]), null=nullm)
        log("  %-24s dim B %3d  dim H %3d  dim H-B %3d  gap %.3f  "
            "null fracs B %.3f H-B %.3f out %.3f"
            % (g, kB, kH, QHmB.shape[1], gap, nullm[0], nullm[1], nullm[4]))

    # ------------------------------------------------------------ every pair
    rows = []
    for s in groups:
        bs = basis[s]
        for t in groups:
            if s == t:
                continue
            V, keys, _ = conditioned_shift(T, "ec", "dom", s, t, a.min_n)
            if len(V) < 3:
                continue
            if bs["W"] is not None:
                V = V @ bs["W"]
            vbar = V.mean(0)
            R = V - vbar
            def pack(v, pre):
                fB = frac_in(v, bs["QB"]); fHmB = frac_in(v, bs["QHmB"])
                fH = frac_in(v, bs["QH"]); fBmH = frac_in(v, bs["QBmH"])
                fout = 1.0 - frac_in(v, bs["QBH"])
                n = bs["null"]
                return {
                    pre + "_in_B": fB, pre + "_in_HminusB": fHmB, pre + "_outside": fout,
                    pre + "_in_H": fH, pre + "_in_BminusH": fBmH,
                    pre + "_in_B_excess": fB - n[0],
                    pre + "_in_HminusB_excess": fHmB - n[1],
                    pre + "_in_H_excess": fH - n[2],
                    pre + "_in_BminusH_excess": fBmH - n[3],
                    pre + "_outside_excess": fout - n[4],
                }
            row = dict(source=s, target=t, n_cells=len(V),
                       dim_B=bs["kB"], dim_H=bs["kH"], dim_HminusB=bs["kHmB"],
                       gap=round(bs["gap"], 4),
                       mag_over_gap=float(np.linalg.norm(vbar) / bs["gap"]),
                       diff_over_gap=float(np.linalg.norm(R, axis=1).mean() / bs["gap"]))
            row.update(pack(vbar, "shared"))
            # the differential component, averaged over cells
            dif = [pack(r, "diff") for r in R]
            for k in dif[0]:
                row[k] = float(np.nanmean([d[k] for d in dif]))
            # absolute (not fractional) displacement in each piece, in gap units
            for nm, Q in (("B", bs["QB"]), ("HminusB", bs["QHmB"]), ("H", bs["QH"])):
                row["shared_abs_" + nm] = float(np.linalg.norm(Q.T @ vbar) / bs["gap"])
            rows.append(row)
        log("  source %s done (%d pairs)" % (s, sum(1 for r in rows if r["source"] == s)))

    log("%d ordered pairs decomposed" % len(rows))

    # ------------------------------------------------------------- join outcomes
    outcomes = {}
    if os.path.exists(a.join):
        with open(a.join) as f:
            for d in csv.DictReader(f):
                key = (d["source"], d["target"])
                if key in outcomes and d.get("budget") not in ("500", None, ""):
                    continue
                outcomes[key] = d
        log("joined %d outcome rows from %s" % (len(outcomes), a.join))
    for r in rows:
        d = outcomes.get((r["source"], r["target"]), {})
        for c in ("retained", "r_star_budget", "r_star", "zeroshot_f1", "ceiling_f1",
                  "n_classes"):
            v = d.get(c)
            try:
                r[c] = float(v)
            except (TypeError, ValueError):
                r[c] = float("nan")

    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); cols.append(k)
    p = os.path.join(a.outdir, "shift_decomposition%s.csv" % a.tag)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)
    log("wrote %s" % p)

    # ----------------------------------------------------------- correlations
    doms = sorted({r["source"] for r in rows} | {r["target"] for r in rows})
    look = {(r["source"], r["target"]): r for r in rows}
    rng = np.random.default_rng(3)

    def perm_p(field, outcome, obs):
        """Permute whole GROUPS, not pairs -- the 210 pairs share only 15 groups."""
        cnt = 0
        for _ in range(a.n_perm):
            mp = {d: e for d, e in zip(doms, rng.permutation(doms))}
            xs, ys = [], []
            for r in rows:
                rr = look.get((mp[r["source"]], mp[r["target"]]))
                if rr is None:
                    continue
                xs.append(rr[field]); ys.append(r[outcome])
            rho, n = spearman(xs, ys)
            if n > 5 and abs(rho) >= obs:
                cnt += 1
        return cnt / a.n_perm

    fields = [c for c in cols if c.startswith(("shared_", "diff_")) or
              c in ("mag_over_gap", "diff_over_gap")]
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "config": vars(a),
           "n_pairs": len(rows), "dims": {g: {k: basis[g][k] for k in
                                              ("kB", "kH", "kHmB", "kBmH")} for g in groups},
           "correlations": {}}
    for outcome in ("retained", "r_star_budget"):
        vals = [r[outcome] for r in rows]
        if not np.isfinite(vals).any():
            continue
        table = []
        for fld in fields:
            rho, n = spearman([r[fld] for r in rows], vals)
            if not np.isfinite(rho):
                continue
            pr = partial_spearman([r[fld] for r in rows], vals,
                                  [r["mag_over_gap"] for r in rows])
            table.append(dict(predictor=fld, rho=round(rho, 4), n=n,
                              partial_given_magnitude=round(pr, 4),
                              p_group_perm=perm_p(fld, outcome, abs(rho))))
        table.sort(key=lambda x: -abs(x["rho"]))
        res["correlations"][outcome] = table
        print("\n=== %s : which piece of the real shift tracks it? (n = %d pairs) ==="
              % (outcome, len(rows)))
        print("  %-34s %7s %7s %8s" % ("predictor", "rho", "partial", "p_perm"))
        print("  " + "-" * 60)
        for x in table[:18]:
            print("  %-34s %7.3f %7.3f %8.4f%s"
                  % (x["predictor"], x["rho"], x["partial_given_magnitude"],
                     x["p_group_perm"], " *" if x["p_group_perm"] < 0.05 else ""))

    q = os.path.join(a.outdir, "shift_decomposition%s.json" % a.tag)
    with open(q, "w") as f:
        json.dump(res, f, indent=2, default=str)
    log("wrote %s" % q)
    print("\nNOTE: %d predictors were tested against each outcome; at p<0.05 about "
          "%.1f false positives are expected by chance."
          % (len(fields), 0.05 * len(fields)))


if __name__ == "__main__":
    main()
