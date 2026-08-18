#!/usr/bin/env python3
"""T2.2 -- percent identity against retention, on all 210 ordered pairs.

Every pident claim in this project rested on 14 points from a single source
group, and the document flagged any identity-against-r* claim as suggestive until
BLAST had been run out of every source. It now has been (job 47525602, 225 hits
files, 1h17 on 32 cores), so this script answers the two questions that were open:

  1. Does the identity-to-retention correlation hold when the source is not held
     fixed at gammaproteobacteria?
  2. Does the ~40% median-identity boundary that coincides with the
     bacterial-against-non-bacterial retention split generalise off that row?

Question 2 must be asked **relative to each source's own clade** -- for a
eukaryotic source the near targets are the other eukaryotes -- which is what
`clades.targets_by_clade` is for. Asking it with a fixed "bacterial" set is the
mistake documented at the top of `clades.py`.
"""
import argparse, csv, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clades import clade_of, targets_by_clade, USABLE  # noqa: E402

SEQID = "/scratch/lmk04992/ec_rstar/seqid_allpairs"
FLAT = "/scratch/lmk04992/ec_rstar/rstar_allpairs_flat.csv"
JOINED = "/scratch/lmk04992/ec_rstar/rstar_vs_distance_P500_joined.csv"


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 4:
        return float("nan"), len(a)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1]), len(a)


def partial(x, y, z):
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 5:
        return float("nan")
    R = np.stack([np.argsort(np.argsort(v[m])).astype(float) for v in (x, y, z)])
    R = R - R.mean(1, keepdims=True)
    C = np.corrcoef(R)
    den = np.sqrt(max((1 - C[0, 2] ** 2) * (1 - C[1, 2] ** 2), 1e-12))
    return float((C[0, 1] - C[0, 2] * C[1, 2]) / den)


def perm_p(rows, xf, yf, obs, n_perm=2000, seed=5):
    """Permute whole groups: 210 pairs come from 15 groups (landmine 4)."""
    rng = np.random.default_rng(seed)
    doms = sorted({r["source"] for r in rows} | {r["target"] for r in rows})
    look = {(r["source"], r["target"]): r for r in rows}
    cnt = 0
    for _ in range(n_perm):
        mp = dict(zip(doms, rng.permutation(doms)))
        xs, ys = [], []
        for r in rows:
            rr = look.get((mp[r["source"]], mp[r["target"]]))
            if rr is None:
                continue
            xs.append(rr.get(xf)); ys.append(r.get(yf))
        rho, n = sp(xs, ys)
        if n > 5 and abs(rho) >= obs:
            cnt += 1
    return cnt / n_perm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqid", default=SEQID)
    ap.add_argument("--out", default="/scratch/lmk04992/ec_rstar/pident_allpairs.json")
    ap.add_argument("--n_perm", type=int, default=2000)
    a = ap.parse_args()

    pid = {}
    for f in sorted(glob.glob(os.path.join(a.seqid, "seq_identity_src_*.json"))):
        for r in json.load(open(f))["rows"]:
            if r["source"] != r["target"]:
                pid[(r["source"], r["target"])] = r
    print("loaded %d ordered pairs of BLAST summaries" % len(pid))

    ret, geo = {}, {}
    with open(FLAT) as f:
        for d in csv.DictReader(f):
            if d["budget"] == "500":
                ret[(d["source"], d["target"])] = d
    if os.path.exists(JOINED):
        with open(JOINED) as f:
            for d in csv.DictReader(f):
                geo[(d["source"], d["target"])] = d

    rows = []
    for k, r in pid.items():
        if k not in ret:
            continue
        g = geo.get(k, {})
        def fl(d, c):
            try:
                return float(d[c])
            except (KeyError, TypeError, ValueError):
                return float("nan")
        rows.append(dict(source=k[0], target=k[1],
                         clade_source=clade_of(k[0]), clade_target=clade_of(k[1]),
                         same_clade=int(clade_of(k[0]) == clade_of(k[1])),
                         retained=fl(ret[k], "retained"),
                         r_star_budget=fl(ret[k], "r_star_budget"),
                         proxy_a_dist=fl(g, "proxy_a_dist"),
                         **{c: r.get(c, float("nan")) for c in
                            ("pident_median", "pident_mean", "pident_p10",
                             "frac_below_30", "frac_below_40", "frac_nohit")}))

    out = {"n_pairs": len(rows), "correlations": {}, "boundary": {}}
    print("\n=== 1. identity against retention, n = %d (was 14) ===" % len(rows))
    print("  %-18s %8s %10s %10s %8s" % ("predictor", "rho", "partial|A", "rho_gamma", "p_perm"))
    gam = [r for r in rows if r["source"] == "gammaproteobacteria"]
    for fld in ("pident_median", "pident_mean", "pident_p10",
                "frac_below_30", "frac_below_40", "frac_nohit"):
        v = [r[fld] for r in rows]
        R = [r["retained"] for r in rows]
        rho, n = sp(v, R)
        pa = partial(v, R, [r["proxy_a_dist"] for r in rows])
        rg, ng = sp([r[fld] for r in gam], [r["retained"] for r in gam])
        pp = perm_p(rows, fld, "retained", abs(rho), a.n_perm)
        out["correlations"][fld] = dict(rho=round(rho, 4), n=n,
                                       partial_given_proxy_a=round(pa, 4),
                                       rho_gamma_only=round(rg, 4), n_gamma=ng,
                                       p_group_perm=pp)
        print("  %-18s %+8.3f %10.3f %10.3f %8.4f%s"
              % (fld, rho, pa, rg, pp, " *" if pp < 0.05 else ""))

    print("\n=== 2. the identity boundary, asked relative to EACH source's clade ===")
    print("  %-24s %6s %9s %11s %8s" % ("source", "clade", "min within", "max outside", "sep"))
    n_sep = 0
    for src in sorted({r["source"] for r in rows}):
        within, outside = targets_by_clade(src)
        w = [r["pident_median"] for r in rows
             if r["source"] == src and r["target"] in within]
        o = [r["pident_median"] for r in rows
             if r["source"] == src and r["target"] in outside]
        if not w or not o:
            continue
        sep = min(w) > max(o)
        n_sep += int(sep)
        out["boundary"][src] = dict(clade=clade_of(src), min_within=round(min(w), 2),
                                    max_outside=round(max(o), 2), separated=bool(sep),
                                    gap=round(min(w) - max(o), 2),
                                    n_within=len(w), n_outside=len(o))
        print("  %-24s %6s %8.1f%% %10.1f%% %8s  (gap %+.1f pts)"
              % (src, clade_of(src)[:6], min(w), max(o), sep, min(w) - max(o)))
    out["boundary_summary"] = dict(n_sources_separated=n_sep,
                                  n_sources_tested=len(out["boundary"]))
    print("\n  separated in %d of %d source groups"
          % (n_sep, len(out["boundary"])))
    # where the boundary actually sits, pooled
    allw = [r["pident_median"] for r in rows if r["same_clade"]]
    allo = [r["pident_median"] for r in rows if not r["same_clade"]]
    out["pooled_boundary"] = dict(
        within_clade_min=round(float(np.min(allw)), 2),
        within_clade_median=round(float(np.median(allw)), 2),
        outside_clade_max=round(float(np.max(allo)), 2),
        outside_clade_median=round(float(np.median(allo)), 2),
        n_within=len(allw), n_outside=len(allo))
    print("  pooled: within-clade pairs %.1f%% (min %.1f) vs outside-clade %.1f%% (max %.1f)"
          % (np.median(allw), np.min(allw), np.median(allo), np.max(allo)))

    with open(a.out, "w") as f:
        json.dump({"rows": rows, **out}, f, indent=2, default=str)
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
