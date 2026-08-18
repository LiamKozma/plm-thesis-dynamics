#!/usr/bin/env python3
"""Is a subspace special, or is it only where the low-variance directions live?

The two arms of `subspace_rstar.py` disagree, and each normalisation is confounded
in the opposite direction:

  * matched EUCLIDEAN length -- H-minus-B looks worst, but at one EC gap H-minus-B
    is a 14-SD intervention while B is a 3-SD one.
  * matched WITHIN-CLASS SD -- H looks worst, but 20 SD along H is a 5.9-gap move
    while 20 SD along B-minus-H is a 0.48-gap move.

So neither arm can be read as "subspace X is special". Pool all three arms and ask
whether the condition label explains anything once BOTH magnitude measures are in
the model. If it does not, the honest statement is that the homology subspace is
not special in itself: it is special because it is where the function-discriminative
directions with small within-class spread live.

Fitted on ranks, so it is the same currency as every Spearman in the project.
"""
import argparse, csv, glob, itertools, json, os
import numpy as np


def rank(v):
    v = np.asarray(v, float)
    return np.argsort(np.argsort(v)).astype(float)


def ols_r2(X, y):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - (resid ** 2).sum() / ss_tot, beta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="/scratch/lmk04992/subspace_rstar")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--out", default="/scratch/lmk04992/subspace_rstar/analysis.json")
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, "subspace_rstar_summary*.csv"))):
        arm = os.path.basename(f).replace("subspace_rstar_summary", "").replace(".csv", "")
        with open(f) as fh:
            for d in csv.DictReader(fh):
                if int(float(d["budget"])) != a.budget:
                    continue
                if d["condition"] == "none":
                    continue
                try:
                    sd = float(d["mag_in_within_class_sd"])
                    gaps = float(d["magnitude_gaps"])
                    ret = float(d["zero_shot_over_ceiling"])
                except (TypeError, ValueError):
                    continue
                rows.append(dict(arm=arm, condition=d["condition"], sd=sd, gaps=gaps,
                                 retained=ret, whitened=int(d["whitened"]),
                                 cond_dim=int(d["cond_dim"])))
    print("pooled %d cells from %d arms at budget %d"
          % (len(rows), len({r["arm"] for r in rows}), a.budget))
    for w in (0, 1):
        sub = [r for r in rows if r["whitened"] == w]
        if len(sub) < 12:
            continue
        name = "whitened" if w else "raw"
        y = rank([r["retained"] for r in sub])
        rsd = rank([r["sd"] for r in sub])
        rgp = rank([r["gaps"] for r in sub])
        conds = sorted({r["condition"] for r in sub})
        dummies = [[1.0 if r["condition"] == c else 0.0 for r in sub] for c in conds[1:]]

        r2_sd, _ = ols_r2([rsd], y)
        r2_gp, _ = ols_r2([rgp], y)
        r2_both, _ = ols_r2([rsd, rgp], y)
        r2_full, _ = ols_r2([rsd, rgp] + dummies, y)
        r2_cond, _ = ols_r2(dummies, y)
        k1, k2 = 2, 2 + len(dummies)
        n = len(sub)
        # partial F for adding the condition dummies on top of both magnitudes
        num = (r2_full - r2_both) / max(len(dummies), 1)
        den = (1 - r2_full) / max(n - k2 - 1, 1)
        F = num / den if den > 0 else float("nan")
        print("\n=== %s coordinates, n = %d cells, %d conditions ===" % (name, n, len(conds)))
        print("  R2 from within-class SD alone            %.3f" % r2_sd)
        print("  R2 from Euclidean length in gaps alone   %.3f" % r2_gp)
        print("  R2 from BOTH magnitude measures          %.3f" % r2_both)
        print("  R2 from condition label alone            %.3f" % r2_cond)
        print("  R2 from both magnitudes + condition      %.3f" % r2_full)
        print("  condition adds %+.3f R2 on top of magnitude   (partial F = %.2f on %d, %d df)"
              % (r2_full - r2_both, F, len(dummies), n - k2 - 1))
        if r2_full - r2_both < 0.05:
            print("  -> the condition label adds essentially nothing. The subspace is not")
            print("     special in itself; it is special because of the spread of the")
            print("     directions it contains.")
        # per-condition mean SD needed per gap: the mechanism, stated plainly
        print("\n  within-class SD per EC gap, by condition (the mechanism):")
        for c in conds:
            v = [r["sd"] / r["gaps"] for r in sub if r["condition"] == c and r["gaps"] > 0]
            if v:
                print("    %-20s %6.1f SD per gap" % (c, float(np.mean(v))))
    with open(a.out, "w") as f:
        json.dump({"rows": rows, "budget": a.budget}, f, indent=2)
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
