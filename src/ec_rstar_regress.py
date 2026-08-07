#!/usr/bin/env python
"""Which distance predicts r*? -- the join, the regression, and the honest null.

Takes every r* that got measured and every distance that got measured, joins them
on (source, target), and asks which distance ranks the pairs the way r* does.

Three things this script exists to get right, all of them landmines from the
brief:

  * PERMUTE GROUPS, NOT PAIRS (landmine 4). 210 ordered pairs come from 15
    groups and are nowhere near independent. `perm_p` below is copied from
    `beta_diagnosis.py` and relabels whole groups.
  * SAY HOW MANY PREDICTORS WERE TESTED (landmine 5). Printed with every table.
  * r* IS CENSORED. A pair that never reaches the bar at any r on the grid has
    no r*, and dropping those pairs biases the result towards the easy ones --
    they are precisely the hardest pairs. Every correlation is reported twice:
    on the uncensored subset, and with censored pairs pushed to a value worse
    than any observed (`--censor_at`). If the two disagree, neither is quoted
    without the other.

Also reports partial correlations controlling for the number of classes, because
the per-pair label set makes K vary from ~30 to ~112 and a 30-way problem is
easier than a 112-way one -- so K is a confound for anything that correlates with
group size.
"""
import argparse
import csv
import json
import math
import os

import numpy as np


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / d) if d > 1e-12 else float("nan")


def partial_spearman(x, y, z):
    """Spearman of x,y after rank-residualising both on z."""
    def rk(v):
        v = np.asarray(v, float)
        return np.argsort(np.argsort(v)).astype(float)
    rx, ry, rz = rk(x), rk(y), rk(z)
    A = np.c_[np.ones(len(rz)), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    d = np.linalg.norm(ex) * np.linalg.norm(ey)
    return float(ex @ ey / d) if d > 1e-12 else float("nan")


def load_json(path):
    return json.load(open(path)) if path and os.path.exists(path) else None


def read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        x = float(v)
        return x if not math.isnan(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rstar_mlp", default="", help="rstar_summary.csv from the ladder")
    ap.add_argument("--rstar_lr", default="", help="allpairs JSON from ec_rstar_allpairs")
    ap.add_argument("--distances", default="", help="ec_distance_metrics.json")
    ap.add_argument("--seq_identity", default="", help="ec_seq_identity parse output")
    ap.add_argument("--geometry", default="", help="beta_diagnosis.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=0,
                    help="restrict the MLP arm to one budget (0 = the smallest present)")
    ap.add_argument("--censor_at", type=float, default=1.25)
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # ------------------------------------------------------------------ join
    rec = {}

    def slot(s, t):
        return rec.setdefault((s, t), {"source": s, "target": t})

    mlp = [r for r in read_csv(a.rstar_mlp) if fnum(r.get("holdout", 0)) == 0.0]
    if mlp:
        budgets = sorted({int(fnum(r["budget"])) for r in mlp})
        B = a.budget or budgets[0]
        for r in mlp:
            if int(fnum(r["budget"])) != B:
                continue
            d = slot(r["source"], r["target"])
            d["mlp_rstar"] = fnum(r["r_star"])
            if r.get("r_star_matched") not in (None, ""):
                d["mlp_rstar_matched"] = fnum(r["r_star_matched"])
                d["mlp_ceiling_matched"] = fnum(r.get("ceiling_matched"))
            d["mlp_ceiling"] = fnum(r["ceiling"])
            d["mlp_budget_ceiling"] = fnum(r.get("budget_ceiling"))
            d["mlp_zeroshot"] = fnum(r["zero_shot"])
            d["mlp_retained"] = (fnum(r["zero_shot"]) / fnum(r["ceiling"])
                                 if fnum(r["ceiling"]) > 0 else float("nan"))
            d["n_classes"] = fnum(r["n_classes"])
            d["n_tgt_train"] = fnum(r.get("n_tgt_train"))
        print(f"MLP arm: {sum('mlp_rstar' in v for v in rec.values())} pairs "
              f"at budget {B} (budgets present: {budgets})")

    lr = load_json(a.rstar_lr)
    if lr:
        for row in lr["rows"]:
            d = slot(row["source"], row["target"])
            bs = sorted(int(k) for k in row["budgets"])
            key = str(a.budget) if str(a.budget) in row["budgets"] else str(bs[0])
            b = row["budgets"][key]
            d["lr_rstar"] = (b["r_star"] if b["r_star"] is not None else float("nan"))
            d["lr_ceiling"] = row["ceiling_f1"]
            d["lr_zeroshot"] = row["zeroshot_f1"]
            d["lr_retained"] = row["retained"]
            d.setdefault("n_classes", row["n_classes"])
            d["lr_budget"] = int(key)
        print(f"LR arm: {len(lr['rows'])} pairs")

    dist = load_json(a.distances)
    if dist:
        for row in dist["rows"]:
            d = slot(row["source"], row["target"])
            for k in ("mmd_rbf", "energy_dist", "proxy_a_dist", "feat_wasserstein",
                      "frechet", "mean_shift_norm", "mean_shift_over_gap",
                      "n_shared_lineage"):
                if row.get(k) is not None:
                    d[k] = float(row[k])
            d["lca_named_rank"] = row.get("lca_named_rank")
        print(f"distances: {len(dist['rows'])} pairs")

    sid = load_json(a.seq_identity)
    if sid:
        for row in sid["rows"]:
            d = slot(row["source"], row["target"])
            for k in ("pident_median", "pident_mean", "pident_p10",
                      "frac_below_30", "frac_below_30_censored", "frac_nohit"):
                if row.get(k) is not None:
                    d[k] = float(row[k])
        print(f"sequence identity: {len(sid['rows'])} groups")

    geo = load_json(a.geometry)
    if geo and "part_b_predictors" in geo:
        for row in geo["part_b_predictors"]["rows"]:
            d = slot(row["source"], row["target"])
            for k in ("mag_over_gap", "alpha", "beta_shared", "diff_abs",
                      "diff_inB_abs", "offB_abs", "inB_abs", "procrustes",
                      "logit_spread", "gap_ratio"):
                if row.get(k) is not None:
                    d["geo_" + k] = float(row[k])
            d["geo_retained"] = row.get("retained")
        print(f"geometry: {len(geo['part_b_predictors']['rows'])} pairs")

    rows = list(rec.values())
    if not rows:
        raise SystemExit("nothing joined -- check the input paths")

    # ------------------------------------------------------------ regression
    PRED = [k for k in sorted({k for r in rows for k in r})
            if k.startswith(("geo_", "mmd", "energy", "proxy", "feat_", "frechet",
                             "mean_shift", "pident", "frac_below", "frac_nohit",
                             "n_shared_lineage"))
            and k not in ("geo_retained",)]

    def perm_p(pred, targ, obs, rows_used):
        """Relabel whole GROUPS, not pairs -- copied from beta_diagnosis.perm_p."""
        rng = np.random.default_rng(a.seed)
        doms = sorted({r["source"] for r in rows_used} |
                      {r["target"] for r in rows_used})
        look = {(r["source"], r["target"]): r for r in rows_used}
        cnt = 0
        for _ in range(a.n_perm):
            mp = dict(zip(doms, rng.permutation(doms)))
            xs, ys = [], []
            for r in rows_used:
                rr = look.get((mp[r["source"]], mp[r["target"]]))
                if rr is not None and pred in rr and targ in r:
                    xv, yv = fnum(rr[pred]), fnum(r[targ])
                    if xv == xv and yv == yv:
                        xs.append(xv)
                        ys.append(yv)
            if len(xs) > 5 and abs(spearman(xs, ys)) >= obs:
                cnt += 1
        return cnt / a.n_perm

    report = {"n_rows": len(rows), "predictors_tested": PRED, "tables": {}}
    out_lines = []

    def emit(msg):
        print(msg, flush=True)
        out_lines.append(msg)

    for targ in ("mlp_rstar", "mlp_rstar_matched", "lr_rstar",
                 "mlp_retained", "lr_retained"):
        have = [r for r in rows if targ in r]
        if len(have) < 6:
            continue
        for mode in ("uncensored", "censored"):
            use = []
            for r in have:
                v = fnum(r[targ])
                if v != v:                       # NaN == never reached the bar
                    if mode == "censored" and targ.endswith("rstar"):
                        r = dict(r)
                        r[targ] = a.censor_at
                        use.append(r)
                    continue
                use.append(r)
            n_cens = len(have) - sum(1 for r in have if fnum(r[targ]) == fnum(r[targ]))
            if mode == "censored" and (n_cens == 0 or not targ.endswith("rstar")):
                continue
            if len(use) < 6:
                continue
            emit(f"\n### {targ}  [{mode}]  n = {len(use)} pairs, "
                 f"{n_cens} censored (never reached the bar)")
            emit(f"  {'predictor':<26s} {'rho':>7s} {'p_group':>8s} "
                 f"{'rho|K':>7s} {'n':>5s}")
            emit("  " + "-" * 58)
            tab = []
            for p in PRED:
                xs = [fnum(r[p]) for r in use if p in r]
                ys = [fnum(r[targ]) for r in use if p in r]
                ks = [fnum(r.get("n_classes", float("nan")))
                      for r in use if p in r]
                ok = [i for i in range(len(xs)) if xs[i] == xs[i] and ys[i] == ys[i]]
                if len(ok) < 6:
                    continue
                x = [xs[i] for i in ok]
                y = [ys[i] for i in ok]
                rho = spearman(x, y)
                sub = [r for r in use if p in r and fnum(r[p]) == fnum(r[p])
                       and fnum(r[targ]) == fnum(r[targ])]
                pv = perm_p(p, targ, abs(rho), sub)
                kk = [ks[i] for i in ok]
                prho = (partial_spearman(x, y, kk)
                        if all(v == v for v in kk) else float("nan"))
                tab.append((abs(rho), p, rho, pv, prho, len(ok)))
            for _, p, rho, pv, prho, n in sorted(tab, reverse=True):
                star = " *" if pv < 0.05 else ""
                emit(f"  {p:<26s} {rho:+7.3f} {pv:8.3f} {prho:+7.3f} {n:5d}{star}")
            emit(f"  {len(tab)} predictors tested -> about "
                 f"{0.05*len(tab):.1f} false positives at p<0.05 by chance.")
            emit("  rho|K = partial Spearman controlling for the number of EC classes.")
            report["tables"][f"{targ}_{mode}"] = [
                dict(predictor=p, rho=rho, p_group_perm=pv, partial_rho_given_K=prho,
                     n=n) for _, p, rho, pv, prho, n in sorted(tab, reverse=True)]

    # agreement between the two estimators on the pairs they share
    both = [r for r in rows if "mlp_rstar" in r and "lr_rstar" in r]
    ok = [r for r in both if fnum(r["mlp_rstar"]) == fnum(r["mlp_rstar"])
          and fnum(r["lr_rstar"]) == fnum(r["lr_rstar"])]
    if len(ok) >= 5:
        rho = spearman([fnum(r["mlp_rstar"]) for r in ok],
                       [fnum(r["lr_rstar"]) for r in ok])
        emit(f"\n### estimator agreement on the {len(ok)} shared pairs "
             f"(of {len(both)}): Spearman(MLP r*, LR r*) = {rho:+.3f}")
        emit("  These are NOT the same estimator; this says whether they rank "
             "targets the same way, not whether the numbers match.")
        report["estimator_agreement"] = {"n": len(ok), "spearman": rho}

    report["rows"] = rows
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(a.out.replace(".json", ".txt"), "w") as f:
        f.write("\n".join(out_lines) + "\n")
    fields = sorted({k for r in rows for k in r})
    with open(a.out.replace(".json", "_joined.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out} (+ .txt, _joined.csv)")


if __name__ == "__main__":
    main()
