#!/usr/bin/env python
"""r* for EC labels over ALL ordered group pairs, with a linear probe.

Companion to `ec_recovery_threshold.py`. That script uses the MLP adaptation loop
so its r* is comparable to the synthetic sweep; it is therefore only affordable
for one source domain (14 pairs). This script trades comparability for breadth:
a linear probe over all 210 ordered pairs, which is what gives the
distance -> r* regression enough points to survive a group-level permutation
null (the brief's landmine 4: permute groups, not pairs).

    THE TWO r* VALUES ARE NOT THE SAME NUMBER AND MUST NOT BE POOLED.
    The MLP r* warm-starts from the source model and takes one pass over the
    pool. A logistic regression has no warm start, so here "adaptation" is a
    refit on the mixed pool. Report them in separate columns; the 14 pairs they
    share are the check on whether they rank targets the same way.

Everything else is held identical to the MLP arm on purpose: the same shared-EC
label set rule, the same fixed-budget pool with a target fraction r, the same
"90% of the achievable ceiling" rule, and the ceiling reported next to r*.
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_ec_geometry import load_table, drop_groups  # noqa: E402

_CTX = {}


def macro_f1(pred, y):
    f1s = []
    for c in np.unique(y):
        tp = float(((pred == c) & (y == c)).sum())
        fp = float(((pred == c) & (y != c)).sum())
        fn = float(((pred != c) & (y == c)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def fit_eval(Xtr, ytr, Xte, yte, seed, scaler):
    if len(np.unique(ytr)) < 2:
        return 0.0
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(scaler.transform(Xtr), ytr)
    return macro_f1(clf.predict(scaler.transform(Xte)), yte)


def stratified_split(rng, y, idx, test_frac):
    tr, te = [], []
    for c in np.unique(y[idx]):
        ci = rng.permutation(idx[y[idx] == c])
        n_te = max(1, int(round(test_frac * len(ci))))
        n_te = min(n_te, len(ci) - 1) if len(ci) > 1 else len(ci)
        te.append(ci[:n_te])
        tr.append(ci[n_te:])
    return np.concatenate(tr), np.concatenate(te)


def _pair_row(st):
    s, t = st
    T = _CTX["T"]
    a = _CTX["a"]
    lab, dom, X = T["ec"], T["dom"], T["X"]
    keys = [k for k in sorted(set(lab))
            if ((lab == k) & (dom == s)).sum() >= a.min_n
            and ((lab == k) & (dom == t)).sum() >= a.min_n]
    if len(keys) < 3:
        return None
    k2i = {k: i for i, k in enumerate(keys)}
    keep = np.isin(lab, keys)
    y_all = np.array([k2i.get(v, -1) for v in lab], dtype=np.int64)
    K = len(keys)

    rng = np.random.default_rng(abs(hash((s, t))) % (2 ** 31))
    s_tr, _ = stratified_split(rng, y_all, np.where(keep & (dom == s))[0], a.test_frac)
    t_tr, t_te = stratified_split(rng, y_all, np.where(keep & (dom == t))[0], a.test_frac)
    if a.max_train and len(s_tr) > a.max_train:
        s_tr = rng.permutation(s_tr)[:a.max_train]
    if a.max_train and len(t_tr) > a.max_train:
        t_tr = rng.permutation(t_tr)[:a.max_train]
    if a.max_test and len(t_te) > a.max_test:
        t_te = rng.permutation(t_te)[:a.max_test]

    scaler = StandardScaler().fit(X[s_tr])          # source-only, as in the MLP arm
    Xs, ys = X[s_tr], y_all[s_tr]
    Xt, yt = X[t_tr], y_all[t_tr]
    Xe, ye = X[t_te], y_all[t_te]

    ceil = fit_eval(Xt, yt, Xe, ye, 0, scaler)
    zs = fit_eval(Xs, ys, Xe, ye, 0, scaler)
    budgets = [p for p in a.budgets if p <= len(t_tr)]
    if not budgets or ceil <= 0:
        return None

    per_budget = {}
    for P in budgets:
        fin = {}
        for r in a.ood_fracs:
            nt, ns = int(round(r * P)), P - int(round(r * P))
            vals = []
            for seed in range(a.n_rep):
                r2 = np.random.default_rng(9000 + seed)
                ti = r2.permutation(len(Xt))[:nt]
                si = r2.permutation(len(Xs))[:ns]
                px = np.concatenate([Xt[ti], Xs[si]]) if ns and nt else (
                    Xt[ti] if nt else Xs[si])
                py = np.concatenate([yt[ti], ys[si]]) if ns and nt else (
                    yt[ti] if nt else ys[si])
                vals.append(fit_eval(px, py, Xe, ye, seed, scaler))
            fin[r] = float(np.mean(vals))
        bar = a.recover_at * ceil
        rstar = next((r for r in sorted(a.ood_fracs) if fin[r] >= bar), None)
        per_budget[str(P)] = {
            "bar": round(bar, 4),
            "r_star": (rstar if rstar is not None else None),
            "final": {str(r): round(v, 4) for r, v in fin.items()}}
    return dict(source=s, target=t, n_classes=K, n_src_train=len(s_tr),
                n_tgt_train=len(t_tr), n_tgt_test=len(t_te),
                ceiling_f1=round(ceil, 4), zeroshot_f1=round(zs, 4),
                retained=round(zs / ceil, 4), budgets=per_budget)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--ec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ec_level", type=int, default=3)
    ap.add_argument("--drop_groups", default="other_bacteria,other_eukaryota")
    ap.add_argument("--min_n", type=int, default=15)
    ap.add_argument("--budgets", default="200,500")
    ap.add_argument("--ood_fracs", default="0.0,0.05,0.1,0.2,0.3,0.5,0.75,1.0")
    ap.add_argument("--n_rep", type=int, default=3)
    ap.add_argument("--test_frac", type=float, default=0.4)
    ap.add_argument("--max_train", type=int, default=6000)
    ap.add_argument("--max_test", type=int, default=3000)
    ap.add_argument("--recover_at", type=float, default=0.9)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    a.budgets = [int(x) for x in a.budgets.split(",") if x]
    a.ood_fracs = [float(x) for x in a.ood_fracs.split(",") if x]

    T = load_table(a.emb, a.meta, a.ec, a.ec_level)
    T = drop_groups(T, a.drop_groups.split(","))
    groups = sorted(set(T["dom"]))
    pairs = list(itertools.permutations(groups, 2))
    print(f"{len(T['X'])} proteins, {len(groups)} groups, {len(pairs)} ordered pairs, "
          f"{a.jobs} worker(s)", flush=True)

    _CTX.update(T=T, a=a)
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.jobs) as pool:
            rows = pool.map(_pair_row, pairs, chunksize=1)
    else:
        rows = [_pair_row(p) for p in pairs]
    rows = [r for r in rows if r is not None]
    print(f"{len(rows)} pairs produced an r* row", flush=True)

    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "rows": rows}, f, indent=2)

    flat = []
    for r in rows:
        for P, b in r["budgets"].items():
            flat.append({k: v for k, v in r.items() if k != "budgets"}
                        | {"budget": int(P), "bar": b["bar"],
                           "r_star": (b["r_star"] if b["r_star"] is not None
                                      else float("nan"))})
    if flat:
        csv_path = a.out.replace(".json", "_flat.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)
        print(f"wrote {a.out} and {csv_path}", flush=True)


if __name__ == "__main__":
    main()
